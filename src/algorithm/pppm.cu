#include "pppm.h"
#include "pppm_direct.h"
namespace pppm
{
__global__ void solve_fdtd_far_kernel(PPPMSolver solver)
{
    int grid_idx = blockIdx.x;
    if (grid_idx >= solver.pg.neighbor_3_square_nonempty.size())
        return;
    int3 grid_coord = solver.pg.neighbor_3_square_nonempty[grid_idx].coord;
    auto &neighbor_list = solver.pg.neighbor_3_square_list(grid_coord);
    auto &cache = solver.grid_cache;
    int t = solver.time_idx();
    __shared__ float fdtd_near_field;
    if (threadIdx.x == 0)
    {
        fdtd_near_field = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < neighbor_list.size(); i += blockDim.x)
    {
        int tri_idx = neighbor_list[i];
        auto &tri = solver.pg.triangles[tri_idx];
        int3 tri_coord = tri.grid_coord;
        auto w = cache.fdtd_near_weight(tri_idx, tri_coord, grid_coord);
        auto &neumann = solver.neumann[tri_idx];
        auto &dirichlet = solver.dirichlet[tri_idx];
        atomicAdd_block(&fdtd_near_field, w.convolution(neumann, dirichlet, t));
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float far_field = solver.pg.fdtd.grids[t](grid_coord) - fdtd_near_field;
        solver.grid_far_field[t](grid_coord) =
            far_field;  // far field of empty grid need to be initialized with FDTD solution
    }
}

void PPPMSolver::solve_fdtd_far(bool log_time)
{
    START_TIME(log_time)
    grid_far_field[time_idx()].assign(pg.fdtd.grids[time_idx()]);
    cuExecuteBlock(pg.neighbor_3_square_nonempty.size(), 64, solve_fdtd_far_kernel, *this);
    LOG_TIME("solve_fdtd_far")
}

__global__ void solve_fdtd_near_kernel(PPPMSolver solver)
{
    int grid_idx = blockIdx.x;
    if (grid_idx >= solver.pg.neighbor_3_square_nonempty.size())
        return;
    int3 grid_coord = solver.pg.neighbor_3_square_nonempty[grid_idx].coord;
    auto &neighbor_list = solver.pg.neighbor_3_square_list(grid_coord);
    auto &cache = solver.grid_cache;
    int t = solver.time_idx();
    __shared__ float accurate_near_field;
    if (threadIdx.x == 0)
    {
        accurate_near_field = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < neighbor_list.size(); i += blockDim.x)
    {
        int tri_idx = neighbor_list[i];
        auto &tri = solver.pg.triangles[tri_idx];
        int3 tri_coord = tri.grid_coord;
        auto w = cache.bem_near_weight(tri_idx, tri_coord, grid_coord);
        auto &neumann = solver.neumann[tri_idx];
        auto &dirichlet = solver.dirichlet[tri_idx];
        atomicAdd_block(&accurate_near_field, w.convolution(neumann, dirichlet, t));
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        solver.pg.fdtd.grids[t](grid_coord) = solver.grid_far_field[t](grid_coord) + accurate_near_field;
    }
}

void PPPMSolver::solve_fdtd_near(bool log_time)
{
    START_TIME(log_time)
    cuExecuteBlock(pg.neighbor_3_square_nonempty.size(), 64, solve_fdtd_near_kernel, *this);
    LOG_TIME("solve_fdtd_near")
}

__global__ void solve_face_far_kernel(PPPMSolver solver)
{
    int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_idx >= solver.pg.triangles.size())
        return;
    auto &tri = solver.pg.triangles[tri_idx];
    auto &w = solver.face_cache.interpolation_weight;
    int t = solver.time_idx();
    float far_sum = 0;
    // caculate the far field from interpolation
#pragma unroll
    for (int dx = 0; dx < 2; dx++)
#pragma unroll
        for (int dy = 0; dy < 2; dy++)
#pragma unroll
            for (int dz = 0; dz < 2; dz++)
            {
                int weight_idx = dx * 4 + dy * 2 + dz;
                int3 coord = tri.grid_base_coord + make_int3(dx, dy, dz);
                far_sum += w(tri_idx, weight_idx) * solver.grid_far_field[t](coord);
            }
    solver.face_far_field[tri_idx] = far_sum;
    solver.dirichlet[tri_idx][t] = 0;
}

__global__ void solve_face_near_kernel(PPPMSolver solver)
{
    int base_coord_idx = blockIdx.x;
    if (base_coord_idx >= solver.pg.base_coord_nonempty.size())
        return;
    int3 base_coord = solver.pg.base_coord_nonempty[base_coord_idx].coord;
    auto &neighbor_list = solver.pg.neighbor_4_square_list(base_coord);
    auto &center_triangle_list = solver.pg.base_coord_face_list(base_coord);
    int center_num = center_triangle_list.size();
    int neighbor_num = neighbor_list.size();
    int total_num = center_num * neighbor_num;
    int t = solver.time_idx();
    for (int i = threadIdx.x; i < total_num; i += blockDim.x)
    {
        int neighbor_i = i / center_num;
        int center_i = i % center_num;
        int neighbor_face_idx = neighbor_list[neighbor_i];
        int center_face_idx = center_triangle_list[center_i];
        auto &w = solver.face_cache.face2face_weight(center_face_idx, neighbor_face_idx);
        auto &neumann = solver.neumann[neighbor_face_idx];
        auto &dirichlet = solver.dirichlet[neighbor_face_idx];
        solver.face_near_field(center_face_idx, neighbor_i) = w.convolution(neumann, dirichlet, t);
        solver.face_factor(center_face_idx, neighbor_i) = (float)w.double_layer[0] * w.max_double_layer_abs;
    }
}

__global__ void update_dirichlet_kernel(PPPMSolver solver)
{
    int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_idx >= solver.pg.triangles.size())
        return;
    auto &tri = solver.pg.triangles[tri_idx];
    auto &neighbor_list = solver.pg.neighbor_4_square_list(tri.grid_base_coord);
    // Equation (2.12) in Paper:https://epubs.siam.org/doi/pdf/10.1137/090775981
    float near_field_sum = 0, factor_sum = 0;
    for (int i = 0; i < neighbor_list.size(); i++)
    {
        near_field_sum += solver.face_near_field(tri_idx, i);
        factor_sum += solver.face_factor(tri_idx, i);
    }
    solver.dirichlet[tri_idx][solver.time_idx()] =
        (solver.face_far_field[tri_idx] * 0.83 + near_field_sum) / (0.5 * tri.area - factor_sum);
}

void PPPMSolver::update_dirichlet(bool log_time)
{
    START_TIME(log_time)
    cuExecute(pg.triangles.size(), solve_face_far_kernel, *this);
    LOG_TIME("solve_face_far")
    cuExecuteBlock(pg.base_coord_nonempty.size(), 64, solve_face_near_kernel, *this);
    LOG_TIME("solve_face_near")
    cuExecute(pg.triangles.size(), update_dirichlet_kernel, *this);
    LOG_TIME("update_dirichlet")
}

__global__ void set_neumann_condition_kernel(GArr<History> neumann, GArr<float> neuuman_condition, int t)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < neumann.size())
    {
        neumann[i][t] = neuuman_condition[i];
    }
}

void PPPMSolver::set_neumann_condition(CArr<float> neuuman_condition, bool log_time)
{
    START_TIME(log_time)
    current_neumann.assign(neuuman_condition);
    cuExecute(neumann.size(), set_neumann_condition_kernel, neumann, current_neumann, time_idx());
    LOG_TIME("set_neumann_condition")
}

void PPPMSolver::solve_fdtd_far_simple(bool log_time)
{
    START_TIME(log_time)
    pg.fdtd.step(log_time);
    direct_fdtd_far(*this);
    LOG_TIME("Direct Far Solve")
}

void PPPMSolver::solve_fdtd_near_simple(bool log_time)
{
    START_TIME(log_time)
    direct_correction_fdtd_near(*this);
    LOG_TIME("Direct Near Solve")
}

}  // namespace pppm