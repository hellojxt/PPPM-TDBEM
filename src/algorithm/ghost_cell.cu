#include "ghost_cell.h"
#include "ghost_cell_util.h"
#include "macro.h"
namespace pppm
{

CGPU_FUNC inline int3 neighbor_idx_to_coord(int idx)
{
    return make_int3(idx % 2, (idx / 2) % 2, idx / 4);
}

// normalize the coordinate to the range [-1, 1]^3
GPU_FUNC inline float3 get_normalized_coord(float3 coord, int3 base_coord, GhostCellSolver &solver)
{
    float3 base_point = solver.grid.getCenter(base_coord);
    return (coord - base_point) / solver.grid_size() * 2 + make_float3(-1, -1, -1);
}

// return the base coordinate of the 8 interpolation points of the reflect point
GPU_FUNC inline int3 get_base_coord_for_reflect(CellInfo ghost_cell, GhostCellSolver &solver)
{
    float grid_size = solver.grid_size();
    float3 reflect_point = ghost_cell.reflect_point;
    int3 base_coord = make_int3(reflect_point / grid_size - 0.5f);
#ifdef MEMORY_CHECK
    float3 base_point = solver.grid.getCenter(base_coord);
    float3 offset = reflect_point - base_point;
    if (offset.x < 0 || offset.y < 0 || offset.z < 0 || offset.x > grid_size || offset.y > grid_size ||
        offset.z > grid_size)
        printf("interpolation error in construct_phi_matrix_kernel in ghost_cell.cu");
#endif
    return base_coord;
}

__global__ void construct_ghost_cell_list(GhostCellSolver solver)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = solver.cell_data.size.x;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;
    CellInfo cell = solver.cell_data(x, y, z);
    if (cell.type == CellType::GHOST)
    {
        solver.ghost_cells[cell.ghost_idx] = make_int3(x, y, z);
    }
}

void GhostCellSolver::precompute_cell_data(bool log_time)
{
    START_TIME(log_time)
    ghost_cell_num = fill_cell_data(grid, cell_data, (condition_number_threshold == 0));
    ghost_cells.resize(ghost_cell_num);
    cuExecute3D(grid.grid_dim, construct_ghost_cell_list, *this);
    if (ghost_cell_num <= 0)
        LOG_ERROR("No ghost cell found!");
    LOG_TIME("Precompute Cell Data")
};

__global__ void construct_phi_matrix_kernel(GArr3D<float> phi, GhostCellSolver solver)
{
    int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ghost_idx >= solver.ghost_cell_num)
        return;
    int3 ghost_cell_coord = solver.ghost_cells[ghost_idx];
    CellInfo ghost_cell = solver.cell_data(ghost_cell_coord);
    float3 normal = normalize(ghost_cell.reflect_point - ghost_cell.nearst_point);
    int3 base_coord = get_base_coord_for_reflect(ghost_cell, solver);

    for (int i = 0; i < GHOST_CELL_NEIGHBOR_NUM; i++)
    {
        int3 dcoord = neighbor_idx_to_coord(i);
        int3 neighbor_coord = base_coord + dcoord;
        bool is_self = (neighbor_coord.x == ghost_cell_coord.x && neighbor_coord.y == ghost_cell_coord.y &&
                        neighbor_coord.z == ghost_cell_coord.z);
        if (is_self)
        {
            // dn phi = normal dot D phi
            float3 coord = get_normalized_coord(ghost_cell.nearst_point, base_coord, solver);
            phi(ghost_idx, i, 0) = dot(normal, make_float3(coord.y * coord.z, coord.x * coord.z, coord.x * coord.y));
            phi(ghost_idx, i, 1) = dot(normal, make_float3(coord.y, coord.x, 0));
            phi(ghost_idx, i, 2) = dot(normal, make_float3(0, coord.z, coord.y));
            phi(ghost_idx, i, 3) = dot(normal, make_float3(coord.z, 0, coord.x));
            phi(ghost_idx, i, 4) = dot(normal, make_float3(1, 0, 0));
            phi(ghost_idx, i, 5) = dot(normal, make_float3(0, 1, 0));
            phi(ghost_idx, i, 6) = dot(normal, make_float3(0, 0, 1));
            phi(ghost_idx, i, 7) = 0;
        }
        else
        {
            // phi[ghost_idx][i] = [xyz, xy, xz, yz, x, y, z, 1];
            float3 coord = make_float3(dcoord * 2 - 1);
            phi(ghost_idx, i, 0) = coord.x * coord.y * coord.z;
            phi(ghost_idx, i, 1) = coord.x * coord.y;
            phi(ghost_idx, i, 2) = coord.y * coord.z;
            phi(ghost_idx, i, 3) = coord.x * coord.z;
            phi(ghost_idx, i, 4) = coord.x;
            phi(ghost_idx, i, 5) = coord.y;
            phi(ghost_idx, i, 6) = coord.z;
            phi(ghost_idx, i, 7) = 1;
        }
    }
}

__global__ void precompute_p_weight_kernel(SVDResult svd_result, GhostCellSolver solver)
{
    int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ghost_idx >= solver.ghost_cell_num)
        return;
    float condition_num = svd_result.S(ghost_idx, 0) / svd_result.S(ghost_idx, GHOST_CELL_NEIGHBOR_NUM - 1);
    // printf("condition_num: %f , threshold: %f\n", condition_num, solver.condition_number_threshold);
    if (condition_num > solver.condition_number_threshold)
    {
        solver.ghost_order[ghost_idx] = AccuracyOrder::FIRST_ORDER;
        for (int i = 0; i < GHOST_CELL_NEIGHBOR_NUM; i++)
        {
            solver.p_weight(ghost_idx, i) = 0;
        }
    }
    else
    {
        solver.ghost_order[ghost_idx] = AccuracyOrder::SECOND_ORDER;
        int3 ghost_cell_coord = solver.ghost_cells[ghost_idx];
        auto ghost_cell = solver.cell_data(ghost_cell_coord);
        int3 base_coord = get_base_coord_for_reflect(ghost_cell, solver);
        float3 r = get_normalized_coord(ghost_cell.reflect_point, base_coord, solver);
        // printf("r: %f %f %f\n", r.x, r.y, r.z);
        float phi_r[GHOST_CELL_NEIGHBOR_NUM] = {r.x * r.y * r.z, r.x * r.y, r.y * r.z, r.x * r.z, r.x, r.y, r.z, 1};
        // p_weight = inv_A.T * phi_r
        for (int i = 0; i < GHOST_CELL_NEIGHBOR_NUM; i++)
        {
            float sum = 0;
            for (int j = 0; j < GHOST_CELL_NEIGHBOR_NUM; j++)
            {
                sum += svd_result.inv_A(ghost_idx, j, i) * phi_r[j];
            }
            solver.p_weight(ghost_idx, i) = sum;
        }
    }
}

template <bool CONSTRUCT_MATRIX = true, bool CONSTRUCT_RHS = true>
__global__ void construct_equation_kernel(GhostCellSolver solver)
{
    int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ghost_idx >= solver.ghost_cell_num)
        return;
    auto acc_order = solver.ghost_order[ghost_idx];
    int offset = ghost_idx * (GHOST_CELL_NEIGHBOR_NUM + 1);
    int3 ghost_cell_coord = solver.ghost_cells[ghost_idx];
    if (acc_order == AccuracyOrder::FIRST_ORDER)
    {
        if (CONSTRUCT_MATRIX)
        {
            solver.A.rows[offset] = ghost_idx;
            solver.A.cols[offset] = ghost_idx;
            solver.A.vals[offset] = 1;
        }
        if (CONSTRUCT_RHS)
        {
            int3 dcoord[6] = {make_int3(1, 0, 0),  make_int3(-1, 0, 0), make_int3(0, 1, 0),
                              make_int3(0, -1, 0), make_int3(0, 0, 1),  make_int3(0, 0, -1)};
            auto cell = solver.cell_data(ghost_cell_coord);
            for (int i = 0; i < 6; i++)
            {
                int3 coord = ghost_cell_coord + dcoord[i];
                if (solver.cell_data(coord).type == AIR)
                {
                    solver.b[ghost_idx] = -solver.neuuman_data[cell.nearest_particle_idx] * solver.grid_size() +
                                          solver.fdtd.grids[solver.fdtd.t](coord);  // p_g - p_n = l * rho * a_n
                    break;
                }
            }
        }
    }
    else if (acc_order == AccuracyOrder::SECOND_ORDER)
    {
        auto ghost_cell = solver.cell_data(ghost_cell_coord);
        int3 base_coord = get_base_coord_for_reflect(ghost_cell, solver);
        float b_value = 0;
        if (CONSTRUCT_RHS)
            b_value += -solver.neuuman_data[ghost_cell.nearest_particle_idx] * ghost_cell.nearst_distance * 2;
        for (int i = 0; i < GHOST_CELL_NEIGHBOR_NUM; i++)
        {
            int3 dcoord = neighbor_idx_to_coord(i);
            int3 neighbor_coord = base_coord + dcoord;
            auto neighbor_cell = solver.cell_data(neighbor_coord);
            bool is_self = (neighbor_cell.ghost_idx == ghost_idx);
            if (is_self)  // ghost cell self, p = -rho*an(nearst_point)
            {
                if (CONSTRUCT_RHS)
                {
                    float scale_factor =
                        solver.grid_size() / 2;  // correction factor as stencils are transformed to the [−1, 1]^3
                    b_value += solver.p_weight(ghost_idx, i) *
                               (scale_factor * solver.neuuman_data[ghost_cell.nearest_particle_idx]);
                }
            }
            else if (neighbor_cell.type == GHOST)  // other ghost cell, add matrix element
            {
                if (CONSTRUCT_MATRIX)
                {
                    solver.A.rows[offset + i] = ghost_idx;
                    solver.A.cols[offset + i] = neighbor_cell.ghost_idx;
                    // solver.A.vals[offset + i] = 0;
                    solver.A.vals[offset + i] = -solver.p_weight(ghost_idx, i);
                    // printf("A(%d, %d) = %f\n", ghost_idx, neighbor_cell.ghost_idx, solver.p_weight(ghost_idx, i));
                }
            }
            else  // neighbor cell is air
            {
                if (CONSTRUCT_RHS)
                {
                    b_value += solver.p_weight(ghost_idx, i) * solver.fdtd.grids[solver.fdtd.t](neighbor_coord);
                }
            }
        }
        if (CONSTRUCT_RHS)
        {
            solver.b[ghost_idx] = b_value;
        }

        if (CONSTRUCT_MATRIX)
        {
            solver.A.rows[offset + GHOST_CELL_NEIGHBOR_NUM] = ghost_idx;
            solver.A.cols[offset + GHOST_CELL_NEIGHBOR_NUM] = ghost_idx;
            solver.A.vals[offset + GHOST_CELL_NEIGHBOR_NUM] = 1;
        }
    }
}

void GhostCellSolver::precompute_ghost_matrix(bool log_time)
{
    START_TIME(log_time)
    A.resize(ghost_cell_num, ghost_cell_num, ghost_cell_num * (GHOST_CELL_NEIGHBOR_NUM + 1));
    A.reset();  // set A to zero matrix
    p_weight.resize(ghost_cell_num, GHOST_CELL_NEIGHBOR_NUM);
    b.resize(ghost_cell_num);
    x.resize(ghost_cell_num);
    GArr3D<float> phi;
    phi.resize(ghost_cell_num, GHOST_CELL_NEIGHBOR_NUM, GHOST_CELL_NEIGHBOR_NUM);
    cuExecute(ghost_cell_num, construct_phi_matrix_kernel, phi, *this);
    LOG_TIME("Construct phi matrix")
    auto svd_result = cusolver_svd(phi);
    svd_result.solve_inverse();
    LOG_TIME("SVD")
    ghost_order.resize(ghost_cell_num);
    cuExecute(ghost_cell_num, precompute_p_weight_kernel, svd_result, *this);
    LOG_TIME("Precompute p weight")
    if (condition_number_threshold > 0.0f)
    {
        auto construct_matrix_kernel = construct_equation_kernel<true, false>;
        cuExecute(ghost_cell_num, construct_matrix_kernel, *this);
        A.eliminate_zeros();
        A.sort_by_row();
        linear_solver.set_coo_matrix(A);
        LOG_TIME("Construct matrix A")
    }
    phi.clear();
    svd_result.clear();
}

__global__ void update_ghost_cell_kernel(GArr<float> x, GhostCellSolver solver)
{
    int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ghost_idx >= solver.ghost_cell_num)
        return;
    int3 ghost_cell_coord = solver.ghost_cells[ghost_idx];
    solver.fdtd.grids[solver.fdtd.t](ghost_cell_coord) = x[ghost_idx];
}

void GhostCellSolver::solve_ghost_cell(bool log_time)
{
    START_TIME(log_time)
    b.reset();
    auto construct_rhs_kernel = construct_equation_kernel<false, true>;
    cuExecute(ghost_cell_num, construct_rhs_kernel, *this);
    LOG_TIME("Construct rhs b")
    if (condition_number_threshold == 0.0f)
    {
        cuExecute(ghost_cell_num, update_ghost_cell_kernel, b, *this);
    }
    else
    {
        linear_solver.solve(b, x, 100);
        cuExecute(ghost_cell_num, update_ghost_cell_kernel, x, *this);
        LOG_TIME("Solve equation for ghost cell")
    }
}

}  // namespace pppm