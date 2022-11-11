#include "ghost_cell.h"
#include "ghost_cell_util.h"
#include "macro.h"
namespace pppm
{

CGPU_FUNC inline int3 neighbor_idx_to_coord(int idx)
{
    return make_int3(idx % 2, (idx / 2) % 2, idx / 4);
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

void GhostCellSolver::precompute_cell_data()
{
    ghost_cell_num = fill_cell_data(grid, cell_data);
    ghost_cells.resize(ghost_cell_num);
    cuExecute3D(grid.grid_dim, construct_ghost_cell_list, *this);
};

#define SINGULAR_EPS 0.02f

__global__ void construct_phi_matrix_kernel(GArr3D<float> phi, GhostCellSolver solver)
{
    int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ghost_idx >= solver.ghost_cell_num)
        return;
    int3 ghost_cell_coord = solver.ghost_cells[ghost_idx];
    CellInfo ghost_cell = solver.cell_data(ghost_cell_coord);
    float3 nearest_point = ghost_cell.nearst_point;
    float3 reflect_point = ghost_cell.reflect_point;
    float grid_size = solver.grid.grid_size;
    int3 base_coord = make_int3(reflect_point / grid_size - 0.5f);
    float3 base_point = solver.grid.getCenter(base_coord);
#ifdef MEMORY_CHECK
    float3 offset = reflect_point - base_point;
    if (offset.x < 0 || offset.y < 0 || offset.z < 0 || offset.x > grid_size || offset.y > grid_size ||
        offset.z > grid_size)
        printf("interpolation error in construct_phi_matrix_kernel in ghost_cell.cu");
#endif
    float3 normal = solver.grid.particles[ghost_cell.nearest_particle_idx].normal;
    for (int i = 0; i < GHOST_CELL_NEIGHBOR_NUM; i++)
    {
        int3 dcoord = neighbor_idx_to_coord(i);
        int3 coord = base_coord + dcoord;
        float distance = length(solver.grid.getCenter(coord) - nearest_point);
        if (distance > grid_size * SINGULAR_EPS)
        {
            // phi[ghost_idx][i] = [xyz, xy, xz, yz, x, y, z, 1];
            phi(ghost_idx, i, 0) = coord.x * coord.y * coord.z;
            phi(ghost_idx, i, 1) = coord.x * coord.y;
            phi(ghost_idx, i, 2) = coord.y * coord.z;
            phi(ghost_idx, i, 3) = coord.x * coord.z;
            phi(ghost_idx, i, 4) = coord.x;
            phi(ghost_idx, i, 5) = coord.y;
            phi(ghost_idx, i, 6) = coord.z;
            phi(ghost_idx, i, 7) = 1;
        }
        else
        {
            // dn phi = normal dot D phi
            phi(ghost_idx, i, 0) = dot(normal, make_float3(coord.y * coord.z, coord.x * coord.z, coord.x * coord.y));
            phi(ghost_idx, i, 1) = dot(normal, make_float3(coord.y, coord.x, 0));
            phi(ghost_idx, i, 2) = dot(normal, make_float3(0, coord.z, coord.y));
            phi(ghost_idx, i, 3) = dot(normal, make_float3(coord.z, 0, coord.x));
            phi(ghost_idx, i, 4) = dot(normal, make_float3(1, 0, 0));
            phi(ghost_idx, i, 5) = dot(normal, make_float3(0, 1, 0));
            phi(ghost_idx, i, 6) = dot(normal, make_float3(0, 0, 1));
            phi(ghost_idx, i, 7) = 0;
        }
    }
}

__global__ void precompute_p_weight_kernel(SVDResult svd_result, GhostCellSolver solver)
{
    int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ghost_idx >= solver.ghost_cell_num)
        return;
    int condition_num = svd_result.S(ghost_idx, 0) / svd_result.S(ghost_idx, GHOST_CELL_NEIGHBOR_NUM - 1);
    if (condition_num > CONDITION_NUMBER_THRESHOLD)
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
        float3 r = solver.cell_data(ghost_cell_coord).reflect_point;
        float phi_r[GHOST_CELL_NEIGHBOR_NUM] = {r.x * r.y * r.z, r.x * r.y, r.y * r.z, r.x * r.z, r.x, r.y, r.z, 1};
        for (int i = 0; i < GHOST_CELL_NEIGHBOR_NUM; i++)
        {
            int sum = 0;
            for (int j = 0; j < GHOST_CELL_NEIGHBOR_NUM; j++)
            {
                sum += svd_result.inv_A(ghost_idx, j, i);
            }
            solver.p_weight(ghost_idx, i) = sum * phi_r[i];
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
    if (CONSTRUCT_MATRIX)
    {
        solver.A.rows[offset] = ghost_idx;
        solver.A.cols[offset] = ghost_idx;
        solver.A.vals[offset] = 1;
    }
    int3 ghost_cell_coord = solver.ghost_cells[ghost_idx];
    if (acc_order == AccuracyOrder::FIRST_ORDER)
    {
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
                    solver.b[ghost_idx] =
                        solver.neuuman_data[cell.nearest_particle_idx] * AIR_DENSITY * solver.grid_size() +
                        solver.fdtd.grids[solver.fdtd.t](coord);  // p_g - p_n = l * rho * a_n
                    break;
                }
            }
        }
    }
    else if (acc_order == AccuracyOrder::SECOND_ORDER)
    {
        auto ghost_cell = solver.cell_data(ghost_cell_coord);
        float3 nearest_point = ghost_cell.nearst_point;
        float3 reflect_point = ghost_cell.reflect_point;
        int3 base_coord = make_int3(reflect_point / solver.grid_size() - 0.5f);
        float b_value = solver.neuuman_data[ghost_cell.nearest_particle_idx] * AIR_DENSITY * solver.grid_size();
        for (int i = 0; i < GHOST_CELL_NEIGHBOR_NUM; i++)
        {
            int3 dcoord = neighbor_idx_to_coord(i);
            int3 coord = base_coord + dcoord;
            float distance = length(solver.grid.getCenter(coord) - nearest_point);
            auto neighbor_cell = solver.cell_data(coord);
            if ((distance > solver.grid_size() * SINGULAR_EPS) && (neighbor_cell.type == GHOST))
            {
                if (CONSTRUCT_MATRIX)
                {
                    solver.A.rows[offset + i + 1] = ghost_idx;
                    solver.A.cols[offset + i + 1] = neighbor_cell.ghost_idx;
                    solver.A.vals[offset + i + 1] = -solver.p_weight(ghost_idx, i);
                }
            }
            else
            {
                if (CONSTRUCT_RHS)
                {
                    if (neighbor_cell.type == GHOST)
                    {
                        b_value += solver.p_weight(ghost_idx, i) *
                                   solver.neuuman_data[ghost_cell.nearest_particle_idx] * AIR_DENSITY;
                    }
                    else
                    {
                        b_value += solver.p_weight(ghost_idx, i) * solver.fdtd.grids[solver.fdtd.t](coord);
                    }
                }
            }
        }
        if (CONSTRUCT_RHS)
        {
            solver.b[ghost_idx] = b_value;
        }
    }
}

void GhostCellSolver::precompute_ghost_matrix()
{
    A.resize(ghost_cell_num, ghost_cell_num, ghost_cell_num * (GHOST_CELL_NEIGHBOR_NUM + 1));
    A.reset();  // set A to zero matrix
    p_weight.resize(ghost_cell_num, GHOST_CELL_NEIGHBOR_NUM);
    b.resize(ghost_cell_num);
    GArr3D<float> phi;
    phi.resize(ghost_cell_num, GHOST_CELL_NEIGHBOR_NUM, GHOST_CELL_NEIGHBOR_NUM);
    cuExecute(ghost_cell_num, construct_phi_matrix_kernel, phi, *this);
    auto svd_result = cusolver_svd(phi);
    svd_result.solve_inverse();
    cuExecute(ghost_cell_num, precompute_p_weight_kernel, svd_result, *this);
    auto construct_matrix_kernel = construct_equation_kernel<true, false>;
    cuExecute(ghost_cell_num, construct_matrix_kernel, *this);
    phi.clear();
    svd_result.clear();
}

}  // namespace pppm