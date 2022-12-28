#include "ghost_cell.h"
#include "ghost_cell_util.h"
#include "helper_math.h"
namespace pppm
{

__global__ void fill_in_nearest_kernel(GArr3D<CellInfo> cell_data, ParticleGrid grid, bool thin_shell)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = grid.grid_dim;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;

    float3 grid_center = grid.getCenter(x, y, z);
    CellInfo result;
    result.type = UNKNOWN;
    result.nearst_distance = MAX_FLOAT;

    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids
            {
                int3 coord = make_int3(x + dx, y + dy, z + dz);
                if (coord.x < 0 || coord.x >= grid_dim || coord.y < 0 || coord.y >= grid_dim || coord.z < 0 ||
                    coord.z >= grid_dim)
                    continue;

                auto &neighbor_list = grid.grid_face_list(coord);
                for (int i = 0; i < neighbor_list.size(); i++)
                {
                    int face_idx = neighbor_list[i];
                    int3 triID = grid.triangles[face_idx].indices;
                    float3 nearest_point = get_nearest_triangle_point(grid_center, grid.vertices[triID.x],
                                                                      grid.vertices[triID.y], grid.vertices[triID.z]);
                    float curr_len = length(grid_center - nearest_point);
                    if (curr_len < result.nearst_distance)
                    {
                        result.nearst_distance = curr_len;
                        result.nearst_point = nearest_point;
                        result.nearest_particle_idx = face_idx;
                        result.reflect_point = 2 * nearest_point - grid_center;
                    }
                }
            }
    if (result.nearst_distance < MAX_FLOAT)
    {
        auto nearest_particle = grid.triangles[result.nearest_particle_idx];
        float direction = dot(nearest_particle.normal, grid_center - result.nearst_point);
        result.type = (direction > 0) ? AIR : SOLID;
    }
    if (thin_shell && grid.grid_face_list(x, y, z).size() > 0)
        result.type = SOLID;
    cell_data(x, y, z) = result;
    return;
}

__global__ void cell_classfication_kernel(GArr3D<CellInfo> cell_data,
                                          ParticleGrid grid,
                                          GArr3D<CellType> type_arr,
                                          GArr3D<int> ghost_idx_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = cell_data.size.x;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;
    CellType type = cell_data(x, y, z).type;
    int3 neighbor_list[6] = {make_int3(x + 1, y, z), make_int3(x - 1, y, z), make_int3(x, y + 1, z),
                             make_int3(x, y - 1, z), make_int3(x, y, z + 1), make_int3(x, y, z - 1)};
    if (type == SOLID)
    {
        for (int i = 0; i < 6; i++)
        {
            int3 coord = neighbor_list[i];
            if (coord.x < 0 || coord.x >= grid_dim || coord.y < 0 || coord.y >= grid_dim || coord.z < 0 ||
                coord.z >= grid_dim)
                continue;
            if (cell_data(coord).type == AIR && cell_data(coord).nearst_distance < MAX_FLOAT)
            {
                type = GHOST;
            }
        }
    }
    type_arr(x, y, z) = type;
    ghost_idx_arr(x, y, z) = (type == GHOST);
}

__global__ void post_cell_classification_kernel(ParticleGrid grid, GArr3D<CellType> type_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= grid.grid_dim || y < 0 || y >= grid.grid_dim)
        return;
    auto current_type = AIR;
    for (int z = 0; z < grid.grid_dim; z++)
    {
        auto &t = type_arr(x, y, z);
        if (t == SOLID || t == GHOST)
        {
            current_type = SOLID;
        }
        else if (t == AIR)
        {
            current_type = AIR;
        }
        else
        {
            type_arr(x, y, z) = current_type;
        }
    }
}

__global__ void apply_cell_type_kernel(GArr3D<CellInfo> cell_data,
                                       ParticleGrid grid,
                                       GArr3D<CellType> type_arr,
                                       GArr3D<int> ghost_idx_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = cell_data.size.x;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;
    cell_data(x, y, z).type = type_arr(x, y, z);
    if (type_arr(x, y, z) == GHOST)
    {
        cell_data(x, y, z).ghost_idx = ghost_idx_arr(x, y, z) - 1;
    }
    else
    {
        cell_data(x, y, z).ghost_idx = -1;
    }
}

int fill_cell_data(ParticleGrid grid, GArr3D<CellInfo> cell_data, bool thin_shell)
{
    auto grid_dim_3D = make_int3(grid.grid_dim, grid.grid_dim, grid.grid_dim);
    cuExecute3D(grid_dim_3D, fill_in_nearest_kernel, cell_data, grid, thin_shell);
    GArr3D<CellType> type_arr;
    GArr3D<int> ghost_idx_arr;
    type_arr.resize(grid_dim_3D);
    ghost_idx_arr.resize(grid_dim_3D);
    cuExecute3D(grid_dim_3D, cell_classfication_kernel, cell_data, grid, type_arr, ghost_idx_arr);
    cuExecute2D(dim2(grid.grid_dim, grid.grid_dim), post_cell_classification_kernel, grid, type_arr);
    thrust::inclusive_scan(thrust::device, ghost_idx_arr.begin(), ghost_idx_arr.end(), ghost_idx_arr.begin());
    cuExecute3D(grid_dim_3D, apply_cell_type_kernel, cell_data, grid, type_arr, ghost_idx_arr);
    int ghost_cell_num = ghost_idx_arr.data.last_item();
    type_arr.clear();
    ghost_idx_arr.clear();
    return ghost_cell_num;
}

};  // namespace pppm