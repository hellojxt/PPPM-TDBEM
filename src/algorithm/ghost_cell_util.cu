#include "ghost_cell_util.h"
#include "helper_math.h"
namespace pppm
{

__global__ void fill_in_nearest_kernel(GArr3D<CellInfo> cell_data, ParticleGrid grid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = grid.grid_dim.x;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;

    float3 grid_center = grid.getCenter(x, y, z);
    CellInfo result;
    result.type = AIR;
    result.nearst_distance = MAX_FLOAT;

    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids
            {
                int3 coord = make_int3(x + dx, y + dy, z + dz);
                if (coord.x < 0 || coord.x >= grid_dim || coord.y < 0 || coord.y >= grid_dim || coord.z < 0 ||
                    coord.z >= grid_dim)
                    continue;

                Range &neighbors = grid.grid_hash_map(coord);
                for (int i = neighbors.start; i < neighbors.end; i++)
                {
                    int3 triID = grid.particles[i].indices;
                    float3 nearest_point = get_nearest_triangle_point(grid_center, grid.vertices[triID.x],
                                                                      grid.vertices[triID.y], grid.vertices[triID.z]);
                    float curr_len = length(grid_center - nearest_point);
                    if (curr_len < result.nearst_distance)
                    {
                        result.nearst_distance = curr_len;
                        result.nearst_point = nearest_point;
                        result.nearest_particle = grid.particles[i];
                    }
                }
            }
    float direction = dot(result.nearest_particle.normal, grid_center - result.nearst_point);
    result.type = (direction > 0) ? AIR : SOLID;
    cell_data(x, y, z) = result;
    return;
}

__global__ void cell_classfication_kernel(GArr3D<CellInfo> cell_data, ParticleGrid grid, GArr3D<CellType> type_arr)
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
}

__global__ void apply_cell_type_kernel(GArr3D<CellInfo> cell_data, ParticleGrid grid, GArr3D<CellType> type_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = cell_data.size.x;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;
    cell_data(x, y, z).type = type_arr(x, y, z);
}

void fill_cell_data(ParticleGrid grid, GArr3D<CellInfo> cell_data)
{
    cuExecute3D(grid.grid_dim, fill_in_nearest_kernel, cell_data, grid);
    GArr3D<CellType> type_arr;
    type_arr.resize(grid.grid_dim);
    cuExecute3D(grid.grid_dim, cell_classfication_kernel, cell_data, grid, type_arr);
    cuExecute3D(grid.grid_dim, apply_cell_type_kernel, cell_data, grid, type_arr);
    type_arr.clear();
}

};  // namespace pppm