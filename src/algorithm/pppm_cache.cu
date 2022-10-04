#include "macro.h"
#include "pppm_cache.h"

namespace pppm
{
__global__ void get_involved_grid(PPPMSolver pppm, PPPMCache cache)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int res = pppm.fdtd.res;
    if (x >= res - 1 || x < 1 || y >= res - 1 || y < 1 || z >= res - 1 || z < 1)
        return;
    int neighbor_particle_num = 0;
    int3 center = make_int3(x, y, z);
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids
            {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                int3 coord = make_int3(x + dx, y + dy, z + dz);
                Range r = pppm.pg.grid_hash_map(coord);
                neighbor_particle_num += r.length();
            }
    cache.grid_neighbor_num(center) = neighbor_particle_num;
    cache.grid_neighbor_nonzero(center) = neighbor_particle_num > 0;
}

__global__ void fill_grid_info(PPPMSolver pppm, PPPMCache cache)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int res = pppm.fdtd.res;
    if (x >= res - 1 || x < 1 || y >= res - 1 || y < 1 || z >= res - 1 || z < 1)
        return;

    int3 center = make_int3(x, y, z);
    int range_end = cache.grid_neighbor_num(center);
    int range_start = range_end;

    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids
            {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                int3 coord = make_int3(x + dx, y + dy, z + dz);
                Range r = pppm.pg.grid_hash_map(coord);
                for (int i = r.start; i < r.end; i++)
                {
                    range_start--;
                    cache.grid_data[range_start].particle_id = i;       // fill particle id
                    cache.grid_fdtd_data[range_start].particle_id = i;  // fill particle id
                }
            }
    if (range_start != range_end)
    {
        int grid_index = cache.grid_neighbor_nonzero(center) - 1;
        cache.grid_map[grid_index] = GridMap(center, Range(range_start, range_end));
    }
}

void set_cache_grid_size(PPPMSolver &pppm)
{
    int res = pppm.fdtd.res;
    PPPMCache &cache = pppm.cache;
    cache.grid_neighbor_nonzero.resize(res, res, res);
    cache.grid_neighbor_nonzero.reset();
    cache.grid_neighbor_num.resize(res, res, res);
    cache.grid_neighbor_num.reset();
    cuExecute3D(dim3(res, res, res), get_involved_grid, pppm, cache);
    thrust::inclusive_scan(thrust::device, cache.grid_neighbor_num.begin(), cache.grid_neighbor_num.end(),
                           cache.grid_neighbor_num.begin(),
                           thrust::plus<int>());               // calculate the prefix sum of neighbor_nums
    int total_num = cache.grid_neighbor_num.data.last_item();  // last item is the total number of neighbor particles
    thrust::inclusive_scan(thrust::device, cache.grid_neighbor_nonzero.begin(), cache.grid_neighbor_nonzero.end(),
                           cache.grid_neighbor_nonzero.begin(),
                           thrust::plus<int>());  // calculate the prefix sum of non-zero-neighbor particle
    int total_nonzero =
        cache.grid_neighbor_nonzero.data.last_item();  // last item is the total number of particles have neighbors
    cache.grid_map.resize(total_nonzero);
    cache.grid_data.resize(total_num);
    cache.grid_fdtd_data.resize(total_num);
    cuExecute3D(dim3(res, res, res), fill_grid_info, pppm, cache);
}

__global__ void precompute_grid_data(PPPMSolver pppm, PPPMCache cache)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= cache.grid_map.size())
        return;
    GridMap grid_map = cache.grid_map[grid_id];
    Range r = grid_map.range;
    for (int i = r.start; i < r.end; i++)
    {
        BEMCache e;
        e.particle_id = cache.grid_data[i].particle_id;
    }
}

void cache_grid_data(PPPMSolver &pppm, PPPMCache &cache) {}

}  // namespace pppm