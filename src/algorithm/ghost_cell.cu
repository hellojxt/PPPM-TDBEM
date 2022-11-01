#include "ghost_cell.h"

namespace pppm
{
    __device__ float GetMagnitude(float3 vec)
    {
        return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    }

    __global__ void fill_in_nearest_kernel(GArr<int3> &center2Indices, ParticleGrid &grid)
    {
        int id = threadIdx.x;
        auto gridParticleID = grid.grid_dense_map[id].start;
        auto &gridParticle = grid.particles[gridParticleID];

        float3 pos = gridParticle.pos;
        int3 gridCellID = make_int3((pos - grid.min_pos) / grid.grid_size);
        int3 gridLimit = grid.grid_dim;

        float minMagnitude = 1e5f;
        int3 result;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) // iterate over all the 3x3x3 grids
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;
                    int3 coord = make_int3(gridCellID.x + dx, gridCellID.y + dy, gridCellID.z + dz);
                    if (coord.x < 0 || coord.x >= gridLimit.x ||
                        coord.y < 0 || coord.y >= gridLimit.y ||
                        coord.z < 0 || coord.z >= gridLimit.z)
                        continue;

                    Range &neighbors = grid.grid_hash_map(coord);
                    for (int i = neighbors.start; i < neighbors.end; i++)
                    {
                        auto currParticlePos = grid.particles[i].pos;
                        auto currMagnitude = GetMagnitude(currParticlePos - pos);
                        if (currMagnitude < minMagnitude)
                        {
                            result = grid.particles[i].indices;
                            minMagnitude = currMagnitude;
                        }
                    }
                }

        center2Indices[id] = result;
        return;
    }

    void GhostCells::fill_in_nearest()
    {
        int size = center2Indices.size();
        fill_in_nearest_kernel<<<1, size>>>(center2Indices, grid);
        return;
    };

}