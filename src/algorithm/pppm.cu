#include "pppm.h"

namespace pppm
{

    __global__ void get_grid_far_field(FDTD fdtd, ParticleGrid pg, TDBEM bem, GArr<BoundaryHistory> particle_history, GridArr far_field)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= fdtd.res - 1 || x < 1 || y >= fdtd.res - 1 || y < 1 || z >= fdtd.res - 1 || z < 1)
            return;
        float far_field_value = 0;
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;
                    int3 coord = make_int3(x + dx, y + dy, z + dz);
                    float3 grid_center = fdtd.getCenter(coord);
                    Range r = pg.grid_hash_map(coord);
                    float near_field_value = 0;
                    for (int i = r.start; i < r.end; i++)
                    {
                        BElement &e = pg.particles[i];
                        near_field_value += bem.laplace(
                            pg.vertices.data(),
                            PairInfo(e.indices, grid_center),
                            particle_history[i].neumann,
                            particle_history[i].dirichlet,
                            fdtd.t);
                    }
                }
            }
        }
    }

    void PPPMSolver::step()
    {
        // update particle far field (interpolate the far field to particle positions)

        // update particle near field (using neighbor particles)

        // correct fdtd result at current step by updating the near field of each grid cell

        // step fdtd simulation
        fdtd.step();

        // get far field by subtracting near field from total field of each grid cell
    }

}