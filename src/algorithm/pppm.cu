#include "pppm.h"

namespace pppm
{

    __global__ void cell_near_field_kernel(GArr3D<Range> grid_hash_map, GArr2D<BoundaryData> particle_data, GArr3D<cpx> near_field){
        int x = blockIdx.x / grid_hash_map.size.y;
        int y = blockIdx.x % grid_hash_map.size.y;
        int z = threadIdx.x;
        if (x >= grid_hash_map.size.x || y >= grid_hash_map.size.y || z >= grid_hash_map.size.z) return;
        cpx near_field_value = 0;
        for (int dx = -1; dx <= 1; dx++){
            for (int dy = -1; dy <= 1; dy++){
                for (int dz = -1; dz <= 1; dz++){
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if (nx < 0 || nx >= grid_hash_map.size.x || ny < 0 || ny >= grid_hash_map.size.y || nz < 0 || nz >= grid_hash_map.size.z) continue;
                    Range range = grid_hash_map(nx, ny, nz);
                    for (int i = range.start; i < range.end; i++){
                        
                    }
                }
            }
        }
    }


    void PPPMSolver::step()
    {
        // step fdtd simulation
        fdtd.step();
        // approximate the near field of each grid cell after fdtd simulation
        

        // get far field by subtracting near field from total field of each grid cell


        // update particle far field (interpolate the far field to particle positions)
        


        // update particle near field (using neighbor particles)


        // correct fdtd result at current step by updating the near field of each grid cell


    }

}