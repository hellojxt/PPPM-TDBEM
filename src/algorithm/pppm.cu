#include "pppm.h"

namespace pppm
{

    __global__ void get_grid_far_field(FDTD fdtd, ParticleGrid pg, TDBEM bem, GArr<BoundaryHistory> particle_history, GArr3D<float> far_field){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;


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