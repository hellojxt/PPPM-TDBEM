#include "pppm.h"
#include "pppm_kernel.h"

namespace pppm
{

__global__ void solve_far_field_kernel(PPPMSolver pppm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int res = pppm.fdtd.res;
    if (x >= res - 1 || x < 1 || y >= res - 1 || y < 1 || z >= res - 1 || z < 1)
        return;
    int3 coord = make_int3(x, y, z);
    int t = pppm.fdtd.t;
    float c = pppm.fdtd.c, dt = pppm.fdtd.dt;
    float fdtd_near_field = 0;
    fdtd_near_field += 2 * near_field(pppm, coord, coord, t - 1);
    fdtd_near_field += (c * c * dt * dt) * laplacian_near_field(pppm, coord, coord, t - 1);
    fdtd_near_field -= near_field(pppm, coord, coord, t - 2);
    pppm.far_field[t](coord) = pppm.fdtd.grids[t](coord) - fdtd_near_field;
}

__global__ void correction_fdtd_near_field_kernel(PPPMSolver pppm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int res = pppm.fdtd.res;
    if (x >= res - 1 || x < 1 || y >= res - 1 || y < 1 || z >= res - 1 || z < 1)
        return;
    int3 coord = make_int3(x, y, z);
    int t = pppm.fdtd.t;
    float accurate_near_field = near_field(pppm, coord, coord, t);
    pppm.fdtd.grids[t](coord) = pppm.far_field[t](coord) + accurate_near_field;
}

void PPPMSolver::solve_fdtd_simple()
{
    TICK(near_field);
    cuExecute3D(dim3(fdtd.res, fdtd.res, fdtd.res), correction_fdtd_near_field_kernel,
                *this);  // correct fdtd result at last step by updating the near field of each grid cell
    TOCK(near_field);
    TICK(fdtd);
    fdtd.step();
    TOCK(fdtd);
    TICK(fdtd_correction);
    cuExecute3D(dim3(fdtd.res, fdtd.res, fdtd.res), solve_far_field_kernel,
                *this);  // get far field by subtracting near field from total field of each grid cell
    TOCK(fdtd_correction);
}

void PPPMSolver::update_particle_data() {}

void PPPMSolver::step()
{
    solve_fdtd_simple();
    update_particle_data();
}

}  // namespace pppm