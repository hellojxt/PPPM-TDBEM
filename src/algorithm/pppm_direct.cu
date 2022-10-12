#include "pppm_direct.h"

namespace pppm
{

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

void direct_correction_fdtd_near(PPPMSolver &pppm)
{
    int res = pppm.fdtd.res;
    cuExecute3D(dim3(res, res, res), correction_fdtd_near_field_kernel, pppm);
}

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
    // fdtd_near_field += (c * c * dt * dt) * laplacian_near_field(pppm, coord, coord, t - 1);
    // fdtd_near_field -= near_field(pppm, coord, coord, t - 2);
    pppm.far_field[t](coord) = pppm.fdtd.grids[t](coord) - fdtd_near_field;
}

void direct_fdtd_far(PPPMSolver &pppm)
{
    int res = pppm.fdtd.res;
    auto his = pppm.particle_history;
    cuExecute3D(dim3(res, res, res), solve_far_field_kernel, pppm);
}

}  // namespace pppm