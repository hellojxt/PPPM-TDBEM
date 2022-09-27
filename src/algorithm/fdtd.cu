#include "fdtd.h"

namespace pppm
{

void FDTD::init(int res_, float dl_, float dt_)
{
    res = res_;
    for (int i = 0; i < GRID_TIME_SIZE; i++)
    {
        grids[i].resize(res, res, res);
        grids[i].reset();
    }
    t = -1;
    dl = dl_;
    dt = dt_;
    c = 343.2f;
}

GPU_FUNC inline value laplacian(GArr3D<value> &grid, int i, int j, int k, float h)
{
    float sum = 0;
    sum += grid(i - 1, j, k);
    sum += grid(i + 1, j, k);
    sum += grid(i, j - 1, k);
    sum += grid(i, j + 1, k);
    sum += grid(i, j, k - 1);
    sum += grid(i, j, k + 1);
    sum -= 6 * grid(i, j, k);
    return sum / (h * h);
}

__global__ void fdtd_inner_kernel(FDTD fdtd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i >= fdtd.res - 1 || i <= 0 || j >= fdtd.res - 1 || j <= 0 || k >= fdtd.res - 1 || k <= 0)
        return;
    float t = fdtd.t;
    float h = fdtd.dl;
    float dt = fdtd.dt;
    float c = fdtd.c;
    fdtd.grids[t](i, j, k) = 2 * fdtd.grids[t - 1](i, j, k) +
                             (c * c * dt * dt) * laplacian(fdtd.grids[t - 1], i, j, k, h) - fdtd.grids[t - 2](i, j, k);
}

/*
 * t++ first, then compute the FDTD kernel
 */
void FDTD::step()
{
    t++;
    cuExecute3D(dim3(res - 2, res - 2, res - 2), fdtd_inner_kernel, *this);
}

void FDTD::clear()
{
    for (int i = 0; i < GRID_TIME_SIZE; i++)
    {
        grids[i].clear();
    }
}

}  // namespace pppm