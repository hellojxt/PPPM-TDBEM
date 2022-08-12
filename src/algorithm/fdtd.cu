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
        t = 0;
        dl = dl_;
        dt = dt_;
        c = 343.2f;
    }

    GPU_FUNC float laplacian(GArr3D<float> &grid, int i, int j, int k, float h)
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
        if (blockIdx.x >= (fdtd.res - 2) * (fdtd.res - 2) || threadIdx.x >= (fdtd.res - 2))
            return;
        int i = blockIdx.x / (fdtd.res - 2) + 1;
        int j = blockIdx.x % (fdtd.res - 2) + 1;
        int k = threadIdx.x + 1;
        int n2 = fdtd.getGridIndex(0);
        int n1 = fdtd.getGridIndex(-1);
        int n0 = fdtd.getGridIndex(-2);
        float h = fdtd.dl;
        float dt = fdtd.dt;
        float c = fdtd.c;
        fdtd.grids[n2](i, j, k) = 2 * fdtd.grids[n1](i, j, k) + (c * c * dt * dt) * laplacian(fdtd.grids[n1], i, j, k, h) - fdtd.grids[n0](i, j, k);
    }

    void FDTD::update()
    {
        t++;
        cuExecuteBlock((res - 2) * (res - 2), (res - 2), fdtd_inner_kernel, *this);
    }

    void FDTD::clear()
    {
        for (int i = 0; i < GRID_TIME_SIZE; i++)
        {
            grids[i].clear();
        }
    }

}