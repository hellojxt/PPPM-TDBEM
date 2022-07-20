#include "fdtd.h"

namespace pppm
{

    __global__ void copy_clip_kernel(FDTD fdtd, GArr3D<float> data, cpx_phase phase, int clip_idx)
    {
        int x = blockIdx.x;
        int y = threadIdx.x;
        int n = fdtd.t % GRID_TIME_SIZE;
        if (x < fdtd.res && y < fdtd.res)
        {
            int z = clip_idx;
            float real = fdtd.grids[n](x, y, z).real();
            float imag = fdtd.grids[n](x, y, z).imag();
            if (phase == CPX_REAL)
            {
                data(fdtd.t, x, y) = real;
            }
            else if (phase == CPX_IMAG)
            {
                data(fdtd.t, x, y) = imag;
            }
            else if (phase == CPX_ABS)
            {
                data(fdtd.t, x, y) = sqrt(real * real + imag * imag);
            }
        }
    }

    GPU_FUNC cpx laplacian(GArr3D<cpx> &grid, int i, int j, int k, float h)
    {
        cpx sum = 0;
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
        int n2 = fdtd.t % GRID_TIME_SIZE;
        int n1 = (fdtd.t - 1 + GRID_TIME_SIZE) % GRID_TIME_SIZE;
        int n0 = (fdtd.t - 2 + GRID_TIME_SIZE) % GRID_TIME_SIZE;
        float h = fdtd.dl;
        float dt = fdtd.dt;
        float c = fdtd.c;
        fdtd.grids[n2](i, j, k) = 2 * fdtd.grids[n1](i, j, k) + (c * c * dt * dt) * laplacian(fdtd.grids[n1], i, j, k, h) - fdtd.grids[n0](i, j, k);
    }

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

    
    void FDTD::update()
    {
        t++;
        cuExecuteBlock((res - 2) * (res - 2), 64, fdtd_inner_kernel, *this);
    }

    void FDTD::copy_clip(GArr3D<float> &data, int clip_idx, cpx_phase phase)
    {
        if (clip_idx == -1)
        {
            clip_idx = res / 2;
        }
        cuExecuteBlock(res, res, copy_clip_kernel, *this, data, phase, clip_idx);
    }

}