#include "fdtd.h"
#include "helper_math.h"
namespace pppm
{

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= fdtd.res - 2 || i <= 1 || j >= fdtd.res - 2 || j <= 1 || k >= fdtd.res - 2 || k <= 1)
        return;
    int t = fdtd.t;
    float h = fdtd.dl, dt = fdtd.dt, c = fdtd.c;
    fdtd.grids[t + 1](i, j, k) = 2 * fdtd.grids[t](i, j, k) + (c * c * dt * dt) * laplacian(fdtd.grids[t], i, j, k, h) -
                                 fdtd.grids[t - 1](i, j, k);
}

// credit : https://stackoverflow.com/a/9614511/15582103
GPU_FUNC inline void Solve4x4Linear(float a[4][4], float b[4], float result[4])
{

    float s0 = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    float s1 = a[0][0] * a[1][2] - a[1][0] * a[0][2];
    float s2 = a[0][0] * a[1][3] - a[1][0] * a[0][3];
    float s3 = a[0][1] * a[1][2] - a[1][1] * a[0][2];
    float s4 = a[0][1] * a[1][3] - a[1][1] * a[0][3];
    float s5 = a[0][2] * a[1][3] - a[1][2] * a[0][3];

    float c5 = a[2][2] * a[3][3] - a[3][2] * a[2][3];
    float c4 = a[2][1] * a[3][3] - a[3][1] * a[2][3];
    float c3 = a[2][1] * a[3][2] - a[3][1] * a[2][2];
    float c2 = a[2][0] * a[3][3] - a[3][0] * a[2][3];
    float c1 = a[2][0] * a[3][2] - a[3][0] * a[2][2];
    float c0 = a[2][0] * a[3][1] - a[3][0] * a[2][1];

    float invdet = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
    float inv[4][4];
    inv[0][0] = (a[1][1] * c5 - a[1][2] * c4 + a[1][3] * c3) * invdet;
    inv[0][1] = (-a[0][1] * c5 + a[0][2] * c4 - a[0][3] * c3) * invdet;
    inv[0][2] = (a[3][1] * s5 - a[3][2] * s4 + a[3][3] * s3) * invdet;
    inv[0][3] = (-a[2][1] * s5 + a[2][2] * s4 - a[2][3] * s3) * invdet;
    inv[1][0] = (-a[1][0] * c5 + a[1][2] * c2 - a[1][3] * c1) * invdet;
    inv[1][1] = (a[0][0] * c5 - a[0][2] * c2 + a[0][3] * c1) * invdet;
    inv[1][2] = (-a[3][0] * s5 + a[3][2] * s2 - a[3][3] * s1) * invdet;
    inv[1][3] = (a[2][0] * s5 - a[2][2] * s2 + a[2][3] * s1) * invdet;
    inv[2][0] = (a[1][0] * c4 - a[1][1] * c2 + a[1][3] * c0) * invdet;
    inv[2][1] = (-a[0][0] * c4 + a[0][1] * c2 - a[0][3] * c0) * invdet;
    inv[2][2] = (a[3][0] * s4 - a[3][1] * s2 + a[3][3] * s0) * invdet;
    inv[2][3] = (-a[2][0] * s4 + a[2][1] * s2 - a[2][3] * s0) * invdet;
    inv[3][0] = (-a[1][0] * c3 + a[1][1] * c1 - a[1][2] * c0) * invdet;
    inv[3][1] = (a[0][0] * c3 - a[0][1] * c1 + a[0][2] * c0) * invdet;
    inv[3][2] = (-a[3][0] * s3 + a[3][1] * s1 - a[3][2] * s0) * invdet;
    inv[3][3] = (a[2][0] * s3 - a[2][1] * s1 + a[2][2] * s0) * invdet;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        result[i] = 0;
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            result[i] += inv[i][j] * b[j];
        }
    }
}

#define GET_JUDGE(x) (-((int)((x) == 1)) + ((int)((x) == fdtd.res - 2)))
#define X_IDX(x) ((x))
#define Y_IDX(x) (((x) + 1) % 3)
#define Z_IDX(x) (((x) + 2) % 3)

__device__ void ab_condition_solve(FDTD &fdtd, int i, int j, int k)
{
    int t = fdtd.t;
    float dt = fdtd.dt, h = fdtd.dl, c = fdtd.c;
    int judge[3] = {GET_JUDGE(i), GET_JUDGE(j), GET_JUDGE(k)};  // -1 for negative, 0 for middle, 1 for positive
    int3 d_coord[3] = {make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1)};
    int3 coord = make_int3(i, j, k);

    float a[4][4], b[4];
    // for wave equation (4th equation with index 3)
    float coef = c * c * dt * dt / (h * h);
    a[3][3] = 1;
#pragma unroll
    for (int dim = 0; dim < 3; dim++)
        a[3][dim] = -coef * (judge[dim] != 0);
    b[3] = -fdtd.grids[t - 1](coord) + 2 * fdtd.grids[t](coord) - coef * (6 * fdtd.grids[t](coord));
#pragma unroll
    for (int dim = 0; dim < 3; dim++)
        b[3] += coef * fdtd.grids[t](coord + d_coord[dim]) * (judge[dim] <= 0) +
                coef * fdtd.grids[t](coord - d_coord[dim]) * (judge[dim] >= 0);

    // for ab condition (index 0, 1, 2)
    float coef1 = 1 / (dt * dt), coef2 = c / (2 * h * dt), coef3 = -(c * c) / (2 * h * h);
#pragma unroll
    for (int idx = 0; idx < 3; idx++)
    {
        int x = X_IDX(idx), y = Y_IDX(idx), z = Z_IDX(idx);
        a[idx][3] = coef1;
        a[idx][x] = coef2;
        a[idx][y] = coef3 * (judge[y] != 0);
        a[idx][z] = coef3 * (judge[z] != 0);
        b[idx] = -coef1 * (fdtd.grids[t - 1](coord) - 2 * fdtd.grids[t](coord)) +
                 -coef2 * (fdtd.grids[t](coord + d_coord[x]) * (-(judge[x] < 0)) +
                           fdtd.grids[t](coord - d_coord[x]) * (-(judge[x] > 0)) +
                           fdtd.grids[t - 1](coord + d_coord[x]) * (-judge[x]) +
                           fdtd.grids[t - 1](coord - d_coord[x]) * judge[x]) +
                 -coef3 * (fdtd.grids[t](coord + d_coord[y]) * (judge[y] <= 0) +
                           fdtd.grids[t](coord - d_coord[y]) * (judge[y] >= 0) +
                           fdtd.grids[t](coord + d_coord[z]) * (judge[z] <= 0) +
                           fdtd.grids[t](coord - d_coord[z]) * (judge[z] >= 0) - 4 * fdtd.grids[t](coord));
    }
    // solve linear equation
    float result[4];
    Solve4x4Linear(a, b, result);
    fdtd.grids[t + 1](coord) = result[3];
#pragma unroll
    for (int dim = 0; dim < 3; dim++)
        if (judge[dim] != 0)
            fdtd.grids[t](coord + d_coord[dim] * judge[dim]) = result[dim];
}

__global__ void fdtd_boundary_kernel(FDTD fdtd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = i % 6;
    i = i / 6;
    if (i >= fdtd.res - 1 || i == 0 || j >= fdtd.res - 1 || j == 0)
        return;
    int params[6][3] = {{i, j, 1}, {i, j, fdtd.res - 2}, {i, 1, j}, {i, fdtd.res - 2, j},
                        {1, i, j}, {fdtd.res - 2, i, j}};
    ab_condition_solve(fdtd, params[offset][0], params[offset][1], params[offset][2]);
}

void FDTD::step_inner_grid()
{
    cuExecute3D(dim3(res, res, res), fdtd_inner_kernel, *this);
}

void FDTD::step_boundary_grid()
{
    cuExecute2D(dim2(6 * res, res), fdtd_boundary_kernel, *this);
}

}  // namespace pppm