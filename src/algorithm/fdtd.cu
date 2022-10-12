#include "fdtd.h"

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

// credit : https://stackoverflow.com/a/9614511/15582103
__device__ void Solve4x4Linear(float a[4][4], float b[4], float result[4])
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

    // Should check for 0 determinant
    float invdet = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);

    result[0] = ( 
    ( a[1][1] * c5 - a[1][2] * c4 + a[1][3] * c3) * b[0] + // inv[0][0]
    (-a[0][1] * c5 + a[0][2] * c4 - a[0][3] * c3) * b[1] + // inv[0][1]
    ( a[3][1] * s5 - a[3][2] * s4 + a[3][3] * s3) * b[2] + // inv[0][2]
    (-a[2][1] * s5 + a[2][2] * s4 - a[2][3] * s3) * b[3]) * invdet;// inv[0][3] 

    result[1] = (
    (-a[1][0] * c5 + a[1][2] * c2 - a[1][3] * c1) * b[0] + // inv[1][0]
    ( a[0][0] * c5 - a[0][2] * c2 + a[0][3] * c1) * b[1] + // inv[1][1]
    (-a[3][0] * s5 + a[3][2] * s2 - a[3][3] * s1) * b[2] + // inv[1][2]
    ( a[2][0] * s5 - a[2][2] * s2 + a[2][3] * s1) * b[3]) * invdet;// inv[1][3]

    result[2] = (
    ( a[1][0] * c4 - a[1][1] * c2 + a[1][3] * c0) * b[0] + // inv[2][0]
    (-a[0][0] * c4 + a[0][1] * c2 - a[0][3] * c0) * b[1] + // inv[2][1]
    ( a[3][0] * s4 - a[3][1] * s2 + a[3][3] * s0) * b[2] + // inv[2][2]
    (-a[2][0] * s4 + a[2][1] * s2 - a[2][3] * s0) * b[3]) * invdet;// inv[2][3]

    result[3] = (
    (-a[1][0] * c3 + a[1][1] * c1 - a[1][2] * c0) * b[0] + // inv[3][0]
    ( a[0][0] * c3 - a[0][1] * c1 + a[0][2] * c0) * b[1] + // inv[3][1]
    (-a[3][0] * s3 + a[3][1] * s1 - a[3][2] * s0) * b[2] + // inv[3][2]
    ( a[2][0] * s3 - a[2][1] * s1 + a[2][2] * s0) * b[3]) * invdet;// inv[3][3]

    return;
}


#define POS 1
#define NEG 2
#define NPNN 0

#define GET_JUDGE(x) ((((int)((x) == 0)) << 1) + (((int)((x) == fdtd.res - 1))))

__device__ void fdtd_fill_matrix(FDTD& fdtd, int i, int j, int k)
{
    int xjudge = GET_JUDGE(i), yjudge = GET_JUDGE(j), zjudge = GET_JUDGE(k);
    int t = fdtd.t;
    float dt = fdtd.dt;
    float h = fdtd.dl;
    float c = fdtd.c;

    // coeff1/2是后三个方程会用到的，coeff3是第一个方程会用到的。
    float coeff1 = c * dt / (2 * h), coeff2 = c * c / (2 * h * h), coeff3 = 2 * coeff2 * (dt * dt);
    
    float a[4][4];
    float b[4];
    
    // 填充确定是常数的常数项和第一行的系数；因为第一个方程不管变量是什么系数都不变。
    b[0] = (6 * coeff3 - 2) * fdtd.grids[t - 1](i, j, k) + fdtd.grids[t - 2](i, j, k);
    b[1] = b[2] = b[3] = -fdtd.grids[t - 2](i, j, k) + (2 + 4 * coeff2) * fdtd.grids[t - 1](i, j, k);
    a[0][1] = a[0][2] = a[0][3] = coeff3;
    a[0][0] = -1;
    if(xjudge == NPNN) // i不实际具有需要解的变量，其方程无效
    {
        // 把第二列这一代表自己变量的列置为0，为了保持方程有解（即行列式不为0）在自己这一行保留一个1。
        a[0][1] = a[2][1] = a[3][1] = 0;
        a[1][1] = 1;
        // 其他方程可以减去i所代表的常数。
        float temp0 = (fdtd.grids[t - 1](i + 1, j, k) + fdtd.grids[t - 1](i - 1, j, k));
        float temp = coeff2 * temp0;
        b[0] -= coeff3 * temp0;
        b[2] -= temp;
        b[3] -= temp;
    }
    else if(xjudge == POS) // i+1是变量，方程有效。
    {
        // 把第二列填写上i+1对应的系数。
        a[1][1] = coeff1; a[2][1] = a[3][1] = coeff2;
        // 常数项挖去一部分；注意ab layer中0代表存储+x的buffer。
        b[0] -= fdtd.grids[t - 1](i - 1, j, k) * coeff3;
        b[1] -= coeff1 * (fdtd.grids[t - 2](i - 1, j, k) - fdtd.grids[t - 1](i - 1, j, k) - fdtd.absorbingLayer[t - 2](0, j, k));
        float temp = coeff2 * fdtd.grids[t - 1](i - 1, j, k);
        b[2] -= temp, b[3] -= temp;
    }
    else{ // xjudge == NEG;
        a[1][1] = -coeff1; a[2][1] = a[3][1] = coeff2;
        b[0] -= fdtd.grids[t - 1](i + 1, j, k) * coeff3;
        b[1] -= coeff1 * (fdtd.grids[t - 1](i + 1, j, k) - fdtd.grids[t - 2](i + 1, j, k) + fdtd.absorbingLayer[t - 2](3, j, k));
        float temp = coeff2 * fdtd.grids[t - 1](i + 1, j, k);
        b[2] -= temp, b[3] -= temp;
    }

    if(yjudge == NPNN)
    {
        a[0][2] = a[1][2] = a[3][2] = 0;
        a[2][2] = 1;

        float temp0 = (fdtd.grids[t - 1](i, j + 1, k) + fdtd.grids[t - 1](i, j - 1, k));
        float temp = coeff2 * temp0;
        b[0] -= coeff3 * temp0;
        b[1] -= temp;
        b[3] -= temp;
    }
    else if(yjudge == POS)
    {
        a[2][2] = coeff1; a[1][2] = a[3][2] = coeff2;
        b[0] -= fdtd.grids[t - 1](i, j - 1, k) * coeff3;
        b[2] -= coeff1 * (fdtd.grids[t - 2](i, j - 1, k) - fdtd.grids[t - 1](i, j - 1, k) - fdtd.absorbingLayer[t - 2](1, i, k));
        float temp = coeff2 * fdtd.grids[t - 1](i, j - 1, k);
        b[1] -= temp, b[3] -= temp;
    }
    else{ // yjudge == NEG;
        a[2][2] = -coeff1; a[1][2] = a[3][2] = coeff2;
        b[0] -= fdtd.grids[t - 1](i, j + 1, k) * coeff3;
        b[2] -= coeff1 * (fdtd.grids[t - 1](i, j + 1, k) - fdtd.grids[t - 2](i, j + 1, k) + fdtd.absorbingLayer[t - 2](4, i, k));
        float temp = coeff2 * fdtd.grids[t - 1](i, j + 1, k);
        b[1] -= temp, b[3] -= temp;
    }

    if(zjudge == NPNN)
    {
        a[0][3] = a[1][3] = a[2][3] = 0;
        a[3][3] = 1;

        float temp0 = (fdtd.grids[t - 1](i, j, k + 1) + fdtd.grids[t - 1](i, j, k - 1));
        float temp = coeff2 * temp0;
        b[0] -= coeff3 * temp0;
        b[1] -= temp;
        b[3] -= temp;
    }
    else if(zjudge == POS)
    {
        a[3][3] = coeff1; a[1][3] = a[2][3] = coeff2;
        b[0] -= fdtd.grids[t - 1](i, j, k - 1) * coeff3;
        b[3] -= coeff1 * (fdtd.grids[t - 2](i, j, k - 1) - fdtd.grids[t - 1](i, j, k - 1) - fdtd.absorbingLayer[t - 2](2, i, j));
        float temp = coeff2 * fdtd.grids[t - 1](i, j, k - 1);
        b[1] -= temp, b[2] -= temp;
    }
    else{ // zjudge == NEG;
        a[3][3] = -coeff1; a[1][3] = a[2][3] = coeff2;
        b[0] -= fdtd.grids[t - 1](i, j, k + 1) * coeff3;
        b[3] -= coeff1 * (fdtd.grids[t - 1](i, j, k + 1) - fdtd.grids[t - 2](i, j, k + 1) + fdtd.absorbingLayer[t - 2](5, i, j));
        float temp = coeff2 * fdtd.grids[t - 1](i, j, k + 1);
        b[1] -= temp, b[2] -= temp;
    }


    // The matrix has been prepared, solve it.
    float result[4];
    Solve4x4Linear(a, b, result);
    fdtd.grids[t](i, j, k) = result[0];
    if(xjudge == POS)
        fdtd.absorbingLayer[t - 1](0, j, k) = result[1];
    else if(xjudge == NEG)
        fdtd.absorbingLayer[t - 1](3, j, k) = result[1];
    
    if(yjudge == POS)
        fdtd.absorbingLayer[t - 1](1, i, k) = result[2];
    else if(yjudge == NEG)
        fdtd.absorbingLayer[t - 1](4, i, k) = result[2];

    if(zjudge == POS)
        fdtd.absorbingLayer[t - 1](2, i, j) = result[3];
    else if(zjudge == NEG)
        fdtd.absorbingLayer[t - 1](5, i, j) = result[3];

    return;
}

__global__ void fdtd_boundary_kernel(FDTD fdtd)
{
    int offsetx = threadIdx.x + blockIdx.x * blockDim.x,
        offsety = threadIdx.y + blockIdx.y * blockDim.y;
    int last = fdtd.res - 1;
    fdtd_fill_matrix(fdtd, 0, offsetx, offsety);
    fdtd_fill_matrix(fdtd, last, offsetx, offsety);
    fdtd_fill_matrix(fdtd, offsetx, 0, offsety);
    fdtd_fill_matrix(fdtd, offsetx, last, offsety);
    fdtd_fill_matrix(fdtd, offsetx, offsety, 0);
    fdtd_fill_matrix(fdtd, offsetx, offsety, last);
    return;
}

void FDTD::step_inner_grid()
{
    cuExecute3D(dim3(res - 2, res - 2, res - 2), fdtd_inner_kernel, *this);
}

void FDTD::step_boundary_grid() {
    cuExecuteBlock(res, res, fdtd_boundary_kernel, *this);
}

}  // namespace pppm