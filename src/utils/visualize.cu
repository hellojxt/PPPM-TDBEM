#include "visualize.h"

namespace pppm
{
__global__ void copy_kernel(GArr3D<float> src, GArr3D<float> dst, int t, int3 face)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst.size.y || y >= dst.size.z)
        return;
    int3 e[3] = {make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1)};
    int idx = (face.y != 0) + (face.z != 0) * 2;
    int3 pos = face + e[(idx + 1) % 3] * x + e[(idx + 2) % 3] * y;
    dst(t, x, y) = src(pos);
}

void RenderElement::assign(int idx, GArr3D<float> src)
{
    cuExecute2D(dim2(data.rows, data.cols), copy_kernel, src, data, idx, plane);
}

}  // namespace pppm
