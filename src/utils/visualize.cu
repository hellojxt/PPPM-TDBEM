#include "visualize.h"

namespace pppm
{
__global__ void copy_kernel(GArr3D<float> src, GArr3D<float> dst, int t, int3 face)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst.size.y || y >= dst.size.z)
        return;
    int3 ex = make_int3(1, 0, 0);
    int3 ey = make_int3(0, 1, 0);
    int3 ez = make_int3(0, 0, 1);
    int3 pos = face;
    if (face.x != 0)
        pos += ey * x + ez * y;
    else if (face.y != 0)
        pos += ex * x + ez * y;
    else if (face.z != 0)
        pos += ex * x + ey * y;
    else
        return;
    dst(t, x, y) = src(pos);
}

void RenderElement::assign(int idx, GArr3D<float> src)
{
    cuExecute2D(dim2(data.rows, data.cols), copy_kernel, src, data, idx, plane);
}

}  // namespace pppm
