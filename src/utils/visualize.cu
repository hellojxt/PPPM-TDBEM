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
    current_data_idx = idx;
}

__global__ void get_time_signal_kernel(GArr3D<float> data, GArr<float> signal, int x, int y)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= data.size.x)
        return;
    signal[t_idx] = data(t_idx, x, y);
}

GArr<float> RenderElement::get_time_siganl(int x, int y)
{
    GArr<float> signal(data.size.x);
    cuExecute(data.size.x, get_time_signal_kernel, data, signal, x, y);
    return signal;
}

void RenderElement::write_image(int idx, std::string filename)
{
    update_mesh(); // 这次update_mesh调用把data从1080*1080变成了72*72并且没有释放内存！
    auto image = render_window.data[idx].cpu();
    write_to_png(filename, image);
}

}  // namespace pppm
