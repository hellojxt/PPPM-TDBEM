#include "gui.h"
#include "window.h"

using namespace pppm;


__global__ void kernel(GArr3D<uchar4> data)
{
    for (int x = blockIdx.x; x < data.rows; x += gridDim.x)
    {
        for (int y = threadIdx.x; y < data.cols; y += blockDim.x)
        {
            for (int b = 0; b < data.batchs; b++)
            {
                data(b, x, y) = make_uchar4(0, b, 0, 255);
            }
            
        }
    }
}

int main(){
    GUI gui;
    CudaRender render;
    GArr3D<uchar4> data;
    data.resize(100, 512, 512);
    cuExecuteBlock(512, 64, kernel, data);
    render.setData(data);
    gui.append(&render);
    gui.start();
    return 0;
}