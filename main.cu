#include "gui.h"
#include "window.h"

using namespace pppm;


__global__ void kernel(GArr3D<float> data)
{
    for (int x = blockIdx.x; x < data.rows; x += gridDim.x)
    {
        for (int y = threadIdx.x; y < data.cols; y += blockDim.x)
        {
            for (int b = 0; b < data.batchs; b++)
            {
                data(b, x, y) = (float) b - 50;
            }
            
        }
    }
}

int main(){
    GUI gui;
    CudaRender render;
    GArr3D<float> data;
    data.resize(100, 32, 32);
    cuExecuteBlock(32, 64, kernel, data);
    render.setData(data);
    gui.append(&render);
    gui.start();
    return 0;
}