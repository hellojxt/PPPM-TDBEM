#pragma once
#include <cuda_runtime.h> // CUDA
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include "array3D.h"

namespace pppm
{
    class Window
    {
    public:
        std::string title;
        virtual void update() = 0;
        virtual void init() = 0;
        void called();
    };

    class CudaRender : public Window
    {
    public:
        GLuint image;
        cudaGraphicsResource_t CudaResource;
        cudaArray *array;
        GArr3D<uchar4> data;
        int width, height;
        int frame_num;
        int frame_idx;
        int frame_idx_last;
        int update_frame_count;
        float play_speed;
        CudaRender()
        {
            this->title = "CudaRender";
        }

        void setData(GArr3D<float> origin_data, float data_max = 1, float upsample_factor = 24);
        void init();
        void update();
        void clear();
    };

}