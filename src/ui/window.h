#pragma once
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>  // CUDA
#include <device_launch_parameters.h>
#include "array3D.h"
#include "helper_math.h"
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
        bool is_inited = false;
        CudaRender() { this->title = "CudaRender"; }
        CudaRender(const char *str) { this->title = str; }

        void set_overall_frame_num(int frame_num_) { frame_num = frame_num_; }

        void setData(GArr3D<float> origin_data,
                     float data_max = -1,
                     float line_width = 0.5f,
                     int start_batch = 0,
                     int end_batch = 0);

        void add_mesh_to_images(GArr<float3> vertices,
                                GArr<int3> triangles,
                                float3 min_pos,
                                float3 max_pos,
                                PlaneType plane,
                                float3 plane_pos,
                                float line_width = 3.0f,
                                int start_batch = 0,
                                int end_batch = 0);
        void init();
        void update();
        void clear();
};

}  // namespace pppm
