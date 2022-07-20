#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <string>
#include "macro.h"
#include <cuda_runtime.h> // CUDA
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include "array3D.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "window.h"

namespace pppm
{
    void Window::called()
    {
        ImGui::Begin(title.c_str());
        update();
        ImGui::End();
    }
    inline __device__ uchar4 JetColor(float v, float vmin, float vmax)
    {
        float3 c = {1.0, 1.0, 1.0}; // white
        double dv;
        if (v < vmin)
            v = vmin;
        if (v > vmax)
            v = vmax;
        dv = vmax - vmin;
        if (v < (vmin + 0.25 * dv))
        {
            c.x = 0;
            c.y = 4 * (v - vmin) / dv;
        }
        else if (v < (vmin + 0.5 * dv))
        {
            c.x = 0;
            c.z = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
        }
        else if (v < (vmin + 0.75 * dv))
        {
            c.x = 4 * (v - vmin - 0.5 * dv) / dv;
            c.z = 0;
        }
        else
        {
            c.y = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
            c.z = 0;
        }
        return make_uchar4(c.x * 255, c.y * 255, c.z * 255, 255);
    }

#define UPSAMPLE_FACTOR 24
    __global__ void preprocess_image_data(GArr3D<float> in_data, GArr3D<uchar4> out_data, float data_max)
    {
        for (int x = blockIdx.x; x < out_data.rows; x += gridDim.x)
        {
            for (int y = threadIdx.x; y < out_data.cols; y += blockDim.x)
            {
                for (int b = 0; b < out_data.batchs; b++)
                {
                    float2 pos_f = make_float2(((float)x) / UPSAMPLE_FACTOR, ((float)y) / UPSAMPLE_FACTOR);
                    int2 pos = make_int2((int)pos_f.x, (int)pos_f.y);
                    float value = in_data(b, pos.x, pos.y);
                    uchar4 line_color = make_uchar4(170, 170, 170, 255);
                    uchar4 pixel_color = JetColor(value, -data_max, data_max);
                    float k = 3.5f;
                    pixel_color.x /= k;
                    pixel_color.y /= k;
                    pixel_color.z /= k;
                    float line_width = 0.03f;
                    float center = 0.5f - 0.5f / UPSAMPLE_FACTOR;
                    if (abs(pos_f.x - pos.x - center) >= 0.5 - line_width || abs(pos_f.y - pos.y - center) >= 0.5 - line_width)
                    {
                        out_data(b, x, y) = line_color;
                    }
                    else
                    {
                        out_data(b, x, y) = pixel_color;
                    }
                }
            }
        }
    }

    void CudaRender::setData(GArr3D<float> origin_data, float data_max)
    {
        data.resize(origin_data.batchs, origin_data.rows * UPSAMPLE_FACTOR, origin_data.cols * UPSAMPLE_FACTOR);
        if (data_max == -1)
        {
            auto min_ptr = thrust::min_element(thrust::device, origin_data.begin(), origin_data.end());
            auto max_ptr = thrust::max_element(thrust::device, origin_data.begin(), origin_data.end());
            GArr<float> min_gpu(min_ptr, 1);
            GArr<float> max_gpu(max_ptr, 1);
            float data_max = max_gpu.last_item();
            float data_min = min_gpu.last_item();
            if (-data_min > data_max)
            {
                data_max = -data_min;
            }
        }
        cuExecuteBlock(data.rows, 64, preprocess_image_data, origin_data, data, data_max);
        frame_num = data.batchs;
        width = data.rows;
        height = data.cols;
        frame_idx = 0;
        frame_idx_last = -1;
        update_frame_count = 0;
        play_speed = 0.5f;
    }

    void CudaRender::init()
    {
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &image);
        glBindTexture(GL_TEXTURE_2D, image);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        cudaGraphicsGLRegisterImage(&CudaResource, image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        cuSafeCall(cudaGraphicsMapResources(1, &CudaResource, 0));
        cuSafeCall(cudaGraphicsSubResourceGetMappedArray(&array, CudaResource, 0, 0));
    }
    void CudaRender::update()
    {
        update_frame_count += 1;
        if (frame_idx != frame_idx_last)
        {
            auto frame_data = data[frame_idx];
            cuSafeCall(cudaMemcpy2DToArray(array, 0, 0, frame_data.begin(), width * sizeof(uchar4),
                                           height * sizeof(uchar4), height, cudaMemcpyDeviceToDevice));
            frame_idx_last = frame_idx;
        }
        ImVec2 wsize = ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin();
        ImVec2 img_size = ImVec2(wsize.x, wsize.y - ImGui::GetFrameHeightWithSpacing() * 2);
        if (img_size.x < img_size.y)
        {
            img_size.y = img_size.x * height / width;
        }
        else
        {
            img_size.x = img_size.y * width / height;
        }
        ImGui::SetCursorPos((wsize - img_size) * 0.5f);
        ImGui::Image((ImTextureID)(uintptr_t)image, img_size, ImVec2(0, 1), ImVec2(1, 0));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetWindowSize().x * 0.15f);
        ImGui::SliderInt("Frame", &frame_idx, 0, frame_num - 1);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetWindowSize().x * 0.15f);
        ImGui::SliderFloat("Speed", &play_speed, 0.05f, 1.0f);
        if (ImGui::IsItemActive() && update_frame_count % (int)(1.0f / play_speed) == 0)
        {
            frame_idx += 1;
            if (frame_idx >= frame_num)
                frame_idx = 0;
        }
    }

    void CudaRender::clear()
    {
        cuSafeCall(cudaGraphicsUnmapResources(1, &CudaResource, 0));
    }

}