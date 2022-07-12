#pragma once
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

namespace pppm
{
    class Window
    {
    public:
        std::string title;
        virtual void update() = 0;
        virtual void init() = 0;
        void called()
        {
            ImGui::Begin(title.c_str());
            update();
            ImGui::End();
        }
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
        void setData(GArr3D<uchar4> data)
        {
            this->data = data;
            frame_num = data.batchs;
            width = data.rows;
            height = data.cols;
            frame_idx = 0;
            frame_idx_last = -1;
            update_frame_count = 0;
            play_speed = 0.5f;
        }
        void init()
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
        void update()
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
            ImVec2 img_size = ImVec2(wsize.x, wsize.y - ImGui::GetFrameHeightWithSpacing()*2);
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
            if (ImGui::IsItemActive() && update_frame_count % (int) (1.0f/play_speed) == 0)
            {
                frame_idx += 1;
                if (frame_idx >= frame_num)
                    frame_idx = 0;
            }
        }

        void clear()
        {
            cuSafeCall(cudaGraphicsUnmapResources(1, &CudaResource, 0));
        }
    };

}