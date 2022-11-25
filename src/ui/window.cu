#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>  // CUDA
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <string>
#include "array3D.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "helper_math.h"
#include "imgui_internal.h"
#include "macro.h"
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
    float3 c = {1.0, 1.0, 1.0};  // white
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

__global__ void preprocess_image_data(GArr3D<float> in_data,
                                      GArr3D<uchar4> out_data,
                                      float data_max,
                                      int upsample_factor,
                                      float line_width,
                                      int start_batch,
                                      int end_batch)
{
    for (int x = blockIdx.x; x < out_data.rows; x += gridDim.x)
    {
        for (int y = threadIdx.x; y < out_data.cols; y += blockDim.x)
        {
            for (int b = start_batch; b < end_batch; b++)
            {
                float2 pos = make_float2(x + 0.5f, y + 0.5f);
                int x_data_idx = (x / upsample_factor);
                int y_data_idx = (y / upsample_factor);
                float2 pos_center = make_float2(x_data_idx * upsample_factor + upsample_factor / 2.0f,
                                                y_data_idx * upsample_factor + upsample_factor / 2.0f);
                float value = in_data(b, x / upsample_factor, y / upsample_factor);
                uchar4 line_color = make_uchar4(255, 255, 255, 255);
                uchar4 pixel_color = JetColor(value, -data_max, data_max);
                float k = 3.5f;
                pixel_color.x /= k;
                pixel_color.y /= k;
                pixel_color.z /= k;

                float dist = upsample_factor / 2.0f - max(abs(pos.x - pos_center.x), abs(pos.y - pos_center.y));
                float gauss;
                if (dist <= line_width)
                {
                    gauss = 0.4f;
                }
                else
                {
                    gauss = 0.0f;
                }
                // float gauss = exp(-dist * dist / (line_width * line_width));
                out_data(b, x, y) = make_uchar4(pixel_color.x * (1 - gauss) + line_color.x * gauss,
                                                pixel_color.y * (1 - gauss) + line_color.y * gauss,
                                                pixel_color.z * (1 - gauss) + line_color.z * gauss, 255);
            }
        }
    }
}

void CudaRender::setData(GArr3D<float> origin_data, float data_max, float line_width, int start_batch, int end_batch)
{
    int upsample_factor = 1080 / origin_data.rows;
    if (start_batch == 0)
    {
        data.resize(origin_data.batchs, origin_data.rows * upsample_factor, origin_data.cols * upsample_factor);
        width = data.rows;
        height = data.cols;
        frame_idx = 0;
        frame_idx_last = -1;
        update_frame_count = 0;
        play_speed = 0.5f;
    }
    cuExecuteBlock(data.rows, 64, preprocess_image_data, origin_data, data, data_max, upsample_factor, line_width,
                   start_batch, end_batch);
}

// calculate interaction between a line segment and a plane
GPU_FUNC bool linePlaneIntersection(float3 &contact, float3 start, float3 end, float3 plane_normal, float3 plane_pos)
{
    float3 dir = end - start;
    float3 normal = normalize(plane_normal);
    float3 plane_to_start = start - plane_pos;
    float l = length(dir);
    float denom = dot(normal, dir);
    if (abs(denom) < 1e-6)
    {
        return false;
    }
    float numer = dot(normal, plane_to_start);
    float t = -(numer / denom);
    float eps = 1e-4 * l;
    if (t <= 0 - eps || t >= 1.0f + eps)
    {
        return false;
    }
    float3 contact_pos = start + t * dir;
    float3 delta = contact - contact_pos;
    if (length(contact - (start + t * dir)) < eps)
    {
        return false;
    }
    else
    {
        contact = start + t * dir;
        return true;
    }
}

GPU_FUNC float2 get_pixel_coord(float3 pos, float3 min_pos, float3 max_pos, int width, int height, PlaneType plane)
{
    float3 pos_scaled = (pos - min_pos) / (max_pos - min_pos);
    float2 coord;
    switch (plane)
    {
        case PlaneType::XY:
            coord = make_float2(pos_scaled.x * width, pos_scaled.y * height);
            break;
        case PlaneType::XZ:
            coord = make_float2(pos_scaled.x * width, pos_scaled.z * height);
            break;
        case PlaneType::YZ:
            coord = make_float2(pos_scaled.y * width, pos_scaled.z * height);
            break;
    }
    return coord;
}

GPU_FUNC float point_line_distance(float2 point, float2 start, float2 end)
{
    if (length(end - start) < 1e-6)
    {
        return length(point - start);
    }
    float2 dir = end - start;
    float2 diff = point - start;
    float t = dot(diff, dir) / dot(dir, dir);
    if (t < 0)
    {
        t = 0;
    }
    else if (t > 1)
    {
        t = 1;
    }
    float2 closest = start + t * dir;
    float dist = length(point - closest);
    return dist;
}

__global__ void add_mesh_kernel(GArr<float3> vertices,
                                GArr<int3> triangles,
                                float3 min_pos,
                                float3 max_pos,
                                GArr3D<uchar4> data,
                                GArr2D<int> gauss,
                                GArr2D<int> gauss_num,
                                PlaneType plane,
                                float3 plane_pos,
                                float line_width)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= triangles.size())
    {
        return;
    }

    int3 tri = triangles[idx];
    float3 v0 = vertices[tri.x];
    float3 v1 = vertices[tri.y];
    float3 v2 = vertices[tri.z];
    float3 plane_normal;
    if (plane == PlaneType::XY)
        plane_normal = make_float3(0, 0, 1);
    else if (plane == PlaneType::XZ)
        plane_normal = make_float3(0, 1, 0);
    else if (plane == PlaneType::YZ)
        plane_normal = make_float3(1, 0, 0);
    float3 contact[5];
    for (int i = 0; i < 5; i++)
    {
        contact[i] = make_float3(0, 0, 0);
    }
    int contact_num = 0;
    if (linePlaneIntersection(contact[contact_num], v0, v1, plane_normal, plane_pos))
        contact_num++;
    if (linePlaneIntersection(contact[contact_num], v1, v2, plane_normal, plane_pos))
        contact_num++;
    if (linePlaneIntersection(contact[contact_num], v2, v0, plane_normal, plane_pos))
        contact_num++;
    if (contact_num >= 2)
    {
        if (length(contact[0] - contact[1]) < 1e-4 && contact_num > 2)
        {
            contact[1] = contact[2];
        }
        float2 coord1 = get_pixel_coord(contact[0], min_pos, max_pos, data.rows, data.cols, plane);
        float2 coord2 = get_pixel_coord(contact[1], min_pos, max_pos, data.rows, data.cols, plane);
        float2 coord_min = fminf(coord1, coord2);
        float2 coord_max = fmaxf(coord1, coord2);
        float extra_width = line_width * 4;
        int x_min = max(0, (int)(coord_min.x - extra_width));
        int x_max = min(data.rows - 1, (int)(coord_max.x + extra_width));
        int y_min = max(0, (int)(coord_min.y - extra_width));
        int y_max = min(data.cols - 1, (int)(coord_max.y + extra_width));
        for (int i = x_min; i <= x_max; i++)
        {
            for (int j = y_min; j <= y_max; j++)
            {
                float2 pixel_center = make_float2(i + 0.5f, j + 0.5f);
                float dist = point_line_distance(pixel_center, coord1, coord2);
                // anti-aliasing
                float alpha = 1.0f;
                if (dist > line_width)
                {
                    alpha = exp(-0.5 * pow((dist - line_width) * 3, 2));
                }
                // printf("dist: %f, alpha: %f\n", dist, alpha);
                atomicMax(&gauss(i, j), alpha * 255);
                atomicAdd(&gauss_num(i, j), 1);
            }
        }
    }
}

void __global__
add_mesh_post_kernel(GArr3D<uchar4> data, GArr2D<int> gauss, GArr2D<int> gauss_num, int start_batch, int end_batch)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= data.rows || j >= data.cols)
    {
        return;
    }
    uchar4 line_color = make_uchar4(255, 255, 255, 255);
    if (gauss_num(i, j) > 0)
    {
        float alpha = gauss(i, j) / 255.0f;
        for (int b = start_batch; b < end_batch; b++)
            data(b, i, j) = make_uchar4((u_char)(data(b, i, j).x * (1.0f - alpha) + line_color.x * alpha),
                                        (u_char)(data(b, i, j).y * (1.0f - alpha) + line_color.y * alpha),
                                        (u_char)(data(b, i, j).z * (1.0f - alpha) + line_color.z * alpha),
                                        (u_char)(data(b, i, j).w * (1.0f - alpha) + line_color.w * alpha));
    }
}

void CudaRender::add_mesh_to_images(GArr<float3> vertices,
                                    GArr<int3> triangles,
                                    float3 min_pos,
                                    float3 max_pos,
                                    PlaneType plane,
                                    float3 plane_pos,
                                    float line_width,
                                    int start_batch,
                                    int end_batch)
{
    GArr2D<int> gauss;
    GArr2D<int> gauss_num;
    gauss.resize(data.rows, data.cols);
    gauss_num.resize(data.rows, data.cols);
    gauss.reset();
    gauss_num.reset();
    cuExecute(triangles.size(), add_mesh_kernel, vertices, triangles, min_pos, max_pos, data, gauss, gauss_num, plane,
              plane_pos, line_width);
    cuExecute2D(dim2(data.rows, data.cols), add_mesh_post_kernel, data, gauss, gauss_num, start_batch, end_batch);
    gauss.clear();
    gauss_num.clear();
}

void CudaRender::init()
{
    is_inited = true;
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
        cuSafeCall(cudaMemcpy2DToArray(array, 0, 0, frame_data.begin(), width * sizeof(uchar4), height * sizeof(uchar4),
                                       height, cudaMemcpyDeviceToDevice));
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
    if (is_inited)
        cuSafeCall(cudaGraphicsUnmapResources(1, &CudaResource, 0));
    data.clear();
}

}  // namespace pppm