#pragma once
#include "gui.h"
#include "objIO.h"
#include "pppm.h"
#include "window.h"
#include "array_writer.h"
namespace pppm
{

static inline PlaneType normal2plane(float3 normal)
{
    if (normal.x != 0)
        return PlaneType::YZ;
    else if (normal.y != 0)
        return PlaneType::XZ;
    else if (normal.z != 0)
        return PlaneType::XY;
    else
    {
        std::cout << "Error: normal2plane: normal is a zero vector" << std::endl;
        exit(1);
    }
}

static inline float3 plane2normal(PlaneType plane)
{
    if (plane == PlaneType::YZ)
        return make_float3(1, 0, 0);
    else if (plane == PlaneType::XZ)
        return make_float3(0, 1, 0);
    else if (plane == PlaneType::XY)
        return make_float3(0, 0, 1);
    else
    {
        std::cout << "Error: plane2normal: plane is not a valid plane" << std::endl;
        exit(1);
    }
}

class RenderElement
{
    public:
        GArr3D<float> data;
        int3 plane;
        int frame_num;
        ParticleGrid &grid;
        float max_abs_value;
        std::string name;
        CudaRender render_window;
        int last_mesh_update;
        int current_data_idx;

        RenderElement(ParticleGrid &grid_, std::string name_) : grid(grid_), name(name_)
        {
            render_window = CudaRender(name.c_str());
            last_mesh_update = -1;
            current_data_idx = -1;
        }

        void set_params(int3 plane_, int frame_num_, float max_value)
        {
            plane = plane_;
            frame_num = frame_num_;
            max_abs_value = max_value;
            data.resize(frame_num, grid.grid_dim.x, grid.grid_dim.y);
            render_window.set_overall_frame_num(frame_num);
        }

        void update_mesh()
        {
            render_window.setData(data, max_abs_value, 0.5f, last_mesh_update + 1, current_data_idx + 1);
            auto p = normal2plane(make_float3(plane));
            auto normal = plane2normal(p);
            auto plane_pos = (grid.max_pos + grid.min_pos) / 2;
            plane_pos += (length(make_float3(plane)) - grid.grid_dim.x / 2.0f + 0.5f) * normal * grid.grid_size;
            render_window.add_mesh_to_images(grid.vertices, grid.triangles, grid.min_pos, grid.max_pos, p, plane_pos,
                                             1.0f, last_mesh_update + 1, current_data_idx + 1);
            last_mesh_update = current_data_idx;
        }

        void last_check()
        {
            if (last_mesh_update != current_data_idx)
            {
                render_window.setData(data, max_abs_value, 0.5f, last_mesh_update + 1, current_data_idx + 1);
            }
        }

        void assign(int idx, GArr3D<float> src);

        GArr<float> get_time_siganl(int x, int y);

        void render()
        {
            GUI gui;
            last_check();
            gui.append(&render_window);
            gui.start();
        }

        void write_image(int idx, std::string filename);
};

static inline void renderArray(RenderElement &e)
{
    GUI gui;
    e.last_check();
    gui.append(&e.render_window);
    gui.start();
}

static inline void renderArray(RenderElement &e1, RenderElement &e2)
{
    GUI gui;
    e1.last_check();
    e2.last_check();
    gui.append(&e1.render_window);
    gui.append(&e2.render_window);
    gui.start();
}

}  // namespace pppm