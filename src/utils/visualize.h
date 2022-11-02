#pragma once
#include "gui.h"
#include "objIO.h"
#include "pppm.h"
#include "window.h"

namespace pppm
{

class RenderElement
{
    public:
        GArr3D<float> data;
        int3 plane;
        int frame_num;
        ParticleGrid &grid;
        float max_abs_value;
        std::string name;
        RenderElement(ParticleGrid &grid_, std::string name_) : grid(grid_), name(name_) {}

        void set_params(int3 plane_, int frame_num_, float max_value)
        {
            plane = plane_;
            frame_num = frame_num_;
            max_abs_value = max_value;
            data.resize(frame_num, grid.grid_dim.x, grid.grid_dim.y);
        }

        void assign(int idx, GArr3D<float> src);
};

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

static inline void renderArray(RenderElement e, bool draw_mesh = true)
{
    GUI gui;
    CudaRender render(e.name.c_str());
    render.setData(e.data, e.max_abs_value);
    if (draw_mesh)
    {
        auto p = normal2plane(make_float3(e.plane));
        auto normal = plane2normal(p);
        auto plane_pos = (e.grid.max_pos + e.grid.min_pos) / 2;
        plane_pos += (length(make_float3(e.plane)) - e.grid.grid_dim.x / 2.0f + 0.5f) * normal * e.grid.grid_size;
        render.add_mesh_to_images(e.grid.vertices, e.grid.triangles, e.grid.min_pos, e.grid.max_pos, p, plane_pos,
                                  1.0f);
    }
    gui.append(&render);
    gui.start();
}

static inline void renderArray(std::vector<RenderElement> es, bool draw_mesh = true)
{
    GUI gui;
    for (auto e : es)
    {
        CudaRender *render = new CudaRender(e.name.c_str());
        render->setData(e.data, e.max_abs_value);
        if (draw_mesh)
        {
            auto p = normal2plane(make_float3(e.plane));
            auto normal = plane2normal(p);
            auto plane_pos = (e.grid.max_pos + e.grid.min_pos) / 2;
            plane_pos += (length(make_float3(e.plane)) - e.grid.grid_dim.x / 2.0f + 0.5f) * normal * e.grid.grid_size;
            render->add_mesh_to_images(e.grid.vertices, e.grid.triangles, e.grid.min_pos, e.grid.max_pos, p, plane_pos,
                                       1.0f);
        }
        gui.append(render);
    }
    gui.start();
}

}  // namespace pppm