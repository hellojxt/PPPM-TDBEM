#include "gui.h"
#include "objIO.h"
#include "pppm.h"
#include "window.h"
using namespace pppm;

int main()
{
    auto filename = std::string(ASSET_DIR) + std::string("/sphere.obj");
    auto mesh = loadOBJ(filename);
    mesh.normalize();
    GArr<float3> vertice = mesh.vertices;
    GArr<int3> triangles = mesh.triangles;
    GUI gui;
    CudaRender render;
    CudaRender render2("additional render");

    int res = 51;
    int step_num = 100;
    float dl = 0.005;
    float dt = 1.0f / 120000;

    GArr3D<float> data;
    data.resize(step_num, res, res);
    FDTD fdtd;
    fdtd.init(res, dl, dt);
    CArr3D<float> init_grid;
    init_grid.resize(res, res, res);
    init_grid.reset();
    init_grid(25, 25, 25) = 1;
    fdtd.grids[-1].assign(init_grid);

    for (int i = 0; i < step_num; i++)
    {
        // LOG_INFO("step" << i);
        fdtd.step();
        data[i].assign(fdtd.grids[i][25]);
    }

    render.setData(data, 0.01f);
    render2.setData(data, 0.01f);
    render.add_mesh_to_images(vertice, triangles, make_float3(-2, -2, -2), make_float3(2, 2, 2), PlaneType::XY,
                              make_float3(0.1, 0.1, 0.1));
    gui.append(&render);
    gui.append(&render2);
    gui.start();
}