#include "gui.h"
#include "window.h"
#include "fdtd.h"
using namespace pppm;

int main(){
    GUI gui;
    CudaRender render;
    GArr3D<float> data;

    int res = 50;
    int step_num = 100;
    float dl = 0.005;
    float dt = 1.0f/120000;

    data.resize(step_num, res, res);
    FDTD fdtd;
    fdtd.init(res, dl, dt);
    CArr3D<cpx> init_grid;
    init_grid.resize(res, res, res);
    init_grid.reset();
    init_grid(25, 25, 25) = cpx(1, 0);
    fdtd.grids[0].assign(init_grid);

    for (int i = 0; i < step_num; i++){
        auto cpu_data = fdtd.grids[0].cpu();
        LOG(i << ": " << cpu_data(25, 25, 25));
        fdtd.copy_clip(data);
        fdtd.update();
    }

    render.setData(data, 0.01f);
    gui.append(&render);
    gui.start();
    return 0;
}