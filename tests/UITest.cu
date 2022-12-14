#include "case_generator.h"
#include "gui.h"
#include "objIO.h"
#include "pppm.h"
#include "visualize.h"
#include "window.h"

using namespace pppm;
// FIXME: need to be fixed for new PPPM
int main()
{
    auto filename = std::string(ASSET_DIR) + std::string("/sphere.obj");
    auto mesh = Mesh::loadOBJ(filename);
    int res = 33;
    auto solver = empty_pppm(res);
    mesh.stretch_to(solver->size().x / 4);
    mesh.move_to(solver->center());
    solver->set_mesh(mesh.vertices, mesh.triangles);
    int step_num = 100;

    RenderElement re(solver->pg, "UItest");
    re.set_params(make_int3(0, 0, res / 2), step_num, 0.01f);

    CArr3D<float> init_grid = solver->pg.fdtd.grids[0].cpu();
    init_grid.reset();
    init_grid(res / 2, res / 2, res / 2) = 1;
    solver->pg.fdtd.grids[-1].assign(init_grid);

    for (int i = 0; i < step_num; i++)
    {
        solver->pg.fdtd.step();
        re.assign(i, solver->pg.fdtd.grids[i]);
    }
    re.update_mesh();
    renderArray(re);
}