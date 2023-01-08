#include "case_generator.h"
#include "ghost_cell.h"
#include "macro.h"
#include "objIO.h"
#include "visualize.h"
using namespace pppm;

void view_cell_data(GhostCellSolver *solver)
{
    GArr3D<float> view_data;
    view_data.resize(solver->cell_data.size);
    thrust::transform(thrust::device, solver->cell_data.begin(), solver->cell_data.end(), view_data.begin(),
                      view_transformer());

    RenderElement re(solver->grid, "distance");
    re.set_params(make_int3(0, 32, 0), 1, 1.0f);
    re.assign(0, view_data);
    re.update_mesh();
    re.write_image(0, EXP_DIR + std::string("test/cell_classification.png"));
}

int main()
{
    GhostCellSolver *solver = empty_ghost_cell_solver(64);
    solver->set_condition_number_threshold(15);
    auto filename = ASSET_DIR + std::string("sphere4.obj");
    auto mesh = Mesh::loadOBJ(filename, true);
    mesh.stretch_to(solver->size().x / 1.5f);
    mesh.move_to(solver->center());

    solver->set_mesh(mesh.vertices, mesh.triangles);

    view_cell_data(solver);
}