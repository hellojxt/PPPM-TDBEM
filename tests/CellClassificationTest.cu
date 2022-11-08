#include "case_generator.h"
#include "ghost_cell.h"
#include "objIO.h"
#include "visualize.h"

struct view_transformer
{
        CGPU_FUNC float operator()(CellInfo &x) const
        {
            float k = (x.type == GHOST) ? 1.0f : -1.0f;
            return 1.0f / (x.nearst_distance + 1e-10) * k;
        }
};

int main()
{
    using namespace pppm;

    GhostCellSolver *solver = empty_ghost_cell_solver(64);
    auto mesh = Mesh::loadOBJ("../assets/sphere.obj", true);
    mesh.stretch_to(solver->size().x / 3.0f);
    mesh.move_to(solver->center());

    solver->set_mesh(mesh.vertices, mesh.triangles);

    solver->precompute_cell_data();

    GArr3D<float> view_data;
    view_data.resize(solver->cell_data.size);
    thrust::transform(thrust::device, solver->cell_data.begin(), solver->cell_data.end(), view_data.begin(),
                      view_transformer());

    RenderElement re(solver->grid, "distance");
    re.set_params(make_int3(0, 0, 32), 1, 1 / solver->grid_size());
    re.assign(0, view_data);
    re.update_mesh();
    re.render();
}