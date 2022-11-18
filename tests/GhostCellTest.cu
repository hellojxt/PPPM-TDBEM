#include "case_generator.h"
#include "ghost_cell.h"
#include "macro.h"
#include "objIO.h"
#include "visualize.h"
using namespace pppm;
struct view_transformer
{
        CGPU_FUNC float operator()(CellInfo &x) const
        {
            if (x.type == SOLID)
                return 0.0f;  // green for solid
            if (x.type == GHOST)
                return 1.0f / (x.nearst_distance + 0.005);  // light red for ghost
            if (x.type == AIR)
                return -1.0f / (x.nearst_distance + 0.005);  // light blue for air
            return -MAX_FLOAT;                               // blue for UNKNOWN
        }
};

void view_cell_data(GhostCellSolver *solver)
{
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
#define MAX_STEP 128
#define USE_UI
int main(int argc, char *argv[])
{
    // get condition number threshold from command line
    float condition_number_threshold = argv[1] ? atof(argv[1]) : 25.0f;
    GhostCellSolver *solver = empty_ghost_cell_solver(64);
    solver->set_condition_number_threshold(condition_number_threshold);
    auto mesh = Mesh::loadOBJ("../assets/ghost_cell_test/2.obj", true);
    mesh.stretch_to(solver->size().x / 2.0f);
    mesh.move_to(solver->center());
    solver->set_mesh(mesh.vertices, mesh.triangles);

    CArr<float> neumann_data;
    neumann_data.resize(solver->grid.particles.size());

#ifdef USE_UI
    RenderElement re(solver->grid, "ghost cell");
    re.set_params(make_int3(0, 32, 0), MAX_STEP, solver->grid_size());
#endif
    float omega = 2 * PI * 8000;
    solver->precompute_cell_data();
    solver->precompute_ghost_matrix();

    for (int i = 0; i < MAX_STEP; i++)
    {
        printf("step %d\n", i);
        float t = i * solver->dt();
        solver->fdtd.step();
        thrust::fill(thrust::host, neumann_data.begin(), neumann_data.end(), sin(t * omega));
        solver->set_boundary_condition(neumann_data);
        solver->solve_ghost_cell();
#ifdef USE_UI
        re.assign(i, solver->fdtd.grids[i]);
#endif
        // solver->A.print();
        // LOG_INFO(solver->p_weight)
        // LOG_INFO(solver->x)
        // break;
    }
    auto solver_order = solver->ghost_order.cpu();
    float order_sum = thrust::reduce(thrust::host, solver_order.begin(), solver_order.end(), 0.0f);
    LOG("condition number threshold: " << condition_number_threshold);
    LOG("1st order replace rate: " << 1.0f - ((float)order_sum) / solver_order.size())
#ifdef USE_UI
    re.update_mesh();
    re.render();
#endif
}