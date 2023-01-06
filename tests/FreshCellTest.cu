#include "case_generator.h"
#include "ghost_cell.h"
#include "macro.h"
#include "objIO.h"
#include "visualize.h"

using namespace pppm;

__global__ void set_grid_potential_kernel(GhostCellSolver solver)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= solver.grid.grid_dim || y >= solver.grid.grid_dim || z >= solver.grid.grid_dim)
        return;
    int3 coord = make_int3(x, y, z);
    float p;
    if (solver.cell_data(coord).type != AIR)
        p = 0;
    else
        p = length(solver.grid.getCenter(coord) - solver.center());
    solver.grid.fdtd.grids[-1](coord) = p;
    solver.grid.fdtd.grids[-2](coord) = p;
}

int main(int argc, char *argv[])
{
    // get condition number threshold from command line
    float condition_number_threshold = 15;
    GhostCellSolver *solver = empty_ghost_cell_solver(64);
    solver->set_condition_number_threshold(condition_number_threshold);
    auto filename = ASSET_DIR + std::string("sphere4.obj");
    auto mesh = Mesh::loadOBJ(filename, true);
    mesh.stretch_to(solver->size().x / 1.5f);
    mesh.move_to(solver->center());
    solver->set_mesh(mesh.vertices, mesh.triangles);
    RenderElement re1(solver->grid, "ghost cell");
    re1.set_params(make_int3(0, 0, 32), 2, solver->grid_size() * 32);
    cuExecute3D(dim3(solver->res(), solver->res(), solver->res()), set_grid_potential_kernel, *solver);
    re1.assign(0, solver->grid.fdtd.grids[-2]);
    re1.assign(1, solver->grid.fdtd.grids[-1]);
    re1.update_mesh();
    re1.write_image(0, EXP_DIR + std::string("test/ghost_cell_before_0.png"));
    re1.write_image(1, EXP_DIR + std::string("test/ghost_cell_before_1.png"));
    re1.clear();

    mesh.move(make_float3(solver->grid.grid_size / 2));
    solver->update_mesh(mesh.vertices, true);
    RenderElement re2(solver->grid, "ghost cell");
    re2.set_params(make_int3(0, 32, 0), 2, solver->grid_size() * 32);
    re2.assign(0, solver->grid.fdtd.grids[-2]);
    re2.assign(1, solver->grid.fdtd.grids[-1]);
    re2.update_mesh();
    re2.write_image(0, EXP_DIR + std::string("test/ghost_cell_after_0.png"));
    re2.write_image(1, EXP_DIR + std::string("test/ghost_cell_after_1.png"));
    re2.clear();
}