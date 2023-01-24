#include <string>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
#include "macro.h"
#include "objIO.h"
#include "particle_grid.h"
#include "pppm.h"
#include "progressbar.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"
#include <filesystem>
#include <fstream>
#include "ghost_cell.h"
#include <sys/stat.h>

using namespace pppm;

#define ALL_STEP 3500

__global__ void update_surf_acc(SineSource sine, GArr<float> surface_accs, GArr<Triangle> triangles, float t)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= surface_accs.size())
        return;
    auto tri = triangles[idx];
    if (tri.normal.y > 0)
    {
        // sine.omega = 2 * PI * 4000 * abs(tri.normal.x);
        surface_accs[idx] = sine(t).real();
    }
    else
    {
        // sine.omega = 2 * PI * 4000 * abs(tri.normal.x);
        surface_accs[idx] = -sine(t).real();
    }
}

template <typename T>
void simulation(T &solver, Mesh &mesh, SineSource &sine, std::string dirname)
{
    GArr<float> surface_accs;
    surface_accs.resize(mesh.triangles.size());
    CHECK_DIR(dirname);
    LOG(dirname);
    float time_cost = 0;
    APPEND_TIME(time_cost, solver.set_mesh(mesh.vertices, mesh.triangles), set_mesh)
    progressbar bar(ALL_STEP);
    auto triangles = solver.get_triangles();
    for (int i = 0; i < ALL_STEP; i++)
    {
        bar.update();
        cuExecute(surface_accs.size(), update_surf_acc, sine, surface_accs, triangles, solver.dt() * i);
        APPEND_TIME(time_cost, solver.update_step(surface_accs), update)
        // save_grid(solver.get_particle_grid(), dirname + "/" + std::to_string(i) + ".png", 0.01f,
        //           make_float3(0, 0, 0.5));
    }
    LOG("time cost: " << time_cost);
}

int main(int argc, char *argv[])
{
    auto dir_name = ROOT_DIR + std::string("dataset/teaser/");
    auto OUT_DIR = dir_name + "output/";
    CHECK_DIR(OUT_DIR);

    float scale = 2;
    float box_size = 0.7;
    float grid_size = 0.017;

    auto mesh = Mesh::loadOBJ(dir_name + "mesh.obj");
    mesh.stretch_to(box_size / scale);
    mesh.fix_mesh(grid_size, OUT_DIR);
    float3 min_pos = mesh.get_center() - box_size / 2;
    float dt = grid_size / (std::sqrt(3) * AIR_WAVE_SPEED * 1.01);
    int res = box_size / grid_size;
    LOG("res: " << res << ", dt: " << dt << ", grid_size: " << grid_size << ", box_size: " << box_size)

    auto sine = SineSource(2 * PI * 4000);

    // PPPM
    PPPMSolver pppm(res, grid_size, dt, min_pos);
    simulation(pppm, mesh, sine, OUT_DIR + "/pppm/");
    pppm.clear();

    // First order Ghost cell
    // GhostCellSolver solver1(min_pos, grid_size, res, dt);
    // solver1.set_condition_number_threshold(0.0f);
    // simulation(solver1, mesh, sine, OUT_DIR + "/ghostcell1/");
    // solver1.clear();

    // Second order Ghost cell
    GhostCellSolver solver2(min_pos, grid_size, res, dt);
    solver2.set_condition_number_threshold(25.0f);
    simulation(solver2, mesh, sine, OUT_DIR + "/ghostcell2/");
    solver2.clear();
}
