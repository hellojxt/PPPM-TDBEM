#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
#include "macro.h"
#include "objIO.h"
#include "pppm.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"

#define ALL_STEP 256
using namespace pppm;

__global__ void set_boundary_value(PPPMSolver pppm, SineSource sine, float3 center)
{
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= pppm.pg.particles.size())
        return;
    float neumann_amp = 1e3;
    int t = pppm.fdtd.t;
    float dt = pppm.fdtd.dt;
    pppm.particle_history[particle_id].neumann[t] = neumann_amp * sine(dt * t).real();
}

int main()
{
    int res = 64;
    PPPMSolver *solver = empty_pppm(res);
    auto mesh = Mesh::loadOBJ("../assets/sphere4.obj");
    LOG(mesh.vertices.size())
    LOG(mesh.triangles.size())
    mesh.stretch_to(solver->size().x / 3);
    mesh.move_to(solver->center());
    solver->set_mesh(mesh.vertices, mesh.triangles);
    RenderElement re(solver->pg, "PPPM");
    re.set_params(make_int3(0, 0, res / 2), ALL_STEP, 2.0f);

    solver->precompute_grid_cache();
    solver->precompute_particle_cache();

    TICK(solve_with_cache)
    for (int i = 0; i < ALL_STEP; i++)
    {
        solver->solve_fdtd_far_with_cache();
        cuExecute(solver->pg.particles.size(), set_boundary_value, *solver, SineSource(2 * PI * 3000),
                  solver->center());
        solver->update_particle_dirichlet();
        solver->solve_fdtd_near_with_cache();
        re.assign(i, solver->fdtd.grids[i]);
    }
    TOCK(solve_with_cache)
    renderArray(re);
}
