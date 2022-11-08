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

#define ALL_STEP 512
#define SET_DIRICHLET false

using namespace pppm;

__global__ void set_boundary_value(PPPMSolver pppm, SineSource sine, MonoPole mp)
{
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= pppm.pg.particles.size())
        return;
    Particle &p = pppm.pg.particles[particle_id];
    int t = pppm.fdtd.t;
    float dt = pppm.fdtd.dt;
    if (SET_DIRICHLET)
        pppm.particle_history[particle_id].dirichlet[t] = (mp.dirichlet(p.pos) * sine(dt * t)).real();
    pppm.particle_history[particle_id].neumann[t] =
        (mp.neumann(p.pos, p.normal) * sine(dt * t)).real() * (t < STEP_NUM * 5);
}

int main()
{
    int res = 50;
    PPPMSolver *solver = empty_pppm(res);
    auto mesh = Mesh::loadOBJ("../assets/sphere.obj", true);
    mesh.stretch_to(solver->size().x / 4.0f);
    LOG("stretch to " << mesh.get_scale())
    mesh.move_to(solver->center());

    solver->set_mesh(mesh.vertices, mesh.triangles);
    RenderElement re(solver->pg, "PPPM");
    int x_idx = res / 8;
    int y_idx = res / 2;
    int z_idx = res / 2;

    re.set_params(make_int3(0, 0, z_idx), ALL_STEP, 1.0f);

    auto sine = SineSource(2 * PI * 3000);
    float wave_number = sine.omega / AIR_WAVE_SPEED;
    LOG("wave number: " << wave_number)
    auto mp = MonoPole(solver->center(), wave_number);

    solver->precompute_grid_cache();
    solver->precompute_particle_cache();

    for (int i = 0; i < ALL_STEP; i++)
    {
        solver->solve_fdtd_far_with_cache();
        cuExecute(solver->pg.particles.size(), set_boundary_value, *solver, sine, mp);
        if (!SET_DIRICHLET)
            solver->update_particle_dirichlet();
        solver->solve_fdtd_near_with_cache();
        re.assign(i, solver->fdtd.grids[i]);
        if ((i + 1) % STEP_NUM == 0 && i / STEP_NUM < 5)
        {
            re.update_mesh();
            mesh.move(make_float3(solver->size().x / 64.0f, 0, 0));
            solver->set_mesh(mesh.vertices, mesh.triangles);
            solver->precompute_grid_cache();
            solver->precompute_particle_cache();
        }
    }

    auto solver_signal = re.get_time_siganl(y_idx, x_idx).cpu();

    write_to_txt("pppm_dynamic_signal.txt", solver_signal);
    re.update_mesh();
    renderArray(re);
}
