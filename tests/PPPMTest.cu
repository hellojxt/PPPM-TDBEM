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
    pppm.particle_history[particle_id].neumann[t] = (mp.neumann(p.pos, p.normal) * sine(dt * t)).real();
}

int main()
{
    int res = 50;
    PPPMSolver *solver = empty_pppm(res);
    auto mesh = Mesh::loadOBJ("../assets/bunny.obj", true);
    mesh.stretch_to(solver->size().x / 4.0f);
    LOG("stretch to " << mesh.get_scale())
    mesh.move_to(solver->center());

    solver->set_mesh(mesh.vertices, mesh.triangles);
    RenderElement re(solver->pg, "PPPM");
    int x_idx = res / 6;
    int y_idx = res / 2;
    int z_idx = res / 2;

    re.set_params(make_int3(0, 0, z_idx), ALL_STEP, 1.0f);

    auto sine = SineSource(2 * PI * 3000);
    float wave_number = sine.omega / AIR_WAVE_SPEED;
    LOG("wave number: " << wave_number)
    auto mp = MonoPole(solver->center(), wave_number);

    solver->precompute_grid_cache();
    solver->precompute_particle_cache();
    TICK(solve_with_cache)
    for (int i = 0; i < ALL_STEP; i++)
    {
        solver->solve_fdtd_far_with_cache();
        cuExecute(solver->pg.particles.size(), set_boundary_value, *solver, sine, mp);
        if (!SET_DIRICHLET)
            solver->update_particle_dirichlet();
        solver->solve_fdtd_near_with_cache();
        re.assign(i, solver->fdtd.grids[i]);
    }
    TOCK(solve_with_cache)

    TDBEM &bem = solver->bem;
    auto vertices = mesh.vertices;
    auto paticles = solver->pg.particles.cpu();
    float3 trg_pos = solver->pg.getCenter(x_idx, y_idx, z_idx);
    cpx bem_sum = 0;
    for (int p_id = 0; p_id < paticles.size(); p_id++)
    {
        auto &p = paticles[p_id];
        auto pair_info = PairInfo(p.indices, trg_pos);
        bem_sum +=
            bem.helmholtz(vertices.data(), pair_info, mp.neumann(p.pos, p.normal), mp.dirichlet(p.pos), wave_number);
    }

    auto solver_signal = re.get_time_siganl(y_idx, x_idx).cpu();
    CArr<float> helmholtz_result(ALL_STEP);
    for (int i = 0; i < ALL_STEP; i++)
        helmholtz_result[i] = (bem_sum * sine(solver->fdtd.dt * i)).real();
    CArr<float> analytic_result(ALL_STEP);
    for (int i = 0; i < ALL_STEP; i++)
        analytic_result[i] = (mp.dirichlet(trg_pos) * sine(solver->fdtd.dt * i)).real();

    LOG("bem sum: " << bem_sum)
    LOG("analytic weight: " << mp.dirichlet(trg_pos))
    write_to_txt("pppm_signal.txt", solver_signal);
    write_to_txt("helmholtz_signal.txt", helmholtz_result);
    write_to_txt("analytic_signal.txt", analytic_result);
    re.update_mesh();
    renderArray(re);
}
