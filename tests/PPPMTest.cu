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

__global__ void set_boundary_value(PPPMSolver pppm, SineSource sine, MonoPole mp, GArr<float> surface_accs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pppm.pg.triangles.size())
        return;
    auto &p = pppm.pg.triangles[i];
    int t = pppm.time_idx();
    float dt = pppm.dt();
    if (SET_DIRICHLET)
        pppm.dirichlet[i][t] = (mp.dirichlet(p.center) * sine(dt * t)).real();
    if (t == 0)
        pppm.neumann[i][t] = surface_accs[i];
    else
        pppm.neumann[i][t] = 0;
}

int main()
{
    int res = 32;
    PPPMSolver *solver = empty_pppm(res);
    auto filename =
        "/home/jxt/PPPM-TDBEM/experiments/rigidbody/output/lego/surface.obj";  // ASSET_DIR +
                                                                               // std::string("sphere3.obj");
    auto surface_accs_cpu = read_from_txt("/home/jxt/PPPM-TDBEM/experiments/rigidbody/output/lego/surface_accs.txt");
    GArr<float> surface_accs(surface_accs_cpu);
    auto mesh = Mesh::loadOBJ(filename, true);
    mesh.stretch_to(solver->size().x / 2.0f);
    LOG("stretch to " << mesh.get_scale())
    mesh.move_to(solver->center());

    solver->set_mesh(mesh.vertices, mesh.triangles);
    RenderElement re(solver->pg, "PPPM");
    int x_idx = res / 6;
    int y_idx = res / 2;
    int z_idx = res / 2;

    re.set_params(make_int3(0, 0, z_idx), ALL_STEP, 0.01f);

    auto sine = SineSource(2 * PI * 3000);
    float wave_number = sine.omega / AIR_WAVE_SPEED;
    LOG("wave number: " << wave_number)
    auto mp = MonoPole(solver->center(), wave_number);

    TICK(solve_with_cache)
    for (int i = 0; i < ALL_STEP; i++)
    {
        solver->pg.fdtd.step();
        solver->solve_fdtd_far();
        cuExecute(solver->pg.triangles.size(), set_boundary_value, *solver, sine, mp, surface_accs);
        if (!SET_DIRICHLET)
            solver->update_dirichlet();
        solver->solve_fdtd_near();
        re.assign(i, solver->pg.fdtd.grids[i]);
    }
    TOCK(solve_with_cache)

    TDBEM &bem = solver->bem;
    auto vertices = mesh.vertices;
    auto paticles = solver->pg.triangles.cpu();
    float3 trg_pos = solver->pg.getCenter(x_idx, y_idx, z_idx);
    cpx bem_sum = 0;
    for (int p_id = 0; p_id < paticles.size(); p_id++)
    {
        auto &p = paticles[p_id];
        auto pair_info = PairInfo(p.indices, trg_pos);
        bem_sum += bem.helmholtz(vertices.data(), pair_info, mp.neumann(p.center, p.normal), mp.dirichlet(p.center),
                                 wave_number);
    }

    auto solver_signal = re.get_time_siganl(y_idx, x_idx).cpu();
    CArr<float> helmholtz_result(ALL_STEP);
    for (int i = 0; i < ALL_STEP; i++)
        helmholtz_result[i] = (bem_sum * sine(solver->dt() * i)).real();
    CArr<float> analytic_result(ALL_STEP);
    for (int i = 0; i < ALL_STEP; i++)
        analytic_result[i] = (mp.dirichlet(trg_pos) * sine(solver->dt() * i)).real();

    LOG("bem sum: " << bem_sum)
    LOG("analytic weight: " << mp.dirichlet(trg_pos))
    write_to_txt("pppm_signal.txt", solver_signal);
    write_to_txt("helmholtz_signal.txt", helmholtz_result);
    write_to_txt("analytic_signal.txt", analytic_result);
    re.update_mesh();
    // re.write_image(ALL_STEP / 2, "pppm.png");
    renderArray(re);
}
