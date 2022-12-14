#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdio>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
#include "macro.h"
#include "pppm.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"

#define TEST_MAX_STEP 128

using namespace pppm;

__global__ void set_boundary_value(PPPMSolver pppm, SineSource sine)
{
    float neumann_amp = 1e4;
    float dirichlet_amp = 1e4;
    int t = pppm.fdtd.t;
    float dt = pppm.fdtd.dt;
    pppm.neumann[0][t] = neumann_amp * sine(dt * t).real();
    pppm.dirichlet[0][t] = dirichlet_amp * sine(dt * t).real();
}
// FIXME: need to be fixed for new PPPM
int main()
{
    int res = 65;
    auto solver = empty_pppm(res);
    FDTD &fdtd = solver->fdtd;
    int3 coord = make_int3(res / 2, res / 2, res / 2);
    float3 center = solver->pg.getCenter(coord);
    CArr<float3> vertices(3);
    vertices[0] = center + make_float3(0.0f, 0.1f, 0.3f) * fdtd.dl;
    vertices[1] = center + make_float3(0.0f, 0.0f, 0.1f) * fdtd.dl;
    vertices[2] = center + make_float3(0.1f, 0.0f, 0.3f) * fdtd.dl;
    CArr<int3> triangles(1);
    triangles[0] = make_int3(0, 1, 2);
    solver->set_mesh(vertices, triangles);
    float omega = 2.0f * M_PI * 4000.0f;
    SineSource sine(omega);

    RenderElement re_fdtd(solver->pg, "FDTD");
    re_fdtd.set_params(make_int3(res / 2, 0, 0), TEST_MAX_STEP, 0.5f);
    RenderElement re_far(solver->pg, "Farfield");
    re_far.set_params(make_int3(res / 2, 0, 0), TEST_MAX_STEP, 0.5f);

    solver->precompute_grid_cache();
    for (int i = 0; i < TEST_MAX_STEP; i++)
    {
        solver->solve_fdtd_far_with_cache();
        cuExecuteBlock(1, 1, set_boundary_value, *solver, sine);
        solver->solve_fdtd_near_with_cache();
        re_fdtd.assign(i, fdtd.grids[fdtd.t]);
        re_far.assign(i, solver->far_field[fdtd.t]);
    }
    re_fdtd.update_mesh();
    re_far.update_mesh();
    printf("Done\n");
    // visualizer
    renderArray(re_fdtd, re_far);
}
