#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdio>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "gui.h"
#include "macro.h"
#include "pppm.h"
#include "sound_source.h"
#include "window.h"

#define TEST_MAX_STEP 32

using namespace pppm;

__global__ void set_boundary_value(PPPMSolver pppm, SineSource sine)
{
    float neumann_amp = 1e2;
    float dirichlet_amp = 1e2;
    int t = pppm.fdtd.t + 1;
    float dt = pppm.fdtd.dt;
    pppm.particle_history[0].neumann[t] = neumann_amp * sine(dt * t).imag();
    pppm.particle_history[0].dirichlet[t] = dirichlet_amp * sine(dt * t).imag();
}

int main()
{
    int res = 33;
    float dl = 0.005;
    float dt = 1.0f / 120000;
    PPPMSolver solver(res, dl, dt);
    FDTD &fdtd = solver.fdtd;
    int3 coord = make_int3(16, 16, 16);
    float3 center = fdtd.getCenter(coord);
    CArr<float3> vertices(3);
    vertices[0] = center + make_float3(0.0f, 0.1f, 0.0f) * fdtd.dl;
    vertices[1] = center + make_float3(0.0f, 0.0f, 0.0f) * fdtd.dl;
    vertices[2] = center + make_float3(0.1f, 0.0f, 0.0f) * fdtd.dl;
    CArr<int3> triangles(1);
    triangles[0] = make_int3(0, 1, 2);
    solver.set_mesh(vertices, triangles);
    float omega = 2.0f * M_PI * 4000.0f;
    SineSource sine(omega);

    GArr3D<float> visual_data_far_field(TEST_MAX_STEP, res, res);

    while (fdtd.t < TEST_MAX_STEP - 1)
    {
        printf("t = %d\n", fdtd.t);
        cuExecuteBlock(1, 1, set_boundary_value, solver, sine);
        solver.solve_fdtd_simple();
        visual_data_far_field[fdtd.t].assign(solver.far_field[fdtd.t][16]);
    }

    // visualizer
    GUI gui;
    CudaRender render_far_field("far field visualizer");
    render_far_field.setData(visual_data_far_field, 0.005f);
    gui.append(&render_far_field);
    gui.start();
}
