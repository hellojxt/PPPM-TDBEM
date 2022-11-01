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

#define TEST_MAX_STEP 64

using namespace pppm;

__global__ void set_boundary_value(PPPMSolver pppm, SineSource sine)
{
    float neumann_amp = 1e4;
    float dirichlet_amp = 1e4;
    int t = pppm.fdtd.t;
    float dt = pppm.fdtd.dt;
    pppm.particle_history[0].neumann[t] = neumann_amp * sine(dt * t).real();
    pppm.particle_history[0].dirichlet[t] = dirichlet_amp * sine(dt * t).real();
}

__global__ void copy_kernel(GArr3D<float> src, GArr3D<float> dst, int t, int3 face)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst.size.y || y >= dst.size.z)
        return;
    int3 e[3] = {make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1)};
    int idx = (face.y != 0) + (face.z != 0) * 2;
    int3 pos = face + e[(idx + 1) % 3] * x + e[(idx + 2) % 3] * y;
    dst(t, x, y) = src(pos);
}

int main()
{
    int res = 33;
    float dl = 0.005;
    float dt = 1.0f / 120000;
    PPPMSolver solver(res, dl, dt);
    FDTD &fdtd = solver.fdtd;
    int3 coord = make_int3(res / 2, res / 2, res / 2);
    float3 center = fdtd.getCenter(coord);
    CArr<float3> vertices(3);
    vertices[0] = center + make_float3(0.0f, 0.1f, 0.3f) * fdtd.dl;
    vertices[1] = center + make_float3(0.0f, 0.0f, 0.1f) * fdtd.dl;
    vertices[2] = center + make_float3(0.1f, 0.0f, 0.3f) * fdtd.dl;
    CArr<int3> triangles(1);
    triangles[0] = make_int3(0, 1, 2);
    solver.set_mesh(vertices, triangles);
    float omega = 2.0f * M_PI * 4000.0f;
    SineSource sine(omega);
    TICK(all_time)
    GArr3D<float> visual_data_far_field(TEST_MAX_STEP, res, res);
    GArr3D<float> visual_data_fdtd(TEST_MAX_STEP, res, res);
    solver.precompute_grid_cache();
    while (fdtd.t < TEST_MAX_STEP - 1)
    {
        printf("t = %d\n", fdtd.t);
        solver.solve_fdtd_far_with_cache();
        cuExecuteBlock(1, 1, set_boundary_value, solver, sine);
        solver.solve_fdtd_near_with_cache();
        cuExecute2D(dim2(res, res), copy_kernel, solver.far_field[fdtd.t], visual_data_far_field, fdtd.t,
                    make_int3(16, 0, 0));
        cuExecute2D(dim2(res, res), copy_kernel, fdtd.grids[fdtd.t], visual_data_fdtd, fdtd.t, make_int3(16, 0, 0));
    }
    TOCK(all_time)
    // visualizer
    renderArray(RenderElement(visual_data_far_field, 0.5f, "far_field"), RenderElement(visual_data_fdtd, 0.5f, "fdtd"));
}
