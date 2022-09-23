#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "pppm.h"
#include "sound_source.h"

#define TEST_MAX_STEP 256
using namespace pppm;

__global__ void set_boundary_sine_signal(PPPMSolver pppm)
{
    int t = pppm.fdtd.t;
}

int main()
{
    int res  = 64;
    float dl = 0.01;
    float dt = 1.0f / 120000;
    PPPMSolver solver(res, dl, dt);
    FDTD &fdtd    = solver.fdtd;
    int3 coord    = make_int3(32, 32, 32);
    float3 center = fdtd.getCenter(coord);
    CArr<float3> vertices(3);
    vertices[0] = center + make_float3(0.0f, 0.0f, 0.0f) * fdtd.dl;
    vertices[1] = center + make_float3(0.0f, 0.0f, 0.1f) * fdtd.dl;
    vertices[2] = center + make_float3(0.0f, 0.1f, 0.0f) * fdtd.dl;
    CArr<int3> triangles(1);
    triangles[0] = make_int3(0, 1, 2);
    solver.set_mesh(vertices, triangles);
    float neumann_amp   = 1.0f;
    float dirichlet_amp = 1.0f;
    float omega         = 2.0f * M_PI * 4000.0f;
    SineSource sine(omega);

    GArr3D<float> visual_data_fdtd(TEST_MAX_STEP, res, res);
    GArr3D<float> visual_data_far_field(TEST_MAX_STEP, res, res);

    //   while (fdtd.t < TEST_MAX_STEP) {
    //     int t = fdtd.t + 1;
    //     neumann[t] = neumann_amp * sine(t * fdtd.dt).imag();
    //     dirichlet[t] = dirichlet_amp * sine(t * fdtd.dt).imag();

    //     auto grid = fdtd.grids[t].cpu();
    //     for (int dx = -1; dx <= 1; dx++) {
    //       for (int dy = -1; dy <= 1; dy++) {
    //         for (int dz = -1; dz <= 1; dz++) {
    //           int3 c = coord + make_int3(dx, dy, dz);
    //           grid(c) = bem.laplace(vertices, PairInfo(src_face,
    //           fdtd.getCenter(c)),
    //                                 neumann, dirichlet, t);
    //         }
    //       }
    //     }
    //     fdtd.grids[t].assign(grid);
    //     check_fdtd[t] = grid(check_coord);
    //     check_bem[t] =
    //         bem.laplace(vertices, PairInfo(src_face,
    //         fdtd.getCenter(check_coord)),
    //                     neumann, dirichlet, t);
    //   }
    // write_to_txt("check_fdtd.txt", check_fdtd, LC_TEST_MAX_STEP);
    // write_to_txt("check_bem.txt", check_bem, LC_TEST_MAX_STEP);
}
