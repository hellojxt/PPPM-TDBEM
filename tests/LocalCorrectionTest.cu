#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "fdtd.h"
#include "sound_source.h"

// FIXME: need to be fixed for new PPPM
int main()
{
#define TEST_MAX_STEP 256

    using namespace pppm;
    FDTD fdtd;
    fdtd.init(64, 0.01, 1.0f / 120000);
    TDBEM bem;
    bem.init(fdtd.dt);
    int3 coord = make_int3(32, 32, 32);
    float3 center = fdtd.getCenter(coord);
    float3 vertices[3] = {center + make_float3(0.3f, 0.0f, 0.0f) * fdtd.dl,
                          center + make_float3(0.3f, 0.0f, 0.1f) * fdtd.dl,
                          center + make_float3(0.3f, 0.1f, 0.0f) * fdtd.dl};
    int3 src_face = make_int3(0, 1, 2);
    History neumann;
    History dirichlet;
    neumann.reset();
    dirichlet.reset();
    float neumann_amp = 0.01f;
    float dirichlet_amp = 0.01f;
    float omega = 2.0f * M_PI * 4000.0f;
    SineSource sine(omega);

    float check_fdtd[TEST_MAX_STEP];
    float check_bem[TEST_MAX_STEP];
    int3 check_coord = make_int3(30, 32, 28);

    while (fdtd.t + 1 < TEST_MAX_STEP)
    {
        fdtd.step();
        int t = fdtd.t;
        neumann[t] = neumann_amp * sine(t * fdtd.dt).imag();
        dirichlet[t] = dirichlet_amp * sine(t * fdtd.dt).imag();
        auto grid = fdtd.grids[t].cpu();
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    int3 c = coord + make_int3(dx, dy, dz);
                    grid(c) = bem.laplace(vertices, PairInfo(src_face, fdtd.getCenter(c)), neumann, dirichlet, t);
                }
            }
        }
        fdtd.grids[t].assign(grid);
        check_fdtd[t] = grid(check_coord);
        check_bem[t] = bem.laplace(vertices, PairInfo(src_face, fdtd.getCenter(check_coord)), neumann, dirichlet, t);
    }
    write_to_txt("check_fdtd.txt", check_fdtd, TEST_MAX_STEP);
    write_to_txt("check_bem.txt", check_bem, TEST_MAX_STEP);
}
