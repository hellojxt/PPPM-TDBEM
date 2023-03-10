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

#define TEST_MAX_STEP 256

using namespace pppm;

__global__ void set_boundary_value(PPPMSolver pppm, SineSource sine)
{
    float neumann_amp = 1e4;
    float dirichlet_amp = 1e4;
    int t = pppm.time_idx();
    float dt = pppm.dt();
    pppm.neumann[0][t] = neumann_amp * sine(dt * t).real();
    pppm.dirichlet[0][t] = dirichlet_amp * sine(dt * t).real();
}

__device__ float CheckError(const GArr3D<float> &grid, int3 coord)
{
    float error = 0;
    for (int i = max(0, coord.x - 1); i <= min(grid.batchs - 1, coord.x + 1); i++)
    {
        for (int j = max(0, coord.y - 1); j <= min(grid.rows - 1, coord.y + 1); j++)
        {
            for (int k = max(0, coord.z - 1); k <= min(grid.cols - 1, coord.z + 1); k++)
            {
                if (coord.x == i && coord.y == j && coord.z == k)
                    continue;
                error += grid(i, j, k);
            }
        }
    }
    return error / 8;
}

__global__ void SetError(GArr3D<float> grid, int3 coord, GArr<float> result)
{
    result[0] = CheckError(grid, coord);
    return;
}

void empty_far_field_test()
{
    int res = 65;
    auto solver = empty_pppm(res);

    FDTD &fdtd = solver->pg.fdtd;
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

    CArr<float> errors;
    GArr<float> tempErr;
    tempErr.resize(1);
    for (int i = 0; i < TEST_MAX_STEP; i++)
    {
        fdtd.step();
        solver->solve_fdtd_far();
        cuExecuteBlock(1, 1, set_boundary_value, *solver, sine);
        solver->solve_fdtd_near();
        auto &grid = solver->grid_far_field[fdtd.t];
        SetError<<<1, 1>>>(grid, coord, tempErr);
        errors.pushBack(tempErr.cpu()[0]);
    }
    write_to_txt(EXP_DIR + std::string("test/error.txt"), errors);
}

void basic_time_and_space_step_test()
{
    using namespace pppm;
    float hs[3] = {0.005, 0.01, 0.02};
    float dts[3] = {1.0f / 120000 / 2, dts[0] * 2, dts[1] * 2};

    for (int i = 0, p = 1; i < 3; i++, p *= 2)
    {
        float h = hs[i];
        float dt = dts[i];
        FDTD fdtd;
        fdtd.init(64, h, dt);
        TDBEM bem;
        bem.init(fdtd.dt, h);
        int3 coord = make_int3(32, 32, 32);
        float3 center = fdtd.getCenter(coord);

        float3 vertices[3] = {center + make_float3(0.4f, 0.0f, 0.0f) * fdtd.dl,
                              center + make_float3(0.4f, 0.0f, 0.1f) * fdtd.dl,
                              center + make_float3(0.4f, 0.1f, 0.0f) * fdtd.dl};
        int3 src_face = make_int3(0, 1, 2);
        History neumann;
        History dirichlet;
        neumann.reset();
        dirichlet.reset();
        float neumann_amp = 0.01f;
        float dirichlet_amp = 0.01f;
        float omega = 2.0f * M_PI * 2000.0f;
        SineSource sine(omega);

        float check_fdtd[TEST_MAX_STEP];
        float check_bem[TEST_MAX_STEP];
        int3 offset = make_int3(-8 / p, 0, -4 / p);
        int3 check_coord = coord + offset;
        // std::cout << fdtd.getCenter(check_coord) - fdtd.getCenter(coord) << "\n";
        // continue;

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

        CArr<float> errors;
        errors.resize(TEST_MAX_STEP);
        for (int i = 0; i < TEST_MAX_STEP; i++)
        {
            errors[i] = check_bem[i] - check_fdtd[i];
            // errors[i] /= check_bem[i];
        }
        write_to_txt(EXP_DIR + std::string("test/error" + std::to_string(i) + ".txt"), errors);
        fdtd.reset();
    }
}

void advanced_time_and_space_step_test()
{
    float h = 0.01;
    float dt = 1.0f / 120000;
    FDTD fdtd;
    fdtd.init(64, h, dt);
    int3 coord = make_int3(32, 32, 32);
    float3 center = fdtd.getCenter(coord);

    float3 vertices[3] = {center + make_float3(0.4f, 0.0f, 0.0f) * fdtd.dl,
                          center + make_float3(0.4f, 0.0f, 0.1f) * fdtd.dl,
                          center + make_float3(0.4f, 0.1f, 0.0f) * fdtd.dl};
    int3 src_face = make_int3(0, 1, 2);
    History neumann;
    History dirichlet;
    neumann.reset();
    dirichlet.reset();
    float neumann_amp = 0.01f;
    float dirichlet_amp = 0.01f;
    float omega = 2.0f * M_PI * 2000.0f;
    SineSource sine(omega);

    float check_fdtd[TEST_MAX_STEP];
    int3 check_coord = make_int3(30, 32, 31);

    while (fdtd.t + 1 < TEST_MAX_STEP)
    {
        fdtd.step();
        int t = fdtd.t;
        check_fdtd[t] = fdtd.grids[t](check_coord);
    }

    CArr<float> errors;
    errors.resize(TEST_MAX_STEP);
    for (int i = 0; i < TEST_MAX_STEP; i++)
    {
        errors[i] = check_bem[i] - check_fdtd[i];
        // errors[i] /= check_bem[i];
    }
    write_to_txt(EXP_DIR + std::string("test/error" + std::to_string(i) + ".txt"), errors);
    fdtd.reset();
}

int main()
{
    basic_time_and_space_step_test();
    return 0;
}
