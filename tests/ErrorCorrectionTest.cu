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

__device__ float CheckError(const GArr3D<float>& grid, int3 coord)
{
    float error = 0;
    for(int i = max(0, coord.x - 1); i <= min(grid.batchs - 1, coord.x + 1); i++)
    {
        for(int j = max(0, coord.y - 1); j <= min(grid.rows - 1, coord.y + 1); j++)
        {
            for (int k = max(0, coord.z - 1); k <= min(grid.cols - 1, coord.z + 1); k++)
            {
                if(coord.x == i && coord.y == j && coord.z == k)
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

int main()
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
        auto& grid = solver->grid_far_field[fdtd.t];
        SetError<<<1, 1>>>(grid, coord, tempErr);
        errors.pushBack(tempErr.cpu()[0]);
    }
    write_to_txt(EXP_DIR + std::string("test/error.txt"), errors);
    return 0;
}
