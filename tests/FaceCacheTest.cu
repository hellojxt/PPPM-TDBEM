#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
#include "macro.h"
#include "pppm.h"
#include "sound_source.h"
#include "window.h"

using Catch::Approx;

__global__ void set_signal_kernel(PPPMSolver pppm, SineSource sine)
{
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float neumann_amp = 1e10;
    float dirichlet_amp = 1e10;
    float dt = pppm.dt();
    float t = pppm.time_idx();
    pppm.neumann[face_idx][t] = neumann_amp * sine(dt * t, (face_idx + 1)).real();
    pppm.dirichlet[face_idx][t] = dirichlet_amp * sine(dt * t, (face_idx + 1)).imag();
}

TEST_CASE("FaceCache", "[pc]")
{
    using namespace pppm;
    PPPMSolver *solver = random_pppm(1024, 32);
    auto triangles = solver->pg.triangles.cpu();
    auto vertices = solver->pg.vertices.cpu();
    auto cache = solver->face_cache;

    auto cache_data = cache.cache.cpu();
    auto cache_size = cache.cache_size.cpu();
    auto cache_index = cache.cache_index.cpu();
    auto interpolation_weight = cache.interpolation_weight.cpu();
    for (int i = 0; i < triangles.size(); i++)
    {
        auto &tri = triangles[i];
        SECTION("test interpolation weights")
        {
            float func_params[8];
            for (int j = 0; j < 8; j++)
                func_params[j] = RAND_F;
            auto func = [func_params](float3 coord) {
                int x = coord.x;
                int y = coord.y;
                int z = coord.z;
                return func_params[0] * x + func_params[1] * y + func_params[2] * z + func_params[3] * x * y +
                       func_params[4] * x * z + func_params[5] * y * z + func_params[6] * x * y * z + func_params[7];
            };
            float interpolation_value = 0;
            for (int j = 0; j < 8; j++)
            {
                int3 coord = tri.grid_base_coord + make_int3((j >> 2) & 1, (j >> 1) & 1, j & 1);
                interpolation_value += func(solver->pg.getCenter(coord)) * interpolation_weight(i, j);
            }
            float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
            float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
            float3 dst_v[3] = {{vertices[tri.indices.x]}, {vertices[tri.indices.y]}, {vertices[tri.indices.z]}};
            float trg_jacobian = jacobian(dst_v);
            float ground_truth = 0;
            for (int i = 0; i < TRI_GAUSS_NUM; i++)
            {
                float3 v = local_to_global(guass_x[i][0], guass_x[i][1], dst_v);
                ground_truth += 0.5 * guass_w[i] * func(v) * trg_jacobian;
            }
            REQUIRE(interpolation_value == Approx(ground_truth).margin(trg_jacobian * 1e-5));
        }
    }

    float3 center = make_float3(16, 16, 16) + make_float3(1, 1, 1) * 0.1;
    float3 near_test = make_float3(16, 16, 16) + make_float3(1, 1, 1) * (1.8);
    solver->clear();
    solver = empty_pppm(32);
    float frequency = 3000;
    float omega = 2 * M_PI * frequency;
    SineSource source(omega);

#define PRECOMPUTE_STEP 128
    CArr<float> face_far_field(PRECOMPUTE_STEP);
    CArr<float> face_far_field_from_solver(PRECOMPUTE_STEP);

    solver->clear();
    solver = empty_pppm(32);
    CArr<float> face_near_field(PRECOMPUTE_STEP);
    CArr<float> face_near_field_from_solver(PRECOMPUTE_STEP);

    add_small_triangles(solver, {center, near_test}, 0.1);
    vertices = solver->pg.vertices.cpu();
    triangles = solver->pg.triangles.cpu();
    face_near_field.reset();
    face_near_field_from_solver.reset();
    for (int i = 0; i < PRECOMPUTE_STEP; i++)
    {
        // printf("step: %d\n", i);
        solver->pg.fdtd.step();
        solver->solve_fdtd_far();
        cuExecuteBlock(1, 2, set_signal_kernel, *solver, source);
        solver->update_dirichlet();
        auto neumann = solver->neumann.cpu();
        auto dirichlet = solver->dirichlet.cpu();
        face_near_field_from_solver[i] = dirichlet[0][solver->time_idx()];
        dirichlet[0][solver->time_idx()] = 0;
        float near1 = solver->bem.laplace(vertices.data(), PairInfo(triangles[1].indices, triangles[0].indices),
                                          neumann[1], dirichlet[1], solver->time_idx());
        float near2 = solver->bem.laplace(vertices.data(), PairInfo(triangles[0].indices, triangles[0].indices),
                                          neumann[0], dirichlet[0], solver->time_idx());
        face_near_field[i] = near1 + near2;
        neumann.reset();
        dirichlet.reset();
        dirichlet[0][solver->time_idx()] = 1;
        float factor = 0.5f * triangles[0].area -
                       solver->bem.laplace(vertices.data(), PairInfo(triangles[0].indices, triangles[0].indices),
                                           neumann[0], dirichlet[0], solver->time_idx());
        // printf("factor: %e, self: %e, other: %e, dirichlet: %e\n", factor, near2, near1, face_near_field[i] /
        // factor);
        face_near_field[i] = face_near_field[i] / factor;
        cuExecuteBlock(1, 2, set_signal_kernel, *solver, source);
        solver->solve_fdtd_near();
        // break;
    }
    write_to_txt("face_near_field.txt", face_near_field);
    write_to_txt("face_near_field_from_solver.txt", face_near_field_from_solver);
}
