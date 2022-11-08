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
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float neumann_amp = 1e10;
    float dirichlet_amp = 1e10;
    float dt = pppm.fdtd.dt;
    float t = pppm.fdtd.t;
    pppm.particle_history[particle_idx].neumann[t] = neumann_amp * sine(dt * t, (particle_idx + 1)).real();
    pppm.particle_history[particle_idx].dirichlet[t] = dirichlet_amp * sine(dt * t, (particle_idx + 1)).imag();
}

TEST_CASE("ParticleCache", "[pc]")
{
    using namespace pppm;
    PPPMSolver *solver = random_pppm(1024, 32);
    solver->precompute_particle_cache();
    auto particle_map = solver->cache.particle_map.cpu();
    auto particle_data = solver->cache.particle_data.cpu();
    auto particles = solver->pg.particles.cpu();
    auto vertices = solver->pg.vertices.cpu();
    REQUIRE(particle_map.size() == particles.size());

    for (int i = 0; i < particles.size(); i++)
    {
        auto particle = particles[i];
        auto r = particle_map[i].range;
        auto base_coord = particle_map[i].base_coord;
        auto dcoord = particle.pos - solver->pg.getCenter(base_coord);
        SECTION("test particle cache info (cache size and neighbor list)")
        {

            REQUIRE((dcoord.x >= 0 && dcoord.x <= solver->fdtd.dl));
            REQUIRE((dcoord.y >= 0 && dcoord.y <= solver->fdtd.dl));
            REQUIRE((dcoord.z >= 0 && dcoord.z <= solver->fdtd.dl));
            float3 center = (solver->pg.getCenter(base_coord) + solver->pg.getCenter(base_coord + 1)) / 2;
            int neighbor_number = 0;
            for (int j = 0; j < particles.size(); j++)
            {
                auto other_particle = particles[j];
                auto other_dcoord = other_particle.pos - center;
                auto max_dim_length = std::max(std::max(abs(other_dcoord.x), abs(other_dcoord.y)), abs(other_dcoord.z));
                if (max_dim_length < solver->fdtd.dl * 2)
                {
                    int same_particle_id_num = 0;
                    for (int k = r.start; k < r.end; k++)
                    {
                        if (particle_data[k].particle_id == j)
                        {
                            same_particle_id_num++;
                        }
                    }
                    REQUIRE(same_particle_id_num == 1);
                    neighbor_number++;
                }
            }
            REQUIRE(r.end - r.start == neighbor_number);
        }
        SECTION("test weights in particle cahe")
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
            auto &weights = particle_map[i].weight;
            float interpolation_value = 0;
            for (int j = 0; j < 8; j++)
            {
                int3 coord = base_coord + make_int3((j >> 2) & 1, (j >> 1) & 1, j & 1);
                interpolation_value += func(solver->pg.getCenter(coord)) * weights[j];
            }
            float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
            float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
            float3 dst_v[3] = {
                {vertices[particle.indices.x]}, {vertices[particle.indices.y]}, {vertices[particle.indices.z]}};
            float trg_jacobian = jacobian(dst_v);
            float ground_truth = 0;
            for (int i = 0; i < TRI_GAUSS_NUM; i++)
            {
                float3 v = local_to_global(guass_x[i][0], guass_x[i][1], dst_v);
                ground_truth += 0.5 * guass_w[i] * func(v) * trg_jacobian;
            }
            REQUIRE(interpolation_value == Approx(ground_truth).margin(1e-3));
        }
    }

    float3 center_offset = make_float3(RAND_SIGN, RAND_SIGN, RAND_SIGN) * 0.3;
    float3 center = make_float3(16, 16, 16) + center_offset;
    float3 near_test_offset = make_float3(RAND_SIGN, RAND_SIGN, RAND_SIGN) * (1.4 + RAND_F * 0.5);
    float3 near_test = make_float3(16, 16, 16) + near_test_offset;
    float3 far_test_offset = make_float3(RAND_SIGN, RAND_SIGN, RAND_SIGN) * (4.3 + RAND_F * 0.5);
    float3 far_test = make_float3(16, 16, 16) + far_test_offset;
    LOG("center:" << center);
    LOG("near_test:" << near_test);
    LOG("far_test:" << far_test);
    solver->clear();
    solver = empty_pppm(32);
    float frequency = 3000;
    float omega = 2 * M_PI * frequency;
    SineSource source(omega);

#define PRECOMPUTE_STEP 128
    CArr<float> particle_far_field(PRECOMPUTE_STEP);
    CArr<float> particle_far_field_from_solver(PRECOMPUTE_STEP);
    GArr3D<float> visual_data_far_field(PRECOMPUTE_STEP, 32, 32);

    add_small_triangles(solver, {center, far_test}, 0.1);
    vertices = solver->pg.vertices.cpu();
    particles = solver->pg.particles.cpu();
    solver->precompute_grid_cache();
    solver->precompute_particle_cache();
    particle_far_field.reset();
    particle_far_field_from_solver.reset();
    for (int i = 0; i < PRECOMPUTE_STEP; i++)
    {
        solver->solve_fdtd_far_with_cache();
        cuExecuteBlock(1, 2, set_signal_kernel, *solver, source);
        solver->update_particle_dirichlet();
        auto far_history = solver->particle_history.cpu();
        particle_far_field_from_solver[i] = far_history[0].dirichlet[solver->fdtd.t];
        far_history[0].dirichlet[solver->fdtd.t] = 0;
        particle_far_field[i] =
            solver->bem.laplace(vertices.data(), PairInfo(particles[1].indices, particles[0].indices),
                                far_history[1].neumann, far_history[1].dirichlet, solver->fdtd.t) +
            solver->bem.laplace(vertices.data(), PairInfo(particles[0].indices, particles[0].indices),
                                far_history[0].neumann, far_history[0].dirichlet, solver->fdtd.t);
        far_history.reset();
        far_history[0].dirichlet[solver->fdtd.t] = 1;
        float factor =
            1.0f / 2 - solver->bem.laplace(vertices.data(), PairInfo(particles[0].indices, particles[0].indices),
                                           far_history[0].neumann, far_history[0].dirichlet, solver->fdtd.t);
        particle_far_field[i] = particle_far_field[i] / factor;
        cuExecuteBlock(1, 2, set_signal_kernel, *solver, source);
        solver->solve_fdtd_near_with_cache();
        visual_data_far_field[i].assign(solver->far_field[i][15]);
        // printf("%d: far field: %e, %e\n", i, particle_far_field[i], particle_far_field_from_solver[i]);
    }
    write_to_txt("particle_far_field.txt", particle_far_field);
    write_to_txt("particle_far_field_from_solver.txt", particle_far_field_from_solver);
    // renderArray(RenderElement(visual_data_far_field, 2e10f, "far_field"));

    solver->clear();
    solver = empty_pppm(32);
    CArr<float> particle_near_field(PRECOMPUTE_STEP);
    CArr<float> particle_near_field_from_solver(PRECOMPUTE_STEP);

    add_small_triangles(solver, {center, near_test}, 0.1);
    vertices = solver->pg.vertices.cpu();
    particles = solver->pg.particles.cpu();
    solver->precompute_grid_cache();
    solver->precompute_particle_cache();
    particle_near_field.reset();
    particle_near_field_from_solver.reset();
    for (int i = 0; i < PRECOMPUTE_STEP; i++)
    {
        solver->solve_fdtd_far_with_cache();
        cuExecuteBlock(1, 2, set_signal_kernel, *solver, source);
        solver->update_particle_dirichlet();
        auto near_history = solver->particle_history.cpu();
        particle_near_field_from_solver[i] = near_history[0].dirichlet[solver->fdtd.t];
        near_history[0].dirichlet[solver->fdtd.t] = 0;
        particle_near_field[i] =
            solver->bem.laplace(vertices.data(), PairInfo(particles[1].indices, particles[0].indices),
                                near_history[1].neumann, near_history[1].dirichlet, solver->fdtd.t) +
            solver->bem.laplace(vertices.data(), PairInfo(particles[0].indices, particles[0].indices),
                                near_history[0].neumann, near_history[0].dirichlet, solver->fdtd.t);
        near_history.reset();
        near_history[0].dirichlet[solver->fdtd.t] = 1;
        float factor =
            1.0f / 2 - solver->bem.laplace(vertices.data(), PairInfo(particles[0].indices, particles[0].indices),
                                           near_history[0].neumann, near_history[0].dirichlet, solver->fdtd.t);
        particle_near_field[i] = particle_near_field[i] / factor;
        cuExecuteBlock(1, 2, set_signal_kernel, *solver, source);
        solver->solve_fdtd_near_with_cache();
        // printf("%d: near field: %e, %e\n", i, particle_near_field[i], particle_near_field_from_solver[i]);
    }
    write_to_txt("particle_near_field.txt", particle_near_field);
    write_to_txt("particle_near_field_from_solver.txt", particle_near_field_from_solver);
}
