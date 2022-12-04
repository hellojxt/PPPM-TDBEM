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

using namespace pppm;
using Catch::Approx;
template <typename T>
void check_same(CArr<T> const &a, CArr<T> const &b)
{
    REQUIRE(a.size() == b.size());
    for (int i = 0; i < a.size(); ++i)
    {
        REQUIRE(a[i] == b[i]);
    }
}

void check_grid_cache()
{
    auto solver = bunny_pppm(32);
    LOG("Simple algorithm");
    solver->precompute_grid_cache_simple(true);
    auto grid_map_simple = solver->cache.grid_map.cpu();
    auto grid_fdtd_data_simple = solver->cache.grid_fdtd_data.cpu();
    auto grid_data_simple = solver->cache.grid_data.cpu();
    LOG("Fast algorithm");
    solver->precompute_grid_cache(true);
    auto grid_map = solver->cache.grid_map.cpu();
    auto grid_fdtd_data = solver->cache.grid_fdtd_data.cpu();
    auto grid_data = solver->cache.grid_data.cpu();
    check_same(grid_map_simple, grid_map);
    check_same(grid_fdtd_data_simple, grid_fdtd_data);
    check_same(grid_data_simple, grid_data);
    solver->clear();
}

void check_particle_cache()
{
    auto solver = bunny_pppm(64);
    LOG("Simple algorithm");
    solver->precompute_particle_cache_simple(true);
    auto particle_map_simple = solver->cache.particle_map.cpu();
    auto particle_data_simple = solver->cache.particle_data.cpu();
    LOG("Fast algorithm");
    solver->precompute_particle_cache(true);
    auto particle_map = solver->cache.particle_map.cpu();
    auto particle_data = solver->cache.particle_data.cpu();
    for (int i = 0; i < particle_map_simple.size(); ++i)
    {
        REQUIRE(particle_map_simple[i] == particle_map[i]);
    }
    for (int i = 0; i < particle_data_simple.size(); ++i)
    {
        for (int j = 0; j < STEP_NUM; j++)
        {
            // printf("%d %d: %e %e\n", i, j, particle_data_simple[i].weight.single_layer[j],
            //        particle_data[i].weight.single_layer[j]);
            REQUIRE(particle_data_simple[i].weight.single_layer[j] ==
                    Approx(particle_data[i].weight.single_layer[j]).margin(1e-11));
            REQUIRE(particle_data_simple[i].weight.double_layer[j] ==
                    Approx(particle_data[i].weight.double_layer[j]).epsilon(1e-11));
        }
    }
    solver->clear();
}

void check_particle_solver()
{
    auto solver = bunny_pppm(64);
    solver->fdtd.step();
    solver->precompute_particle_cache(true);
    solver->precompute_grid_cache(true);
    auto far_field = solver->far_field[0].cpu();
    for (int i = 0; i < far_field.data.size(); i++)
    {
        far_field.data[i] = RAND_F;
    }
    auto dirichlet = solver->dirichlet.cpu();
    auto neumann = solver->neumann.cpu();
    for (int i = 0; i < dirichlet.size(); i++)
    {
        for (int j = 0; j < 2 * STEP_NUM; j++)
        {
            dirichlet[i][j] = RAND_F;
            neumann[i][j] = RAND_F;
        }
    }
    solver->neumann.assign(neumann);
    solver->dirichlet.assign(dirichlet);
    solver->far_field[0].assign(far_field);
    LOG("Simple algorithm");
    solver->update_particle_dirichlet_simple(true);
    auto dirichlet_solved_simple = solver->dirichlet.cpu();
    solver->neumann.assign(neumann);
    solver->dirichlet.assign(dirichlet);
    solver->far_field[0].assign(far_field);
    LOG("Fast algorithm");
    solver->update_particle_dirichlet(true);
    auto dirichlet_solved = solver->dirichlet.cpu();
    REQUIRE(dirichlet_solved.size() == dirichlet_solved_simple.size());
    for (int i = 0; i < dirichlet_solved.size(); i++)
    {
        for (int j = 0; j < 2 * STEP_NUM; j++)
        {
            REQUIRE(dirichlet_solved[i][j] == Approx(dirichlet_solved_simple[i][j]).margin(1e-3));
        }
    }
    solver->clear();
}

TEST_CASE("GridCache", "[gc]")
{
    // check_grid_cache();
    // check_particle_cache();
    check_particle_solver();
}

/**
----------------Cost Time--------------
Set Grid Cache Size time: 0.343622 ms
Cache Grid Data time: 6521.72 ms
Set Particle Cache Size time: 1.44124 ms
Cache Particle Data time: 12913.7 ms
FDTD time: 0.035987 ms
Far Solve from Cache time: 0.110819 ms
Particle Solve from Cache time: 4.2936 ms
Near Solve from Cache time: 0.114656 ms
*/