#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "case_generator.h"
#include "particle_grid.h"

TEST_CASE("Particle Grid", "[pg]")
{
    using namespace pppm;
    PPPMSolver *solver = random_pppm();
    auto &pg = solver->pg;
    int res = solver->fdtd.res;
    float3 min_pos = solver->pg.min_pos;
    float grid_size = solver->pg.grid_size;

    auto particles = pg.particles.cpu();
    auto grid_dense_map = pg.grid_dense_map.cpu();
    auto grid_hash_map = pg.grid_hash_map.cpu();
    CArr<int> particle_flag;
    particle_flag.resize(particles.size());
    particle_flag.reset();
    int grid_num = 0;
    for (int x = 0; x < res; x++)
    {
        for (int y = 0; y < res; y++)
        {
            for (int z = 0; z < res; z++)
            {
                auto range = grid_hash_map(x, y, z);
                if (range.length() == 0)
                {
                    continue;
                }
                grid_num++;
                for (uint i = range.start; i < range.end; i++)
                {
                    SECTION("no particle in 2 cells")
                    {
                        REQUIRE(particle_flag[i] == 0);
                    }
                    particle_flag[i] = 1;
                    auto particle = particles[i];
                    SECTION("particle in its cell")
                    {
                        REQUIRE(particle.pos.x >= x * grid_size + min_pos.x);
                        REQUIRE(particle.pos.x < (x + 1) * grid_size + min_pos.x);
                        REQUIRE(particle.pos.y >= y * grid_size + min_pos.y);
                        REQUIRE(particle.pos.y < (y + 1) * grid_size + min_pos.y);
                        REQUIRE(particle.pos.z >= z * grid_size + min_pos.z);
                        REQUIRE(particle.pos.z < (z + 1) * grid_size + min_pos.z);
                    }
                    SECTION("cell coord is correct")
                    {
                        REQUIRE(particle.cell_coord.x == x);
                        REQUIRE(particle.cell_coord.y == y);
                        REQUIRE(particle.cell_coord.z == z);
                    }
                }
            }
        }
    }

    for (int grid_idx = 0; grid_idx < grid_num; grid_idx++)
    {
        auto first_particle = particles[grid_dense_map[grid_idx].start];
        auto cell_coord = first_particle.cell_coord;
        auto range = grid_hash_map(cell_coord.x, cell_coord.y, cell_coord.z);
        SECTION("Grid hash map and dense map are consistent")
        {
            INFO("grid index is " << grid_idx);
            REQUIRE(grid_dense_map[grid_idx].start == range.start);
            REQUIRE(grid_dense_map[grid_idx].end == range.end);
        }
    }
    SECTION("Grid hash map and dense map are consistent")
    {
        REQUIRE(grid_num == grid_dense_map.size());
    }

    SECTION("all particles in grid")
    {
        REQUIRE(particle_flag.sum() == particles.size());
    }
    pg.clear();
}
