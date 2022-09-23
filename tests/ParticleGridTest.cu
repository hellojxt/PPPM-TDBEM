#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "particle_grid.h"

TEST_CASE("Particle Grid", "[pg]")
{
    using namespace pppm;
    float3 min_pos  = make_float3(0.0f, 0.0f, 0.0f);
    float grid_size = RAND_F;
    int res         = GENERATE(8, 16);
    ParticleGrid pg;
    pg.init(min_pos, grid_size, res);
    int triangle_count = GENERATE(10, 100);
    CArr<float3> vertices;
    CArr<int3> triangles;
    vertices.resize(triangle_count * 3);
    triangles.resize(triangle_count);
    for (int i = 0; i < triangle_count; i++)
    {
        float3 v0           = make_float3(0.0f, 0.0f, 1.0f);
        float3 v1           = make_float3(1.0f, 0.0f, 0.0f);
        float3 v2           = make_float3(0.0f, 1.0f, 0.0f);
        float3 offset       = make_float3(RAND_F, RAND_F, RAND_F) * make_float3(res - 1) * grid_size;
        vertices[i * 3 + 0] = v0 * grid_size + offset;
        vertices[i * 3 + 1] = v1 * grid_size + offset;
        vertices[i * 3 + 2] = v2 * grid_size + offset;
        triangles[i]        = make_int3(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }
    pg.set_mesh(vertices, triangles);
    pg.construct_grid();
    auto particles      = pg.particles.cpu();
    auto grid_dense_map = pg.grid_dense_map.cpu();
    auto grid_hash_map  = pg.grid_hash_map.cpu();
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
                    auto particle    = particles[i];
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
        auto cell_coord     = first_particle.cell_coord;
        auto range          = grid_hash_map(cell_coord.x, cell_coord.y, cell_coord.z);
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
