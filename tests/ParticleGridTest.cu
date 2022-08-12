#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "particle_grid.h"

TEST_CASE("Particle Grid", "[pg]")
{
    using namespace pppm;
    float3 min_pos = make_float3(0.0f, 0.0f, 0.0f);
    float grid_size = RAND_F;
    int res = GENERATE(8, 16);
    int3 grid_dim = make_int3(res, res, res);
    ParticleGrid pg;
    pg.init(min_pos, grid_size, grid_dim);
    int triangle_count = GENERATE(10, 100);
    CArr<float3> vertices;
    CArr<int3> triangles;
    vertices.resize(triangle_count * 3);
    triangles.resize(triangle_count);
    for (int i = 0; i < triangle_count; i++)
    {
        float3 v0 = make_float3(0.0f, 0.0f, 1.0f);
        float3 v1 = make_float3(1.0f, 0.0f, 0.0f);
        float3 v2 = make_float3(0.0f, 1.0f, 0.0f);
        float3 offset = make_float3(RAND_F, RAND_F, RAND_F) * make_float3(grid_dim - 1) * grid_size;
        vertices[i * 3 + 0] = v0 * grid_size + offset;
        vertices[i * 3 + 1] = v1 * grid_size + offset;
        vertices[i * 3 + 2] = v2 * grid_size + offset;
        triangles[i] = make_int3(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }
    pg.set_mesh(vertices, triangles);
    pg.construct_grid();
    auto particles = pg.particles.cpu();
    auto particle_map = pg.particle_map.cpu();
    auto grid_hash_map = pg.grid_hash_map.cpu();
    CArr<int> particle_map_flag;
    particle_map_flag.resize(particles.size());
    particle_map_flag.reset();

    for (int x = 0; x < grid_dim.x; x++)
    {
        for (int y = 0; y < grid_dim.y; y++)
        {
            for (int z = 0; z < grid_dim.z; z++)
            {
                int cell_id = grid_hash_map(x, y, z);
                if (cell_id == -1)
                    continue;
                uint start = particle_map[cell_id].start;
                uint end = particle_map[cell_id].end;
                for (uint i = start; i < end; i++)
                {
                    SECTION("no particle in 2 cells"){
                        REQUIRE(particle_map_flag[i] == 0);
                    }
                    particle_map_flag[i] = 1;
                    auto particle = particles[i];
                    SECTION("particle in its cell"){
                        REQUIRE(particle.pos.x >= x * grid_size + min_pos.x);
                        REQUIRE(particle.pos.x < (x + 1) * grid_size + min_pos.x);
                        REQUIRE(particle.pos.y >= y * grid_size + min_pos.y);
                        REQUIRE(particle.pos.y < (y + 1) * grid_size + min_pos.y);
                        REQUIRE(particle.pos.z >= z * grid_size + min_pos.z);
                        REQUIRE(particle.pos.z < (z + 1) * grid_size + min_pos.z);
                    }
                    SECTION("cell coord is morton code of cell id"){
                        REQUIRE(particle.cell_id == encode_morton(particle.cell_coord));
                    }
                    SECTION("cell coord is correct"){
                        REQUIRE(particle.cell_coord.x == x);
                        REQUIRE(particle.cell_coord.y == y);
                        REQUIRE(particle.cell_coord.z == z);
                    }
                }
            }
        }
    }
    SECTION("all particles in grid"){
        REQUIRE(particle_map_flag.sum() == particles.size());
    }
    particle_map_flag.clear();
}
