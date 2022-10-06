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

void sub_test(PPPMSolver *solver)
{
    PPPMCache &cache = solver->cache;
    int res = solver->fdtd.res;
    float grid_size = solver->pg.grid_size;
    auto grid_hash_map = solver->pg.grid_hash_map.cpu();
    int pair_num = 0;
    int involved_grid_num = 0;
    SECTION("check size of grid_map and grid_data")
    {
        for (int x = 1; x < res - 1; x++)
        {
            for (int y = 1; y < res - 1; y++)
            {
                for (int z = 1; z < res - 1; z++)
                {
                    int3 center = make_int3(x, y, z);
                    int neighbor_num = 0;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                if (dx == 0 && dy == 0 && dz == 0)
                                    continue;
                                int3 neighbor = make_int3(x + dx, y + dy, z + dz);
                                neighbor_num += grid_hash_map(neighbor).length();
                            }
                        }
                    }
                    pair_num += neighbor_num;
                    if (neighbor_num > 0)
                    {
                        involved_grid_num++;
                    }
                }
            }
        }
        solver->precompute_grid_cache();
        REQUIRE(pair_num == solver->cache.grid_data.size());
        REQUIRE(pair_num == solver->cache.grid_fdtd_data.size());
        REQUIRE(involved_grid_num == solver->cache.grid_map.size());
    }

    SECTION("check particle id")
    {
        auto grid_map = cache.grid_map.cpu();
        auto grid_data = cache.grid_data.cpu();
        auto grid_fdtd_data = cache.grid_fdtd_data.cpu();
        auto particles = solver->pg.particles.cpu();
        CArr<int> particle_flag(particles.size());
        particle_flag.reset();
        for (int grid_id = 0; grid_id < grid_map.size(); grid_id++)
        {
            GridMap m = grid_map[grid_id];
            int3 coord = m.coord;
            float3 center = solver->fdtd.getCenter(coord);
            Range r = m.range;
            for (int i = r.start; i < r.end; i++)
            {
                BEMCache e1 = grid_data[i];
                BEMCache e2 = grid_fdtd_data[i];
                REQUIRE(e1.particle_id == e2.particle_id);
                REQUIRE(particle_flag[e1.particle_id] == 0);
                particle_flag[e1.particle_id] = 1;
                BElement b = particles[e1.particle_id];
                float3 dist = fabs(b.pos - center);
                REQUIRE(dist.x < 1.5 * grid_size);
                REQUIRE(dist.y < 1.5 * grid_size);
                REQUIRE(dist.z < 1.5 * grid_size);
            }
        }
    }

    SECTION("check cache weight")
    {
        CArr<BoundaryHistory> particle_history = solver->particle_history.cpu();
    }
}
TEST_CASE("GridCache", "[gc]")
{
    using namespace pppm;
    PPPMSolver *solver;
    SECTION("point PPPM")
    {
        solver = point_pppm();
        sub_test(solver);
        solver->clear();
    }
    // SECTION("random PPPM")
    // {
    //     solver = random_pppm();
    //     sub_test(solver);
    //     solver->clear();
    // }
}
