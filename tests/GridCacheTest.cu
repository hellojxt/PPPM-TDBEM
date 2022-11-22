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

void test_precompute_grid_cache(PPPMSolver *solver)
{
    int pair_num = 0;
    int involved_grid_num = 0;
    int res = solver->fdtd.res;
    float grid_size = solver->pg.grid_size;
    auto grid_hash_map = solver->pg.grid_hash_map.cpu();
    PPPMCache &cache = solver->cache;
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
        for (int grid_id = 0; grid_id < grid_map.size(); grid_id++)
        {
            GridMap m = grid_map[grid_id];
            int3 coord = m.coord;
            float3 center = solver->pg.getCenter(coord);
            Range r = m.range;
            for (int i = r.start; i < r.end; i++)
            {
                BEMCache e1 = grid_data[i];
                BEMCache e2 = grid_fdtd_data[i];
                REQUIRE(e1.particle_id == e2.particle_id);
                Particle b = particles[e1.particle_id];
                float3 dist = fabs(b.pos - center);
                REQUIRE(dist.x < 1.5 * grid_size);
                REQUIRE(dist.y < 1.5 * grid_size);
                REQUIRE(dist.z < 1.5 * grid_size);
            }
        }
    }
}

void test_cache_weight(PPPMSolver *solver)
{

    SECTION("check cache weight")
    {
        auto dirichlet = solver->dirichlet.cpu();
        int test_idx = 0;
        for (int i = 0; i < dirichlet.size(); i++)
        {
            dirichlet[i][test_idx - 1] = 1;
        }
        solver->dirichlet.assign(dirichlet);

        solver->solve_fdtd_far_simple();
        solver->solve_fdtd_near_simple();
        CArr3D<float> far_field_simple = solver->far_field[0].cpu();
        CArr3D<float> fdtd_grid_simple = solver->fdtd.grids[0].cpu();

        solver->precompute_grid_cache();

        solver->fdtd.reset();
        solver->solve_fdtd_far_with_cache();
        solver->solve_fdtd_near_with_cache();
        CArr3D<float> far_field = solver->far_field[0].cpu();
        CArr3D<float> fdtd_grid = solver->fdtd.grids[0].cpu();
        for (int x = solver->fdtd.res / 2 - 2; x <= solver->fdtd.res / 2 + 2; x++)
        {
            for (int y = solver->fdtd.res / 2 - 2; y <= solver->fdtd.res / 2 + 2; y++)
            {
                for (int z = solver->fdtd.res / 2 - 2; z <= solver->fdtd.res / 2 + 2; z++)
                {
                    if (far_field_simple(x, y, z) > 0)
                    {
                        REQUIRE(abs(far_field(x, y, z) - far_field_simple(x, y, z)) <
                                abs(far_field_simple(x, y, z)) * 1e-3);
                    }
                }
            }
        }
    }
}
TEST_CASE("GridCache", "[gc]")
{
    using namespace pppm;
    PPPMSolver *solver[2] = {point_pppm(), random_pppm(256)};

    for (int i = 0; i < 2; i++)
    {
        test_precompute_grid_cache(solver[i]);
        test_cache_weight(solver[i]);
        solver[i]->clear();
        delete solver[i];
    }
}
