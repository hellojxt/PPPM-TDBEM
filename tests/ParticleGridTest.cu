#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "case_generator.h"
#include "particle_grid.h"

using Catch::Approx;

TEST_CASE("Particle Grid", "[pg]")
{
    using namespace pppm;
    PPPMSolver *solver = random_pppm(1024);
    auto &pg = solver->pg;
    int res = solver->res();
    float grid_size = pg.grid_size;
    auto vertices = pg.vertices.cpu();
    // randomly move the vertices
    for (int t = 0; t < 5; t++)
    {
        for (int i = 0; i < vertices.size(); i++)
        {
            vertices[i] += make_float3(RAND_F, RAND_F, RAND_F) * grid_size * 0.1;
        }
        solver->update_mesh(vertices);
    }
    auto triangles = pg.triangles.cpu();
    auto indices = pg.faces.cpu();

    // check the triangles
    for (int i = 0; i < triangles.size(); i++)
    {
        auto &tri = triangles[i];
        float3 v1 = vertices[indices[i].x];
        float3 v2 = vertices[indices[i].y];
        float3 v3 = vertices[indices[i].z];
        REQUIRE(tri.indices.x == indices[i].x);
        REQUIRE(tri.indices.y == indices[i].y);
        REQUIRE(tri.indices.z == indices[i].z);
        REQUIRE(tri.area == Approx(0.5f * length(cross(v2 - v1, v3 - v1))).epsilon(1e-3));
        auto normal = normalize(cross(v2 - v1, v3 - v1));
        REQUIRE(tri.normal.x == Approx(normal.x).epsilon(1e-3));
        REQUIRE(tri.normal.y == Approx(normal.y).epsilon(1e-3));
        REQUIRE(tri.normal.z == Approx(normal.z).epsilon(1e-3));
        auto center = (v1 + v2 + v3) / 3.0f;
        REQUIRE(tri.center.x == Approx(center.x).epsilon(1e-3));
        REQUIRE(tri.center.y == Approx(center.y).epsilon(1e-3));
        REQUIRE(tri.center.z == Approx(center.z).epsilon(1e-3));
    }

    // check the grid face list
    auto grid_face_list = pg.grid_face_list.cpu();
    int all_face_num = 0;
    for (int x = 0; x < grid_face_list.size.x; x++)
    {
        for (int y = 0; y < grid_face_list.size.y; y++)
        {
            for (int z = 0; z < grid_face_list.size.z; z++)
            {
                auto &grid = grid_face_list(x, y, z);
                // printf("x: %d, y: %d, z: %d, face num: %d\n", x, y, z, grid.size());
                for (int i = 0; i < grid.size(); i++)
                {
                    auto &tri = triangles[grid[i]];
                    auto pos = tri.center;
                    auto center = pg.getCenter(x, y, z);
                    // printf("x: %d, y: %d, z: %d, face num: %d\n", x, y, z, grid.size());
                    // printf("grid[%d]: %d\n", i, grid[i]);
                    // printf("pos.x: %f, center.x: %f, grid_size: %f\n", pos.x, center.x, grid_size);
                    // printf("pos.y: %f, center.y: %f, grid_size: %f\n", pos.y, center.y, grid_size);
                    // printf("pos.z: %f, center.z: %f, grid_size: %f\n", pos.z, center.z, grid_size);
                    REQUIRE(pos.x >= center.x - grid_size / 2.0f);
                    REQUIRE(pos.x < center.x + grid_size / 2.0f);
                    REQUIRE(pos.y >= center.y - grid_size / 2.0f);
                    REQUIRE(pos.y < center.y + grid_size / 2.0f);
                    REQUIRE(pos.z >= center.z - grid_size / 2.0f);
                    REQUIRE(pos.z < center.z + grid_size / 2.0f);
                }
                all_face_num += grid.size();
            }
        }
    }
    REQUIRE(all_face_num == triangles.size());
    // check the base coord face list
    auto base_coord_face_list = pg.base_coord_face_list.cpu();
    int base_coord_nonempty_num = 0;
    all_face_num = 0;
    for (int x = 0; x < base_coord_face_list.size.x; x++)
    {
        for (int y = 0; y < base_coord_face_list.size.y; y++)
        {
            for (int z = 0; z < base_coord_face_list.size.z; z++)
            {
                auto &grid = base_coord_face_list(x, y, z);
                for (int i = 0; i < grid.size(); i++)
                {
                    auto &tri = triangles[grid[i]];
                    auto pos = tri.center;
                    auto base_coord_pos = pg.getCenter(x, y, z);
                    REQUIRE(pos.x >= base_coord_pos.x);
                    REQUIRE(pos.x < base_coord_pos.x + grid_size);
                    REQUIRE(pos.y >= base_coord_pos.y);
                    REQUIRE(pos.y < base_coord_pos.y + grid_size);
                    REQUIRE(pos.z >= base_coord_pos.z);
                    REQUIRE(pos.z < base_coord_pos.z + grid_size);
                }
                all_face_num += grid.size();
                if (grid.size() > 0)
                {
                    base_coord_nonempty_num++;
                }
            }
        }
    }

    REQUIRE(all_face_num == triangles.size());
    // check the neighbor list
    auto neighbor_3_square_list = pg.neighbor_3_square_list.cpu();
    auto non_empty_data = pg.neighbor_3_square_nonempty.data.cpu();
    auto non_empty_size = pg.neighbor_3_square_nonempty.non_zero_size;
    int non_empty_num = 0;
    for (int x = 0; x < pg.grid_dim; x++)
    {
        for (int y = 0; y < pg.grid_dim; y++)
        {
            for (int z = 0; z < pg.grid_dim; z++)
            {
                int face_num = 0;
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dz = -1; dz <= 1; dz++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nz = z + dz;
                            if (nx < 0 || nx >= pg.grid_dim || ny < 0 || ny >= pg.grid_dim || nz < 0 ||
                                nz >= pg.grid_dim)
                            {
                                continue;
                            }
                            auto &grid = grid_face_list(nx, ny, nz);
                            face_num += grid.size();
                        }
                    }
                }
                if (face_num > 0)
                {
                    non_empty_num++;
                }
            }
        }
    }
    REQUIRE(non_empty_num == non_empty_size);
    for (int i = 0; i < non_empty_size; i++)
    {
        auto coord = non_empty_data[i].coord;
        auto &neighbor_list = neighbor_3_square_list(coord);
        int neighbor_face_num = 0;
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    int nx = coord.x + dx;
                    int ny = coord.y + dy;
                    int nz = coord.z + dz;
                    if (nx < 0 || nx >= pg.grid_dim || ny < 0 || ny >= pg.grid_dim || nz < 0 || nz >= pg.grid_dim)
                    {
                        continue;
                    }
                    auto &grid = grid_face_list(nx, ny, nz);
                    neighbor_face_num += grid.size();
                }
            }
        }
        REQUIRE(neighbor_face_num == neighbor_list.size());
        for (int j = 0; j < neighbor_list.size(); j++)
        {
            auto &tri = triangles[neighbor_list[j]];
            auto pos = tri.center;
            auto center = pg.getCenter(coord);
            REQUIRE(pos.x >= center.x - grid_size * 1.5f);
            REQUIRE(pos.x < center.x + grid_size * 1.5f);
            REQUIRE(pos.y >= center.y - grid_size * 1.5f);
            REQUIRE(pos.y < center.y + grid_size * 1.5f);
            REQUIRE(pos.z >= center.z - grid_size * 1.5f);
            REQUIRE(pos.z < center.z + grid_size * 1.5f);
        }
    }
    // check the neighbor list (base coord)
    auto neighbor_4_square_list = pg.neighbor_4_square_list.cpu();
    auto base_coord_nonempty_data = pg.base_coord_nonempty.data.cpu();
    auto base_coord_nonempty_size = pg.base_coord_nonempty.non_zero_size;
    REQUIRE(base_coord_nonempty_num == base_coord_nonempty_size);
    for (int i = 0; i < base_coord_nonempty_size; i++)
    {
        auto coord = base_coord_nonempty_data[i].coord;
        auto &neighbor_list = neighbor_4_square_list(coord);
        int neighbor_face_num = 0;
        for (int dx = -1; dx <= 2; dx++)
        {
            for (int dy = -1; dy <= 2; dy++)
            {
                for (int dz = -1; dz <= 2; dz++)
                {
                    int nx = coord.x + dx;
                    int ny = coord.y + dy;
                    int nz = coord.z + dz;
                    if (nx < 0 || nx >= pg.grid_dim || ny < 0 || ny >= pg.grid_dim || nz < 0 || nz >= pg.grid_dim)
                    {
                        continue;
                    }
                    auto &grid = grid_face_list(nx, ny, nz);
                    neighbor_face_num += grid.size();
                }
            }
        }
        REQUIRE(neighbor_face_num == neighbor_list.size());
        for (int j = 0; j < neighbor_list.size(); j++)
        {
            auto &tri = triangles[neighbor_list[j]];
            auto pos = tri.center;
            auto center = pg.getCenter(coord);
            REQUIRE(pos.x >= center.x - grid_size * 1.5f);
            REQUIRE(pos.x < center.x + grid_size * 2.5f);
            REQUIRE(pos.y >= center.y - grid_size * 1.5f);
            REQUIRE(pos.y < center.y + grid_size * 2.5f);
            REQUIRE(pos.z >= center.z - grid_size * 1.5f);
            REQUIRE(pos.z < center.z + grid_size * 2.5f);
        }
    }
}
