#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "ghost_cell.h"
#include "pppm.h"
#include "objIO.h"

using namespace pppm;

static PPPMSolver *empty_pppm(int res)
{
    float3 min_pos = make_float3(0.0f, 0.0f, 0.0f);
    float grid_size = 0.005;
    float dt = 8e-6f;
    PPPMSolver *pppm = new PPPMSolver(res, grid_size, dt);
    return pppm;
}

static PPPMSolver *bunny_pppm(int res)
{
    PPPMSolver *solver = empty_pppm(res);
    auto filename = ASSET_DIR + std::string("sphere.obj");
    auto mesh = Mesh::loadOBJ(filename, true);
    mesh.stretch_to(solver->size().x / 4.0f);
    mesh.move_to(solver->center());
    solver->set_mesh(mesh.vertices, mesh.triangles);
    return solver;
}

static PPPMSolver *random_pppm(int triangle_count, int res = 64)
{
    PPPMSolver *pppm = empty_pppm(res);
    CArr<float3> vertices;
    CArr<int3> triangles;
    vertices.resize(triangle_count * 3);
    triangles.resize(triangle_count);
    for (int i = 0; i < triangle_count; i++)
    {
        float3 v0 = make_float3(0.0f, 0.0f, 1.0f) * RAND_F * RAND_SIGN;
        float3 v1 = make_float3(1.0f, 0.0f, 0.0f) * RAND_F * RAND_SIGN;
        float3 v2 = make_float3(0.0f, 1.0f, 0.0f) * RAND_F * RAND_SIGN;
        float3 offset = make_float3(RAND_F, RAND_F, RAND_F) * make_float3(pppm->fdtd.res - 4) * pppm->pg.grid_size +
                        make_float3(2.0f, 2.0f, 2.0f) * pppm->pg.grid_size;
        vertices[i * 3 + 0] = v0 * pppm->pg.grid_size + offset;
        vertices[i * 3 + 1] = v1 * pppm->pg.grid_size + offset;
        vertices[i * 3 + 2] = v2 * pppm->pg.grid_size + offset;
        triangles[i] = make_int3(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }
    pppm->set_mesh(vertices, triangles);
    return pppm;
}

static void add_small_triangles(PPPMSolver *pppm, const std::vector<float3> &points, float radius)
{
    CArr<float3> vertices;
    CArr<int3> triangles;
    vertices.resize(points.size() * 3);
    triangles.resize(points.size());
    for (int i = 0; i < points.size(); i++)
    {
        float3 v0 = make_float3(0.0f, 0.0f, 1.0f) * radius * RAND_SIGN;
        float3 v1 = make_float3(1.0f, 0.0f, 0.0f) * radius * RAND_SIGN;
        float3 v2 = make_float3(0.0f, 1.0f, 0.0f) * radius * RAND_SIGN;
        float3 offset = points[i] * pppm->pg.grid_size;
        vertices[i * 3 + 0] = v0 * pppm->pg.grid_size + offset;
        vertices[i * 3 + 1] = v1 * pppm->pg.grid_size + offset;
        vertices[i * 3 + 2] = v2 * pppm->pg.grid_size + offset;
        triangles[i] = make_int3(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }
    pppm->set_mesh(vertices, triangles);
}

static PPPMSolver *point_pppm(int res = 64)
{
    PPPMSolver *pppm = empty_pppm(res);
    int center_idx = pppm->fdtd.res / 2;
    int3 coord = make_int3(center_idx, center_idx, center_idx);
    float3 center = pppm->pg.getCenter(coord);
    CArr<float3> vertices(3);
    vertices[0] = center + make_float3(0.0f, 0.1f, 0.0f) * pppm->fdtd.dl;
    vertices[1] = center + make_float3(0.0f, 0.0f, 0.0f) * pppm->fdtd.dl;
    vertices[2] = center + make_float3(0.1f, 0.0f, 0.0f) * pppm->fdtd.dl;
    CArr<int3> triangles(1);
    triangles[0] = make_int3(0, 1, 2);
    pppm->set_mesh(vertices, triangles);
    return pppm;
}

static GhostCellSolver *empty_ghost_cell_solver(int res)
{
    float3 min_pos = make_float3(0.0f, 0.0f, 0.0f);
    float grid_size = 0.005;
    float dt = 8e-6f;
    GhostCellSolver *solver = new GhostCellSolver(min_pos, grid_size, res, dt);
    return solver;
}
