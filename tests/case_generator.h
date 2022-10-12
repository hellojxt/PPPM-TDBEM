#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "pppm.h"

using namespace pppm;

static PPPMSolver *empty_pppm()
{
    float3 min_pos = make_float3(0.0f, 0.0f, 0.0f);
    float grid_size = 0.005;
    int res = 64;
    float dt = 8e-6f;
    PPPMSolver *pppm = new PPPMSolver(res, grid_size, dt);
    return pppm;
}

static PPPMSolver *random_pppm()
{
    PPPMSolver *pppm = empty_pppm();
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
        float3 offset = make_float3(RAND_F, RAND_F, RAND_F) * make_float3(pppm->fdtd.res - 1) * pppm->pg.grid_size;
        vertices[i * 3 + 0] = v0 * pppm->pg.grid_size + offset;
        vertices[i * 3 + 1] = v1 * pppm->pg.grid_size + offset;
        vertices[i * 3 + 2] = v2 * pppm->pg.grid_size + offset;
        triangles[i] = make_int3(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }
    pppm->set_mesh(vertices, triangles);
    return pppm;
}

static PPPMSolver *point_pppm()
{
    PPPMSolver *pppm = empty_pppm();
    int center_idx = pppm->fdtd.res / 2;
    int3 coord = make_int3(center_idx, center_idx, center_idx);
    float3 center = pppm->fdtd.getCenter(coord);
    CArr<float3> vertices(3);
    vertices[0] = center + make_float3(0.0f, 0.1f, 0.0f) * pppm->fdtd.dl;
    vertices[1] = center + make_float3(0.0f, 0.0f, 0.0f) * pppm->fdtd.dl;
    vertices[2] = center + make_float3(0.1f, 0.0f, 0.0f) * pppm->fdtd.dl;
    CArr<int3> triangles(1);
    triangles[0] = make_int3(0, 1, 2);
    pppm->set_mesh(vertices, triangles);
    return pppm;
}
