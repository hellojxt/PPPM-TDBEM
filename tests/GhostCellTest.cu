#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "case_generator.h"
#include "ghost_cell.h"

// credit : https://github.com/davideberly/GeometricTools/blob/master/GTE/Mathematics/DistPointTriangle.h
namespace CorrectAnswer
{
    class DCPQuery
    {
    public:
        struct Result
        {
            Result()
                : distance(static_cast<float>(0)),
                  sqrDistance(static_cast<float>(0)),
                  barycentric{static_cast<float>(0), static_cast<float>(0), static_cast<float>(0)},
                  closest{make_float3(0.0f), make_float3(0.0f)}
            {
            }

            float distance, sqrDistance;
            std::array<float, 3> barycentric;
            std::array<float3, 2> closest;
        };

        // The query is designed to be robust when using floating-point
        // arithmetic. For arbitrary-precision arithmetic, use the function
        // operator()(...).
        Result UseConjugateGradient(float3 const &point, float3 const &v0,
                                    float3 const &v1, float3 const &v2)
        {
            float const zero = static_cast<float>(0);
            float const one = static_cast<float>(1);
            float3 diff = point - v0;
            float3 edge0 = v1 - v0;
            float3 edge1 = v2 - v0;
            float a00 = dot(edge0, edge0);
            float a01 = dot(edge0, edge1);
            float a11 = dot(edge1, edge1);
            float b0 = -dot(diff, edge0);
            float b1 = -dot(diff, edge1);

            float f00 = b0;
            float f10 = b0 + a00;
            float f01 = b0 + a01;

            std::array<float, 2> p0{}, p1{}, p{};
            float dt1{}, h0{}, h1{};

            // Compute the endpoints p0 and p1 of the segment. The segment is
            // parameterized by L(z) = (1-z)*p0 + z*p1 for z in [0,1] and the
            // directional derivative of half the quadratic on the segment is
            // H(z) = dot(p1-p0,gradient[Q](L(z))/2), where gradient[Q]/2 =
            // (F,G). By design, F(L(z)) = 0 for cases (2), (4), (5), and
            // (6). Cases (1) and (3) can correspond to no-intersection or
            // intersection of F = 0 with the triangle.
            if (f00 >= zero)
            {
                if (f01 >= zero)
                {
                    // (1) p0 = (0,0), p1 = (0,1), H(z) = G(L(z))
                    GetMinEdge02(a11, b1, p);
                }
                else
                {
                    // (2) p0 = (0,t10), p1 = (t01,1-t01),
                    // H(z) = (t11 - t10)*G(L(z))
                    p0[0] = zero;
                    p0[1] = f00 / (f00 - f01);
                    p1[0] = f01 / (f01 - f10);
                    p1[1] = one - p1[0];
                    dt1 = p1[1] - p0[1];
                    h0 = dt1 * (a11 * p0[1] + b1);
                    if (h0 >= zero)
                    {
                        GetMinEdge02(a11, b1, p);
                    }
                    else
                    {
                        h1 = dt1 * (a01 * p1[0] + a11 * p1[1] + b1);
                        if (h1 <= zero)
                        {
                            GetMinEdge12(a01, a11, b1, f10, f01, p);
                        }
                        else
                        {
                            GetMinInterior(p0, h0, p1, h1, p);
                        }
                    }
                }
            }
            else if (f01 <= zero)
            {
                if (f10 <= zero)
                {
                    // (3) p0 = (1,0), p1 = (0,1), H(z) = G(L(z)) - F(L(z))
                    GetMinEdge12(a01, a11, b1, f10, f01, p);
                }
                else
                {
                    // (4) p0 = (t00,0), p1 = (t01,1-t01), H(z) = t11*G(L(z))
                    p0[0] = f00 / (f00 - f10);
                    p0[1] = zero;
                    p1[0] = f01 / (f01 - f10);
                    p1[1] = one - p1[0];
                    h0 = p1[1] * (a01 * p0[0] + b1);
                    if (h0 >= zero)
                    {
                        p = p0; // GetMinEdge01
                    }
                    else
                    {
                        h1 = p1[1] * (a01 * p1[0] + a11 * p1[1] + b1);
                        if (h1 <= zero)
                        {
                            GetMinEdge12(a01, a11, b1, f10, f01, p);
                        }
                        else
                        {
                            GetMinInterior(p0, h0, p1, h1, p);
                        }
                    }
                }
            }
            else if (f10 <= zero)
            {
                // (5) p0 = (0,t10), p1 = (t01,1-t01),
                // H(z) = (t11 - t10)*G(L(z))
                p0[0] = zero;
                p0[1] = f00 / (f00 - f01);
                p1[0] = f01 / (f01 - f10);
                p1[1] = one - p1[0];
                dt1 = p1[1] - p0[1];
                h0 = dt1 * (a11 * p0[1] + b1);
                if (h0 >= zero)
                {
                    GetMinEdge02(a11, b1, p);
                }
                else
                {
                    h1 = dt1 * (a01 * p1[0] + a11 * p1[1] + b1);
                    if (h1 <= zero)
                    {
                        GetMinEdge12(a01, a11, b1, f10, f01, p);
                    }
                    else
                    {
                        GetMinInterior(p0, h0, p1, h1, p);
                    }
                }
            }
            else
            {
                // (6) p0 = (t00,0), p1 = (0,t11), H(z) = t11*G(L(z))
                p0[0] = f00 / (f00 - f10);
                p0[1] = zero;
                p1[0] = zero;
                p1[1] = f00 / (f00 - f01);
                h0 = p1[1] * (a01 * p0[0] + b1);
                if (h0 >= zero)
                {
                    p = p0; // GetMinEdge01
                }
                else
                {
                    h1 = p1[1] * (a11 * p1[1] + b1);
                    if (h1 <= zero)
                    {
                        GetMinEdge02(a11, b1, p);
                    }
                    else
                    {
                        GetMinInterior(p0, h0, p1, h1, p);
                    }
                }
            }

            Result result{};
            result.closest[0] = point;
            result.closest[1] = v0 + p[0] * edge0 + p[1] * edge1;
            diff = result.closest[0] - result.closest[1];
            result.sqrDistance = dot(diff, diff);
            result.distance = std::sqrt(result.sqrDistance);
            result.barycentric[0] = one - p[0] - p[1];
            result.barycentric[1] = p[0];
            result.barycentric[2] = p[1];
            return result;
        }

    private:
        void GetMinEdge02(float const &a11, float const &b1, std::array<float, 2> &p)
        {
            float const zero = static_cast<float>(0);
            float const one = static_cast<float>(1);
            p[0] = zero;
            if (b1 >= zero)
            {
                p[1] = zero;
            }
            else if (a11 + b1 <= zero)
            {
                p[1] = one;
            }
            else
            {
                p[1] = -b1 / a11;
            }
        }

        inline void GetMinEdge12(float const &a01, float const &a11, float const &b1,
                                 float const &f10, float const &f01, std::array<float, 2> &p)
        {
            float const zero = static_cast<float>(0);
            float const one = static_cast<float>(1);
            float h0 = a01 + b1 - f10;
            if (h0 >= zero)
            {
                p[1] = zero;
            }
            else
            {
                float h1 = a11 + b1 - f01;
                if (h1 <= zero)
                {
                    p[1] = one;
                }
                else
                {
                    p[1] = h0 / (h0 - h1);
                }
            }
            p[0] = one - p[1];
        }

        inline void GetMinInterior(std::array<float, 2> const &p0, float const &h0,
                                   std::array<float, 2> const &p1, float const &h1, std::array<float, 2> &p)
        {
            float z = h0 / (h0 - h1);
            float omz = static_cast<float>(1) - z;
            p[0] = omz * p0[0] + z * p1[0];
            p[1] = omz * p0[1] + z * p1[1];
        }
    };
}

void GetCorrectAnswer(pppm::ParticleGrid &grid, pppm::CArr<pppm::Particle> particles,
                      pppm::CArr<float3> &vertices, pppm::CArr3D<Range> &cpuHashMap,
                      int3 grid_cell_id, CArr3D<CorrectAnswer::DCPQuery::Result> &results)
{
    // here we assume grid_dim is (a, a, a).
    int grid_dim = grid.grid_dim.x;

    float3 grid_center = (make_float3(grid_cell_id) + make_float3(0.5f, 0.5f, 0.5f)) * grid.grid_size + grid.min_pos;

    float min_len = 1e5f;
    CorrectAnswer::DCPQuery::Result result;
    result.distance = -1;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) // iterate over all the 3x3x3 grids
            {
                int3 coord = make_int3(grid_cell_id.x + dx, grid_cell_id.y + dy, grid_cell_id.z + dz);
                if (coord.x < 0 || coord.x >= grid_dim ||
                    coord.y < 0 || coord.y >= grid_dim ||
                    coord.z < 0 || coord.z >= grid_dim)
                    continue;

                Range &neighbors = cpuHashMap(coord);
                for (int i = neighbors.start; i < neighbors.end; i++)
                {
                    int3 triID = particles[i].indices;
                    auto nearest_point = CorrectAnswer::DCPQuery().UseConjugateGradient(grid_center, vertices[triID.x],
                                                                                        vertices[triID.y], vertices[triID.z]);
                    if (nearest_point.distance < min_len)
                    {
                        result = nearest_point;
                        min_len = nearest_point.distance;
                    }
                }
            }

    results(grid_cell_id) = result;
}

TEST_CASE("Ghost cell", "[gc]")
{
    using namespace ghost_cell;

    GhostCellSolver *solver = random_ghost_cell(1024);

    TICK(solver_fill_time);
    solver->fill_in_nearest();
    TOCK(solver_fill_time);

    int res = solver->fdtd.res;
    auto cVerts = solver->grid.vertices.cpu();
    auto cTris = solver->grid.triangles.cpu();

    auto myResult = solver->cells_nearest_facet.cpu();
    CArr3D<CorrectAnswer::DCPQuery::Result> correctResult;
    correctResult.resize(res, res, res);

    auto cParticles = solver->grid.particles.cpu();
    auto cHashMap = solver->grid.grid_hash_map.cpu();
    for (int i = 0; i < res; i++)
    {
        for (int j = 0; j < res; j++)
        {
            for (int k = 0; k < res; k++)
            {
                GetCorrectAnswer(solver->grid, cParticles, cVerts, cHashMap, make_int3(i, j, k), correctResult);
            }
        }
    }

    for (int i = 0; i < res; i++)
    {
        for (int j = 0; j < res; j++)
        {
            for (int k = 0; k < res; k++)
            {
                auto tempCorrectResult = correctResult(i, j, k);
                if (tempCorrectResult.distance < 0) // not detected
                    continue;

                REQUIRE(length(tempCorrectResult.closest[1] - myResult(i, j, k)) < 1e-3f);
            }
        }
    }
    return;
}