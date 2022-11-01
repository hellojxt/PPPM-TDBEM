#include "ghost_cell.h"

namespace ghost_cell
{
    __device__ float2 get_min_edge02(float a11, float b1)
    {
        int a = b1 < 0, b = a11 + b1 <= 0;
        // y equals to b1 >= 0 ? 0 : (a11 + b1 <= 0 ? 1 : -b1 / a11)
        return make_float2(0.0f, a * (b - (1 - b) * (b1 / a11)));
    }

    __device__ float2 get_min_edge12(float a01, float a11, float b1, float f10, float f01)
    {
        float h0 = a01 + b1 - f10, h1 = a11 + b1 - f01;
        int a = h0 < 0, b = h1 <= 0;
        float y = a * (b + (1 - b) * h0 / (h0 - h1));
        return make_float2(1 - y, y);
    }

    __device__ float2 get_min_interior(float2 p0, float h0, float2 p1, float h1)
    {
        float z = h0 / (h0 - h1);
        float omz = 1 - z;
        return make_float2(omz * p0.x + z * p1.x, omz * p0.y + z * p1.y);
    }

    __device__ float3 get_nearest_triangle_point(float3 point, float3 v0, float3 v1, float3 v2)
    {
        float3 diff = point - v0, edge0 = v1 - v0, edge1 = v2 - v0;
        float a00 = dot(edge0, edge0), a01 = dot(edge0, edge1), a11 = dot(edge1, edge1),
              b0 = -dot(diff, edge0), b1 = -dot(diff, edge1);

        float f00 = b0, f10 = b0 + a00, f01 = b0 + a01;

        float2 p0 = make_float2(0.0f), p1 = make_float2(0.0f), p = make_float2(0.0f);
        float dt1 = 0.0f, h0 = 0.0f, h1 = 0.0f;

        if (f00 >= 0)
        {
            if (f01 >= 0)
            {
                p = get_min_edge02(a11, b1);
            }
            else
            {
                p0 = make_float2(0, f00 / (f00 - f01));
                p1.x = f01 / (f01 - f10);
                p1.y = 1 - p1.x;
                dt1 = p1.y - p0.y;
                h0 = dt1 * (a11 * p0.y + b1);

                if (h0 >= 0)
                {
                    p = get_min_edge02(a11, b1);
                }
                else
                {
                    h1 = dt1 * (a01 * p1.x + a11 * p1.y + b1);
                    if (h1 <= 0)
                    {
                        p = get_min_edge12(a01, a11, b1, f10, f01);
                    }
                    else
                    {
                        p = get_min_interior(p0, h0, p1, h1);
                    }
                }
            }
        }
        else if (f01 <= 0)
        {
            if (f10 <= 0)
            {
                // (3) p0 = (1,0), p1 = (0,1), H(z) = G(L(z)) - F(L(z))
                p = get_min_edge12(a01, a11, b1, f10, f01);
            }
            else
            {
                // (4) p0 = (t00,0), p1 = (t01,1-t01), H(z) = t11*G(L(z))
                p0 = make_float2(f00 / (f00 - f10), 0.0f);
                p1.x = f01 / (f01 - f10);
                p1.y = 1 - p1.x;
                h0 = p1.y * (a01 * p0.x + b1);
                if (h0 >= 0)
                {
                    p = p0; // GetMinEdge01
                }
                else
                {
                    h1 = p1.y * (a01 * p1.x + a11 * p1.y + b1);
                    if (h1 <= 0)
                    {
                        p = get_min_edge12(a01, a11, b1, f10, f01);
                    }
                    else
                    {
                        p = get_min_interior(p0, h0, p1, h1);
                    }
                }
            }
        }
        else if (f10 <= 0)
        {
            // (5) p0 = (0,t10), p1 = (t01,1-t01),
            // H(z) = (t11 - t10)*G(L(z))
            p0 = make_float2(0.0f, f00 / (f00 - f01));
            p1.x = f01 / (f01 - f10);
            p1.y = 1 - p1.x;
            dt1 = p1.y - p0.y;
            h0 = dt1 * (a11 * p0.y + b1);
            if (h0 >= 0)
            {
                p = get_min_edge02(a11, b1);
            }
            else
            {
                h1 = dt1 * (a01 * p1.x + a11 * p1.y + b1);
                if (h1 <= 0)
                {
                    p = get_min_edge12(a01, a11, b1, f10, f01);
                }
                else
                {
                    p = get_min_interior(p0, h0, p1, h1);
                }
            }
        }
        else
        {
            // (6) p0 = (t00,0), p1 = (0,t11), H(z) = t11*G(L(z))
            p0 = make_float2(f00 / (f00 - f10), 0.0f);
            p1.x = 0.0f;
            p1.y = f00 / (f00 - f01);
            h0 = p1.y * (a01 * p0.x + b1);
            if (h0 >= 0)
            {
                p = p0; // GetMinEdge01
            }
            else
            {
                h1 = p1.y * (a11 * p1.y + b1);
                if (h1 <= 0)
                {
                    p = get_min_edge02(a11, b1);
                }
                else
                {
                    p = get_min_interior(p0, h0, p1, h1);
                }
            }
        }

        return v0 + p.x * edge0 + p.y * edge1;
    }

    __global__ void fill_in_nearest_kernel(GArr3D<float3> cells_nearest_facet, pppm::ParticleGrid grid)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        // here we assume grid_dim is (a, a, a).
        int grid_dim = grid.grid_dim.x;
        if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
            return;

        float3 grid_center = (make_float3(x, y, z) + make_float3(0.5f, 0.5f, 0.5f)) * grid.grid_size + grid.min_pos;

        float min_len = 1e5f;
        float3 result = {1e5f, 1e5f, 1e5f};
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) // iterate over all the 3x3x3 grids
                {
                    int3 coord = make_int3(x + dx, y + dy, z + dz);
                    if (coord.x < 0 || coord.x >= grid_dim ||
                        coord.y < 0 || coord.y >= grid_dim ||
                        coord.z < 0 || coord.z >= grid_dim)
                        continue;

                    pppm::Range &neighbors = grid.grid_hash_map(coord);
                    for (int i = neighbors.start; i < neighbors.end; i++)
                    {
                        int3 triID = grid.particles[i].indices;
                        float3 nearest_point =
                            get_nearest_triangle_point(grid_center, grid.vertices[triID.x],
                                                       grid.vertices[triID.y], grid.vertices[triID.z]);
                        float curr_len = length(grid_center - nearest_point);
                        if (curr_len < min_len)
                        {
                            result = nearest_point;
                            min_len = curr_len;
                        }
                    }
                }

        cells_nearest_facet(x, y, z) = result;
        return;
    }

    void GhostCellSolver::fill_in_nearest()
    {
        using namespace pppm;
        int3 size = grid.grid_dim;
        cuExecute3D(size, fill_in_nearest_kernel, cells_nearest_facet, grid);
        return;
    };

}