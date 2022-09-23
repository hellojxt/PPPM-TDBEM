#pragma once
#include "array3D.h"
#include "helper_math.h"
#include "macro.h"
#include "morton.h"

namespace pppm
{

class BElement
{
    public:
        uint3 cell_coord;
        float3 pos;
        float3 normal;
        int3 indices;
        friend std::ostream &operator<<(std::ostream &out, const BElement &be);
};

class ParticleGrid
{
    public:
        GArr<float3> vertices;
        GArr<int3> triangles;
        float3 min_pos;
        float3 max_pos;
        float grid_size;
        int3 grid_dim;
        GArr<BElement> particles;  // particles sorted by morton code
        GArr<Range>
            grid_dense_map;  // grid_dense_map(i) is the index range (index for particles) of the elements in the
                             // i-th non-empty grid cell. Range is [start, end) and grid is sorted by morton code
        GArr3D<Range> grid_hash_map;  // grid_hash_map(i,j,k) is the index range (index for particles) of the elements
                                      // in the grid cells. If the grid cell is empty, the range is [0,0).

        void init(float3 min_pos_, float grid_size_, int grid_dim_)
        {
            min_pos = min_pos_;
            grid_size = grid_size_;
            grid_dim = make_int3(grid_dim_, grid_dim_, grid_dim_);
            max_pos = min_pos + make_float3(grid_dim.x, grid_dim.y, grid_dim.z) * grid_size;
        }

        void set_mesh(CArr<float3> vertices_, CArr<int3> triangles_)
        {
            vertices.assign(vertices_);
            triangles.assign(triangles_);
        }

        void clear()
        {
            vertices.clear();
            triangles.clear();
            particles.clear();
            grid_dense_map.clear();
            grid_hash_map.clear();
        }

        void construct_grid();
};
}  // namespace pppm