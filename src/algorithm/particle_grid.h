#pragma once
#include "array3D.h"
#include "macro.h"
#include "morton.h"
#include "helper_math.h"

namespace pppm
{

    class BElement
    {
    public:
        uint cell_id;
        uint3 cell_coord;
        uint particle_id;
        float3 pos;
        float3 normal;
        float area;
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
        GArr<BElement> particles;
        GArr<Range> particle_map;
        GArr3D<int> grid_hash_map;

        ParticleGrid(float3 min_pos_, float grid_size_, int3 grid_dim_)
        {
            min_pos = min_pos_;
            grid_size = grid_size_;
            grid_dim = grid_dim_;
            max_pos = min_pos + make_float3(grid_dim.x, grid_dim.y, grid_dim.z) * grid_size;
        }

        void set_mesh(CArr<float3> vertices_, CArr<int3> triangles_)
        {
            vertices.assign(vertices_);
            triangles.assign(triangles_);
        }

        void construct();
        void validate_data();
        static void randomly_test();
    };
}