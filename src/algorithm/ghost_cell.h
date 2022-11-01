#pragma once

#ifndef GHOST_CELL_H
#define GHOST_CELL_H

#include "particle_grid.h"

namespace pppm
{
    class GhostCells
    {
    public:
        GhostCells(float3 min_pos_, float grid_size_, int grid_dim_,
                   CArr<float3> &vertices_, CArr<int3> &triangles_)
        {
            grid.init(min_pos_, grid_size_, grid_dim_);
            grid.set_mesh(vertices_, triangles_);
            grid.construct_grid();
            center2Indices.resize(grid.grid_dense_map.size());
        };

        void fill_in_nearest();

        GArr<int3> center2Indices;
        ParticleGrid grid;
    };

}

#endif