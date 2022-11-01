#pragma once

#ifndef GHOST_CELL_H
#define GHOST_CELL_H

#include "particle_grid.h"
#include "fdtd.h"

namespace ghost_cell
{
    template <typename T>
    using CArr = pppm::CArr<T>;

    template <typename T>
    using GArr3D = pppm::GArr3D<T>;

    class GhostCellSolver
    {
    public:
        GhostCellSolver(float3 min_pos_, float grid_size_, int grid_dim_, float dt,
                        CArr<float3> &vertices_, CArr<int3> &triangles_)
        {
            fdtd.init(grid_dim_, grid_size_, dt);

            grid.init(min_pos_, grid_size_, grid_dim_);
            grid.set_mesh(vertices_, triangles_);
            grid.construct_grid();
            cells_nearest_facet.resize(grid_dim_, grid_dim_, grid_dim_);
            return;
        };

        void fill_in_nearest();

        GArr3D<float3> cells_nearest_facet;
        pppm::ParticleGrid grid;
        pppm::FDTD fdtd;
    };

}

#endif