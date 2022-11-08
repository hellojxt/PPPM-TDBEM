#pragma once
#include "fdtd.h"
#include "particle_grid.h"

namespace pppm
{

enum CellType
{
    AIR,
    SOLID,
    GHOST
};

class CellInfo
{
    public:
        Particle nearest_particle;
        float3 nearst_point;
        float nearst_distance;
        CellType type;
};

class GhostCellSolver
{
    public:
        GhostCellSolver(float3 min_pos_, float grid_size_, int grid_dim_, float dt)
        {
            fdtd.init(grid_dim_, grid_size_, dt);
            grid.init(min_pos_, grid_size_, grid_dim_);
            cell_data.resize(grid_dim_, grid_dim_, grid_dim_);
        };

        void set_mesh(CArr<float3> &vertices_, CArr<int3> &triangles_)
        {
            grid.set_mesh(vertices_, triangles_);
            grid.construct_grid();
            precompute_cell_data();
        }

        void precompute_cell_data();

        CGPU_FUNC float inline dt() { return fdtd.dt; }

        CGPU_FUNC float inline dl() { return fdtd.dl; }

        CGPU_FUNC float inline grid_size() { return grid.grid_size; }

        CGPU_FUNC int inline res() { return fdtd.res; }

        CGPU_FUNC float3 inline min_coord() { return grid.min_pos; }

        CGPU_FUNC float3 inline max_coord() { return grid.max_pos; }

        CGPU_FUNC float3 inline center() { return (grid.max_pos + grid.min_pos) / 2; }

        CGPU_FUNC float3 inline size() { return grid.max_pos - grid.min_pos; }

        GArr3D<CellInfo> cell_data;
        ParticleGrid grid;
        FDTD fdtd;
};

}  // namespace pppm
