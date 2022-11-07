#pragma once
#include <cstdio>
#include "array3D.h"
#include "bem.h"
#include "fdtd.h"
#include "particle_grid.h"

namespace pppm
{

class BEMCache
{
    public:
        int particle_id;
        LayerWeight weight;
        CGPU_FUNC BEMCache() {}
};

class GridMap
{
    public:
        int3 coord;
        Range range;  // index range in grid data
        CGPU_FUNC GridMap() {}
        CGPU_FUNC GridMap(int3 coord_, Range range_) : coord(coord_), range(range_) {}
};

class ParticleMap
{
    public:
        int3 base_coord;  // lowest coordinate of the 2*2*2 grid cell
        float weight[8];  // weight of the 8 grid cells
        Range range;      // index range in particle data
        CGPU_FUNC ParticleMap() {}
};

class PPPMCache
{
    public:
        /* data for cache */
        GArr<GridMap> grid_map;
        GArr<BEMCache> grid_data;       // for solving accurate near field of BEM
        GArr<BEMCache> grid_fdtd_data;  // for solving inaccurate near field of FDTD

        GArr<ParticleMap> particle_map;  // mapping particle id to index range in particle data
                                         // (first index is the data of particle self)
        GArr<BEMCache> particle_data;    // for solving accurate near field of BEM

        /* data for precomputation of cache size */
        // some grid cells have no neighbors
        GArr3D<int> grid_neighbor_nonzero;
        GArr3D<int> grid_neighbor_num;
        // All particles have neighbors
        GArr<int> particle_neighbor_num;

        void clear()
        {
            grid_map.clear();
            grid_data.clear();
            grid_fdtd_data.clear();
            particle_map.clear();
            particle_data.clear();
        }

        void reset()
        {
            grid_map.reset();
            grid_data.reset();
            grid_fdtd_data.reset();
            particle_map.reset();
            particle_data.reset();
        }
};

class BoundaryHistory
{
    public:
        History dirichlet;
        History neumann;
        friend std::ostream &operator<<(std::ostream &out, const BoundaryHistory &h)
        {
            out << "dirichlet: " << h.dirichlet << std::endl;
            out << "neumann: " << h.neumann << std::endl;
            return out;
        }
};

class PPPMSolver
{
    public:
        FDTD fdtd;  // The left corner of the fdtd grid is at (0,0,0)
        ParticleGrid pg;
        GArr<BoundaryHistory> particle_history;  // history boundary data of particles
        GridArr far_field;                       // far field potential of grid cells
        TDBEM bem;                               // boundary element method solver
        PPPMCache cache;                         // cache for near field computation weights

        /**
         *   Constructor of PPPMSolver
         *   @param res_: resolution of the fdtd grid
         *   @param dl_: grid cell size
         *   @param dt_: time step for the FDTD solver
         */
        PPPMSolver(int res_, float dl_, float dt_)
        {
            fdtd.init(res_, dl_, dt_);
            pg.init(make_float3(0, 0, 0), dl_, res_);
            bem.init(dt_);
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                far_field[i].resize(res_, res_, res_);
                far_field[i].reset();
            }
        }

        CGPU_FUNC float inline dt() { return fdtd.dt; }

        CGPU_FUNC float inline dl() { return fdtd.dl; }

        CGPU_FUNC float inline grid_size() { return pg.grid_size; }

        CGPU_FUNC int inline res() { return fdtd.res; }

        CGPU_FUNC float3 inline min_coord() { return pg.min_pos; }

        CGPU_FUNC float3 inline max_coord() { return pg.max_pos; }

        CGPU_FUNC float3 inline center() { return (pg.max_pos + pg.min_pos) / 2; }

        CGPU_FUNC float3 inline size() { return pg.max_pos - pg.min_pos; }

        // set mesh for the particle grid
        void set_mesh(CArr<float3> &verts_, CArr<int3> &tris_)
        {
            if (fdtd.t > -1)
                remove_current_mesh();
            pg.set_mesh(verts_, tris_);
            pg.construct_grid();
            particle_history.resize(pg.particles.size());
            particle_history.reset();
        }

        void remove_current_mesh()
        {
            pg.clear();
            particle_history.clear();
            int t = fdtd.t;
            for (int i = 0; i < GRID_TIME_SIZE; i++)
                far_field[t - i].assign(fdtd.grids[t - i]);
        }

        void clear()
        {
            fdtd.clear();
            pg.clear();
            particle_history.clear();
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                far_field[i].clear();
            }
            cache.clear();
        }

        void reset()
        {
            fdtd.reset();
            pg.reset();
            particle_history.reset();
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                far_field[i].reset();
            }
            cache.reset();
        }

        /*
            1. solve fdtd
            2. update far field
        */
        void solve_fdtd_far_simple();

        void solve_fdtd_near_simple();

        void precompute_grid_cache();

        void solve_fdtd_far_with_cache();

        void solve_fdtd_near_with_cache();

        void precompute_particle_cache();

        // update particle near field (using neighbor particles) + far field (interpolation from neighbor grid cells)
        void update_particle_dirichlet();
};

}  // namespace pppm