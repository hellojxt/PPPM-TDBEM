#pragma once
#include <cstdio>
#include "array3D.h"
#include "bem.h"
#include "fdtd.h"
#include "particle_grid.h"

namespace pppm
{

class GridMap
{
    public:
        int3 coord;
        Range range;  // index range in grid data
        CGPU_FUNC GridMap() {}
        CGPU_FUNC GridMap(int3 coord_, Range range_) : coord(coord_), range(range_) {}
        friend bool operator==(const GridMap &a, const GridMap &b) { return a.coord == b.coord && a.range == b.range; }
};

class ParticleMap
{
    public:
        int3 base_coord;  // lowest coordinate of the 2*2*2 grid cell
        float weight[8];  // weight of the 8 grid cells
        Range range;      // index range in particle data
        CGPU_FUNC ParticleMap() {}
        friend bool operator==(const ParticleMap &a, const ParticleMap &b)
        {
            for (int i = 0; i < 8; i++)
                if (a.weight[i] != b.weight[i])
                    return false;
            return a.base_coord == b.base_coord && a.range == b.range;
        }
};

class BEMCache
{
    public:
        ParticleMap *particle_map;
        int trg_particle_id;
        int3 trg_coord;
        int particle_id;
        LayerWeight weight;
        CGPU_FUNC BEMCache() {}
        friend bool operator==(const BEMCache &a, const BEMCache &b)
        {
            return a.particle_id == b.particle_id && a.weight == b.weight;
        }
};

class PPPMCache
{
    public:
        /* data for cache */
        GArr<GridMap> grid_map;
        GArr<BEMCache> grid_data;  // for solving accurate near field of BEM

        GArr<BEMCache> grid_fdtd_data;   // for solving inaccurate near field of FDTD
        GArr<ParticleMap> particle_map;  // mapping particle id to index range in particle data
                                         // (first index is the data of particle self)
        GArr<BEMCache> particle_data;    // for solving accurate near field of BEM

        /* data for precomputation of cache size */
        // some grid cells have no neighbors
        GArr3D<int> grid_neighbor_nonzero;
        GArr3D<int> grid_neighbor_num;
        // All particles have neighbors
        GArr<int> particle_neighbor_num;

        GArr<float> particle_far_field;   // cache of particle far field
        GArr<float> particle_near_field;  // cache of particle near field
        GArr<float> particle_factor;      // cache of particle factor

        void clear()
        {
            grid_map.clear();
            grid_data.clear();
            grid_fdtd_data.clear();
            particle_map.clear();
            particle_data.clear();
            grid_neighbor_nonzero.clear();
            grid_neighbor_num.clear();
            particle_neighbor_num.clear();
            particle_far_field.clear();
            particle_near_field.clear();
            particle_factor.clear();
        }

        void reset()
        {
            grid_map.reset();
            grid_data.reset();
            grid_fdtd_data.reset();
            particle_map.reset();
            particle_data.reset();
            grid_neighbor_nonzero.reset();
            grid_neighbor_num.reset();
            particle_neighbor_num.reset();
            particle_far_field.reset();
            particle_near_field.reset();
            particle_factor.reset();
        }
};

class PPPMSolver
{
    public:
        FDTD fdtd;  // The left corner of the fdtd grid is at (0,0,0)
        ParticleGrid pg;
        GArr<History> dirichlet;  // Dirichlet boundary condition
        GArr<History> neumann;    // Neumann boundary condition
        GridArr far_field;        // far field potential of grid cells
        TDBEM bem;                // boundary element method solver
        PPPMCache cache;          // cache for near field computation weights

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
            dirichlet.resize(pg.particles.size());
            dirichlet.reset();
            neumann.resize(pg.particles.size());
            neumann.reset();
            cache.particle_factor.resize(pg.particles.size());
            cache.particle_factor.reset();
            cache.particle_far_field.resize(pg.particles.size());
            cache.particle_far_field.reset();
            cache.particle_near_field.resize(pg.particles.size());
            cache.particle_near_field.reset();
        }

        void remove_current_mesh()
        {
            pg.clear();
            dirichlet.clear();
            neumann.clear();
            int t = fdtd.t;
            for (int i = 0; i < GRID_TIME_SIZE; i++)
                far_field[t - i].assign(fdtd.grids[t - i]);
        }

        void clear()
        {
            fdtd.clear();
            pg.clear();
            dirichlet.clear();
            neumann.clear();
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
            dirichlet.reset();
            neumann.reset();
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
        void solve_fdtd_far_simple(bool log_time = false);

        void solve_fdtd_near_simple(bool log_time = false);

        void precompute_grid_cache_simple(bool log_time = false);

        void precompute_grid_cache(bool log_time = false);

        void solve_fdtd_far_with_cache(bool log_time = false);

        void solve_fdtd_near_with_cache(bool log_time = false);

        void precompute_particle_cache_simple(bool log_time = false);

        void precompute_particle_cache(bool log_time = false);

        // update particle near field (using neighbor particles) + far field (interpolation from neighbor grid cells)
        void update_particle_dirichlet(bool log_time = false);

        void update_particle_dirichlet_simple(bool log_time = false);

        void set_neumann_condition(CArr<float> neuuman_condition);
};

}  // namespace pppm