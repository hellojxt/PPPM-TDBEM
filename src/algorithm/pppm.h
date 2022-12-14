#pragma once
#include <cstdio>
#include <fstream>
#include "array3D.h"
#include "pppm_cache.h"

namespace pppm
{

class PPPMSolver
{
    public:
        TDBEM bem;        // boundary element method solver
        ParticleGrid pg;  // The left corner of the fdtd grid is at (0,0,0)

        GArr<History> dirichlet;      // Dirichlet boundary condition
        GArr<History> neumann;        // Neumann boundary condition
        GArr<float> current_neumann;  // Neumann boundary condition for current
        GridArr grid_far_field;       // far field potential of grid cells

        // following are used for updating the dirichlet boundary values
        GArr<float> face_far_field;     // far field potential of faces
        GArr2D<float> face_near_field;  // near field potential of faces
        GArr2D<float> face_factor;      // current on faces

        // following are used for accelerating the computation (cache the weights)
        GridCache grid_cache;  // cache for near field computation weights
        FaceCache face_cache;  // cache for near field computation weights

        /**
         *   Constructor of PPPMSolver
         *   @param res_: resolution of the fdtd grid
         *   @param dl_: grid cell size
         *   @param dt_: time step for the FDTD solver
         */
        PPPMSolver(int res_, float dl_, float dt_)
        {
            pg.init(make_float3(0, 0, 0), dl_, res_, dt_);
            bem.init(dt_);
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                grid_far_field[i].resize(res_, res_, res_);
                grid_far_field[i].reset();
            }
        }

        CGPU_FUNC float inline dt() { return pg.fdtd.dt; }

        CGPU_FUNC float inline dl() { return pg.fdtd.dl; }

        CGPU_FUNC float inline grid_size() { return pg.grid_size; }

        CGPU_FUNC int inline res() { return pg.fdtd.res; }

        CGPU_FUNC float3 inline min_coord() { return pg.min_pos; }

        CGPU_FUNC float3 inline max_coord() { return pg.max_pos; }

        CGPU_FUNC float3 inline center() { return (pg.max_pos + pg.min_pos) / 2; }

        CGPU_FUNC float3 inline size() { return pg.max_pos - pg.min_pos; }

        CGPU_FUNC int inline time_idx() { return pg.time_idx(); }

        // set mesh for the particle grid
        void set_mesh(CArr<float3> &verts_, CArr<int3> &tris_)
        {
            pg.set_mesh(verts_, tris_);
            pg.construct_neighbor_lists();
            grid_cache.init(tris_.size());
            face_cache.init(tris_.size());
            grid_cache.update_cache(pg, bem);
            face_cache.update_cache(pg, bem);
            dirichlet.resize(tris_.size());
            dirichlet.reset();
            neumann.resize(tris_.size());
            neumann.reset();
            face_far_field.resize(tris_.size());
            face_near_field.resize(tris_.size(), BUFFER_SIZE_NEIGHBOR_NUM_4_4_4);
            face_factor.resize(tris_.size(), BUFFER_SIZE_NEIGHBOR_NUM_4_4_4);
        }

        void update_mesh(CArr<float3> &verts_)
        {
            pg.update_mesh(verts_);
            pg.construct_neighbor_lists();
            grid_cache.update_cache(pg, bem);
            face_cache.update_cache(pg, bem);
        }

        void clear()
        {
            pg.clear();
            dirichlet.clear();
            neumann.clear();
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                grid_far_field[i].clear();
            }
            grid_cache.clear();
            face_cache.clear();
        }

        void solve_fdtd_far(bool log_time = false);

        void solve_fdtd_near(bool log_time = false);

        void update_dirichlet(bool log_time = false);

        void set_neumann_condition(CArr<float> neuuman_condition, bool log_time = false);

        void update_grid_and_face(CArr<float> neuuman_condition, bool log_time = false)
        {
            pg.fdtd.step(log_time);
            set_neumann_condition(neuuman_condition, log_time);
            solve_fdtd_far(log_time);
            update_dirichlet(log_time);
            solve_fdtd_near(log_time);
        }

        void solve_fdtd_far_simple(bool log_time = false);

        void solve_fdtd_near_simple(bool log_time = false);
};

}  // namespace pppm