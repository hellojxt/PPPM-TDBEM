#pragma once
#include <cstdio>
#include "array3D.h"
#include "bem.h"
#include "fdtd.h"
#include "particle_grid.h"

namespace pppm
{

class BoundaryHistory
{
    public:
        History dirichlet;
        History neumann;
};

class PPPMSolver
{
    public:
        FDTD fdtd;  // The left corner of the fdtd grid is at (0,0,0)
        ParticleGrid pg;
        GArr<BoundaryHistory> particle_history;  // history boundary data of particles
        GridArr far_field;                       // far field potential of grid cells
        TDBEM bem;                               // boundary element method solver

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

        // set mesh for the particle grid
        void set_mesh(CArr<float3> &verts_, CArr<int3> &tris_)
        {
            pg.set_mesh(verts_, tris_);
            pg.construct_grid();
            particle_history.resize(pg.particles.size());
            particle_history.reset();
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
        }

        // step: solve_fdtd -> update_particle_data -> step
        void step();

        /*
            1. update fdtd near field
            2. solve fdtd
            3. update far field
        */
        void solve_fdtd_simple();

        // update particle near field (using neighbor particles) + far field (interpolation from neighbor grid cells)
        void update_particle_data();
};
}  // namespace pppm