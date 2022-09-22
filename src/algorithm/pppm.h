#pragma once
#include "fdtd.h"
#include "array3D.h"
#include "particle_grid.h"
#include "bem.h"

namespace pppm
{

    class BoundaryHistory
    {
        History dirichlet;
        History neumann;
    };

    class PPPMSolver
    {
    public:
        FDTD fdtd; // The left corner of the fdtd grid is at (0,0,0)
        ParticleGrid pg;
        GArr<BoundaryHistory> particle_history; // history boundary data of particles
        GArr3D<float> far_field;  // far field potential of grid cells
        TDBEM bem;              // boundary element method solver

        /**
         *   Constructor of PPPMSolver
         *   @param res_: resolution of the fdtd grid
         *   @param dl_: grid cell size
         *   @param dt_: time step for the FDTD solver
         */
        PPPMSolver(int res_, float dl_, float dt_);
        void set_mesh(CArr<float3> &verts_, CArr<int3> &tris_); // set mesh for the particle grid
        void clear();
        void step();
    };

    PPPMSolver::PPPMSolver(int res_, float dl_, float dt_)
    {
        fdtd.init(res_, dl_, dt_);
        pg.init(make_float3(0, 0, 0), dl_, res_);
        bem.init(dt_);
        far_field.resize(res_, res_, res_);
    }

    void PPPMSolver::set_mesh(CArr<float3> &verts_, CArr<int3> &tris_)
    {
        pg.set_mesh(verts_, tris_);
        pg.construct_grid();
        particle_history.resize(pg.particles.size());
    }

    void PPPMSolver::clear()
    {
        fdtd.clear();
        pg.clear();
        far_field.clear();
        particle_history.clear();
    }

}