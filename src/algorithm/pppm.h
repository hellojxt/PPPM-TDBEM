#pragma once
#include "fdtd.h"
#include "array3D.h"
#include "particle_grid.h"
namespace pppm
{

#define MAX_NEIGHBOR_DIST 3
    class PPPMSolver
    {
    public:
        FDTD fdtd;
        ParticleGrid pg;



        PPPMSolver(int res_, float dl_, float dt_);
        void set_mesh(CArr<float3> &verts_, CArr<int3> &tris_);
        void clear();
    };

    PPPMSolver::PPPMSolver(int res_, float dl_, float dt_)
    {
        fdtd.init(res_, dl_, dt_);
        float3 min_pos = make_float3(-res_ * dl_ / 2, -res_ * dl_ / 2, -res_ * dl_ / 2);
        float3 max_pos = make_float3(res_ * dl_ / 2, res_ * dl_ / 2, res_ * dl_ / 2);
        int3 grid_dim = make_int3(res_, res_, res_);
        pg.init(min_pos, max_pos, grid_dim);
    }

    void PPPMSolver::set_mesh(CArr<float3> &verts_, CArr<int3> &tris_)
    {
        pg.set_mesh(verts_, tris_);
        pg.construct_grid();
    }

    void PPPMSolver::clear()
    {
        fdtd.clear();
        pg.clear();
    }

}