#pragma once
#include "fdtd.h"
#include "array3D.h"
#include "particle_grid.h"
namespace pppm
{

#define MAX_P_TABLE_SIZE 128
#define MAX_NEIGHBOR_DIST 3
    class PPPMSolver
    {
    public:
        FDTD fdtd;
        ParticleGrid pg;
        GArr3D<float> p_table[MAX_P_TABLE_SIZE];
        int p_table_size;

        PPPMSolver(int res_, float dl_, float dt_);
        void precompute_p_table(int step);
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

    void PPPMSolver::precompute_p_table(int step)
    {
        p_table_size = step;
        for (int i = 0; i < step; i++)
        {
            p_table[i].resize(MAX_NEIGHBOR_DIST, MAX_NEIGHBOR_DIST, MAX_NEIGHBOR_DIST);
        }
        FDTD fdtd_tmp;
        int res_tmp = step * 2 + 1;
        fdtd_tmp.init(res_tmp, fdtd.dl, fdtd.dt);
        CArr3D<float> init_data;
        init_data.resize(res_tmp, res_tmp, res_tmp);
        init_data.reset();
        init_data(step, step, step) = 1;
        fdtd_tmp.grids[0].assign(init_data);
        CArr<float> p_sum;
        p_sum.resize(step);
        p_sum.reset();
        for (int t = 0; t < step; t++)
        {
            auto p_table_cpu = p_table[t].cpu();
            auto grid_data = fdtd_tmp.grids[fdtd_tmp.t % GRID_TIME_SIZE].cpu();
            for (int i = 0; i < MAX_NEIGHBOR_DIST; i++)
            {
                for (int j = 0; j < MAX_NEIGHBOR_DIST; j++)
                {
                    for (int k = 0; k < MAX_NEIGHBOR_DIST; k++)
                    {
                        p_table_cpu(i, j, k) = grid_data(i + step, j + step, k + step);
                        p_sum[t] += p_table_cpu(i, j, k);
                    }
                }
            }
            p_table[t].assign(p_table_cpu);
            fdtd_tmp.update();
            if (t == 1)
            {
                for (int i = 0; i < MAX_NEIGHBOR_DIST; i++)
                    for (int j = 0; j < MAX_NEIGHBOR_DIST; j++)
                        for (int k = 0; k < MAX_NEIGHBOR_DIST; k++)
                            LOG("("<< i << "," << j << "," << k << ") " << p_table_cpu(i, j, k));
                LOG(p_sum[t]);
            }
        }
        LOG("neighbor p sum: \n" << p_sum);
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
        for (int i = 0; i < MAX_P_TABLE_SIZE; i++)
        {
            p_table[i].clear();
        }
    }

}