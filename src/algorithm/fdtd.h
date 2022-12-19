#pragma once
#include "array3D.h"

namespace pppm
{

#define GRID_TIME_SIZE 3
typedef float value;
typedef CircularArray<GArr3D<value>, GRID_TIME_SIZE> GridArr;

/**
 * @brief The FDTD class
 * This class implements the finite difference time domain method.
 * It is used to simulate the propagation of sound waves in a 3D space.
 */
class FDTD
{
    public:
        GridArr grids;  // 3D grids with 3 history time steps
        float c;        // speed of sound
        int t;          // current time (index) initialized to -1
        int res;        // resolution of the grid
        int res_bound;  // resolution of the boundary
        float dl;       // grid spacing
        float dt;       // time step

        /**
         *  This function is used to initialize the FDTD grid.
         *  It allocates memory for the grid and sets the initial values to zero.
         *  The left corner of the grid is at (0,0,0).
         *  @param res the resolution of the grid
         *  @param dl the grid spacing
         *  @param dt the time step
         */
        void init(int res_, float dl_, float dt_)
        {
            res = res_;
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                grids[i].resize(res, res, res);
                grids[i].reset();
            }
            t = -1;
            dl = dl_;
            dt = dt_;
            c = AIR_WAVE_SPEED;
        }

        void step_inner_grid();

        void step_boundary_grid();

        /**
         * @brief step forward in time,
         * t++ first, then compute the FDTD kernel
         */
        void step(bool log_time = false)
        {
            START_TIME(log_time)
            step_inner_grid();
            step_boundary_grid();
            t++;
            LOG_TIME("FDTD")
        }

        void clear()
        {
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                grids[i].clear();
            }
        }

        void reset()
        {
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                grids[i].reset();
            }
            t = -1;
        }

        CGPU_FUNC inline float3 getCenter(int i, int j, int k) const
        {
            return make_float3((i + 0.5f) * dl, (j + 0.5f) * dl, (k + 0.5f) * dl);
        }
        CGPU_FUNC inline float3 getCenter(int3 c) const { return getCenter(c.x, c.y, c.z); }
        CGPU_FUNC inline float3 getCenter(uint3 c) const { return getCenter(c.x, c.y, c.z); }
};

}  // namespace pppm