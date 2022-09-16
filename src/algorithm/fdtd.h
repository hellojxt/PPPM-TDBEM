#pragma once
#include "array3D.h"

namespace pppm
{

#define GRID_TIME_SIZE 3

    class FDTD
    {
    public:
        GArr3D<float> grids[GRID_TIME_SIZE]; // 3D grids with 3 history time steps
        float c;                             // speed of sound
        int t;                               // current time (index)
        int res;                             // resolution of the grid
        float dl;                            // grid spacing
        float dt;                            // time step

        /**
         *  Get the index of the grid at time t
         *  @param delta the time index difference from the current time
         */
        CGPU_FUNC inline int getGridIndex(int delta = 0)
        {
            return (t + delta + GRID_TIME_SIZE) % GRID_TIME_SIZE;
        }

        /**
         *  This function is used to initialize the FDTD grid.
         *  It allocates memory for the grid and sets the initial values to zero.
         *  @param res the resolution of the grid
         *  @param dl the grid spacing
         *  @param dt the time step
         */
        void init(int res_, float dl_, float dt_);

        /**
         * @brief step forward in time
         */
        void step();
        void clear();
    };

}