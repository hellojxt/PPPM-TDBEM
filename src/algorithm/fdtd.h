#pragma once
#include "array3D.h"

namespace pppm
{

#define GRID_TIME_SIZE 3

    class FDTD
    {
    public:
        GArr3D<float> grids[GRID_TIME_SIZE];
        float c;
        int t;
        int res;
        float dl;
        float dt;

        CGPU_FUNC inline int getGridIndex(int delta = 0)
        {
            return (t + delta + GRID_TIME_SIZE) % GRID_TIME_SIZE;
        }
        void init(int res_, float dl_, float dt_);
        void update();
        void clear();
    };

}