#pragma once
#include "array3D.h"

namespace pppm
{

#define GRID_TIME_SIZE 3

    class FDTD
    {
    public:
        GArr3D<cpx> grids[GRID_TIME_SIZE];
        float c;
        int t;
        int res;
        float dl;
        float dt;

        void init(int res_, float dl_, float dt_);
        void update();
        void copy_clip(GArr3D<float> &data, int clip_idx = -1, cpx_phase phase = CPX_REAL);
    };

}