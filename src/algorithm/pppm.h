#pragma once
#include "fdtd.h"
#include "array3D.h"

namespace pppm{

    class PPPMSolver{
        public:
        FDTD fdtd;
        GArr<float3> verts;
        GArr<int3> tris;

        PPPMSolver(int res_, float dl_, float dt_);
        void load_data(CArr<float3>& verts_, CArr<int3>& tris_);
        

    };












}