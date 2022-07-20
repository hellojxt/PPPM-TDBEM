#include "pppm.h"

namespace pppm
{

    PPPMSolver::PPPMSolver(int res_, float dl_, float dt_)
    {
        fdtd.init(res_, dl_, dt_);
    }

    void PPPMSolver::load_data(CArr<float3> &verts_, CArr<int3> &tris_)
    {
        verts.assign(verts_);
        tris.assign(tris_);
    }

    
}