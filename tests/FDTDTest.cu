#include "case_generator.h"
#include "fdtd.h"
#include "gui.h"
#include "objIO.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"

using namespace pppm;

__global__ void set_center_signal(FDTD fdtd, SineSource s)
{
    int3 center = make_int3(fdtd.res / 2, fdtd.res / 2, fdtd.res / 2);
    fdtd.grids[fdtd.t](center) = s((fdtd.t) * fdtd.dt).real();
}

int main()
{

    int step_num = 300;
    int res = 65;
    auto solver = empty_pppm(65);

    RenderElement re(solver->pg, "FDTD Test");
    re.set_params({res / 2, 0, 0}, step_num, 0.005);
    SineSource s(5000 * 2 * M_PI);

    int3 center_coord = make_int3(0, 5, 0);
    int3 normal = make_int3(0, 1, 0);
    solver->pg.fdtd.set_reflect_boundary(center_coord, normal, true);
    for (int i = 0; i < step_num; i++)
    {
        solver->pg.fdtd.step();
        if (i < 20)
            cuExecuteBlock(1, 1, set_center_signal, solver->pg.fdtd, s);
        re.assign(i, solver->pg.fdtd.grids[i]);
    }
    renderArray(re);
}