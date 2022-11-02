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
    re.set_params({res / 2, 0, 0}, step_num, 0.02);
    SineSource s(5000 * 2 * M_PI);

    for (int i = 0; i < step_num; i++)
    {
        solver->fdtd.step();
        cuExecuteBlock(1, 1, set_center_signal, solver->fdtd, s);
        re.assign(i, solver->fdtd.grids[i]);
    }
    renderArray(re, false);
}