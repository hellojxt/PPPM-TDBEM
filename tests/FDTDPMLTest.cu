#include "case_generator.h"
#include "fdtd.h"
#include "gui.h"
#include "macro.h"
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

    int step_num = 500;
    int res = 65;
    auto solver = empty_pppm(65);

    RenderElement re(solver->pg, "FDTD PML");
    re.set_params({res / 2, 0, 0}, step_num, 0.001);
    SineSource s(5000 * 2 * M_PI);

    FDTD fdtd;
    fdtd.init(solver->res(), solver->dl(), solver->dt(), 5);
    LOG("start simulation")
    for (int i = 0; i < step_num; i++)
    {
        fdtd.step();
        if (i < 20)
        {
            cuExecuteBlock(1, 1, set_center_signal, fdtd, s);
        }
        re.assign(i, fdtd.grids[i]);
    }
    LOG("end simulation")
    renderArray(re);
}