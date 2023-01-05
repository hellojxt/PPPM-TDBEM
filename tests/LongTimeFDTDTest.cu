#include "case_generator.h"
#include "fdtd.h"
#include "gui.h"
#include "objIO.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"
#include "array_writer.h"

using namespace pppm;

__global__ void set_center_signal(FDTD fdtd, SineSource s)
{
    int3 center = make_int3(fdtd.res / 2 - 10, fdtd.res / 2 + 10, fdtd.res / 2);
    fdtd.grids[fdtd.t](center) = 1000;
}

int main()
{

    int step_num = 50000;
    int res = 64;
    float3 min_pos = make_float3(-0.0886002, -0.0927672, -0.0219455);
    float grid_size = 0.00382634;
    float dt = grid_size / AIR_WAVE_SPEED / std::sqrt(3) / 1.01;
    PPPMSolver solver(res, grid_size, dt);

    SineSource s(5000 * 2 * M_PI);
    CArr<float> result;
    result.resize(step_num);
    for (int i = 0; i < step_num; i++)
    {
        solver.pg.fdtd.step();
        if (i < 10)
            cuExecuteBlock(1, 1, set_center_signal, solver.pg.fdtd, s);
        result[i] = solver.pg.fdtd.grids[i](to_cpu(res - 2, res - 2, res - 2));
    }
    write_to_txt("result.txt", result);
}