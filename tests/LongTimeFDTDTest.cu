#include "case_generator.h"
#include "fdtd.h"
#include "gui.h"
#include "objIO.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"
#include "array_writer.h"
#include "progressbar.h"

using namespace pppm;

__global__ void set_center_signal(FDTD fdtd, SineSource s)
{
    int3 center = make_int3(fdtd.res / 2 - 10, fdtd.res / 2 + 10, fdtd.res / 2);
    fdtd.grids[fdtd.t](center) = s((fdtd.t) * fdtd.dt).real();
}

void UpdateFDTDSignal(FDTD& fdtd, SineSource& sineSource, int step, int totalStep)
{
    if (step < totalStep)
        cuExecuteBlock(1, 1, set_center_signal, fdtd, sineSource);
    return;
}

int main()
{
    int step_num = 50000;
    int res = 64;
    float3 min_pos = make_float3(-0.0886002, -0.0927672, -0.0219455);
    float grid_size = 0.002;
    float dt = grid_size / AIR_WAVE_SPEED / std::sqrt(3) / 1.01;
    PPPMSolver solver(res, grid_size, dt);

    SineSource s(5000 * 2 * M_PI);
    CArr<float> result;
    result.resize(step_num);
    result.reset();

    // progressbar bar(step_num, "No chunk version.");
    // for (int i = 0; i < step_num; i++)
    // {        
    //     solver.pg.fdtd.step();
    //     UpdateFDTDSignal(solver.pg.fdtd, s, i, step_num);
    //     result[i] = solver.pg.fdtd.grids[i](to_cpu(res - 2, res - 2, res - 2));
    //     bar.update();
    // }
    // std::cout << "\n";
    // write_to_txt("result.txt", result);
    
    result.reset();
    solver.pg.fdtd.reset();
    progressbar bar2(step_num, "Chunk version.");
    int chunkSize = 5000;
    int backStepSize = 175;
    for (int i = 0; i < step_num; i++)
    {        
        if(i % chunkSize == chunkSize - backStepSize)
        {
            // update rest of the last chunk.
            for(int j = i; j < i + chunkSize && j < step_num; j++)
            {
                solver.pg.fdtd.step();
                result[j] += solver.pg.fdtd.grids[i](to_cpu(res - 2, res - 2, res - 2));
            }
            solver.pg.fdtd.reset();
        }
        solver.pg.fdtd.step();
        UpdateFDTDSignal(solver.pg.fdtd, s, i, step_num);
        result[i] += solver.pg.fdtd.grids[i](to_cpu(res - 2, res - 2, res - 2));
        bar2.update();
    }
    std::cout << "\n";
    write_to_txt("result3.txt", result);
    
    return 0;
}