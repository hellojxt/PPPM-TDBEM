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

void Test1()
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
    
    // result.reset();
    // solver.pg.fdtd.reset();

    progressbar bar2(step_num, "Chunk version.");
    int chunkSize = 5000;
    int backStepSize = 300;
    for (int i = 0; i < step_num; i++)
    {        
        if(i % chunkSize == chunkSize - backStepSize)
        {
            auto endStep = std::min(i + backStepSize, step_num + 1);
            auto savedT = solver.pg.fdtd.t;
            // update rest of the last chunk.
            for(int j = i; j < endStep; j++)
            {
                solver.pg.fdtd.step();
                UpdateFDTDSignal(solver.pg.fdtd, s, j, step_num);
                result[j] = solver.pg.fdtd.grids[j](to_cpu(res - 2, res - 2, res - 2));
            }
            solver.pg.fdtd.reset();
            solver.neumann.reset();
            solver.dirichlet.reset();
            solver.pg.fdtd.t = savedT;
            for(int j = i; j < endStep; j++)
            {
                solver.pg.fdtd.step();
                UpdateFDTDSignal(solver.pg.fdtd, s, j, step_num);
                bar2.update();
            }
            i = endStep - 1;
            continue;
        }
        solver.pg.fdtd.step();
        UpdateFDTDSignal(solver.pg.fdtd, s, i, step_num);
        result[i] = solver.pg.fdtd.grids[i](to_cpu(res - 2, res - 2, res - 2));
        bar2.update();
    }
    std::cout << "\n";
    write_to_txt("result3.txt", result);
    return;
}

// void UpdateFDTDSignal2(GArr<float> &acceleration, float dt, int step,
//                        float accDt, int& accCnt)
// {
//     float currTime = step * dt;
//     float nextTime = (accCnt + 1) * accDt;
//     if(currTime >= nextTime)
//     {
//         accCnt = static_cast<int>(currTime / accDt) + 1;
//         thrust::fill(thrust::device, acceleration.data(), acceleration.data() + acceleration.size(), 1.0f);
//     }
//     else
//     {
//         thrust::fill(thrust::device, acceleration.data(), acceleration.data() + acceleration.size(), 0.0f);
//     }
// }

void UpdateSolver(PPPMSolver &solver, CArr<Triangle> &triangles,
                  SineSource &sine, MonoPole &mp, int step, int totalStep)
{
    CArr<float> neumann_condition;
    neumann_condition.resize(triangles.size());
    for(int i = 0; i < triangles.size(); i++)
    {
        auto &triangle = triangles[i];
        neumann_condition[i] = (mp.neumann(triangle.center, triangle.normal) *
                                sine(solver.dt() * step)).real();
    }
    solver.set_neumann_condition(neumann_condition);
    solver.update_dirichlet();
    solver.solve_fdtd_near(); 

    return;
}

void Test2(bool needAnswer = false)
{
    int step_num = 5000;
    int res = 64;
    Mesh mesh = Mesh::loadOBJ("/home/jiaming/Self/PPPM-github/PPPM-TDBEM/assets/sphere3.obj");
    BBox bbox = mesh.bbox();

    float3 min_pos = bbox.center() - bbox.length();
    float grid_size = bbox.length() * 2 / res;
    float dt = grid_size / AIR_WAVE_SPEED / std::sqrt(3) / 1.01;
    PPPMSolver solver(res, grid_size, dt, min_pos);

    CArr<float> result;
    result.resize(step_num);
    result.reset();

    // GArr<float> acceleration;
    // acceleration.resize(mesh.triangles.size());

    auto sine = SineSource(2 * PI * 1000);
    float wave_number = sine.omega / AIR_WAVE_SPEED; 
    auto mp = MonoPole(bbox.center(), wave_number);

    auto checkCoord = to_cpu(res - 2, res - 2, res - 2);
    auto checkPoint = 1.0f * (res - 2) / res * bbox.length() / 2 + bbox.center();

    // should be same if sphere has center (0, 0, 0).
    LOG(checkPoint);
    LOG(1.0f * (res - 2) / res * bbox.max);

    solver.set_mesh(mesh.vertices, mesh.triangles);
    CArr<Triangle> triangles = solver.pg.triangles.cpu();
    
    progressbar bar2(step_num, "Chunk version.");
    int chunkSize = 1000;
    int backStepSize = 100;
    
    for (int i = 0; i < step_num; i++)
    {   
        if(i % chunkSize == chunkSize - backStepSize)
        {
            auto endStep = std::min(i + backStepSize, step_num + 1);
            auto savedT = solver.pg.fdtd.t;
            // update rest of the last chunk.
            for(int j = i; j < endStep; j++)
            {
                UpdateSolver(solver, triangles, sine, mp, j, step_num);

                result[j] = solver.pg.fdtd.grids[j](checkCoord);
            }
            solver.pg.fdtd.reset();
            solver.neumann.reset();
            solver.dirichlet.reset();
            solver.pg.fdtd.t = savedT;
            for(int j = i; j < endStep; j++)
            {
                UpdateSolver(solver, triangles, sine, mp, j, step_num);
                bar2.update();
            }
            i = endStep - 1;
            continue;
        }
        UpdateSolver(solver, triangles, sine, mp, i, step_num);
        result[i] = solver.pg.fdtd.grids[i](checkCoord);
        bar2.update();
    }
    std::cout << "\n";
    write_to_txt("result5.txt", result);
    
    if(needAnswer)
    {
        CArr<float> analytical_solution(step_num);
        for (int i = 0; i < step_num; i++)
            analytical_solution[i] = (mp.dirichlet(checkPoint) * sine(dt * i)).real();

        write_to_txt("result6.txt", analytical_solution);
    }
    return;
}

int main()
{
    Test2();
    return 0;
}