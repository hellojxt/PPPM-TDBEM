#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "gui.h"
#include "macro.h"
#include "objIO.h"
#include "pppm.h"
#include "sound_source.h"
#include "visualize.h"
#include "window.h"
#include <filesystem>
#include <fstream>
#include "ghost_cell.h"
#include "pppm.h"
#include "RigidBody.h"
#include "progressbar.h"

using namespace pppm;
std::string obj_name = "spolling_bowl";

void test_PPPM()
{
    std::string OUT_DIR = EXP_DIR + std::string("rigidbody/output/") + obj_name + "/pppm";
    float3 grid_center = make_float3(0.0, 0.08, 0.015);
    float grid_length = 0.005 * 50;
    int res = 40;
    float grid_size = grid_length / res;
    int boundary_size = 5;
    float3 min_pos = grid_center - grid_length / 2 - grid_size * boundary_size;
    res = res + boundary_size * 2;
    float dt = grid_size / (std::sqrt(3) * AIR_WAVE_SPEED * 1.2);
    int frame_rate = 1.0f / dt;
    dt = 1.0f / frame_rate;
    float max_time = 2.5;

    printf("grid size: %e\n", grid_size);
    printf("dt: %e\n", dt);
    printf("frame rate: %d\n", frame_rate);

    PPPMSolver solver(res, grid_size, dt, min_pos);

    float3 reflect_center = make_float3(0, -grid_size, 0);
    int3 reflect_coord = solver.pg.getGridCoord(reflect_center);
    int3 reflect_normal = make_int3(0, 1, 0);
    solver.pg.fdtd.set_reflect_boundary(reflect_coord, reflect_normal);

    RigidBody rigidbody(DATASET_DIR + obj_name, frame_rate, "polystyrene");
    rigidbody.fix_mesh(2e-2, OUT_DIR);
    // rigidbody.export_surface_mesh(OUT_DIR);
    rigidbody.move_to_first_impulse();
    int frame_num = (max_time - rigidbody.current_time) / dt;
    auto IMG_DIR = OUT_DIR + "/img_pppm/";
    CHECK_DIR(IMG_DIR)
    int3 check_coord = make_int3(res - boundary_size);
    CArr<float> resultPPPM(frame_num + 2);
    CArr<float> origin_signal(frame_num + 2);
    resultPPPM.reset();
    origin_signal.reset();
    resultPPPM[0] = frame_rate;
    origin_signal[0] = frame_rate;

    progressbar bar(frame_num);
    while (bar.get_progress() <= frame_num)
    {
        if (bar.get_progress() == 0)
        {
            solver.set_mesh(rigidbody.tetVertices, rigidbody.tetSurfaces);
        }
        rigidbody.audio_step();
        // if (bar.get_progress() < 50800)
        // {
        //     bar.update();
        //     continue;
        // }
        if (rigidbody.mesh_is_updated)
        {
            solver.update_mesh(rigidbody.tetVertices);
        }
        for (int j = 0; j < rigidbody.cpuQ.size(); j++)
        {
            origin_signal[bar.get_progress() + 1] += rigidbody.cpuQ[j];
        }
        solver.update_grid_and_face(rigidbody.surfaceAccs);
        resultPPPM[bar.get_progress() + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
        bar.update();
        // if (bar.get_progress() <= 50826 && bar.get_progress() >= 50820)
        // {
        //     auto sub_img_dir = IMG_DIR + "grid" + std::to_string(bar.get_progress()) + "/";
        //     CHECK_DIR(sub_img_dir)
        //     save_all_grid(solver.pg, sub_img_dir, 100);
        // }
        // if (bar.get_progress() == 50826)
        //     break;
    }
    std::cout << "Done" << std::endl;
    write_to_txt(OUT_DIR + "/result.txt", resultPPPM);
    write_to_txt(OUT_DIR + "/origin.txt", origin_signal);
    rigidbody.clear();
    solver.clear();
}

void test_Ghost()
{
    std::string OUT_DIR = EXP_DIR + std::string("rigidbody/output/") + obj_name + "/ghost";
    float3 grid_center = make_float3(0.0, 0.08, 0.015);
    float grid_length = 0.005 * 50;
    int res = 50;
    float grid_size = grid_length / res;
    int boundary_size = 5;
    float3 min_pos = grid_center - grid_length / 2 - grid_size * boundary_size;
    res = res + boundary_size * 2;
    float dt = grid_size / (std::sqrt(3) * AIR_WAVE_SPEED * 1.2);
    int frame_rate = 1.0f / dt;
    dt = 1.0f / frame_rate;
    float max_time = 2.5;

    printf("grid size: %f\n", grid_size);
    printf("dt: %e\n", dt);
    printf("frame rate: %d\n", frame_rate);

    GhostCellSolver ghost_cell_solver(min_pos, grid_size, res, dt);
    ghost_cell_solver.set_condition_number_threshold(10);
    float3 reflect_center = make_float3(0, -grid_size / 2, 0);
    int3 reflect_coord = ghost_cell_solver.grid.getGridCoord(reflect_center);
    int3 reflect_normal = make_int3(0, 1, 0);
    ghost_cell_solver.grid.fdtd.set_reflect_boundary(reflect_coord, reflect_normal);

    RigidBody rigidbody(DATASET_DIR + obj_name, frame_rate, "polystyrene");
    rigidbody.fix_mesh(2e-2, OUT_DIR);
    rigidbody.move_to_first_impulse();
    int frame_num = (max_time - rigidbody.current_time) / dt;
    auto IMG_DIR = OUT_DIR + "/img/";
    CHECK_DIR(IMG_DIR)
    int3 check_coord = make_int3(res - boundary_size);
    CArr<float> resultGhost(frame_num + 2);
    CArr<float> origin_signal(frame_num + 2);
    resultGhost.reset();
    origin_signal.reset();
    resultGhost[0] = frame_rate;
    origin_signal[0] = frame_rate;

    progressbar bar(frame_num);
    while (bar.get_progress() <= frame_num)
    {
        if (bar.get_progress() == 0)
        {
            ghost_cell_solver.set_mesh(rigidbody.tetVertices, rigidbody.tetSurfaces);
        }
        rigidbody.audio_step();
        if (rigidbody.mesh_is_updated)
        {
            ghost_cell_solver.update_mesh(rigidbody.tetVertices);
        }
        for (int j = 0; j < rigidbody.cpuQ.size(); j++)
        {
            origin_signal[bar.get_progress() + 1] += rigidbody.cpuQ[j];
        }
        // if (bar.get_progress() > 0)
        //     rigidbody.surfaceAccs.reset();
        ghost_cell_solver.update(rigidbody.surfaceAccs);
        resultGhost[bar.get_progress() + 1] =
            ghost_cell_solver.grid.fdtd.grids[ghost_cell_solver.grid.fdtd.t](to_cpu(check_coord));
        bar.update();
        // if (bar.get_progress() < 200)
        // {
        //     save_grid(ghost_cell_solver.grid, IMG_DIR + "grid" + std::to_string(bar.get_progress()) + ".png", 1e-4f);
        // }
        // else
        //     break;
        // if (bar.get_progress() > 5000)
        //     break;
    }
    std::cout << "Done" << std::endl;
    write_to_txt(OUT_DIR + "/result.txt", resultGhost);
    write_to_txt(OUT_DIR + "/origin.txt", origin_signal);
    rigidbody.clear();
    ghost_cell_solver.clear();
}

int main()
{
    CHECK_DIR(EXP_DIR + std::string("rigidbody/output/") + obj_name);
    // test_PPPM();
    test_Ghost();
    // RigidBody rigidbody(DATASET_DIR + obj_name, 44100, "polystyrene");
    // rigidbody.export_signal(EXP_DIR + std::string("rigidbody/output/") + obj_name + "/ghost", 2.5);
    // rigidbody.export_mesh_with_modes(EXP_DIR + std::string("rigidbody/output/") + obj_name + "/ghost");
    return 0;
}
