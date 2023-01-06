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

int main()
{
    std::string data_dir = DATASET_DIR + std::string("/bowl");
    std::string OUT_DIR = data_dir + "/output/ghost";
    CHECK_DIR(OUT_DIR);
    RigidBody rigidbody(data_dir, "polystyrene");
    BBox bbox = rigidbody.get_bbox();
    LOG(bbox)
    float3 grid_center = bbox.center();
    float grid_length = bbox.length() * 2;
    int res = 128;
    float grid_size = grid_length / res;
    float3 min_pos = grid_center - grid_length / 2;
    int frame_rate = 1.01f / (grid_size / std::sqrt(3) / AIR_WAVE_SPEED);
    rigidbody.set_sample_rate(frame_rate);
    float dt = 1.0f / frame_rate;
    float max_time = 2.5;
    LOG("min pos: " << min_pos);
    LOG("grid size: " << grid_size)
    LOG("dt: " << dt)
    LOG("frame rate: " << frame_rate)
    rigidbody.move_to_first_impulse();
    GhostCellSolver ghost_cell_solver(min_pos, grid_size, res, dt);
    ghost_cell_solver.set_condition_number_threshold(0);

    int frame_num = (max_time - rigidbody.current_time) / dt;
    auto IMG_DIR = OUT_DIR + "/img/";
    CHECK_DIR(IMG_DIR)
    int3 check_coord = make_int3(res / 8 * 7);

    int mute_frame_num = int(rigidbody.current_time / dt);
    CArr<float> result(mute_frame_num + frame_num + 2);
    CArr<float> origin_signal(mute_frame_num + frame_num + 2);
    result.reset();
    origin_signal.reset();
    result[0] = frame_rate;
    origin_signal[0] = frame_rate;

    progressbar bar(frame_num);

    while (bar.get_progress() <= frame_num)
    {
        // printf("frame: %d\n", bar.get_progress());
        // if (mute_frame_num + bar.get_progress() < 141220)
        // {
        //     bar.update();
        //     rigidbody.audio_step();
        //     continue;
        // }
        rigidbody.audio_step();
        if (!ghost_cell_solver.mesh_set)
        {
            ghost_cell_solver.set_mesh(rigidbody.tetVertices, rigidbody.tetSurfaces);
        }
        else if (rigidbody.mesh_is_updated)
        {
            ghost_cell_solver.update_mesh(rigidbody.tetVertices);
        }
        for (int j = 0; j < rigidbody.cpuQ.size(); j++)
        {
            origin_signal[mute_frame_num + bar.get_progress() + 1] += rigidbody.cpuQ[j];
        }
        // if (bar.get_progress() > 0)
        // {
        //     // rigidbody.surfaceAccs.reset();
        //     ghost_cell_solver.grid.fdtd.step();
        // }
        //
        if (ghost_cell_solver.fresh_error.cpu()[0] == 1)
        {
            auto sub_filename = IMG_DIR + "grid" + std::to_string(mute_frame_num + bar.get_progress()) + ".png";
            auto sub_img_dir = IMG_DIR + "grid" + std::to_string(mute_frame_num + bar.get_progress()) + "/";
            // save_grid(ghost_cell_solver.grid, sub_filename, 10000.0f);
            save_all_grid(ghost_cell_solver.grid, sub_img_dir, 1e10);
            return 0;
        }
        ghost_cell_solver.update(rigidbody.surfaceAccs);
        result[mute_frame_num + bar.get_progress() + 1] =
            ghost_cell_solver.grid.fdtd.grids[ghost_cell_solver.grid.fdtd.t](to_cpu(check_coord));

        // if (bar.get_progress() <= 20000 && bar.get_progress() % 10 == 0)
        // {
        //     auto sub_filename = IMG_DIR + "grid" + std::to_string(ghost_cell_solver.grid.fdtd.t) + ".png";
        //     auto sub_img_dir = IMG_DIR + "grid" + std::to_string(ghost_cell_solver.grid.fdtd.t) + "/";
        //     auto sub_cell_data_dir = IMG_DIR + "grid" + std::to_string(ghost_cell_solver.grid.fdtd.t) + "_cell/";
        //     save_grid(ghost_cell_solver.grid, sub_filename, 1000.0f);
        //     // save_all_grid(ghost_cell_solver.grid, sub_img_dir, 1e10);
        //     // save_cell_data(ghost_cell_solver, sub_cell_data_dir);
        // }
        // if (bar.get_progress() == 100000)
        // {
        //     break;
        // }

        if (isnan(result[mute_frame_num + bar.get_progress() + 1]))
        {
            LOG("NAN")
            break;
        }
        bar.update();
    }
    std::cout << "Done" << std::endl;
    write_to_txt(OUT_DIR + "/result.txt", result);
    write_to_txt(OUT_DIR + "/origin.txt", origin_signal);
    rigidbody.clear();
    ghost_cell_solver.clear();
}
