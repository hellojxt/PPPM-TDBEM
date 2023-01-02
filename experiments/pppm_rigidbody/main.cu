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
    std::string OUT_DIR = data_dir + "/pppm";
    CHECK_DIR(OUT_DIR);

    BBox bbox;
    bbox.load_from_txt(data_dir + "/bounding_box.txt");
    LOG("bbox: " << bbox.min << " " << bbox.max)

    float3 grid_center = bbox.center();
    float grid_length = bbox.length();
    int res = 20;
    float grid_size = grid_length / res;
    int boundary_size = 3;
    float3 min_pos = grid_center - grid_length / 2 - grid_size * boundary_size;
    res = res + boundary_size * 2;
    float dt = grid_size / (std::sqrt(3) * AIR_WAVE_SPEED * 1.2);
    int frame_rate = 1.0f / dt;
    dt = 1.0f / frame_rate;
    float max_time = 2.5;

    LOG("grid size: " << grid_size)
    LOG("dt: " << dt)
    LOG("frame rate: " << frame_rate)

    PPPMSolver solver(res, grid_size, dt, min_pos);

    RigidBody rigidbody(data_dir, frame_rate, "polystyrene");
    // for debug
    rigidbody.fix_mesh(grid_size, OUT_DIR);
    rigidbody.export_mesh_with_modes(OUT_DIR);
    // rigidbody.export_mesh_sequence(OUT_DIR + "/mesh");
    // rigidbody.export_signal(OUT_DIR, 2.5);
    return 0;

    rigidbody.fix_mesh(grid_size, OUT_DIR);
    rigidbody.move_to_first_impulse();

    int frame_num = (max_time - rigidbody.current_time) / dt;
    auto IMG_DIR = OUT_DIR + "/img/";
    CHECK_DIR(IMG_DIR)
    int3 check_coord = make_int3(res - boundary_size);
    CArr<float> result(frame_num + 2);
    CArr<float> origin_signal(frame_num + 2);
    result.reset();
    origin_signal.reset();
    result[0] = frame_rate;
    origin_signal[0] = frame_rate;

    progressbar bar(frame_num);
    while (bar.get_progress() <= frame_num)
    {
        if (bar.get_progress() == 0)
        {
            solver.set_mesh(rigidbody.tetVertices, rigidbody.tetSurfaces);
        }
        rigidbody.audio_step();
        // if (bar.get_progress() < 20000)
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
        result[bar.get_progress() + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
        bar.update();
        if (bar.get_progress() <= 31005 && bar.get_progress() >= 31000)
        {
            auto sub_img_dir = IMG_DIR + "grid" + std::to_string(bar.get_progress()) + ".png";
            // CHECK_DIR(sub_img_dir)
            save_grid(solver.pg, sub_img_dir, 100);
        }
        // if (bar.get_progress() == 33000)
        //     break;
    }
    std::cout << "Done" << std::endl;
    write_to_txt(OUT_DIR + "/result.txt", result);
    write_to_txt(OUT_DIR + "/origin.txt", origin_signal);
    rigidbody.clear();
    solver.clear();
}
