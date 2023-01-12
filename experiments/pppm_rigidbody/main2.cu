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
#include "progressbar.h"
#include "ObjectCollection.h"
#include "RigidBody.h"

using namespace pppm;

void ModalAndManualTest()
{
    std::string data_dir = DATASET_DIR;
    std::string OUT_DIR = data_dir + "total/output/pppm4";
    CHECK_DIR(OUT_DIR);

    ObjectCollection collection(DATASET_DIR, 
        std::vector<std::pair<std::string, ObjectInfo::SoundType>>{ 
            {"bowl", ObjectInfo::SoundType::Modal},
            {"plane", ObjectInfo::SoundType::Manual}
        }, std::vector<std::any>{
            std::string{ "polystyrene" }, 
            {}
    });

    BBox bbox = collection.GetBBox();
    float3 grid_center = bbox.center();
    float grid_length = bbox.length() * 2;
    int res = 40;
    float grid_size = grid_length / res;
    float3 min_pos = grid_center - grid_length / 2;
    int frame_rate = 1.01f / (grid_size / std::sqrt(3) / AIR_WAVE_SPEED);

    RigidBody* bowl = static_cast<RigidBody*>(collection.objects[0].get());
    bowl->set_sample_rate(frame_rate);

    collection.UpdateTimeStep();
    collection.FixMesh(grid_size);

    bowl->move_to_first_impulse();
    collection.UpdateMesh();

    float dt = 1.0f / frame_rate;
    float max_time = 2.5f;
    LOG("min pos: " << min_pos);
    LOG("grid size: " << grid_size)
    LOG("dt: " << dt)
    LOG("frame rate: " << frame_rate)

    PPPMSolver solver(res, grid_size, dt, min_pos);

    int frame_num = (max_time - bowl->current_time) / dt;
    auto IMG_DIR = OUT_DIR + "/img/";
    CHECK_DIR(IMG_DIR);
    int3 check_coord = make_int3(res / 8 * 7);

    int mute_frame_num = int(bowl->current_time / dt);
    CArr<float> result(mute_frame_num + frame_num + 2);
    result.reset();
    result[0] = frame_rate;

    auto currTime = bowl->current_time;
    auto timeStep = collection.timeStep;
    progressbar bar(frame_num);
    while (bar.get_progress() <= frame_num)
    {
        bool meshUpdated = false;
        for(auto& object : collection.objects)
        {
            meshUpdated = meshUpdated || object->UpdateUntil(currTime + timeStep);
        }
        currTime += timeStep;
        collection.UpdateMesh();
        collection.UpdateAcc();

        if (!solver.mesh_set)
        {
            solver.set_mesh(collection.tetVertices, collection.tetSurfaces);
        }
        else if (meshUpdated)
        {
            solver.update_mesh(collection.tetVertices);
        }
        solver.update_grid_and_face(collection.surfaceAccs);
        result[mute_frame_num + bar.get_progress() + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
        if (isnan(result[mute_frame_num + bar.get_progress() + 1]))
        {
            LOG("NAN")
            break;
        }
        bar.update();
    }
    std::cout << "Done" << std::endl;
    write_to_txt(OUT_DIR + "/result.txt", result);
    bowl->clear();
    solver.clear();
    return;
}

void AudioAndManualTest()
{
    std::string data_dir = DATASET_DIR "test";
    std::string OUT_DIR = DATASET_DIR "phone/output/pppm";
    CHECK_DIR(OUT_DIR);

    ObjectCollection collection(data_dir, 
        std::vector<std::pair<std::string, ObjectInfo::SoundType>>{ 
            {"phone", ObjectInfo::SoundType::Audio},
            // {"cup", ObjectInfo::SoundType::Manual}
        }, std::vector<std::any>{
            {},
            // std::string{ "polystyrene" }, 
            // {}
    });

    BBox bbox = collection.GetBBox();
    float3 grid_center = bbox.center();
    float grid_length = bbox.length() * 2;
    int res = 40;
    float grid_size = grid_length / res;
    float3 min_pos = grid_center - grid_length / 2;
    int frame_rate = 1.01f / (grid_size / std::sqrt(3) / AIR_WAVE_SPEED);

    collection.UpdateTimeStep();
    collection.FixMesh(grid_size);
    collection.UpdateMesh();

    float dt = 1.0f / frame_rate;
    float max_time = 2.5f;
    LOG("min pos: " << min_pos);
    LOG("grid size: " << grid_size)
    LOG("dt: " << dt)
    LOG("frame rate: " << frame_rate)

    PPPMSolver solver(res, grid_size, dt, min_pos);

    int frame_num = max_time/ dt;
    int3 check_coord = make_int3(res / 8 * 7);

    const int mute_frame_num = 0;
    CArr<float> result(mute_frame_num + frame_num + 2);
    result.reset();
    result[0] = frame_rate;

    float currTime = 0.0f;
    progressbar bar(frame_num);
    while (bar.get_progress() <= frame_num)
    {
        bool meshUpdated = false;
        for(auto& object : collection.objects)
        {
            meshUpdated = meshUpdated || object->UpdateUntil(currTime + dt);
        }
        currTime += dt;
        collection.UpdateMesh();
        collection.UpdateAcc();

        if (!solver.mesh_set)
        {
            solver.set_mesh(collection.tetVertices, collection.tetSurfaces);
        }
        else if (meshUpdated)
        {
            solver.update_mesh(collection.tetVertices);
        }
        solver.update_grid_and_face(collection.surfaceAccs);
        result[mute_frame_num + bar.get_progress() + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
        if (isnan(result[mute_frame_num + bar.get_progress() + 1]))
        {
            LOG("NAN")
            break;
        }
        bar.update();
    }
    std::cout << "Done" << std::endl;
    write_to_txt(OUT_DIR + "/result.txt", result);
    solver.clear();
    return;   
}

int main()
{
    AudioAndManualTest();
    return 0;
}
