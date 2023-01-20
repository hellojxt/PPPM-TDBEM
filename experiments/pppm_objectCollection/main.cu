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
#include "simple_json_reader.h"

using namespace pppm;

template <typename T>
void SaveGridIf(T func, bool saveAll, const std::string &dir, int frameNum, ParticleGrid &grid, float max_value = 1.0f)
{
    if (func(frameNum))
    {
        if (saveAll)
        {
            save_all_grid(grid, dir + "/" + std::to_string(frameNum) + "/", max_value);
        }
        else
        {
            save_grid(grid, dir + "/" + std::to_string(frameNum) + ".png", max_value, make_float3(0.5f, 0, 0));
        }
    }
}

void ModalAndManualTest()
{
    std::string data_dir = DATASET_DIR;
    std::string OUT_DIR = "/home/jiaming/Self/PPPM-github/render-result/bowl-plane";
    CHECK_DIR(OUT_DIR);

    ObjectCollection collection(DATASET_DIR,
                                std::vector<std::pair<std::string, ObjectInfo::SoundType>>{
                                    {"bowl", ObjectInfo::SoundType::Modal}, {"plane", ObjectInfo::SoundType::Manual}},
                                std::vector<std::any>{std::string{"polystyrene"}, {}});

    BBox bbox = collection.GetBBox();
    float3 grid_center = bbox.center();
    float grid_length = bbox.length() * 2;
    int res = 40;
    float grid_size = grid_length / res;
    float3 min_pos = grid_center - grid_length / 2;
    int frame_rate = 1.01f / (grid_size / std::sqrt(3) / AIR_WAVE_SPEED);

    RigidBody *bowl = static_cast<RigidBody *>(collection.objects[0].get());
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
    auto CheckNaN = [&]() {
        if (isnan(result[mute_frame_num + bar.get_progress() + 1]))
        {
            LOG("NAN")
            return false;
        }
        return true;
    };

    auto UpdateSound = [&]() {
        bool meshUpdated = false;
        for (auto &object : collection.objects)
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
        // SaveGridIf([](int frame_num) { return frame_num % 10000 == 0; }, true, IMG_DIR, bar.get_progress(),
        // solver.pg,
        //            1e-5);
        return;
    };

    int chunkSize = 10000;
    int backStepSize = 500;
    bool success = false;
    while (bar.get_progress() <= frame_num)
    {
        int i = bar.get_progress();
        if (i % chunkSize == chunkSize - backStepSize)
        {
            auto endStep = std::min(i + backStepSize, frame_num + 1);
            auto savedT = solver.pg.fdtd.t;
            auto savedCurrTime = currTime;
            for (int k = 0; k < collection.objects.size(); k++)
            {
                collection.objects[k]->SaveState(*(collection.objectInfos[k].state));
            }
            // update rest of the last chunk.
            for (int j = i; j < endStep; j++)
            {
                UpdateSound();
                result[mute_frame_num + j + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
                success = CheckNaN();
                if (!success)
                    break;
            }
            if (!success)
                break;

            solver.pg.fdtd.reset();
            solver.neumann.reset();
            solver.dirichlet.reset();
            solver.pg.fdtd.t = savedT;
            currTime = savedCurrTime;
            for (int k = 0; k < collection.objects.size(); k++)
            {
                collection.objects[k]->LoadState(*(collection.objectInfos[k].state));
            }

            for (int j = i; j < endStep; j++)
            {
                UpdateSound();
                success = CheckNaN();
                if (!success)
                    break;
                bar.update();
            }
            if (!success)
                break;

            bar.update();
            continue;
        }

        UpdateSound();
        result[mute_frame_num + i + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
        bar.update();
        if (!CheckNaN())
            break;
    }

    std::cout << "Done" << std::endl;
    write_to_txt(OUT_DIR + "/result.txt", result);
    bowl->clear();
    solver.clear();
    return;
}

void AudioAndManualTest(std::string configDir)
{
    SimpleJsonReader reader(configDir);
    auto &inputDir = reader.dirMap["inputDir"], &outputDir = reader.dirMap["outputDir"];
    CHECK_DIR(outputDir);

    std::vector<std::pair<std::string, ObjectInfo::SoundType>> nativeSceneInfo;
    for (auto &info : reader.sceneInfoMap)
    {
        ObjectInfo::SoundType nativeType;
        auto &type = info["type"];
        if (type == "Audio")
            nativeType = ObjectInfo::SoundType::Audio;
        else if (type == "Manual")
            nativeType = ObjectInfo::SoundType::Manual;
        else
            assert(false);
        nativeSceneInfo.emplace_back(std::move(info["name"]), nativeType);
    }

    // the last parameter is useless for collection with only audio and manual.
    ObjectCollection collection(inputDir, nativeSceneInfo, {});

    BBox bbox = collection.GetBBox();
    float3 grid_center = bbox.center();
    float grid_length = bbox.length() * reader.numMap["gridLengthFactor"];
    int res = reader.numMap["res"];
    float grid_size = grid_length / res;
    float3 min_pos = grid_center - grid_length / 2;
    int frame_rate = 1.01f / (grid_size / std::sqrt(3) / AIR_WAVE_SPEED);

    collection.UpdateTimeStep();
    collection.FixMesh(grid_size);
    collection.UpdateMesh();

    float dt = 1.0f / frame_rate;
    float max_time = reader.numMap["maxTime"];
    LOG("res: " << res);
    LOG("max time: " << max_time);
    LOG("grid length factor: " << (grid_length / bbox.length()));

    LOG("bbox: " << bbox)
    LOG("min pos: " << min_pos);
    LOG("max pos: " << min_pos + grid_size * res)
    LOG("grid size: " << grid_size)
    LOG("dt: " << dt)
    LOG("frame rate: " << frame_rate)

    PPPMSolver solver(res, grid_size, dt, min_pos);

    int frame_num = max_time / dt;
    auto IMG_DIR = outputDir + "/img/";
    CHECK_DIR(IMG_DIR)

    int3 check_coord = make_int3(reader.numMap["checkCoordX"] * res, reader.numMap["checkCoordY"] * res,
                                 reader.numMap["checkCoordZ"] * res);

    LOG("check_coord: " << check_coord.x << " " << check_coord.y << " " << check_coord.z);

    const int mute_frame_num = 0;
    CArr<float> result(mute_frame_num + frame_num + 2);
    result.reset();
    result[0] = frame_rate;

    float currTime = 0.0f;
    progressbar bar(frame_num - currTime / dt);

    int chunkSize = 500;
    int backStepSize = 50;

    auto CheckNaN = [&]() {
        if (isnan(result[mute_frame_num + bar.get_progress() + 1]))
        {
            LOG("NAN")
            return false;
        }
        return true;
    };

    auto UpdateSound = [&]() {
        bool meshUpdated = false;
        for (auto &object : collection.objects)
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
        // SaveGridIf([](int frame_num) { return frame_num % 10 == 0; }, false, IMG_DIR, bar.get_progress(), solver.pg,
        //            1e-2);
        return;
    };

    bool success = false;
    while (bar.get_progress() <= frame_num)
    {
        int i = bar.get_progress();
        if (i % chunkSize == chunkSize - backStepSize)
        {
            auto endStep = std::min(i + backStepSize, frame_num + 1);
            auto savedT = solver.pg.fdtd.t;
            auto savedCurrTime = currTime;
            for (int k = 0; k < collection.objects.size(); k++)
            {
                collection.objects[k]->SaveState(*(collection.objectInfos[k].state));
            }
            // update rest of the last chunk.
            for (int j = i; j < endStep; j++)
            {
                UpdateSound();
                result[mute_frame_num + j + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
                success = CheckNaN();
                if (!success)
                    break;
            }
            if (!success)
                break;

            solver.pg.fdtd.reset();
            solver.neumann.reset();
            solver.dirichlet.reset();
            solver.pg.fdtd.t = savedT;
            currTime = savedCurrTime;
            for (int k = 0; k < collection.objects.size(); k++)
            {
                collection.objects[k]->LoadState(*(collection.objectInfos[k].state));
            }

            for (int j = i; j < endStep; j++)
            {
                UpdateSound();
                success = CheckNaN();
                if (!success)
                    break;
                bar.update();
            }
            if (!success)
                break;

            bar.update();
            continue;
        }

        UpdateSound();
        result[mute_frame_num + i + 1] = solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
        bar.update();
        if (!CheckNaN())
            break;
    }
    std::cout << "Done" << std::endl;
    write_to_txt(outputDir + "/result.txt", result);
    solver.clear();
    return;
}

int main(int argc, char **argv)
{
    AudioAndManualTest(argv[1]);
    // ModalAndManualTest();
    return 0;
}
