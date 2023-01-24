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
            save_grid(grid, dir + "/" + std::to_string(frameNum) + ".png", max_value);
        }
    }
}

void AudioAndManualTest(std::string configDir)
{
    SimpleJsonReader reader(configDir);
    auto &inputDir = reader.dirMap["inputDir"], &outputDir = reader.dirMap["outputDir"];
    CHECK_DIR(outputDir);
    bool is_water = false;
    CArr<float2> water_motion;
    int water_motion_idx = 0;
    std::vector<std::pair<std::string, ObjectInfo::SoundType>> nativeSceneInfo;
    std::vector<std::any> aditionalParams;
    for (auto &info : reader.sceneInfoMap)
    {
        ObjectInfo::SoundType nativeType;
        auto &type = info["type"];
        if (type == "Audio")
        {
            nativeType = ObjectInfo::SoundType::Audio;
            aditionalParams.emplace_back();
        }
        else if (type == "Manual")
        {
            nativeType = ObjectInfo::SoundType::Manual;
            aditionalParams.emplace_back();
        }
        else if (type == "Modal")
        {
            nativeType = ObjectInfo::SoundType::Modal;
            aditionalParams.emplace_back(std::string{"plastic"});
        }
        else
            throw std::runtime_error("Unknown sound type: " + type);
        if (info["name"] == "water")
        {
            is_water = true;
            LOG("water detected");
            std::ifstream f_motion(inputDir + "/water/motion.txt");
            if (!f_motion.good())
            {
                LOG_ERROR("Fail to load water motion file.");
                std::exit(EXIT_FAILURE);
            }
            std::string line;
            while (getline(f_motion, line))
            {
                if (line.empty())
                    continue;
                std::istringstream iss(line);
                float t, val, tmp;
                iss >> t >> tmp >> tmp >> val >> tmp >> tmp >> tmp >> tmp;
                water_motion.pushBack(make_float2(t, val));
                // LOG("water motion: " << t << " " << val)
            }
        }
        nativeSceneInfo.emplace_back(std::move(info["name"]), nativeType);
    }
    // the last parameter is useless for collection with only audio and manual.
    ObjectCollection collection(inputDir, nativeSceneInfo, aditionalParams);

    BBox bbox = collection.GetBBox();
    float3 grid_center = bbox.center();
    float grid_length = bbox.length() * reader.numMap["gridLengthFactor"];
    float reflect_coeff = reader.numMap["reflectCoeff"];
    int res = reader.numMap["res"];
    float grid_size = grid_length / res;
    float3 min_pos = grid_center - grid_length / 2;
    int frame_rate = 1.01f / (grid_size / std::sqrt(3) / AIR_WAVE_SPEED);

    for (int obj_id = 0; obj_id < collection.objectInfos.size(); obj_id++)
    {
        if (collection.objectInfos[obj_id].type == ObjectInfo::SoundType::Modal)
        {
            RigidBody *rb = static_cast<RigidBody *>(collection.objects[obj_id].get());
            rb->set_sample_rate(frame_rate);
        }
    }
    collection.UpdateTimeStep();
    collection.FixMesh(grid_size);
    float currTime = 0;
    for (int obj_id = 0; obj_id < collection.objectInfos.size(); obj_id++)
    {
        if (collection.objectInfos[obj_id].type == ObjectInfo::SoundType::Modal)
        {
            RigidBody *rb = static_cast<RigidBody *>(collection.objects[obj_id].get());
            rb->move_to_first_impulse();
            currTime = std::max(currTime, rb->current_time);
        }
    }
    const int mute_frame_num = currTime * frame_rate;
    collection.UpdateMesh();

    float dt = 1.0f / frame_rate;
    currTime += reader.numMap["startTime"];
    float max_time = reader.numMap["maxTime"] + currTime;
    LOG("res: " << res);
    LOG("max time: " << max_time);
    LOG("grid length factor: " << (grid_length / bbox.length()));

    LOG("bbox: " << bbox)
    LOG("min pos: " << min_pos);
    LOG("max pos: " << min_pos + grid_size * res)
    LOG("grid size: " << grid_size)
    LOG("dt: " << dt)
    LOG("frame rate: " << frame_rate)

    PPPMSolver solver(res, grid_size, dt, min_pos, 0, reflect_coeff);

    int frame_num = (max_time - currTime) / dt;
    auto IMG_DIR = outputDir + "/img/";
    CHECK_DIR(IMG_DIR)

    int3 check_coord = make_int3(reader.numMap["checkCoordX"] * res, reader.numMap["checkCoordY"] * res,
                                 reader.numMap["checkCoordZ"] * res);

    LOG("check_coord: " << check_coord.x << " " << check_coord.y << " " << check_coord.z);

    CArr<float> result(mute_frame_num + frame_num + 2);
    result.reset();
    result[0] = frame_rate;
    progressbar bar(frame_num);

    int chunkSize = reader.numMap["chunkSize"];
    int backStepSize = reader.numMap["backStepSize"];

    auto CheckNaN = [&]() {
        if (isnan(result[mute_frame_num + bar.get_progress() + 1]))
        {
            LOG("NAN")
            return false;
        }
        return true;
    };

    auto UpdateSound = [&](bool mute = false) {
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
        if (mute)
        {
            collection.surfaceAccs.reset();
            solver.update_grid_and_face(collection.surfaceAccs);
        }
        else
            solver.update_grid_and_face(collection.surfaceAccs);
        if (is_water)
        {
            while (water_motion_idx < water_motion.size() && water_motion[water_motion_idx].x < currTime)
            {
                water_motion_idx++;
            }
            solver.set_water_mute(water_motion[water_motion_idx].y);
        }
        // SaveGridIf([](int frame_num) { return frame_num % 1 == 0; }, true, IMG_DIR, bar.get_progress(), solver.pg,
        //            1e-1);
        return;
    };

    bool success = false;
    LOG("start simulation in " << frame_num << " frames")
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
                UpdateSound(true);
                float factor = (float)(j - i) / (float)(endStep - i);
                if (reader.numMap["debug"] == 1 && j % 50 == 0)
                    save_grid(solver.pg, IMG_DIR + "/" + std::to_string(j) + ".png", reader.numMap["debug_max_value"],
                              make_float3(reader.numMap["debug_x_idx"], reader.numMap["debug_y_idx"],
                                          reader.numMap["debug_z_idx"]));
                // save_all_grid(solver.pg, IMG_DIR + "/" + std::to_string(j) + "/", reader.numMap["debug_max_value"]);
                result[mute_frame_num + j + 1] =
                    solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord)) * (1 - factor);
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
                result[mute_frame_num + j + 1] += solver.pg.fdtd.grids[solver.pg.fdtd.t](to_cpu(check_coord));
                success = CheckNaN();
                if (!success)
                    break;
                bar.update();
            }
            if (!success)
                break;
            // break;
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
