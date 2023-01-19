#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
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
#include "RigidBody.h"
#include "ObjectCollection.h"
#include "simple_json_reader.h"

using namespace pppm;

int main()
{
    std::string configDir = "/home/jiaming/Self/PPPM-github/PPPM-TDBEM/assets/scene.cfg";
    SimpleJsonReader reader(configDir);
    auto &inputDir = reader.dirMap["inputDir"],
         &outputDir = reader.dirMap["outputDir"];
    CHECK_DIR(outputDir);

    std::vector<std::pair<std::string, ObjectInfo::SoundType>> nativeSceneInfo;
    for(auto& info : reader.sceneInfoMap)
    {
        ObjectInfo::SoundType nativeType;
        auto& type = info["type"];
        if(type == "Audio")
            nativeType = ObjectInfo::SoundType::Audio;
        else if(type == "Manual")
            nativeType = ObjectInfo::SoundType::Manual;
        else
            assert(false);
        nativeSceneInfo.emplace_back(std::move(info["name"]), nativeType);
    }

    // the last parameter is useless for collection with only audio and manual.
    ObjectCollection collection(inputDir, nativeSceneInfo, {});
    collection.export_mesh_sequence(outputDir + "/mesh_sequence");
    return 0;
}
