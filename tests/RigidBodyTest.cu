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

using namespace pppm;

int main()
{
    std::string obj_name = "bowl";
    std::string OUT_DIR = DATASET_DIR + "total" + std::string("/output/pppm");
    // RigidBody rigidbody(DATASET_DIR + obj_name, "polystyrene");
    // rigidbody.set_sample_rate(44100);
    // rigidbody.fix_mesh(1e-2, OUT_DIR);
    // rigidbody.export_mesh_with_modes(OUT_DIR + "/correctAnswer");
    // rigidbody.export_signal(OUT_DIR, 2.5);
    // rigidbody.export_mesh_sequence(OUT_DIR + "/mesh_sequence");

    ObjectCollection collection(DATASET_DIR, 
        std::vector<std::pair<std::string, ObjectInfo::SoundType>>{ 
            {obj_name, ObjectInfo::SoundType::Modal},
            {"plane", ObjectInfo::SoundType::Manual}
        }, std::vector<std::any>{
            { std::string{"polystyrene"}, {} }
        });
    collection.export_modes(OUT_DIR);
    // collection.export_mesh_sequence(OUT_DIR + "/mesh_sequence");
    return 0;
}
