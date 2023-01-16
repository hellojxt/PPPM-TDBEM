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
    std::string OUT_DIR = "/home/jiaming/Self/PPPM-github/render-result/glass-water";
    // RigidBody rigidbody(DATASET_DIR + obj_name, "polystyrene");
    // rigidbody.set_sample_rate(44100);
    // rigidbody.fix_mesh(1e-2, OUT_DIR);
    // rigidbody.export_mesh_with_modes(OUT_DIR + "/correctAnswer");
    // rigidbody.export_signal(OUT_DIR, 2.5);
    // rigidbody.export_mesh_sequence(OUT_DIR + "/mesh_sequence");

    // ObjectCollection collection(DATASET_DIR,
    //                             std::vector<std::pair<std::string, ObjectInfo::SoundType>>{
    //                                 {"bowl", ObjectInfo::SoundType::Modal}, {"plane", ObjectInfo::SoundType::Manual}},
    //                             std::vector<std::any>{std::string{"polystyrene"}, {}});
    // static_cast<RigidBody*>(collection.objects[0].get())->set_sample_rate(44100);

    // ObjectCollection collection("/home/jiaming/Downloads/cup_phone/test",
    //                             std::vector<std::pair<std::string, ObjectInfo::SoundType>>{
    //                                 {"phone", ObjectInfo::SoundType::Audio}, 
    //                                 {"cup", ObjectInfo::SoundType::Manual}
    //                             },
    //                             std::vector<std::any>{
    //                                 {},
    //                                 // std::string{ "polystyrene" },
    //                                 {}
    //                             });

    // ObjectCollection collection("/home/jiaming/Downloads/trumpet/test3",
    //                             std::vector<std::pair<std::string, ObjectInfo::SoundType>>{
    //                                 {"trumpet_horn_speaker", ObjectInfo::SoundType::Audio},
    //                                 {"plunger", ObjectInfo::SoundType::Audio}, // for plunger has motion.
    //                                 {"trumpet_horn", ObjectInfo::SoundType::Manual}},
    //                             {});

    // ObjectCollection collection("/home/jiaming/Downloads/talk_fan/test",
    //                             std::vector<std::pair<std::string, ObjectInfo::SoundType>>{
    //                                 {"head", ObjectInfo::SoundType::Audio},
    //                                 {"blade1", ObjectInfo::SoundType::Audio},
    //                                 {"blade2", ObjectInfo::SoundType::Audio},
    //                                 {"blade3", ObjectInfo::SoundType::Audio},
    //                                 {"fan_other_part", ObjectInfo::SoundType::Manual}},
    //                             {});

    ObjectCollection collection("/home/jiaming/Downloads/倒水demo",
                                std::vector<std::pair<std::string, ObjectInfo::SoundType>>{
                                    {"water", ObjectInfo::SoundType::Audio},
                                    {"glass", ObjectInfo::SoundType::Manual}},
                                {});

    // collection.export_modes(OUT_DIR);
    // collection.objects[0]->UpdateUntil(8.5);
    // collection.UpdateMesh();
    // collection.export_mesh(OUT_DIR + "/mesh_sequence2/surface2.obj");
    collection.export_mesh_sequence(OUT_DIR + "/mesh_sequence");
    return 0;
}
