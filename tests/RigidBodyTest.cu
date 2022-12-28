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

using namespace pppm;

int main()
{
    std::string obj_name = "spolling_bowl";
    std::string OUT_DIR = EXP_DIR + std::string("rigidbody/output/") + obj_name;
    RigidBody rigidbody(DATASET_DIR + obj_name, 44100, "polystyrene");
    rigidbody.fix_mesh(3e-2, OUT_DIR);
    rigidbody.export_mesh_with_modes(OUT_DIR);
    rigidbody.export_signal(OUT_DIR);
    // rigidbody.export_mesh_sequence(OUT_DIR + "/mesh_sequence");
    return 0;
}
