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
    RigidBody rigidbody("/home/jxt/PPPM-TDBEM/dataset/lego", 44100, "polystyrene");
    // rigidbody.export_mesh_with_modes("/home/jxt/PPPM-TDBEM/experiments/rigidbody/output");
    rigidbody.export_signal("/home/jxt/PPPM-TDBEM/experiments/rigidbody/output");
    return 0;
}