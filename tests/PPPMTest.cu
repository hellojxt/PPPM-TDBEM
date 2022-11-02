#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "case_generator.h"
#include "gui.h"
#include "macro.h"
#include "objIO.h"
#include "pppm.h"
#include "sound_source.h"
#include "window.h"

#define ALL_STEP 128
int main()
{
    using namespace pppm;
    PPPMSolver *solver = empty_pppm(64);
    auto mesh = Mesh::loadOBJ("../assets/sphere3.obj");
    LOG(mesh.vertices.size())
    LOG(mesh.triangles.size())
    mesh.stretch_to(solver->size().x / 4);
    mesh.move_to(solver->center());
}
