#include "pppm.h"
#include "pppm_cache.h"
#include "pppm_direct.h"

namespace pppm
{

void PPPMSolver::solve_fdtd_simple()
{
    TICK(correction_fdtd);
    direct_correction_fdtd_near(*this);
    TOCK(correction_fdtd);
    TICK(fdtd);
    fdtd.step();
    TOCK(fdtd);
    TICK(solve_far);
    direct_fdtd_far(*this);
    TOCK(solve_far);
}

void PPPMSolver::update_particle_data() {}

void PPPMSolver::step()
{
    solve_fdtd_simple();
    update_particle_data();
}

}  // namespace pppm