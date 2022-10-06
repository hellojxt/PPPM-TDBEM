#include "pppm.h"
#include "pppm_cache.h"
#include "pppm_direct.h"

namespace pppm
{

void PPPMSolver::solve_fdtd_simple()
{
    TICK(correction_fdtd)
    direct_correction_fdtd_near(*this);
    TOCK(correction_fdtd)
    TICK(fdtd)
    fdtd.step();
    TOCK(fdtd)
    TICK(solve_far)
    direct_fdtd_far(*this);
    TOCK(solve_far)
}

void PPPMSolver::precompute_grid_cache()
{
    TICK(set_cache_size)
    set_cache_grid_size(*this);
    TOCK(set_cache_size)
    TICK(get_cache_data)
    cache_grid_data(*this);
    TOCK(get_cache_data)
}
void PPPMSolver::solve_fdtd_with_cache()
{
    TICK(solve_from_cache)
    solve_from_cache(*this);
    TOCK(solve_from_cache)
}

void PPPMSolver::precompute_particle_cache() {}
void PPPMSolver::update_particle_dirichlet() {}

}  // namespace pppm