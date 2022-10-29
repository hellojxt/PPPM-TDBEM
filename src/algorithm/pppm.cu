#include "pppm.h"
#include "pppm_cache.h"
#include "pppm_direct.h"

namespace pppm
{

void PPPMSolver::solve_fdtd_simple()
{
    TICK(fdtd)
    fdtd.step();
    TOCK(fdtd)
    TICK(solve_far)
    direct_fdtd_far(*this);
    TOCK(solve_far)
    TICK(correction_fdtd)
    direct_correction_fdtd_near(*this);
    TOCK(correction_fdtd)
}

void PPPMSolver::precompute_grid_cache()
{
    TICK(set_grid_cache_size)
    set_grid_cache_size(*this);
    TOCK(set_grid_cache_size)
    TICK(get_grid_cache_data)
    cache_grid_data(*this);
    TOCK(get_grid_cache_data)
}
void PPPMSolver::solve_fdtd_with_cache()
{
    TICK(fdtd)
    fdtd.step();
    TOCK(fdtd)
    TICK(solve_grid_from_cache)
    solve_grid_from_cache(*this);
    TOCK(solve_grid_from_cache)
}

void PPPMSolver::precompute_particle_cache()
{
    TICK(set_particle_cache_size)
    set_particle_cache_size(*this);
    TOCK(set_particle_cache_size)
    TICK(get_particle_cache_data)
    cache_particle_data(*this);
    TOCK(get_particle_cache_data)
}

void PPPMSolver::update_particle_dirichlet()
{
    TICK(update_particle_dirichlet)
    solve_particle_from_cache(*this);
    TOCK(update_particle_dirichlet)
}

}  // namespace pppm