#include "pppm.h"
#include "pppm_cache.h"
#include "pppm_direct.h"

namespace pppm
{

void PPPMSolver::solve_fdtd_far_simple()
{
    TICK(fdtd)
    fdtd.step();
    TOCK(fdtd)
    TICK(solve_far)
    direct_fdtd_far(*this);
    TOCK(solve_far)
}

void PPPMSolver::solve_fdtd_near_simple()
{
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

void PPPMSolver::solve_fdtd_far_with_cache()
{
    TICK(fdtd)
    fdtd.step();
    TOCK(fdtd)
    TICK(solve_far)
    solve_fdtd_far_field_from_cache(*this);
    TOCK(solve_far)
}

void PPPMSolver::solve_fdtd_near_with_cache()
{
    TICK(correction_fdtd)
    solve_fdtd_near_field_from_cache(*this);
    TOCK(correction_fdtd)
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