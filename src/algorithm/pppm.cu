#include "pppm.h"
#include "pppm_cache.h"
#include "pppm_direct.h"

namespace pppm
{

void PPPMSolver::solve_fdtd_far_simple()
{
    fdtd.step();
    direct_fdtd_far(*this);
}

void PPPMSolver::solve_fdtd_near_simple()
{
    direct_correction_fdtd_near(*this);
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
    fdtd.step();
    solve_fdtd_far_field_from_cache(*this);
}

void PPPMSolver::solve_fdtd_near_with_cache()
{
    solve_fdtd_near_field_from_cache(*this);
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
    solve_particle_from_cache(*this);
}

}  // namespace pppm