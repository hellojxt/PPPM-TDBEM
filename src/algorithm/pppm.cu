#include "pppm.h"
#include "pppm_cache.h"
#include "pppm_direct.h"

namespace pppm
{

void PPPMSolver::solve_fdtd_far_simple(bool log_time)
{
    START_TIME(log_time)
    fdtd.step();
    LOG_TIME("FDTD")
    direct_fdtd_far(*this);
    LOG_TIME("Direct Far Solve")
}

void PPPMSolver::solve_fdtd_near_simple(bool log_time)
{
    START_TIME(log_time)
    direct_correction_fdtd_near(*this);
    LOG_TIME("Direct Near Solve")
}

void PPPMSolver::precompute_grid_cache(bool log_time)
{
    START_TIME(log_time)
    set_grid_cache_size(*this);
    LOG_TIME("Set Grid Cache Size")
    cache_grid_data(*this);
    LOG_TIME("Cache Grid Data")
}

void PPPMSolver::solve_fdtd_far_with_cache(bool log_time)
{
    START_TIME(log_time)
    fdtd.step();
    LOG_TIME("FDTD")
    solve_fdtd_far_field_from_cache(*this);
    LOG_TIME("Far Solve from Cache")
}

void PPPMSolver::solve_fdtd_near_with_cache(bool log_time)
{
    START_TIME(log_time)
    solve_fdtd_near_field_from_cache(*this);
    LOG_TIME("Near Solve from Cache")
}

void PPPMSolver::precompute_particle_cache(bool log_time)
{
    START_TIME(log_time)
    set_particle_cache_size(*this);
    LOG_TIME("Set Particle Cache Size")
    cache_particle_data(*this);
    LOG_TIME("Cache Particle Data")
}

void PPPMSolver::update_particle_dirichlet(bool log_time)
{
    START_TIME(log_time)
    solve_particle_from_cache(*this);
    LOG_TIME("Particle Solve from Cache")
}

__global__ void set_neumann_condition_kernel(GArr<History> neumann, GArr<float> neuuman_condition, int t)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < neumann.size())
    {
        neumann[i][t] = neuuman_condition[i];
    }
}

void PPPMSolver::set_neumann_condition(CArr<float> neuuman_condition)
{
    GArr<float> neuuman_condition_gpu(neuuman_condition);
    cuExecute(neumann.size(), set_neumann_condition_kernel, neumann, neuuman_condition_gpu, fdtd.t);
    neuuman_condition_gpu.clear();
}
}  // namespace pppm