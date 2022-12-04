#pragma once
#include "array3D.h"
#include "bem.h"
#include "fdtd.h"
#include "macro.h"
#include "pppm.h"

namespace pppm
{

void set_grid_cache_size(PPPMSolver &pppm);
void set_particle_cache_size(PPPMSolver &pppm);
void cache_grid_data(PPPMSolver &pppm);
void cache_grid_data_fast(PPPMSolver &pppm);
void cache_particle_data(PPPMSolver &pppm);
void cache_particle_data_fast(PPPMSolver &pppm);
void solve_fdtd_far_field_from_cache(PPPMSolver &pppm);
void solve_fdtd_near_field_from_cache(PPPMSolver &pppm);
void solve_particle_from_cache(PPPMSolver &pppm);
void solve_particle_from_cache_fast(PPPMSolver &pppm, bool log_time = false);

}  // namespace pppm