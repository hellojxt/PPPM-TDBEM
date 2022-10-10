#pragma once
#include "array3D.h"
#include "bem.h"
#include "fdtd.h"
#include "macro.h"
#include "pppm.h"

namespace pppm
{

void set_cache_grid_size(PPPMSolver &pppm);
void cache_grid_data(PPPMSolver &pppm);
void solve_from_cache(PPPMSolver &pppm);

}  // namespace pppm