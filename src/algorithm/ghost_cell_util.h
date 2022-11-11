#pragma once
#include "fdtd.h"
#include "ghost_cell.h"
#include "particle_grid.h"

namespace pppm
{

// fill cell data and return ghost cell number
int fill_cell_data(ParticleGrid grid, GArr3D<CellInfo> cell_data);

};  // namespace pppm