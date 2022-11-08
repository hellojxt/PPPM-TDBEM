#pragma once
#include "fdtd.h"
#include "ghost_cell.h"
#include "particle_grid.h"

namespace pppm
{

void fill_cell_data(ParticleGrid grid, GArr3D<CellInfo> cell_data);

};