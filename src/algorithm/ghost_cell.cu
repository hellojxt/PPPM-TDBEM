#include "ghost_cell_util.h"
namespace pppm
{

void GhostCellSolver::precompute_cell_data()
{
    fill_cell_data(grid, cell_data);
};

}  // namespace pppm