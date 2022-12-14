#pragma once
#include "fdtd.h"
#include "particle_grid.h"
#include "sparse.h"
#include "svd.h"

namespace pppm
{
#define GHOST_CELL_NEIGHBOR_NUM 8

enum CellType
{
    AIR,
    SOLID,
    GHOST,
    UNKNOWN
};

enum AccuracyOrder
{
    FIRST_ORDER,
    SECOND_ORDER
};

// implement cout << AccurayOrder
static std::ostream &operator<<(std::ostream &os, const AccuracyOrder &order)
{
    if (order == FIRST_ORDER)
        os << "FIRST_ORDER";
    else if (order == SECOND_ORDER)
        os << "SECOND_ORDER";
    else
        os << "UNKNOWN";
    return os;
}

class CellInfo
{
    public:
        int nearest_particle_idx;
        float3 nearst_point;
        float3 reflect_point;
        float nearst_distance;
        CellType type;
        int ghost_idx;
};

class GhostCellSolver
{
    public:
        GhostCellSolver(float3 min_pos_, float grid_size_, int grid_dim_, float dt)
        {
            grid.init(min_pos_, grid_size_, grid_dim_, dt);
            cell_data.resize(grid_dim_, grid_dim_, grid_dim_);
            set_condition_number_threshold(25.0f);
        };

        void set_mesh(CArr<float3> &vertices_, CArr<int3> &triangles_)
        {
            grid.set_mesh(vertices_, triangles_);
            grid.construct_grid();
        }

        void set_boundary_condition(CArr<float> neuuman_condition) { neuuman_data.assign(neuuman_condition); }

        void set_condition_number_threshold(float threshold) { condition_number_threshold = threshold; }

        void precompute_cell_data(bool log_time = false);

        void precompute_ghost_matrix(bool log_time = false);  // caclulate ghost matrix and p_weight matrix

        void solve_ghost_cell(bool log_time = false);  // update right hand side of ghost cell solver and solve it

        CGPU_FUNC float inline dt() { return grid.fdtd.dt; }

        CGPU_FUNC float inline dl() { return grid.fdtd.dl; }

        CGPU_FUNC float inline grid_size() { return grid.grid_size; }

        CGPU_FUNC int inline res() { return grid.fdtd.res; }

        CGPU_FUNC float3 inline min_coord() { return grid.min_pos; }

        CGPU_FUNC float3 inline max_coord() { return grid.max_pos; }

        CGPU_FUNC float3 inline center() { return (grid.max_pos + grid.min_pos) / 2; }

        CGPU_FUNC float3 inline size() { return grid.max_pos - grid.min_pos; }

        void clear()
        {
            cell_data.clear();
            grid.clear();
            A.clear();
            b.clear();
            p_weight.clear();
            ghost_cells.clear();
            ghost_order.clear();
            neuuman_data.clear();
            linear_solver.clear();
        }

        GArr3D<CellInfo> cell_data;
        ParticleGrid grid;
        COOMatrix A;                      // matrix for ghost cell solver
        GArr<float> b;                    // right hand side of ghost cell solver
        GArr<float> x;                    // solution of ghost cell solver
        GArr2D<float> p_weight;           // weight of 8 neighbor cells
        GArr<int3> ghost_cells;           // list of ghost cells idx
        int ghost_cell_num;               // number of ghost cells
        GArr<AccuracyOrder> ghost_order;  // order of ghost cell
        GArr<float> neuuman_data;         // neuuman data for boundary condition
        BiCGSTAB_Solver linear_solver;
        float condition_number_threshold;  // threshold of condition number
};

}  // namespace pppm
