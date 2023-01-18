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

__device__ inline void print_type(CellType type)
{
    if (type == SOLID)
        printf("SOLID\n");
    else if (type == AIR)
        printf("AIR\n");
    else if (type == UNKNOWN)
        printf("UNKNOWN\n");
    else if (type == GHOST)
        printf("GHOST\n");
    else
        printf("ERROR\n");
}

struct cell_fresh_info
{
        int3 coord;
        bool is_fresh;
        CGPU_FUNC bool is_zero() const { return !is_fresh; }
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
        GhostCellSolver(float3 min_pos_, float grid_size_, int grid_dim_, float dt, int pml_width = 0)
        {
            grid.init(min_pos_, grid_size_, grid_dim_, dt, pml_width);
            cell_data.resize(grid_dim_, grid_dim_, grid_dim_);
            fresh_cell_list.reserve(grid_dim_ * grid_dim_ * grid_dim_);
            set_condition_number_threshold(25.0f);
            linear_solver.cache_stored = false;
            fresh_error.resize(1);
            fresh_error.reset();
        };

        template <typename T1, typename T2>
        void set_mesh(T1 &vertices_, T2 &triangles_, bool log_time = false)
        {
            START_TIME(log_time)
            grid.set_mesh(vertices_, triangles_);
            LOG_TIME("set mesh")
            precompute_cell_data();
            LOG_TIME("precompute cell data")
            precompute_ghost_matrix(log_time);
            LOG_TIME("precompute ghost matrix")
            neuuman_data.resize(triangles_.size());
            neuuman_data_old.resize(triangles_.size());
            neuuman_data.reset();
            neuuman_data_old.reset();
            mesh_set = true;
        }
        template <typename T>
        void update_mesh(T &verts_, bool log_time = false)
        {
            START_TIME(log_time)
            grid.update_mesh(verts_);
            LOG_TIME("update mesh")
            cell_data_old.assign(cell_data);
            precompute_cell_data();
            LOG_TIME("precompute cell data")
            // LOG("value before fresh: " << grid.fdtd.grids[grid.fdtd.t](to_cpu(37, 33, 27)))
            fill_in_fresh_cell(log_time);
            // LOG("value: after fresh: " << grid.fdtd.grids[grid.fdtd.t](to_cpu(37, 33, 27)))
            LOG_TIME("fill in fresh cell")
            precompute_ghost_matrix(log_time);
            LOG_TIME("precompute ghost matrix")
        }

        void fill_in_fresh_cell(bool log_time = false);

        template <typename T>
        void set_boundary_condition(T neuuman_condition)
        {
            neuuman_data_old.assign(neuuman_data);
            neuuman_data.assign(neuuman_condition);
        }

        void set_condition_number_threshold(float threshold) { condition_number_threshold = threshold; }

        void precompute_cell_data(bool log_time = false);

        void precompute_ghost_matrix(bool log_time = false);  // caclulate ghost matrix and p_weight matrix

        void solve_ghost_cell(bool log_time = false);  // update right hand side of ghost cell solver and solve it

        template <typename T>
        void update(T neuuman_condition, bool log = false)
        {
            START_TIME(log)
            // LOG("value before: " << grid.fdtd.grids[grid.fdtd.t](to_cpu(37, 33, 27)))
            grid.fdtd.step();
            // LOG("value: after: " << grid.fdtd.grids[grid.fdtd.t](to_cpu(37, 33, 27)))
            LOG_TIME("fdtd step")
            set_boundary_condition(neuuman_condition);
            solve_ghost_cell();
            set_solid_cell_zero();
            LOG_TIME("solve ghost cell")
        }

        template <typename T>
        void update_step(T neuuman_condition, bool log_time = false)
        {
            update(neuuman_condition, log_time);
        }

        GArr3D<float> get_grid() { return grid.fdtd.grids[grid.fdtd.t]; }

        GArr<Triangle> get_triangles() { return grid.triangles; }

        ParticleGrid &get_particle_grid() { return grid; }

        void set_solid_cell_zero();

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
            fresh_cell_list.clear();
            cell_data.clear();
            cell_data_old.clear();
            grid.clear();
            A.clear();
            b.clear();
            x.clear();
            p_weight.clear();
            ghost_cells.clear();
            ghost_order.clear();
            neuuman_data.clear();
            neuuman_data_old.clear();
            if (condition_number_threshold > 0)
                linear_solver.clear();
        }
        CompactIndexArray<cell_fresh_info> fresh_cell_list;  // list of fresh cells idx
        GArr3D<CellInfo> cell_data;                          // cell data (cell type, nearest particle idx, etc.)
        GArr3D<CellInfo> cell_data_old;                      // cell data before last mesh update
        ParticleGrid grid;
        COOMatrix A;                      // matrix for ghost cell solver
        GArr<float> b;                    // right hand side of ghost cell solver
        GArr<float> x;                    // solution of ghost cell solver
        GArr2D<float> p_weight;           // weight of 8 neighbor cells
        GArr<int3> ghost_cells;           // list of ghost cells idx
        int ghost_cell_num;               // number of ghost cells
        GArr<AccuracyOrder> ghost_order;  // order of ghost cell
        GArr<float> neuuman_data;         // neuuman data for boundary condition
        GArr<float> neuuman_data_old;     // neuuman data before last mesh update
        BiCGSTAB_Solver linear_solver;
        float condition_number_threshold;  // threshold of condition number
        GArr<int> fresh_error;             // for debug
        bool mesh_set = false;
        bool no_ghost_cell = false;
};

}  // namespace pppm
