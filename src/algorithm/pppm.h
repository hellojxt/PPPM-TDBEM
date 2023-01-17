#pragma once
#include <cstdio>
#include <fstream>
#include "array3D.h"
#include "particle_grid.h"
#include "pppm_cache.h"

namespace pppm
{

class PPPMSolver
{
    public:
        TDBEM bem;        // boundary element method solver
        ParticleGrid pg;  // The left corner of the fdtd grid is at (0,0,0)
        // pg储存了每个顶点、边、三角面、每个网格包含的面，每个网格用来插值的面，非空网格，3x3x3范围邻居面及其对应非空网格，4x4x4范围邻居面
        GArr<History> dirichlet;      // Dirichlet boundary condition
        GArr<History> neumann;        // Neumann boundary condition
        GArr<float> current_neumann;  // Neumann boundary condition for current
        GridArr grid_far_field;       // far field potential of grid cells 远场是用作什么的？

        // following are used for updating the dirichlet boundary values
        GArr<float> face_far_field;     // far field potential of faces
        GArr2D<float> face_near_field;  // near field potential of faces
        GArr2D<float> face_factor;      // current on faces

        // following are used for accelerating the computation (cache the weights)
        GridCache grid_cache;  // cache for near field computation weights
        FaceCache face_cache;  // cache for near field computation weights

        bool mesh_set = false;

        /**
         *   Constructor of PPPMSolver
         *   @param res_: resolution of the fdtd grid
         *   @param dl_: grid cell size
         *   @param dt_: time step for the FDTD solver
         */
        PPPMSolver(int res_, float dl_, float dt_, float3 min_pos = make_float3(0, 0, 0), int pml_width = 0)
        {
            pg.init(min_pos, dl_, res_, dt_, pml_width);
            bem.init(dt_, dl_);
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                grid_far_field[i].resize(res_, res_, res_);
                grid_far_field[i].reset();
            }
        }

        CGPU_FUNC float inline dt() { return pg.fdtd.dt; }

        CGPU_FUNC float inline dl() { return pg.fdtd.dl; }

        CGPU_FUNC float inline grid_size() { return pg.grid_size; }

        CGPU_FUNC int inline res() { return pg.fdtd.res; }

        CGPU_FUNC float3 inline min_coord() { return pg.min_pos; }

        CGPU_FUNC float3 inline max_coord() { return pg.max_pos; }

        CGPU_FUNC float3 inline center() { return (pg.max_pos + pg.min_pos) / 2; }

        CGPU_FUNC float3 inline size() { return pg.max_pos - pg.min_pos; }

        CGPU_FUNC int inline time_idx() { return pg.time_idx(); }

        // set mesh for the particle grid
        template <typename T1, typename T2>
        void set_mesh(T1 &verts_, T2 &tris_, bool log_time = false)
        {
            START_TIME(log_time)
            pg.set_mesh(verts_, tris_);  // 初始化网格(包括物体节点面元与每个网格与物体面的包含关系)
            LOG_TIME("particle grid: set_mesh")
            pg.construct_neighbor_lists();  // 初始化网格邻居关系
            LOG_TIME("particle grid: construct_neighbor_lists")
            grid_cache.init(tris_.size());
            face_cache.init(tris_.size());
            grid_cache.update_cache(pg, bem, log_time);
            face_cache.update_cache(pg, bem, log_time);
            dirichlet.resize(tris_.size());  // 初始化dirichlet和neumann边界条件
            dirichlet.reset();
            neumann.resize(tris_.size());
            neumann.reset();
            face_far_field.resize(tris_.size());
            face_near_field.resize(tris_.size(), BUFFER_SIZE_NEIGHBOR_NUM_4_4_4);
            face_factor.resize(tris_.size(), BUFFER_SIZE_NEIGHBOR_NUM_4_4_4);
            mesh_set = true;
        }

        template <typename T>
        void update_mesh(T &verts_, bool log_time = false)
        {
            START_TIME(log_time)
            pg.update_mesh(verts_);
            LOG_TIME("particle grid: update_mesh")
            pg.construct_neighbor_lists();
            LOG_TIME("particle grid: construct_neighbor_lists")
            grid_cache.update_cache(pg, bem, log_time);
            face_cache.update_cache(pg, bem, log_time);
            // neumann.reset();
            // dirichlet.reset();
        }

        void clear()
        {
            pg.clear();
            dirichlet.clear();
            neumann.clear();
            current_neumann.clear();
            face_far_field.clear();
            face_near_field.clear();
            face_factor.clear();
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                grid_far_field[i].clear();
            }
            grid_cache.clear();
            face_cache.clear();
        }

        void solve_fdtd_far(bool log_time = false);

        void solve_fdtd_near(bool log_time = false);

        void update_dirichlet(bool log_time = false);

        template <typename T>
        void set_neumann_condition(T neuuman_condition, bool log_time = false);

        template <typename T>
        void update_grid_and_face(T neuuman_condition, bool log_time = false)
        {
            pg.fdtd.step(log_time);
            solve_fdtd_far(log_time);
            set_neumann_condition(neuuman_condition, log_time);
            update_dirichlet(log_time);
            solve_fdtd_near(log_time);
        }

        template <typename T>
        void update_step(T neuuman_condition, bool log_time = false)
        {
            update_grid_and_face(neuuman_condition, log_time);
        }

        GArr3D<float> get_grid() { return pg.fdtd.grids[pg.fdtd.t]; }

        GArr<Triangle> get_triangles() { return pg.triangles; }
        ParticleGrid &get_particle_grid() { return pg; }

        void solve_fdtd_far_simple(bool log_time = false);

        void solve_fdtd_near_simple(bool log_time = false);

        void export_dirichlet(std::string file_name);
};

}  // namespace pppm