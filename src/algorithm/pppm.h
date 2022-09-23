#pragma once
#include "array3D.h"
#include "bem.h"
#include "fdtd.h"
#include "particle_grid.h"

namespace pppm
{

class BoundaryHistory
{
    public:
        History dirichlet;
        History neumann;
};

class PPPMSolver
{
    public:
        FDTD fdtd;  // The left corner of the fdtd grid is at (0,0,0)
        ParticleGrid pg;
        GArr<BoundaryHistory> particle_history;  // history boundary data of particles
        GridArr far_field;                       // far field potential of grid cells
        TDBEM bem;                               // boundary element method solver

        /**
         *   Constructor of PPPMSolver
         *   @param res_: resolution of the fdtd grid
         *   @param dl_: grid cell size
         *   @param dt_: time step for the FDTD solver
         */
        PPPMSolver(int res_, float dl_, float dt_)
        {
            fdtd.init(res_, dl_, dt_);
            pg.init(make_float3(0, 0, 0), dl_, res_);
            bem.init(dt_);
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                far_field[i].resize(res_, res_, res_);
                far_field[i].reset();
            }
        }

        // set mesh for the particle grid
        void set_mesh(CArr<float3> &verts_, CArr<int3> &tris_)
        {
            pg.set_mesh(verts_, tris_);
            pg.construct_grid();
            particle_history.resize(pg.particles.size());
            particle_history.reset();
        }

        void clear()
        {
            fdtd.clear();
            pg.clear();
            particle_history.clear();
            for (int i = 0; i < GRID_TIME_SIZE; i++)
            {
                far_field[i].clear();
            }
        }

        /**
         *  solve the near field potential of dst_grid from the boundary elements in
         *  the neighbor grids of src_center (3x3x3 grids).
         *  @param src_center: the center grid of all the source grids (3x3x3 grids)
         *  @param dst_grid: the destination grid (the center is used)
         *  @param t: time step index
         */
        GPU_FUNC inline float near_field(int3 src_center, int3 dst_grid, int t)
        {
            float near_field_value = 0;
            float3 dst_point = fdtd.getCenter(dst_grid);  // use the center of the grid cell as destination point
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids around src_center
                    {
                        int3 src = src_center + make_int3(dx, dy, dz);
                        Range r = pg.grid_hash_map(src);
                        for (int i = r.start; i < r.end; i++)
                        {
                            BElement &e = pg.particles[i];  // source boundary element
                            near_field_value +=
                                bem.laplace(pg.vertices.data(), PairInfo(e.indices, dst_point),
                                            particle_history[i].neumann, particle_history[i].dirichlet, t);
                        }
                    }
            return near_field_value;
        }

        /**
         *  laplacian of the near field potential of dst_grid from the boundary elements in
         *  the neighbor grids of src_center (3x3x3 grids).
         *  ∇^2 p(i,j,k) * h^2 = p(i-1,j,k) + p(i+1,j,k) + p(i,j-1,k) + p(i,j+1,k) + p(i,j,k-1) + p(i,j,k+1) -
         * 6*p(i,j,k)
         *  @param src_center: the center grid of all the source grids (3x3x3 grids)
         *  @param dst_grid: the destination grid (the center is used)
         *  @param t: time step index
         */
        GPU_FUNC inline float laplacian_near_field(int3 src_center, int3 dst_grid, int t)
        {
            float result = 0;
            result += near_field(src_center, dst_grid + make_int3(-1, 0, 0), t);
            result += near_field(src_center, dst_grid + make_int3(1, 0, 0), t);
            result += near_field(src_center, dst_grid + make_int3(0, -1, 0), t);
            result += near_field(src_center, dst_grid + make_int3(0, 1, 0), t);
            result += near_field(src_center, dst_grid + make_int3(0, 0, -1), t);
            result += near_field(src_center, dst_grid + make_int3(0, 0, 1), t);
            result -= 6 * near_field(src_center, dst_grid, t);
            return result / (fdtd.dl * fdtd.dl);
        }

        // step: solve_fdtd -> update_particle_data -> step
        void step();

        /*
            1. update fdtd near field
            2. solve fdtd
            3. update far field
        */
        void solve_fdtd();

        // update particle near field (using neighbor particles) + far field (interpolation from neighbor grid cells)
        void update_particle_data();
};
}  // namespace pppm