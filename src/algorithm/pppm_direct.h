#pragma once
#include "macro.h"
#include "pppm.h"

namespace pppm
{

/**
 *  solve the near field potential of dst_grid from the boundary elements in
 *  the neighbor grids of src_center (3x3x3 grids).
 *  @param src_center: the center grid of all the source grids (3x3x3 grids)
 *  @param dst_grid: the destination grid (the center is used)
 *  @param t: time step index
 */
GPU_FUNC inline float near_field(PPPMSolver &pppm, int3 src_center, int3 dst_grid, int t)
{
    float near_field_value = 0;
    float3 dst_point = pppm.fdtd.getCenter(dst_grid);  // use the center of the grid cell as destination point
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids around src_center
            {
                int3 src = src_center + make_int3(dx, dy, dz);
                if ((src.x == src_center.x) && (src.y == src_center.y) && (src.z == src_center.z) ||
                    (src.x == dst_grid.x) && (src.y == dst_grid.y) && (src.z == dst_grid.z))
                    continue;  // skip the center grid and the destination grid
                Range r = pppm.pg.grid_hash_map(src);
                for (int i = r.start; i < r.end; i++)
                {
                    Particle &e = pppm.pg.particles[i];  // source boundary element
                    near_field_value +=
                        pppm.bem.laplace(pppm.pg.vertices.data(), PairInfo(e.indices, dst_point),
                                         pppm.particle_history[i].neumann, pppm.particle_history[i].dirichlet, t);
                }
            }
    return near_field_value;
}

/**
 *  laplacian of the near field potential of dst_grid from the boundary elements in
 *  the neighbor grids of src_center (3x3x3 grids).
 *  ∇^2 p(i,j,k) * h^2 = p(i-1,j,k) + p(i+1,j,k) + p(i,j-1,k) + p(i,j+1,k) + p(i,j,k-1) + p(i,j,k+1) -
 *  6*p(i,j,k)
 *  @param src_center: the center grid of all the source grids (3x3x3 grids)
 *  @param dst_grid: the destination grid (the center is used)
 *  @param t: time step index
 */
GPU_FUNC inline float laplacian_near_field(PPPMSolver &pppm, int3 src_center, int3 dst_grid, int t)
{
    float result = 0;
    result += near_field(pppm, src_center, dst_grid + make_int3(-1, 0, 0), t);
    result += near_field(pppm, src_center, dst_grid + make_int3(1, 0, 0), t);
    result += near_field(pppm, src_center, dst_grid + make_int3(0, -1, 0), t);
    result += near_field(pppm, src_center, dst_grid + make_int3(0, 1, 0), t);
    result += near_field(pppm, src_center, dst_grid + make_int3(0, 0, -1), t);
    result += near_field(pppm, src_center, dst_grid + make_int3(0, 0, 1), t);
    result -= 6 * near_field(pppm, src_center, dst_grid, t);
    return result / (pppm.fdtd.dl * pppm.fdtd.dl);
}

void direct_correction_fdtd_near(PPPMSolver &pppm);

void direct_fdtd_far(PPPMSolver &pppm);

}  // namespace pppm