#pragma once
#include "array3D.h"
#include "bem.h"
#include "fdtd.h"
#include "macro.h"
#include "pppm.h"

namespace pppm
{

GPU_FUNC inline void add_grid_near_field(PPPMSolver &pppm, BEMCache &e, int3 dst, int scale = 1)
{
    BElement &particle = pppm.pg.particles[e.particle_id];
    uint3 src_uint = particle.cell_coord;
    int3 src = make_int3(src_uint.x, src_uint.y, src_uint.z);
    if (src.x == dst.x && src.y == dst.y && src.z == dst.z)
        return;
    float3 dst_point = pppm.fdtd.getCenter(dst);  // use the center of the grid cell as destination point
    LayerWeight w;
    pppm.bem.laplace_weight(pppm.pg.vertices.data(), PairInfo(particle.indices, dst_point), &w);
    e.weight.add(w, scale);
}

GPU_FUNC inline void grid_laplacian_near_field(PPPMSolver &pppm, BEMCache &e, int3 dst)
{
    add_grid_near_field(pppm, e, dst + make_int3(-1, 0, 0));
    add_grid_near_field(pppm, e, dst + make_int3(1, 0, 0));
    add_grid_near_field(pppm, e, dst + make_int3(0, -1, 0));
    add_grid_near_field(pppm, e, dst + make_int3(0, 1, 0));
    add_grid_near_field(pppm, e, dst + make_int3(0, 0, -1));
    add_grid_near_field(pppm, e, dst + make_int3(0, 0, 1));
    add_grid_near_field(pppm, e, dst, -6);
    e.weight.divide(pppm.fdtd.dl * pppm.fdtd.dl);
}

void set_cache_grid_size(PPPMSolver &pppm);

}  // namespace pppm