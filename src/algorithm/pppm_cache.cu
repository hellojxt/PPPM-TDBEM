#include "gauss/triangle.h"
#include "macro.h"
#include "pppm_cache.h"

namespace pppm
{
__global__ void get_involved_grid(PPPMSolver pppm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int res = pppm.fdtd.res;
    if (x >= res - 1 || x < 1 || y >= res - 1 || y < 1 || z >= res - 1 || z < 1)
        return;
    int neighbor_particle_num = 0;
    int3 center = make_int3(x, y, z);
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids
            {
                int3 coord = make_int3(x + dx, y + dy, z + dz);
                Range r = pppm.pg.grid_hash_map(coord);
                neighbor_particle_num += r.length();
            }
    pppm.cache.grid_neighbor_num(center) = neighbor_particle_num;
    pppm.cache.grid_neighbor_nonzero(center) = neighbor_particle_num > 0;
}

__global__ void fill_grid_info(PPPMSolver pppm)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int res = pppm.fdtd.res;
    if (x >= res - 1 || x < 1 || y >= res - 1 || y < 1 || z >= res - 1 || z < 1)
        return;
    PPPMCache &cache = pppm.cache;
    int3 center = make_int3(x, y, z);
    int range_end = cache.grid_neighbor_num(center);
    int range_start = range_end;

    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids
            {
                int3 coord = make_int3(x + dx, y + dy, z + dz);
                Range r = pppm.pg.grid_hash_map(coord);
                for (int i = r.start; i < r.end; i++)
                {
                    range_start--;
                    cache.grid_data[range_start].particle_id = i;       // fill particle id
                    cache.grid_fdtd_data[range_start].particle_id = i;  // fill particle id
                }
            }
    if (range_start != range_end)
    {
        int grid_index = cache.grid_neighbor_nonzero(center) - 1;
        cache.grid_map[grid_index] = GridMap(center, Range(range_start, range_end));
    }
}

void set_grid_cache_size(PPPMSolver &pppm)
{
    int res = pppm.fdtd.res;
    PPPMCache &cache = pppm.cache;
    cache.grid_neighbor_nonzero.resize(res, res, res);
    cache.grid_neighbor_nonzero.reset();
    cache.grid_neighbor_num.resize(res, res, res);
    cache.grid_neighbor_num.reset();
    cuExecute3D(dim3(res, res, res), get_involved_grid, pppm);
    thrust::inclusive_scan(thrust::device, cache.grid_neighbor_num.begin(), cache.grid_neighbor_num.end(),
                           cache.grid_neighbor_num.begin(),
                           thrust::plus<int>());               // calculate the prefix sum of neighbor_nums
    int total_num = cache.grid_neighbor_num.data.last_item();  // last item is the total number of neighbor particles
    thrust::inclusive_scan(thrust::device, cache.grid_neighbor_nonzero.begin(), cache.grid_neighbor_nonzero.end(),
                           cache.grid_neighbor_nonzero.begin(),
                           thrust::plus<int>());  // calculate the prefix sum of non-zero-neighbor particle
    int total_nonzero =
        cache.grid_neighbor_nonzero.data.last_item();  // last item is the total number of particles have neighbors
    cache.grid_map.resize(total_nonzero);
    cache.grid_data.resize(total_num);
    cache.grid_fdtd_data.resize(total_num);
    cuExecute3D(dim3(res, res, res), fill_grid_info, pppm);
}

GPU_FUNC inline void add_grid_near_field(PPPMSolver &pppm, BEMCache &e, int3 dst, float scale, int offset)
{
    Particle &particle = pppm.pg.particles[e.particle_id];
    int3 src = make_int3(particle.cell_coord);
    if (src.x == dst.x && src.y == dst.y && src.z == dst.z)
        return;
    float3 dst_point = pppm.fdtd.getCenter(dst);  // use the center of the grid cell as destination point
    LayerWeight w;
    pppm.bem.laplace_weight(pppm.pg.vertices.data(), PairInfo(particle.indices, dst_point), &w);
    e.weight.add(w, scale, offset);
}

GPU_FUNC inline void add_laplacian_near_field(PPPMSolver &pppm, BEMCache &e, int3 dst, float scale = 1, int offset = 0)
{
    scale = scale / (pppm.fdtd.dl * pppm.fdtd.dl);
    add_grid_near_field(pppm, e, dst + make_int3(-1, 0, 0), scale, offset);
    add_grid_near_field(pppm, e, dst + make_int3(1, 0, 0), scale, offset);
    add_grid_near_field(pppm, e, dst + make_int3(0, -1, 0), scale, offset);
    add_grid_near_field(pppm, e, dst + make_int3(0, 1, 0), scale, offset);
    add_grid_near_field(pppm, e, dst + make_int3(0, 0, -1), scale, offset);
    add_grid_near_field(pppm, e, dst + make_int3(0, 0, 1), scale, offset);
    add_grid_near_field(pppm, e, dst, -6 * scale, offset);
}

__global__ void precompute_grid_data(PPPMSolver pppm)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    PPPMCache &cache = pppm.cache;
    if (grid_id >= cache.grid_map.size())
        return;
    GridMap grid_map = cache.grid_map[grid_id];
    int3 coord = grid_map.coord;
    Range r = grid_map.range;
    float c = pppm.fdtd.c, dt = pppm.fdtd.dt;
    for (int i = r.start; i < r.end; i++)
    {
        BEMCache e;
        e.particle_id = cache.grid_data[i].particle_id;
        e.weight.reset();
        add_grid_near_field(pppm, e, coord, 2, -1);
        add_laplacian_near_field(pppm, e, coord, c * c * dt * dt, -1);
        add_grid_near_field(pppm, e, coord, -1, -2);
        // auto simple = pppm.far_field[0](coord);
        // if ((e.weight.double_layer[1] - simple) / simple > 1e-3)
        // printf("coord: (%d, %d, %d), double: %e, simple: %e\n", coord.x, coord.y, coord.z, e.weight.double_layer[1],
        //        simple);
        pppm.cache.grid_fdtd_data[i] = e;
        e.weight.reset();
        add_grid_near_field(pppm, e, coord, 1, 0);
        pppm.cache.grid_data[i] = e;
    }
}

void cache_grid_data(PPPMSolver &pppm)
{
    int total_num = pppm.cache.grid_data.size();
    cuExecute(total_num, precompute_grid_data, pppm);
}

__global__ void solve_fdtd_far_field_from_cache_kernel(PPPMSolver pppm)
{
    int grid_id = blockIdx.x;
    PPPMCache &cache = pppm.cache;
    if (grid_id >= cache.grid_map.size())
        return;
    GridMap grid_map = cache.grid_map[grid_id];
    int3 coord = grid_map.coord;
    Range r = grid_map.range;
    int t = pppm.fdtd.t;
    __shared__ float fdtd_near_field;
    if (threadIdx.x == 0)
    {
        fdtd_near_field = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < r.end - r.start; i += blockDim.x)
    {
        BEMCache e = cache.grid_fdtd_data[r.start + i];
        BoundaryHistory &history = pppm.particle_history[e.particle_id];
        atomicAdd_block(&fdtd_near_field, e.weight.convolution(history.neumann, history.dirichlet, t));
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float far_field = pppm.fdtd.grids[t](coord) - fdtd_near_field;
        pppm.far_field[t](coord) = far_field;  // far field of empty grid need to be initialized with FDTD solution
    }
}

__global__ void solve_fdtd_near_field_from_cache_kernel(PPPMSolver pppm)
{
    int grid_id = blockIdx.x;
    PPPMCache &cache = pppm.cache;
    if (grid_id >= cache.grid_map.size())
        return;
    GridMap grid_map = cache.grid_map[grid_id];
    int3 coord = grid_map.coord;
    Range r = grid_map.range;
    int t = pppm.fdtd.t;
    __shared__ float accurate_near_field;
    if (threadIdx.x == 0)
    {
        accurate_near_field = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < r.end - r.start; i += blockDim.x)
    {
        BEMCache e = cache.grid_data[r.start + i];
        BoundaryHistory &history = pppm.particle_history[e.particle_id];
        atomicAdd_block(&accurate_near_field, e.weight.convolution(history.neumann, history.dirichlet, t));
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        pppm.fdtd.grids[t](coord) = pppm.far_field[t](coord) + accurate_near_field;
    }
}

void solve_fdtd_far_field_from_cache(PPPMSolver &pppm)
{
    int total_num = pppm.cache.grid_map.size();
    int t = pppm.fdtd.t;
    pppm.far_field[t].assign(pppm.fdtd.grids[t]);
    cuExecuteBlock(total_num, 64, solve_fdtd_far_field_from_cache_kernel, pppm);
}

void solve_fdtd_near_field_from_cache(PPPMSolver &pppm)
{
    int total_num = pppm.cache.grid_map.size();
    cuExecuteBlock(total_num, 64, solve_fdtd_near_field_from_cache_kernel, pppm);
}

// https://en.wikipedia.org/wiki/Trilinear_interpolation
// xyz      xyz      xyz      xyz      xyz      xyz      xyz      xyz
// 000 = 0, 001 = 1, 010 = 2, 011 = 3, 100 = 4, 101 = 5, 110 = 6, 111 = 7
CGPU_FUNC inline void trilinear_interpolation(float3 d, float *weight)
{
    weight[0] = (1 - d.x) * (1 - d.y) * (1 - d.z);
    weight[1] = (1 - d.x) * (1 - d.y) * d.z;
    weight[2] = (1 - d.x) * d.y * (1 - d.z);
    weight[3] = (1 - d.x) * d.y * d.z;
    weight[4] = d.x * (1 - d.y) * (1 - d.z);
    weight[5] = d.x * (1 - d.y) * d.z;
    weight[6] = d.x * d.y * (1 - d.z);
    weight[7] = d.x * d.y * d.z;
}

CGPU_FUNC inline void weight_add(float *w, float scale, float *out)
{
    for (int i = 0; i < 8; i++)
        out[i] += scale * w[i];
}

__global__ void get_particle_neighbor_sum(PPPMSolver pppm)
{
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= pppm.pg.particles.size())
        return;
    float sum = 0;
    Particle &p = pppm.pg.particles[particle_id];
    // calculate the base coordinate (lowest coord) of the 2*2*2 grids
    float3 diff = p.pos - pppm.fdtd.getCenter(p.cell_coord);
    int3 base_coord = make_int3(p.cell_coord) - make_int3((diff.x < 0), (diff.y < 0), (diff.z < 0));
    ParticleMap pm;
    pm.base_coord = base_coord;

    // calculate the weights of 2*2*2 grid with guass interpolation
    float w_temp[8];
    for (int i = 0; i < 8; i++)
        pm.weight[i] = 0;
    float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
    float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
    float3 dst_v[3] = {
        {pppm.pg.vertices[p.indices.x]}, {pppm.pg.vertices[p.indices.y]}, {pppm.pg.vertices[p.indices.z]}};
    float trg_jacobian = jacobian(dst_v);
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
    {
        float3 v = local_to_global(guass_x[i][0], guass_x[i][1], dst_v);
        float3 d = (v - pppm.fdtd.getCenter(base_coord)) / pppm.pg.grid_size;
        trilinear_interpolation(d, w_temp);
        weight_add(w_temp, 0.5 * guass_w[i] * trg_jacobian, pm.weight);
        // weight_add(w_temp, 1.0f / TRI_GAUSS_NUM, pm.weight);
    }

    pppm.cache.particle_map[particle_id] = pm;
    // calculate the sum of neighbor particles
    for (int dx = -1; dx <= 2; dx++)
        for (int dy = -1; dy <= 2; dy++)
            for (int dz = -1; dz <= 2; dz++)
            {
                int3 coord = base_coord + make_int3(dx, dy, dz);
                Range r = pppm.pg.grid_hash_map(coord);
                sum += r.length();
            }
    pppm.cache.particle_neighbor_num[particle_id] = sum;
}

__global__ void fill_particle_info(PPPMSolver pppm)
{
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= pppm.pg.particles.size())
        return;
    int3 base_coord = pppm.cache.particle_map[particle_id].base_coord;
    int range_end = pppm.cache.particle_neighbor_num[particle_id];
    int range_start = range_end;
    for (int dx = -1; dx <= 2; dx++)
        for (int dy = -1; dy <= 2; dy++)
            for (int dz = -1; dz <= 2; dz++)
            {
                int3 coord = base_coord + make_int3(dx, dy, dz);
                Range r = pppm.pg.grid_hash_map(coord);
                for (int i = r.start; i < r.end; i++)
                {
                    range_start--;
                    pppm.cache.particle_data[range_start].particle_id = i;
                }
            }
    pppm.cache.particle_map[particle_id].range = Range(range_start, range_end);
}

void set_particle_cache_size(PPPMSolver &pppm)
{
    int particle_num = pppm.pg.particles.size();
    PPPMCache &cache = pppm.cache;
    cache.particle_map.resize(pppm.pg.particles.size());
    cache.particle_neighbor_num.resize(particle_num);
    cuExecute(particle_num, get_particle_neighbor_sum, pppm);
    thrust::inclusive_scan(thrust::device, cache.particle_neighbor_num.begin(), cache.particle_neighbor_num.end(),
                           cache.particle_neighbor_num.begin(),
                           thrust::plus<int>());  // calculate the prefix sum of neighbor_nums
    int total_num = cache.particle_neighbor_num.last_item();
    cache.particle_data.resize(total_num);
    cuExecute(particle_num, fill_particle_info, pppm);
}

template <typename T>
GPU_FUNC inline void add_particle_near_field(PPPMSolver &pppm,
                                             LayerWeight &layer_weight,
                                             int3 src,
                                             T dst,
                                             float scale,
                                             int offset)
{
    LayerWeight w;
    pppm.bem.laplace_weight(pppm.pg.vertices.data(), PairInfo(src, dst), &w);
    layer_weight.add(w, scale, offset);
}

__global__ void precompute_cache_data(PPPMSolver pppm)
{
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= pppm.pg.particles.size())
        return;
    Particle &current_particle = pppm.pg.particles[particle_id];
    ParticleMap &pm = pppm.cache.particle_map[particle_id];
    Range r = pm.range;
    for (int i = r.start; i < r.end; i++)
    {
        LayerWeight w;
        w.reset();
        auto neighbor_particle_id = pppm.cache.particle_data[i].particle_id;
        Particle &neighbor_particle = pppm.pg.particles[neighbor_particle_id];
        add_particle_near_field(pppm, w, neighbor_particle.indices, current_particle.indices, 1, 0);

        for (int dx = 0; dx < 2; dx++)
            for (int dy = 0; dy < 2; dy++)
                for (int dz = 0; dz < 2; dz++)
                {
                    int weight_idx = dx * 4 + dy * 2 + dz;
                    int3 coord = pm.base_coord + make_int3(dx, dy, dz);
                    auto center = pppm.fdtd.getCenter(coord);
                    if (length(center - neighbor_particle.pos) > pppm.pg.grid_size * 1.5)
                    {
                        add_particle_near_field(pppm, w, neighbor_particle.indices, center, -pm.weight[weight_idx], 0);
                    }
                }
        pppm.cache.particle_data[i].weight = w;
    }
}

void cache_particle_data(PPPMSolver &pppm)
{
    int particle_num = pppm.pg.particles.size();
    printf("particle num: %d\n", particle_num);
    cuExecute(particle_num, precompute_cache_data, pppm);
}

__global__ void solve_particle_from_cache_kernel(PPPMSolver pppm)
{
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= pppm.pg.particles.size())
        return;
    ParticleMap &pm = pppm.cache.particle_map[particle_id];
    BoundaryHistory &history = pppm.particle_history[particle_id];
    Range r = pm.range;
    float sum = 0;
    int t = pppm.fdtd.t;
    history.dirichlet[t] = 0;
    // caculate the far field from interpolation
    for (int dx = 0; dx < 2; dx++)
        for (int dy = 0; dy < 2; dy++)
            for (int dz = 0; dz < 2; dz++)
            {
                int weight_idx = dx * 4 + dy * 2 + dz;
                int3 coord = pm.base_coord + make_int3(dx, dy, dz);
                sum += pm.weight[weight_idx] * pppm.far_field[t](coord);
            }

    // caculate the near field from cache
    float factor = 0;  // the factor of G_t
    for (int i = r.start; i < r.end; i++)
    {
        auto &data = pppm.cache.particle_data[i];
        sum += data.weight.convolution(history.neumann, history.dirichlet, t);
        factor +=
            data.weight.double_layer[0];  // "lumped mass" for G_t, however, the factor is very small compared to 0.5
    }
    // Equation (2.12) in Paper:https://epubs.siam.org/doi/pdf/10.1137/090775981
    history.dirichlet[pppm.fdtd.t] = sum / (0.5 - factor);
}

void solve_particle_from_cache(PPPMSolver &pppm)
{
    int particle_num = pppm.pg.particles.size();
    cuExecute(particle_num, solve_particle_from_cache_kernel, pppm);
}

}  // namespace pppm