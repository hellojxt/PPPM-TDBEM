#include "gauss/triangle.h"
#include "macro.h"
#include "pppm_cache.h"

namespace pppm
{
GPU_FUNC bool triangle_in_grid_cell(ParticleGrid &pg, Triangle &tri, int3 grid_coord)
{
    // float3 nearst_point = get_nearest_triangle_point(pg.getCenter(grid_coord), pg.vertices[tri.indices.x],
    //                                                  pg.vertices[tri.indices.y], pg.vertices[tri.indices.z]);
    // float3 grid_center = pg.getCenter(grid_coord);
    // return abs(nearst_point.x - grid_center.x) < pg.grid_size / 2 &&
    //        abs(nearst_point.y - grid_center.y) < pg.grid_size / 2 &&
    //        abs(nearst_point.z - grid_center.z) < pg.grid_size / 2;
    return (tri.grid_coord.x == grid_coord.x) && (tri.grid_coord.y == grid_coord.y) &&
           (tri.grid_coord.z == grid_coord.z);
}

template <typename T>
GPU_FUNC inline void
add_grid_near_field(ParticleGrid &pg, TDBEM &bem, LayerWeight<T> &w, int src_face_id, int3 dst, float scale, int offset)
{
    Triangle &tri = pg.triangles[src_face_id];
    if (triangle_in_grid_cell(pg, tri, dst))
        return;
    float3 dst_point = pg.getCenter(dst);  // use the center of the grid cell as destination point
    LayerWeight<T> w_current;
    bem.laplace_weight(pg.vertices.data(), PairInfo(tri.indices, dst_point), &w_current);
    w.add(w_current, scale, offset);
}

GPU_FUNC inline void batch_grid_near_field(ParticleGrid &pg,
                                           TDBEM &bem,
                                           LayerWeight<cpx> &w,
                                           int src_face_id,
                                           TargetCoordArray &dst)
{
    Triangle &tri = pg.triangles[src_face_id];
    for (int k = 0; k <= STEP_NUM / 2; k++)
    {
        w.single_layer[k] =
            face2PointIntegrand(pg.vertices.data(), tri.indices, dst, bem.wave_numbers[k], SINGLE_LAYER);
        w.double_layer[k] =
            face2PointIntegrand(pg.vertices.data(), tri.indices, dst, bem.wave_numbers[k], DOUBLE_LAYER);
    }
    for (int k = STEP_NUM / 2 + 1; k < STEP_NUM; k++)
    {
        w.single_layer[k] = conj(w.single_layer[STEP_NUM - k]);
        w.double_layer[k] = conj(w.double_layer[STEP_NUM - k]);
    }
}

__global__ void construct_compute_list_kernel(GridCache gc, ParticleGrid pg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gc.cache.size())
        return;
    GridCache::CacheElement &e = gc.cache[idx];
    Triangle &tri = pg.triangles[idx];
    gc.recompute_list[idx].face_idx = idx;
    gc.recompute_list[idx].need_recompute = gc.empty_cache || gc.need_recompute(e, tri, pg.grid_size); // 将需要计算的三角面元放入缓存
}
// 这个函数在干啥?
__global__ void update_grid_cache_kernel(GridCache gc, ParticleGrid pg, TDBEM bem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_CACHE_SIZE * gc.recompute_list.size())
        return;
    float c = AIR_WAVE_SPEED, dt = pg.delta_t;
    int face_idx = gc.recompute_list[idx / GRID_CACHE_SIZE].face_idx;
    GridCache::CacheElement &e = gc.cache[face_idx];
    Triangle &tri = pg.triangles[face_idx];
    int3 tri_coord = tri.grid_coord;
    int weight_idx = idx % GRID_CACHE_SIZE;
    int3 neighbor_coord = tri_coord + gc.weight_coord(weight_idx);
    LayerWeight<float> w;
    w.reset();
    LayerWeight<cpx> w_original;
    w_original.reset();
    TargetCoordArray dst;
    dst.size = 0;
    int3 dst_coord[8] = {neighbor_coord,
                         neighbor_coord + make_int3(-1, 0, 0),
                         neighbor_coord + make_int3(1, 0, 0),
                         neighbor_coord + make_int3(0, -1, 0),
                         neighbor_coord + make_int3(0, 1, 0),
                         neighbor_coord + make_int3(0, 0, -1),
                         neighbor_coord + make_int3(0, 0, 1),
                         neighbor_coord};
    float scale = c * c * dt * dt / (pg.grid_size * pg.grid_size);
    float dst_scale[8] = {2, scale, scale, scale, scale, scale, scale, -6 * scale};
    for (int i = 0; i < 8; i++)
    {
        if (!triangle_in_grid_cell(pg, tri, dst_coord[i]))
        {
            dst.data[dst.size] = pg.getCenter(dst_coord[i]);
            dst.scale[dst.size] = dst_scale[i];
            dst.size++;
        }
    }
    batch_grid_near_field(pg, bem, w_original, face_idx, dst);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord, 2, 0);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord + make_int3(-1, 0, 0), scale, 0);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord + make_int3(1, 0, 0), scale, 0);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord + make_int3(0, -1, 0), scale, 0);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord + make_int3(0, 1, 0), scale, 0);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord + make_int3(0, 0, -1), scale, 0);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord + make_int3(0, 0, 1), scale, 0);
    // add_grid_near_field(pg, bem, w_original, face_idx, neighbor_coord, -6 * scale, 0);
    bem.scaledFFT(&w_original, &w);
    w.move(-1);
    add_grid_near_field(pg, bem, w, face_idx, neighbor_coord, -1, -2);
    e.fdtd_near_weight[weight_idx].set(w);
    w.reset();
    add_grid_near_field(pg, bem, w, face_idx, neighbor_coord, 1, 0);
    e.bem_near_weight[weight_idx].set(w);
    if (weight_idx == 0)
    {
        e.area = tri.area;
        e.center = tri.center;
        e.normal = tri.normal;
    }
}

void GridCache::update_cache(const ParticleGrid &pg, const TDBEM &bem, bool log_time)
{
    START_TIME(log_time)
    cuExecute(pg.triangles.size(), construct_compute_list_kernel, *this, pg); // 建立需要计算的三角面表,放入缓存
    recompute_list.remove_zeros();
    LOG_TIME("construct compute list")
    cuExecute(GRID_CACHE_SIZE * recompute_list.size(), update_grid_cache_kernel, *this, pg, bem); // 这个函数在干啥?
    LOG_TIME("update grid cache")
    empty_cache = false;
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

__global__ void update_interpolation_weight_kernel(FaceCache pc, ParticleGrid pg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pg.triangles.size())
        return;
    Triangle &tri = pg.triangles[idx];
    // calculate the weights of 2*2*2 grid with guass interpolation
    float w_temp[8];
    float w[8];
    for (int i = 0; i < 8; i++)
        w[i] = 0;
    float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
    float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
    float3 dst_v[3] = {{pg.vertices[tri.indices.x]}, {pg.vertices[tri.indices.y]}, {pg.vertices[tri.indices.z]}};
    float trg_jacobian = jacobian(dst_v);
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
    {
        float3 v = local_to_global(guass_x[i][0], guass_x[i][1], dst_v);
        float3 d = (v - pg.getCenter(tri.grid_base_coord)) / pg.grid_size;
        trilinear_interpolation(d, w_temp);
        weight_add(w_temp, 0.5 * guass_w[i] * trg_jacobian, w);
    }
    for (int i = 0; i < 8; i++)
        pc.interpolation_weight(idx, i) = w[i];
}

__global__ void compute_face_compute_list_kernel(FaceCache pc, ParticleGrid pg)
{
    int base_coord_idx = blockIdx.x;
    if (base_coord_idx >= pg.base_coord_nonempty.size())
        return;
    int3 base_coord = pg.base_coord_nonempty[base_coord_idx].coord;
    auto &neighbor_list = pg.neighbor_4_square_list(base_coord);
    auto &center_triangle_list = pg.base_coord_face_list(base_coord);
    int center_num = center_triangle_list.size();
    int neighbor_num = neighbor_list.size();
    int total_num = center_num * neighbor_num;
    LayerWeight w;
    for (int i = threadIdx.x; i < total_num; i += blockDim.x) // total_num ~ 2e6
    {
        int neighbor_i = i / center_num;
        int center_i = i % center_num;
        int neighbor_face_idx = neighbor_list[neighbor_i];
        auto &neighbor_face = pg.triangles[neighbor_face_idx];
        int center_face_idx = center_triangle_list[center_i];
        auto &center_face = pg.triangles[center_face_idx];
        float normal_angle = dot(neighbor_face.normal, center_face.normal);
        float distance = length(neighbor_face.center - center_face.center);
        auto &e = pc.get_cache_element_with_check(center_face_idx, neighbor_face_idx);
        int index = center_face_idx * PARTICLE_CACHE_SIZE + neighbor_i;
        pc.recompute_list[index].src_idx = neighbor_face_idx;
        pc.recompute_list[index].dst_idx = center_face_idx;
        pc.recompute_list[index].common_vertex = triangle_common_vertex_num(neighbor_face.indices, center_face.indices);
        if (e.empty_cache || pc.need_recompute(e, normal_angle, distance))
        {
            e.normal_angle = normal_angle;
            e.distance = distance;
            e.empty_cache = false;
            pc.recompute_list[index].need_recompute = true;
        }
    }
}

template <typename T>
GPU_FUNC inline void add_particle_near_field(ParticleGrid &pg,
                                             TDBEM &bem,
                                             LayerWeight<float> &layer_weight,
                                             int3 src,
                                             T dst,
                                             float scale,
                                             int offset)
{
    LayerWeight w;
    bem.laplace_weight(pg.vertices.data(), PairInfo(src, dst), &w);
    layer_weight.add(w, scale, offset);
}
// 这个函数又是做什么的?
__global__ void update_particle_cache_kernel(FaceCache pc, ParticleGrid pg, TDBEM bem)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.recompute_list.size())
        return;
    int neighbor_face_idx = pc.recompute_list[i].src_idx;
    int center_face_idx = pc.recompute_list[i].dst_idx;
    auto &neighbor_face = pg.triangles[neighbor_face_idx];
    auto &center_face = pg.triangles[center_face_idx];
    int3 base_coord = center_face.grid_base_coord;
    LayerWeight<float> w;
    LayerWeight<cpx> w_temp1;
    LayerWeight<cpx> w_temp2;
    bem.laplace_weight(pg.vertices.data(), PairInfo(neighbor_face.indices, center_face.indices), &w_temp1);
    // add_particle_near_field(pg, bem, w, neighbor_face.indices, center_face.indices, 1, 0);
    TargetCoordArray dst;
    dst.size = 0;
    for (int dx = 0; dx < 2; dx++)
        for (int dy = 0; dy < 2; dy++)
            for (int dz = 0; dz < 2; dz++)
            {
                int weight_idx = dx * 4 + dy * 2 + dz;
                int3 coord = base_coord + make_int3(dx, dy, dz);
                auto center = pg.getCenter(coord);
                if (abs(center.x - neighbor_face.center.x) > pg.grid_size * 1.5 ||
                    abs(center.y - neighbor_face.center.y) > pg.grid_size * 1.5 ||
                    abs(center.z - neighbor_face.center.z) > pg.grid_size * 1.5)
                {
                    dst.data[dst.size] = center;
                    dst.scale[dst.size] = -pc.interpolation_weight(center_face_idx, weight_idx);
                    dst.size++;
                }
            }
    batch_grid_near_field(pg, bem, w_temp2, neighbor_face_idx, dst);
    w_temp1.add(w_temp2);
    bem.scaledFFT(&w_temp1, &w);
    pc.face2face_weight(center_face_idx, neighbor_face_idx).set(w);
}

void FaceCache::update_cache(const ParticleGrid &pg, const TDBEM &bem, bool log_time)
{
    START_TIME(log_time)
    cuExecute(pg.triangles.size(), update_interpolation_weight_kernel, *this, pg); // 计算插值权重
    LOG_TIME("update interpolation weight")
    recompute_list.reset();
    cuExecuteBlock(pg.base_coord_nonempty.size(), 64, compute_face_compute_list_kernel, *this, pg); // 大概是计算哪些面需要重新计算?
    recompute_list.remove_zeros();
    recompute_list.sort();
    LOG_TIME("compute face compute list")
    if (log_time)
    {
        LOG("recompute list size: " << recompute_list.size());
    }
    cuExecute(recompute_list.size(), update_particle_cache_kernel, *this, pg, bem); // 这是在干啥?
    LOG_TIME("update particle cache")
}
}  // namespace pppm