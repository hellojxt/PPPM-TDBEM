#include "gauss/triangle.h"
#include "macro.h"
#include "pppm_cache.h"

namespace pppm
{

GPU_FUNC inline void add_grid_near_field(ParticleGrid &pg,
                                         TDBEM &bem,
                                         LayerWeight<float> &w,
                                         int src_face_id,
                                         int3 dst,
                                         float scale,
                                         int offset)
{
    Triangle &tri = pg.triangles[src_face_id];
    if (tri.grid_coord.x == dst.x && tri.grid_coord.y == dst.y && tri.grid_coord.z == dst.z)
        return;
    float3 dst_point = pg.getCenter(dst);  // use the center of the grid cell as destination point
    LayerWeight w_current;
    bem.laplace_weight(pg.vertices.data(), PairInfo(tri.indices, dst_point), &w_current);
    w.add(w_current, scale, offset);
}

GPU_FUNC inline void add_laplacian_near_field(ParticleGrid &pg,
                                              TDBEM &bem,
                                              LayerWeight<float> &w,
                                              int src_face_id,
                                              int3 dst,
                                              float scale = 1,
                                              int offset = 0)
{
    scale = scale / (pg.grid_size * pg.grid_size);
    add_grid_near_field(pg, bem, w, src_face_id, dst + make_int3(-1, 0, 0), scale, offset);
    add_grid_near_field(pg, bem, w, src_face_id, dst + make_int3(1, 0, 0), scale, offset);
    add_grid_near_field(pg, bem, w, src_face_id, dst + make_int3(0, -1, 0), scale, offset);
    add_grid_near_field(pg, bem, w, src_face_id, dst + make_int3(0, 1, 0), scale, offset);
    add_grid_near_field(pg, bem, w, src_face_id, dst + make_int3(0, 0, -1), scale, offset);
    add_grid_near_field(pg, bem, w, src_face_id, dst + make_int3(0, 0, 1), scale, offset);
    add_grid_near_field(pg, bem, w, src_face_id, dst, -6 * scale, offset);
}

__global__ void update_grid_cache_kernel(GridCache gc, ParticleGrid pg, TDBEM bem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gc.cache.size())
        return;
    float c = AIR_WAVE_SPEED, dt = pg.delta_t;
    GridCache::CacheElement &e = gc.cache[idx];
    Triangle &tri = pg.triangles[idx];
    int3 tri_coord = tri.grid_coord;
    if (gc.need_recompute(e, tri, pg.grid_size) || gc.empty_cache)
    {
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                {
                    int3 neighbor_coord = tri_coord + make_int3(dx, dy, dz);
                    int weight_idx = gc.weight_idx(neighbor_coord, tri_coord);
                    LayerWeight<float> w;
                    w.reset();
                    add_grid_near_field(pg, bem, w, idx, neighbor_coord, 2, -1);
                    add_laplacian_near_field(pg, bem, w, idx, neighbor_coord, c * c * dt * dt, -1);
                    add_grid_near_field(pg, bem, w, idx, neighbor_coord, -1, -2);
                    e.fdtd_near_weight[weight_idx] = w;
                    w.reset();
                    add_grid_near_field(pg, bem, w, idx, neighbor_coord, 1, 0);
                    e.bem_near_weight[weight_idx] = w;
                }
        e.area = tri.area;
        e.center = tri.center;
        e.normal = tri.normal;
    }
}

void GridCache::update_cache(const ParticleGrid &pg, const TDBEM &bem)
{
    cuExecute(pg.triangles.size(), update_grid_cache_kernel, *this, pg, bem);
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

// angle between two vectors
GPU_FUNC inline float angle_between(float3 normal1, float3 normal2)
{
    float dot = normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z;
    float det = normal1.x * normal2.y * normal2.z + normal1.y * normal2.x * normal2.z +
                normal1.z * normal2.x * normal2.y - normal1.x * normal2.y * normal2.z -
                normal1.y * normal2.x * normal2.z - normal1.z * normal2.x * normal2.y;
    return atan2(det, dot);
}

__global__ void update_particle_cache_kernel(FaceCache pc, ParticleGrid pg, TDBEM bem)
{
    int base_coord_idx = blockIdx.x;
    if (base_coord_idx >= pg.base_coord_nonempty.size())
        return;
    int3 base_coord = pg.base_coord_nonempty.coord(base_coord_idx);
    auto &neighbor_list = pg.neighbor_4_square_list(base_coord);
    auto &center_triangle_list = pg.base_coord_face_list(base_coord);
    int center_num = center_triangle_list.size();
    int neighbor_num = neighbor_list.size();
    int total_num = center_num * neighbor_num;
    LayerWeight w;
    for (int i = threadIdx.x; i < total_num; i += blockDim.x)
    {
        int neighbor_i = i / center_num;
        int center_i = i % center_num;
        int neighbor_face_idx = neighbor_list[neighbor_i];
        auto &neighbor_face = pg.triangles[neighbor_face_idx];
        int center_face_idx = center_triangle_list[center_i];
        auto &center_face = pg.triangles[center_face_idx];
        int normal_angle = angle_between(neighbor_face.normal, center_face.normal);
        int distance = length(neighbor_face.center - center_face.center);
        auto &e = pc.get_cache_element_with_check(center_face_idx, neighbor_face_idx);
        if (e.empty_cache || pc.need_recompute(e, normal_angle, distance))
        {
            e.normal_angle = normal_angle;
            e.distance = distance;
            e.empty_cache = false;
            w.reset();
            add_particle_near_field(pg, bem, w, neighbor_face.indices, center_face.indices, 1, 0);
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
                            add_particle_near_field(pg, bem, w, neighbor_face.indices, center,
                                                    -pc.interpolation_weight(center_face_idx, weight_idx), 0);
                        }
                    }
            pc.face2face_weight(center_face_idx, neighbor_face_idx) = w;
        }
    }
}

void FaceCache::update_cache(const ParticleGrid &pg, const TDBEM &bem)
{
    cuExecute(pg.triangles.size(), update_interpolation_weight_kernel, *this, pg);
    cuExecuteBlock(pg.base_coord_nonempty.size(), 64, update_particle_cache_kernel, *this, pg, bem);
}
}  // namespace pppm