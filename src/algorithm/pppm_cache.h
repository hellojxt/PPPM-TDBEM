#pragma once
#include "bem.h"
#include "particle_grid.h"

namespace pppm
{

#define RECOMPUTE_THRESHOLD 0.01

class GridCache
{
    public:
        struct CacheElement
        {
                LayerWeight<float> fdtd_near_weight[27];
                LayerWeight<float> bem_near_weight[27];
                float3 center;
                float3 normal;
                float area;
        };
        GArr<CacheElement> cache;
        bool empty_cache = true;
        void init(int triangle_num) { cache.resize(triangle_num); }
        void update_cache(const ParticleGrid &pg, const TDBEM &bem);

        GPU_FUNC int weight_idx(int3 grid_coord, int3 tri_coord)
        {
            int3 coord = grid_coord - tri_coord + make_int3(1, 1, 1);
            return coord.x + coord.y * 3 + coord.z * 9;
        }

        GPU_FUNC LayerWeight<float> &fdtd_near_weight(int tri_idx, int3 tri_coord, int3 grid_coord)
        {
            return cache[tri_idx].fdtd_near_weight[weight_idx(grid_coord, tri_coord)];
        }

        GPU_FUNC LayerWeight<float> &bem_near_weight(int tri_idx, int3 tri_coord, int3 grid_coord)
        {
            return cache[tri_idx].bem_near_weight[weight_idx(grid_coord, tri_coord)];
        }

        CGPU_FUNC bool need_recompute(const CacheElement &old, const Triangle &tri, float grid_size)
        {
            return (length(old.center - tri.center) > RECOMPUTE_THRESHOLD * grid_size ||
                    length(old.normal - tri.normal) > RECOMPUTE_THRESHOLD ||
                    abs(old.area - tri.area) > RECOMPUTE_THRESHOLD * tri.area);
        }
        void clear()
        {
            empty_cache = true;
            cache.clear();
        }
};

#define PARTICLE_CACHE_SIZE 256
#define INTERPOLATION_WEIGHT_SIZE 8
class FaceCache
{
    public:
        struct CacheElement
        {
                LayerWeight<float> weight;
                float normal_angle;
                float distance;
                bool empty_cache;
        };
        GArr2D<int> cache_index;
        GArr<int> cache_size;
        GArr2D<CacheElement> cache;
        GArr2D<float> interpolation_weight;
        void init(int triangle_num)
        {
            cache_index.resize(triangle_num, triangle_num);
            cache_index.reset_minus_one();
            cache.resize(triangle_num, PARTICLE_CACHE_SIZE);
            cache_size.resize(triangle_num);
            cache_size.reset();
            interpolation_weight.resize(triangle_num, INTERPOLATION_WEIGHT_SIZE);
        }
        void update_cache(const ParticleGrid &pg, const TDBEM &bem);

        GPU_FUNC LayerWeight<float> &face2face_weight(int i, int j) { return cache(i, cache_index(i, j)).weight; }

        CGPU_FUNC bool need_recompute(const CacheElement &old, const float normal_angle, const float distance)
        {
            return (abs(old.normal_angle - normal_angle) > RECOMPUTE_THRESHOLD * 2 * M_PI ||
                    abs(old.distance - distance) > RECOMPUTE_THRESHOLD * distance);
        }

        GPU_FUNC CacheElement &get_cache_element_with_check(int i, int j)
        {
            int idx = cache_index(i, j);
            if (idx == -1)
            {
                // atomic add cache_size[i]
                idx = atomicAdd(&cache_size[i], 1);
#ifdef MEMORY_CHECK
                assert(idx < PARTICLE_CACHE_SIZE);
#endif
                cache_index(i, j) = idx;
                cache(i, idx).empty_cache = true;
            }
            return cache(i, idx);
        }

        void clear()
        {
            cache_index.clear();
            cache_size.clear();
            cache.clear();
            interpolation_weight.clear();
        }
};

}  // namespace pppm