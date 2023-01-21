#pragma once
#include "bem.h"
#include "particle_grid.h"

namespace pppm
{

#define RECOMPUTE_THRESHOLD 0.01
#define GRID_CACHE_SIZE 27
class GridCache
{
    public:
        struct CacheElement
        {
                LayerWeightHalf fdtd_near_weight[GRID_CACHE_SIZE];
                LayerWeightHalf bem_near_weight[GRID_CACHE_SIZE];
                float3 center;
                float3 normal;
                float area;
        };
        struct FaceIndex
        {
                int face_idx;
                bool need_recompute;
                CGPU_FUNC bool is_zero() const { return need_recompute == false; }
        };

        CompactIndexArray<FaceIndex> recompute_list;
        GArr<CacheElement> cache;
        bool empty_cache = true;
        void init(int triangle_num)
        {
            cache.resize(triangle_num);
            recompute_list.reserve(triangle_num);
        }
        void update_cache(const ParticleGrid &pg, const TDBEM &bem, bool log_time = false);

        GPU_FUNC int weight_idx(int3 grid_coord, int3 tri_coord)
        {
            int3 coord = grid_coord - tri_coord + make_int3(1, 1, 1);
            return coord.x + coord.y * 3 + coord.z * 9;
        }

        GPU_FUNC int3 weight_coord(int weight_idx)
        {
            int3 coord;
            coord.z = weight_idx / 9;
            weight_idx -= coord.z * 9;
            coord.y = weight_idx / 3;
            weight_idx -= coord.y * 3;
            coord.x = weight_idx;
            return coord - make_int3(1, 1, 1);
        }

        GPU_FUNC LayerWeightHalf &fdtd_near_weight(int tri_idx, int3 tri_coord, int3 grid_coord)
        {
            return cache[tri_idx].fdtd_near_weight[weight_idx(grid_coord, tri_coord)];
        }

        GPU_FUNC LayerWeightHalf &bem_near_weight(int tri_idx, int3 tri_coord, int3 grid_coord)
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
            recompute_list.clear();
        }
};

#define PARTICLE_CACHE_SIZE 1024
#define INTERPOLATION_WEIGHT_SIZE 8
class FaceCache
{
    public:
        struct CacheElement
        {
                LayerWeightHalf weight;
                float normal_angle;
                float distance;
                bool empty_cache;
        };
        struct FacePairIndex
        {
                int src_idx;
                int dst_idx;
                int common_vertex;
                bool need_recompute;
                CGPU_FUNC bool is_zero() const { return need_recompute == false; }
                CGPU_FUNC bool operator<(const FacePairIndex &other) const
                {
                    return common_vertex < other.common_vertex;
                }
        };
        GArr2D<int> cache_index;
        GArr<int> cache_size;
        GArr2D<CacheElement> cache;
        GArr2D<float> interpolation_weight;
        CompactIndexArray<FacePairIndex> recompute_list;

        void init(int triangle_num)
        {
            cache_index.resize(triangle_num, triangle_num);
            cache_index.reset_minus_one();
            cache.resize(triangle_num, PARTICLE_CACHE_SIZE);
            recompute_list.reserve(triangle_num * PARTICLE_CACHE_SIZE);
            cache_size.resize(triangle_num);
            cache_size.reset();
            interpolation_weight.resize(triangle_num, INTERPOLATION_WEIGHT_SIZE);
        }
        void update_cache(const ParticleGrid &pg, const TDBEM &bem, bool log_time = false);

        GPU_FUNC LayerWeightHalf &face2face_weight(int i, int j) { return cache(i, cache_index(i, j)).weight; }

        CGPU_FUNC bool need_recompute(const CacheElement &old, const float normal_angle, const float distance)
        {
            bool ret = (abs(old.normal_angle - normal_angle) > RECOMPUTE_THRESHOLD ||
                        abs(old.distance - distance) > RECOMPUTE_THRESHOLD * distance);
            return ret;
        }

        GPU_FUNC CacheElement &get_cache_element_with_check(int i, int j)
        {
            int idx = cache_index(i, j);
            if (idx == -1)
            {
                // atomic add cache_size[i]
                idx = atomicAdd(&cache_size[i], 1);
                idx = idx % PARTICLE_CACHE_SIZE;
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
            recompute_list.clear();
        }
};

}  // namespace pppm