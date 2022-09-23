#include "particle_grid.h"

namespace pppm
{
std::ostream &operator<<(std::ostream &out, const BElement &be)
{
    out << " cell_coord: " << be.cell_coord << " pos: " << be.pos << " normal: " << be.normal << std::endl;
    return out;
}

__global__ void construct_kernel(GArr<float3> vertices,
                                 GArr<int3> triangles,
                                 BBox bbox,
                                 int level,
                                 GArr<BElement> particles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= triangles.size())
        return;
    float3 v0     = vertices[triangles[idx].x];
    float3 v1     = vertices[triangles[idx].y];
    float3 v2     = vertices[triangles[idx].z];
    float3 normal = normalize(cross(v1 - v0, v2 - v0));
    float3 center = (v0 + v1 + v2) / 3.0f;
    BElement particle;
    particle.normal  = normal;
    particle.pos     = center;
    particle.indices = triangles[idx];
    // calculate coord for morton code
    uint MAX_MORTON_CODE = 1U << level;
    float3 coord         = (center - bbox.min) / bbox.width * MAX_MORTON_CODE;
    uint3 ucoord         = make_uint3((unsigned int)coord.x, (unsigned int)coord.y, (unsigned int)coord.z);
    particle.cell_coord  = ucoord;
    particles[idx]       = particle;
}

__global__ void calculate_differ_kernel(GArr<BElement> particles, GArr<int> differ)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= particles.size() - 1)
        return;
    differ[idx] = (encode_morton(particles[idx].cell_coord) != encode_morton(particles[idx + 1].cell_coord));
}

__global__ void grid_dense_map_kernel(GArr<int> differ, GArr<Range> grid_dense_map)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= differ.size())
        return;
    if (idx == 0 || differ[idx - 1] != differ[idx])
        grid_dense_map[differ[idx]].start = idx;
}

__global__ void grid_dense_map_post_kernel(GArr<BElement> particles,
                                           GArr<Range> grid_dense_map,
                                           GArr3D<Range> grid_hash_map)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= grid_dense_map.size())
        return;
    if (idx == grid_dense_map.size() - 1)
        grid_dense_map[idx].end = particles.size();
    else
        grid_dense_map[idx].end = grid_dense_map[idx + 1].start;

    BElement particle                        = particles[grid_dense_map[idx].start];
    uint3 coord                              = particle.cell_coord;
    grid_hash_map(coord.x, coord.y, coord.z) = grid_dense_map[idx];
}

void ParticleGrid::construct_grid()
{
    particles.resize(triangles.size());
    // extend to the 2^n resolution and construct the grid
    int max_grid_dim = std::max(grid_dim.x, std::max(grid_dim.y, grid_dim.z));
    int n            = ceil(log2(max_grid_dim));
    auto extend_res  = pow(2, n);
    BBox bbox;
    bbox.min   = min_pos;
    bbox.width = grid_size * extend_res;
    bbox.max   = bbox.min + bbox.width;
    cuExecute(particles.size(), construct_kernel, vertices, triangles, bbox, n, particles);
    // sort the particles by the morton code
    thrust::sort(thrust::device, particles.begin(), particles.end(), [] GPU_FUNC(const BElement &a, const BElement &b) {
        return encode_morton(a.cell_coord) < encode_morton(b.cell_coord);
    });
    // calculate differ and prefix sum of differ
    GArr<int> differ(particles.size());
    cuExecute(particles.size(), calculate_differ_kernel, particles, differ);
    thrust::exclusive_scan(thrust::device, differ.begin(), differ.end(), differ.begin(), 0);
    // calculate the particle map
    grid_dense_map.resize(differ.last_item() + 1);
    cuExecute(differ.size(), grid_dense_map_kernel, differ, grid_dense_map);
    differ.clear();
    // post process particle map
    grid_hash_map.resize(grid_dim.x, grid_dim.y, grid_dim.z);
    grid_hash_map.reset();
    cuExecute(particles.size(), grid_dense_map_post_kernel, particles, grid_dense_map, grid_hash_map);
}

}  // namespace pppm