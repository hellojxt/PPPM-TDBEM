#include "particle_grid.h"

namespace pppm
{
    std::ostream &operator<<(std::ostream &out, const BElement &be)
    {
        // print cell id in binary format
        out << "cell_id: ";
        uint cell_id_binary = be.cell_id;
        for (int i = 24; i >= 0; i--)
        {
            out << ((cell_id_binary >> i) & 1);
        }
        out << " cell_coord: " << be.cell_coord << " particle_id: " << be.particle_id
            << " pos: " << be.pos << " normal: " << be.normal << " area: " << be.area << std::endl;
        return out;
    }

    __global__ void construct_kernel(GArr<float3> vertices, GArr<int3> triangles, BBox bbox, int level, GArr<BElement> particles)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= triangles.size())
            return;
        float3 v0 = vertices[triangles[idx].x];
        float3 v1 = vertices[triangles[idx].y];
        float3 v2 = vertices[triangles[idx].z];
        float3 normal = normalize(cross(v1 - v0, v2 - v0));
        float area = 0.5f * length(cross(v1 - v0, v2 - v0));
        float3 center = (v0 + v1 + v2) / 3.0f;
        BElement particle;
        particle.particle_id = idx;
        particle.normal = normal;
        particle.area = area;
        particle.pos = center;
        // calculate morton code
        uint MAX_MORTON_CODE = 1U << level;
        float3 coord = (center - bbox.min) / bbox.width * MAX_MORTON_CODE;
        uint3 ucoord = make_uint3((unsigned int)coord.x, (unsigned int)coord.y, (unsigned int)coord.z);
        particle.cell_id = encode_morton(ucoord);
        particle.cell_coord = ucoord;
        particles[idx] = particle;
    }

    __global__ void calculate_differ_kernel(GArr<BElement> particles, GArr<int> differ)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= particles.size() - 1)
            return;
        differ[idx] = (particles[idx].cell_id != particles[idx + 1].cell_id);
    }

    __global__ void particle_map_kernel(GArr<int> differ, GArr<Range> particle_map)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= differ.size())
            return;
        if (idx == 0 || differ[idx - 1] != differ[idx])
            particle_map[differ[idx]].start = idx;
    }

    __global__ void particle_map_post_kernel(GArr<BElement> particles, GArr<Range> particle_map, GArr3D<int> grid_hash_map)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= particle_map.size())
            return;
        if (idx == particle_map.size() - 1)
            particle_map[idx].end = particles.size();
        else
            particle_map[idx].end = particle_map[idx + 1].start;

        BElement particle = particles[particle_map[idx].start];
        uint3 coord = particle.cell_coord;
        grid_hash_map(coord.x, coord.y, coord.z) = idx;
    }

    void ParticleGrid::construct_grid()
    {
        particles.resize(triangles.size());
        // extend to the 2^n resoltion and construct the grid
        int max_grid_dim = max(grid_dim.x, max(grid_dim.y, grid_dim.z));
        int n = ceil(log2(max_grid_dim));
        auto extend_res = pow(2, n);
        BBox bbox;
        bbox.min = min_pos;
        bbox.width = grid_size * extend_res;
        bbox.max = bbox.min + bbox.width;
        cuExecute(particles.size(), construct_kernel, vertices, triangles, bbox, n, particles);
        // sort the particles by the morton code
        thrust::sort(thrust::device, particles.begin(), particles.end(), [] GPU_FUNC(const BElement &a, const BElement &b)
                     { return a.cell_id < b.cell_id; });
        // calculate differ and prefix sum of differ
        GArr<int> differ(particles.size());
        cuExecute(particles.size(), calculate_differ_kernel, particles, differ);
        thrust::exclusive_scan(thrust::device, differ.begin(), differ.end(), differ.begin(), 0);
        // calculate the particle map
        particle_map.resize(differ.last_item() + 1);
        cuExecute(differ.size(), particle_map_kernel, differ, particle_map);
        differ.clear();
        // post process particle map
        grid_hash_map.resize(grid_dim.x, grid_dim.y, grid_dim.z);
        grid_hash_map.reset_minus_one();
        cuExecute(particles.size(), particle_map_post_kernel, particles, particle_map, grid_hash_map);
    }

    void ParticleGrid::validate_data()
    {
        auto particles = this->particles.cpu();
        auto particle_map = this->particle_map.cpu();
        auto grid_hash_map = this->grid_hash_map.cpu();
        CArr<int> particle_map_flag;
        particle_map_flag.resize(particles.size());
        particle_map_flag.reset();
        bool valid_pass = true;
        for (int x = 0; x < grid_dim.x; x++)
        {
            for (int y = 0; y < grid_dim.y; y++)
            {
                for (int z = 0; z < grid_dim.z; z++)
                {
                    int cell_id = grid_hash_map(x, y, z);
                    if (cell_id == -1)
                        continue;
                    uint start = particle_map[cell_id].start;
                    uint end = particle_map[cell_id].end;
                    for (uint i = start; i < end; i++)
                    {
                        particle_map_flag[i] = 1;
                        auto particle = particles[i];
                        if (particle.cell_id != encode_morton(particle.cell_coord))
                        {
                            valid_pass = false;
                            LOG_ERROR("particle cell id error: " << particle.cell_id << " != " << encode_morton(particle.cell_coord));
                        }

                        if (particle.cell_coord.x != x || particle.cell_coord.y != y || particle.cell_coord.z != z)
                        {
                            valid_pass = false;
                            LOG_ERROR("particle cell coord error: " << particle.cell_coord << " != " << make_uint3(x, y, z));
                        }
                        auto grid_min_pos = min_pos + make_float3(x * grid_size, y * grid_size, z * grid_size);
                        auto grid_max_pos = grid_min_pos + grid_size;
                        if (particle.pos.x < grid_min_pos.x || particle.pos.x >= grid_max_pos.x ||
                            particle.pos.y < grid_min_pos.y || particle.pos.y >= grid_max_pos.y ||
                            particle.pos.z < grid_min_pos.z || particle.pos.z >= grid_max_pos.z)
                        {
                            valid_pass = false;
                            LOG_ERROR("particle pos error: " << particle.pos << " != " << grid_min_pos << " ~ " << grid_max_pos);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < particles.size(); i++)
        {
            if (particle_map_flag[i] == 0)
            {
                valid_pass = false;
                LOG_ERROR("particle map error: particle " << i << " not in any cell");
            }
        }
        if (valid_pass)
            LOG_INFO("particle grid validation passed");
        particle_map_flag.clear();
    }

    void ParticleGrid::randomly_test()
    {
        float3 min_pos = make_float3(0.0f, 0.0f, 0.0f);
        float grid_size = RAND_F;
        int3 grid_dim = make_int3(8, 8, 8);
        ParticleGrid pg;
        pg.init(min_pos, grid_size, grid_dim);
        int triangle_count = 400;
        CArr<float3> vertices;
        CArr<int3> triangles;
        vertices.resize(triangle_count * 3);
        triangles.resize(triangle_count);
        for (int i = 0; i < triangle_count; i++)
        {
            float3 v0 = make_float3(0.0f, 0.0f, 1.0f);
            float3 v1 = make_float3(1.0f, 0.0f, 0.0f);
            float3 v2 = make_float3(0.0f, 1.0f, 0.0f);
            float3 offset = make_float3(RAND_F, RAND_F, RAND_F) * make_float3(grid_dim - 1) * grid_size;
            vertices[i * 3 + 0] = v0 * grid_size + offset;
            vertices[i * 3 + 1] = v1 * grid_size + offset;
            vertices[i * 3 + 2] = v2 * grid_size + offset;
            triangles[i] = make_int3(i * 3 + 0, i * 3 + 1, i * 3 + 2);
        }
        pg.set_mesh(vertices, triangles);
        pg.construct_grid();
        // SHOW(pg.vertices)
        // SHOW(pg.triangles)
        // SHOW(pg.particles)
        // SHOW(pg.particle_map)
        pg.validate_data();
    }

}