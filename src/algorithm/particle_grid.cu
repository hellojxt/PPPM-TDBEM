#include "particle_grid.h"

namespace pppm
{
__global__ void update_vertices_grid_kernel(ParticleGrid pg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pg.vertices.size())
        return;
    float3 &v = pg.vertices[i];
    int3 coord = pg.getGridCoord(v);
    int3 base_coord = pg.getGridBaseCoord(v);
    pg.grid_face_list(coord).atomic_append(i);
    pg.base_coord_face_list(base_coord).atomic_append(i);
}

void ParticleGrid::construct_vertices_grid()
{
    cuExecute(vertices.size(), update_vertices_grid_kernel, *this);
}
// update the grid and the grid face list
__global__ void update_triangles_kernel(ParticleGrid pg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pg.faces.size())
        return;
    Triangle tri;
    Triangle &old_tri = pg.triangles[i];
    int3 indices = pg.faces[i];
    float3 v0 = pg.vertices[indices.x];
    float3 v1 = pg.vertices[indices.y];
    float3 v2 = pg.vertices[indices.z];
    tri.indices = indices;
    tri.center = (v0 + v1 + v2) / 3.0f;
    tri.normal = normalize(cross(v1 - v0, v2 - v0));
    tri.area = length(cross(v1 - v0, v2 - v0)) / 2.0f;
    tri.grid_coord = pg.getGridCoord(tri.center);
    tri.grid_base_coord = pg.getGridBaseCoord(tri.center);
    if (!pg.empty_grid)
    {
        int3 old_coord = old_tri.grid_coord;
        int3 old_base_coord = old_tri.grid_base_coord;
        if (tri.grid_coord.x != old_coord.x || tri.grid_coord.y != old_coord.y || tri.grid_coord.z != old_coord.z)
        {
            pg.grid_face_list(old_coord)[old_tri.grid_index] = -1;  // mark as deleted
            tri.grid_index = pg.grid_face_list(tri.grid_coord).atomic_append(i);
        }
        else
        {
            tri.grid_index = old_tri.grid_index;
        }
        if (tri.grid_base_coord.x != old_base_coord.x || tri.grid_base_coord.y != old_base_coord.y ||
            tri.grid_base_coord.z != old_base_coord.z)
        {
            pg.base_coord_face_list(old_base_coord)[old_tri.grid_base_index] = -1;  // mark as deleted
            tri.grid_base_index = pg.base_coord_face_list(tri.grid_base_coord).atomic_append(i);
        }
        else
        {
            tri.grid_base_index = old_tri.grid_base_index;
        }
    }
    else
    {
        tri.grid_index = pg.grid_face_list(tri.grid_coord).atomic_append(i);
        tri.grid_base_index = pg.base_coord_face_list(tri.grid_base_coord).atomic_append(i);
    }
    pg.triangles[i] = tri;
}

// remove the out of bound faces
__global__ void update_grid_list_kernel(ParticleGrid pg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= pg.grid_dim || j >= pg.grid_dim || k >= pg.grid_dim)
        return;
    int3 coord = make_int3(i, j, k);
    auto &face_list = pg.grid_face_list(coord);
    auto &base_coord_face_list = pg.base_coord_face_list(coord);
    int face_list_size = 0;
    int base_coord_face_list_size = 0;
    for (int idx = 0; idx < face_list.size(); idx++)
    {
        int face_idx = face_list[idx];
        if (face_idx != -1)
        {
            if (idx != face_list_size)
            {
                face_list[face_list_size] = face_idx;
                pg.triangles[face_idx].grid_index = face_list_size;
            }

            face_list_size++;
        }
    }
    face_list.num = face_list_size;
    for (int idx = 0; idx < base_coord_face_list.size(); idx++)
    {
        int face_idx = base_coord_face_list[idx];
        if (face_idx != -1)
        {
            if (idx != base_coord_face_list_size)
            {
                base_coord_face_list[base_coord_face_list_size] = face_idx;
                pg.triangles[face_idx].grid_base_index = base_coord_face_list_size;
            }
            base_coord_face_list_size++;
        }
    }
    base_coord_face_list.num = base_coord_face_list_size;
}

void ParticleGrid::construct_grid()
{
    cuExecute(faces.size(), update_triangles_kernel, *this);
    cuExecute3D(dim3(grid_dim, grid_dim, grid_dim), update_grid_list_kernel, *this);
}

__global__ void init_neigbor_list_size_kernel(ParticleGrid pg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= pg.grid_dim - 1 || i < 1 || j >= pg.grid_dim - 1 || j < 1 || k >= pg.grid_dim - 1 || k < 1)
        return;
    int3 coord = make_int3(i, j, k);
    int neighbor_3_num = 0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)  // iterate over all the 3x3x3 grids
            {
                int3 neighbor_coord = coord + make_int3(dx, dy, dz);
                auto &face_list = pg.grid_face_list(neighbor_coord);
                neighbor_3_num += face_list.size();
            }
    int coord_idx = pg.neighbor_3_square_list.index(i, j, k);
    pg.neighbor_3_square_nonempty[coord_idx].coord = coord;
    pg.neighbor_3_square_nonempty[coord_idx].num = neighbor_3_num;
    pg.base_coord_nonempty[coord_idx].coord = coord;
    pg.base_coord_nonempty[coord_idx].num = pg.base_coord_face_list(coord).size();
}

// 3x3x3 neighbor list (-1, -1, -1) -> (1, 1, 1)
__global__ void fill_neighbor_list_3_kernel(ParticleGrid pg)
{
    int i = blockIdx.x;
    if (i >= pg.neighbor_3_square_nonempty.size())
        return;
    int3 coord = pg.neighbor_3_square_nonempty[i].coord;
    auto &neighbor_list = pg.neighbor_3_square_list(coord);
    int neighbor_num_prefix_sum[27];
    int neighbor_num_sum = 0;
    for (int idx = 0; idx < 27; idx++)
    {
        int3 neighbor_coord = coord + make_int3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 - 1);
        neighbor_num_prefix_sum[idx] = neighbor_num_sum;
        neighbor_num_sum += pg.grid_face_list(neighbor_coord).size();
    }
    for (int idx = threadIdx.x; idx < 27; idx += blockDim.x)
    {
        int3 neighbor_coord = coord + make_int3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 - 1);
        auto &face_list = pg.grid_face_list(neighbor_coord);
        for (int face_idx = 0; face_idx < face_list.size(); face_idx++)
        {
            neighbor_list[neighbor_num_prefix_sum[idx] + face_idx] = face_list[face_idx];
        }
    }
    neighbor_list.num = neighbor_num_sum;
}

// 4x4x4 neighbor list (-1, -1, -1) -> (2, 2, 2)
__global__ void fill_neighbor_list_4_kernel(ParticleGrid pg)
{
    int i = blockIdx.x;
    if (i >= pg.base_coord_nonempty.size())
        return;
    int3 coord = pg.base_coord_nonempty[i].coord;
    auto &neighbor_list = pg.neighbor_4_square_list(coord);
    int neighbor_num_prefix_sum[64];
    int neighbor_num_sum = 0;
    for (int idx = 0; idx < 64; idx++)
    {
        int3 neighbor_coord = coord + make_int3(idx % 4 - 1, idx / 4 % 4 - 1, idx / 16 - 1);
        neighbor_num_prefix_sum[idx] = neighbor_num_sum;
        neighbor_num_sum += pg.grid_face_list(neighbor_coord).size();
    }
    for (int idx = threadIdx.x; idx < 64; idx += blockDim.x)
    {
        int3 neighbor_coord = coord + make_int3(idx % 4 - 1, idx / 4 % 4 - 1, idx / 16 - 1);
        auto &face_list = pg.grid_face_list(neighbor_coord);
        for (int face_idx = 0; face_idx < face_list.size(); face_idx++)
        {
            neighbor_list[neighbor_num_prefix_sum[idx] + face_idx] = face_list[face_idx];
        }
    }
    neighbor_list.num = neighbor_num_sum;
}

struct empty_neighbor_num
{
        __host__ __device__ bool operator()(const NeighborNum &n) { return n.num == 0; }
};

void ParticleGrid::construct_neighbor_lists()
{
    neighbor_3_square_nonempty.reset();
    base_coord_nonempty.reset();
    cuExecute3D(dim3(grid_dim, grid_dim, grid_dim), init_neigbor_list_size_kernel, *this);
    neighbor_3_square_nonempty.remove_zeros();
    base_coord_nonempty.remove_zeros();
    neighbor_3_square_list.reset();
    neighbor_4_square_list.reset();
    cuExecuteBlock(neighbor_3_square_nonempty.size(), 32, fill_neighbor_list_3_kernel, *this);
    cuExecuteBlock(base_coord_nonempty.size(), 64, fill_neighbor_list_4_kernel, *this);
}

}  // namespace pppm