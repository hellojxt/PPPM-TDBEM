#include "ghost_cell.h"
#include "ghost_cell_util.h"
#include "helper_math.h"
namespace pppm
{

__global__ void fill_in_nearest_kernel(GArr3D<CellInfo> cell_data, ParticleGrid grid, bool thin_shell)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = grid.grid_dim;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;

    float3 grid_center = grid.getCenter(x, y, z);
    CellInfo result;
    result.nearst_distance = MAX_FLOAT;
    float3 nearest_coeff;
    bool is_found = false;
    for (int dx = -3; dx <= 3; dx++)
        for (int dy = -3; dy <= 3; dy++)
            for (int dz = -3; dz <= 3; dz++)  // iterate over all the 3x3x3 grids
            {
                int3 coord = make_int3(x + dx, y + dy, z + dz);
                if (coord.x < 0 || coord.x >= grid_dim || coord.y < 0 || coord.y >= grid_dim || coord.z < 0 ||
                    coord.z >= grid_dim)
                    continue;
                auto &neighbor_list = grid.grid_face_list(coord);
                for (int i = 0; i < neighbor_list.size(); i++)
                {
                    int face_idx = neighbor_list[i];
                    int3 triID = grid.triangles[face_idx].indices;
                    float3 v0 = grid.vertices[triID.x];
                    float3 v1 = grid.vertices[triID.y];
                    float3 v2 = grid.vertices[triID.z];
                    float3 coeff = get_nearest_triangle_point_coeff(grid_center, v0, v1, v2);
                    float3 nearest_point = coeff.x * v0 + coeff.y * v1 + coeff.z * v2;
                    // float3 nearest_point = get_nearest_triangle_point(grid_center, v0, v1, v2);
                    float curr_len = length(grid_center - nearest_point);
                    if (curr_len < result.nearst_distance)
                    {
                        result.nearst_distance = curr_len;
                        result.nearst_point = nearest_point;
                        result.nearest_particle_idx = face_idx;
                        result.reflect_point = 2 * nearest_point - grid_center;
                        is_found = true;
                        nearest_coeff = coeff;
                    }
                }
            }
    if (is_found)
    {
        auto nearst_tri = grid.triangles[result.nearest_particle_idx];
        float3 normals[3];
        int v_ids[3] = {nearst_tri.indices.x, nearst_tri.indices.y, nearst_tri.indices.z};
        for (int i = 0; i < 3; i++)
        {
            int v_id = v_ids[i];
            float3 normal = make_float3(0, 0, 0);
            for (int j = 0; j < grid.vertex_neigbor_face_list[v_id].size(); j++)
            {
                int face_id = grid.vertex_neigbor_face_list[v_id][j];
                auto &neighbor_tri = grid.triangles[face_id];
                // calculate the angle between the two neighbor vertices of the one shared in the edge/corner
                // of the triangle
                int vs[2];
                int vs_num = 0;
                if (neighbor_tri.indices.x != v_id)
                {
                    vs[vs_num++] = neighbor_tri.indices.x;
                }
                if (neighbor_tri.indices.y != v_id)
                {
                    vs[vs_num++] = neighbor_tri.indices.y;
                }
                if (neighbor_tri.indices.z != v_id)
                {
                    vs[vs_num++] = neighbor_tri.indices.z;
                }
                float3 e1 = grid.vertices[vs[0]] - grid.vertices[v_id];
                float3 e2 = grid.vertices[vs[1]] - grid.vertices[v_id];
                float angle = acos(dot(e1, e2) / (length(e1) * length(e2)));
                normal += grid.triangles[face_id].normal * angle;
            }
            normals[i] = normalize(normal);
        }
        float3 normal_nearst_point =
            nearest_coeff.x * normals[0] + nearest_coeff.y * normals[1] + nearest_coeff.z * normals[2];
        float direction = dot(normal_nearst_point, grid_center - result.nearst_point);
        float3 d_vec = result.nearst_point - grid_center;
        // if (x == 30 && y == 30 && z == 44)
        // {
        //     printf(
        //         "direction: %f, normal: %f %f %f, grid_center: %f %f %f, nearest_point: %f %f %f, grid_center - "
        //         "nearest_point: %f %f %f\n",
        //         direction, normal_nearst_point.x, normal_nearst_point.y, normal_nearst_point.z, grid_center.x,
        //         grid_center.y, grid_center.z, result.nearst_point.x, result.nearst_point.y, result.nearst_point.z,
        //         grid_center.x - result.nearst_point.x, grid_center.y - result.nearst_point.y,
        //         grid_center.z - result.nearst_point.z);
        //     printf("d_vec: %f %f %f, grid_size: %f\n", d_vec.x, d_vec.y, d_vec.z, grid.grid_size);
        // }

        if (direction <= 0 || (thin_shell && fabs(d_vec.x) < (grid.grid_size / 2) &&
                               fabs(d_vec.y) < (grid.grid_size / 2) && fabs(d_vec.z) < (grid.grid_size / 2)))
            result.type = SOLID;
        else
            result.type = AIR;
    }
    else
    {
        result.type = UNKNOWN;
    }

    cell_data(x, y, z) = result;
    return;
}

__global__ void cell_classfication_kernel(GArr3D<CellInfo> cell_data,
                                          ParticleGrid grid,
                                          GArr3D<CellType> type_arr,
                                          GArr3D<int> ghost_idx_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = cell_data.size.x;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;
    CellType type = cell_data(x, y, z).type;
    int3 neighbor_list[6] = {make_int3(x + 1, y, z), make_int3(x - 1, y, z), make_int3(x, y + 1, z),
                             make_int3(x, y - 1, z), make_int3(x, y, z + 1), make_int3(x, y, z - 1)};
    if (type == SOLID)
    {
        for (int i = 0; i < 6; i++)
        {
            int3 coord = neighbor_list[i];
            if (coord.x < 0 || coord.x >= grid_dim || coord.y < 0 || coord.y >= grid_dim || coord.z < 0 ||
                coord.z >= grid_dim)
                continue;
            if (cell_data(coord).type == AIR)
            {
                type = GHOST;
            }
        }
    }
    type_arr(x, y, z) = type;
    ghost_idx_arr(x, y, z) = (type == GHOST);
    // if (x == 30 && y == 30 && z == 44)
    // {
    //     print_type(type);
    // }
}

__global__ void post_cell_classification_kernel(ParticleGrid grid, GArr3D<CellType> type_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= grid.grid_dim || y < 0 || y >= grid.grid_dim)
        return;
    auto current_type = AIR;
    auto first_found = false;
    for (int z = 0; z < grid.grid_dim; z++)
    {
        if (!first_found && type_arr(x, y, z) != UNKNOWN)
        {
            first_found = true;
            current_type = type_arr(x, y, z);
            for (int k = 0; k < z; k++)
                type_arr(x, y, k) = current_type;
            continue;
        }
        auto &t = type_arr(x, y, z);
        if (t == SOLID || t == GHOST)
        {
            current_type = SOLID;
        }
        else if (t == AIR)
        {
            current_type = AIR;
        }
        else
        {
            type_arr(x, y, z) = current_type;
        }
    }
}

__global__ void apply_cell_type_kernel(GArr3D<CellInfo> cell_data,
                                       ParticleGrid grid,
                                       GArr3D<CellType> type_arr,
                                       GArr3D<int> ghost_idx_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // here we assume grid_dim is (a, a, a).
    int grid_dim = cell_data.size.x;
    if (x < 0 || x >= grid_dim || y < 0 || y >= grid_dim || z < 0 || z >= grid_dim)
        return;
    cell_data(x, y, z).type = type_arr(x, y, z);
    if (type_arr(x, y, z) == GHOST)
    {
        cell_data(x, y, z).ghost_idx = ghost_idx_arr(x, y, z) - 1;
    }
    else
    {
        cell_data(x, y, z).ghost_idx = -1;
    }
}

int fill_cell_data(ParticleGrid grid, GArr3D<CellInfo> cell_data, bool thin_shell)
{
    auto grid_dim_3D = make_int3(grid.grid_dim, grid.grid_dim, grid.grid_dim);
    cuExecute3D(grid_dim_3D, fill_in_nearest_kernel, cell_data, grid, thin_shell);
    GArr3D<CellType> type_arr;
    GArr3D<int> ghost_idx_arr;
    type_arr.resize(grid_dim_3D);
    ghost_idx_arr.resize(grid_dim_3D);
    cuExecute3D(grid_dim_3D, cell_classfication_kernel, cell_data, grid, type_arr, ghost_idx_arr);
    cuExecute2D(dim2(grid.grid_dim, grid.grid_dim), post_cell_classification_kernel, grid, type_arr);
    thrust::inclusive_scan(thrust::device, ghost_idx_arr.begin(), ghost_idx_arr.end(), ghost_idx_arr.begin());
    cuExecute3D(grid_dim_3D, apply_cell_type_kernel, cell_data, grid, type_arr, ghost_idx_arr);
    int ghost_cell_num = ghost_idx_arr.data.last_item();
    type_arr.clear();
    ghost_idx_arr.clear();
    return ghost_cell_num;
}

};  // namespace pppm