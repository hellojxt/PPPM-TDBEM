#pragma once
#include "array3D.h"
#include "helper_math.h"
#include "macro.h"
#include "fdtd.h"

namespace pppm
{
#define BUFFER_SIZE_FACE_NUM_PER_CELL 32
#define BUFFER_SIZE_NEIGHBOR_NUM_3_3_3 256
#define BUFFER_SIZE_NEIGHBOR_NUM_4_4_4 512

// attention: need to be used as reference for performance
template <typename T, int N>
class GridElementList
{
    public:
        int num;
        T list[N];
        CGPU_FUNC GridElementList() : num(0) {}
        GPU_FUNC inline int atomic_append(T i)
        {
            int index = atomicAdd(&num, 1);
#ifdef MEMORY_CHECK
            assert(index < N);
#endif
            list[index] = i;
            return index;
        }
        CGPU_FUNC T &operator[](int i) const
        {
#ifdef MEMORY_CHECK
            assert(i < num);
#endif
            return list[i];
        }
        CGPU_FUNC T &operator[](int i)
        {
#ifdef MEMORY_CHECK
            assert(i < num);
#endif
            return list[i];
        }
        CGPU_FUNC void set_size(int i)
        {
#ifdef MEMORY_CHECK
            assert(i <= N);
#endif
            num = i;
        }
        CGPU_FUNC int size() const { return num; }
        CGPU_FUNC void clear() { num = 0; }
};

using FaceList = GridElementList<int, BUFFER_SIZE_FACE_NUM_PER_CELL>;
using Neighbor3SquareList = GridElementList<int, BUFFER_SIZE_NEIGHBOR_NUM_3_3_3>;
using Neighbor4SquareList = GridElementList<int, BUFFER_SIZE_NEIGHBOR_NUM_4_4_4>;

struct NeighborNum
{
        int3 coord;
        int num;
        CGPU_FUNC bool is_zero() const { return num == 0; }
};

using CompactCoordArray = CompactIndexArray<NeighborNum>;

class Triangle
{
    public:
        int3 indices;
        float3 normal;
        float3 center;
        int3 grid_coord;
        int grid_index;
        int3 grid_base_coord;
        int grid_base_index;
        float area;
        CGPU_FUNC Triangle() {}
};

class ParticleGrid
{
    public:
        FDTD fdtd;
        float3 min_pos;
        float3 max_pos;
        float grid_size;
        int grid_dim;
        float delta_t;
        bool empty_grid = true;
        GArr<float3> vertices;  // vertex position
        // GArr<GridElementList<int, 32>> vertex_neigbor_face_list;  // vertex_neigbor_face_list(i) contain the index of
        // all the faces that contain the vertex i
        GArr<GridElementList<int, 32>> vertex_neigbor_face_list;
        GArr<int3> faces;                       // directly store the indices of vertices
        GArr<Triangle> triangles;               // store the triangle information
        GArr3D<FaceList> grid_face_list;        // grid_face_list(i,j,k) contain the index of all the faces in the cell
        GArr3D<FaceList> base_coord_face_list;  // base_coord_face_list(i,j,k) contain the index of the faces in
                                                // the interpolation cube (with 2*2*2 neighbor cell centers as vertices)
        CompactCoordArray base_coord_nonempty;  // contain the index of the nonempty base_coord_face_list(i,j,k)
        GArr3D<Neighbor3SquareList> neighbor_3_square_list;  // neighbor_3_square_list(i,j,k) contain the index of the
                                                             // neighbor triangles in the 3x3x3 square
        CompactCoordArray neighbor_3_square_nonempty;        // contain the index of the nonempty neighbors (3*3)
        GArr3D<Neighbor4SquareList> neighbor_4_square_list;  // neighbor_4_square_list(i,j,k) contain the index of the
                                                             // neighbor triangles in the 4x4x4 square

        void init(float3 min_pos_, float grid_size_, int grid_dim_, float delta_t_, int pml_width = 0)
        {
            min_pos = min_pos_;
            grid_size = grid_size_;
            grid_dim = grid_dim_;
            delta_t = delta_t_;
            max_pos = min_pos + make_float3(grid_dim, grid_dim, grid_dim) * grid_size;
            fdtd.init(grid_dim, grid_size, delta_t);              // 211
            grid_face_list.resize(grid_dim, grid_dim, grid_dim);  // 914
            grid_face_list.reset();
            base_coord_face_list.resize(grid_dim, grid_dim, grid_dim);  // 914
            base_coord_face_list.reset();
            neighbor_3_square_list.resize(grid_dim, grid_dim, grid_dim);  // 1824
            neighbor_3_square_list.reset();
            neighbor_4_square_list.resize(grid_dim, grid_dim, grid_dim);  // 3648
            neighbor_4_square_list.reset();
            base_coord_nonempty.reserve(grid_dim * grid_dim * grid_dim);
            neighbor_3_square_nonempty.reserve(grid_dim * grid_dim * grid_dim);
        }

        template <typename T1, typename T2>
        void set_mesh(T1 vertices_, T2 faces_)
        {
            vertices.assign(vertices_);
            faces.assign(faces_);
            triangles.resize(faces.size());
            grid_face_list.reset();
            vertex_neigbor_face_list.resize(vertices.size());
            construct_grid();
            update_vertex_neighbor_face_list();
            empty_grid = false;
        }

        void update_vertex_neighbor_face_list();

        template <typename T>
        void update_mesh(T vertices_)
        {
            vertices.assign(vertices_);
            construct_grid();
        }

        void clear()
        {
            fdtd.clear();
            vertices.clear();
            faces.clear();
            triangles.clear();
            vertex_neigbor_face_list.clear();
            grid_face_list.clear();
            base_coord_face_list.clear();
            neighbor_3_square_list.clear();
            neighbor_4_square_list.clear();
            base_coord_nonempty.clear();
            neighbor_3_square_nonempty.clear();
            empty_grid = true;
        }

        void construct_grid();

        void construct_vertices_grid();

        void construct_neighbor_lists();

        CGPU_FUNC inline float3 getCenter(int i, int j, int k) const
        {
            return make_float3((i + 0.5f) * grid_size, (j + 0.5f) * grid_size, (k + 0.5f) * grid_size) + min_pos;
        }
        CGPU_FUNC inline float3 getCenter(int3 c) const { return getCenter(c.x, c.y, c.z); }
        CGPU_FUNC inline float3 getCenter(uint3 c) const { return getCenter(c.x, c.y, c.z); }
        CGPU_FUNC inline int3 getGridCoord(float3 pos) const { return make_int3((pos - min_pos) / grid_size); }
        // lower left corner of the 2*2*2  neighbor cell centers, used for far field interpolation
        CGPU_FUNC inline int3 getGridBaseCoord(float3 pos) const
        {
            int3 c = getGridCoord(pos);
            float3 diff = pos - getCenter(c);
            return c - make_int3(diff.x < 0, diff.y < 0, diff.z < 0);
        }
        CGPU_FUNC inline int time_idx() const { return fdtd.t; }
};
}  // namespace pppm