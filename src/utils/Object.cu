#include "Object.h"
#include "helper_math.h"
#include <fstream>
namespace pppm
{
__device__ inline float3 rotate(const float4 q, const float3 v)
{
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

__global__ void Transform(GArr<float3> vertices, GArr<float3> standard_vertices, float3 translation, float4 rotation)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= vertices.size())
        return;
    vertices[id] = rotate(rotation, standard_vertices[id]) + translation;
    return;
}

__global__ void FillIf(float *arr, int *judge, float num, size_t size)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) * 64;
    for (int i = 0; i < 64 && id + i < size; i++)
    {
        if (judge[id + i] > 0)
        {
            arr[id + i] = num;
        }

        else
            arr[id + i] = 0;
    }
    return;
}

__global__ void FindNearestVertex(GArr<float3> origin_vertices,
                                  GArr<int3> origin_surfaces,
                                  GArr<float3> vertices,
                                  GArr<int3> surfaces,
                                  GArr<int> judge,
                                  GArr<int> selectedVertices)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= selectedVertices.size())
        return;

    int min_dist_id = -1;
    float min_dist = MAX_FLOAT;
    int v_id = selectedVertices[id];
    float3 curr_vert = origin_vertices[v_id];

    for (int f_id = 0; f_id < surfaces.size(); f_id++)
    {
        int3 face = surfaces[f_id];
        float3 verts[3] = {vertices[face.x], vertices[face.y], vertices[face.z]};
        float3 nearest_p = get_nearest_triangle_point(curr_vert, verts[0], verts[1], verts[2]);
        float dist = length(nearest_p - curr_vert);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_dist_id = f_id;
        }
    }
    // printf("cv id:%d, current vertex: (%f, %f, %f), selected face id: %d\n", selectedVertices[id], curr_vert.x,
    //        curr_vert.y, curr_vert.z, min_dist_id);
    atomicAdd(&judge[min_dist_id], 1);
    return;
}

std::pair<CArr<int3>, CArr<float3>> FindAllSurfaces(CArr<int4> &tetrahedrons, CArr<float3> &tetraVerts)
{
    CArr<int3> surfaceTriangles;
    CArr<float3> surfaceNorms;

    struct iVec3Hash
    {
            int operator()(const int3 &vec) const { return (vec.x << 20) + (vec.y << 10) + vec.z; }
    };
    struct iVec3Eq
    {
            bool operator()(const int3 &vec1, const int3 &vec2) const
            {
                return vec1.x == vec2.x && vec1.y == vec2.y && vec1.z == vec2.z;
            }
    };
    struct TriInfo
    {
            int cnt = 0;
            int tetID = 0;
            int exceptVertID = 0;
    };

    std::unordered_map<int3, TriInfo, iVec3Hash, iVec3Eq> candidateTriangles;
    int3 currTri;
    int int3::*int3Members[] = {&int3::x, &int3::y, &int3::z};
    int int4::*int4Members[] = {&int4::x, &int4::y, &int4::z, &int4::w};

    for (int i = 0, size = tetrahedrons.size(); i < size; i++)
    {
        int4 &currVertIDs = tetrahedrons[i];
        for (int i0 = 0; i0 < 4; i0++)
        {
            for (int j = 0, k = 0; j < 4; j++)
            {
                if (j == i0)
                    continue;
                currTri.*(int3Members[k++]) = currVertIDs.*(int4Members[j]);
            }
            auto &currInfo = candidateTriangles[currTri];
            currInfo.cnt++;
            currInfo.tetID = i;
            currInfo.exceptVertID = i0;
        }
    }

    for (auto &candidateTriangle : candidateTriangles)
    {
        if (candidateTriangle.second.cnt != 1)
            continue;

        int3 currTri = candidateTriangle.first;

        int4 &currTet = tetrahedrons[candidateTriangle.second.tetID];
        int exceptVertID = candidateTriangle.second.exceptVertID;

        float3 center = (tetraVerts[currTri.x] + tetraVerts[currTri.y] + tetraVerts[currTri.z]) / 3;
        float3 exceptVec = tetraVerts[currTet.*(int4Members[exceptVertID])] - center;
        float3 e1 = tetraVerts[currTri.y] - tetraVerts[currTri.x], e2 = tetraVerts[currTri.z] - tetraVerts[currTri.x];
        float3 normVec = normalize(cross(e1, e2));
        if (dot(normVec, exceptVec) > 0)
        {
            std::swap(currTri.y, currTri.z);
            normVec = -normVec;
        }
        surfaceTriangles.pushBack(currTri);
        surfaceNorms.pushBack(normVec);
    }

    return {surfaceTriangles, surfaceNorms};
}

void Object::LoadTetMesh_(const std::string &vertsPath,
                          const std::string &tetPath,
                          GArr<float3> &tetVertices,
                          GArr<int3> &tetSurfaces,
                          GArr<float3> &tetSurfaceNorms)
{
    std::ifstream f_verts(vertsPath);
    CArr<float3> cpuTetVertices;
    float3 vert;
    if (!f_verts.good())
    {
        LOG_ERROR("Fail to load tet mesh file.");
        std::exit(EXIT_FAILURE);
    }
    std::string line;
    while (getline(f_verts, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        iss >> vert.x >> vert.y >> vert.z;
        cpuTetVertices.pushBack(vert);
    }
    f_verts.close();

    std::ifstream f_tet(tetPath);
    CArr<int4> tetrahedrons;
    float4 tet;
    if (!f_tet.good())
    {
        LOG_ERROR("Fail to load tet mesh file.");
        std::exit(EXIT_FAILURE);
    }
    while (getline(f_tet, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        iss >> tet.x >> tet.y >> tet.z >> tet.w;
        int idxs[4] = {F2I(tet.x), F2I(tet.y), F2I(tet.z), F2I(tet.w)};
        std::sort(idxs, idxs + 4);
        tetrahedrons.pushBack(make_int4(idxs[0], idxs[1], idxs[2], idxs[3]));
    }

    tetVertices.assign(cpuTetVertices);
    auto [surfaceTris, surfaceNorms] = FindAllSurfaces(tetrahedrons, cpuTetVertices);
    tetSurfaces.assign(surfaceTris);
    tetSurfaceNorms.assign(surfaceNorms);
    return;
}

void Object::LoadMotion_(const std::string &path,
                         CArr<float3> &translations,
                         CArr<float4> &rotations,
                         CArr<float> &frameTime)
{
    std::ifstream fin(path);

    float currTime = 0;
    float3 currTranslation;
    float4 currRotation;
    if (!fin.good())
    {
        LOG_ERROR("Fail to load displacement file at " << path << "\n");
        return;
    }
    std::string line;
    while (getline(fin, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        iss >> currTime >> currTranslation.x >> currTranslation.y >> currTranslation.z >> currRotation.x >>
            currRotation.y >> currRotation.z >> currRotation.w;
        translations.pushBack(currTranslation);
        rotations.pushBack(currRotation);
        frameTime.pushBack(currTime);
    }
    return;
};

void AudioObject::LoadAccs_(const std::string &path)
{
    std::ifstream fin(path);
    if (!fin.good())
    {
        LOG_ERROR("Fail to load acceleration file at " << path << "\n");
        return;
    }
    float currAcc = 0;
    while (true)
    {
        fin >> currAcc;
        if (!fin.good())
        {
            assert(fin.eof());
            break;
        }
        accelerations.pushBack(currAcc);
    }
    return;
}

void AudioObject::LoadCover_(const std::string &path)
{
    std::ifstream fin(path);
    if (!fin.good())
    {
        LOG_ERROR("Fail to load selected vertices file at " << path << "\n");
        return;
    }
    CArr<int> cpuCoverVertices;
    int currVertex = 0;
    while (true)
    {
        fin >> currVertex;
        cpuCoverVertices.pushBack(currVertex);
        if (!fin.good())
        {
            assert(fin.eof());
            break;
        }
    }
    selectedVertices.assign(cpuCoverVertices);
    return;
}

}  // namespace pppm