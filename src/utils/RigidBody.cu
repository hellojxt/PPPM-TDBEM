#include "RigidBody.h"
#include <fstream>
#include <cmath>
#include <unordered_map>
#include "helper_math.h"
#include "progressbar.h"
#include "particle_grid.h"

namespace pppm
{
void ModalInfo::SetCoeffs(float timestep, float eigenVal, MaterialParameters &material)
{
    float lambda = eigenVal / material.density;
    float omega = std::sqrt(lambda);
    float ksi = (material.alpha + material.beta * lambda) / (2 * omega);
    float omega_prime = omega * std::sqrt(1 - ksi * ksi);
    float epsilon = std::exp(-ksi * omega * timestep);
    float sqrEpsilon = epsilon * epsilon;
    float theta = omega_prime * timestep;
    float gamma = std::asin(ksi);
    coeff1 = 2 * epsilon * std::cos(theta);
    coeff2 = -sqrEpsilon;

    float coeff3_item1 = epsilon * std::cos(theta + gamma);
    float coeff3_item2 = sqrEpsilon * std::cos(2 * theta + gamma);
    coeff3 = 2 * (coeff3_item1 - coeff3_item2) * omega_prime / (3 * omega);
    return;
}

void RigidBody::load_data(const std::string &objPath,
                          const std::string &displacementPath,
                          const std::string &implusePath,
                          const std::string &eigenPath,
                          const std::string &tetPath)
{
    mesh = Mesh::loadOBJ(objPath);
    impulseTimeStamp = 0;
    gpuVertices.assign(mesh.vertices);
    LoadDisplacement_(displacementPath);
    LoadImpulses_(implusePath);
    LoadTetMesh_(tetPath);
    LoadEigen_(eigenPath);
    InitIIR_();
    current_time = 0;
    animationTimeStamp = 0;
}

void RigidBody::LoadDisplacement_(const std::string &displacementPath)
{
    std::ifstream fin(displacementPath, std::ios::binary);

    double currTime = 0;
    double3 currTranslation;
    double4 currRotation;
    int termination = 0;
    int cnt = 0;
    int omit = 0;
    CArr<float3> cpuTranslations;
    CArr<float4> cpuRotations;

    while (true)
    {
        fin.read(reinterpret_cast<char *>(&currTime), sizeof(double));

        if (fin.eof())
            break;
        else if (!fin.good())
        {
            LOG_ERROR("Fail to load displacement file.\n");
            std::exit(EXIT_FAILURE);
        }

        frameTime.pushBack(static_cast<float>(currTime));

        fin.read(reinterpret_cast<char *>(&omit), sizeof(int));
        assert(fin.good());

        fin.read(reinterpret_cast<char *>(&currTranslation), sizeof(double) * 3);
        assert(fin.good());
        cpuTranslations.pushBack(make_float3(currTranslation.x, currTranslation.y, currTranslation.z));

        fin.read(reinterpret_cast<char *>(&currRotation), sizeof(double) * 4);
        assert(fin.good());
        cpuRotations.pushBack(make_float4(currRotation.x, currRotation.y, currRotation.z, currRotation.w));
        fin.read(reinterpret_cast<char *>(&termination), sizeof(int));
        assert(termination == -1);

        cnt++;
    }
    translations.assign(cpuTranslations);
    rotations.assign(cpuRotations);
    return;
};

void RigidBody::LoadImpulses_(const std::string &impulsePath)
{
    std::ifstream fin(impulsePath);
    double currTime, lastTime = 0.0;
    int vertexID, objID;
    double relativeSpeed;
    double3 impulse;
    char TorS, CorP;

    int cnt = 0;
    while (true)
    {
        fin >> currTime;
        if (fin.eof())
            break;
        else if (!fin.good())
        {
            LOG_ERROR("Fail to load impulse file.");
            std::exit(EXIT_FAILURE);
        }
        fin >> objID >> vertexID >> relativeSpeed >> impulse.x >> impulse.y >> impulse.z >> TorS >> CorP;

        // Here we assume the timestamp is monocratic.
        if (lastTime > currTime)
        {
            std::cout << cnt << " " << lastTime << " " << currTime << "\n";
            assert(false);
        };
        impulses.pushBack(Impulse{.currTime = static_cast<float>(currTime),
                                  .vertexID = vertexID,
                                  .impulseVec = make_float3(impulse.x, impulse.y, impulse.z)});
        lastTime = currTime;
        cnt++;
    }

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

void RigidBody::LoadTetMesh_(const std::string &tetPath)
{
    std::ifstream fin(tetPath, std::ios::binary);

    int tetVertAmount = 0;
    // skip alignment.
    fin.seekg(4);
    fin.read(reinterpret_cast<char *>(&tetVertAmount), sizeof(int));
    CArr<float3> cpuTetVertices;
    cpuTetVertices.resize(tetVertAmount);

    auto FillFloat3Member = [&fin](float3 & num, float float3::*member) __attribute__((always_inline))
    {
        double tempD = 0;
        fin.read(reinterpret_cast<char *>(&tempD), sizeof(double));
        assert(fin.good());
        num.*member = static_cast<float>(tempD);
    };

    for (int i = 0; i < tetVertAmount; i++)
    {
        FillFloat3Member(cpuTetVertices[i], &float3::x);
        FillFloat3Member(cpuTetVertices[i], &float3::y);
        FillFloat3Member(cpuTetVertices[i], &float3::z);
    }

    CArr<int4> tetrahedrons;
    int tetAmount = 0;
    fin.read(reinterpret_cast<char *>(&tetAmount), sizeof(int));
    tetrahedrons.resize(tetAmount);
    assert(tetAmount != 0);

    for (int i = 0; i < tetAmount; i++)
    {
        fin.read(reinterpret_cast<char *>(&tetrahedrons[i]), sizeof(int4));
        assert(fin.good() && tetrahedrons[i].x < tetVertAmount && tetrahedrons[i].y < tetVertAmount &&
               tetrahedrons[i].z < tetVertAmount && tetrahedrons[i].w < tetVertAmount);
        std::sort(reinterpret_cast<int *>(tetrahedrons.data() + i),
                  reinterpret_cast<int *>(tetrahedrons.data() + i + 1));
    }

    tetVertices.assign(cpuTetVertices);
    auto [surfaceTris, surfaceNorms] = FindAllSurfaces(tetrahedrons, cpuTetVertices);
    tetSurfaces.assign(surfaceTris);
    tetSurfaceNorms.assign(surfaceNorms);
}

void RigidBody::LoadEigen_(const std::string &eigenPath)
{
    std::ifstream fin(eigenPath, std::ios::binary);
    int vecDim, modalSize;
    double temp;

    fin.read(reinterpret_cast<char *>(&vecDim), sizeof(int));
    assert(fin.good());
    assert(vecDim == tetVertices.size() * 3);

    fin.read(reinterpret_cast<char *>(&modalSize), sizeof(int));
    assert(fin.good());

    CArr<float> eigenVals_;
    eigenVals_.resize(modalSize);
    int modalSize_ = modalSize;
    for (int i = 0; i < modalSize; i++)
    {
        fin.read(reinterpret_cast<char *>(&temp), sizeof(double));
        assert(fin.good());
        eigenVals_[i] = static_cast<float>(temp);
        if (std::sqrt(eigenVals_[i] / material.density) / (2 * M_PI) < 20000.0f)
            modalSize_ = i + 1;
    }

    eigenVals.resize(modalSize_);
    for (int i = 0; i < modalSize_; i++)
        eigenVals[i] = eigenVals_[i];
    eigenVecs.resize(vecDim, modalSize_);

    // store transpose of U
    for (int j = 0; j < modalSize; j++)
        for (int i = 0; i < vecDim; i++)
        {
            fin.read((char *)&temp, sizeof(double));
            assert(fin.good());
            if (j < modalSize_)
                eigenVecs(i, j) = static_cast<float>(temp);
        }
    return;
};

void RigidBody::InitIIR_()
{
    timestep = 1.0f / sample_rate;
    int eigenNum = eigenVals.size();

    modalInfos.resize(eigenNum);
    for (int i = 0; i < eigenNum; i++)
    {
        modalInfos[i].SetCoeffs(timestep, eigenVals[i], material);
    }
    cpuQ.resize(eigenNum);
    gpuQ.resize(eigenNum);
    vertAccs.resize(tetVertices.size());
    modalMatrix.assign(eigenVecs);
    update_surf_matrix();
    return;
}

__global__ void MatrixVertToTri(GArr2D<float> modalMatrix,
                                GArr2D<float> modelMatrixSurf,
                                GArr<int3> surfaces,
                                GArr<float3> surfaceNorms)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= surfaces.size())
        return;
    int3 verts = surfaces[id];
    for (int i = 0; i < modelMatrixSurf.size.y; i++)
    {
        float3 surf_vec = (make_float3(modalMatrix(verts.x * 3, i), modalMatrix(verts.x * 3 + 1, i),
                                       modalMatrix(verts.x * 3 + 2, i)) +
                           make_float3(modalMatrix(verts.y * 3, i), modalMatrix(verts.y * 3 + 1, i),
                                       modalMatrix(verts.y * 3 + 2, i)) +
                           make_float3(modalMatrix(verts.z * 3, i), modalMatrix(verts.z * 3 + 1, i),
                                       modalMatrix(verts.z * 3 + 2, i))) /
                          3;
        modelMatrixSurf(id, i) = dot(surf_vec, surfaceNorms[id]);
    }
    return;
}

void RigidBody::update_surf_matrix()
{
    surfaceAccs.resize(tetSurfaces.size());
    modelMatrixSurf.resize(tetSurfaces.size(), eigenVals.size());
    cuExecute(tetSurfaces.size(), MatrixVertToTri, modalMatrix, modelMatrixSurf, tetSurfaces, tetSurfaceNorms);
}

__global__ void update_surf_acc_kernel(GArr<float> gpuQ, GArr2D<float> modelMatrixSurf, GArr<float> surfaceAccs)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= surfaceAccs.size())
        return;
    float acc = 0;
    for (int i = 0; i < gpuQ.size(); i++)
    {
        acc += modelMatrixSurf(id, i) * gpuQ[i];
    }
    surfaceAccs[id] = acc;
}

void RigidBody::Q_to_Accs_()
{
    cuExecute(surfaceAccs.size(), update_surf_acc_kernel, gpuQ, modelMatrixSurf, surfaceAccs);
}

void RigidBody::CalculateIIR_()
{

    float currTime = current_time;
    int size = modalInfos.size();

    while (impulseTimeStamp < impulses.size() && currTime >= impulses[impulseTimeStamp].currTime)
    {
        int id = impulses[impulseTimeStamp].vertexID;
        float3 currImpluse = impulses[impulseTimeStamp].impulseVec;
        for (int i = 0; i < size; i++)
        {
            modalInfos[i].f += eigenVecs(id * 3, i) * currImpluse.x + eigenVecs(id * 3 + 1, i) * currImpluse.y +
                               eigenVecs(id * 3 + 2, i) * currImpluse.z;
        }
        impulseTimeStamp++;
    };

    for (int i = 0; i < size; i++)
    {
        auto &modalInfo = modalInfos[i];
        cpuQ[i] = modalInfo.coeff1 * modalInfo.q1 + modalInfo.coeff2 * modalInfo.q2 + modalInfo.coeff3 * modalInfo.f;
        modalInfo.q2 = modalInfo.q1, modalInfo.q1 = cpuQ[i];
        modalInfo.f = 0;
    }

    gpuQ.assign(cpuQ);
    Q_to_Accs_();
    return;
}

__device__ __forceinline__ float3 rotate(const float4 q, const float3 v)
{
    float t2 = q.x * q.y;
    float t3 = q.x * q.z;
    float t4 = q.x * q.w;
    float t5 = -q.y * q.y;
    float t6 = q.y * q.z;
    float t7 = q.y * q.w;
    float t8 = -q.z * q.z;
    float t9 = q.z * q.w;
    float t10 = -q.w * q.w;
    return make_float3(2.0f * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
                       2.0f * ((t4 + t6) * v.x + (t5 + t10) * v.y + (t9 - t2) * v.z) + v.y,
                       2.0f * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z);
}

__global__ void Transform(GArr<float3> vertices, float3 translation, float4 rotation)
{
    int id = threadIdx.y * blockIdx.x + threadIdx.x;
    if (id > vertices.size())
    {
        return;
    }
    vertices[id].x += translation.x, vertices[id].y += translation.y, vertices[id].z += translation.z;
    vertices[id] = rotate(rotation, vertices[id]);
    return;
}

void RigidBody::animation_step()
{
    cuExecute(gpuVertices.size(), Transform, gpuVertices, translations[animationTimeStamp],
              rotations[animationTimeStamp]);

    animationTimeStamp++;
}

void RigidBody::audio_step()
{
    if (frameTime[animationTimeStamp] < current_time)
    {
        animation_step();
    }
    current_time += timestep;
    CalculateIIR_();
}

__global__ void update_fixed_surface_kernel(ParticleGrid pg,
                                            GArr<float3> vertices,
                                            GArr<float3> vertices_fixed,
                                            GArr<int3> surfaces,
                                            GArr<float3> surfaceNorms)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= surfaces.size())
        return;

    int3 surface = surfaces[id];
    float3 vs[3] = {vertices_fixed[surface.x], vertices_fixed[surface.y], vertices_fixed[surface.z]};
    int indices[3] = {0, 0, 0};
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float3 v = vs[i];
        int3 coord = pg.getGridCoord(v);
        auto &v_neighbors = pg.neighbor_3_square_list(coord);
        float min_distance = MAXFLOAT;
        int min_v_id = -1;
        for (int j = 0; j < v_neighbors.size(); j++)
        {
            int v_id = v_neighbors[j];
            float3 v_neighbor = vertices[v_id];
            float distance = length(v - v_neighbor);
            if (distance < min_distance)
            {
                min_distance = distance;
                min_v_id = v_id;
            }
        }
        indices[i] = min_v_id;
    }
    surfaces[id] = make_int3(indices[0], indices[1], indices[2]);
    surfaceNorms[id] = normalize(cross(vs[1] - vs[0], vs[2] - vs[0]));
}

void RigidBody::fix_mesh(float precision, std::string tmp_dir)
{
    CHECK_DIR(tmp_dir);
    std::string python_src_dir = ROOT_DIR + std::string("python_scripts/");
    std::string python_src_name = "fix_mesh.py";
    export_surface_mesh(tmp_dir);
    std::string in_mesh_name = "surface.obj";
    std::string out_mesh_name = "surface_fixed.obj";
    std::string cmd = "docker run -it --rm -v " + tmp_dir + ":/models " + "-v " + python_src_dir + ":/scripts " +
                      "pymesh/pymesh /scripts/" + python_src_name + " --detail " + std::to_string(precision) +
                      " /models/" + in_mesh_name + " /models/" + out_mesh_name;
    // std::cout << cmd << std::endl;
    system(cmd.c_str());
    Mesh fixedMesh(tmp_dir + "/" + out_mesh_name);
    Mesh originMesh(tetVertices.cpu(), tetSurfaces.cpu());
    float3 min_pos = originMesh.bbox().min;
    float length = originMesh.bbox().width;
    int res = 64;
    float grid_size = length / res;
    ParticleGrid pg;
    pg.init(min_pos - grid_size * 2, length / res, res + 4, 0.1f);
    pg.set_only_vertices(originMesh.vertices);
    pg.construct_neighbor_lists();
    GArr<float3> gpuVerticesFixed(fixedMesh.vertices);
    tetSurfaces.assign(fixedMesh.triangles);
    tetSurfaceNorms.resize(fixedMesh.triangles.size());
    std::cout << "update fixed surface...";
    cuExecute(fixedMesh.triangles.size(), update_fixed_surface_kernel, pg, tetVertices, gpuVerticesFixed, tetSurfaces,
              tetSurfaceNorms);
    std::cout << "done" << std::endl;
    gpuVerticesFixed.clear();
    pg.clear();
    update_surf_matrix();
}

void RigidBody::export_surface_mesh(const std::string &output_path)
{
    CHECK_DIR(output_path);
    Mesh surfaceMesh(tetVertices, tetSurfaces);
    surfaceMesh.writeOBJ(output_path + "/surface.obj");
}

// export the surface mesh with all the modes.
void RigidBody::export_mesh_with_modes(const std::string &output_path)
{
    CHECK_DIR(output_path);
    export_surface_mesh(output_path);
    std::ofstream fout(output_path + "/modes.txt");
    int modalAmount = modalMatrix.cols;
    printf("surface triangle amount: %d\n", tetSurfaces.size());
    printf("modalAmount: %d\n", modalAmount);
    std::cout << "frequency of modes: " << std::endl;
    for (int i = 0; i < modalAmount; i++)
    {
        // print frequency
        std::cout << int(sqrt(eigenVals[i] / material.density) / (2 * M_PI)) << ", ";
    }
    std::cout << std::endl;
    progressbar bar(modalAmount, "exporting modes");
    for (int i = 0; i < modalAmount; i++)
    {
        bar.update();
        for (int j = 0; j < modalAmount; j++)
        {
            cpuQ[j] = 0;
        }
        cpuQ[i] = 1;
        gpuQ.assign(cpuQ);
        Q_to_Accs_();
        auto surf_accs = surfaceAccs.cpu();
        for (int j = 0; j < surf_accs.size(); j++)
        {
            fout << surf_accs[j] << " ";
        }
        fout << std::endl;
    }
    fout.close();
    std::cout << std::endl;
    return;
}

void RigidBody::export_signal(const std::string &output_path)
{
    CHECK_DIR(output_path);
    std::ofstream fout(output_path + "/signal.txt");
    int frame_num = frameTime.last() / timestep;
    progressbar bar(frame_num, "exporting signal");
    for (int i = 0; i < frame_num; i++)
    {
        bar.update();
        audio_step();
        float s = 0;
        for (int j = 0; j < cpuQ.size(); j++)
        {
            s += cpuQ[j];
        }
        fout << s << std::endl;
    }
    fout.close();
    std::cout << std::endl;
    return;
}

}  // namespace pppm