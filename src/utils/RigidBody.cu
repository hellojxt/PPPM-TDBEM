#include "RigidBody.h"
#include <fstream>
#include <cmath>
#include <unordered_map>
#include "helper_math.h"
#include "progressbar.hpp"

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

CArr<int3> FindAllSurfaces(CArr<int4> &tetrahedrons)
{
    CArr<int3> surfaceTriangles;

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

    std::unordered_map<int3, int, iVec3Hash, iVec3Eq> candidateTriangles;
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
            candidateTriangles[currTri]++;
        }
    }

    for (auto &candidateTriangle : candidateTriangles)
    {
        if (candidateTriangle.second != 1)
            continue;

        surfaceTriangles.pushBack(candidateTriangle.first);
    }

    return surfaceTriangles;
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
        // fin.read(reinterpret_cast<char *>(&tempD), sizeof(double));
        // assert(fin.good());
        // cpuTetVertices[i].x = static_cast<float>(tempD);

        // fin.read(reinterpret_cast<char *>(&tempD), sizeof(double));
        // assert(fin.good());
        // cpuTetVertices[i].y = static_cast<float>(tempD);

        // fin.read(reinterpret_cast<char *>(&tempD), sizeof(double));
        // assert(fin.good());
        // cpuTetVertices[i].z = static_cast<float>(tempD);
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
    tetSurfaces.assign(FindAllSurfaces(tetrahedrons));
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
    surfaceAccs.resize(tetSurfaces.size());
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
    return;
}

__global__ void GetVertAccs(GArr2D<float> gpuModalMatrix, GArr<float> gpuQ, GArr<float3> gpuVertAccArr)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= gpuVertAccArr.size())
        return;
    int modalAmount = gpuModalMatrix.cols;
    gpuVertAccArr[id].x = 0;
    gpuVertAccArr[id].y = 0;
    gpuVertAccArr[id].z = 0;
    for (int i = 0; i < modalAmount; i++)
    {
        gpuVertAccArr[id].x += gpuModalMatrix(id * 3, i) * gpuQ[i];
        gpuVertAccArr[id].y += gpuModalMatrix(id * 3 + 1, i) * gpuQ[i];
        gpuVertAccArr[id].z += gpuModalMatrix(id * 3 + 2, i) * gpuQ[i];
    }
    return;
}

__global__ void CollectAccToTri(GArr<float3> gpuVertAccArr,
                                GArr<float> gpuTriAccArr,
                                GArr<int3> gpuTriangleArr,
                                GArr<float3> vertices)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= gpuTriAccArr.size())
        return;
    int3 verts = gpuTriangleArr[id];
    float3 vertAccs = (gpuVertAccArr[verts.x] + gpuVertAccArr[verts.y] + gpuVertAccArr[verts.z]) / 3;
    gpuTriAccArr[id] = length(vertAccs);
    return;
}

void RigidBody::Q_to_Accs_()
{
    cuExecute(vertAccs.size(), GetVertAccs, modalMatrix, gpuQ, vertAccs);
    cuExecute(surfaceAccs.size(), CollectAccToTri, vertAccs, surfaceAccs, tetSurfaces, tetVertices);
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

// export the surface mesh with all the modes.
void RigidBody::export_mesh_with_modes(const std::string &output_path)
{
    Mesh surfaceMesh(tetVertices, tetSurfaces);
    surfaceMesh.writeOBJ(output_path + "/surface.obj");
    std::ofstream fout(output_path + "/modes.txt");
    int modalAmount = modalMatrix.cols;
    printf("surface triangle amount: %d\n", tetSurfaces.size());
    printf("modalAmount: %d\n", modalAmount);
    progressbar bar(modalAmount);
    for (int i = 0; i < modalAmount; i++)
    {
        // print frequency
        std::cout << "frequency of mode " << i << ": " << sqrt(eigenVals[i] / material.density) / (2 * M_PI)
                  << std::endl;
    }
    std::cout << std::endl;
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
    std::ofstream fout(output_path + "/signal.txt");
    int frame_num = frameTime.last() / timestep;
    progressbar bar(frame_num);
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