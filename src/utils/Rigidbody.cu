#include "RigidBody.h"
#include <fstream>
#include <cmath>
#include <unordered_map>

namespace pppm
{
    void ModalInfo::SetCoeffs(float timestep, float eigenVal)
    {
        const float alpha = 30, beta = 1e-6f;
        const float rho = 1.29f;

        float lambda = eigenVal;

        float omega = std::sqrt(lambda);
        float ksi = (alpha + beta * lambda) / (2 * omega);
        float omega_prime = omega * std::sqrt(1 - ksi * ksi);
        float epsilon = std::exp(-ksi * omega * timestep);
        float sqrEpsilon = epsilon * epsilon;
        float theta = omega_prime * timestep;
        float gamma = std::asin(ksi);

        coeff1 = 2 * epsilon * std::cos(theta);
        coeff2 = -sqrEpsilon;

        float coeff3_item1 = epsilon * std::cos(theta + gamma);
        float coeff3_item2 = sqrEpsilon * std::cos(2 * theta + gamma);
        coeff3 = 2 * (coeff3_item1 - coeff3_item2) * omega_prime * rho / (3 * omega);
        return;
    }

    RigidBody::RigidBody(const std::string &objPath,
                         const std::string &displacementPath,
                         const std::string &implusePath,
                         const std::string &eigenPath,
                         const std::string &tetPath)
        : mesh(Mesh::loadOBJ(objPath)), t(0), impulseTimeStamp(0)
    {
        gpuVertices.assign(mesh.vertices);
        LoadDisplacement_(displacementPath);
        LoadImpulses_(implusePath);
        LoadTetMesh_(tetPath);
        LoadEigen_(eigenPath);
        InitIIR_();
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
            cpuTranslations.pushBack(make_float3(currTranslation.x, currTranslation.y,
                                                 currTranslation.z));

            fin.read(reinterpret_cast<char *>(&currRotation), sizeof(double) * 4);
            assert(fin.good());
            cpuRotations.pushBack(make_float4(currRotation.x, currRotation.y,
                                              currRotation.z, currRotation.w));
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
            impulses.pushBack(Impulse{
                .currTime = static_cast<float>(currTime),
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
            int operator()(const int3 &vec) const
            {
                return (vec.x << 20) + (vec.y << 10) + vec.z;
            }
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

        auto FillFloat3Member = [&fin](float3 & num, float float3::*member)
            __attribute__((always_inline))
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
            assert(fin.good() && tetrahedrons[i].x < tetVertAmount &&
                   tetrahedrons[i].y < tetVertAmount &&
                   tetrahedrons[i].z < tetVertAmount &&
                   tetrahedrons[i].w < tetVertAmount);
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

        eigenVals.resize(modalSize);
        eigenVecs.resize(modalSize);
        for (auto &vec : eigenVecs)
        {
            vec.resize(vecDim);
        }

        for (int i = 0; i < modalSize; i++)
        {
            fin.read(reinterpret_cast<char *>(&temp), sizeof(double));
            assert(fin.good());
            eigenVals[i] = static_cast<float>(temp);
        }

        // store transpose of U
        for (int j = 0; j < vecDim; j++)
        {
            for (int i = 0; i < modalSize; i++)
            {
                fin.read((char *)&temp, sizeof(double));
                assert(fin.good());
                eigenVecs[i][j] = static_cast<float>(temp);
            }
        }
        return;
    };

    void RigidBody::InitIIR_()
    {
        size_t triNum = tetSurfaces.size();
        surfaceAccs.resize(triNum);

        float timestep = frameTime[1] - frameTime[0];
        int eigenNum = eigenVals.size();
        modalInfos.resize(eigenNum);
        for (int i = 0; i < eigenNum; i++)
        {
            modalInfos[i].SetCoeffs(timestep, eigenVals[i]);
        }
        cpuQ.resize(eigenNum);
        gpuQ.resize(eigenNum);

        vertAccs.resize(tetVertices.size());
        size_t size = eigenVecs[0].size();
        modalMatrix.resize(size, eigenNum);
        for (int i = 0; i < eigenNum; i++)
        {
            cudaMemcpy(modalMatrix.data.data() + size * i, eigenVecs[i].data(),
                       size * sizeof(float), cudaMemcpyHostToDevice);
        }
        return;
    }

    __global__ void GetVertAccs(GArr2D<float> gpuModalMatrix, GArr<float> gpuQ,
                                GArr<float3> gpuVertAccArr)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        int modalAmount = gpuModalMatrix.rows, vertDim = gpuModalMatrix.cols;
        int innerOffset = id % 4;
        int colOffset = (id / 4) * 64;

        for (int i = 0; i < modalAmount; i++)
        {
            float q = gpuQ[i];
            for (int j = innerOffset + colOffset;
                 j < colOffset + 64 && j < vertDim; j += 4)
            {
                gpuVertAccArr[j].x += gpuModalMatrix.index(i, j) * q;
                gpuVertAccArr[j + 1].y += gpuModalMatrix.index(i, j + 1) * q;
                gpuVertAccArr[j + 2].z += gpuModalMatrix.index(i, j + 2) * q;
            }
        }
        return;
    }

    __global__ void CollectAccToTri(GArr<float3> gpuVertAccArr, GArr<float> gpuTriAccArr,
                                    GArr<int3> gpuTriangleArr)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= gpuTriangleArr.size())
            return;

        int3 verts = gpuTriangleArr[id];
        float3 vertAccs{
            (gpuVertAccArr[verts.x].x + gpuVertAccArr[verts.y].x +
             gpuVertAccArr[verts.z].x) /
                3,
            (gpuVertAccArr[verts.x].y + gpuVertAccArr[verts.y].y +
             gpuVertAccArr[verts.z].y) /
                3,
            (gpuVertAccArr[verts.x].z + gpuVertAccArr[verts.y].z +
             gpuVertAccArr[verts.z].z) /
                3};

        gpuTriAccArr[id] = norm3d(vertAccs.x, vertAccs.y, vertAccs.z);
        return;
    }

    void RigidBody::CalculateIIR_()
    {

        float currTime = frameTime[t];
        int size = modalInfos.size();
        while (impulseTimeStamp < impulses.size() &&
               currTime >= impulses[impulseTimeStamp].currTime)
        {
            int id = impulses[impulseTimeStamp].vertexID;
            float3 currImpluse = impulses[impulseTimeStamp].impulseVec;
            for (int i = 0; i < size; i++)
            {
                const auto &currVec = eigenVecs[i];
                modalInfos[i].f += currVec[id * 3] * currImpluse.x +
                                   currVec[id * 3 + 1] * currImpluse.y +
                                   currVec[id * 3 + 2] * currImpluse.z;
            }
            impulseTimeStamp++;
        };

        for (int i = 0; i < size; i++)
        {
            auto &modalInfo = modalInfos[i];
            cpuQ[i] = modalInfo.coeff1 * modalInfo.q1 +
                      modalInfo.coeff2 * modalInfo.q2 +
                      modalInfo.coeff3 * modalInfo.f;
            modalInfo.q1 = modalInfo.q2, modalInfo.q2 = cpuQ[i];
            modalInfo.f = 0;
        }

        gpuQ.assign(cpuQ);
        auto vertDim = tetVertices.size();
        // let each thread do 16 work.
        cuExecute(vertDim / 16, GetVertAccs, modalMatrix, gpuQ, vertAccs);
        auto triNum = tetSurfaces.size();
        cuExecute(triNum, CollectAccToTri, vertAccs, surfaceAccs, tetSurfaces);
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
        return make_float3(
            2.0f * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
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
        vertices[id].x += translation.x, vertices[id].y += translation.y,
            vertices[id].z += translation.z;
        vertices[id] = rotate(rotation, vertices[id]);
        return;
    }

    void RigidBody::TransformToNextFrame()
    {
        cuExecute(gpuVertices.size(), Transform, gpuVertices, translations[t],
                  rotations[t]);
        CalculateIIR_();
        t++;
        if (t == translations.size())
            t = 0;
        return;
    }
}