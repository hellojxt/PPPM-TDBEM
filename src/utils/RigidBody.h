#pragma once
#ifndef RIGID_BODY_H
#define RIGID_BODY_H

#include <string>
#include "objIO.h"
#include "array2D.h"
#include "macro.h"

namespace pppm
{
    struct Impulse
    {
        float currTime;
        int vertexID;
        float3 impulseVec;
    };

    struct ModalInfo
    {
    public:
        float coeff1 = 0.0f;
        float coeff2 = 0.0f;
        float coeff3 = 0.0f;
        float f = 0.0f;
        float q1 = 0;
        float q2 = 0;

        void SetCoeffs(float timestep, float eigenVal);
    };

    class RigidBody
    {
    public:
        RigidBody(const std::string &objPath, const std::string &displacementPath,
                  const std::string &implusePath, const std::string &eigenPath,
                  const std::string &tetPath);

        void TransformToNextFrame();
        Mesh mesh;
        GArr<float3> gpuVertices;

        CArr<float> frameTime;
        GArr<float3> tetVertices;
        GArr<int3> tetSurfaces;

        CArr<float3> translations;
        CArr<float4> rotations;

        CArr<CArr<float>> eigenVecs;
        GArr2D<float> modalMatrix;
        CArr<float> eigenVals;
        CArr<float> cpuQ;
        GArr<float> gpuQ;
        CArr<ModalInfo> modalInfos;

        GArr<float3> vertAccs;
        GArr<float> surfaceAccs;

        int t;
        int impulseTimeStamp;
        CArr<Impulse> impulses;

    private:
        void LoadDisplacement_(const std::string &displacementPath);
        void LoadImpulses_(const std::string &);
        void LoadTetMesh_(const std::string &);
        void LoadEigen_(const std::string &);
        void InitIIR_();
        void CalculateIIR_();
    };

}

#endif // RIGID_BODY_H