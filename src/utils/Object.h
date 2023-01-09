#pragma once
#ifndef M_OBJECT_H
#define M_OBJECT_H
#include "array.h"
#include "objIO.h"
#include "helper_math.h"

namespace pppm
{

__device__ inline float3 rotate(const float4 q, const float3 v);

__global__ void Transform(GArr<float3> vertices, GArr<float3> standard_vertices,
                          float3 translation, float4 rotation);

__global__ void Fill(float* arr, float num, size_t size);

class Object
{
public:
    virtual float GetLastFrameTime() = 0;
    virtual void UpdateUntil(float time) = 0;
    virtual GArr<float3>& GetVertices() = 0;
    virtual GArr<int3>& GetSurfaces() = 0;
    virtual void SubmitAccelerations(float* begin) = 0;
    virtual ~Object(){};
protected:
    void LoadTetMesh_(const std::string &vertsPath, const std::string &tetPath,
                      GArr<float3> &tetVertices, GArr<int3> &tetSurfaces, 
                      GArr<float3> &tetSurfaceNorms);
    void LoadMotion_(const std::string &path, CArr<float3> &translations,
                     CArr<float4> &rotations, CArr<float>& frameTime);
};

class ManualObject : public Object
{
public:
    ManualObject(const std::string& dir) {
        Mesh mesh = Mesh::loadOBJ(dir + "/mesh.obj");
        vertices.assign(mesh.vertices);
        surfaces.assign(mesh.triangles);
        return;
    }
    virtual float GetLastFrameTime() override { return FLT_MAX; }
    virtual void UpdateUntil(float time) override { return; } // no motion.
    virtual GArr<float3> &GetVertices() override { return vertices; };
    virtual GArr<int3>& GetSurfaces() override { return surfaces; };
    virtual void SubmitAccelerations(float* begin) override { 
        cudaMemset(begin, 0, surfaces.size() * sizeof(float));
        return;
    }
private:
    GArr<float3> vertices;
    GArr<int3> surfaces;
};

class AudioObject : public Object
{
public:
    AudioObject(const std::string& dir) {
        Mesh mesh = Mesh::loadOBJ(dir + "/mesh.obj");
        vertices.assign(mesh.vertices);
        standardVertices.assign(vertices);
        surfaces.assign(mesh.triangles);
        LoadMotion_(dir + "/motion.txt", translations, rotations, frameTime);
        LoadAccs_(dir + "/accs.txt");
        return;
    }
    void SetSampleRate(float sampleRate) { sampleTime = 1 / sampleRate; return; }
    virtual float GetLastFrameTime() override { return frameTime.last(); }
    virtual void UpdateUntil(float time) override { 
        AnimationUpdateUntil_(time);
        AccelerationUpdateUntil_(time);
    } // no motion.
    virtual GArr<float3> &GetVertices() override { return vertices; };
    virtual GArr<int3>& GetSurfaces() override { return surfaces; };
    virtual void SubmitAccelerations(float* begin) override {
        cuExecute(surfaces.size() / 64, Fill, begin,
                  accelerations[accTimeStep - 1], surfaces.size());
        return;
    }

private:
    GArr<float3> vertices;
    GArr<float3> standardVertices;
    GArr<int3> surfaces;
    CArr<float> accelerations;
    CArr<float3> translations;
    CArr<float4> rotations;
    CArr<float> frameTime;
    int animationTimeStep = 0;
    int accTimeStep = 0;
    float sampleTime = 0.0f;

    void AnimationUpdateUntil_(float time)
    {
        while(frameTime[animationTimeStep] < time)
        {
            cuExecute(vertices.size(), Transform, vertices, standardVertices,
                      translations[animationTimeStep], rotations[animationTimeStep]);
            animationTimeStep++;
        }
        return;
    };
    
    void AccelerationUpdateUntil_(float time)
    {
        while(sampleTime * accTimeStep < time)
        {
            accTimeStep++;
        }
        return;
    };

    void LoadAccs_(const std::string& path);
};

}
#endif // M_OBJECT_H