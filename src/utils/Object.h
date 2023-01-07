#pragma once
#ifndef M_OBJECT_H
#define M_OBJECT_H
#include "array.h"
#include "objIO.h"

namespace pppm
{

class Object
{
public:
    virtual float GetLastFrameTime() = 0;
    virtual void UpdateUntil(float time) = 0;
    virtual GArr<float3>& GetVertices() = 0;
    virtual GArr<int3>& GetSurfaces() = 0;
    virtual ~Object(){};
protected:
    void LoadTetMesh_(const std::string &vertsPath, const std::string &tetPath,
        GArr<float3>& tetVertices, GArr<int3>& tetSurfaces, GArr<float3>& tetSurfaceNorms);
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

private:
    GArr<float3> vertices;
    GArr<int3> surfaces;
};
}
#endif // M_OBJECT_H