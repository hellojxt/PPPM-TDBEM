#pragma once
#ifndef OBJECT_COLLECTION_H
#define OBJECT_COLLECTION_H
#include "Object.h"
#include <memory>
#include <any>
#include <filesystem>
#include <vector>
#include <string>

namespace pppm
{
struct ObjectInfo
{
    enum class SoundType{
        Modal,
        Manual,
        Audio
    } type;
    size_t verticesOffset;
    size_t surfacesOffset;
};

class ObjectCollection
{
public:
    ObjectCollection(const std::filesystem::path &dir,
                     const std::vector<std::pair<std::string, ObjectInfo::SoundType>> &objects,
                     const std::vector<std::any>& additionalParameters);

    void export_mesh(const std::string &output_path);
    void export_mesh_sequence(const std::string &output_path);
    void export_modes(const std::string &output_path);
    void UpdateMesh();
    void UpdateAcc();
    void UpdateTimeStep();
    void FixMesh(float);
    BBox GetBBox();

    CArr<ObjectInfo> objectInfos;
    std::vector<std::unique_ptr<Object>> objects;

    GArr<float3> tetVertices;
    GArr<int3> tetSurfaces;
    GArr<float3> tetSurfaceNorms;
    GArr<float> surfaceAccs;

    float timeStep;
    std::filesystem::path rootDir;
private:
    void LoadObjectMesh_(int objID);
};

}
#endif // OBJECT_COLLECTION_H