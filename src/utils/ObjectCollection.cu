#include "ObjectCollection.h"
#include "RigidBody.h"
#include "objIO.h"
#include "progressbar.h"

namespace pppm
{
ObjectCollection::ObjectCollection(const std::filesystem::path& dir,
    const std::vector<std::pair<std::string, ObjectInfo::SoundType>>& objectNames,
    const std::vector<std::any>& additionalParameters)
{
    size_t verticesSize = 0, surfacesSize = 0;
    for(int i = 0 ; i < objectNames.size(); i++)
    {
        auto& objectName = objectNames[i];
        if(objectName.second == ObjectInfo::SoundType::Modal)
        {
            const auto parameter = std::any_cast<std::string>(additionalParameters[i]);
            objects.push_back(
                std::make_unique<RigidBody>((dir / objectName.first).string(), parameter));
        }
        else if(objectName.second == ObjectInfo::SoundType::Manual)
        {

        }
        else if(objectName.second == ObjectInfo::SoundType::Audio)
        {

        }
        auto& currObject = objects.back();
        auto currObjectVerticesSize = currObject->GetVertices().size(),
            currObjectSurfacesSize = currObject->GetSurfaces().size();
        objectInfos.pushBack(ObjectInfo{ objectName.second, verticesSize, surfacesSize});
        verticesSize += currObjectVerticesSize, surfacesSize += currObjectSurfacesSize;
    }
    tetVertices.resize(verticesSize), tetSurfaces.resize(surfacesSize);
    return;
}

void ObjectCollection::export_mesh_sequence(const std::string &output_path)
{
    CHECK_DIR(output_path);
    float animation_export_timestep = 1.0f / 60.0f;
    float lastFrameTime = FLT_MAX;
    for(auto& object : objects)
    {
        lastFrameTime = object->GetLastFrameTime();
    }
    int frame_num = lastFrameTime / animation_export_timestep;
    progressbar bar(frame_num - 1, "exporting mesh sequence");
    
    auto rawTetVerticesPtr = tetVertices.data();
    auto rawTetSurfacesPtr = tetSurfaces.data();

    for (int i = 1; i < frame_num; i++)
    {
        for(int j = 0; j < objects.size(); j++)
        {
            auto& object = objects[j];
            object->UpdateUntil(i * animation_export_timestep);
            auto& objectVertices = object->GetVertices();
            auto& objectSurfaces = object->GetSurfaces();

            auto& objectInfo = objectInfos[j];
            cudaMemcpy(rawTetVerticesPtr + objectInfo.verticesOffset, objectVertices.data(), 
                objectVertices.size(), cudaMemcpyDeviceToDevice);
            cudaMemcpy(rawTetSurfacesPtr + objectInfo.surfacesOffset, objectSurfaces.data(),
                objectSurfaces.size(), cudaMemcpyDeviceToDevice);
        }
        
        Mesh surfaceMesh(tetVertices, tetSurfaces);
        surfaceMesh.remove_isolated_vertices();
        surfaceMesh.writeOBJ(output_path + "/surface_" + std::to_string(i) + ".obj");
        
        bar.update();
    }
    std::cout << std::endl;
    return;
}

}