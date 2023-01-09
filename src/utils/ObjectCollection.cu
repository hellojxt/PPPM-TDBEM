#include "ObjectCollection.h"
#include "RigidBody.h"
#include "objIO.h"
#include "progressbar.h"

namespace pppm
{
__global__ void AttachSurfaceIndicesOffset(int3* indices, size_t size, size_t offset)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= size)   
        return;

    indices[id].x += offset;
    indices[id].y += offset;
    indices[id].z += offset;

    return;
};

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
            auto ptr = std::make_unique<RigidBody>(dir / objectName.first, parameter);
            ptr->set_sample_rate(44100);
            objects.push_back(std::move(ptr));
        }
        else if(objectName.second == ObjectInfo::SoundType::Manual)
        {
            objects.push_back(
                std::make_unique<ManualObject>(dir / objectName.first)
            );
        }
        else if(objectName.second == ObjectInfo::SoundType::Audio)
        {
            auto ptr = std::make_unique<AudioObject>(dir / objectName.first);
            ptr->SetSampleRate(44100);
            objects.push_back(std::move(ptr));
        }
        auto& currObject = objects.back();
        auto currObjectVerticesSize = currObject->GetVertices().size(),
            currObjectSurfacesSize = currObject->GetSurfaces().size();
        objectInfos.pushBack(ObjectInfo{ objectName.second, verticesSize, surfacesSize});
        verticesSize += currObjectVerticesSize, surfacesSize += currObjectSurfacesSize;
    }
    tetVertices.resize(verticesSize), tetSurfaces.resize(surfacesSize);
    surfaceAccs.resize(surfacesSize);
    return;
}

void ObjectCollection::export_mesh_sequence(const std::string &output_path)
{
    CHECK_DIR(output_path);
    float animation_export_timestep = 1.0f / 60.0f;
    float lastFrameTime = FLT_MAX;
    for(auto& object : objects)
    {
        lastFrameTime = std::min(lastFrameTime, object->GetLastFrameTime());
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
                       objectVertices.size() * sizeof(float3), cudaMemcpyDeviceToDevice);
            cudaMemcpy(rawTetSurfacesPtr + objectInfo.surfacesOffset, objectSurfaces.data(),
                       objectSurfaces.size() * sizeof(int3), cudaMemcpyDeviceToDevice);
            cuExecute(objectSurfaces.size(), AttachSurfaceIndicesOffset,
                      rawTetSurfacesPtr + objectInfo.surfacesOffset, objectSurfaces.size(),
                      objectInfo.verticesOffset);
        }
        
        Mesh surfaceMesh(tetVertices, tetSurfaces);
        surfaceMesh.remove_isolated_vertices();
        surfaceMesh.writeOBJ(output_path + "/surface_" + std::to_string(i) + ".obj");
        
        bar.update();
    }
    std::cout << std::endl;
    return;
}

void ObjectCollection::export_modes(const std::string& output_path)
{
    CHECK_DIR(output_path);
    std::ofstream fout(output_path + "/modes.txt");
    std::cout << std::endl;
    for(int i = 0; i < objects.size(); i++)
    {
        auto& object = objects[i];
        auto& objectInfo = objectInfos[i];
        if(objectInfo.type == ObjectInfo::SoundType::Modal)
        {
            static_cast<RigidBody*>(object.get())->separate_mode(0);
        }
        object->SubmitAccelerations(surfaceAccs.data() + objectInfo.surfacesOffset);
    }
    auto surf_accs = surfaceAccs.cpu();
    for (int j = 0; j < surf_accs.size(); j++)
    {
        fout << surf_accs[j] << " ";
    }
    fout << std::endl;

    fout.close();
    return;
}

}