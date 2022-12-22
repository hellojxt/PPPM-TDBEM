#pragma once

#include <string>
#include "objIO.h"
#include "array2D.h"
#include "macro.h"
#include <filesystem>
#include "material.h"

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

        void SetCoeffs(float timestep, float eigenVal, MaterialParameters &material);
};

class RigidBody
{
    public:
        RigidBody(const std::string &data_dir, float sample_rate_, std::string material_name)
        {
            sample_rate = sample_rate_;
            material.set_parameters(material_name);
            std::string model_dir = data_dir + "/model/";
            std::string model_subdir = std::filesystem::directory_iterator(model_dir)->path();
            std::string objPath, eigenPath, tetPath;
            for (const auto &entry : std::filesystem::directory_iterator(model_subdir))
            {
                if (entry.path().extension() == ".obj")
                {
                    objPath = entry.path();
                }
                if (entry.path().extension() == ".modes")
                {
                    eigenPath = entry.path();
                }
                if (entry.path().extension() == ".tet")
                {
                    tetPath = entry.path();
                }
            }
            std::string displacementPath = data_dir + "/animation/displace.bin";
            std::string implusePath = data_dir + "/shader/modalImpulses.txt";
            load_data(objPath, displacementPath, implusePath, eigenPath, tetPath);
        }

        void load_data(const std::string &objPath,
                       const std::string &displacementPath,
                       const std::string &implusePath,
                       const std::string &eigenPath,
                       const std::string &tetPath);

        void export_mesh_with_modes(const std::string &output_path);
        void export_signal(const std::string &output_path);  // export the signal without considering the acoustics

        void animation_step();
        void audio_step();
        bool end() { return current_time <= frameTime.last(); }

        Mesh mesh;
        GArr<float3> gpuVertices;

        CArr<float> frameTime;
        GArr<float3> tetVertices;
        GArr<int3> tetSurfaces;

        CArr<float3> translations;
        CArr<float4> rotations;

        CArr2D<float> eigenVecs;
        GArr2D<float> modalMatrix;
        CArr<float> eigenVals;
        CArr<float> cpuQ;
        GArr<float> gpuQ;
        CArr<ModalInfo> modalInfos;

        GArr<float3> vertAccs;
        GArr<float> surfaceAccs;

        int animationTimeStamp;
        int impulseTimeStamp;
        float sample_rate;
        float current_time;
        float timestep;
        CArr<Impulse> impulses;
        MaterialParameters material;

    private:
        void LoadDisplacement_(const std::string &displacementPath);
        void LoadImpulses_(const std::string &);
        void LoadTetMesh_(const std::string &);
        void LoadEigen_(const std::string &);
        void InitIIR_();
        void CalculateIIR_();
        void Q_to_Accs_();
};

}  // namespace pppm
