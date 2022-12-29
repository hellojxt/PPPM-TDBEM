#pragma once

#include <string>
#include "objIO.h"
#include "array2D.h"
#include "macro.h"
#include <filesystem>
#include "material.h"
#include <deque>

namespace pppm
{

#define CONTACT_TIME_SCALE 0.00005f

struct Impulse
{
        float currTime;
        int vertexID;
        float3 impulseVec;
        float impulseRelativeSpeed;
};

class ImpulseSine
{
    public:
        Impulse imp;
        ImpulseSine(Impulse imp_) : imp(imp_) {}
        inline float amp(float time)
        {
            float tau = (CONTACT_TIME_SCALE / pow(abs(imp.impulseRelativeSpeed), 0.2));
            float signal = sin(M_PI * (time - imp.currTime) / tau) / tau;
            if (isnan(signal))
            {
                printf("tau: %f, time: %f, currTime: %f, sin: %f, amp: %f\n", tau, time, imp.currTime,
                       sin(M_PI * (time - imp.currTime) / tau), signal);
                printf("pow(imp.impulseRelativeSpeed, 0.2): %f\n", pow(imp.impulseRelativeSpeed, 0.2));
            }
            return signal;
        }
        inline bool dead(float time) { return time - imp.currTime > CONTACT_TIME_SCALE; }
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
        float q3 = 0;

        void SetCoeffs(float timestep, float eigenVal, MaterialParameters &material);
};

class RigidBody
{
    public:
        RigidBody(const std::string &data_dir,
                  float sample_rate_,
                  std::string material_name,
                  float max_frequncy_ = 20000.0f)
        {
            sample_rate = sample_rate_;
            max_frequncy = max_frequncy_;
            material.set_parameters(material_name);
            std::string model_dir = data_dir + "/model/";
            std::string model_subdir = std::filesystem::directory_iterator(model_dir)->path();
            std::string objPath, eigenPath, tetPath, mapPath;
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
            obj_filename = objPath;
            std::string displacementPath = data_dir + "/animation/displace.bin";
            std::string implusePath = data_dir + "/shader/modalImpulses.txt";
            load_data(objPath, displacementPath, implusePath, eigenPath, tetPath);
        }

        void load_data(const std::string &objPath,
                       const std::string &displacementPath,
                       const std::string &implusePath,
                       const std::string &eigenPath,
                       const std::string &tetPath);
        void fix_mesh(float precision, std::string tmp_dir);
        void update_surf_matrix();
        void export_mesh_with_modes(const std::string &output_path);
        void export_surface_mesh(const std::string &output_path);
        void export_signal(const std::string &output_path,
                           float max_time);  // export the signal without considering the acoustics
        void export_mesh_sequence(const std::string &output_path);
        void export_surface_accs(const std::string &filename);
        void move_to_first_impulse();
        void animation_step();
        void audio_step();
        bool end() { return current_time <= frameTime.last(); }
        void clear()
        {
            gpuVertices.clear();
            frameTime.clear();
            tetVertices.clear();
            standardTetVertices.clear();
            tetSurfaces.clear();
            tetSurfaceNorms.clear();
            translations.clear();
            rotations.clear();
            eigenVecs.clear();
            modalMatrix.clear();
            modelMatrixSurf.clear();
            eigenVals.clear();
            cpuQ.clear();
            gpuQ.clear();
            modalInfos.clear();
            vertAccs.clear();
            surfaceAccs.clear();
            impulses.clear();
        }

        Mesh mesh;
        GArr<float3> gpuVertices;

        CArr<float> frameTime;
        GArr<float3> tetVertices;
        GArr<float3> standardTetVertices;
        GArr<int3> tetSurfaces;
        GArr<float3> tetSurfaceNorms;

        CArr<float3> translations;
        CArr<float4> rotations;

        CArr2D<float> eigenVecs;
        GArr2D<float> modalMatrix;
        GArr2D<float> modelMatrixSurf;
        CArr<float> eigenVals;
        CArr<float> cpuQ;
        GArr<float> gpuQ;
        CArr<ModalInfo> modalInfos;
        std::deque<ImpulseSine> currentImpulseSines;
        GArr<float3> vertAccs;
        GArr<float> surfaceAccs;
        bool mesh_is_updated;  // used after audio_step is called

        int animationTimeStamp;
        int impulseTimeStamp;
        float sample_rate;
        float current_time;
        float timestep;
        float max_frequncy;
        std::string obj_filename;
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
