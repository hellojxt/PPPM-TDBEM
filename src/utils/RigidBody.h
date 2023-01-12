#pragma once
#ifndef RIGID_BODY_H
#define RIGID_BODY_H

#include <string>
#include "objIO.h"
#include "array2D.h"
#include "macro.h"
#include <filesystem>
#include "material.h"
#include <deque>
#include "Object.h"

namespace pppm
{

#define CONTACT_TIME_SCALE 0.0005f

struct Impulse
{
        float currTime;
        int vertexID;
        float3 impulseVec;
        inline float amp(float time)
        {
            float tau = CONTACT_TIME_SCALE;
            float signal = sin(M_PI * (time - currTime) / tau);
            return signal;
        }
        inline bool dead(float time) { return time - currTime > CONTACT_TIME_SCALE; }
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

class RigidBody : public Object
{
    public:
        RigidBody(const std::string &data_dir, const std::string material_name, float max_frequncy_ = 20000.0f)
        {
            max_frequncy = max_frequncy_;
            material.set_parameters(material_name);
            load_data(data_dir);
        }

        void set_sample_rate(float sample_rate_)
        {
            sample_rate = sample_rate_;
            InitIIR_();
            current_time = 0;
            animationTimeStamp = 0;
        }

        virtual float GetLastFrameTime() override;
        virtual bool UpdateUntil(float time) override;
        virtual GArr<float3> &GetVertices() override;
        virtual GArr<int3> &GetSurfaces() override;
        virtual void SubmitAccelerations(float* begin) override;
        virtual float GetTimeStep() override { return timestep; };

        void separate_mode(int mode);
        void load_data(const std::string &data_dir);
        virtual void fix_mesh(float precision, std::string tmp_dir);
        void update_surf_matrix();
        void export_mesh_with_modes(const std::string &output_path);
        void export_surface_mesh(const std::string &output_path);
        void export_signal(const std::string &output_path,
                           float max_time);  // export the signal without considering the acoustics
        void export_mesh_sequence(const std::string &output_path);
        void export_surface_accs(const std::string &filename);
        void move_to_first_impulse();
        virtual BBox get_bbox();
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
            surfaceAccs.clear();
            impulses.clear();
        }

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
        std::deque<Impulse> currentImpulse;
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
        void LoadMotion_(const std::string &);
        void LoadImpulses_(const std::string &);
        void LoadTetMesh_(const std::string &, const std::string &);
        void LoadEigen_(const std::string &, const std::string &);
        void InitIIR_();
        void CalculateIIR_();
        void Q_to_Accs_();
};

}  // namespace pppm

#endif