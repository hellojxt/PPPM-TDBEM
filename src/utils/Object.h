#pragma once
#ifndef M_OBJECT_H
#    define M_OBJECT_H
#    include "array.h"
#    include "objIO.h"
#    include "helper_math.h"
#    include "progressbar.h"

namespace pppm
{

struct bbox_minimum
{
        __device__ float3 operator()(const float3 &a, const float3 &b) const { return fminf(a, b); }
};

struct bbox_maximum
{
        __device__ float3 operator()(const float3 &a, const float3 &b) const { return fmaxf(a, b); }
};

__device__ inline float3 rotate(const float4 q, const float3 v);

__global__ void Transform(GArr<float3> vertices, GArr<float3> standard_vertices, float3 translation, float4 rotation);

__global__ void FillIf(float *arr, int *judge, float num, size_t size);

__global__ void FindNearestVertex(GArr<float3> origin_vertices,
                                  GArr<int3> origin_surfaces,
                                  GArr<float3> vertices,
                                  GArr<int3> surfaces,
                                  GArr<int> judge,
                                  GArr<int> selectedVertices);

struct ObjectState
{};

struct AudioObjectState : public ObjectState
{
    public:
        CArr<float3> vertices;
        int animationTimeStep;
        int accTimeStep;
};

class Object
{
    public:
        virtual float GetLastFrameTime() = 0;
        virtual float GetTimeStep() { return FLT_MAX; };
        virtual bool UpdateUntil(float time) = 0;
        virtual GArr<float3> &GetVertices() = 0;
        virtual GArr<int3> &GetSurfaces() = 0;
        virtual void SubmitAccelerations(float *begin) = 0;
        virtual bool WillUpdateMesh(float time) { return false; }
        virtual BBox get_bbox() = 0;
        virtual void fix_mesh(float precision, std::string tmp_dir) = 0;
        virtual ~Object(){};
        virtual void SaveState(ObjectState &state) = 0;
        virtual void LoadState(ObjectState &state) = 0;
        std::string name;

    protected:
        void LoadTetMesh_(const std::string &vertsPath,
                          const std::string &tetPath,
                          GArr<float3> &tetVertices,
                          GArr<int3> &tetSurfaces,
                          GArr<float3> &tetSurfaceNorms);
        void LoadMotion_(const std::string &path,
                         CArr<float3> &translations,
                         CArr<float4> &rotations,
                         CArr<float> &frameTime);
};

class ManualObject : public Object
{
    public:
        ManualObject(const std::string &dir)
        {
            Mesh mesh = Mesh::loadOBJ(dir + "/mesh.obj");
            vertices.assign(mesh.vertices);
            surfaces.assign(mesh.triangles);
            return;
        }
        virtual void SaveState(ObjectState &state) override{};
        virtual void LoadState(ObjectState &state) override{};
        virtual float GetLastFrameTime() override { return FLT_MAX; }
        virtual bool UpdateUntil(float time) override { return false; }  // no motion.
        virtual GArr<float3> &GetVertices() override { return vertices; };
        virtual GArr<int3> &GetSurfaces() override { return surfaces; };
        virtual void SubmitAccelerations(float *begin) override
        {
            cudaMemset(begin, 0, surfaces.size() * sizeof(float));
            return;
        }
        virtual void fix_mesh(float precision, std::string tmp_dir)
        {
            CHECK_DIR(tmp_dir);
            std::string python_src_dir = ROOT_DIR + std::string("python_scripts/");
            std::string python_src_name = "fix_mesh.py";
            std::string in_mesh_name = "mesh.obj";
            std::string out_mesh_name = "surface_fixed.obj";
            std::string cmd = "docker run -it --rm -v " + tmp_dir + ":/models " + "-v " + python_src_dir +
                              ":/scripts " + "pymesh/pymesh /scripts/" + python_src_name + " --detail " +
                              std::to_string(precision) + " /models/" + in_mesh_name + " /models/" + out_mesh_name;
            // std::cout << cmd << std::endl;
            system(cmd.c_str());
            Mesh fixedMesh(tmp_dir + "/" + out_mesh_name);
            vertices.assign(fixedMesh.vertices);
            surfaces.assign(fixedMesh.triangles);
        };

        virtual BBox get_bbox()
        {
            Mesh mesh(vertices.cpu(), surfaces.cpu());
            return mesh.bbox();
        };

        virtual ~ManualObject()
        {
            vertices.clear();
            surfaces.clear();
        }

    private:
        GArr<float3> vertices;
        GArr<int3> surfaces;
};

class AudioObject : public Object
{
    public:
        AudioObject(const std::string &dir)
        {
            Mesh mesh = Mesh::loadOBJ(dir + "/mesh.obj");
            vertices.assign(mesh.vertices);
            standardVertices.assign(vertices);
            surfaces.assign(mesh.triangles);
            LoadMotion_(dir + "/motion.txt", translations, rotations, frameTime);
            LoadAccs_(dir + "/accs.txt");
            LoadCover_(dir + "/selected_vertices.txt");
            return;
        }
        virtual void SaveState(ObjectState &state)
        {
            // Don't operator on state directly to prevent slicing.
            AudioObjectState &actualState = static_cast<AudioObjectState &>(state);
            actualState.vertices.assign(vertices);
            actualState.accTimeStep = accTimeStep;
            actualState.animationTimeStep = animationTimeStep;
            return;
        };
        virtual void LoadState(ObjectState &state)
        {
            AudioObjectState &actualState = static_cast<AudioObjectState &>(state);
            vertices.assign(actualState.vertices);
            accTimeStep = actualState.accTimeStep;
            animationTimeStep = actualState.animationTimeStep;
            return;
        };
        virtual float GetTimeStep() { return sampleTime; };
        void SetSampleRate(float sampleRate)
        {
            sampleTime = 1 / sampleRate;
            return;
        }
        virtual float GetLastFrameTime() override { return frameTime.isEmpty() ? FLT_MAX : frameTime.last(); }
        virtual bool UpdateUntil(float time) override
        {
            bool animationUpdate = AnimationUpdateUntil_(time);
            AccelerationUpdateUntil_(time);
            return animationUpdate;
        }
        virtual GArr<float3> &GetVertices() override { return vertices; };
        virtual GArr<int3> &GetSurfaces() override { return surfaces; };
        virtual void SubmitAccelerations(float *begin) override
        {
            if (accTimeStep >= accelerations.size())
            {
                cudaMemset(begin, 0, surfaces.size() * sizeof(float));
                return;
            }
            cuExecute(surfaces.size() / 64, FillIf, begin, selectedSurfacesJudgement.data(), accelerations[accTimeStep],
                      surfaces.size());
            return;
        }

        virtual void fix_mesh(float precision, std::string tmp_dir)
        {
            CHECK_DIR(tmp_dir);
            std::string python_src_dir = ROOT_DIR + std::string("python_scripts/");
            std::string python_src_name = "fix_mesh.py";
            std::string in_mesh_name = "mesh.obj";
            std::string out_mesh_name = "surface_fixed.obj";
            std::string cmd = "docker run -it --rm -v " + tmp_dir + ":/models " + "-v " + python_src_dir +
                              ":/scripts " + "pymesh/pymesh /scripts/" + python_src_name + " --detail " +
                              std::to_string(precision) + " /models/" + in_mesh_name + " /models/" + out_mesh_name;
            // std::cout << cmd << std::endl;
            system(cmd.c_str());

            Mesh fixedMesh(tmp_dir + "/" + out_mesh_name);

            if (accelerations.isEmpty())
            {
                vertices.assign(fixedMesh.vertices);
                surfaces.assign(fixedMesh.triangles);
                standardVertices.assign(vertices);
                return;
            }
            // all surfaces have.
            if (selectedVertices.isEmpty())
            {
                selectedSurfacesJudgement.resize(fixedMesh.triangles.size());
                thrust::fill(thrust::device, selectedSurfacesJudgement.data(),
                             selectedSurfacesJudgement.data() + selectedSurfacesJudgement.size(), 1);
            }
            else
            {
                auto gpuFixedVertices = fixedMesh.vertices.gpu();
                auto gpuFixedSurfaces = fixedMesh.triangles.gpu();

                selectedSurfacesJudgement.resize(gpuFixedSurfaces.size());
                selectedSurfacesJudgement.reset();
                // find the covered surfaces.
                LOG("selectedVertices.size() = " << selectedVertices.size());
                cuExecute(selectedVertices.size(), FindNearestVertex, vertices, surfaces, gpuFixedVertices,
                          gpuFixedSurfaces, selectedSurfacesJudgement, selectedVertices);
                gpuFixedVertices.clear();
                gpuFixedSurfaces.clear();
            }
            vertices.assign(fixedMesh.vertices);
            surfaces.assign(fixedMesh.triangles);
            standardVertices.assign(vertices);
        };

        virtual BBox get_bbox()
        {
            BBox box;
            box.min = make_float3(1e10, 1e10, 1e10);
            box.max = make_float3(-1e10, -1e10, -1e10);
            if (translations.size() == 0)
            {
                Mesh mesh(vertices.cpu(), surfaces.cpu());
                box = mesh.bbox();
                return box;
            }

            GArr<float3> verts;
            verts.assign(vertices);
            progressbar bar(translations.size(), "Calculating BBox");
            for (int t = 0; t < translations.size(); t++)
            {
                bar.update();
                cuExecute(vertices.size(), Transform, verts, standardVertices, translations[t], rotations[t]);
                box.min = fminf(box.min, thrust::reduce(thrust::device, verts.begin(), verts.end(),
                                                        make_float3(1e10, 1e10, 1e10), bbox_minimum()));
                box.max = fmaxf(box.max, thrust::reduce(thrust::device, verts.begin(), verts.end(),
                                                        make_float3(-1e10, -1e10, -1e10), bbox_maximum()));
            }
            std::cout << std::endl;
            verts.clear();
            return box;
        };

        virtual ~AudioObject()
        {
            vertices.clear();
            standardVertices.clear();
            surfaces.clear();
            selectedVertices.clear();
            selectedSurfacesJudgement.clear();
        }

    private:
        GArr<int> selectedVertices;
        GArr<int> selectedSurfacesJudgement;
        GArr<float3> vertices;
        GArr<float3> standardVertices;
        GArr<int3> surfaces;
        CArr<float> accelerations;
        CArr<float3> translations;
        CArr<float4> rotations;
        CArr<float> frameTime;
        int animationTimeStep = -1;
        int accTimeStep = 0;
        float sampleTime = 0.0f;

        bool AnimationUpdateUntil_(float time)
        {
            if (animationTimeStep + 1 >= frameTime.size())
                return false;

            bool flag = false;
            while (frameTime[animationTimeStep + 1] <= time)
            {
                animationTimeStep++;
                flag = true;
            }

            if (flag)
            {
                cuExecute(vertices.size(), Transform, vertices, standardVertices, translations[animationTimeStep],
                          rotations[animationTimeStep]);
            }
            return flag;
        };

        void AccelerationUpdateUntil_(float time)
        {
            accTimeStep++;
            while (sampleTime * accTimeStep <= time)
            {
                accTimeStep++;
            }
            accTimeStep--;
            return;
        };

        void LoadAccs_(const std::string &path);
        void LoadCover_(const std::string &path);
};

}  // namespace pppm
#endif  // M_OBJECT_H