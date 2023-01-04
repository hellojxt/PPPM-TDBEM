#include "RigidBody.h"
#include <fstream>
#include <cmath>
#include <unordered_map>
#include "helper_math.h"
#include "progressbar.h"
#include "particle_grid.h"

namespace pppm
{
void ModalInfo::SetCoeffs(float timestep, float eigenVal, MaterialParameters &material)
{
    float lambda = eigenVal;
    float omega = std::sqrt(lambda);
    float ksi = (material.alpha + material.beta * lambda) / (2 * omega);
    float omega_prime = omega * std::sqrt(1 - ksi * ksi);
    float epsilon = std::exp(-ksi * omega * timestep);
    float sqrEpsilon = epsilon * epsilon;
    float theta = omega_prime * timestep;
    float gamma = std::asin(ksi);
    coeff1 = 2 * epsilon * std::cos(theta);
    coeff2 = -sqrEpsilon;

    float coeff3_item1 = epsilon * std::cos(theta + gamma);
    float coeff3_item2 = sqrEpsilon * std::cos(2 * theta + gamma);
    coeff3 = 2 * (coeff3_item1 - coeff3_item2) / (3 * omega * omega_prime);
    return;
}

void RigidBody::load_data(const std::string &data_dir)
{
    impulseTimeStamp = 0;
    LoadMotion_(std::filesystem::path(data_dir) / "motion.txt");
    LoadImpulses_(std::filesystem::path(data_dir) / "contact.txt");
    LoadTetMesh_(std::filesystem::path(data_dir) / "vertices.txt", std::filesystem::path(data_dir) / "tets.txt");
    LoadEigen_(std::filesystem::path(data_dir) / "eigenvalues.txt",
               std::filesystem::path(data_dir) / "eigenvectors.txt");
}

void RigidBody::LoadMotion_(const std::string &displacementPath)
{
    std::ifstream fin(displacementPath);

    float currTime = 0;
    float3 currTranslation;
    float4 currRotation;
    CArr<float3> cpuTranslations;
    CArr<float4> cpuRotations;

    if (!fin.good())
    {
        LOG_ERROR("Fail to load displacement file.\n");
        std::exit(EXIT_FAILURE);
    }
    std::string line;
    while (getline(fin, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        iss >> currTime >> currTranslation.x >> currTranslation.y >> currTranslation.z >> currRotation.x >>
            currRotation.y >> currRotation.z >> currRotation.w;
        cpuTranslations.pushBack(currTranslation);
        cpuRotations.pushBack(currRotation);
        frameTime.pushBack(currTime);
    }
    translations.assign(cpuTranslations);
    rotations.assign(cpuRotations);
    return;
};

void RigidBody::LoadImpulses_(const std::string &impulsePath)
{
    std::ifstream fin(impulsePath);
    float currTime;
    float vertexID;
    float3 impulseNormal;
    float impulseMagnitude;

    if (!fin.good())
    {
        LOG_ERROR("Fail to load impulse file.");
        std::exit(EXIT_FAILURE);
    }
    std::string line;
    while (getline(fin, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        iss >> currTime >> vertexID >> impulseNormal.x >> impulseNormal.y >> impulseNormal.z >> impulseMagnitude;

        Impulse imp;
        imp.currTime = currTime;
        imp.vertexID = F2I(vertexID);
        imp.impulseVec = impulseNormal * impulseMagnitude;
        impulses.pushBack(imp);
    }
    return;
}

std::pair<CArr<int3>, CArr<float3>> FindAllSurfaces(CArr<int4> &tetrahedrons, CArr<float3> &tetraVerts)
{
    CArr<int3> surfaceTriangles;
    CArr<float3> surfaceNorms;

    struct iVec3Hash
    {
            int operator()(const int3 &vec) const { return (vec.x << 20) + (vec.y << 10) + vec.z; }
    };
    struct iVec3Eq
    {
            bool operator()(const int3 &vec1, const int3 &vec2) const
            {
                return vec1.x == vec2.x && vec1.y == vec2.y && vec1.z == vec2.z;
            }
    };
    struct TriInfo
    {
            int cnt = 0;
            int tetID = 0;
            int exceptVertID = 0;
    };

    std::unordered_map<int3, TriInfo, iVec3Hash, iVec3Eq> candidateTriangles;
    int3 currTri;
    int int3::*int3Members[] = {&int3::x, &int3::y, &int3::z};
    int int4::*int4Members[] = {&int4::x, &int4::y, &int4::z, &int4::w};

    for (int i = 0, size = tetrahedrons.size(); i < size; i++)
    {
        int4 &currVertIDs = tetrahedrons[i];
        for (int i0 = 0; i0 < 4; i0++)
        {
            for (int j = 0, k = 0; j < 4; j++)
            {
                if (j == i0)
                    continue;
                currTri.*(int3Members[k++]) = currVertIDs.*(int4Members[j]);
            }
            auto &currInfo = candidateTriangles[currTri];
            currInfo.cnt++;
            currInfo.tetID = i;
            currInfo.exceptVertID = i0;
        }
    }

    for (auto &candidateTriangle : candidateTriangles)
    {
        if (candidateTriangle.second.cnt != 1)
            continue;

        int3 currTri = candidateTriangle.first;

        int4 &currTet = tetrahedrons[candidateTriangle.second.tetID];
        int exceptVertID = candidateTriangle.second.exceptVertID;

        float3 center = (tetraVerts[currTri.x] + tetraVerts[currTri.y] + tetraVerts[currTri.z]) / 3;
        float3 exceptVec = tetraVerts[currTet.*(int4Members[exceptVertID])] - center;
        float3 e1 = tetraVerts[currTri.y] - tetraVerts[currTri.x], e2 = tetraVerts[currTri.z] - tetraVerts[currTri.x];
        float3 normVec = normalize(cross(e1, e2));
        if (dot(normVec, exceptVec) > 0)
        {
            std::swap(currTri.y, currTri.z);
            normVec = -normVec;
        }
        surfaceTriangles.pushBack(currTri);
        surfaceNorms.pushBack(normVec);
    }

    return {surfaceTriangles, surfaceNorms};
}

void RigidBody::LoadTetMesh_(const std::string &vertsPath, const std::string &tetPath)
{
    std::ifstream f_verts(vertsPath);
    CArr<float3> cpuTetVertices;
    float3 vert;
    if (!f_verts.good())
    {
        LOG_ERROR("Fail to load tet mesh file.");
        std::exit(EXIT_FAILURE);
    }
    std::string line;
    while (getline(f_verts, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        iss >> vert.x >> vert.y >> vert.z;
        cpuTetVertices.pushBack(vert);
    }
    f_verts.close();

    std::ifstream f_tet(tetPath);
    CArr<int4> tetrahedrons;
    float4 tet;
    if (!f_tet.good())
    {
        LOG_ERROR("Fail to load tet mesh file.");
        std::exit(EXIT_FAILURE);
    }
    while (getline(f_tet, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        iss >> tet.x >> tet.y >> tet.z >> tet.w;
        int idxs[4] = {F2I(tet.x), F2I(tet.y), F2I(tet.z), F2I(tet.w)};
        std::sort(idxs, idxs + 4);
        tetrahedrons.pushBack(make_int4(idxs[0], idxs[1], idxs[2], idxs[3]));
    }

    tetVertices.assign(cpuTetVertices);
    standardTetVertices.assign(cpuTetVertices);
    auto [surfaceTris, surfaceNorms] = FindAllSurfaces(tetrahedrons, cpuTetVertices);
    tetSurfaces.assign(surfaceTris);
    tetSurfaceNorms.assign(surfaceNorms);
}

void RigidBody::LoadEigen_(const std::string &eigenvalPath, const std::string &eigenvecPath)
{
    std::ifstream f_val(eigenvalPath);
    CArr<float> eigenVals_;
    int modalSize;

    if (!f_val.good())
    {
        LOG_ERROR("Fail to load eigenval file.");
        std::exit(EXIT_FAILURE);
    }
    std::string line;
    while (getline(f_val, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        float val;
        iss >> val;
        eigenVals_.pushBack(val);
        if (std::sqrt(val) / (2 * M_PI) < max_frequncy)
            modalSize = eigenVals_.size();
    }
    int vecDim = tetVertices.size() * 3;
    eigenVals.resize(modalSize);
    for (int i = 0; i < modalSize; i++)
        eigenVals[i] = eigenVals_[i];
    eigenVecs.resize(vecDim, modalSize);

    std::ifstream f_vec(eigenvecPath);
    if (!f_vec.good())
    {
        LOG_ERROR("Fail to load eigenvec file.");
        std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < vecDim; i++)
    {
        for (int j = 0; j < eigenVals_.size(); j++)
        {
            float val;
            f_vec >> val;
            if (j < modalSize)
                eigenVecs(i, j) = val;
        }
    }
    return;
};

void RigidBody::InitIIR_()
{
    timestep = 1.0f / sample_rate;
    int eigenNum = eigenVals.size();

    modalInfos.resize(eigenNum);
    for (int i = 0; i < eigenNum; i++)
    {
        modalInfos[i].SetCoeffs(timestep, eigenVals[i], material);
    }
    cpuQ.resize(eigenNum);
    gpuQ.resize(eigenNum);
    modalMatrix.assign(eigenVecs);
    update_surf_matrix();
    currentImpulse.clear();
    return;
}

__global__ void MatrixVertToTri(GArr2D<float> modalMatrix,
                                GArr2D<float> modelMatrixSurf,
                                GArr<int3> surfaces,
                                GArr<float3> surfaceNorms)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= surfaces.size())
        return;
    int3 verts = surfaces[id];
    for (int i = 0; i < modelMatrixSurf.size.y; i++)
    {
        float3 surf_vec = (make_float3(modalMatrix(verts.x * 3, i), modalMatrix(verts.x * 3 + 1, i),
                                       modalMatrix(verts.x * 3 + 2, i)) +
                           make_float3(modalMatrix(verts.y * 3, i), modalMatrix(verts.y * 3 + 1, i),
                                       modalMatrix(verts.y * 3 + 2, i)) +
                           make_float3(modalMatrix(verts.z * 3, i), modalMatrix(verts.z * 3 + 1, i),
                                       modalMatrix(verts.z * 3 + 2, i))) /
                          3;

        modelMatrixSurf(id, i) = dot(surf_vec, surfaceNorms[id]);
    }
    return;
}

void RigidBody::update_surf_matrix()
{
    surfaceAccs.resize(tetSurfaces.size());
    modelMatrixSurf.resize(tetSurfaces.size(), eigenVals.size());
    cuExecute(tetSurfaces.size(), MatrixVertToTri, modalMatrix, modelMatrixSurf, tetSurfaces, tetSurfaceNorms);
}

__global__ void update_surf_acc_kernel(GArr<float> gpuQ, GArr2D<float> modelMatrixSurf, GArr<float> surfaceAccs)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= surfaceAccs.size())
        return;
    float acc = 0;
    for (int i = 0; i < gpuQ.size(); i++)
    {
        acc += modelMatrixSurf(id, i) * gpuQ[i];
    }
    surfaceAccs[id] = acc;
}

void RigidBody::Q_to_Accs_()
{
    cuExecute(surfaceAccs.size(), update_surf_acc_kernel, gpuQ, modelMatrixSurf, surfaceAccs);
}

void RigidBody::CalculateIIR_()
{

    float currTime = current_time;
    int size = modalInfos.size();
    while (impulseTimeStamp < impulses.size() && currTime >= impulses[impulseTimeStamp].currTime)
    {
        currentImpulse.push_back(impulses[impulseTimeStamp]);
        impulseTimeStamp++;
    };
    while (currentImpulse.size() > 0 && currentImpulse[0].dead(currTime))
    {
        currentImpulse.erase(currentImpulse.begin());
    }
    for (int i = 0; i < size; i++)
    {
        modalInfos[i].f = 0;
        for (int j = 0; j < currentImpulse.size(); j++)
        {
            int id = currentImpulse[j].vertexID;
            float3 currImpluse = currentImpulse[j].impulseVec * currentImpulse[j].amp(currTime);
            modalInfos[i].f += eigenVecs(id * 3, i) * currImpluse.x + eigenVecs(id * 3 + 1, i) * currImpluse.y +
                               eigenVecs(id * 3 + 2, i) * currImpluse.z;
        }
    }

    for (int i = 0; i < size; i++)
    {
        auto &modalInfo = modalInfos[i];
        float q0 = modalInfo.coeff1 * modalInfo.q1 + modalInfo.coeff2 * modalInfo.q2 + modalInfo.coeff3 * modalInfo.f;
        modalInfo.q3 = modalInfo.q2;
        modalInfo.q2 = modalInfo.q1;
        modalInfo.q1 = q0;
        cpuQ[i] = (modalInfo.q1 + modalInfo.q3 - 2 * modalInfo.q2) / (timestep * timestep);
    }

    gpuQ.assign(cpuQ);
    Q_to_Accs_();
    return;
}

__device__ inline float3 rotate(const float4 q, const float3 v)
{
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

__global__ void Transform(GArr<float3> vertices, GArr<float3> standard_vertices, float3 translation, float4 rotation)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= vertices.size())
        return;
    vertices[id] = rotate(rotation, standard_vertices[id]) + translation;
    return;
}

void RigidBody::animation_step()
{
    cuExecute(tetVertices.size(), Transform, tetVertices, standardTetVertices, translations[animationTimeStamp],
              rotations[animationTimeStamp]);
    animationTimeStamp++;
}

struct bbox_minimum
{
        __device__ float3 operator()(const float3 &a, const float3 &b) const { return fminf(a, b); }
};

struct bbox_maximum
{
        __device__ float3 operator()(const float3 &a, const float3 &b) const { return fmaxf(a, b); }
};

BBox RigidBody::get_bbox()
{
    BBox box;
    box.min = make_float3(1e10, 1e10, 1e10);
    box.max = make_float3(-1e10, -1e10, -1e10);
    GArr<float3> verts;
    verts.assign(tetVertices);
    int t = 1;
    while (frameTime[t] < impulses[0].currTime)
        t++;
    t--;
    progressbar bar(translations.size() - t, "Calculating BBox");
    for (; t < translations.size(); t++)
    {
        bar.update();
        cuExecute(tetVertices.size(), Transform, verts, standardTetVertices, translations[t], rotations[t]);
        box.min = fminf(box.min, thrust::reduce(thrust::device, verts.begin(), verts.end(),
                                                make_float3(1e10, 1e10, 1e10), bbox_minimum()));
        box.max = fmaxf(box.max, thrust::reduce(thrust::device, verts.begin(), verts.end(),
                                                make_float3(-1e10, -1e10, -1e10), bbox_maximum()));
    }
    std::cout << std::endl;
    verts.clear();
    return box;
}

void RigidBody::audio_step()
{
    if (frameTime[animationTimeStamp] <= current_time)
    {
        animation_step();
        mesh_is_updated = true;
    }
    else
    {
        mesh_is_updated = false;
    }
    current_time += timestep;
    CalculateIIR_();
}

void RigidBody::move_to_first_impulse()
{
    while (current_time + timestep < impulses[0].currTime)
    {
        audio_step();
    };
    printf("Move to first impulse at time %f\n", current_time);
}

__global__ void update_surf_matrix_for_fixed_mesh(GArr2D<float> modalMatrix,
                                                  GArr2D<float> modelMatrixSurf,
                                                  GArr<float3> origin_vertices,
                                                  GArr<int3> origin_surfaces,
                                                  GArr<float3> vertices,
                                                  GArr<int3> surfaces)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= surfaces.size())
        return;
    int3 surface = surfaces[id];
    float3 verts[3] = {vertices[surface.x], vertices[surface.y], vertices[surface.z]};
    float3 surf_normal = normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));
    int min_dist_id[3] = {-1, -1, -1};
    float min_dist[3] = {MAX_FLOAT, MAX_FLOAT, MAX_FLOAT};
#pragma unroll
    for (int f_id = 0; f_id < origin_surfaces.size(); f_id++)
    {
        int3 face = origin_surfaces[f_id];
        float3 o_verts[3] = {origin_vertices[face.x], origin_vertices[face.y], origin_vertices[face.z]};
        for (int i = 0; i < 3; i++)
        {
            float3 nearest_p = get_nearest_triangle_point(verts[i], o_verts[0], o_verts[1], o_verts[2]);
            float dist = length(nearest_p - verts[i]);
            if (dist < min_dist[i])
            {
                min_dist[i] = dist;
                min_dist_id[i] = f_id;
            }
        }
    }
    for (int j = 0; j < modalMatrix.size.y; j++)
    {
        modelMatrixSurf(id, j) = 0;
    }
    for (int i = 0; i < 3; i++)
    {
        int3 face = origin_surfaces[min_dist_id[i]];
        float3 o_verts[3] = {origin_vertices[face.x], origin_vertices[face.y], origin_vertices[face.z]};
        float3 v = verts[i];
        // interpolate in the triangle
        float dist[3] = {length(o_verts[0] - v), length(o_verts[1] - v), length(o_verts[2] - v)};
        double coeff[3] = {1.0 / (dist[0] + 1e-10), 1.0 / (dist[1] + 1e-10), 1.0 / (dist[2] + 1e-10)};
        double sum = coeff[0] + coeff[1] + coeff[2];
        coeff[0] /= sum;
        coeff[1] /= sum;
        coeff[2] /= sum;
        for (int j = 0; j < modalMatrix.size.y; j++)
        {
            modelMatrixSurf(id, j) += (modalMatrix(face.x * 3, j) * coeff[0] + modalMatrix(face.y * 3, j) * coeff[1] +
                                       modalMatrix(face.z * 3, j) * coeff[2]) *
                                      surf_normal.x;
            modelMatrixSurf(id, j) +=
                (modalMatrix(face.x * 3 + 1, j) * coeff[0] + modalMatrix(face.y * 3 + 1, j) * coeff[1] +
                 modalMatrix(face.z * 3 + 1, j) * coeff[2]) *
                surf_normal.y;
            modelMatrixSurf(id, j) +=
                (modalMatrix(face.x * 3 + 2, j) * coeff[0] + modalMatrix(face.y * 3 + 2, j) * coeff[1] +
                 modalMatrix(face.z * 3 + 2, j) * coeff[2]) *
                surf_normal.z;
        }
    }
    for (int j = 0; j < modalMatrix.size.y; j++)
    {
        modelMatrixSurf(id, j) /= 3;
    }
}

void RigidBody::fix_mesh(float precision, std::string tmp_dir)
{
    CHECK_DIR(tmp_dir);
    std::string python_src_dir = ROOT_DIR + std::string("python_scripts/");
    std::string python_src_name = "fix_mesh.py";
    export_surface_mesh(tmp_dir);
    std::string in_mesh_name = "surface.obj";
    std::string out_mesh_name = "surface_fixed.obj";
    std::string cmd = "docker run -it --rm -v " + tmp_dir + ":/models " + "-v " + python_src_dir + ":/scripts " +
                      "pymesh/pymesh /scripts/" + python_src_name + " --detail " + std::to_string(precision) +
                      " /models/" + in_mesh_name + " /models/" + out_mesh_name;
    // std::cout << cmd << std::endl;
    system(cmd.c_str());
    Mesh fixedMesh(tmp_dir + "/" + out_mesh_name);
    GArr<float3> fixedVertices = fixedMesh.vertices.gpu();
    GArr<int3> fixedSurfaces = fixedMesh.triangles.gpu();
    modelMatrixSurf.resize(fixedSurfaces.size(), modalMatrix.size.y);
    cuExecute(fixedSurfaces.size(), update_surf_matrix_for_fixed_mesh, modalMatrix, modelMatrixSurf, tetVertices,
              tetSurfaces, fixedVertices, fixedSurfaces);
    tetVertices.assign(fixedVertices);
    fixedVertices.clear();
    tetSurfaces.assign(fixedSurfaces);
    fixedSurfaces.clear();
    standardTetVertices.assign(tetVertices);
    surfaceAccs.resize(tetSurfaces.size());
}

void RigidBody::export_surface_mesh(const std::string &output_path)
{
    CHECK_DIR(output_path);
    Mesh surfaceMesh(tetVertices, tetSurfaces);
    printf("export surface mesh to %s\n", output_path.c_str());
    surfaceMesh.writeOBJ(output_path + "/surface.obj");
}

// export the surface mesh with all the modes.
void RigidBody::export_mesh_with_modes(const std::string &output_path)
{
    CHECK_DIR(output_path);
    export_surface_mesh(output_path);
    std::ofstream fout(output_path + "/modes.txt");
    int modalAmount = modalMatrix.cols;
    printf("surface triangle amount: %d\n", tetSurfaces.size());
    printf("modalAmount: %d\n", modalAmount);
    std::cout << "frequency of modes: " << std::endl;
    for (int i = 0; i < modalAmount; i++)
    {
        // print frequency
        std::cout << int(sqrt(eigenVals[i]) / (2 * M_PI)) << ", ";
    }
    std::cout << std::endl;
    progressbar bar(modalAmount, "exporting modes");
    for (int i = 0; i < modalAmount; i++)
    {
        bar.update();
        for (int j = 0; j < modalAmount; j++)
        {
            cpuQ[j] = 0;
        }
        cpuQ[i] = 1;
        gpuQ.assign(cpuQ);
        Q_to_Accs_();
        auto surf_accs = surfaceAccs.cpu();
        for (int j = 0; j < surf_accs.size(); j++)
        {
            fout << surf_accs[j] << " ";
        }
        fout << std::endl;
    }
    fout.close();
    std::cout << std::endl;
    return;
}

void RigidBody::export_signal(const std::string &output_path, float max_time)
{
    CHECK_DIR(output_path);
    std::ofstream fout(output_path + "/signal.txt");
    std::ofstream force_out(output_path + "/force.txt");
    int frame_num = max_time / timestep;
    printf("frame_num: %d\n", frame_num);
    progressbar bar(frame_num, "exporting signal");
    for (int i = 0; i < eigenVals.size(); i++)
    {
        force_out << eigenVals[i] / material.density << " ";
    }
    fout << sample_rate << std::endl;
    force_out << std::endl;
    for (int i = 0; i < frame_num; i++)
    {
        bar.update();
        audio_step();
        float s = 0;
        for (int j = 0; j < cpuQ.size(); j++)
        {
            s += cpuQ[j];
        }
        for (int j = 0; j < modalInfos.size(); j++)
        {
            force_out << modalInfos[j].f << " ";
        }
        force_out << std::endl;
        fout << s << std::endl;
    }
    fout.close();
    force_out.close();
    std::cout << std::endl;
    return;
}

void RigidBody::export_mesh_sequence(const std::string &output_path)
{
    current_time = 0;
    CHECK_DIR(output_path);
    float animation_export_timestep = 1.0f / 60.0f;
    int frame_num = frameTime.last() / animation_export_timestep;
    progressbar bar(frame_num - 1, "exporting mesh sequence");
    for (int i = 1; i < frame_num; i++)
    {
        while (current_time < i * animation_export_timestep)
        {
            audio_step();
        }
        bar.update();
        Mesh surfaceMesh(tetVertices, tetSurfaces);
        surfaceMesh.remove_isolated_vertices();
        surfaceMesh.writeOBJ(output_path + "/surface_" + std::to_string(i) + ".obj");
    }
    std::cout << std::endl;
    return;
}

void RigidBody::export_surface_accs(const std::string &filename)
{
    std::ofstream fout(filename);
    auto surf_accs = surfaceAccs.cpu();
    for (int j = 0; j < surf_accs.size(); j++)
    {
        fout << surf_accs[j] << std::endl;
    }
    fout.close();
    return;
}

}  // namespace pppm