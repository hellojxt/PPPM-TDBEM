#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/io.h>
#include <unistd.h>
#include "helper_math.h"
#include "objIO.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <cstddef>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace pppm
{

Mesh::Mesh(CArr<float3> vertices_, CArr<int3> triangles_)
{
    vertices = vertices_;
    triangles = triangles_;
}
void Mesh::print()
{
    std::cout << "Vertices:\n";
    for (auto v : vertices.m_data)
        std::cout << "(" << v.x << "," << v.y << "," << v.z << ")\n";
    std::cout << "Triangles:\n";
    for (auto f : triangles.m_data)
        std::cout << "[" << f.x << "," << f.y << "," << f.z << "]\n";
}

void Mesh::writeOBJ(std::string filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(1);
    }
    for (auto v : vertices.m_data)
    {
        file << "v " << v.x << " " << v.y << " " << v.z << std::endl;
    }
    for (auto f : triangles.m_data)
    {
        file << "f " << f.x + 1 << " " << f.y + 1 << " " << f.z + 1 << std::endl;
    }
    file.close();
}

void Mesh::stretch(float scale)
{
    for (auto &v : vertices.m_data)
        v *= scale;
}

float Mesh::get_scale()
{
    float3 min = vertices[0];
    float3 max = vertices[0];
    for (auto &v : vertices.m_data)
    {
        min = fminf(min, v);
        max = fmaxf(max, v);
    }
    float3 size = max - min;
    float max_size = fmaxf(size.x, fmaxf(size.y, size.z));
    return max_size;
}

void Mesh::stretch_to(float scale)
{
    auto max_size = get_scale();
    stretch(scale / max_size);
}

void Mesh::move(float3 offset)
{
    for (auto &v : vertices.m_data)
        v += offset;
}

void Mesh::move_to(float3 pos)
{
    float3 center = get_center();
    for (auto &v : vertices.m_data)
        v += pos - center;
}

BBox Mesh::bbox()
{
    float3 min_p = vertices[0];
    float3 max_p = vertices[0];
    for (auto v : vertices.m_data)
    {
        min_p.x = std::min(min_p.x, v.x);
        min_p.y = std::min(min_p.y, v.y);
        min_p.z = std::min(min_p.z, v.z);
        max_p.x = std::max(max_p.x, v.x);
        max_p.y = std::max(max_p.y, v.y);
        max_p.z = std::max(max_p.z, v.z);
    }
    return BBox(min_p, max_p);
}

void Mesh::remove_isolated_vertices()
{
    std::vector<bool> is_isolated(vertices.size(), true);
    for (auto f : triangles.m_data)
    {
        is_isolated[f.x] = false;
        is_isolated[f.y] = false;
        is_isolated[f.z] = false;
    }
    std::vector<int> new_indices(vertices.size());
    int new_index = 0;
    for (int i = 0; i < vertices.size(); i++)
    {
        if (!is_isolated[i])
        {
            new_indices[i] = new_index;
            new_index++;
        }
    }
    CArr<float3> new_vertices(new_index);
    for (int i = 0; i < vertices.size(); i++)
    {
        if (!is_isolated[i])
        {
            new_vertices[new_indices[i]] = vertices[i];
        }
    }
    for (auto &f : triangles.m_data)
    {
        f.x = new_indices[f.x];
        f.y = new_indices[f.y];
        f.z = new_indices[f.z];
    }
    vertices = new_vertices;
}

float3 Mesh::get_center()
{
    float3 min_pos = vertices[0];
    float3 max_pos = vertices[0];
    for (int i = 1; i < vertices.size(); i++)
    {
        min_pos.x = std::min(min_pos.x, vertices[i].x);
        min_pos.y = std::min(min_pos.y, vertices[i].y);
        min_pos.z = std::min(min_pos.z, vertices[i].z);
        max_pos.x = std::max(max_pos.x, vertices[i].x);
        max_pos.y = std::max(max_pos.y, vertices[i].y);
        max_pos.z = std::max(max_pos.z, vertices[i].z);
    }
    float3 center = (min_pos + max_pos) / 2.0f;
    return center;
}

void Mesh::normalize()
{
    stretch_to(1.0f);
    move_to(make_float3(0.0f));
}

Mesh Mesh::loadOBJ(std::string file_name, bool log)
{

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(file_name))
    {
        std::cerr << "Failed to load " << file_name << std::endl;
        exit(1);
    }
    if (log)
    {
        if (!reader.Warning().empty())
        {
            std::cout << "WARN: " << reader.Warning() << std::endl;
        }
    }

    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();

    auto vertices = CArr<float3>(attrib.vertices.size() / 3);
    for (size_t v = 0; v < attrib.vertices.size() / 3; v++)
    {
        vertices[v] = make_float3(attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]);
    }

    int triangle_num = 0;
    for (auto &shape : shapes)
    {
        triangle_num += shape.mesh.num_face_vertices.size();
    }

    auto triangles = CArr<int3>(triangle_num);

    for (auto &shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int vertice_num_per_face = shape.mesh.num_face_vertices[f];
            assert(vertice_num_per_face == 3);
            tinyobj::index_t idx0 = shape.mesh.indices[f * 3 + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[f * 3 + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[f * 3 + 2];
            triangles[f + index_offset] = make_int3(idx0.vertex_index, idx1.vertex_index, idx2.vertex_index);
        }
        index_offset += shape.mesh.num_face_vertices.size();
    }

    if (log)
    {
        // LOG
        std::cout << "Vertices number: " << vertices.size() << "\n";
        std::cout << "Triangles number: " << triangles.size() << "\n";
        // Loaded success
        std::cout << "OBJ file:" << file_name << " loaded!"
                  << "\n";
    }

    return Mesh(vertices, triangles);
}

// xcx add below 2 functions
void Mesh::fix_mesh(float precision, std::string tmp_dir, std::string mesh_name)
{
    CHECK_DIR(tmp_dir);
    std::string python_src_dir = ROOT_DIR + std::string("python_scripts/");
    std::string python_src_name = "fix_mesh.py";
    std::string in_mesh_name = tmp_dir + "/" + mesh_name;
    std::string out_mesh_name = tmp_dir + "/fixed_" + mesh_name;
    writeOBJ(in_mesh_name);
    std::string cmd = "docker run -it --rm -v " + tmp_dir + ":/models " + "-v " + python_src_dir + ":/scripts " +
                      "pymesh/pymesh /scripts/" + python_src_name + " --detail " + std::to_string(precision) +
                      " /models/" + in_mesh_name + " /models/" + out_mesh_name;
    system(cmd.c_str());
    Mesh fixedMesh(out_mesh_name);
    vertices.assign(fixedMesh.vertices);
    triangles.assign(fixedMesh.triangles);
}

}  // namespace pppm
