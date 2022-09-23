#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "helper_math.h"
#include "objIO.h"
namespace pppm
{

Mesh::Mesh(CArr<float3> vertices_, CArr<int3> triangles_)
{
    vertices  = vertices_;
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

void Mesh::writeOBJ(std::string filename) {}

void Mesh::move(float3 offset)
{
    for (auto &v : vertices.m_data)
        v += offset;
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

void Mesh::normalize()
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
    float3 center   = (min_pos + max_pos) / 2.0f;
    float3 scale_f3 = (max_pos - min_pos) / 2.0f;
    float scale     = std::max(std::max(scale_f3.x, scale_f3.y), scale_f3.z);
    for (int i = 0; i < vertices.size(); i++)
    {
        vertices[i] = (vertices[i] - center) / scale;
    }
}

Mesh loadOBJ(std::string file_name, bool log)
{
    CArr<float3> vertices;
    CArr<int3> triangles;
    std::stringstream ss;
    std::ifstream in_file(file_name);
    std::string line   = "";
    std::string prefix = "";

    // std::cout << "Start reading\n";
    // File open error check
    if (!in_file.is_open())
    {
        std::cout << "Error opening file: " << file_name << "\n";
        exit(1);
    }

    // Read one line at a time
    while (std::getline(in_file, line))
    {
        // Get the prefix of the line
        ss.clear();
        ss.str(line);
        ss >> prefix;

        if (prefix == "#")
        {}
        else if (prefix == "o")
        {}
        else if (prefix == "s")
        {}
        else if (prefix == "use_mtl")
        {}
        else if (prefix == "v")  // Vertex position
        {
            float3 tmp;
            ss >> tmp.x >> tmp.y >> tmp.z;
            vertices.pushBack(tmp);
        }
        else if (prefix == "vt")
        {
            // ss >> temp_vec2.x >> temp_vec2.y;
            // vertex_texcoords.push_back(temp_vec2);
        }
        else if (prefix == "vn")
        {
            // ss >> temp_vec3.x >> temp_vec3.y >> temp_vec3.z;
            // vertex_normals.push_back(temp_vec3);
        }
        else if (prefix == "f")
        {
            int tmp;
            int counter = 0;
            std::vector<int> tmp_inds;
            while (ss >> tmp)
            {
                // Pushing indices into correct arrays
                if (counter == 0)
                    tmp_inds.push_back(tmp - 1);
                // else if (counter == 1)
                // 	vertex_texcoord_indicies.push_back(temp_glint);
                // else if (counter == 2)
                // 	vertex_normal_indicies.push_back(temp_glint);

                // Handling characters
                if (ss.peek() == '/')
                {
                    ++counter;
                    ss.ignore(1, '/');
                }
                else if (ss.peek() == ' ')
                {
                    counter = 0;
                    ss.ignore(1, ' ');
                }

                // Reset the counter
                if (counter > 2)
                    counter = 0;
            }
            triangles.pushBack(make_int3(tmp_inds[0], tmp_inds[1], tmp_inds[2]));
        }
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

}  // namespace pppm
