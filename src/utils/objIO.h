#pragma once
#include <string>
#include "array.h"

namespace pppm
{

class Mesh
{
    public:
        CArr<float3> vertices;
        CArr<int3> triangles;
        CArr<float3> normal;
        Mesh(){};
        Mesh(std::string file_name, bool log = false) { *this = loadOBJ(file_name, log); }
        Mesh(CArr<float3> vertices_, CArr<int3> triangles_);
        Mesh(CArr<float3> vertices_, CArr<int3> triangles_, CArr<float3> normal_);
        void print();
        void writeOBJ(std::string filename);
        void move(float3 offset);
        void move_to(float3 pos);
        float3 get_center();
        float get_scale();
        void stretch(float scale);
        void stretch_to(float scale);
        void normalize();
        void remove_isolated_vertices();
        BBox bbox();
        static Mesh loadOBJ(std::string file_name, bool log = false);
};

}  // namespace pppm
