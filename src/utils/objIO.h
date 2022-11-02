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
        Mesh(CArr<float3> vertices_, CArr<int3> triangles_);
        Mesh(CArr<float3> vertices_, CArr<int3> triangles_, CArr<float3> normal_);
        void print();
        void writeOBJ(std::string filename);
        void move(float3 offset);
        void move_to(float3 pos);
        float3 get_center();
        void stretch(float scale);
        void stretch_to(float scale);
        void normalize();
        BBox bbox();
        static Mesh loadOBJ(std::string file_name, bool log = false);
};

}  // namespace pppm
