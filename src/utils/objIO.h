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
    void normalize();
    BBox bbox();
};

Mesh loadOBJ(std::string file_name, bool log = false);

}  // namespace pppm
