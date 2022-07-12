#pragma once
#include "array.h"
#include <string>

namespace pppm{

    struct BBox{
        float3 min;
        float3 max;
        BBox(float3 _min, float3 _max){
            min = _min;
            max = _max;
        }
        friend std::ostream &operator<<(std::ostream &os, const BBox &bbox){
            os << "BBox: " << bbox.min << "-" << bbox.max;
            return os;
        }
    };


    struct Mesh{
        CArr<float3> vertices;
        CArr<int3> triangles; 
        CArr<float3> normal;
        Mesh(CArr<float3> vertices_, CArr<int3> triangles_);
        Mesh(CArr<float3> vertices_, CArr<int3> triangles_, CArr<float3> normal_);
        void print();
        void writeOBJ(std::string filename);
        void move(float3 offset);
        BBox bbox();
    };

    Mesh loadOBJ(const char* file_name);

}

#include "objIO.inl"