#pragma once
#ifndef M_OBJECT_H
#define M_OBJECT_H
#include "array.h"

namespace pppm
{

class Object
{
public:
    Object() = default;
    virtual float GetLastFrameTime() = 0;
    virtual void UpdateUntil(float time) = 0;
    virtual GArr<float3>& GetVertices() = 0;
    virtual GArr<int3>& GetSurfaces() = 0;
    virtual ~Object(){};
};
}
#endif // M_OBJECT_H