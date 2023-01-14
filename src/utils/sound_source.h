#pragma once
#include "helper_math.h"
#include "macro.h"
namespace pppm
{
/**
 * @brief The class that represents a cos function (complex number)
 */
class SineSource
{
    public:
        float omega;
        SineSource(float omega) { this->omega = omega; }
        CGPU_FUNC cpx inline operator()(float t) { return exp(cpx(0.0f, omega * t)); }
        CGPU_FUNC cpx inline operator()(float t, int freq_factor) { return exp(cpx(0.0f, omega * freq_factor * t)); }
};

class MonoPole
{
    public:
        float3 center;
        cpx wave_number;
        MonoPole(float3 center, cpx wave_number)
        {
            this->center = center;
            this->wave_number = wave_number;
        }
        CGPU_FUNC cpx inline dirichlet(float3 pos)
        {
            float r = length(pos - center);
            return exp(-cpx(0, 1) * wave_number * r) / (4 * PI * r);
        }

        CGPU_FUNC cpx inline neumann(float3 pos, float3 normal)
        {
            float r = length(pos - center);
            cpx ikr = cpx(0, 1) * r * wave_number; // 表示当前振动到了什么地方
            return -exp(-ikr) / (4 * PI * r * r * r) * (1 + ikr) * dot(normal, pos - center); // 表示当前的振动速度吗？
        }
};
}  // namespace pppm
