#pragma once
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
}  // namespace pppm
