#pragma once
#include "array3D.h"
#include "helper_math.h"
#include "macro.h"

namespace pppm
{
enum PotentialType
{
    SINGLE_LAYER,
    DOUBLE_LAYER
};

#define MAX_TARGET_COORD_NUM 16
struct TargetCoordArray
{
        float3 data[MAX_TARGET_COORD_NUM];
        float scale[MAX_TARGET_COORD_NUM];
        int size;
};

CGPU_FUNC inline cpx single_layer_potential(float3 src_coord, float3 trg_coord, cpx k)
{
    cpx potential(0, 0);
    float3 s2t = trg_coord - src_coord;
    real_t r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
    if (r2 != 0)
    {
        real_t r = sqrt(r2);
        potential += exp(cpx(0, 1) * r * k) / (4 * PI * r);
        // printf("r = %f, k = %f, potential = %f + %f i\n", r, k.real(), potential.real(), potential.imag());
    }
    return potential;
}

CGPU_FUNC inline cpx double_layer_potential(float3 src_coord, float3 trg_coord, float3 trial_norm, cpx k)
{
    cpx potential(0, 0);
    float3 s2t = src_coord - trg_coord;
    real_t r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
    if (r2 != 0)
    {
        real_t r = sqrt(r2);
        cpx ikr = cpx(0, 1) * r * k;
        potential += -exp(ikr) / (4 * PI * r2 * r) * (1 - ikr) * dot(s2t, trial_norm);
    }
    return potential;
}

CGPU_FUNC inline cpx layer_potential(float3 src_coord, float3 trg_coord, float3 trial_norm, cpx k, PotentialType type)
{
    if (type == SINGLE_LAYER)
        return single_layer_potential(src_coord, trg_coord, k);
    else
        return double_layer_potential(src_coord, trg_coord, trial_norm, k);
}

CGPU_FUNC inline cpx single_layer_potential(float3 src_coord, TargetCoordArray &trg_coord, cpx k)
{
    cpx potential(0, 0);
    for (int i = 0; i < trg_coord.size; i++)
    {
        float3 s2t = trg_coord.data[i] - src_coord;
        real_t r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
        if (r2 != 0)
        {
            real_t r = sqrt(r2);
            potential += exp(cpx(0, 1) * r * k) / (4 * PI * r) * trg_coord.scale[i];
        }
    }
    return potential;
}

CGPU_FUNC inline cpx double_layer_potential(float3 src_coord, TargetCoordArray &trg_coord, float3 trial_norm, cpx k)
{
    cpx potential(0, 0);
    for (int i = 0; i < trg_coord.size; i++)
    {
        float3 s2t = src_coord - trg_coord.data[i];
        real_t r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
        if (r2 != 0)
        {
            real_t r = sqrt(r2);
            cpx ikr = cpx(0, 1) * r * k;
            potential += -exp(ikr) / (4 * PI * r2 * r) * (1 - ikr) * dot(s2t, trial_norm) * trg_coord.scale[i];
        }
    }
    return potential;
}

CGPU_FUNC inline cpx layer_potential(float3 src_coord,
                                     TargetCoordArray &trg_coord,
                                     float3 trial_norm,
                                     cpx k,
                                     PotentialType type)
{
    if (type == SINGLE_LAYER)
        return single_layer_potential(src_coord, trg_coord, k);
    else
        return double_layer_potential(src_coord, trg_coord, trial_norm, k);
}

}  // namespace pppm