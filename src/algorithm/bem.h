#pragma once
#include "array3D.h"
#include "integrand.h"

namespace pppm
{

#define STEP_NUM 64  // number of time steps for history of each particle

/**
 * @brief particle to particle calculation type
 * @param pair_type can be either "FACE_TO_FACE" or "FACE_TO_POINT"
 */
class PairInfo
{
    public:
        int3 src;
        int3 dst_face;
        float3 dst_point;
        PairType pair_type;
        CGPU_FUNC PairInfo(int3 src, int3 dst_face) : src(src), dst_face(dst_face), pair_type(FACE_TO_FACE) {}
        CGPU_FUNC PairInfo(int3 src, float3 dst_point) : src(src), dst_point(dst_point), pair_type(FACE_TO_POINT) {}
};

typedef CircularArray<float, STEP_NUM * 2> History;

CGPU_FUNC inline cpx pair_integrand(const float3 *vertices,
                                    PairInfo pair,
                                    cpx wave_number,
                                    PotentialType potential_type)
{
    if (pair.pair_type == FACE_TO_FACE)
        return face2FaceIntegrand(vertices, pair.src, pair.dst_face, wave_number, potential_type);
    else
        return face2PointIntegrand(vertices, pair.src, pair.dst_point, wave_number, potential_type);
};

class TDBEM
{
    public:
        cpx wave_numbers[STEP_NUM];  // wave number for transforming from lapalace
                                     // domain to time domain and vice versa
        float lambda;                // origin for integration
        float dt;                    // time step

        TDBEM(){};
        void init(float dt)
        {
            this->dt = dt;
            lambda =
                std::max(pow((double)(dt * AIR_WAVE_SPEED), 3.0 / STEP_NUM), pow((double)EPS, 1.0 / (2 * STEP_NUM)));
            for (int k = 0; k < STEP_NUM; k++)
            {
                cpx s_k = lambda * exp(-2 * PI * cpx(0, 1) / STEP_NUM * k);
                wave_numbers[k] = BDF2(s_k) / (dt * AIR_WAVE_SPEED) / cpx(0, 1);
            }
        }

        CGPU_FUNC inline void scaledDFT(cpx *in, cpx *out)
        {
            for (int k = 0; k < STEP_NUM; k++)
            {
                out[k] = 0;
                for (int i = 0; i < STEP_NUM; i++)
                {
                    out[k] += in[i] * exp(2 * PI * cpx(0, 1) / STEP_NUM * k * i);
                }
                out[k] = out[k] / STEP_NUM * pow(lambda, -k);
            }
        }

        CGPU_FUNC inline void laplace_weight(const float3 *vertices,
                                             PairInfo pair,
                                             PotentialType potential_type,
                                             cpx *weight)
        {
            cpx v[STEP_NUM];
            for (int k = 0; k < STEP_NUM; k++)
            {
                v[k] = pair_integrand(vertices, pair, wave_numbers[k], potential_type);
            }
            scaledDFT(v, weight);
        }

        CGPU_FUNC inline float laplace(const float3 *vertices,
                                       PairInfo pair,
                                       History src_neumann,
                                       History src_dirichlet,
                                       int t)
        {
            cpx single_layer_weight[STEP_NUM];
            laplace_weight(vertices, pair, SINGLE_LAYER, single_layer_weight);
            cpx double_layer_weight[STEP_NUM];
            laplace_weight(vertices, pair, DOUBLE_LAYER, double_layer_weight);
            float result = 0;
            for (int k = 0; k < STEP_NUM; k++)
            {
                result += -single_layer_weight[k].real() * src_neumann[t - k] +
                          double_layer_weight[k].real() * src_dirichlet[t - k];
            }
            return result;
        }

        CGPU_FUNC inline cpx helmholtz(const float3 *vertices,
                                       PairInfo pair,
                                       cpx src_neumann,
                                       cpx src_dirichlet,
                                       cpx wave_number)
        {
            cpx single_layer_weight = pair_integrand(vertices, pair, wave_number, SINGLE_LAYER);
            cpx double_layer_weight = pair_integrand(vertices, pair, wave_number, DOUBLE_LAYER);
            return -single_layer_weight * src_neumann + double_layer_weight * src_dirichlet;
        }
};

}  // namespace pppm