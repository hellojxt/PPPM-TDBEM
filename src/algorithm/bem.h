#pragma once
#include <cstdio>
#include "array3D.h"
#include "integrand.h"
#include "macro.h"

namespace pppm
{

#define STEP_NUM_POWER 5  // 2^STEP_NUM_POWER = STEP_NUM
#define STEP_NUM 32       // number of time steps for history of each particle
#define pureImag cpx(0, 1)

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

class LayerWeight
{
    public:
        float single_layer[STEP_NUM];
        float double_layer[STEP_NUM];
        inline CGPU_FUNC LayerWeight() {}
        inline CGPU_FUNC void reset()
        {
            for (int i = 0; i < STEP_NUM; i++)
            {
                single_layer[i] = 0;
                double_layer[i] = 0;
            }
        }

        /*
         * offset have to be in range [-STEP_NUM, 0].
         * some last weight will be ignored.
         */
        inline CGPU_FUNC void add(const LayerWeight &other, float weight = 1, int offset = 0)
        {
            for (int i = -offset; i < STEP_NUM; i++)
            {
                single_layer[i] += other.single_layer[i + offset] * weight;
                double_layer[i] += other.double_layer[i + offset] * weight;
            }
        }

        inline CGPU_FUNC void add(float k)
        {
            for (int i = 0; i < STEP_NUM; i++)
            {
                single_layer[i] += k;
                double_layer[i] += k;
            }
        }

        inline CGPU_FUNC void divide(float k)
        {
            for (int i = 0; i < STEP_NUM; i++)
            {
                single_layer[i] /= k;
                double_layer[i] /= k;
            }
        }

        inline CGPU_FUNC void multiply(float k)
        {
            for (int i = 0; i < STEP_NUM; i++)
            {
                single_layer[i] *= k;
                double_layer[i] *= k;
            }
        }

        inline CGPU_FUNC float convolution(History &neumann, History &dirichlet, int t)
        {
            float result = 0;
            for (int k = 0; k < STEP_NUM; k++)
            {
                result += -single_layer[k] * neumann[t - k] + double_layer[k] * dirichlet[t - k];
            }
            return result;
        }
};

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

        CGPU_FUNC inline void scaledIDFT(cpx *in, float *out)
        {
            for (int k = 0; k < STEP_NUM; k++)
            {
                cpx result = 0;
                for (int i = 0; i < STEP_NUM; i++)
                {
                    result += in[i] * exp(2 * PI * cpx(0, 1) / STEP_NUM * k * i);
                }
                out[k] = result.real() / STEP_NUM * pow(lambda, -k);
            }
        }

        CGPU_FUNC inline void FFT(cpx *in, cpx *out, cpx *buffer, int step = 1)
        {
            if (step == STEP_NUM)
            {
                out[0] = in[0];
                return;
            }
            int newStep = step * 2;
            FFT(in, out, buffer, newStep);
            FFT(in + step, out + step, buffer + step, newStep);
            for (int i = 0; i < STEP_NUM; i += step * 2)
            {
                buffer[i / 2] = out[i];
                buffer[(STEP_NUM + i) / 2] = out[i + step];
            }
            for (int i = 0; i < STEP_NUM / 2; i += step)
            {
                auto temp = exp(2 * PI * i / STEP_NUM * pureImag) * buffer[i + STEP_NUM / 2];
                auto temp2 = buffer[i];
                out[i] = temp2 + temp;
                out[i + STEP_NUM / 2] = temp2 - temp;
            }
            return;
        }

        CGPU_FUNC inline void scaledFFT(cpx *in, float *out)
        {
            cpx buffer1[STEP_NUM];
            cpx buffer2[STEP_NUM];
            FFT(in, buffer1, buffer2);
            float temp = 1.0f;
            for (int i = 0; i < STEP_NUM; i++)
            {
                out[i] = buffer1[i].real() / STEP_NUM * temp;
                temp /= lambda;
            }
            return;
        }

        CGPU_FUNC inline void laplace_weight(const float3 *vertices,
                                             PairInfo pair,
                                             PotentialType potential_type,
                                             float *weight)
        {
            cpx v[STEP_NUM];
            for (int k = 0; k <= STEP_NUM / 2; k++)
            {
                v[k] = pair_integrand(vertices, pair, wave_numbers[k], potential_type);
            }
            for (int k = STEP_NUM / 2 + 1; k < STEP_NUM; k++)
            {
                v[k] = conj(v[STEP_NUM - k]);
            }
            scaledIDFT(v, weight);
        }

        CGPU_FUNC inline void laplace_weight(const float3 *vertices, PairInfo pair, LayerWeight *weight)
        {
            laplace_weight(vertices, pair, SINGLE_LAYER, weight->single_layer);
            laplace_weight(vertices, pair, DOUBLE_LAYER, weight->double_layer);
        }

        CGPU_FUNC inline float laplace(const float3 *vertices,
                                       PairInfo pair,
                                       History src_neumann,
                                       History src_dirichlet,
                                       int t)
        {
            LayerWeight weight;
            laplace_weight(vertices, pair, &weight);
            return weight.convolution(src_neumann, src_dirichlet, t);
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