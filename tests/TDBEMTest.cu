#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "array_writer.h"
#include "bem.h"
#include "sound_source.h"

TEST_CASE("TDBEM", "[bem]")
{
#define BEM_EPS 0.1f
    using namespace pppm;
    float scale = 0.01f;
    float3 vertices[6] = {
        make_float3(RAND_F, RAND_F, RAND_F) * scale, make_float3(RAND_F, RAND_F, RAND_F) * scale,
        make_float3(RAND_F, RAND_F, RAND_F) * scale, make_float3(RAND_F, RAND_F, RAND_F) * scale,
        make_float3(RAND_F, RAND_F, RAND_F) * scale, make_float3(RAND_F, RAND_F, RAND_F) * scale,
    };
    int3 src = make_int3(0, 1, 2);
    int3 trg_face = make_int3(3, 4, 5);
    float3 trg_point = make_float3(RAND_F, RAND_F, RAND_F) * scale;

    std::vector<PairInfo> pair_infos;
    pair_infos.push_back(PairInfo(src, trg_face));
    pair_infos.push_back(PairInfo(src, trg_point));

    for (auto pair_info : pair_infos)
    {
        float dt = RAND_I(100, 200) * 1e-7f;
        float freq = RAND_I(1000, 4000);
        float omega = 2.0f * M_PI * freq;
        cpx wave_number = omega / AIR_WAVE_SPEED;
        TDBEM bem;
        bem.init(dt);
        History neumann;
        History dirichlet;
        float neumann_amp = RAND_F;
        float dirichlet_amp = RAND_F;
        SineSource sine(omega);

        for (int t = -STEP_NUM; t < 0; t++)
        {
            neumann[t] = sine(t * dt).real() * neumann_amp;
            dirichlet[t] = sine(t * dt).real() * dirichlet_amp;
        }
        float laplace_result[STEP_NUM];
        cpx helmholtz_result[STEP_NUM];
        for (int t = 0; t < STEP_NUM; t++)
        {
            // be careful, boundary data of current time need to be set before calling laplace ！！！
            neumann[t] = sine(t * dt).real() * neumann_amp;
            dirichlet[t] = sine(t * dt).real() * dirichlet_amp;
            laplace_result[t] = bem.laplace(vertices, pair_info, neumann, dirichlet, t);
        }

        cpx helmholtz_weight = bem.helmholtz(vertices, pair_info, neumann_amp, dirichlet_amp, wave_number);
        float amplitude = 0;
        for (int t = 0; t < STEP_NUM; t++)
        {
            helmholtz_result[t] = helmholtz_weight * sine(t * dt);
            if (abs(helmholtz_result[t]) > amplitude)
                amplitude = abs(helmholtz_result[t]);
        }
        for (int t = 0; t < STEP_NUM; t++)
        {
            REQUIRE(abs(laplace_result[t] - helmholtz_result[t].real()) / amplitude < BEM_EPS);
        }
        write_to_txt("laplace_result.txt", laplace_result, STEP_NUM);
        write_to_txt("helmholtz_result.txt", helmholtz_result, STEP_NUM);
    }
}
