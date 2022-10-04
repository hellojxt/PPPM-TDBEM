#include "bem.h"
#include <thrust/random.h>
#include <random>
#include <chrono>

int main()
{
    pppm::TDBEM obj;
    obj.lambda = 1;
    thrust::complex<float> in[STEP_NUM];
    float out[STEP_NUM], out2[STEP_NUM];

    std::random_device rd{};
    thrust::uniform_real_distribution<float> distribution(-50.0f, 50.0f);
    thrust::default_random_engine engine(123);
    thrust::generate(in, in + STEP_NUM, [&](){ return thrust::complex<float>{ distribution(engine), distribution(engine)}; });
    TICK(t1);
    obj.scaledIDFT(in, out);
    TOCK(t1);
    TICK(t2);
    obj.scaledFFT(in, out2);
    TOCK(t2);
    for (size_t i = 0; i < STEP_NUM; i++)
    {
        if(out[i] - out2[i] > 1e-3f)
        {
            std::cout << out[i] << " " << out2[i];
        }
    }
    std::cout << std::endl;
    return 0;
}