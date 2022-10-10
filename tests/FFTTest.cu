#include "bem.h"
#include <thrust/random.h>
#include <random>
#include <chrono>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <thrust/device_vector.h>

#ifdef pureImag
#undef pureImag
#endif
#define pureImag thrust::complex<float>(0, 1)

template<int step = 1>
__device__ void FFT2(thrust::complex<float>* in, thrust::complex<float>* out,
    thrust::complex<float>* buffer)
{
    const int newStep = step * 2;
    FFT2<newStep>(in, out, buffer);
    FFT2<newStep>(in + step, out + step, buffer + step);
    for (int i = 0; i < STEP_NUM; i += newStep)
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

template<>
__device__ void FFT2<STEP_NUM>(thrust::complex<float>* in, thrust::complex<float>* out,
    thrust::complex<float>* buffer)
{
    out[0] = in[0];
    return;
}

#define BLOCK_SIZE 64

__global__ void scaledFFT2(thrust::complex<float>* in, float* out, float lambda)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ thrust::complex<float> buffer1[BLOCK_SIZE][STEP_NUM];
    __shared__ thrust::complex<float> buffer2[BLOCK_SIZE][STEP_NUM];
    FFT2(in + id * STEP_NUM, buffer1[threadIdx.x], buffer2[threadIdx.x]);
    out += id * STEP_NUM;
    float temp = 1.0f;
    for(int i = 0; i < STEP_NUM; i++)
    {
        out[i] = buffer1[threadIdx.x][i].real() / STEP_NUM * temp;
        temp /= lambda;
    }
    return;
};

__global__ void scaledFFT3(thrust::complex<float>* in, float* out, float lambda)
{
    thrust::complex<float> buffer1[BLOCK_SIZE];
    thrust::complex<float> buffer2[BLOCK_SIZE];
    FFT2(in + threadIdx.x * STEP_NUM, buffer1, buffer2);
    out += threadIdx.x * STEP_NUM;
    float temp = 1.0f;
    for(int i = 0; i < STEP_NUM; i++)
    {
        out[i] = buffer1[i].real() / STEP_NUM * temp;
        temp /= lambda;
    }
    return;
};


__global__ void scaledIDFT2(thrust::complex<float>* in, float *out, float lambda)
{
    in += threadIdx.x * STEP_NUM, out += threadIdx.x * STEP_NUM;
    for (int k = 0; k < STEP_NUM; k++)
    {
        thrust::complex<float> result = 0;
        for (int i = 0; i < STEP_NUM; i++)
        {
            result += in[i] * exp(2 * PI * pureImag / STEP_NUM * k * i);
        }
        out[k] = result.real() / STEP_NUM * pow(lambda, -k);
    }
}

std::random_device rd{};
struct GenRand
{
    int seed = rd();
    __device__ thrust::complex<float> operator () (int idx)
    {
        thrust::default_random_engine randEng(seed);
        thrust::uniform_real_distribution<float> uniDist(-50.0f, 50.0f);
        randEng.discard(idx);
        return {uniDist(randEng), uniDist(randEng)};
    }
};


#define KERNEL_NUM 512
thrust::device_vector<float> kernelOut1(STEP_NUM * KERNEL_NUM),
    kernelOut2(STEP_NUM * KERNEL_NUM);
thrust::device_vector<thrust::complex<float>> kernelIn(STEP_NUM * KERNEL_NUM);

TEST_CASE("FFT", "[bem]")
{
    pppm::TDBEM obj;
    obj.lambda = 1;
    thrust::complex<float> in[STEP_NUM];
    float out[STEP_NUM], out2[STEP_NUM];

    thrust::uniform_real_distribution<float> distribution(-50.0f, 50.0f);
    thrust::default_random_engine engine(123);
    thrust::generate(in, in + STEP_NUM, [&](){ return thrust::complex<float>{ distribution(engine), distribution(engine)}; });
    TICK(cpuDFT);
    obj.scaledIDFT(in, out);
    TOCK(cpuDFT);
    TICK(cpuFFT);
    obj.scaledFFT(in, out2);
    TOCK(cpuFFT);
    for (size_t i = 0; i < STEP_NUM; i++)
    {
        REQUIRE(abs(out[i] - out2[i]) < 1e-3f);
    }

    thrust::device_vector<float> kernelOut01(STEP_NUM), kernelOut02(STEP_NUM);
    thrust::device_vector<thrust::complex<float>> kernelIn0(in, in + STEP_NUM);
    auto gpuVecTest = thrust::raw_pointer_cast(kernelIn0.data());
    scaledIDFT2<<<1,1>>>(gpuVecTest, thrust::raw_pointer_cast(kernelOut01.data()), 1.0f);
    scaledFFT2<<<1,1>>>(gpuVecTest, thrust::raw_pointer_cast(kernelOut02.data()), 1.0f);
    cudaDeviceSynchronize();
    for(size_t i = 0; i < STEP_NUM; i++)
    {
        REQUIRE(abs(out[i] - kernelOut01[i]) < 1e-3f);
        REQUIRE(abs(out2[i] - kernelOut02[i]) < 1e-3f);
    }

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(int(kernelIn.size())),
        kernelIn.begin(),
        GenRand());

    auto gpuVec = thrust::raw_pointer_cast(kernelIn.data());
    TICK(gpuDFT);
    scaledIDFT2<<<1, KERNEL_NUM>>>(gpuVec, thrust::raw_pointer_cast(kernelOut1.data()), 1.0f);
    cudaDeviceSynchronize();
    TOCK(gpuDFT);

    TICK(gpuSharedFFT);
    scaledFFT2<<<BLOCK_SIZE, KERNEL_NUM / BLOCK_SIZE>>>(gpuVec, thrust::raw_pointer_cast(kernelOut2.data()), 1.0f);
    cudaDeviceSynchronize();
    TOCK(gpuSharedFFT);

    for (size_t i = 0; i < STEP_NUM * KERNEL_NUM; i++)
    {
        REQUIRE(abs(kernelOut1[i] - kernelOut2[i]) < 1e-3f);
    }

    TICK(gpuLocalFFT);
    scaledFFT3<<<1, KERNEL_NUM>>>(gpuVec, thrust::raw_pointer_cast(kernelOut2.data()), 1.0f);
    cudaDeviceSynchronize();
    TOCK(gpuLocalFFT);

    return;
}