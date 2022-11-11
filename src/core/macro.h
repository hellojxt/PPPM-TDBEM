#pragma once
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#include <bitset>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

namespace pppm
{
#define LOG_ERROR_COLOR "\033[1;31m"
#define LOG_WARNING_COLOR "\033[1;33m"
#define LOG_INFO_COLOR "\033[1;32m"
#define LOG_DEBUG_COLOR "\033[1;34m"
#define LOG_RESET_COLOR "\033[0m"
#define LOG_BOLD_COLOR "\033[1m"

#define LOG_FILE std::cout
#define LOG(x) LOG_FILE << x << std::endl;
#define LOG_INFO(x) LOG_FILE << #x << " = " << x << std::endl;
#define LOG_ERROR(x) LOG_FILE << LOG_ERROR_COLOR << x << LOG_RESET_COLOR << std::endl;
#define LOG_WARNING(x) LOG_FILE << LOG_WARNING_COLOR << x << LOG_RESET_COLOR << std::endl;
#define LOG_DEBUG(x) LOG_FILE << LOG_DEBUG_COLOR << x << LOG_RESET_COLOR << std::endl;
#define LOG_LINE(x) \
    LOG_FILE << LOG_BOLD_COLOR << "---------------" << x << "---------------" << LOG_RESET_COLOR << std::endl;
#define LOG_INLINE(x) LOG_FILE << x;

#define TICK_PRECISION 3
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TICK_INLINE(x) auto bench_inline_##x = std::chrono::steady_clock::now();
#define TOCK(x)                                                                                              \
    std::streamsize ss_##x = std::cout.precision();                                                          \
    std::cout.precision(TICK_PRECISION);                                                                     \
    LOG_FILE << #x ": "                                                                                      \
             << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - \
                                                                          bench_##x)                         \
                    .count()                                                                                 \
             << "s" << std::endl;                                                                            \
    std::cout.precision(ss_##x);
#define TOCK_INLINE(x)                                                                                       \
    std::streamsize ss_##x = std::cout.precision();                                                          \
    std::cout.precision(TICK_PRECISION);                                                                     \
    LOG_FILE << #x ": "                                                                                      \
             << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - \
                                                                          bench_inline_##x)                  \
                    .count()                                                                                 \
             << "s"                                                                                          \
             << "  \t";                                                                                      \
    std::cout.precision(ss_##x);

#define PI 3.14159265359
#define AIR_WAVE_SPEED 340.29f
#define AIR_DENSITY 1.225f
#define CGPU_FUNC __host__ __device__
#define GPU_FUNC __device__
#define CPU_FUNC __host__
#define RAND_F (float)rand() / (float)RAND_MAX
#define RAND_I(min, max) (min + rand() % (max - min + 1))
#define RAND_SIGN (rand() % 2 == 0 ? 1 : -1)
#define BDF2(x) 0.5 * ((x) * (x)-4 * (x) + 3)
#define MAX_FLOAT 3.402823466e+38F

typedef float real_t;
const real_t EPS = 1e-8f;
typedef thrust::complex<real_t> cpx;

enum cpx_phase
{
    CPX_REAL,
    CPX_IMAG,
    CPX_ABS,
};

enum PlaneType
{
    XY,
    XZ,
    YZ,
};

typedef std::vector<real_t> RealVec;
typedef std::vector<cpx> ComplexVec;

static inline void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        // printf( "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

#define atomicAddCpx(dst, value)                         \
    {                                                    \
        atomicAdd((float *)(dst), (value).real());       \
        atomicAdd(((float *)(dst)) + 1, (value).imag()); \
    }
#define atomicAddCpxBlock(dst, value)                          \
    {                                                          \
        atomicAdd_block((float *)(dst), (value).real());       \
        atomicAdd_block(((float *)(dst)) + 1, (value).imag()); \
    }

#ifdef DEBUG
#    define cuSafeCall(X) X
#else
#    define cuSafeCall(X) \
        X;                \
        checkCudaError(#X);
#endif

/**
 * @brief Macro to check cuda errors
 *
 */
#ifdef DEBUG
#    define cuSynchronize() \
        {}
#else
#    define cuSynchronize()                                                                                        \
        {                                                                                                          \
            char str[200];                                                                                         \
            cudaDeviceSynchronize();                                                                               \
            cudaError_t err = cudaGetLastError();                                                                  \
            if (err != cudaSuccess)                                                                                \
            {                                                                                                      \
                sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
                throw std::runtime_error(std::string(str));                                                        \
            }                                                                                                      \
        }
#endif

static uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
static uint cudaGridSize(uint totalSize, uint blockSize)
{
    int dim = iDivUp(totalSize, blockSize);
    return dim == 0 ? 1 : dim;
}

#define CUDA_BLOCK_SIZE 64
/**
 * @brief Macro definition for execuation of cuda kernels, note that at lease one block will be executed.
 *
 * size: indicate how many threads are required in total.
 * Func: kernel function
 */

#define cuExecute(size, Func, ...)                              \
    {                                                           \
        uint pDims = cudaGridSize((uint)size, CUDA_BLOCK_SIZE); \
        Func<<<pDims, CUDA_BLOCK_SIZE>>>(__VA_ARGS__);          \
        cuSynchronize();                                        \
    }

#define CUDA_3D_BLOCK_SIZE 4
#define cuExecute3D(size, Func, ...)                                                                           \
    {                                                                                                          \
        uint pDimx = cudaGridSize((uint)size.x, CUDA_3D_BLOCK_SIZE);                                           \
        uint pDimy = cudaGridSize((uint)size.y, CUDA_3D_BLOCK_SIZE);                                           \
        uint pDimz = cudaGridSize((uint)size.z, CUDA_3D_BLOCK_SIZE);                                           \
        Func<<<dim3(pDimx, pDimy, pDimz), dim3(CUDA_3D_BLOCK_SIZE, CUDA_3D_BLOCK_SIZE, CUDA_3D_BLOCK_SIZE)>>>( \
            __VA_ARGS__);                                                                                      \
        cuSynchronize();                                                                                       \
    }

#define CUDA_2D_BLOCK_SIZE 8
#define cuExecute2D(size, Func, ...)                                                                   \
    {                                                                                                  \
        uint pDimx = cudaGridSize((uint)size.x, CUDA_2D_BLOCK_SIZE);                                   \
        uint pDimy = cudaGridSize((uint)size.y, CUDA_2D_BLOCK_SIZE);                                   \
        Func<<<dim3(pDimx, pDimy, 1), dim3(CUDA_2D_BLOCK_SIZE, CUDA_2D_BLOCK_SIZE, 1)>>>(__VA_ARGS__); \
        cuSynchronize();                                                                               \
    }

#define cuExecuteBlock(size1, size2, Func, ...) \
    {                                           \
        Func<<<size1, size2>>>(__VA_ARGS__);    \
        cuSynchronize();                        \
    }

#define cuExecuteDyArr(size1, size2, size3, Func, ...) \
    {                                                  \
        Func<<<size1, size2, size3>>>(__VA_ARGS__);    \
        cuSynchronize();                               \
    }

#define SHOW(x) std::cout << #x << ":\n" << x << std::endl;

static std::ostream &operator<<(std::ostream &o, float3 const &f)
{
    return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
}
static std::ostream &operator<<(std::ostream &o, uint3 const &f)
{
    return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
}
static std::ostream &operator<<(std::ostream &o, int3 const &f)
{
    return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
}
static std::ostream &operator<<(std::ostream &o, int4 const &a)
{
    return o << "(" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << ")";
}

class Range
{
    public:
        uint start;
        uint end;
        CGPU_FUNC Range(uint start, uint end) : start(start), end(end) {}
        CGPU_FUNC Range() : start(0), end(0) {}
        inline CGPU_FUNC int length() { return end - start; }
        friend std::ostream &operator<<(std::ostream &os, const Range &r)
        {
            os << "[" << r.start << "," << r.end << ")";
            return os;
        }
};

class dim2
{
    public:
        int x, y;
        dim2(int x_, int y_)
        {
            x = x_;
            y = y_;
        }
};

struct is_not_empty
{
        CGPU_FUNC int operator()(const int &x) const { return x != 0; }
};

class BBox
{
    public:
        float3 min;
        float3 max;
        float width;
        BBox() {}
        BBox(float3 _min, float3 _max)
        {
            min = _min;
            max = _max;
        }
        friend std::ostream &operator<<(std::ostream &os, const BBox &b)
        {
            os << "[" << b.min << "," << b.max << "]";
            return os;
        }
};
}  // namespace pppm