#include "array3D.h"
#include "svd.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace pppm
{
// CUDA API error checking
#define CUDA_CHECK(err)                                                   \
    do                                                                    \
    {                                                                     \
        cudaError_t err_ = (err);                                         \
        if (err_ != cudaSuccess)                                          \
        {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                       \
        }                                                                 \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                   \
    do                                                                        \
    {                                                                         \
        cusolverStatus_t err_ = (err);                                        \
        if (err_ != CUSOLVER_STATUS_SUCCESS)                                  \
        {                                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                       \
        }                                                                     \
    } while (0)

SVDResult cusolver_svd(GArr3D<float> A)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    int batchSize = A.batchs;
    int m = A.rows;
    int n = A.cols;
    if (m != n)
    {
        throw std::runtime_error("cusolver_svd: m != n");
    }
    int lda = m;
    int ldu = m;
    int ldv = m;
    int rank = m;
    long long int strideA = static_cast<long long int>(lda * m);
    long long int strideS = m;
    long long int strideU = static_cast<long long int>(ldu * m);
    long long int strideV = static_cast<long long int>(ldv * m);

    float *d_A = A.begin();
    GArr2D<float> S(A.batchs, m); /* singular values */
    float *d_S = S.begin();
    GArr3D<float> U(A.batchs, m, m); /* left singular vectors */
    float *d_U = U.begin();
    GArr3D<float> V(A.batchs, m, m); /* right singular vectors */
    float *d_V = V.begin();
    int *d_info = nullptr; /* error info */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * batchSize));

    int lwork = 0;           /* size of workspace */
    float *d_work = nullptr; /* device workspace for getrf */

    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute eigenvectors */

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
        cusolverH, jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                          /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank,             /* number of singular values */
        m,                /* nubmer of rows of Aj, 0 <= m */
        n,                /* number of columns of Aj, 0 <= n  */
        d_A,              /* Aj is m-by-n */
        lda,              /* leading dimension of Aj */
        strideA,          /* >= lda*n */
        d_S,              /* Sj is rank-by-1, singular values in descending order */
        strideS,          /* >= rank */
        d_U,              /* Uj is m-by-rank */
        ldu,              /* leading dimension of Uj, ldu >= max(1,m) */
        strideU,          /* >= ldu*rank */
        d_V,              /* Vj is n-by-rank */
        ldv,              /* leading dimension of Vj, ldv >= max(1,n) */
        strideV,          /* >= ldv*rank */
        &lwork, batchSize /* number of matrices */
        ));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));
    std::vector<double> RnrmF(batchSize, 0); /* residual norm */
    /* step 4: compute SVD */
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
        cusolverH, jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                         /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank,            /* number of singular values */
        m,               /* nubmer of rows of Aj, 0 <= m */
        n,               /* number of columns of Aj, 0 <= n  */
        d_A,             /* Aj is m-by-n */
        lda,             /* leading dimension of Aj */
        strideA,         /* >= lda*n */
        d_S,             /* Sj is rank-by-1 */
                         /* the singular values in descending order */
        strideS,         /* >= rank */
        d_U,             /* Uj is m-by-rank */
        ldu,             /* leading dimension of Uj, ldu >= max(1,m) */
        strideU,         /* >= ldu*rank */
        d_V,             /* Vj is n-by-rank */
        ldv,             /* leading dimension of Vj, ldv >= max(1,n) */
        strideV,         /* >= ldv*rank */
        d_work, lwork, d_info, RnrmF.data(), batchSize /* number of matrices */
        ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    /* free resources */
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return SVDResult(S, U, V);
}
};  // namespace pppm
