
#include "sparse.h"

namespace pppm
{
/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (status != cudaSuccess)                                                                                    \
        {                                                                                                             \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(1);                                                                                                  \
        }                                                                                                             \
    }

#define CHECK_CUSPARSE(func)                                                                                         \
    {                                                                                                                \
        cusparseStatus_t status = (func);                                                                            \
        if (status != CUSPARSE_STATUS_SUCCESS)                                                                       \
        {                                                                                                            \
            printf("cuSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), \
                   status);                                                                                          \
            exit(1);                                                                                                 \
        }                                                                                                            \
    }

#define CHECK_CUBLAS(func)                                                             \
    {                                                                                  \
        cublasStatus_t status = (func);                                                \
        if (status != CUBLAS_STATUS_SUCCESS)                                           \
        {                                                                              \
            printf("CUBLAS API failed at line %d with error: %d\n", __LINE__, status); \
            exit(1);                                                                   \
        }                                                                              \
    }

int solve_BiCGStab_cusparse(cublasHandle_t cublasHandle,
                            cusparseHandle_t cusparseHandle,
                            int m,
                            cusparseSpMatDescr_t matA,
                            cusparseSpMatDescr_t matM_lower,
                            cusparseSpMatDescr_t matM_upper,
                            Vec d_B,
                            Vec d_X,
                            Vec d_R0,
                            Vec d_R,
                            Vec d_P,
                            Vec d_P_aux,
                            Vec d_S,
                            Vec d_S_aux,
                            Vec d_V,
                            Vec d_T,
                            Vec d_tmp,
                            void *d_bufferMV,
                            int maxIterations,
                            float tolerance)
{
    const float zero = 0.0;
    const float one = 1.0;
    const float minus_one = -1.0;
    //--------------------------------------------------------------------------
    // Create opaque data structures that holds analysis data between calls
    float coeff_tmp;
    size_t bufferSizeL, bufferSizeU;
    void *d_bufferL, *d_bufferU;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL))
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_lower,
                                           d_P.vec, d_tmp.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL,
                                           &bufferSizeL))
    CHECK_CUDA(cudaMalloc(&d_bufferL, bufferSizeL))
    CHECK_CUSPARSE(cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_lower,
                                         d_P.vec, d_tmp.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL,
                                         d_bufferL))

    // Calculate UPPER buffersize
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrU))
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_upper,
                                           d_tmp.vec, d_P_aux.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                                           &bufferSizeU))
    CHECK_CUDA(cudaMalloc(&d_bufferU, bufferSizeU))
    CHECK_CUSPARSE(cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_upper,
                                         d_tmp.vec, d_P_aux.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                                         d_bufferU))
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA(cudaMemcpy(d_R0.ptr, d_B.ptr, m * sizeof(float), cudaMemcpyDeviceToDevice))
    //    (b) compute R = -A * X0 + R
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, d_X.vec, &one,
                                d_R0.vec, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
    //--------------------------------------------------------------------------
    float alpha, delta, delta_prev, omega;
    CHECK_CUBLAS(cublasSdot(cublasHandle, m, d_R0.ptr, 1, d_R0.ptr, 1, &delta))
    delta_prev = delta;
    // R = R0
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_R0.ptr, m * sizeof(float), cudaMemcpyDeviceToDevice))
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    float nrm_R;
    CHECK_CUBLAS(cublasSnrm2(cublasHandle, m, d_R0.ptr, 1, &nrm_R))
    float threshold = tolerance * nrm_R;

    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 1; i <= maxIterations; i++)
    {

        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        CHECK_CUDA(cudaMemcpy(d_P.ptr, d_R.ptr, m * sizeof(float), cudaMemcpyDeviceToDevice))
        if (i > 1)
        {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            CHECK_CUBLAS(cublasSdot(cublasHandle, m, d_R0.ptr, 1, d_R.ptr, 1, &delta))
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            float beta = (delta / delta_prev) * (alpha / omega);
            delta_prev = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            float minus_omega = -omega;
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, m, &minus_omega, d_V.ptr, 1, d_P.ptr, 1))
            //    (b) P = beta * P
            CHECK_CUBLAS(cublasSscal(cublasHandle, m, &beta, d_P.ptr, 1))
            //    (c) P = R + P
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, m, &one, d_R.ptr, 1, d_P.ptr, 1))
        }
        //----------------------------------------------------------------------
        // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        //    (a) M_L^-1 P_i => tmp    (triangular solver)
        CHECK_CUDA(cudaMemset(d_tmp.ptr, 0x0, m * sizeof(float)))
        CHECK_CUDA(cudaMemset(d_P_aux.ptr, 0x0, m * sizeof(float)))
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lower, d_P.vec,
                                          d_tmp.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL))
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_upper, d_tmp.vec,
                                          d_P_aux.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU))
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, d_P_aux.vec, &zero,
                                    d_V.vec, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
        //    (b) denominator = R'0 * V
        float denominator;
        CHECK_CUBLAS(cublasSdot(cublasHandle, m, d_R0.ptr, 1, d_V.ptr, 1, &denominator))
        alpha = delta / denominator;
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, m, &alpha, d_P_aux.ptr, 1, d_X.ptr, 1))
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        CHECK_CUDA(cudaMemcpy(d_S.ptr, d_R.ptr, m * sizeof(float), cudaMemcpyDeviceToDevice))
        //    (b) S = -alpha * V + R_i-1
        float minus_alpha = -alpha;
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, m, &minus_alpha, d_V.ptr, 1, d_S.ptr, 1))
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        float nrm_S;
        CHECK_CUBLAS(cublasSnrm2(cublasHandle, m, d_S.ptr, 1, &nrm_S))
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        //    (a) M_L^-1 S => tmp    (triangular solver)
        cudaMemset(d_tmp.ptr, 0x0, m * sizeof(float));
        cudaMemset(d_S_aux.ptr, 0x0, m * sizeof(float));
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lower, d_S.vec,
                                          d_tmp.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL))
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_upper, d_tmp.vec,
                                          d_S_aux.vec, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU))
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, d_S_aux.vec, &zero,
                                    d_T.vec, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
        //    (b) omega_num = (A * S_aux, s)
        float omega_num, omega_den;
        CHECK_CUBLAS(cublasSdot(cublasHandle, m, d_T.ptr, 1, d_S.ptr, 1, &omega_num))
        //    (c) omega_den = (A * S_aux, A * S_aux)
        CHECK_CUBLAS(cublasSdot(cublasHandle, m, d_T.ptr, 1, d_T.ptr, 1, &omega_den))
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, m, &omega, d_S_aux.ptr, 1, d_X.ptr, 1))
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        CHECK_CUDA(cudaMemcpy(d_R.ptr, d_S.ptr, m * sizeof(float), cudaMemcpyDeviceToDevice))
        //    (a) R_i+1 = -omega * T + R
        float minus_omega = -omega;
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, m, &minus_omega, d_T.ptr, 1, d_R.ptr, 1))
        // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        CHECK_CUBLAS(cublasSnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
        if (nrm_R < threshold)
            break;
    }
    //--------------------------------------------------------------------------
    //    (a) copy b in R
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(float), cudaMemcpyDeviceToDevice))
    // R = -A * X + R
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, d_X.vec, &one,
                                d_R.vec, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
    // check ||R||
    CHECK_CUBLAS(cublasSnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
    //--------------------------------------------------------------------------
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrL))
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrU))
    CHECK_CUDA(cudaFree(d_bufferL))
    CHECK_CUDA(cudaFree(d_bufferU))
    return EXIT_SUCCESS;
}

BiCGSTAB_Solver::BiCGSTAB_Solver()
{
    // ### cuSPARSE Handle and descriptors initialization ###
    CHECK_CUBLAS(cublasCreate(&cublasHandle))
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle))
    cache_stored = false;
}
void BiCGSTAB_Solver::set_csr_matrix(GArr<int> &A_rows,
                                     GArr<int> &A_cols,
                                     GArr<float> &A_vals,
                                     int num_rows,
                                     int num_cols)
{
    if (cache_stored)
        clear_cache();
    int *d_A_rows = A_rows.data();
    int *d_A_columns = A_cols.data();
    float *d_A_values = A_vals.data();
    int nnz = A_vals.size();
    int m = num_rows;
    CHECK_CUDA(cudaMalloc((void **)&d_M_values, nnz * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_R.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_R0.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_P.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_P_aux.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_S.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_S_aux.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_V.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_T.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMemcpy(d_M_values, d_A_values, nnz * sizeof(float), cudaMemcpyDeviceToDevice))

    // Create dense vectors
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_R.vec, m, d_R.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_R0.vec, m, d_R0.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_P.vec, m, d_P.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_P_aux.vec, m, d_P_aux.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_S.vec, m, d_S.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_S_aux.vec, m, d_S_aux.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_V.vec, m, d_V.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_T.vec, m, d_T.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_32F))
    cusparseIndexBase_t baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    int *d_M_rows = d_A_rows;
    int *d_M_columns = d_A_columns;
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, m, nnz, d_A_rows, d_A_columns, d_A_values, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, baseIdx, CUDA_R_32F))
    // M_lower
    CHECK_CUSPARSE(cusparseCreateCsr(&matM_lower, m, m, nnz, d_M_rows, d_M_columns, d_M_values, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, baseIdx, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower)))
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit)))
    // M_upper
    CHECK_CUSPARSE(cusparseCreateCsr(&matM_upper, m, m, nnz, d_M_rows, d_M_columns, d_M_values, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, baseIdx, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper)))
    CHECK_CUSPARSE(
        cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit)))
    const float alpha = 1.0f;
    size_t bufferSizeMV;
    float beta = 0.0;
    Vec d_X_tmp, d_B_tmp;
    CHECK_CUDA(cudaMalloc((void **)&d_X_tmp.ptr, m * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&d_B_tmp.ptr, m * sizeof(float)))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_X_tmp.vec, m, d_X_tmp.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_B_tmp.vec, m, d_B_tmp.ptr, CUDA_R_32F))
    // ### cuSPARSE SpMV buffer size ###
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, d_X_tmp.vec,
                                           &beta, d_B_tmp.vec, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV))
    CHECK_CUDA(cudaMalloc(&d_bufferMV, bufferSizeMV))
    csrilu02Info_t infoM = NULL;
    cusparseMatDescr_t matLU;
    int bufferSizeLU = 0;
    void *d_bufferLU;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&matLU))
    CHECK_CUSPARSE(cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(matLU, baseIdx))
    CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&infoM))

    CHECK_CUSPARSE(cusparseScsrilu02_bufferSize(cusparseHandle, m, nnz, matLU, d_M_values, d_A_rows, d_A_columns, infoM,
                                                &bufferSizeLU))
    CHECK_CUDA(cudaMalloc(&d_bufferLU, bufferSizeLU))
    CHECK_CUSPARSE(cusparseScsrilu02_analysis(cusparseHandle, m, nnz, matLU, d_M_values, d_A_rows, d_A_columns, infoM,
                                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU))
    int structural_zero;
    CHECK_CUSPARSE(cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM, &structural_zero))
    // M = L * U
    CHECK_CUSPARSE(cusparseScsrilu02(cusparseHandle, m, nnz, matLU, d_M_values, d_A_rows, d_A_columns, infoM,
                                     CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU))
    // Find numerical zero
    int numerical_zero;
    CHECK_CUSPARSE(cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM, &numerical_zero))

    CHECK_CUSPARSE(cusparseDestroyCsrilu02Info(infoM))
    CHECK_CUSPARSE(cusparseDestroyMatDescr(matLU))
    CHECK_CUDA(cudaFree(d_bufferLU))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_B_tmp.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_X_tmp.vec))
    CHECK_CUDA(cudaFree(d_B_tmp.ptr))
    CHECK_CUDA(cudaFree(d_X_tmp.ptr))
    cache_stored = true;
}

void BiCGSTAB_Solver::set_coo_matrix(GArr<int> &A_rows,
                                     GArr<int> &A_cols,
                                     GArr<float> &A_vals,
                                     int num_rows,
                                     int num_cols)
{
    GArr<int> A_rows_csr;
    A_rows_csr.resize(num_rows + 1);
    CHECK_CUSPARSE(cusparseXcoo2csr(cusparseHandle, A_rows.data(), A_rows.size(), num_rows, A_rows_csr.data(),
                                    CUSPARSE_INDEX_BASE_ZERO))
    set_csr_matrix(A_rows_csr, A_cols, A_vals, num_rows, num_cols);
}

void BiCGSTAB_Solver::clear_cache()
{
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_B.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_X.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_R.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_R0.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_P.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_P_aux.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_S.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_S_aux.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_V.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_T.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_tmp.vec))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matM_lower))
    CHECK_CUSPARSE(cusparseDestroySpMat(matM_upper))

    CHECK_CUDA(cudaFree(d_R.ptr))
    CHECK_CUDA(cudaFree(d_R0.ptr))
    CHECK_CUDA(cudaFree(d_P.ptr))
    CHECK_CUDA(cudaFree(d_P_aux.ptr))
    CHECK_CUDA(cudaFree(d_S.ptr))
    CHECK_CUDA(cudaFree(d_S_aux.ptr))
    CHECK_CUDA(cudaFree(d_V.ptr))
    CHECK_CUDA(cudaFree(d_T.ptr))
    CHECK_CUDA(cudaFree(d_tmp.ptr))
    CHECK_CUDA(cudaFree(d_M_values))
    CHECK_CUDA(cudaFree(d_bufferMV))
}

void BiCGSTAB_Solver::clear()
{
    clear_cache();
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle))
    CHECK_CUBLAS(cublasDestroy(cublasHandle))
}

GArr<float> BiCGSTAB_Solver::solve(GArr<float> &b, int maxIterations, float tolerance)
{
    GArr<float> X;
    X.resize(b.size());
    X.reset();
    d_X.ptr = X.data();
    d_B.ptr = b.data();
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_B.vec, b.size(), d_B.ptr, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_X.vec, b.size(), d_X.ptr, CUDA_R_32F))
    solve_BiCGStab_cusparse(cublasHandle, cusparseHandle, b.size(), matA, matM_lower, matM_upper, d_B, d_X, d_R0, d_R,
                            d_P, d_P_aux, d_S, d_S_aux, d_V, d_T, d_tmp, d_bufferMV, maxIterations, tolerance);
    return X;
}

__global__ void eliminate_zeros_kernel(GArr<int> index,
                                       GArr<int> rows,
                                       GArr<int> cols,
                                       GArr<float> vals,
                                       GArr<int> rows_shrink,
                                       GArr<int> cols_shrink,
                                       GArr<float> vals_shrink)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < index.size() && vals[i] != 0)
    {
        int idx = index[i] - 1;
        rows_shrink[idx] = rows[i];
        cols_shrink[idx] = cols[i];
        vals_shrink[idx] = vals[i];
    }
}

struct nonzero_vals
{
        __host__ __device__ int operator()(const float x) { return x != 0; }
};

void COOMatrix::eliminate_zeros()
{
    GArr<int> index;
    index.resize(vals.size());
    thrust::transform(thrust::device, vals.begin(), vals.end(), index.begin(), nonzero_vals());
    thrust::inclusive_scan(thrust::device, index.begin(), index.end(), index.begin());
    int num_nonzeros = index.last_item();
    GArr<int> rows_shrink;
    GArr<int> cols_shrink;
    GArr<float> vals_shrink;
    rows_shrink.resize(num_nonzeros);
    cols_shrink.resize(num_nonzeros);
    vals_shrink.resize(num_nonzeros);
    cuExecute(vals.size(), eliminate_zeros_kernel, index, rows, cols, vals, rows_shrink, cols_shrink, vals_shrink);
    rows.clear();
    cols.clear();
    vals.clear();
    rows = rows_shrink;
    cols = cols_shrink;
    vals = vals_shrink;
}

};  // namespace pppm
