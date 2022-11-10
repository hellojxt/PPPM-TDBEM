#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>   // fopen
#include <stdlib.h>  // EXIT_FAILURE
#include <string.h>  // strtok
#include "array3D.h"

namespace pppm
{

typedef struct VecStruct
{
        cusparseDnVecDescr_t vec;
        float *ptr;
} Vec;

class COOMatrix
{
    public:
        GArr<int> rows;
        GArr<int> cols;
        GArr<float> vals;
        int rows_num;
        int cols_num;
};

class BiCGSTAB_Solver
{
    public:
        Vec d_B, d_X, d_R, d_R0, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T, d_tmp;
        cublasHandle_t cublasHandle = NULL;
        cusparseHandle_t cusparseHandle = NULL;
        cusparseSpMatDescr_t matA, matM_lower, matM_upper;
        void *d_bufferMV;
        float *d_M_values;
        bool cache_stored;

        BiCGSTAB_Solver();

        void set_csr_matrix(GArr<int> &A_rows, GArr<int> &A_cols, GArr<float> &A_vals, int num_rows, int num_cols);
        void set_coo_matrix(GArr<int> &A_rows, GArr<int> &A_cols, GArr<float> &A_vals, int num_rows, int num_cols);

        GArr<float> solve(GArr<float> &b, int maxIterations = 20, float tolerance = 1e-10);

        void clear_cache();

        void clear();
};

};  // namespace pppm