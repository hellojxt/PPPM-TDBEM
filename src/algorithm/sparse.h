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
        COOMatrix()
        {
            rows_num = 0;
            cols_num = 0;
        }
        void resize(int rows_num, int cols_num, int nnz)
        {
            this->rows_num = rows_num;
            this->cols_num = cols_num;
            rows.resize(nnz);
            cols.resize(nnz);
            vals.resize(nnz);
        }

        void reset()
        {
            rows.reset();
            cols.reset();
            vals.reset();
        }

        void clear()
        {
            rows.clear();
            cols.clear();
            vals.clear();
        }

        void set_matrix(int rows_num, int cols_num, GArr<int> &rows, GArr<int> &cols, GArr<float> &vals)
        {
            this->rows_num = rows_num;
            this->cols_num = cols_num;
            this->rows = rows;
            this->cols = cols;
            this->vals = vals;
        }
        void eliminate_zeros();

        void sort_by_row();

        void print()
        {
            auto rows_cpu = rows.cpu();
            auto cols_cpu = cols.cpu();
            auto vals_cpu = vals.cpu();
            // print dense matrix
            for (int i = 0; i < rows_num; i++)
            {
                for (int j = 0; j < cols_num; j++)
                {
                    bool found = false;
                    for (int k = 0; k < rows.size(); k++)
                    {
                        if (rows_cpu[k] == i && cols_cpu[k] == j)
                        {
                            printf("%f ", vals_cpu[k]);
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        printf("%f ", 0.0f);
                    }
                }
                printf("\n");
            }
        }
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

        // must be sorted!!! (although the cusparse documentation does not mention this)
        void set_csr_matrix(GArr<int> &A_rows, GArr<int> &A_cols, GArr<float> &A_vals, int num_rows, int num_cols);
        void set_coo_matrix(GArr<int> &A_rows, GArr<int> &A_cols, GArr<float> &A_vals, int num_rows, int num_cols);
        void set_coo_matrix(COOMatrix &A) { set_coo_matrix(A.rows, A.cols, A.vals, A.rows_num, A.cols_num); }

        void solve(GArr<float> &b, GArr<float> &x, int maxIterations = 100, float tolerance = 1e-6);

        void clear_cache();

        void clear();
};

};  // namespace pppm