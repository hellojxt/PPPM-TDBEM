#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "array.h"
#include "macro.h"
#include "sparse.h"

using namespace pppm;
#define TEST_NUM 4
using Catch::Approx;

void test_sort()
{
    std::vector<int> row = {3, 2, 0, 3, 0, 1, 0, 2, 2};
    std::vector<int> col = {1, 0, 0, 3, 2, 1, 3, 2, 3};
    std::vector<float> val = {8.0, 5.0, 1.0, 9.0, 2.0, 4.0, 3.0, 6.0, 7.0};
    int m = 4;
    GArr<int> A_rows, A_cols;
    GArr<float> A_vals, b_vals, x_vals;
    A_rows.assign(row);
    A_cols.assign(col);
    A_vals.assign(val);

    COOMatrix mat;
    mat.set_matrix(m, m, A_rows, A_cols, A_vals);
    mat.eliminate_zeros();
    mat.sort_by_row();

    int h_rows_ref[] = {
        0, 0, 0, 1, 2, 2, 2, 3, 3,
    };                                                  // sorted
    int h_columns_ref[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};  // sorted
    double h_values_ref[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    auto h_rows = mat.rows.cpu();
    auto h_columns = mat.cols.cpu();
    auto h_values = mat.vals.cpu();
    for (int i = 0; i < h_values.size(); i++)
    {
        REQUIRE(h_rows[i] == h_rows_ref[i]);
        REQUIRE(h_columns[i] == h_columns_ref[i]);
        REQUIRE(h_values[i] == Approx(h_values_ref[i]));
    }
}

void test_BiCGStab_solver()
{
    BiCGSTAB_Solver solver;

    int m = 100;
    CArr2D<float> A(m, m);
    CArr<float> x(m);
    CArr<float> b(m);
    for (int _test = 0; _test < TEST_NUM; _test++)
    {
        // generate a sparse random matrix with diagonal dominant
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                A(i, j) = 0;
            }
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int k = rand() % m;
                A(i, k) = rand() % 100 - 50;
            }
            A(i, i) = 1;
            for (int j = 0; j < m; j++)
            {
                A(i, i) += abs(A(i, j));
            }
            float scale = A(i, i);
            for (int j = 0; j < m; j++)
            {
                A(i, j) /= scale;
            }
        }
        // generate a unit vector
        for (int i = 0; i < m; i++)
        {
            x[i] = i + 1;
        }
        // generate b from A and x
        for (int i = 0; i < m; i++)
        {
            b[i] = 0;
            for (int j = 0; j < m; j++)
            {
                b[i] += A(i, j) * x[j];
            }
        }
        // convert A to COO format
        std::vector<int> row;
        std::vector<int> col;
        std::vector<float> val;
        for (int i = 0; i < m; i++)
        {
            for (int j = m / 2; j < m; j++)
            {
                row.push_back(i);
                col.push_back(j);
                val.push_back(A(i, j));
            }
            for (int j = 0; j < m / 2; j++)
            {
                row.push_back(i);
                col.push_back(j);
                val.push_back(A(i, j));
            }
        }
        // solve Ax = b
        GArr<int> A_rows, A_cols;
        GArr<float> A_vals, b_vals, x_vals;
        A_rows.assign(row);
        A_cols.assign(col);
        A_vals.assign(val);

        COOMatrix mat;
        mat.set_matrix(m, m, A_rows, A_cols, A_vals);
        mat.eliminate_zeros();
        mat.sort_by_row();

        b_vals.assign(b);
        solver.set_coo_matrix(mat);
        for (int _test_inner = 0; _test_inner < TEST_NUM; _test_inner++)
        {
            x_vals = solver.solve(b_vals);
            auto x_vals_cpu = x_vals.cpu();
            x_vals.clear();
            for (int i = 0; i < m; i++)
            {
                REQUIRE(x_vals_cpu[i] == Approx(x[i]).margin(5e-3));
            }
        }
    }
}

TEST_CASE("cusparse", "[cs]")
{
    test_sort();
    test_BiCGStab_solver();
}