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

TEST_CASE("cusparse", "[cs]")
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
            for (int j = 0; j < 10; j++)
            {
                int k = rand() % m;
                A(i, k) = rand() % 100;
            }
            A(i, i) = 0;
            for (int j = 0; j < m; j++)
            {
                A(i, i) += A(i, j);
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
            for (int j = 0; j < m; j++)
            {
                if (A(i, j) != 0)
                {
                    row.push_back(i);
                    col.push_back(j);
                    val.push_back(A(i, j));
                }
            }
        }
        // solve Ax = b
        GArr<int> A_rows, A_cols;
        GArr<float> A_vals, b_vals, x_vals;
        A_rows.assign(row);
        A_cols.assign(col);
        A_vals.assign(val);
        b_vals.assign(b);
        solver.set_coo_matrix(A_rows, A_cols, A_vals, m, m);
        for (int _test_inner = 0; _test_inner < TEST_NUM; _test_inner++)
        {
            x_vals = solver.solve(b_vals);
            auto x_vals_cpu = x_vals.cpu();
            x_vals.clear();
            for (int i = 0; i < m; i++)
            {
                REQUIRE(x_vals_cpu[i] == Approx(x[i]).margin(1e-3));
            }
        }
    }
}