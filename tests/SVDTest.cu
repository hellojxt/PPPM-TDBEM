#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "array.h"
#include "macro.h"
#include "svd.h"
using namespace pppm;

using Catch::Approx;

TEST_CASE("SVD", "[svd]")
{

    CArr3D<float> A(128, 8, 8);
    // A = rand(128, 8, 8);
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                A(i, j, k) = rand() / (float)RAND_MAX;
            }
        }
    }
    GArr3D<float> A_GPU(A);
    auto result = cusolver_svd(A_GPU);
    auto single_values = result.S.cpu();
    // check if the singular values are sorted
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            REQUIRE(single_values(i, j) >= single_values(i, j + 1));
        }
    }
    result.solve_inverse();
    auto A_inv = result.inv_A.cpu();
    // check A_inv * A = I
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                float sum = 0;
                for (int l = 0; l < 8; l++)
                {
                    sum += A_inv(i, j, l) * A(i, l, k);
                }
                if (j == k)
                {
                    REQUIRE(sum == Approx(1.0f).margin(1e-3));
                }
                else
                {
                    REQUIRE(sum == Approx(0.0f).margin(1e-3));
                }
            }
        }
    }
}