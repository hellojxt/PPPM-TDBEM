#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "array3D.h"

namespace pppm
{

GArr3D<float> inverse_from_svd(GArr2D<float> S, GArr3D<float> U, GArr3D<float> V);

class SVDResult
{
    public:
        GArr2D<float> S;
        GArr3D<float> U;
        GArr3D<float> V;
        GArr<int> info;
        GArr3D<float> inv_A;

        SVDResult(GArr2D<float> &S, GArr3D<float> &U, GArr3D<float> &V, GArr<int> &info)
            : S(S), U(U), V(V), info(info){};

        void solve_inverse() { inv_A = inverse_from_svd(S, U, V); }

        void clear()
        {
            S.clear();
            U.clear();
            V.clear();
            info.clear();
            inv_A.clear();
        }
};

SVDResult cusolver_svd(GArr3D<float> A);

};  // namespace pppm