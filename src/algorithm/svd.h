#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "array3D.h"

namespace pppm
{

class SVDResult
{
    public:
        GArr2D<float> S;
        GArr3D<float> U;
        GArr3D<float> V;
        SVDResult(GArr2D<float> &S, GArr3D<float> &U, GArr3D<float> &V)
        {
            this->S = S;
            this->U = U;
            this->V = V;
        }
};

SVDResult cusolver_svd(GArr3D<float> A);

};  // namespace pppm