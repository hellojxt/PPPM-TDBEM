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

        SVDResult(GArr2D<float> &S, GArr3D<float> &U, GArr3D<float> &V, GArr<int> &info)
            : S(S), U(U), V(V), info(info){};

        void clear()
        {
            S.clear();
            U.clear();
            V.clear();
            info.clear();
            m_inv_A.clear();
        }

        GArr3D<float> get_inv_A()
        {
            if (m_inv_A.batchs == 0)
            {
                m_inv_A = inverse_from_svd(S, U, V);
            }
            return m_inv_A;
        }

    private:
        GArr3D<float> m_inv_A;
};

SVDResult cusolver_svd(GArr3D<float> A);

};  // namespace pppm