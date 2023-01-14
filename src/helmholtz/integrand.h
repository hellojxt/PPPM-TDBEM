#pragma once
#include "gauss/line.h"
#include "gauss/triangle.h"
#include "potential.h"

namespace pppm
{

inline CGPU_FUNC int triangle_common_vertex_num(int3 ind1, int3 ind2)
{
    return (ind1.x == ind2.x) + (ind1.x == ind2.y) + (ind1.x == ind2.z) + (ind1.y == ind2.x) + (ind1.y == ind2.y) +
           (ind1.y == ind2.z) + (ind1.z == ind2.x) + (ind1.z == ind2.y) + (ind1.z == ind2.z);
}

// normalized normal of a triangle
inline CGPU_FUNC float3 triangle_norm(float3 *verts)
{
    float3 v1 = verts[1] - verts[0];
    float3 v2 = verts[2] - verts[0];
    float3 n = cross(v1, v2);
    return n / length(n);
}

inline CGPU_FUNC float jacobian(float3 *v)
{
    return length(cross(v[1] - v[0], v[2] - v[0]));
}

inline CGPU_FUNC float jacobian(float3 v1, float3 v2, float3 v3)
{
    return length(cross(v2 - v1, v3 - v1));
}

inline CGPU_FUNC float jacobian(float3 *verts, int3 ind)
{
    return jacobian(verts[ind.x], verts[ind.y], verts[ind.z]);
}

// unit triangle (0, 0), (1, 0), (0, 1)
inline CGPU_FUNC float3 local_to_global(float x1, float x2, float3 *v)
{
    return (1 - x1 - x2) * v[0] + x1 * v[1] + x2 * v[2];
}

// unit triangle (0, 0), (1, 0), (1, 1)
inline CGPU_FUNC float3 local_to_global2(float x1, float x2, float3 *v)
{
    return (1 - x1) * v[0] + (x1 - x2) * v[1] + x2 * v[2];
}

inline CGPU_FUNC cpx singular_point(float xsi,
                                    float eta1,
                                    float eta2,
                                    float eta3,
                                    float weight,
                                    float3 *trial_v,
                                    float3 *test_v,
                                    float3 trial_norm,
                                    int neighbor_num,
                                    cpx s,
                                    PotentialType type)
{
    cpx result = cpx(0, 0);
    xsi = 0.5 * (xsi + 1);
    eta1 = 0.5 * (eta1 + 1);
    eta2 = 0.5 * (eta2 + 1);
    eta3 = 0.5 * (eta3 + 1);
    switch (neighbor_num)
    {
        case 3:
        {  // Indentical Panels
            float w = xsi * xsi * xsi * eta1 * eta1 * eta2;
            float eta12 = eta1 * eta2;
            float eta123 = eta1 * eta2 * eta3;
            float3 v1, v2;
            // Region 1
            v1 = local_to_global2(xsi, xsi * (1.0 - eta1 + eta12), trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta123), xsi * (1.0 - eta1), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            // Region 2
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * (1.0 - eta1), trial_v);
            v2 = local_to_global2(xsi, xsi * (1.0 - eta1 + eta12), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            // Region 3
            v1 = local_to_global2(xsi, xsi * (eta1 - eta12 + eta123), trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta12), xsi * (eta1 - eta12), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            // Region 4
            v1 = local_to_global2(xsi * (1.0 - eta12), xsi * (eta1 - eta12), trial_v);
            v2 = local_to_global2(xsi, xsi * (eta1 - eta12 + eta123), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            // Region 5
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * (eta1 - eta123), trial_v);
            v2 = local_to_global2(xsi, xsi * (eta1 - eta12), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            // Region 6
            v1 = local_to_global2(xsi, xsi * (eta1 - eta12), trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta123), xsi * (eta1 - eta123), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            result = result * w * weight;
            break;
        }
        case 2:
        {  // Common Edge
            float w = xsi * xsi * xsi * eta1 * eta1;
            float eta12 = eta1 * eta2;
            float eta123 = eta1 * eta2 * eta3;
            float3 v1, v2;
            // Region 1
            v1 = local_to_global2(xsi, xsi * eta1 * eta3, trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            // Region 2
            v1 = local_to_global2(xsi, xsi * eta1, trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta123), xsi * eta1 * eta2 * (1 - eta3), test_v);
            result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
            // Region 3
            v1 = local_to_global2(xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2), trial_v);
            v2 = local_to_global2(xsi, xsi * eta123, test_v);
            result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
            // Region 4
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * eta12 * (1.0 - eta3), trial_v);
            v2 = local_to_global2(xsi, xsi * eta1, test_v);
            result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
            // Region 5
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * eta1 * (1.0 - eta2 * eta3), trial_v);
            v2 = local_to_global2(xsi, xsi * eta12, test_v);
            result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
            result = result * w * weight;
            break;
        }
        case 1:
        {  // Common Vertex
            float w = xsi * xsi * xsi * eta2;
            float3 v1, v2;
            // Region 1
            v1 = local_to_global2(xsi, xsi * eta1, trial_v);
            v2 = local_to_global2(xsi * eta2, xsi * eta2 * eta3, test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            // Region 2
            v1 = local_to_global2(xsi * eta2, xsi * eta2 * eta3, trial_v);
            v2 = local_to_global2(xsi, xsi * eta1, test_v);
            result += layer_potential(v1, v2, trial_norm, s, type);
            result = result * w * weight;
            break;
        }
    }
    return result;
}

inline CGPU_FUNC cpx singular_integrand(float3 *trial_v,
                                        float3 *test_v,
                                        float trial_jacobian,
                                        float test_jacobian,
                                        cpx s,
                                        int neighbor_num,
                                        PotentialType type)
{
    float guass_x[LINE_GAUSS_NUM] = LINE_GAUSS_XS;
    float guass_w[LINE_GAUSS_NUM] = LINE_GAUSS_WS;
    cpx result = cpx(0, 0);
    float3 trial_norm = triangle_norm(trial_v);
    for (int xsi_i = 0; xsi_i < LINE_GAUSS_NUM; xsi_i++)
        for (int eta1_i = 0; eta1_i < LINE_GAUSS_NUM; eta1_i++)
            for (int eta2_i = 0; eta2_i < LINE_GAUSS_NUM; eta2_i++)
                for (int eta3_i = 0; eta3_i < LINE_GAUSS_NUM; eta3_i++)
                {
                    result += singular_point(guass_x[xsi_i], guass_x[eta1_i], guass_x[eta2_i], guass_x[eta3_i],
                                             guass_w[xsi_i] * guass_w[eta1_i] * guass_w[eta2_i] * guass_w[eta3_i],
                                             trial_v, test_v, trial_norm, neighbor_num, s, type);
                }
    return result * trial_jacobian * test_jacobian / 16;
}

inline CGPU_FUNC cpx
regular_integrand(float3 *trial_v, float3 *test_v, float trial_jacobian, float test_jacobian, cpx s, PotentialType type)
{
    cpx result = cpx(0, 0);
    float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
    float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
    float3 trial_norm = triangle_norm(trial_v);
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
        for (int j = 0; j < TRI_GAUSS_NUM; j++)
        {
            float3 v1 = local_to_global(guass_x[i][0], guass_x[i][1], trial_v);
            float3 v2 = local_to_global(guass_x[j][0], guass_x[j][1], test_v);
            result += 0.25 * guass_w[i] * guass_w[j] * trial_jacobian * test_jacobian *
                      layer_potential(v1, v2, trial_norm, s, type);
        }
    return result;
}

inline CGPU_FUNC cpx potential_integrand(float3 point, float3 *src_v, float src_jacobian, cpx s, PotentialType type)
{
    cpx result = cpx(0, 0);
    float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
    float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
    float3 src_norm = triangle_norm(src_v);
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
    {
        float3 v_in_tri = local_to_global(guass_x[i][0], guass_x[i][1], src_v);
        result += 0.5 * guass_w[i] * src_jacobian * layer_potential(v_in_tri, point, src_norm, s, type);
    }
    return result;
}

inline CGPU_FUNC cpx
potential_integrand(TargetCoordArray &point, float3 *src_v, float src_jacobian, cpx s, PotentialType type)
{
    cpx result = cpx(0, 0);
    float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
    float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
    float3 src_norm = triangle_norm(src_v);
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
    {
        float3 v_in_tri = local_to_global(guass_x[i][0], guass_x[i][1], src_v);
        result += 0.5 * guass_w[i] * src_jacobian * layer_potential(v_in_tri, point, src_norm, s, type);
    }
    return result;
}

enum PairType
{
    FACE_TO_FACE,
    FACE_TO_POINT
};

inline CGPU_FUNC cpx
face2FaceIntegrand(const float3 *vertices, int3 src, int3 trg, cpx k, PotentialType type, float threshold)
{

    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    float3 trg_v[3] = {{vertices[trg.x]}, {vertices[trg.y]}, {vertices[trg.z]}};
    float trg_jacobian = jacobian(trg_v);
    // float3 src_center = (src_v[0] + src_v[1] + src_v[2]) / 3;
    // float3 trg_center = (trg_v[0] + trg_v[1] + trg_v[2]) / 3;
    // float distance = length(src_center - trg_center);
    // if (distance < threshold)
    //     trg = src;
    int neighbor_num = triangle_common_vertex_num(src, trg);
    if (neighbor_num == 0)
        return regular_integrand(src_v, trg_v, src_jacobian, trg_jacobian, k, type);
    else
        return singular_integrand(src_v, trg_v, src_jacobian, trg_jacobian, k, neighbor_num, type);
}

inline CGPU_FUNC cpx face2PointIntegrand(const float3 *vertices, int3 src, float3 trg, cpx k, PotentialType type)
{
    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    return potential_integrand(trg, src_v, src_jacobian, k, type);
}

inline CGPU_FUNC cpx
face2PointIntegrand(const float3 *vertices, int3 src, TargetCoordArray &trg, cpx k, PotentialType type)
{
    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    return potential_integrand(trg, src_v, src_jacobian, k, type);
    // float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    // float src_jacobian = jacobian(src_v);
    // cpx result = cpx(0, 0);
    // float guass_x[2] = {0.3333333333333330, 0.3333333333333330};
    // float guass_w[1] = {1.0f};
    // float3 src_norm = triangle_norm(src_v);
    // float3 v_in_tri = local_to_global(guass_x[0], guass_x[1], src_v);
    // result += 0.5 * guass_w[0] * src_jacobian * layer_potential(v_in_tri, trg, src_norm, k, type);
    // return result;
}

}  // namespace pppm