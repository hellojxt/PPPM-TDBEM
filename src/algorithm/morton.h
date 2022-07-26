#pragma once
#include "macro.h"

namespace pppm{
    typedef uint64_t morton;
    typedef uint Coord;

    CGPU_FUNC inline Coord morton3D_GetThirdBits(const morton m) {
        const morton masks[6] = { 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249 };
        morton x = m & masks[5];
        x = (x ^ (x >> 2)) & masks[4];
        x = (x ^ (x >> 4)) & masks[3];
        x = (x ^ (x >> 8)) & masks[2];
        x = (x ^ (x >> 16)) & masks[1];
        x = (x ^ ((morton)x >> 32)) & masks[0];
        return (Coord) x;
    }

    // DECODE 3D Morton code : Magic bits
    // This method splits the morton codes bits by using certain patterns (magic bits)
    CGPU_FUNC inline uint3 decode_morton(const morton m) {
        Coord z = morton3D_GetThirdBits(m);
        Coord y = morton3D_GetThirdBits(m >> 1);
        Coord x = morton3D_GetThirdBits(m >> 2);
        return make_uint3(x, y, z);
    }



    CGPU_FUNC inline morton morton3D_SplitBy3bits(const Coord a) {
        const morton masks[6] = { 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249 };
        morton x = ((morton)a) & masks[0];
        x = (x | (morton)x << 32) & masks[1]; 
        x = (x | x << 16) & masks[2];
        x = (x | x << 8)  & masks[3];
        x = (x | x << 4)  & masks[4];
        x = (x | x << 2)  & masks[5];
        return x;
    }

    // ENCODE 3D Morton code : Magic bits method
    // This method uses certain bit patterns (magic bits) to split bits in the Coordinates
    CGPU_FUNC inline morton encode_morton(const uint3 coord) {
        return morton3D_SplitBy3bits(coord.z) | (morton3D_SplitBy3bits(coord.y) << 1) | (morton3D_SplitBy3bits(coord.x) << 2);
    }

}
