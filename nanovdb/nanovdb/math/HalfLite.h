// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// The functions HalfToFloat and FloatToHalf are derived from Imath's
// imath_half_to_float / imath_float_to_half (Imath/half.h):
//   Copyright Contributors to the OpenEXR Project
//   SPDX-License-Identifier: BSD-3-Clause
//   Primary original authors: Florian Kainz, Rod Bogart

#ifndef NANOVDB_MATH_HALF_LITE_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_HALF_LITE_HAS_BEEN_INCLUDED

#include <nanovdb/util/Util.h>// for __hostdev__ and lots of other utility functions

namespace nanovdb {// =================================================================

namespace math {// =============================================================

// Standalone scalar conversions extracted from Half.h. Assumes IEEE 754
// binary32 floats and binary16 half bits; uses no intrinsics or lookup tables.
union HalfUnion
{
    uint32_t i;
    float f;
};

inline __hostdev__ float HalfToFloat(uint16_t h_raw)
{
    HalfUnion v;
    uint32_t hexpmant = ((uint32_t)(h_raw) << 17) >> 4;
    v.i = ((uint32_t)(h_raw >> 15)) << 31;

    if (hexpmant >= 0x00800000)
    {
        v.i |= hexpmant;
        if (hexpmant < 0x0f800000)
            v.i += 0x38000000;
        else
            v.i |= 0x7f800000;
    }
    else if (hexpmant != 0)
    {
        // Normalize a subnormal using a portable count-leading-zeros loop.
        uint32_t lc = 0;
        while (0 == ((hexpmant << lc) & 0x80000000))
            ++lc;
        lc -= 8;
        v.i |= 0x38800000;
        v.i |= (hexpmant << lc);
        v.i -= (lc << 23);
    }
    return v.f;
}

// Round to nearest, with ties to even. Overflow produces signed infinity.
inline __hostdev__ uint16_t FloatToHalf(float f)
{
    HalfUnion v;
    uint16_t h_raw;
    uint32_t e, m, ui, r, shift;

    v.f = f;
    ui = (v.i & ~0x80000000);
    h_raw = ((v.i >> 16) & 0x8000);

    if (ui >= 0x38800000)
    {
        // Infinity or NaN: preserve the sign and available payload bits.
        if (ui >= 0x7f800000)
        {
            h_raw |= 0x7c00;
            if (ui == 0x7f800000)
                return h_raw;
            m = (ui & 0x7fffff) >> 13;
            return h_raw | (uint16_t)m | (uint16_t)(m == 0);
        }

        if (ui > 0x477fefff)
            return h_raw | 0x7c00;

        ui -= 0x38000000;
        ui = ((ui + 0x00000fff + ((ui >> 13) & 1)) >> 13);
        return h_raw | (uint16_t)ui;
    }

    // Zero or a value that rounds to zero.
    if (ui < 0x33000001)
        return h_raw;

    // Produce a subnormal half.
    e = (ui >> 23);
    shift = 0x7e - e;
    m = 0x800000 | (ui & 0x7fffff);
    r = m << (32 - shift);
    h_raw |= (m >> shift);
    if (r > 0x80000000 || (r == 0x80000000 && (h_raw & 0x1) != 0))
        ++h_raw;
    return h_raw;
}

} // namespace math ===============================================================

} // namespace nanovdb ===================================================================

#endif // NANOVDB_MATH_HALF_LITE_HAS_BEEN_INCLUDED
