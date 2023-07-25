/**************************  vector_convert.h   *******************************
* Author:        Agner Fog
* Date created:  2014-07-23
* Last modified: 2022-07-20
* Version:       2.02.00
* Project:       vector class library
* Description:
* Header file for conversion between different vector classes with different
* sizes. Also includes verious generic template functions.
*
* (c) Copyright 2012-2022 Agner Fog.
* Apache License version 2.0 or later.
*****************************************************************************/

#ifndef VECTOR_CONVERT_H
#define VECTOR_CONVERT_H

#ifndef VECTORCLASS_H
#include "vectorclass.h"
#endif

#if VECTORCLASS_H < 20200
#error Incompatible versions of vector class library mixed
#endif

#ifdef VCL_NAMESPACE
namespace VCL_NAMESPACE {
#endif

#if MAX_VECTOR_SIZE >= 256

/*****************************************************************************
*
*          Extend from 128 to 256 bit vectors
*
*****************************************************************************/

#if INSTRSET >= 8  // AVX2. 256 bit integer vectors

// sign extend
static inline Vec16s extend (Vec16c const a) {
    return _mm256_cvtepi8_epi16(a);
}

// zero extend
static inline Vec16us extend (Vec16uc const a) {
    return _mm256_cvtepu8_epi16(a);
}

// sign extend
static inline Vec8i extend (Vec8s const a) {
    return _mm256_cvtepi16_epi32(a);
}

// zero extend
static inline Vec8ui extend (Vec8us const a) {
    return _mm256_cvtepu16_epi32(a);
}

// sign extend
static inline Vec4q extend (Vec4i const a) {
    return _mm256_cvtepi32_epi64(a);
}

// zero extend
static inline Vec4uq extend (Vec4ui const a) {
    return _mm256_cvtepu32_epi64(a);
}


#else  // no AVX2. 256 bit integer vectors are emulated

// sign extend and zero extend functions:
static inline Vec16s extend (Vec16c const a) {
    return Vec16s(extend_low(a), extend_high(a));
}

static inline Vec16us extend (Vec16uc const a) {
    return Vec16us(extend_low(a), extend_high(a));
}

static inline Vec8i extend (Vec8s const a) {
    return Vec8i(extend_low(a), extend_high(a));
}

static inline Vec8ui extend (Vec8us const a) {
    return Vec8ui(extend_low(a), extend_high(a));
}

static inline Vec4q extend (Vec4i const a) {
    return Vec4q(extend_low(a), extend_high(a));
}

static inline Vec4uq extend (Vec4ui const a) {
    return Vec4uq(extend_low(a), extend_high(a));
}

#endif  // AVX2

/*****************************************************************************
*
*          Conversions between float and double
*
*****************************************************************************/
#if INSTRSET >= 7  // AVX. 256 bit float vectors

// float to double
static inline Vec4d to_double (Vec4f const a) {
    return _mm256_cvtps_pd(a);
}

// double to float
static inline Vec4f to_float (Vec4d const a) {
    return _mm256_cvtpd_ps(a);
}

#else  // no AVX2. 256 bit float vectors are emulated

// float to double
static inline Vec4d to_double (Vec4f const a) {
    Vec2d lo = _mm_cvtps_pd(a);
    Vec2d hi = _mm_cvtps_pd(_mm_movehl_ps(a, a));
    return Vec4d(lo,hi);
}

// double to float
static inline Vec4f to_float (Vec4d const a) {
    Vec4f lo = _mm_cvtpd_ps(a.get_low());
    Vec4f hi = _mm_cvtpd_ps(a.get_high());
    return _mm_movelh_ps(lo, hi);
}

#endif

/*****************************************************************************
*
*          Reduce from 256 to 128 bit vectors
*
*****************************************************************************/
#if INSTRSET >= 10  // AVX512VL

// compress functions. overflow wraps around
static inline Vec16c compress (Vec16s const a) {
    return _mm256_cvtepi16_epi8(a);
}

static inline Vec16uc compress (Vec16us const a) {
    return _mm256_cvtepi16_epi8(a);
}

static inline Vec8s compress (Vec8i const a) {
    return _mm256_cvtepi32_epi16(a);
}

static inline Vec8us compress (Vec8ui const a) {
    return _mm256_cvtepi32_epi16(a);
}

static inline Vec4i compress (Vec4q const a) {
    return _mm256_cvtepi64_epi32(a);
}

static inline Vec4ui compress (Vec4uq const a) {
    return _mm256_cvtepi64_epi32(a);
}

// compress_saturated functions. overflow saturates
static inline Vec16c compress_saturated (Vec16s const a) {
    return _mm256_cvtsepi16_epi8(a);
}

static inline Vec16uc compress_saturated (Vec16us const a) {
    return _mm256_cvtusepi16_epi8(a);
}

static inline Vec8s compress_saturated (Vec8i const a) {
    return _mm256_cvtsepi32_epi16(a);
}

static inline Vec8us compress_saturated (Vec8ui const a) {
    return _mm256_cvtusepi32_epi16(a);
}

static inline Vec4i compress_saturated (Vec4q const a) {
    return _mm256_cvtsepi64_epi32(a);
}

static inline Vec4ui compress_saturated (Vec4uq const a) {
    return _mm256_cvtusepi64_epi32(a);
}


#else  // no AVX512

// compress functions. overflow wraps around
static inline Vec16c compress (Vec16s const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec16uc compress (Vec16us const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec8s compress (Vec8i const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec8us compress (Vec8ui const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec4i compress (Vec4q const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec4ui compress (Vec4uq const a) {
    return compress(a.get_low(), a.get_high());
}

// compress_saturated functions. overflow saturates
static inline Vec16c compress_saturated (Vec16s const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec16uc compress_saturated (Vec16us const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec8s compress_saturated (Vec8i const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec8us compress_saturated (Vec8ui const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec4i compress_saturated (Vec4q const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec4ui compress_saturated (Vec4uq const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

#endif  // AVX512

#endif // MAX_VECTOR_SIZE >= 256


#if MAX_VECTOR_SIZE >= 512

/*****************************************************************************
*
*          Reduce from 512 to 256 bit vectors
*
*****************************************************************************/
#if INSTRSET >= 10  // AVX512VL

// compress_saturated functions. overflow saturates
static inline Vec32c compress_saturated (Vec32s const a) {
    return _mm512_cvtsepi16_epi8(a);
}

static inline Vec32uc compress_saturated (Vec32us const a) {
    return _mm512_cvtusepi16_epi8(a);
}

static inline Vec16s compress_saturated (Vec16i const a) {
    return _mm512_cvtsepi32_epi16(a);
}

static inline Vec16us compress_saturated (Vec16ui const a) {
    return _mm512_cvtusepi32_epi16(a);
}

static inline Vec8i compress_saturated (Vec8q const a) {
    return _mm512_cvtsepi64_epi32(a);
}

static inline Vec8ui compress_saturated (Vec8uq const a) {
    return _mm512_cvtusepi64_epi32(a);
}

#else  // no AVX512

// compress_saturated functions. overflow saturates
static inline Vec32c compress_saturated (Vec32s const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec32uc compress_saturated (Vec32us const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec16s compress_saturated (Vec16i const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec16us compress_saturated (Vec16ui const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec8i compress_saturated (Vec8q const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

static inline Vec8ui compress_saturated (Vec8uq const a) {
    return compress_saturated(a.get_low(), a.get_high());
}

#endif  // AVX512

/*****************************************************************************
*
*          Extend from 256 to 512 bit vectors
*
*****************************************************************************/

#if INSTRSET >= 9  // AVX512. 512 bit integer vectors

// sign extend
static inline Vec32s extend (Vec32c const a) {
#if INSTRSET >= 10
    return _mm512_cvtepi8_epi16(a);
#else
    return Vec32s(extend_low(a), extend_high(a));
#endif
}

// zero extend
static inline Vec32us extend (Vec32uc const a) {
#if INSTRSET >= 10
    return _mm512_cvtepu8_epi16(a);
#else
    return Vec32us(extend_low(a), extend_high(a));
#endif
}

// sign extend
static inline Vec16i extend (Vec16s const a) {
    return _mm512_cvtepi16_epi32(a);
}

// zero extend
static inline Vec16ui extend (Vec16us const a) {
    return _mm512_cvtepu16_epi32(a);
}

// sign extend
static inline Vec8q extend (Vec8i const a) {
    return _mm512_cvtepi32_epi64(a);
}

// zero extend
static inline Vec8uq extend (Vec8ui const a) {
    return _mm512_cvtepu32_epi64(a);
}

#else  // no AVX512. 512 bit vectors are emulated



// sign extend
static inline Vec32s extend (Vec32c const a) {
    return Vec32s(extend_low(a), extend_high(a));
}

// zero extend
static inline Vec32us extend (Vec32uc const a) {
    return Vec32us(extend_low(a), extend_high(a));
}

// sign extend
static inline Vec16i extend (Vec16s const a) {
    return Vec16i(extend_low(a), extend_high(a));
}

// zero extend
static inline Vec16ui extend (Vec16us const a) {
    return Vec16ui(extend_low(a), extend_high(a));
}

// sign extend
static inline Vec8q extend (Vec8i const a) {
    return Vec8q(extend_low(a), extend_high(a));
}

// zero extend
static inline Vec8uq extend (Vec8ui const a) {
    return Vec8uq(extend_low(a), extend_high(a));
}

#endif  // AVX512


/*****************************************************************************
*
*          Reduce from 512 to 256 bit vectors
*
*****************************************************************************/
#if INSTRSET >= 9  // AVX512F

// compress functions. overflow wraps around
static inline Vec32c compress (Vec32s const a) {
#if INSTRSET >= 10  // AVVX512BW
    return _mm512_cvtepi16_epi8(a);
#else
    return compress(a.get_low(), a.get_high());
#endif
}

static inline Vec32uc compress (Vec32us const a) {
    return Vec32uc(compress(Vec32s(a)));
}

static inline Vec16s compress (Vec16i const a) {
    return _mm512_cvtepi32_epi16(a);
}

static inline Vec16us compress (Vec16ui const a) {
    return _mm512_cvtepi32_epi16(a);
}

static inline Vec8i compress (Vec8q const a) {
    return _mm512_cvtepi64_epi32(a);
}

static inline Vec8ui compress (Vec8uq const a) {
    return _mm512_cvtepi64_epi32(a);
}

#else  // no AVX512

// compress functions. overflow wraps around
static inline Vec32c compress (Vec32s const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec32uc compress (Vec32us const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec16s compress (Vec16i const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec16us compress (Vec16ui const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec8i compress (Vec8q const a) {
    return compress(a.get_low(), a.get_high());
}

static inline Vec8ui compress (Vec8uq const a) {
    return compress(a.get_low(), a.get_high());
}

#endif  // AVX512

/*****************************************************************************
*
*          Conversions between float and double
*
*****************************************************************************/

#if INSTRSET >= 9  // AVX512. 512 bit float vectors

// float to double
static inline Vec8d to_double (Vec8f const a) {
    return _mm512_cvtps_pd(a);
}

// double to float
static inline Vec8f to_float (Vec8d const a) {
    return _mm512_cvtpd_ps(a);
}

#else  // no AVX512. 512 bit float vectors are emulated

// float to double
static inline Vec8d to_double (Vec8f const a) {
    Vec4d lo = to_double(a.get_low());
    Vec4d hi = to_double(a.get_high());
    return Vec8d(lo,hi);
}

// double to float
static inline Vec8f to_float (Vec8d const a) {
    Vec4f lo = to_float(a.get_low());
    Vec4f hi = to_float(a.get_high());
    return Vec8f(lo, hi);
}

#endif

#endif // MAX_VECTOR_SIZE >= 512

// double to float
static inline Vec4f to_float (Vec2d const a) {
    return _mm_cvtpd_ps(a);
}


/*****************************************************************************
*
*          Generic template functions
*
*  These templates define functions for multiple vector types in one template
*
*****************************************************************************/

// concatenate two vectors into one vector of double size
template <typename T> auto concatenate2(T const a, T const b) {
    static_assert(sizeof(T) * 8 < MAX_VECTOR_SIZE, "Maximum vector size exceeded");
    return decltype(extend_z(a))(a, b);   // call constructor for double size vector type
}


// horizontal min/max of vector elements
// implemented with universal template, works for all vector types:

template <typename T> auto horizontal_min(T const x) {
    if constexpr (T::elementtype() >= 15) {
        // T is a float or double vector
        if (horizontal_or(is_nan(x))) {
            // check for NAN because min does not guarantee NAN propagation
            return x[horizontal_find_first(is_nan(x))];
        }
    }
    return horizontal_min1(x);
}

template <typename T> auto horizontal_min1(T const x) {
    if constexpr (T::elementtype() <= 3) {       // boolean vector type
        return horizontal_and(x);
    }
    else if constexpr (sizeof(T) >= 32) {
        // split recursively into smaller vectors
        return horizontal_min1(min(x.get_low(), x.get_high()));
    }
    else if constexpr (T::size() == 2) {
        T a = permute2 <1, V_DC>(x);             // high half
        T b = min(a, x);
        return b[0];
    }
    else if constexpr (T::size() == 4) {
        T a = permute4<2, 3, V_DC, V_DC>(x);     // high half
        T b = min(a, x);
        a = permute4<1, V_DC, V_DC, V_DC>(b);
        b = min(a, b);
        return b[0];
    }
    else if constexpr (T::size() == 8) {
        T a = permute8<4, 5, 6, 7, V_DC, V_DC, V_DC, V_DC>(x);  // high half
        T b = min(a, x);
        a = permute8<2, 3, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = min(a, b);
        a = permute8<1, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = min(a, b);
        return b[0];
    }
    else {
        static_assert(T::size() == 16);          // no other size is allowed
        T a = permute16<8, 9, 10, 11, 12, 13, 14, 15, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC >(x);  // high half
        T b = min(a, x);
        a = permute16<4, 5, 6, 7, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = min(a, b);
        a = permute16<2, 3, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = min(a, b);
        a = permute16<1, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = min(a, b);
        return b[0];
    }
}

template <typename T> auto horizontal_max(T const x) {
    if constexpr (T::elementtype() >= 15) {
        // T is a float or double vector
        if (horizontal_or(is_nan(x))) {
            // check for NAN because max does not guarantee NAN propagation
            return x[horizontal_find_first(is_nan(x))];
        }
    }
    return horizontal_max1(x);
}

template <typename T> auto horizontal_max1(T const x) {
    if constexpr (T::elementtype() <= 3) {       // boolean vector type
        return horizontal_or(x);
    }
    else if constexpr (sizeof(T) >= 32) {
        // split recursively into smaller vectors
        return horizontal_max1(max(x.get_low(), x.get_high()));
    }
    else if constexpr (T::size() == 2) {
        T a = permute2 <1, V_DC>(x);             // high half
        T b = max(a, x);
        return b[0];
    }
    else if constexpr (T::size() == 4) {
        T a = permute4<2, 3, V_DC, V_DC>(x);     // high half
        T b = max(a, x);
        a = permute4<1, V_DC, V_DC, V_DC>(b);
        b = max(a, b);
        return b[0];
    }
    else if constexpr (T::size() == 8) {
        T a = permute8<4, 5, 6, 7, V_DC, V_DC, V_DC, V_DC>(x);  // high half
        T b = max(a, x);
        a = permute8<2, 3, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = max(a, b);
        a = permute8<1, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = max(a, b);
        return b[0];
    }
    else {
        static_assert(T::size() == 16);          // no other size is allowed
        T a = permute16<8, 9, 10, 11, 12, 13, 14, 15, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC >(x);  // high half
        T b = max(a, x);
        a = permute16<4, 5, 6, 7, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = max(a, b);
        a = permute16<2, 3, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = max(a, b);
        a = permute16<1, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>(b);
        b = max(a, b);
        return b[0];
    }
}

// Find first element that is true in a boolean vector
template <typename V>
static inline int horizontal_find_first(V const x) {
    static_assert(V::elementtype() == 2 || V::elementtype() == 3, "Boolean vector expected");
    auto bits = to_bits(x);                      // convert to bits
    if (bits == 0) return -1;
    if constexpr (V::size() < 32) {
        return bit_scan_forward((uint32_t)bits);
    }
    else {
        return bit_scan_forward(bits);
    }
}

// Count the number of elements that are true in a boolean vector
template <typename V>
static inline int horizontal_count(V const x) {
    static_assert(V::elementtype() == 2 || V::elementtype() == 3, "Boolean vector expected");
    auto bits = to_bits(x);                      // convert to bits
    if constexpr (V::size() < 32) {
        return vml_popcnt((uint32_t)bits);
    }
    else {
        return (int)vml_popcnt(bits);
    }
}

// maximum and minimum functions. This version is sure to propagate NANs,
// conforming to the new IEEE-754 2019 standard
template <typename V>
static inline V maximum(V const a, V const b) {
    if constexpr (V::elementtype() < 15) {
        return max(a, b);              // integer type
    }
    else {                             // float or double vector
        V y = select(is_nan(a), a, max(a, b));
#ifdef SIGNED_ZERO                     // pedantic about signed zero
        y = select(a == b, a & b, y);  // maximum(+0, -0) = +0
#endif
        return y;
    }
}

template <typename V>
static inline V minimum(V const a, V const b) {
    if constexpr (V::elementtype() < 15) {
        return min(a, b);              // integer type
    }
    else {                             // float or double vector
        V y = select(is_nan(a), a, min(a, b));
#ifdef SIGNED_ZERO                     // pedantic about signed zero
        y = select(a == b, a | b, y);  // minimum(+0, -0) = -0
#endif
        return y;
    }
}

// floating point remainder
// -denominator/2 <= result < denominator/2
template <typename V>
static inline V fremainder(V const numerator, double const denominator) {
    // (Optimization notice: Calculation of 1/denominator and constants for extended precision reduction
    // may be optimized by a compiler moving loop-invariant code. This is intended)
    static_assert(V::elementtype() == 16 || V::elementtype() == 17, "wrong vector type"); // supports only float and double
    if (denominator > 0.) {                                // denominator must be positive
        if constexpr (V::elementtype() == 16) {            // float
#ifdef __FMA__
            float recipd = float(1.0 / denominator);       // reciprocal denominator
            float fd = float(denominator);                 // denominator rounded to single precision
            float d2 = float(denominator - fd);            // remaining bits for double precision
            V q = round(numerator * recipd);               // divide and round
            V m = nmul_add(q, d2, nmul_add(q, fd, numerator));// double precision reduction
#else  // no FMA. Use extended precision reduction
            union {
                float f;
                uint32_t i;
            } u;
            u.f = float(denominator);
            u.i &= 0xFFFFF000;                             // remove 12 least significant bits for extended precision reduction
            float d2 = denominator - u.f;                  // remaining bits
            float recipd = float(1.0 / denominator);       // reciprocal
            V q = round(numerator * recipd);               // divide and round
            V m = nmul_add(q, d2, nmul_add(q, u.f, numerator));// extended precision reduction
#endif  // FMA
            if (true) { // Check that result is within desired interval. This may be omitted if not essential:
                // This check may be needed in extreme cases of numerator > 1.E5 * denominator
                auto too_high = m >= float( denominator * 0.5);
                auto too_low  = m <  float(-denominator * 0.5);
                m = if_sub(too_high, m, float(denominator));
                m = if_add(too_low,  m, float(denominator));
            }
            return m;
        }
        else if constexpr (V::elementtype() == 17) {       // double precision
#ifdef __FMA__
            double recipd = 1.0 / denominator;             // reciprocal
            V q = round(numerator * recipd);               // divide and round
            V m = nmul_add(q, denominator, numerator);     // nmul_add has extended precision
#else  // no FMA. Use extended precision reduction
            union {
                double f;
                uint64_t i;
            } u;
            u.f = denominator;
            u.i &= 0xFFFFFFFFFF000000;                     // remove 24 least significant bits for extended precision reduction
            double d2 = denominator - u.f;                 // remaining bits
            double recipd = 1.0 / denominator;             // reciprocal
            V q = round(numerator * recipd);               // divide and round
            V m = nmul_add(q, d2, nmul_add(q, u.f, numerator));// extended precision reduction
#endif  // FMA
            if (true) { // Check that result is within desired interval. This may be omitted if not essential:
                // This check is rarely needed except in extreme cases of numerator > 1.E14 * denominator
                auto too_high = m >=  denominator * 0.5;
                auto too_low  = m <  -denominator * 0.5;
                m = if_sub(too_high, m, denominator);
                m = if_add(too_low,  m, denominator);
            }
            return m;
        }
    }
    else {
        return nan_vec<V>(1);                              //  denominator is not positive
    }
}

// floating point modulo
// 0 <= result < denominator
template <typename V>
static inline V fmodulo(V const numerator, double const denominator) {
    // (Optimization notice: Calculation of 1/denominator and constants for extended precision reduction
    // may be optimized by a compiler moving loop-invariant code. This is intended)
    static_assert(V::elementtype() == 16 || V::elementtype() == 17, "wrong vector type"); // supports only float and double
    if (denominator > 0.) {                                // denominator must be positive
        if constexpr (V::elementtype() == 16) {            // float
#ifdef __FMA__
            float recipd = float(1.0 / denominator);       // reciprocal denominator
            float fd = float(denominator);                 // denominator rounded to single precision
            float d2 = float(denominator - fd);            // remaining bits for double precision
            V q = floor(numerator * recipd);               // divide and floor
            V m = nmul_add(q, d2, nmul_add(q, fd, numerator));// double precision reduction
#else  // no FMA. Use extended precision reduction
            union {
                float f;
                uint32_t i;
            } u;
            u.f = float(denominator);
            u.i &= 0xFFFFF000;                             // remove 12 least significant bits for extended precision reduction
            float d2 = denominator - u.f;                  // remaining bits
            float recipd = float(1.0 / denominator);       // reciprocal
            V q = floor(numerator * recipd);               // divide and floor
            V m = nmul_add(q, d2, nmul_add(q, u.f, numerator));// extended precision reduction
#endif  // FMA
            if (true) { // Check that result is within desired interval. This may be omitted if not essential:
                // This check may be needed in extreme cases of numerator > 1.E5 * denominator
                auto too_high = m >= float(denominator);
                auto too_low  = m <  0.f;
                m = if_sub(too_high, m, float(denominator));
                m = if_add(too_low,  m, float(denominator));
            }
            return m;
        }
        else if constexpr (V::elementtype() == 17) {       // double precision
#ifdef __FMA__
            double recipd = 1.0 / denominator;             // reciprocal
            V q = floor(numerator * recipd);               // divide and floor
            V m = nmul_add(q, denominator, numerator);     // nmul_add has extended precision
#else  // no FMA. Use extended precision reduction
            union {
                double f;
                uint64_t i;
            } u;
            u.f = denominator;
            u.i &= 0xFFFFFFFFFF000000;                     // remove 24 least significant bits for extended precision reduction
            double d2 = denominator - u.f;                 // remaining bits
            double recipd = 1.0 / denominator;             // reciprocal
            V q = floor(numerator * recipd);               // divide and floor
            V m = nmul_add(q, d2, nmul_add(q, u.f, numerator));// extended precision reduction
#endif  // FMA
            if (true) { // Check that result is within desired interval. This may be omitted if not essential:
                // This check is rarely needed except in extreme cases of numerator > 1.E14 * denominator
                auto too_high = m >= denominator;
                auto too_low  = m <  0.;
                m = if_sub(too_high, m, denominator);
                m = if_add(too_low,  m, denominator);
            }
            return m;
        }
    }
    else {
        return nan_vec<V>(1);                              //  denominator is not positive
    }
}

#ifdef VCL_NAMESPACE
}
#endif

#endif // VECTOR_CONVERT_H
