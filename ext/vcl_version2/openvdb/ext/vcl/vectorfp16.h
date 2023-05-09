/****************************  vectorfp16.h   *******************************
* Author:        Agner Fog
* Date created:  2022-05-03
* Last modified: 2022-07-20
* Version:       2.02.00
* Project:       vector class library
* Description:
* Header file defining half precision floating point vector classes
* Instruction sets AVX512_FP16 and AVX512VL required
*
* Instructions: see vcl_manual.pdf
*
* The following vector classes are defined here:
* Vec8h     Vector of  8 half precision floating point numbers in 128 bit vector
* Vec16h    Vector of 16 half precision floating point numbers in 256 bit vector
* Vec32h    Vector of 32 half precision floating point numbers in 512 bit vector
*
* This header file defines operators and functions for these vectors.
*
* You need a compiler supporting the AVX512_FP16 instruction to compile for this.
* This code works with the following compilers:
* clang++ version 14.0
* g++ version 12.1 with binutils version 2.34
* Intel c++ compiler version 2022.0
*
* (c) Copyright 2012-2022 Agner Fog.
* Apache License version 2.0 or later.
*****************************************************************************/

#ifndef VECTORFP16_H
#define VECTORFP16_H

#ifndef VECTORCLASS_H
#include "vectorclass.h"
#endif

#if VECTORCLASS_H < 20200
#error Incompatible versions of vector class library mixed
#endif

#if INSTRSET < 10 || !defined(__AVX512FP16__)
// half precision instructions not supported. Use emulation
#include "vectorfp16e.h"
#else

#ifdef VCL_NAMESPACE
namespace VCL_NAMESPACE {
#endif

// type Float16 emulates _Float16 in vectorfp16e.h if _Float16 not defined
typedef _Float16 Float16;  // Float16 needs no emulation


/*****************************************************************************
*
*          Vec8hb: Vector of 8 Booleans for use with Vec8h
*
*****************************************************************************/

typedef Vec8b Vec8hb;  // compact boolean vector


/*****************************************************************************
*
*          Vec8h: Vector of 8 half precision floating point values
*
*****************************************************************************/

class Vec8h {
protected:
    __m128h xmm; // Float vector
public:
    // Default constructor:
    Vec8h() = default;
    // Constructor to broadcast the same value into all elements:
    Vec8h(_Float16 f) {
        xmm = _mm_set1_ph (f);
    }
    // Constructor to build from all elements:
    Vec8h(_Float16 f0, _Float16 f1, _Float16 f2, _Float16 f3, _Float16 f4, _Float16 f5, _Float16 f6, _Float16 f7) {
        xmm = _mm_setr_ph (f0, f1, f2, f3, f4, f5, f6, f7);
    }
    // Constructor to convert from type __m128 used in intrinsics:
    Vec8h(__m128h const x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128 used in intrinsics:
    Vec8h & operator = (__m128h const x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128 used in intrinsics
    operator __m128h() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec8h & load(void const * p) {
        xmm = _mm_loadu_ph (p);
        return *this;
    }
    // Member function to load from array, aligned by 16
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 16. In most cases there is no difference in speed between load and load_a
    Vec8h & load_a(void const * p) {
        xmm = _mm_load_ph (p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        _mm_storeu_ph (p, xmm);
    }
    // Member function storing into array, aligned by 16
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 16.
    void store_a(void * p) const {
        _mm_store_ph (p, xmm);
    }
    // Member function storing to aligned uncached memory (non-temporal store).
    // This may be more efficient than store_a when storing large blocks of memory if it 
    // is unlikely that the data will stay in the cache until it is read again.
    // Note: Will generate runtime error if p is not aligned by 16
    void store_nt(void * p) const {
        _mm_stream_ps((float*)p, _mm_castph_ps(xmm));
    }
    // Partial load. Load n elements and set the rest to 0
    Vec8h & load_partial(int n, void const * p) {
        xmm = _mm_castsi128_ph(_mm_maskz_loadu_epi16(__mmask8((1u << n) - 1), p));
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        _mm_mask_storeu_epi16(p, __mmask8((1u << n) - 1), _mm_castph_si128(xmm));
    }
    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec8h & cutoff(int n) {
        xmm = _mm_castsi128_ph(_mm_maskz_mov_epi16(__mmask8((1u << n) - 1), _mm_castph_si128(xmm)));
        return *this;
    }
    // Member function to change a single element in vector
    Vec8h const insert(int index, _Float16 a) {
        __m128h aa = _mm_set1_ph (a);
        xmm = _mm_castsi128_ph(_mm_mask_mov_epi16(_mm_castph_si128(xmm), __mmask8(1u << index), _mm_castph_si128(aa)));
        return *this;
    }
    // Member function extract a single element from vector
    _Float16 extract(int index) const {
#if INSTRSET >= 10 && defined (__AVX512VBMI2__)
        __m128i x = _mm_maskz_compress_epi16(__mmask8(1u << index), _mm_castph_si128(xmm));
        return _mm_cvtsh_h(_mm_castsi128_ph(x));
#elif 0
        union {
            __m128h v;
            _Float16 f[8];
        } y;
        y.v = xmm;
        return y.f[index & 7];
#else
        Vec4ui x = _mm_maskz_compress_epi32(__mmask8(1u << (index >> 1)), _mm_castph_si128(xmm));  // extract int32_t
        x >>= (index & 1) << 4;  // get upper 16 bits if index odd
        return _mm_cvtsh_h(_mm_castsi128_ph(x));
#endif
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    _Float16 operator [] (int index) const {
        return extract(index);
    }
    static constexpr int size() {
        return 8;
    }
    static constexpr int elementtype() {
        return 15;
    }
    typedef __m128h registertype;
};


/*****************************************************************************
*
*          Operators for Vec8h
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec8h operator + (Vec8h const a, Vec8h const b) {
    return _mm_add_ph(a, b);
}

// vector operator + : add vector and scalar
static inline Vec8h operator + (Vec8h const a, _Float16 b) {
    return a + Vec8h(b);
}
static inline Vec8h operator + (_Float16 a, Vec8h const b) {
    return Vec8h(a) + b;
}

// vector operator += : add
static inline Vec8h & operator += (Vec8h & a, Vec8h const b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec8h operator ++ (Vec8h & a, int) {
    Vec8h a0 = a;
    a = a + _Float16(1.); // 1.0f16 not supported by g++ version 12.1
    return a0;
}

// prefix operator ++
static inline Vec8h & operator ++ (Vec8h & a) {
    a = a +  _Float16(1.);
    return a;
}

// vector operator - : subtract element by element
static inline Vec8h operator - (Vec8h const a, Vec8h const b) {
    return _mm_sub_ph(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec8h operator - (Vec8h const a, float b) {
    return a - Vec8h(b);
}
static inline Vec8h operator - (float a, Vec8h const b) {
    return Vec8h(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec8h operator - (Vec8h const a) {
    return _mm_castps_ph(_mm_xor_ps(_mm_castph_ps(a), _mm_castsi128_ps(_mm_set1_epi32(0x80008000))));
}

// vector operator -= : subtract
static inline Vec8h & operator -= (Vec8h & a, Vec8h const b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec8h operator -- (Vec8h & a, int) {
    Vec8h a0 = a;
    a = a -  _Float16(1.);
    return a0;
}

// prefix operator --
static inline Vec8h & operator -- (Vec8h & a) {
    a = a -  _Float16(1.);
    return a;
}

// vector operator * : multiply element by element
static inline Vec8h operator * (Vec8h const a, Vec8h const b) {
    return _mm_mul_ph(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec8h operator * (Vec8h const a, _Float16 b) {
    return a * Vec8h(b);
}
static inline Vec8h operator * (_Float16 a, Vec8h const b) {
    return Vec8h(a) * b;
}

// vector operator *= : multiply
static inline Vec8h & operator *= (Vec8h & a, Vec8h const b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec8h operator / (Vec8h const a, Vec8h const b) {
    return _mm_div_ph(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec8h operator / (Vec8h const a, _Float16 b) {
    return a / Vec8h(b);
}
static inline Vec8h operator / (_Float16 a, Vec8h const b) {
    return Vec8h(a) / b;
}

// vector operator /= : divide
static inline Vec8h & operator /= (Vec8h & a, Vec8h const b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec8hb operator == (Vec8h const a, Vec8h const b) {
    return _mm_cmp_ph_mask(a, b, 0);
}

// vector operator != : returns true for elements for which a != b
static inline Vec8hb operator != (Vec8h const a, Vec8h const b) {
    return _mm_cmp_ph_mask(a, b, 4);
}

// vector operator < : returns true for elements for which a < b
static inline Vec8hb operator < (Vec8h const a, Vec8h const b) {
    return _mm_cmp_ph_mask(a, b, 1);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec8hb operator <= (Vec8h const a, Vec8h const b) {
    return _mm_cmp_ph_mask(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
static inline Vec8hb operator > (Vec8h const a, Vec8h const b) {
    return _mm_cmp_ph_mask(a, b, 6+8);
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec8hb operator >= (Vec8h const a, Vec8h const b) {
    return _mm_cmp_ph_mask(a, b, 5+8);
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec8h operator & (Vec8h const a, Vec8h const b) {
    return _mm_castps_ph(_mm_and_ps(_mm_castph_ps(a), _mm_castph_ps(b)));
}

// vector operator &= : bitwise and
static inline Vec8h & operator &= (Vec8h & a, Vec8h const b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec8h and Vec8hb
static inline Vec8h operator & (Vec8h const a, Vec8hb const b) {
    return _mm_castsi128_ph(_mm_maskz_mov_epi16(b, _mm_castph_si128(a)));
}
static inline Vec8h operator & (Vec8hb const a, Vec8h const b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec8h operator | (Vec8h const a, Vec8h const b) {
    return _mm_castps_ph(_mm_or_ps(_mm_castph_ps(a), _mm_castph_ps(b)));
}

// vector operator |= : bitwise or
static inline Vec8h & operator |= (Vec8h & a, Vec8h const b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8h operator ^ (Vec8h const a, Vec8h const b) {
    return _mm_castps_ph(_mm_xor_ps(_mm_castph_ps(a), _mm_castph_ps(b)));
}

// vector operator ^= : bitwise xor
static inline Vec8h & operator ^= (Vec8h & a, Vec8h const b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec8hb operator ! (Vec8h const a) {
    return a == Vec8h(0.0);
}


/*****************************************************************************
*
*          Functions for Vec8h
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec8h select(Vec8hb const s, Vec8h const a, Vec8h const b) {
    return _mm_castsi128_ph(_mm_mask_mov_epi16(_mm_castph_si128(b), s, _mm_castph_si128(a)));
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8h if_add(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return _mm_mask_add_ph (a, f, a, b);
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
static inline Vec8h if_sub(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return _mm_mask_sub_ph (a, f, a, b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec8h if_mul(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return _mm_mask_mul_ph (a, f, a, b);
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] / b[i]) : a[i]
static inline Vec8h if_div(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return _mm_mask_div_ph (a, f, a, b);
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec8h(-0.0f16)) gives true, while Vec8h(-0.0f16) < Vec8h(0.0f16) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb sign_bit(Vec8h const a) {
    Vec8s t1 = _mm_castph_si128(a);    // reinterpret as 16-bit integer
    Vec8s t2 = t1 >> 15;               // extend sign bit
    return t2 != 0;
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec8h sign_combine(Vec8h const a, Vec8h const b) {
    return a ^ (b & Vec8h(_Float16(-0.0)));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb is_finite(Vec8h const a) {
    return __mmask8(_mm_fpclass_ph_mask(a, 0x99) ^ 0xFF);
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb is_inf(Vec8h const a) {
    return __mmask8(_mm_fpclass_ph_mask(a, 0x18));
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb is_nan(Vec8h const a) {
    // assume that compiler does not optimize this away with -ffinite-math-only:
    return Vec4fb(_mm_fpclass_ph_mask(a, 0x81));
}

// Function is_subnormal: gives true for elements that are subnormal
// false for finite numbers, zero, NAN and INF
static inline Vec8hb is_subnormal(Vec8h const a) {
    return Vec8hb(_mm_fpclass_ph_mask(a, 0x20));
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal
// false for finite numbers, NAN and INF
static inline Vec8hb is_zero_or_subnormal(Vec8h const a) {
    return Vec8hb(_mm_fpclass_ph_mask(a, 0x26));
}

// Function infinite8h: returns a vector where all elements are +INF
static inline Vec8h infinite8h() {
    return _mm_castsi128_ph(_mm_set1_epi16(0x7C00));
}

// template for producing quiet NAN
template <>
Vec8h nan_vec<Vec8h>(uint32_t payload) {
    if constexpr (Vec8h::elementtype() == 15) {  // _Float16
        union {
            uint16_t i;
            _Float16 f;
        } uf;
        uf.i = 0x7E00 | (payload & 0x01FF);
        return Vec8h(uf.f);
    }
} 

// Function nan8h: returns a vector where all elements are NAN (quiet)
static inline Vec8h nan8h(int n = 0x10) {
    return nan_vec<Vec8h>(n);
}

// This function returns the code hidden in a NAN. The sign bit is ignored
static inline Vec8us nan_code(Vec8h const x) {
    Vec8us a = Vec8us(_mm_castph_si128(x));
    Vec8us const n = 0x3FF;
    return select(is_nan(x), a & n, Vec8us(0));
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline _Float16 horizontal_add(Vec8h const a) {
    //return _mm_reduce_add_ph(a);
    __m128h b = _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(a), _mm_castph_ps(a)));
    __m128h c = _mm_add_ph(a, b);
    __m128h d = _mm_castps_ph(_mm_movehdup_ps( _mm_castph_ps(c)));    
    __m128h e = _mm_add_ph(c, d);
    __m128h f = _mm_castsi128_ph(_mm_shufflelo_epi16(_mm_castph_si128(e), 1));
    __m128h g = _mm_add_sh(e, f);
    return _mm_cvtsh_h(g);
}

#if MAX_VECTOR_SIZE >= 256
// same, with high precision
static inline float horizontal_add_x(Vec8h const a) {
    //Vec8f b = _mm256_cvtph_ps(a); // declaration of _mm256_cvtph_ps has __m128i parameter because it was defined before __m128h was defined
    Vec8f b = _mm256_cvtph_ps(_mm_castph_si128(a));
    return horizontal_add(b);
}
#endif

// function max: a > b ? a : b
static inline Vec8h max(Vec8h const a, Vec8h const b) {
    return _mm_max_ph(a, b);
}

// function min: a < b ? a : b
static inline Vec8h min(Vec8h const a, Vec8h const b) {
    return _mm_min_ph(a, b);
}
// NAN-safe versions of maximum and minimum are in vector_convert.h

// function abs: absolute value
static inline Vec8h abs(Vec8h const a) {
    return _mm_abs_ph(a);
}

// function sqrt: square root
static inline Vec8h sqrt(Vec8h const a) {
    return _mm_sqrt_ph(a);
}

// function square: a * a
static inline Vec8h square(Vec8h const a) {
    return a * a;
}

// The purpose of this template is to prevent implicit conversion of a float
// exponent to int when calling pow(vector, float) and vectormath_exp.h is not included
template <typename TT> static Vec8h pow(Vec8h const a, TT const n);  // = delete

// Raise floating point numbers to integer power n
template <>
inline Vec8h pow<int>(Vec8h const x0, int const n) {
    return pow_template_i<Vec8h>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec8h pow<uint32_t>(Vec8h const x0, uint32_t const n) {
    return pow_template_i<Vec8h>(x0, (int)n);
}

// Raise floating point numbers to integer power n, where n is a compile-time constant:
// Template in vectorf28.h is used
//template <typename V, int n>
//static inline V pow_n(V const a);

// implement as function pow(vector, const_int)
template <int n>
static inline Vec8h pow(Vec8h const a, Const_int_t<n>) {
    return pow_n<Vec8h, n>(a);
}

static inline Vec8h round(Vec8h const a) {
    return _mm_roundscale_ph (a, 8);
}

// function truncate: round towards zero. (result as float vector)
static inline Vec8h truncate(Vec8h const a) {
    return _mm_roundscale_ph(a, 3 + 8);
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec8h floor(Vec8h const a) {
    return _mm_roundscale_ph(a, 1 + 8);
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec8h ceil(Vec8h const a) {
    return _mm_roundscale_ph(a, 2 + 8);
}

// function roundi: round to nearest integer (even). (result as integer vector)
static inline Vec8s roundi(Vec8h const a) {
    // Note: assume MXCSR control register is set to rounding
    return _mm_cvtph_epi16(a);
}

// function truncatei: round towards zero. (result as integer vector)
static inline Vec8s truncatei(Vec8h const a) {
    return _mm_cvttph_epi16(a);
}

// function to_float: convert integer vector to float vector
static inline Vec8h to_float16(Vec8s const a) {
    return _mm_cvtepi16_ph(a);
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec8h to_float16(Vec8us const a) {
    return _mm_cvtepu16_ph(a);
}

// Approximate math functions

// reciprocal (almost exact)
static inline Vec8h approx_recipr(Vec8h const a) {
    return _mm_rcp_ph (a);
}

// reciprocal squareroot (almost exact)
static inline Vec8h approx_rsqrt(Vec8h const a) {
    return _mm_rsqrt_ph(a);
}

// Fused multiply and add functions

// Multiply and add. a*b+c
static inline Vec8h mul_add(Vec8h const a, Vec8h const b, Vec8h const c) {
    return _mm_fmadd_ph(a, b, c);
}

// Multiply and subtract. a*b-c
static inline Vec8h mul_sub(Vec8h const a, Vec8h const b, Vec8h const c) {
    return _mm_fmsub_ph(a, b, c);
}

// Multiply and inverse subtract
static inline Vec8h nmul_add(Vec8h const a, Vec8h const b, Vec8h const c) {
    return _mm_fnmadd_ph(a, b, c);
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec8s exponent(Vec8h const a) {
    Vec8us t1 = _mm_castph_si128(a);   // reinterpret as 16-bit integer
    Vec8us t2 = t1 << 1;               // shift out sign bit
    Vec8us t3 = t2 >> 11;              // shift down logical to position 0
    Vec8s  t4 = Vec8s(t3) - 0x0F;      // subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
static inline Vec8h fraction(Vec8h const a) {
    return _mm_getmant_ph(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  16 gives +INF
// n <= -15 gives 0.0f
// This function will never produce subnormals, and never raise exceptions
static inline Vec8h exp2(Vec8s const n) {
    Vec8s t1 = max(n, -15);            // limit to allowed range
    Vec8s t2 = min(t1, 16);
    Vec8s t3 = t2 + 15;                // add bias
    Vec8s t4 = t3 << 10;               // put exponent into position 10
    return _mm_castsi128_ph(t4);       // reinterpret as float
}
//static Vec8h exp2(Vec8h const x);    // defined in vectormath_exp.h ??


// change signs on vectors Vec8h
// Each index i0 - i7 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8h change_sign(Vec8h const a) {
    if constexpr ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7) == 0) return a;
    __m128i mask = constant4ui<
        (i0 ? 0x8000 : 0) | (i1 ? 0x80000000 : 0), 
        (i2 ? 0x8000 : 0) | (i3 ? 0x80000000 : 0), 
        (i4 ? 0x8000 : 0) | (i5 ? 0x80000000 : 0), 
        (i6 ? 0x8000 : 0) | (i7 ? 0x80000000 : 0) >();
    return  _mm_castps_ph(_mm_xor_ps(_mm_castph_ps(a), _mm_castsi128_ps(mask)));     // flip sign bits
}

/*****************************************************************************
*
*          conversion of precision
*
*****************************************************************************/

// conversions Vec8h <-> Vec4f
// extend precision: Vec8h -> Vec4f. upper half ignored
Vec4f convert8h_4f (Vec8h h) {
    return _mm_cvtph_ps(_mm_castph_si128(h));
}

// reduce precision: Vec4f -> Vec8h. upper half zero
Vec8h convert4f_8h (Vec4f f) {
    return _mm_castsi128_ph(_mm_cvtps_ph(f, 0));
}

#if MAX_VECTOR_SIZE >= 256
// conversions Vec8h <-> Vec8f
// extend precision: Vec8h -> Vec8f
Vec8f to_float (Vec8h h) {
    return _mm256_cvtph_ps(_mm_castph_si128(h));
}

// reduce precision: Vec8f -> Vec8h
Vec8h to_float16 (Vec8f f) {
    return _mm_castsi128_ph(_mm256_cvtps_ph(f, 0));
} 
#endif

/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/

static inline __m128i reinterpret_i(__m128h const x) {
    return _mm_castph_si128(x);
}

static inline __m128h  reinterpret_h(__m128i const x) {
    return _mm_castsi128_ph(x);
}

static inline __m128  reinterpret_f(__m128h const x) {
    return _mm_castph_ps(x);
}

static inline __m128d reinterpret_d(__m128h const x) {
    return _mm_castph_pd(x);
}




/*****************************************************************************
*
*          Vector permute and blend functions
*
******************************************************************************
*
* The permute function can reorder the elements of a vector and optionally
* set some elements to zero.
*
* See vectori128.h for details
*
*****************************************************************************/
// permute vector Vec8h
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8h permute8(Vec8h const a) {
    return _mm_castsi128_ph (permute8<i0, i1, i2, i3, i4, i5, i6, i7>(Vec8s(_mm_castph_si128(a))));
}


/*****************************************************************************
*
*          Vector blend functions
*
*****************************************************************************/

// permute and blend Vec8h
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8h blend8(Vec8h const a, Vec8h const b) {
    return _mm_castsi128_ph (blend8<i0, i1, i2, i3, i4, i5, i6, i7>(Vec8s(_mm_castph_si128(a)), Vec8s(_mm_castph_si128(b))));
}


/*****************************************************************************
*
*          Vector lookup functions
*
******************************************************************************
*
* These functions use vector elements as indexes into a table.
* The table is given as one or more vectors or as an array.
*
*****************************************************************************/

static inline Vec8h lookup8 (Vec8s const index, Vec8h const table) {
    return _mm_castsi128_ph(lookup8(index, Vec8s(_mm_castph_si128(table))));
}

static inline Vec8h lookup16(Vec8s const index, Vec8h const table0, Vec8h const table1) {
    return _mm_castsi128_ph(lookup16(index, Vec8s(_mm_castph_si128(table0)), Vec8s(_mm_castph_si128(table1))));
}

template <int n>
static inline Vec8h lookup(Vec8s const index, void const * table) {
    return _mm_castsi128_ph(lookup<n>(index, (void const *)(table)));
}


/*****************************************************************************
*
*          256 bit vectors
*
*****************************************************************************/

#if MAX_VECTOR_SIZE >= 256


/*****************************************************************************
*
*          Vec16hb: Vector of 16 Booleans for use with Vec16h
*
*****************************************************************************/

typedef Vec16b Vec16hb;  // compact boolean vector


/*****************************************************************************
*
*          Vec16h: Vector of 16 half precision floating point values
*
*****************************************************************************/

class Vec16h {
protected:
    __m256h ymm; // Float vector
public:
    // Default constructor:
    Vec16h() = default;
    // Constructor to broadcast the same value into all elements:
    Vec16h(_Float16 f) {
        ymm = _mm256_set1_ph (f);
    }
    // Constructor to build from all elements:
    Vec16h(_Float16 f0, _Float16 f1, _Float16 f2, _Float16 f3, _Float16 f4, _Float16 f5, _Float16 f6, _Float16 f7,
    _Float16 f8, _Float16 f9, _Float16 f10, _Float16 f11, _Float16 f12, _Float16 f13, _Float16 f14, _Float16 f15) {
        ymm = _mm256_setr_ph (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15);
    }
    // Constructor to build from two Vec8h:
    Vec16h(Vec8h const a0, Vec8h const a1) {     
        ymm = _mm256_castps_ph(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castph_ps(a0)),_mm_castph_ps(a1),1));
    }
    // Constructor to convert from type __m256h used in intrinsics:
    Vec16h(__m256h const x) {
        ymm = x;
    }
    // Assignment operator to convert from type __m256h used in intrinsics:
    Vec16h & operator = (__m256h const x) {
        ymm = x;
        return *this;
    }
    // Type cast operator to convert to __m256h used in intrinsics
    operator __m256h() const {
        return ymm;
    }
    // Member function to load from array (unaligned)
    Vec16h & load(void const * p) {
        ymm = _mm256_loadu_ph (p);
        return *this;
    }
    // Member function to load from array, aligned by 32
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 32. In most cases there is no difference in speed between load and load_a
    Vec16h & load_a(void const * p) {
        ymm = _mm256_load_ph (p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        _mm256_storeu_ph (p, ymm);
    }
    // Member function storing into array, aligned by 32
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 32.
    void store_a(void * p) const {
        _mm256_store_ph (p, ymm);
    }
    // Member function storing to aligned uncached memory (non-temporal store).
    // This may be more efficient than store_a when storing large blocks of memory if it 
    // is unlikely that the data will stay in the cache until it is read again.
    // Note: Will generate runtime error if p is not aligned by 32
    void store_nt(void * p) const {
        _mm256_stream_ps((float*)p, _mm256_castph_ps(ymm));
    }
    // Partial load. Load n elements and set the rest to 0
    Vec16h & load_partial(int n, void const * p) {
        ymm = _mm256_castsi256_ph(_mm256_maskz_loadu_epi16(__mmask16((1u << n) - 1), p));
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        _mm256_mask_storeu_epi16(p, __mmask16((1u << n) - 1), _mm256_castph_si256(ymm));
    }
    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec16h & cutoff(int n) {
        ymm = _mm256_castsi256_ph(_mm256_maskz_mov_epi16(__mmask16((1u << n) - 1), _mm256_castph_si256(ymm)));
        return *this;
    }
    // Member function to change a single element in vector
    Vec16h const insert(int index, _Float16 a) {
        __m256h aa = _mm256_set1_ph (a);
        ymm = _mm256_castsi256_ph(_mm256_mask_mov_epi16(_mm256_castph_si256(ymm), __mmask16(1u << index), _mm256_castph_si256(aa)));
        return *this;
    }
    // Member function extract a single element from vector
    _Float16 extract(int index) const {
#if INSTRSET >= 10 && defined (__AVX512VBMI2__)
        __m256i x = _mm256_maskz_compress_epi16(__mmask16(1u << index), _mm256_castph_si256(ymm));
        return _mm256_cvtsh_h(_mm256_castsi256_ph(x));
#elif 0
        union {
            __m256h v;
            _Float16 f[16];
        } y;
        y.v = ymm;
        return y.f[index & 15];
#else
        Vec8ui x = _mm256_maskz_compress_epi32(__mmask16(1u << (index >> 1)), _mm256_castph_si256(ymm));  // extract int32_t
        x >>= uint32_t((index & 1) << 4);  // get upper 16 bits if index odd
        return _mm256_cvtsh_h(_mm256_castsi256_ph(x));
#endif
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    _Float16 operator [] (int index) const {
        return extract(index);
    }
    Vec8h get_low() const {
        return _mm256_castph256_ph128(ymm);
    }
    Vec8h get_high() const {
        return _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(ymm),1));
    }
    static constexpr int size() {
        return 16;
    }
    static constexpr int elementtype() {
        return 15;
    }
    typedef __m256h registertype;
};


/*****************************************************************************
*
*          Operators for Vec16h
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec16h operator + (Vec16h const a, Vec16h const b) {
    return _mm256_add_ph(a, b);
}

// vector operator + : add vector and scalar
static inline Vec16h operator + (Vec16h const a, _Float16 b) {
    return a + Vec16h(b);
}
static inline Vec16h operator + (_Float16 a, Vec16h const b) {
    return Vec16h(a) + b;
}

// vector operator += : add
static inline Vec16h & operator += (Vec16h & a, Vec16h const b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec16h operator ++ (Vec16h & a, int) {
    Vec16h a0 = a;
    a = a +  _Float16(1.);
    return a0;
}

// prefix operator ++
static inline Vec16h & operator ++ (Vec16h & a) {
    a = a +  _Float16(1.);
    return a;
}

// vector operator - : subtract element by element
static inline Vec16h operator - (Vec16h const a, Vec16h const b) {
    return _mm256_sub_ph(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec16h operator - (Vec16h const a, float b) {
    return a - Vec16h(b);
}
static inline Vec16h operator - (float a, Vec16h const b) {
    return Vec16h(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec16h operator - (Vec16h const a) {
    return _mm256_castps_ph(_mm256_xor_ps(_mm256_castph_ps(a), _mm256_castsi256_ps(_mm256_set1_epi32(0x80008000))));
}

// vector operator -= : subtract
static inline Vec16h & operator -= (Vec16h & a, Vec16h const b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec16h operator -- (Vec16h & a, int) {
    Vec16h a0 = a;
    a = a -  _Float16(1.);
    return a0;
}

// prefix operator --
static inline Vec16h & operator -- (Vec16h & a) {
    a = a -  _Float16(1.);
    return a;
}

// vector operator * : multiply element by element
static inline Vec16h operator * (Vec16h const a, Vec16h const b) {
    return _mm256_mul_ph(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec16h operator * (Vec16h const a, _Float16 b) {
    return a * Vec16h(b);
}
static inline Vec16h operator * (_Float16 a, Vec16h const b) {
    return Vec16h(a) * b;
}

// vector operator *= : multiply
static inline Vec16h & operator *= (Vec16h & a, Vec16h const b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec16h operator / (Vec16h const a, Vec16h const b) {
    return _mm256_div_ph(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec16h operator / (Vec16h const a, _Float16 b) {
    return a / Vec16h(b);
}
static inline Vec16h operator / (_Float16 a, Vec16h const b) {
    return Vec16h(a) / b;
}

// vector operator /= : divide
static inline Vec16h & operator /= (Vec16h & a, Vec16h const b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec16hb operator == (Vec16h const a, Vec16h const b) {
    return _mm256_cmp_ph_mask(a, b, 0);
}

// vector operator != : returns true for elements for which a != b
static inline Vec16hb operator != (Vec16h const a, Vec16h const b) {
    return _mm256_cmp_ph_mask(a, b, 4);
}

// vector operator < : returns true for elements for which a < b
static inline Vec16hb operator < (Vec16h const a, Vec16h const b) {
    return _mm256_cmp_ph_mask(a, b, 1);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec16hb operator <= (Vec16h const a, Vec16h const b) {
    return _mm256_cmp_ph_mask(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
static inline Vec16hb operator > (Vec16h const a, Vec16h const b) {
    return _mm256_cmp_ph_mask(a, b, 6+8);
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec16hb operator >= (Vec16h const a, Vec16h const b) {
    return _mm256_cmp_ph_mask(a, b, 5+8);
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec16h operator & (Vec16h const a, Vec16h const b) {
    return _mm256_castps_ph(_mm256_and_ps(_mm256_castph_ps(a), _mm256_castph_ps(b)));
}

// vector operator &= : bitwise and
static inline Vec16h & operator &= (Vec16h & a, Vec16h const b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec16h and Vec16hb
static inline Vec16h operator & (Vec16h const a, Vec16hb const b) {
    return _mm256_castsi256_ph(_mm256_maskz_mov_epi16(b, _mm256_castph_si256(a)));
}
static inline Vec16h operator & (Vec16hb const a, Vec16h const b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec16h operator | (Vec16h const a, Vec16h const b) {
    return _mm256_castps_ph(_mm256_or_ps(_mm256_castph_ps(a), _mm256_castph_ps(b)));
}

// vector operator |= : bitwise or
static inline Vec16h & operator |= (Vec16h & a, Vec16h const b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16h operator ^ (Vec16h const a, Vec16h const b) {
    return _mm256_castps_ph(_mm256_xor_ps(_mm256_castph_ps(a), _mm256_castph_ps(b)));
}

// vector operator ^= : bitwise xor
static inline Vec16h & operator ^= (Vec16h & a, Vec16h const b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec16hb operator ! (Vec16h const a) {
    return a == Vec16h(0.0);
}


/*****************************************************************************
*
*          Functions for Vec16h
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec16h select(Vec16hb const s, Vec16h const a, Vec16h const b) {
    return _mm256_castsi256_ph(_mm256_mask_mov_epi16(_mm256_castph_si256(b), s, _mm256_castph_si256(a)));
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec16h if_add(Vec16hb const f, Vec16h const a, Vec16h const b) {
    return _mm256_mask_add_ph (a, f, a, b);
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
static inline Vec16h if_sub(Vec16hb const f, Vec16h const a, Vec16h const b) {
    return _mm256_mask_sub_ph (a, f, a, b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec16h if_mul(Vec16hb const f, Vec16h const a, Vec16h const b) {
    return _mm256_mask_mul_ph (a, f, a, b);
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] / b[i]) : a[i]
static inline Vec16h if_div(Vec16hb const f, Vec16h const a, Vec16h const b) {
    return _mm256_mask_div_ph (a, f, a, b);
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec16h(-0.0f16)) gives true, while Vec16h(-0.0f16) < Vec16h(0.0f16) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb sign_bit(Vec16h const a) {
    Vec16s t1 = _mm256_castph_si256(a);    // reinterpret as 16-bit integer
    Vec16s t2 = t1 >> 15;                  // extend sign bit
    return t2 != 0;
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec16h sign_combine(Vec16h const a, Vec16h const b) {
    return a ^ (b & Vec16h(_Float16(-0.0)));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb is_finite(Vec16h const a) {
    return __mmask16(_mm256_fpclass_ph_mask(a, 0x99) ^ 0xFFFF);
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb is_inf(Vec16h const a) {
    return __mmask16(_mm256_fpclass_ph_mask(a, 0x18));
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb is_nan(Vec16h const a) {
    // assume that compiler does not optimize this away with -ffinite-math-only:
    return Vec16sb(_mm256_fpclass_ph_mask(a, 0x81));
}

// Function is_subnormal: gives true for elements that are subnormal
// false for finite numbers, zero, NAN and INF
static inline Vec16hb is_subnormal(Vec16h const a) {
    return Vec16hb(_mm256_fpclass_ph_mask(a, 0x20));
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal
// false for finite numbers, NAN and INF
static inline Vec16hb is_zero_or_subnormal(Vec16h const a) {
    return Vec16hb(_mm256_fpclass_ph_mask(a, 0x26));
}

// Function infinite16h: returns a vector where all elements are +INF
static inline Vec16h infinite16h() {
    return _mm256_castsi256_ph(_mm256_set1_epi16(0x7C00));
}

// template for producing quiet NAN
template <>
Vec16h nan_vec<Vec16h>(uint32_t payload) {
    if constexpr (Vec16h::elementtype() == 15) {  // _Float16
        union {
            uint16_t i;
            _Float16 f;
        } uf;
        uf.i = 0x7E00 | (payload & 0x01FF);
        return Vec16h(uf.f);
    }
} 

// Function nan16h: returns a vector where all elements are NAN (quiet)
static inline Vec16h nan16h(int n = 0x10) {
    return nan_vec<Vec16h>(n);
}

// This function returns the code hidden in a NAN. The sign bit is ignored
static inline Vec16us nan_code(Vec16h const x) {
    Vec16us a = Vec16us(_mm256_castph_si256(x));
    Vec16us const n = 0x3FF;
    return select(is_nan(x), a & n, Vec16us(0));
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline _Float16 horizontal_add(Vec16h const a) {
    return horizontal_add(a.get_low()+a.get_high());
}
#if MAX_VECTOR_SIZE >= 512
// same, with high precision
static inline float horizontal_add_x(Vec16h const a) {
    Vec16f b =  _mm512_cvtph_ps(_mm256_castph_si256(a));
    return horizontal_add(b);
}
#endif

// function max: a > b ? a : b
static inline Vec16h max(Vec16h const a, Vec16h const b) {
    return _mm256_max_ph(a, b);
}

// function min: a < b ? a : b
static inline Vec16h min(Vec16h const a, Vec16h const b) {
    return _mm256_min_ph(a, b);
}
// NAN-safe versions of maximum and minimum are in vector_convert.h

// function abs: absolute value
static inline Vec16h abs(Vec16h const a) {
    return _mm256_abs_ph(a);
}

// function sqrt: square root
static inline Vec16h sqrt(Vec16h const a) {
    return _mm256_sqrt_ph(a);
}

// function square: a * a
static inline Vec16h square(Vec16h const a) {
    return a * a;
}

// The purpose of this template is to prevent implicit conversion of a float
// exponent to int when calling pow(vector, float) and vectormath_exp.h is not included
template <typename TT> static Vec16h pow(Vec16h const a, TT const n);  // = delete

// Raise floating point numbers to integer power n
template <>
inline Vec16h pow<int>(Vec16h const x0, int const n) {
    return pow_template_i<Vec16h>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec16h pow<uint32_t>(Vec16h const x0, uint32_t const n) {
    return pow_template_i<Vec16h>(x0, (int)n);
}

// Raise floating point numbers to integer power n, where n is a compile-time constant:
// Template in vectorf28.h is used
//template <typename V, int n>
//static inline V pow_n(V const a);

// implement as function pow(vector, const_int)
template <int n>
static inline Vec16h pow(Vec16h const a, Const_int_t<n>) {
    return pow_n<Vec16h, n>(a);
}


static inline Vec16h round(Vec16h const a) {
    return _mm256_roundscale_ph (a, 8);
}

// function truncate: round towards zero. (result as float vector)
static inline Vec16h truncate(Vec16h const a) {
    return _mm256_roundscale_ph(a, 3 + 8);
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec16h floor(Vec16h const a) {
    return _mm256_roundscale_ph(a, 1 + 8);
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec16h ceil(Vec16h const a) {
    return _mm256_roundscale_ph(a, 2 + 8);
}

// function roundi: round to nearest integer (even). (result as integer vector)
static inline Vec16s roundi(Vec16h const a) {
    // Note: assume MXCSR control register is set to rounding
    return _mm256_cvtph_epi16(a);
}

// function truncatei: round towards zero. (result as integer vector)
static inline Vec16s truncatei(Vec16h const a) {
    return _mm256_cvttph_epi16(a);
}

// function to_float: convert integer vector to float vector
static inline Vec16h to_float16(Vec16s const a) {
    return _mm256_cvtepi16_ph(a);
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec16h to_float16(Vec16us const a) {
    return _mm256_cvtepu16_ph(a);
}

// Approximate math functions

// reciprocal (almost exact)
static inline Vec16h approx_recipr(Vec16h const a) {
    return _mm256_rcp_ph (a);
}

// reciprocal squareroot (almost exact)
static inline Vec16h approx_rsqrt(Vec16h const a) {
    return _mm256_rsqrt_ph(a);
}

// Fused multiply and add functions

// Multiply and add. a*b+c
static inline Vec16h mul_add(Vec16h const a, Vec16h const b, Vec16h const c) {
    return _mm256_fmadd_ph(a, b, c);
}

// Multiply and subtract. a*b-c
static inline Vec16h mul_sub(Vec16h const a, Vec16h const b, Vec16h const c) {
    return _mm256_fmsub_ph(a, b, c);
}

// Multiply and inverse subtract
static inline Vec16h nmul_add(Vec16h const a, Vec16h const b, Vec16h const c) {
    return _mm256_fnmadd_ph(a, b, c);
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec16s exponent(Vec16h const a) {
    Vec16us t1 = _mm256_castph_si256(a);   // reinterpret as 16-bit integer
    Vec16us t2 = t1 << 1;                  // shift out sign bit
    Vec16us t3 = t2 >> 11;                 // shift down logical to position 0
    Vec16s  t4 = Vec16s(t3) - 0x0F;        // subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
static inline Vec16h fraction(Vec16h const a) {
    return _mm256_getmant_ph(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  16 gives +INF
// n <= -15 gives 0.0f
// This function will never produce subnormals, and never raise exceptions
static inline Vec16h exp2(Vec16s const n) {
    Vec16s t1 = max(n, -15);            // limit to allowed range
    Vec16s t2 = min(t1, 16);
    Vec16s t3 = t2 + 15;                // add bias
    Vec16s t4 = t3 << 10;               // put exponent into position 10
    return _mm256_castsi256_ph(t4);     // reinterpret as float
}
//static Vec16h exp2(Vec16h const x);    // defined in vectormath_exp.h ??


// change signs on vectors Vec16h
// Each index i0 - i15 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline Vec16h change_sign(Vec16h const a) {
    if constexpr ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | i11 | i12 | i13 | i14 | i15) == 0) return a;
    __m256i mask = constant8ui<
        (i0  ? 0x8000 : 0) | (i1  ? 0x80000000 : 0), 
        (i2  ? 0x8000 : 0) | (i3  ? 0x80000000 : 0), 
        (i4  ? 0x8000 : 0) | (i5  ? 0x80000000 : 0), 
        (i6  ? 0x8000 : 0) | (i7  ? 0x80000000 : 0), 
        (i8  ? 0x8000 : 0) | (i9  ? 0x80000000 : 0), 
        (i10 ? 0x8000 : 0) | (i11 ? 0x80000000 : 0), 
        (i12 ? 0x8000 : 0) | (i13 ? 0x80000000 : 0), 
        (i14 ? 0x8000 : 0) | (i15 ? 0x80000000 : 0) >();
    return  _mm256_castps_ph(_mm256_xor_ps(_mm256_castph_ps(a), _mm256_castsi256_ps(mask)));     // flip sign bits
}

/*****************************************************************************
*
*          conversions Vec16h <-> Vec16f
*
*****************************************************************************/
#if MAX_VECTOR_SIZE >= 512
// extend precision: Vec8h -> Vec8f
Vec16f to_float (Vec16h h) {
    return _mm512_cvtph_ps(_mm256_castph_si256(h));
}

// reduce precision: Vec8f -> Vec8h
Vec16h to_float16 (Vec16f f) {
    return _mm256_castsi256_ph(_mm512_cvtps_ph(f, 0));
}
#endif

/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/

static inline __m256i reinterpret_i(__m256h const x) {
    return _mm256_castph_si256(x);
}

static inline __m256h reinterpret_h(__m256i const x) {
    return _mm256_castsi256_ph(x);
}

static inline __m256  reinterpret_f(__m256h const x) {
    return _mm256_castph_ps(x);
}

static inline __m256d reinterpret_d(__m256h const x) {
    return _mm256_castph_pd(x);
}

static inline Vec16h extend_z(Vec8h a) {
    //return _mm256_zextsi128_si256(a);
    return _mm256_zextph128_ph256(a);
}


/*****************************************************************************
*
*          Vector permute and blend functions
*
******************************************************************************
*
* The permute function can reorder the elements of a vector and optionally
* set some elements to zero.
*
* See vectori128.h for details
*
*****************************************************************************/
// permute vector Vec16h
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline Vec16h permute16(Vec16h const a) {
    return _mm256_castsi256_ph (
    permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15> (
    Vec16s(_mm256_castph_si256(a))));
}


/*****************************************************************************
*
*          Vector blend functions
*
*****************************************************************************/

// permute and blend Vec16h
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline Vec16h blend16(Vec16h const a, Vec16h const b) {
    return _mm256_castsi256_ph (
    blend16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15> (
    Vec16s(_mm256_castph_si256(a)), Vec16s(_mm256_castph_si256(b))));
}


/*****************************************************************************
*
*          Vector lookup functions
*
******************************************************************************
*
* These functions use vector elements as indexes into a table.
* The table is given as one or more vectors or as an array.
*
*****************************************************************************/

static inline Vec16h lookup16 (Vec16s const index, Vec16h const table) {
    return _mm256_castsi256_ph(lookup16(index, Vec16s(_mm256_castph_si256(table))));
}

template <int n>
static inline Vec16h lookup(Vec16s const index, void const * table) {
    return _mm256_castsi256_ph(lookup<n>(index, (void const *)(table)));
}


#endif // MAX_VECTOR_SIZE >= 256



/*****************************************************************************
*
*          512 bit vectors
*
*****************************************************************************/

#if MAX_VECTOR_SIZE >= 512


/*****************************************************************************
*
*          Vec32hb: Vector of 32 Booleans for use with Vec32h
*
*****************************************************************************/

typedef Vec32b Vec32hb;  // compact boolean vector


/*****************************************************************************
*
*          Vec32h: Vector of 32 half precision floating point values
*
*****************************************************************************/

class Vec32h {
protected:
    __m512h zmm; // Float vector
public:
    // Default constructor:
    Vec32h() = default;
    // Constructor to broadcast the same value into all elements:
    Vec32h(_Float16 f) {
        zmm = _mm512_set1_ph (f);
    }
    // Constructor to build from all elements:
    Vec32h(_Float16 f0, _Float16 f1, _Float16 f2, _Float16 f3, _Float16 f4, _Float16 f5, _Float16 f6, _Float16 f7,
    _Float16 f8, _Float16 f9, _Float16 f10, _Float16 f11, _Float16 f12, _Float16 f13, _Float16 f14, _Float16 f15,
    _Float16 f16, _Float16 f17, _Float16 f18, _Float16 f19, _Float16 f20, _Float16 f21, _Float16 f22, _Float16 f23,
    _Float16 f24, _Float16 f25, _Float16 f26, _Float16 f27, _Float16 f28, _Float16 f29, _Float16 f30, _Float16 f31) {
        zmm = _mm512_setr_ph (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
        f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31);
    }
    // Constructor to build from two Vec16h:
    Vec32h(Vec16h const a0, Vec16h const a1) {     
        zmm = _mm512_castps_ph(_mm512_insertf32x8(_mm512_castps256_ps512(_mm256_castph_ps(a0)),_mm256_castph_ps(a1),1));
    }
    // Constructor to convert from type __m512h used in intrinsics:
    Vec32h(__m512h const x) {
        zmm = x;
    }
    // Assignment operator to convert from type __m512h used in intrinsics:
    Vec32h & operator = (__m512h const x) {
        zmm = x;
        return *this;
    }
    // Type cast operator to convert to __m512h used in intrinsics
    operator __m512h() const {
        return zmm;
    }
    // Member function to load from array (unaligned)
    Vec32h & load(void const * p) {
        zmm = _mm512_loadu_ph (p);
        return *this;
    }
    // Member function to load from array, aligned by 64
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 64. In most cases there is no difference in speed between load and load_a
    Vec32h & load_a(void const * p) {
        zmm = _mm512_load_ph (p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        _mm512_storeu_ph (p, zmm);
    }
    // Member function storing into array, aligned by 64
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 64.
    void store_a(void * p) const {
        _mm512_store_ph (p, zmm);
    }
    // Member function storing to aligned uncached memory (non-temporal store).
    // This may be more efficient than store_a when storing large blocks of memory if it 
    // is unlikely that the data will stay in the cache until it is read again.
    // Note: Will generate runtime error if p is not aligned by 64
    void store_nt(void * p) const {
        _mm512_stream_ps((float*)p, _mm512_castph_ps(zmm));
    }
    // Partial load. Load n elements and set the rest to 0
    Vec32h & load_partial(int n, void const * p) {
        zmm = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(__mmask32((1u << n) - 1), p));
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        _mm512_mask_storeu_epi16(p, __mmask32((1u << n) - 1), _mm512_castph_si512(zmm));
    }
    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec32h & cutoff(int n) {
        zmm = _mm512_castsi512_ph(_mm512_maskz_mov_epi16(__mmask32((1u << n) - 1), _mm512_castph_si512(zmm)));
        return *this;
    }
    // Member function to change a single element in vector
    Vec32h const insert(int index, _Float16 a) {
        __m512h aa = _mm512_set1_ph (a);
        zmm = _mm512_castsi512_ph(_mm512_mask_mov_epi16(_mm512_castph_si512(zmm), __mmask32(1u << index), _mm512_castph_si512(aa)));
        return *this;
    }
    // Member function extract a single element from vector
    _Float16 extract(int index) const {
#if INSTRSET >= 10 && defined (__AVX512VBMI2__)
        __m512i x = _mm512_maskz_compress_epi16(__mmask32(1u << index), _mm512_castph_si512(zmm));
        return _mm512_cvtsh_h(_mm512_castsi512_ph(x));
#elif 0
        union {
            __m512h v;
            _Float16 f[32];
        } y;
        y.v = zmm;
        return y.f[index & 31];
#else
        Vec16ui x = _mm512_maskz_compress_epi32(__mmask16(1u << (index >> 1)), _mm512_castph_si512(zmm));  // extract int32_t
        x >>= uint32_t((index & 1) << 4);  // get upper 16 bits if index odd
        return _mm512_cvtsh_h(_mm512_castsi512_ph(x));
#endif
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    _Float16 operator [] (int index) const {
        return extract(index);
    }
    Vec16h get_low() const {
        return _mm512_castph512_ph256(zmm);
    }
    Vec16h get_high() const {
        return _mm256_castps_ph(_mm512_extractf32x8_ps(_mm512_castph_ps(zmm),1));
    }
    static constexpr int size() {
        return 32;
    }
    static constexpr int elementtype() {
        return 15;
    }
    typedef __m512h registertype;
};


/*****************************************************************************
*
*          Operators for Vec32h
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec32h operator + (Vec32h const a, Vec32h const b) {
    return _mm512_add_ph(a, b);
}

// vector operator + : add vector and scalar
static inline Vec32h operator + (Vec32h const a, _Float16 b) {
    return a + Vec32h(b);
}
static inline Vec32h operator + (_Float16 a, Vec32h const b) {
    return Vec32h(a) + b;
}

// vector operator += : add
static inline Vec32h & operator += (Vec32h & a, Vec32h const b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec32h operator ++ (Vec32h & a, int) {
    Vec32h a0 = a;
    a = a +  _Float16(1.);
    return a0;
}

// prefix operator ++
static inline Vec32h & operator ++ (Vec32h & a) {
    a = a +  _Float16(1.);
    return a;
}

// vector operator - : subtract element by element
static inline Vec32h operator - (Vec32h const a, Vec32h const b) {
    return _mm512_sub_ph(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec32h operator - (Vec32h const a, float b) {
    return a - Vec32h(b);
}
static inline Vec32h operator - (float a, Vec32h const b) {
    return Vec32h(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec32h operator - (Vec32h const a) {
    return _mm512_castps_ph(_mm512_xor_ps(_mm512_castph_ps(a), _mm512_castsi512_ps(_mm512_set1_epi32(0x80008000))));
}

// vector operator -= : subtract
static inline Vec32h & operator -= (Vec32h & a, Vec32h const b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec32h operator -- (Vec32h & a, int) {
    Vec32h a0 = a;
    a = a -  _Float16(1.);
    return a0;
}

// prefix operator --
static inline Vec32h & operator -- (Vec32h & a) {
    a = a -  _Float16(1.);
    return a;
}

// vector operator * : multiply element by element
static inline Vec32h operator * (Vec32h const a, Vec32h const b) {
    return _mm512_mul_ph(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec32h operator * (Vec32h const a, _Float16 b) {
    return a * Vec32h(b);
}
static inline Vec32h operator * (_Float16 a, Vec32h const b) {
    return Vec32h(a) * b;
}

// vector operator *= : multiply
static inline Vec32h & operator *= (Vec32h & a, Vec32h const b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec32h operator / (Vec32h const a, Vec32h const b) {
    return _mm512_div_ph(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec32h operator / (Vec32h const a, _Float16 b) {
    return a / Vec32h(b);
}
static inline Vec32h operator / (_Float16 a, Vec32h const b) {
    return Vec32h(a) / b;
}

// vector operator /= : divide
static inline Vec32h & operator /= (Vec32h & a, Vec32h const b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec32hb operator == (Vec32h const a, Vec32h const b) {
    return _mm512_cmp_ph_mask(a, b, 0);
}

// vector operator != : returns true for elements for which a != b
static inline Vec32hb operator != (Vec32h const a, Vec32h const b) {
    return _mm512_cmp_ph_mask(a, b, 4);
}

// vector operator < : returns true for elements for which a < b
static inline Vec32hb operator < (Vec32h const a, Vec32h const b) {
    return _mm512_cmp_ph_mask(a, b, 1);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec32hb operator <= (Vec32h const a, Vec32h const b) {
    return _mm512_cmp_ph_mask(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
static inline Vec32hb operator > (Vec32h const a, Vec32h const b) {
    return _mm512_cmp_ph_mask(a, b, 6+8);
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec32hb operator >= (Vec32h const a, Vec32h const b) {
    return _mm512_cmp_ph_mask(a, b, 5+8);
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec32h operator & (Vec32h const a, Vec32h const b) {
    return _mm512_castps_ph(_mm512_and_ps(_mm512_castph_ps(a), _mm512_castph_ps(b)));
}

// vector operator &= : bitwise and
static inline Vec32h & operator &= (Vec32h & a, Vec32h const b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec32h and Vec32hb
static inline Vec32h operator & (Vec32h const a, Vec32hb const b) {
    return _mm512_castsi512_ph(_mm512_maskz_mov_epi16(b, _mm512_castph_si512(a)));
}
static inline Vec32h operator & (Vec32hb const a, Vec32h const b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec32h operator | (Vec32h const a, Vec32h const b) {
    return _mm512_castps_ph(_mm512_or_ps(_mm512_castph_ps(a), _mm512_castph_ps(b)));
}

// vector operator |= : bitwise or
static inline Vec32h & operator |= (Vec32h & a, Vec32h const b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec32h operator ^ (Vec32h const a, Vec32h const b) {
    return _mm512_castps_ph(_mm512_xor_ps(_mm512_castph_ps(a), _mm512_castph_ps(b)));
}

// vector operator ^= : bitwise xor
static inline Vec32h & operator ^= (Vec32h & a, Vec32h const b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec32hb operator ! (Vec32h const a) {
    return a == Vec32h(0.0);
}


/*****************************************************************************
*
*          Functions for Vec32h
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec32h select(Vec32hb const s, Vec32h const a, Vec32h const b) {
    return _mm512_castsi512_ph(_mm512_mask_mov_epi16(_mm512_castph_si512(b), s, _mm512_castph_si512(a)));
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec32h if_add(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return _mm512_mask_add_ph (a, f, a, b);
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
static inline Vec32h if_sub(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return _mm512_mask_sub_ph (a, f, a, b);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec32h if_mul(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return _mm512_mask_mul_ph (a, f, a, b);
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] / b[i]) : a[i]
static inline Vec32h if_div(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return _mm512_mask_div_ph (a, f, a, b);
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec32h(-0.0f16)) gives true, while Vec32h(-0.0f16) < Vec32h(0.0f16) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb sign_bit(Vec32h const a) {
    Vec32s t1 = _mm512_castph_si512(a);    // reinterpret as 16-bit integer
    Vec32s t2 = t1 >> 15;                  // extend sign bit
    return t2 != 0;
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec32h sign_combine(Vec32h const a, Vec32h const b) {
    return a ^ (b & Vec32h(_Float16(-0.0)));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb is_finite(Vec32h const a) {
    return __mmask32(~ _mm512_fpclass_ph_mask(a, 0x99));
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb is_inf(Vec32h const a) {
    return __mmask32(_mm512_fpclass_ph_mask(a, 0x18));
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb is_nan(Vec32h const a) {
    // assume that compiler does not optimize this away with -ffinite-math-only:
    return Vec32sb(_mm512_fpclass_ph_mask(a, 0x81));
}

// Function is_subnormal: gives true for elements that are subnormal
// false for finite numbers, zero, NAN and INF
static inline Vec32hb is_subnormal(Vec32h const a) {
    return Vec32hb(_mm512_fpclass_ph_mask(a, 0x20));
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal
// false for finite numbers, NAN and INF
static inline Vec32hb is_zero_or_subnormal(Vec32h const a) {
    return Vec32hb(_mm512_fpclass_ph_mask(a, 0x26));
}

// Function infinite32h: returns a vector where all elements are +INF
static inline Vec32h infinite32h() {
    return _mm512_castsi512_ph(_mm512_set1_epi16(0x7C00));
}

// template for producing quiet NAN
template <>
Vec32h nan_vec<Vec32h>(uint32_t payload) {
    if constexpr (Vec32h::elementtype() == 15) {  // _Float16
        union {
            uint16_t i;
            _Float16 f;
        } uf;
        uf.i = 0x7E00 | (payload & 0x01FF);
        return Vec32h(uf.f);
    }
} 

// Function nan32h: returns a vector where all elements are NAN (quiet)
static inline Vec32h nan32h(int n = 0x10) {
    return nan_vec<Vec32h>(n);
}

// This function returns the code hidden in a NAN. The sign bit is ignored
static inline Vec32us nan_code(Vec32h const x) {
    Vec32us a = Vec32us(_mm512_castph_si512(x));
    Vec32us const n = 0x3FF;
    return select(is_nan(x), a & n, Vec32us(0));
} 


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline _Float16 horizontal_add(Vec32h const a) {
    return horizontal_add(a.get_low()+a.get_high());
}
// same, with high precision
static inline float horizontal_add_x(Vec32h const a) {
    Vec16f b1 = _mm512_cvtph_ps(_mm256_castph_si256(a.get_low()));   //_mm512_cvtph_ps(a.get_low());
    Vec16f b2 = _mm512_cvtph_ps(_mm256_castph_si256(a.get_high()));
    return horizontal_add(b1 + b2);
}

// function max: a > b ? a : b
static inline Vec32h max(Vec32h const a, Vec32h const b) {
    return _mm512_max_ph(a, b);
}

// function min: a < b ? a : b
static inline Vec32h min(Vec32h const a, Vec32h const b) {
    return _mm512_min_ph(a, b);
}
// NAN-safe versions of maximum and minimum are in vector_convert.h

// function abs: absolute value
static inline Vec32h abs(Vec32h const a) {
    return _mm512_abs_ph(a);
}

// function sqrt: square root
static inline Vec32h sqrt(Vec32h const a) {
    return _mm512_sqrt_ph(a);
}

// function square: a * a
static inline Vec32h square(Vec32h const a) {
    return a * a;
}

// The purpose of this template is to prevent implicit conversion of a float
// exponent to int when calling pow(vector, float) and vectormath_exp.h is not included
template <typename TT> static Vec32h pow(Vec32h const a, TT const n);  // = delete

// Raise floating point numbers to integer power n
template <>
inline Vec32h pow<int>(Vec32h const x0, int const n) {
    return pow_template_i<Vec32h>(x0, n);
}

// allow conversion from unsigned int
template <>
inline Vec32h pow<uint32_t>(Vec32h const x0, uint32_t const n) {
    return pow_template_i<Vec32h>(x0, (int)n);
}

// Raise floating point numbers to integer power n, where n is a compile-time constant:
// Template in vectorf28.h is used
//template <typename V, int n>
//static inline V pow_n(V const a);

// implement as function pow(vector, const_int)
template <int n>
static inline Vec32h pow(Vec32h const a, Const_int_t<n>) {
    return pow_n<Vec32h, n>(a);
}

static inline Vec32h round(Vec32h const a) {
    return _mm512_roundscale_ph (a, 8);
}

// function truncate: round towards zero. (result as float vector)
static inline Vec32h truncate(Vec32h const a) {
    return _mm512_roundscale_ph(a, 3 + 8);
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec32h floor(Vec32h const a) {
    return _mm512_roundscale_ph(a, 1 + 8);
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec32h ceil(Vec32h const a) {
    return _mm512_roundscale_ph(a, 2 + 8);
}

// function roundi: round to nearest integer (even). (result as integer vector)
static inline Vec32s roundi(Vec32h const a) {
    // Note: assume MXCSR control register is set to rounding
    return _mm512_cvtph_epi16(a);
}

// function truncatei: round towards zero. (result as integer vector)
static inline Vec32s truncatei(Vec32h const a) {
    return _mm512_cvttph_epi16(a);
}

// function to_float: convert integer vector to float vector
static inline Vec32h to_float16(Vec32s const a) {
    return _mm512_cvtepi16_ph(a);
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec32h to_float16(Vec32us const a) {
    return _mm512_cvtepu16_ph(a);
}

// Approximate math functions

// reciprocal (almost exact)
static inline Vec32h approx_recipr(Vec32h const a) {
    return _mm512_rcp_ph(a);
}

// reciprocal squareroot (almost exact)
static inline Vec32h approx_rsqrt(Vec32h const a) {
    return _mm512_rsqrt_ph(a);
}

// Fused multiply and add functions

// Multiply and add. a*b+c
static inline Vec32h mul_add(Vec32h const a, Vec32h const b, Vec32h const c) {
    return _mm512_fmadd_ph(a, b, c);
}

// Multiply and subtract. a*b-c
static inline Vec32h mul_sub(Vec32h const a, Vec32h const b, Vec32h const c) {
    return _mm512_fmsub_ph(a, b, c);
}

// Multiply and inverse subtract
static inline Vec32h nmul_add(Vec32h const a, Vec32h const b, Vec32h const c) {
    return _mm512_fnmadd_ph(a, b, c);
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec32s exponent(Vec32h const a) {
    Vec32us t1 = _mm512_castph_si512(a);   // reinterpret as 16-bit integer
    Vec32us t2 = t1 << 1;                  // shift out sign bit
    Vec32us t3 = t2 >> 11;                 // shift down logical to position 0
    Vec32s  t4 = Vec32s(t3) - 0x0F;        // subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
static inline Vec32h fraction(Vec32h const a) {
    return _mm512_getmant_ph(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  16 gives +INF
// n <= -15 gives 0.0f
// This function will never produce subnormals, and never raise exceptions
static inline Vec32h exp2(Vec32s const n) {
    Vec32s t1 = max(n, -15);            // limit to allowed range
    Vec32s t2 = min(t1, 16);
    Vec32s t3 = t2 + 15;                // add bias
    Vec32s t4 = t3 << 10;               // put exponent into position 10
    return _mm512_castsi512_ph(t4);     // reinterpret as float
}
//static Vec32h exp2(Vec32h const x);    // defined in vectormath_exp.h ??


// change signs on vectors Vec32h
// Each index i0 - i31 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15,
int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 >
static inline Vec32h change_sign(Vec32h const a) {
    if constexpr ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | i11 | i12 | i13 | i14 | i15 |
    i16 | i17 | i18 | i19 | i20 | i21 | i22 | i23 | i24 | i25 | i26 | i27 | i28 | i29 | i30 | i31)
    == 0) return a;
    __m512i mask = constant16ui<
        (i0  ? 0x8000 : 0) | (i1  ? 0x80000000 : 0), 
        (i2  ? 0x8000 : 0) | (i3  ? 0x80000000 : 0), 
        (i4  ? 0x8000 : 0) | (i5  ? 0x80000000 : 0), 
        (i6  ? 0x8000 : 0) | (i7  ? 0x80000000 : 0), 
        (i8  ? 0x8000 : 0) | (i9  ? 0x80000000 : 0), 
        (i10 ? 0x8000 : 0) | (i11 ? 0x80000000 : 0), 
        (i12 ? 0x8000 : 0) | (i13 ? 0x80000000 : 0), 
        (i14 ? 0x8000 : 0) | (i15 ? 0x80000000 : 0),        
        (i16 ? 0x8000 : 0) | (i17 ? 0x80000000 : 0), 
        (i18 ? 0x8000 : 0) | (i19 ? 0x80000000 : 0), 
        (i20 ? 0x8000 : 0) | (i21 ? 0x80000000 : 0), 
        (i22 ? 0x8000 : 0) | (i23 ? 0x80000000 : 0), 
        (i24 ? 0x8000 : 0) | (i25 ? 0x80000000 : 0), 
        (i26 ? 0x8000 : 0) | (i27 ? 0x80000000 : 0), 
        (i28 ? 0x8000 : 0) | (i29 ? 0x80000000 : 0), 
        (i30 ? 0x8000 : 0) | (i31 ? 0x80000000 : 0) >();
    return  _mm512_castps_ph(_mm512_xor_ps(_mm512_castph_ps(a), _mm512_castsi512_ps(mask)));     // flip sign bits
}


/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/

static inline __m512i reinterpret_i(__m512h const x) {
    return _mm512_castph_si512(x);
}

static inline __m512h reinterpret_h(__m512i const x) {
    return _mm512_castsi512_ph(x);
}

static inline __m512  reinterpret_f(__m512h const x) {
    return _mm512_castph_ps(x);
}

static inline __m512d reinterpret_d(__m512h const x) {
    return _mm512_castph_pd(x);
}

static inline Vec32h extend_z(Vec16h a) {
    //return _mm512_zextsi256_si512(a);
    return _mm512_zextph256_ph512(a);
}

/*****************************************************************************
*
*          Vector permute and blend functions
*
******************************************************************************
*
* The permute function can reorder the elements of a vector and optionally
* set some elements to zero.
*
* See vectori128.h for details
*
*****************************************************************************/
// permute vector Vec32h
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15,
int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 >
static inline Vec32h permute32(Vec32h const a) {
    return _mm512_castsi512_ph (
    permute32<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
    i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31 > (
    Vec32s(_mm512_castph_si512(a))));
}


/*****************************************************************************
*
*          Vector blend functions
*
*****************************************************************************/

// permute and blend Vec32h
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15,
int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 >
static inline Vec32h blend32(Vec32h const a, Vec32h const b) {
    return _mm512_castsi512_ph (
    blend32<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
    i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31 > (
    Vec32s(_mm512_castph_si512(a)), Vec32s(_mm512_castph_si512(b))));
}


/*****************************************************************************
*
*          Vector lookup functions
*
******************************************************************************
*
* These functions use vector elements as indexes into a table.
* The table is given as one or more vectors or as an array.
*
*****************************************************************************/

static inline Vec32h lookup32 (Vec32s const index, Vec32h const table) {
    return _mm512_castsi512_ph(lookup32(index, Vec32s(_mm512_castph_si512(table))));
}

template <int n>
static inline Vec32h lookup(Vec32s const index, void const * table) {
    return _mm512_castsi512_ph(lookup<n>(index, (void const *)(table)));
}

#endif // MAX_VECTOR_SIZE >= 512


/***************************************************************************************
*
*                       Mathematical functions
*
* This code is designed to be independent of whether the vectormath files are included
*
***************************************************************************************/

// pow(2,n)
template <typename V>
static inline V vh_pow2n (V const n) {           
    typedef decltype(roundi(n)) VI;              // corresponding integer vector type
    const _Float16 pow2_10 =  1024.;             // 2^10
    const _Float16 bias = 15.;                   // bias in exponent
    V  a = n + (bias + pow2_10);                 // put n + bias in least significant bits
    VI b = reinterpret_i(a);                     // bit-cast to integer
    VI c = b << 10;                              // shift left 10 places to get into exponent field
    V  d = reinterpret_h(c);                     // bit-cast back to float16
    return d;
}

// generate INF vector
template <typename VTYPE>
static inline VTYPE infinite_vech();

template <>
inline Vec8h infinite_vech<Vec8h>() {
    return infinite8h();
}
#if MAX_VECTOR_SIZE >= 256
template <>
inline Vec16h infinite_vech<Vec16h>() {
    return infinite16h();
}
#endif
#if MAX_VECTOR_SIZE >= 512
template <>
inline Vec32h infinite_vech<Vec32h>() {
    return infinite32h();
}
#endif


// Template for exp function, half precision
// The limit of abs(x) is defined by max_x below
// Note on accuracy:
// This function does not produce subnormal results
// Max error is 7 ULP
// The input range is slightly reduced. Inputs > 10.75 give INF. INputs < -10.75 give 0.
// The emulated version without __AVX512FP16__ can produce subnormals, has full input range,
// and a precision of 1 ULP

// Template parameters:
// VTYPE:  float vector type
// M1: 0 for exp, 1 for expm1
// BA: 0 for exp, 1 for 0.5*exp, 2 for pow(2,x), 10 for pow(10,x)

template<typename VTYPE, int M1, int BA>
static inline VTYPE exp_h(VTYPE const initial_x) {

    // Taylor coefficients
    const _Float16 P0expf   =  1.f/2.f;
    const _Float16 P1expf   =  1.f/6.f;
    const _Float16 P2expf   =  1.f/24.f;

    VTYPE  x, r, x2, z, n2;                      // data vectors

    // maximum abs(x), value depends on BA, defined below
    // The lower limit of x is slightly more restrictive than the upper limit.
    // We are specifying the lower limit, except for BA = 1 because it is not used for negative x
    _Float16 max_x;

    if constexpr (BA <= 1) {                     // exp(x)
        const _Float16 ln2f  =  0.69314718f;     // ln(2)
        const _Float16 log2e  =  1.44269504089f; // log2(e)
        x = initial_x;
        r = round(initial_x*log2e);
        x = nmul_add(r, VTYPE(ln2f), x);         //  x -= r * ln2f;
        max_x = 10.75f;                          // overflow limit
    }
    else if constexpr (BA == 2) {                // pow(2,x)
        const _Float16 ln2  =  0.69314718f;      // ln(2)
        max_x = 15.5f;
        r = round(initial_x);
        x = initial_x - r;
        x = x * ln2;
    }
    else if constexpr (BA == 10) {               // pow(10,x)
        max_x = 4.667f;
        const _Float16 log10_2 = 0.30102999566f; // log10(2)
        x = initial_x;
        r = round(initial_x*_Float16(3.32192809489f)); // VM_LOG2E*VM_LN10
        x = nmul_add(r, VTYPE(log10_2), x);      //  x -= r * log10_2
        x = x * _Float16(2.30258509299f);        // (float)VM_LN10;
    }
    else  {  // undefined value of BA
        return 0.;
    }
    x2 = x * x;
    // z = polynomial_2(x,P0expf,P1expf,P2expf);
    z = mul_add(x2, P2expf, mul_add(x, P1expf, P0expf));
    z = mul_add(z, x2, x);                       // z *= x2;  z += x;
    if constexpr (BA == 1) r--;                  // 0.5 * exp(x)
    n2 = vh_pow2n(r);                            // multiply by power of 2
    if constexpr (M1 == 0) {                     // exp
        z = (z + _Float16(1.0f)) * n2;
    }
    else {                                       // expm1
        z = mul_add(z, n2, n2 - _Float16(1.0));  //  z = z * n2 + (n2 - 1.0f);
#ifdef SIGNED_ZERO                               // pedantic preservation of signed zero
        z = select(initial_x == _Float16(0.), initial_x, z);
#endif
    }
    // check for overflow
    auto inrange  = abs(initial_x) < VTYPE(max_x);// boolean vector
    // check for INF and NAN
    inrange &= is_finite(initial_x);
    if (horizontal_and(inrange)) {               // fast normal path
        return z;
    }
    else {                                       // overflow, underflow and NAN
        VTYPE const inf = infinite_vech<VTYPE>();          // infinite
        r = select(sign_bit(initial_x), _Float16(0.f-(M1&1)), inf);  // value in case of +/- overflow or INF
        z = select(inrange, z, r);                         // +/- underflow
        z = select(is_nan(initial_x), initial_x, z);       // NAN goes through
        return z;
    }
}

// dummy functions used for type definition in template sincos_h:
static inline Vec8us  unsigned_int_type(Vec8h)  { return 0; }
#if MAX_VECTOR_SIZE >= 256
static inline Vec16us unsigned_int_type(Vec16h) { return 0; }
#endif
#if MAX_VECTOR_SIZE >= 512
static inline Vec32us unsigned_int_type(Vec32h) { return 0; }
#endif


// Template for trigonometric functions.
// Template parameters:
// VTYPE:  vector type
// SC:     1 = sin, 2 = cos, 3 = sincos, 4 = tan, 8 = multiply by pi
// Parameters:
// xx = input x (radians)
// cosret = return pointer (only if SC = 3)
template<typename VTYPE, int SC>
static inline VTYPE sincos_h(VTYPE * cosret, VTYPE const xx) {

    // define constants
    const _Float16 dp1h = 1.57031250f;           // pi/2 with lower bits of mantissa removed
    const _Float16 dp2h = 1.57079632679489661923 - dp1h; // remaining bits

    const _Float16 P0sinf = -1.6666654611E-1f;   // Taylor coefficients
    const _Float16 P1sinf = 8.3321608736E-3f;

    const _Float16 P0cosf = 4.166664568298827E-2f;
    const _Float16 P1cosf = -1.388731625493765E-3f;

    const float pi     = 3.14159265358979323846f;// pi
    const _Float16 c2_pi  = float(2./3.14159265358979323846);  // 2/pi

    typedef decltype(roundi(xx)) ITYPE;          // integer vector type
    typedef decltype(unsigned_int_type(xx)) UITYPE;// unsigned integer vector type
    typedef decltype(xx < xx) BVTYPE;            // boolean vector type

    VTYPE  xa, x, y, x2, s, c, sin1, cos1;       // data vectors
    ITYPE  signsin, signcos;                     // integer vectors
    UITYPE q;                                    // unsigned integer vector for quadrant
    BVTYPE swap, overflow;                       // boolean vectors

    xa = abs(xx);

    // Find quadrant
    if constexpr ((SC & 8) != 0) {               // sinpi
        xa = select(xa > VTYPE(32000.f), VTYPE(0.f), xa); // avoid overflow when multiplying by 2
        y = round(xa * VTYPE(2.0f)); 
    }
    else {                                       // sin
        xa = select(xa > VTYPE(314.25f), VTYPE(0.f), xa); // avoid meaningless results for high x
        y = round(xa * c2_pi);                   // quadrant, as float
    }

    q = UITYPE(roundi(y));                       // quadrant, as unsigned integer
    //      0 -   pi/4 => 0
    //   pi/4 - 3*pi/4 => 1
    // 3*pi/4 - 5*pi/4 => 2
    // 5*pi/4 - 7*pi/4 => 3
    // 7*pi/4 - 8*pi/4 => 4

    if constexpr ((SC & 8) != 0) {               // sinpi
        // modulo 2: subtract 0.5*y
        x = nmul_add(y, VTYPE(0.5f), xa) * VTYPE(pi);
    }
    else {                                       // sin
        // Reduce by extended precision modular arithmetic    
        x = nmul_add(y, dp2h, nmul_add(y, dp1h, xa)); 
    }

    // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    x2 = x * x;

    //x2 = select(is_inf(xx), reinterpret_h(UITYPE(0x7F00)), x2);  // return NAN rather than INF if INF input

    s = mul_add(x2, P1sinf, P0sinf) * (x*x2) + x;
    c = mul_add(x2, P1cosf, P0cosf) * (x2*x2) + nmul_add(_Float16(0.5f), x2, _Float16(1.0f));
    // s = P0sinf * (x*x2) + x;  // 2 ULP error
    // c = P0cosf * (x2*x2) + nmul_add(0.5f, x2, 1.0f);  // 2 ULP error

    // swap sin and cos if odd quadrant
    swap = BVTYPE((q & 1) != 0);

    if constexpr ((SC & 5) != 0) {               // get sin
        sin1 = select(swap, c, s);
        signsin = ((q << 14) ^ ITYPE(reinterpret_i(xx))); // sign
        sin1 = sign_combine(sin1, VTYPE(reinterpret_h(signsin)));
    }

    if constexpr ((SC & 6) != 0) {               // get cos
        cos1 = select(swap, s, c);
        signcos = ((q + 1) & 2) << 14;           // sign
        cos1 ^= reinterpret_h(signcos);
    }
    if      constexpr ((SC & 7) == 1) return sin1;
    else if constexpr ((SC & 7) == 2) return cos1;
    else if constexpr ((SC & 7) == 3) {          // both sin and cos. cos returned through pointer
        *cosret = cos1;
        return sin1;
    }
    else {                                       // (SC & 7) == 4. tan
        if constexpr (SC == 12) {
            // tanpi can give INF result, tan cannot. Get the right sign of INF result according to IEEE 754-2019
            cos1 = select(cos1 == VTYPE(0.f), VTYPE(0.f), cos1); // remove sign of 0
            // the sign of zero output is arbitrary. fixing it would be a waste of code
        }
        return sin1 / cos1;
    }
}

// instantiations of math function templates

static inline Vec8h exp(Vec8h const x) {
    return exp_h<Vec8h, 0, 0>(x);
} 
static inline Vec8h exp2(Vec8h const x) {
    return exp_h<Vec8h, 0, 2>(x);
}
static inline Vec8h exp10(Vec8h const x) {
    return exp_h<Vec8h, 0, 10>(x);
}
static inline Vec8h expm1(Vec8h const x) {
    return exp_h<Vec8h, 1, 0>(x);
}
static inline Vec8h sin(Vec8h const x) {
    return sincos_h<Vec8h, 1>(0, x);
}
static inline Vec8h cos(Vec8h const x) {
    return sincos_h<Vec8h, 2>(0, x);
}
static inline Vec8h sincos(Vec8h * cosret, Vec8h const x) {
    return sincos_h<Vec8h, 3>(cosret, x);
} 
static inline Vec8h tan(Vec8h const x) {
    return sincos_h<Vec8h, 4>(0, x);
}

static inline Vec8h sinpi(Vec8h const x) {
    return sincos_h<Vec8h, 9>(0, x);
}
static inline Vec8h cospi(Vec8h const x) {
    return sincos_h<Vec8h, 10>(0, x);
}
static inline Vec8h sincospi(Vec8h * cosret, Vec8h const x) {
    return sincos_h<Vec8h, 11>(cosret, x);
}
static inline Vec8h tanpi(Vec8h const x) {
    return sincos_h<Vec8h, 12>(0, x);
}

#if MAX_VECTOR_SIZE >= 256

static inline Vec16h exp(Vec16h const x) {
    return exp_h<Vec16h, 0, 0>(x);
}
static inline Vec16h exp2(Vec16h const x) {
    return exp_h<Vec16h, 0, 2>(x);
}
static inline Vec16h exp10(Vec16h const x) {
    return exp_h<Vec16h, 0, 10>(x);
}
static inline Vec16h expm1(Vec16h const x) {
    return exp_h<Vec16h, 1, 0>(x);
} 
static inline Vec16h sin(Vec16h const x) {
    return sincos_h<Vec16h, 1>(0, x);
}
static inline Vec16h cos(Vec16h const x) {
    return sincos_h<Vec16h, 2>(0, x);
}
static inline Vec16h sincos(Vec16h * cosret, Vec16h const x) {
    return sincos_h<Vec16h, 3>(cosret, x);
} 
static inline Vec16h tan(Vec16h const x) {
    return sincos_h<Vec16h, 4>(0, x);
}
static inline Vec16h sinpi(Vec16h const x) {
    return sincos_h<Vec16h, 9>(0, x);
}
static inline Vec16h cospi(Vec16h const x) {
    return sincos_h<Vec16h, 10>(0, x);
}
static inline Vec16h sincospi(Vec16h * cosret, Vec16h const x) {
    return sincos_h<Vec16h, 11>(cosret, x);
} 
static inline Vec16h tanpi(Vec16h const x) {
    return sincos_h<Vec16h, 12>(0, x);
}

#endif  // MAX_VECTOR_SIZE >= 256

#if MAX_VECTOR_SIZE >= 512

static inline Vec32h exp(Vec32h const x) {
    return exp_h<Vec32h, 0, 0>(x);
}
static inline Vec32h exp2(Vec32h const x) {
    return exp_h<Vec32h, 0, 2>(x);
}
static inline Vec32h exp10(Vec32h const x) {
    return exp_h<Vec32h, 0, 10>(x);
}
static inline Vec32h expm1(Vec32h const x) {
    return exp_h<Vec32h, 1, 0>(x);
}
static inline Vec32h sin(Vec32h const x) {
    return sincos_h<Vec32h, 1>(0, x);
}
static inline Vec32h cos(Vec32h const x) {
    return sincos_h<Vec32h, 2>(0, x);
}
static inline Vec32h sincos(Vec32h * cosret, Vec32h const x) {
    return sincos_h<Vec32h, 3>(cosret, x);
} 
static inline Vec32h tan(Vec32h const x) {
    return sincos_h<Vec32h, 4>(0, x);
}
static inline Vec32h sinpi(Vec32h const x) {
    return sincos_h<Vec32h, 9>(0, x);
}
static inline Vec32h cospi(Vec32h const x) {
    return sincos_h<Vec32h, 10>(0, x);
}
static inline Vec32h sincospi(Vec32h * cosret, Vec32h const x) {
    return sincos_h<Vec32h, 11>(cosret, x);
} 
static inline Vec32h tanpi(Vec32h const x) {
    return sincos_h<Vec32h, 12>(0, x);
}

#endif  // MAX_VECTOR_SIZE >= 512


#ifdef VCL_NAMESPACE
}
#endif

#endif // defined(__AVX512FP16__)

#endif // VECTORFP16_H
