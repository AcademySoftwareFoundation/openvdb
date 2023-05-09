/****************************  vectorfp16e.h   *******************************
* Author:        Agner Fog
* Date created:  2022-05-03
* Last modified: 2022-07-20
* Version:       2.02.00
* Project:       vector class library
* Description:
* Header file emulating half precision floating point vector classes
* when instruction set AVX512_FP16 is not defined
*
* Instructions: see vcl_manual.pdf
*
* The following vector classes are defined here:
* Vec8h     Vector of 8 half precision floating point numbers in 128 bit vector
* Vec16h    Vector of 16 half precision floating point numbers in 256 bit vector
* Vec32h    Vector of 32 half precision floating point numbers in 512 bit vector
*
* This header file defines operators and functions for these vectors.
*
* (c) Copyright 2012-2022 Agner Fog.
* Apache License version 2.0 or later.
*****************************************************************************/

#ifndef VECTORFP16E_H
#define VECTORFP16E_H

#ifndef VECTORCLASS_H
#include "vectorclass.h"
#endif

#if VECTORCLASS_H < 20200
#error Incompatible versions of vector class library mixed
#endif

#if MAX_VECTOR_SIZE < 256
#error Emulation of half precision floating point not supported for MAX_VECTOR_SIZE < 256
#endif

#ifdef VCL_NAMESPACE
namespace VCL_NAMESPACE {
#endif


/*****************************************************************************
*
*        Float16: Use _Float16 if it is defined, or emulate it if not
*
*****************************************************************************/


// test if _Float16 is defined
#if defined(FLT16_MAX) || defined(__FLT16_MAX__)
    // _Float16 is defined. 
    typedef _Float16 Float16;
    
    // Define bit-casting between uint16_t <-> Float16
    static inline uint16_t castfp162s(Float16 x) {
        union {
            Float16 f;
            uint16_t i;
        } u;
        u.f = x;
        return u.i;
    }
    static inline Float16 casts2fp16(uint16_t x) {
        union {
            uint16_t i;
            Float16 f;
        } u;
        u.i = x;
        return u.f;
    }
#else
    // _Float16 is not defined
    // define Float16 as a class with constructor, operators, etc. to avoid operators like + from treating Float16 like integer
    class Float16 {
    protected:
        uint16_t x;
    public:
    // Default constructor:
        Float16() = default;
#ifdef __F16C__   // F16C instruction set includes conversion instructions
    Float16(float f) { // Constructor to convert float to fp16
        //x = uint16_t(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set1_ps(f), _MM_FROUND_NO_EXC))); // requires __AVX512FP16__
        x = uint16_t(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set1_ps(f), 0)));
    }
    operator float() const {                     // Type cast operator to convert fp16 to float
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi32(x)));
    }

#else  // F16C instruction set not supported. Make conversion functions
    Float16(float f) {                           // Constructor to convert float to fp16
        union {                                  // single precision float as bitfield
            float f;
            struct {
                uint32_t mant : 23;
                uint32_t expo : 8;
                uint32_t sign : 1;
            };
        } u;
        union {                                  // half precision float as bitfield
            uint16_t h;
            struct {
                uint16_t mant : 10;
                uint16_t expo : 5;
                uint16_t sign : 1;
            };
        } v;
        u.f = f;
        v.expo = u.expo - 0x70;                  // convert exponent
        v.mant = u.mant >> 13;                   // get upper part of mantissa
        if (u.mant & (1 << 12)) {                // round to nearest or even
            if ((u.mant & ((1 << 12) - 1)) || (v.mant & 1)) { // round up if odd or remaining bits are nonzero
                v.h++;                           // overflow here will give infinity
            }
        }
        v.sign = u.sign;                         // copy sign bit
        if (u.expo == 0xFF) {                    // infinity or nan
            v.expo = 0x1F;
            if (u.mant != 0) {                   // Nan
                v.mant = u.mant >> 13;           // NAN payload is left-justified
            }
        }
        else if (u.expo > 0x8E) {
            v.expo = 0x1F;  v.mant = 0;          // overflow -> inf
        }
        else if (u.expo < 0x71) {
            v.expo = 0;                          // subnormals are always supported
            u.expo += 24;
            u.sign = 0;
            //v.mant = int(u.f) & 0x3FF;
            int mants = _mm_cvt_ss2si(_mm_load_ss(&u.f));
            v.mant = mants & 0x3FF; // proper rounding of subnormal
            if (mants == 0x400) v.expo = 1;
        }
        x = v.h;                                 // store result
    }    
    operator float() const {                     // Type cast operator to convert fp16 to float
        union {
            uint32_t hhh;
            float fff;
            struct {
                uint32_t mant : 23;
                uint32_t expo : 8;
                uint32_t sign : 1;
            };
        } u;
        u.hhh = (x & 0x7fff) << 13;              // Exponent and mantissa
        u.hhh += 0x38000000;                     // Adjust exponent bias
        if ((x & 0x7C00) == 0) {                 // Subnormal or zero
            u.hhh = 0x3F800000 - (24 << 23);     // 2^-24
            u.fff *= int(x & 0x3FF);             // subnormal value = mantissa * 2^-24
        }
        if ((x & 0x7C00) == 0x7C00) {            // infinity or nan
            u.expo = 0xFF;
            if (x & 0x3FF) {                     // nan
                u.mant = (x & 0x3FF) << 13;      // NAN payload is left-justified
            }
        }
        u.hhh |= (x & 0x8000) << 16;             // copy sign bit
        return u.fff;
    } 
#endif  // F16C supported

    void setBits(uint16_t a) {
        x = a;
    }
    uint16_t getBits() const {
        return x;
    }
    };

    static inline int16_t castfp162s(Float16 a) {
        return a.getBits();
    }
    static inline Float16 casts2fp16(int16_t a) {
        Float16 f;
        f.setBits(a);
        return f;
    }

    // Define operators for Float16 emulation class

    static inline Float16 operator + (Float16 const a, Float16 const b) {
        return Float16(float(a) + float(b));
    }
    static inline Float16 operator - (Float16 const a, Float16 const b) {
        return Float16(float(a) - float(b));
    }
    static inline Float16 operator * (Float16 const a, Float16 const b) {
        return Float16(float(a) * float(b));
    }
    static inline Float16 operator / (Float16 const a, Float16 const b) {
        return Float16(float(a) / float(b));
    }
    static inline Float16 operator - (Float16 const a) {
        return casts2fp16(castfp162s(a) ^ 0x8000);
    }
    static inline bool operator == (Float16 const a, Float16 const b) {
        return float(a) == float(b);
    }
    static inline bool operator != (Float16 const a, Float16 const b) {
        return float(a) != float(b);
    }
    static inline bool operator < (Float16 const a, Float16 const b) {
        return float(a) < float(b);
    }
    static inline bool operator <= (Float16 const a, Float16 const b) {
        return float(a) <= float(b);
    }
    static inline bool operator > (Float16 const a, Float16 const b) {
        return float(a) > float(b);
    }
    static inline bool operator >= (Float16 const a, Float16 const b) {
        return float(a) >= float(b);
    }

#endif  // Float16 defined


/*****************************************************************************
*
*          Vec8hb: Vector of 8 Booleans for use with Vec8h
*
*****************************************************************************/

#if INSTRSET >= 10
typedef Vec8b Vec8hb;   // compact boolean vector
static inline Vec8hb Vec8fb2hb (Vec8fb const a) {
    return a;
}
#else
typedef Vec8sb Vec8hb;  // broad boolean vector
static inline Vec8hb Vec8fb2hb (Vec8fb const a) {
    // boolean vector needs compression from 32 bits to 16 bits per element
    Vec4ib lo = reinterpret_i(a.get_low());
    Vec4ib hi = reinterpret_i(a.get_high());
    return _mm_packs_epi32(lo, hi);
}
#endif


/*****************************************************************************
*
*          Vec8h: Vector of 8 half precision floating point values
*
*****************************************************************************/

class Vec8h {
protected:
    __m128i xmm; // Float vector
public:
    // Default constructor:
    Vec8h() = default;
    // Constructor to broadcast the same value into all elements:
    Vec8h(Float16 f) {
        xmm = _mm_set1_epi16 (castfp162s(f));
    }
    // Constructor to build from all elements:
    Vec8h(Float16 f0, Float16 f1, Float16 f2, Float16 f3, Float16 f4, Float16 f5, Float16 f6, Float16 f7) {
        xmm = _mm_setr_epi16 (castfp162s(f0), castfp162s(f1), castfp162s(f2), castfp162s(f3), castfp162s(f4), castfp162s(f5), castfp162s(f6), castfp162s(f7));
    }
    // Constructor to convert from type __m128i used in intrinsics:
    Vec8h(__m128i const x) {
        xmm = x;
    }
    // Assignment operator to convert from type __m128i used in intrinsics:
    Vec8h & operator = (__m128i const x) {
        xmm = x;
        return *this;
    }
    // Type cast operator to convert to __m128i used in intrinsics
    operator __m128i() const {
        return xmm;
    }
    // Member function to load from array (unaligned)
    Vec8h & load(void const * p) {
        xmm = _mm_loadu_si128 ((const __m128i *)p);
        return *this;
    }
    // Member function to load from array, aligned by 16
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 16. In most cases there is no difference in speed between load and load_a
    Vec8h & load_a(void const * p) {
        xmm = _mm_load_si128 ((const __m128i *)p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        _mm_storeu_si128 ((__m128i *)p, xmm);
    }
    // Member function storing into array, aligned by 16
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 16.
    void store_a(void * p) const {
        _mm_store_si128 ((__m128i *)p, xmm);
    }
    // Member function storing to aligned uncached memory (non-temporal store).
    // This may be more efficient than store_a when storing large blocks of memory if it 
    // is unlikely that the data will stay in the cache until it is read again.
    // Note: Will generate runtime error if p is not aligned by 16
    void store_nt(void * p) const {
        _mm_stream_si128((__m128i*)p, xmm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec8h & load_partial(int n, void const * p) {
        xmm = Vec8s().load_partial(n, p);
        return *this;
    }
    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        Vec8s(xmm).store_partial(n, p);
    }
    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec8h & cutoff(int n) {
        xmm = Vec8s(xmm).cutoff(n);
        return *this;
    }
    // Member function to change a single element in vector
    Vec8h const insert(int index, Float16 a) {
        xmm = Vec8s(xmm).insert(index, castfp162s(a));
        return *this;
    }
    // Member function extract a single element from vector
    Float16 extract(int index) const {
        Float16 y;
        y = casts2fp16(Vec8s(xmm).extract(index));
        return y;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    Float16 operator [] (int index) const {
        return extract(index);
    }
    static constexpr int size() {
        return 8;
    }
    static constexpr int elementtype() {
        return 15;
    }
    typedef __m128i registertype;
};

/*****************************************************************************
*
*          conversions Vec8h <-> Vec4f
*
*****************************************************************************/

#ifdef __F16C__    // F16C instruction set has conversion instructions

// extend precision: Vec8h -> Vec4f. upper half ignored
Vec4f convert8h_4f (Vec8h h) {
    return _mm_cvtph_ps(h);
}

// reduce precision: Vec4f -> Vec8h. upper half zero
Vec8h convert4f_8h (Vec4f f) {
    return _mm_cvtps_ph(f, 0);
}

#else

// extend precision: Vec8h -> Vec4f. upper half ignored
Vec4f convert8h_4f (Vec8h x) {
    // __m128i a = _mm_cvtepu16_epi32(x);                            // SSE4.1
    __m128i a = _mm_unpacklo_epi16(x, _mm_setzero_si128 ());         // zero extend
    __m128i b = _mm_slli_epi32(a, 16);                               // left-justify
    __m128i c = _mm_and_si128(b, _mm_set1_epi32(0x80000000));        // isolate sign bit
    __m128i d = _mm_andnot_si128(_mm_set1_epi32(0x80000000),b);      // remove sign bit
    __m128i e = _mm_srli_epi32(d, 3);                                // put exponent and mantissa in place
    __m128i f = _mm_add_epi32(e, _mm_set1_epi32(0x38000000));        // adjust exponent bias
    // check for subnormal, INF, and NAN
    __m128i xx = _mm_set1_epi32(0x7C00);                             // exponent field in fp16
    __m128i g  = _mm_and_si128(a, xx);                               // isolate exponent (low position)
    __m128i zd = _mm_cmpeq_epi32(g, _mm_setzero_si128());            // -1 if x is zero or subnormal
    __m128i in = _mm_cmpeq_epi32(g, xx);                             // -1 if x is INF or NAN
    __m128i ma = _mm_and_si128(a, _mm_set1_epi32(0x3FF));            // isolate mantissa
    __m128  sn = _mm_mul_ps(_mm_cvtepi32_ps(ma), _mm_set1_ps(1.f/16777216.f)); // converted subnormal = mantissa * 2^-24
    __m128i snm = _mm_and_si128(_mm_castps_si128(sn), zd);           // converted subnormal, masked
    __m128i inm = _mm_and_si128(in,_mm_set1_epi32(0x7F800000));      // INF or NAN exponent field, masked off if not INF or NAN
    __m128i fm = _mm_andnot_si128(zd, f);                            // normal result, masked off if zero or subnormal
    __m128i r = _mm_or_si128(fm, c);                                 // insert sign bit
    __m128i s = _mm_or_si128(snm, inm);                              // combine branches
    __m128i t = _mm_or_si128(r, s);                                  // combine branches
    return _mm_castsi128_ps(t);                                      // cast result to float
}

// reduce precision: Vec4f -> Vec8h. upper half zero
Vec8h convert4f_8h (Vec4f x) {
    __m128i a = _mm_castps_si128(x);                                 // bit-cast to integer
    // 23 bit mantissa rounded to 10 bits - nearest or even
    __m128i r = _mm_srli_epi32(a, 12);                               // get first discarded mantissa bit
    __m128i s = _mm_and_si128(a, _mm_set1_epi32(0x2FFF));            // 0x2000 indicates if odd, 0x0FFF if remaining bits are nonzero
    __m128i u = _mm_cmpeq_epi32(s, _mm_setzero_si128());             // false if odd or remaining bits nonzero
    __m128i v = _mm_andnot_si128(u, r);                              // bit 0 = 1 if we have to round up
    __m128i w = _mm_and_si128(v, _mm_set1_epi32(1));                 // = 1 if we need to round up
    __m128i m = _mm_srli_epi32(a, 13);                               // get mantissa in place
    __m128i n = _mm_and_si128(m, _mm_set1_epi32(0x3FF));             // mantissa isolated
    __m128i e = _mm_and_si128(a, _mm_set1_epi32(0x7FFFFFFF));        // remove sign bit
    __m128i f = _mm_sub_epi32(e, _mm_set1_epi32(0x70 << 23));        // adjust exponent bias (underflow will be caught by uu below)
    __m128i g = _mm_srli_epi32(f, 13);                               // shift exponent into new place
    __m128i h = _mm_and_si128(g, _mm_set1_epi32(0x3FC00));           // isolate exponent 
    __m128i i = _mm_or_si128(n, h);                                  // combine exponent and mantissa
    Vec4i   j = _mm_add_epi32(i, w);                                 // round mantissa. Overflow will carry into exponent
    // check for overflow and underflow
    Vec4ib  k  = j > 0x7BFF;                                         // overflow
    Vec4i   ee = _mm_srli_epi32(e, 23);                              // exponent at position 0
    Vec4ib  ii = ee == 0xFF;                                         // check for INF and NAN
    Vec4ib  uu = ee < 0x71;                                          // check for exponent underflow
    __m128i pp = _mm_or_si128(j, _mm_set1_epi32(0x7C00));            // insert exponent if INF or NAN
    // compute potential subnormal result
    __m128i ss = _mm_add_epi32(e, _mm_set1_epi32(24 << 23));         // add 24 to exponent
    __m128i tt = _mm_cvtps_epi32(_mm_castsi128_ps(ss));              // convert float to int with rounding
    __m128i vv = _mm_and_si128(tt, _mm_set1_epi32(0x3FF));           // mantissa of subnormal number
    // combine results   
    Vec4i  bb = select(k, 0x7C00, j);                                // select INF if overflow
    Vec4i  dd = select(ii, pp, bb);                                  // select INF or NAN    
    Vec4i  cc = select(uu, vv, dd);                                  // select if subnormal or zero or exponent underflow
    // get sign bit
    Vec4i  sa = Vec4i(a) >> 16;                                      // extend sign bit to avoid saturation in pack instruction below
    Vec4i  const smask = 0xFFFF8000;                                 // extended sign mask
    Vec4i  sb = sa & smask;                                          // isolate sign
    Vec4i  sc = _mm_andnot_si128(smask, cc);                         // isolate exponent and mantissa
    Vec4i  rr = sb | sc;                                             // combine with sign
    Vec4i  rc  = _mm_packs_epi32(rr, _mm_setzero_si128());           // pack into 16-bit words (words are sign extended so they will not saturate)
    return (__m128i)rc;                                              // return as Vec8h
} 

#endif

/*****************************************************************************
*
*          conversions Vec8h <-> Vec8f
*
*****************************************************************************/
#if defined (__F16C__) && INSTRSET >= 8  // F16C instruction set has conversion instructions

// extend precision: Vec8h -> Vec8f
Vec8f to_float (Vec8h h) {
    return _mm256_cvtph_ps(h);
}

// reduce precision: Vec8f -> Vec8h
Vec8h to_float16 (Vec8f f) {
    return _mm256_cvtps_ph(f, 0);
}

#elif INSTRSET >= 8 // __F16C__ not defined, AVX2 supported

// extend precision: Vec8h -> Vec8f
static Vec8f to_float (Vec8h x) {
    __m256i a = _mm256_cvtepu16_epi32(x);                            // zero-extend each element to 32 bits
    __m256i b = _mm256_slli_epi32(a, 16);                            // left-justify
    __m256i c = _mm256_and_si256(b, _mm256_set1_epi32(0x80000000));  // isolate sign bit
    __m256i d = _mm256_andnot_si256(_mm256_set1_epi32(0x80000000),b);// remove sign bit
    __m256i e = _mm256_srli_epi32(d, 3);                             // put exponent and mantissa in place
    __m256i f = _mm256_add_epi32(e, _mm256_set1_epi32(0x38000000));  // adjust exponent bias
    // check for subnormal, INF, and NAN
    __m256i xx = _mm256_set1_epi32(0x7C00);                          // exponent field in fp16
    __m256i g  = _mm256_and_si256(a, xx);                            // isolate exponent (low position)
    __m256i zd = _mm256_cmpeq_epi32(g, _mm256_setzero_si256());      // -1 if x is zero or subnormal
    __m256i in = _mm256_cmpeq_epi32(g, xx);                          // -1 if x is INF or NAN
    __m256i ma = _mm256_and_si256(a, _mm256_set1_epi32(0x3FF));      // isolate mantissa
    __m256  sn = _mm256_mul_ps(_mm256_cvtepi32_ps(ma), _mm256_set1_ps(1.f/16777216.f)); // converted subnormal = mantissa * 2^-24
    __m256i snm = _mm256_and_si256(_mm256_castps_si256(sn), zd);     // converted subnormal, masked
    __m256i inm = _mm256_and_si256(in,_mm256_set1_epi32(0x7F800000));// INF or NAN exponent field, masked off if not INF or NAN
    __m256i fm = _mm256_andnot_si256(zd, f);                         // normal result, masked off if zero or subnormal
    __m256i r = _mm256_or_si256(fm, c);                              // insert sign bit
    __m256i s = _mm256_or_si256(snm, inm);                           // combine branches
    __m256i t = _mm256_or_si256(r, s);                               // combine branches
    return _mm256_castsi256_ps(t);                                   // cast result to float
}

// reduce precision: Vec8f -> Vec8h
static Vec8h to_float16 (Vec8f x) {
    __m256i a = _mm256_castps_si256(x);                              // bit-cast to integer
    // 23 bit mantissa rounded to 10 bits - nearest or even
    __m256i r = _mm256_srli_epi32(a, 12);                            // get first discarded mantissa bit
    __m256i s = _mm256_and_si256(a, _mm256_set1_epi32(0x2FFF));      // 0x2000 indicates if odd, 0x0FFF if remaining bits are nonzero
    __m256i u = _mm256_cmpeq_epi32(s, _mm256_setzero_si256());       // false if odd or remaining bits nonzero
    __m256i v = _mm256_andnot_si256(u, r);                           // bit 0 = 1 if we have to round up
    __m256i w = _mm256_and_si256(v, _mm256_set1_epi32(1));           // = 1 if we need to round up
    __m256i m = _mm256_srli_epi32(a, 13);                            // get mantissa in place
    __m256i n = _mm256_and_si256(m, _mm256_set1_epi32(0x3FF));       // mantissa isolated
    __m256i e = _mm256_and_si256(a, _mm256_set1_epi32(0x7FFFFFFF));  // remove sign bit
    __m256i f = _mm256_sub_epi32(e, _mm256_set1_epi32(0x70 << 23));  // adjust exponent bias (underflow will be caught by uu below)
    __m256i g = _mm256_srli_epi32(f, 13);                            // shift exponent into new place
    __m256i h = _mm256_and_si256(g, _mm256_set1_epi32(0x3FC00));     // isolate exponent 
    __m256i i = _mm256_or_si256(n, h);                               // combine exponent and mantissa
    __m256i j = _mm256_add_epi32(i, w);                              // round mantissa. Overflow will carry into exponent
    // check for overflow and underflow
    __m256i k = _mm256_cmpgt_epi32(j, _mm256_set1_epi32(0x7BFF));    // overflow
    __m256i ee = _mm256_srli_epi32(e, 23);                           // exponent at position 0
    __m256i ii = _mm256_cmpeq_epi32(ee, _mm256_set1_epi32(0xFF));    // check for INF and NAN
    __m256i uu = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x71), ee);    // check for exponent underflow
    __m256i pp = _mm256_or_si256(j, _mm256_set1_epi32(0x7C00));      // insert exponent if INF or NAN
    // compute potential subnormal result
    __m256i ss = _mm256_add_epi32(e, _mm256_set1_epi32(24 << 23));   // add 24 to exponent
    __m256i tt = _mm256_cvtps_epi32(_mm256_castsi256_ps(ss));        // convert float to int with rounding
    __m256i vv = _mm256_and_si256(tt, _mm256_set1_epi32(0x7FF));     // mantissa of subnormal number (possible overflow to normal)
    // combine results
    __m256i bb = _mm256_blendv_epi8(j, _mm256_set1_epi32(0x7C00), k);// select INF if overflow
    __m256i dd = _mm256_blendv_epi8(bb, pp, ii);                     // select INF or NAN    
    __m256i cc = _mm256_blendv_epi8(dd, vv, uu);                     // select if subnormal or zero or exponent underflow
    __m256i sa = _mm256_srai_epi32(a, 16);                           // extend sign bit to avoid saturation in pack instruction below
    __m256i sb = _mm256_and_si256(sa, _mm256_set1_epi32(0xFFFF8000));// isolate sign
    __m256i sc = _mm256_andnot_si256(_mm256_set1_epi32(0xFFFF8000), cc);// isolate exponent and mantissa
    __m256i rr = _mm256_or_si256(sb, sc);                            // combine with sign
    __m128i rl = _mm256_castsi256_si128(rr);                         // low half of results
    __m128i rh = _mm256_extractf128_si256(rr, 1);                    // high half of results
    __m128i rc = _mm_packs_epi32(rl, rh);                            // pack into 16-bit words (words are sign extended so they will not saturate)
    return  rc;                                                      // return as Vec8h
} 

#else // __F16C__ not defined, AVX2 not supported 

// extend precision: Vec8h -> Vec8f
static Vec8f to_float (Vec8h x) {
    Vec8s  xx = __m128i(x);
    Vec4ui a1 = _mm_unpacklo_epi16(xx, _mm_setzero_si128 ());
    Vec4ui a2 = _mm_unpackhi_epi16(xx, _mm_setzero_si128 ());
    Vec4ui b1 = a1 << 16;                        // left-justify
    Vec4ui b2 = a2 << 16;
    Vec4ui c1 = b1 & 0x80000000;                 // isolate sign bit
    Vec4ui c2 = b2 & 0x80000000;
    Vec4ui d1 = _mm_andnot_si128(Vec4ui(0x80000000), b1); // remove sign bit
    Vec4ui d2 = _mm_andnot_si128(Vec4ui(0x80000000), b2);
    Vec4ui e1 = d1 >> 3;                         // put exponent and mantissa in place
    Vec4ui e2 = d2 >> 3;
    Vec4ui f1 = e1 + 0x38000000;                 // adjust exponent bias
    Vec4ui f2 = e2 + 0x38000000;
    Vec4ui g1 = a1 & 0x7C00;                     // isolate exponent (low position)
    Vec4ui g2 = a2 & 0x7C00;
    Vec4ib z1 = g1 == 0;                         // true if x is zero or subnormal (broad boolean vector)
    Vec4ib z2 = g2 == 0;
    Vec4ib i1 = g1 == 0x7C00;                    // true if x is INF or NAN
    Vec4ib i2 = g2 == 0x7C00;
    Vec4ui m1 = a1 & 0x3FF;                      // isolate mantissa (low position)
    Vec4ui m2 = a2 & 0x3FF;
    Vec4f  s1 = to_float(m1) * (1.f/16777216.f); // converted subnormal = mantissa * 2^-24
    Vec4f  s2 = to_float(m2) * (1.f/16777216.f);
    Vec4ui sm1 = Vec4ui(reinterpret_i(s1)) & Vec4ui(z1); // converted subnormal, masked
    Vec4ui sm2 = Vec4ui(reinterpret_i(s2)) & Vec4ui(z2);
    Vec4ui inm1 = Vec4ui(i1) & Vec4ui(0x7F800000); // INF or NAN exponent field, masked off if not INF or NAN 
    Vec4ui inm2 = Vec4ui(i2) & Vec4ui(0x7F800000);
    Vec4ui fm1 = _mm_andnot_si128(Vec4ui(z1), f1); // normal result, masked off if zero or subnormal
    Vec4ui fm2 = _mm_andnot_si128(Vec4ui(z2), f2);
    Vec4ui r1 = fm1 | c1;                        // insert sign bit
    Vec4ui r2 = fm2 | c2;
    Vec4ui q1 = sm1 | inm1;                      // combine branches
    Vec4ui q2 = sm2 | inm2;
    Vec4ui t1 = r1  | q1;                        // combine branches
    Vec4ui t2 = r2  | q2;
    Vec4f  u1 = reinterpret_f(t1);               // bit-cast to float
    Vec4f  u2 = reinterpret_f(t2);
    return Vec8f(u1, u2);                        // combine low and high part
} 

// reduce precision: Vec8f -> Vec8h
static Vec8h to_float16 (Vec8f x) {              
    Vec4ui a1 = _mm_castps_si128(x.get_low());             // low half
    Vec4ui a2 = _mm_castps_si128(x.get_high());            // high half
    Vec4ui r1 = a1 >> 12;                                  // get first discarded mantissa bit
    Vec4ui r2 = a2 >> 12;
    Vec4ui s1 = a1 & 0x2FFF;                               // 0x2000 indicates if odd, 0x0FFF if remaining bits are nonzero
    Vec4ui s2 = a2 & 0x2FFF;
    Vec4ib u1 = s1 == 0;                                   // false if odd or remaining bits nonzero
    Vec4ib u2 = s2 == 0;
    Vec4ui v1 = _mm_andnot_si128(u1, r1);                  // bit 0 = 1 if we have to round up
    Vec4ui v2 = _mm_andnot_si128(u2, r2);
    Vec4ui w1 = v1 & 1;                                    // = 1 if we need to round up
    Vec4ui w2 = v2 & 1;
    Vec4ui m1 = a1 >> 13;                                  // get mantissa in place
    Vec4ui m2 = a2 >> 13;
    Vec4ui n1 = m1 & 0x3FF;                                // mantissa isolated
    Vec4ui n2 = m2 & 0x3FF;
    Vec4ui e1 = a1 & 0x7FFFFFFF;                           // remove sign bit
    Vec4ui e2 = a2 & 0x7FFFFFFF;
    Vec4ui f1 = e1 - (0x70 << 23);                         // adjust exponent bias
    Vec4ui f2 = e2 - (0x70 << 23);
    Vec4ui g1 = f1 >> 13;                                  // shift exponent into new place
    Vec4ui g2 = f2 >> 13;
    Vec4ui h1 = g1 & 0x3FC00;                              // isolate exponent 
    Vec4ui h2 = g2 & 0x3FC00;
    Vec4ui i1 = n1 | h1;                                   // combine exponent and mantissa
    Vec4ui i2 = n2 | h2;
    Vec4ui j1 = i1 + w1;                                   // round mantissa. Overflow will carry into exponent
    Vec4ui j2 = i2 + w2;
    // check for overflow and underflow
    Vec4ib k1 = j1 > 0x7BFF;                               // overflow
    Vec4ib k2 = j2 > 0x7BFF;
    Vec4ui ee1 = e1 >> 23;                                 // exponent at position 0
    Vec4ui ee2 = e2 >> 23;
    Vec4ib ii1 = ee1 == 0xFF;                              // check for INF and NAN
    Vec4ib ii2 = ee2 == 0xFF;
    Vec4ib uu1 = ee1 < 0x71;                               // exponent underflow
    Vec4ib uu2 = ee2 < 0x71;
    Vec4i  pp1 = Vec4i(0x7C00) | j1;                       // insert exponent if INF or NAN
    Vec4i  pp2 = Vec4i(0x7C00) | j2;
    // compute potential subnormal result
    Vec4ui ss1 = e1 + (24 << 23);                          // add 24 to exponent
    Vec4ui ss2 = e2 + (24 << 23);
    Vec4ui tt1 = _mm_cvtps_epi32(_mm_castsi128_ps(ss1));   // convert float to int with rounding
    Vec4ui tt2 = _mm_cvtps_epi32(_mm_castsi128_ps(ss2));
    Vec4ui vv1 = tt1 & 0x7FF;                              // mantissa of subnormal number (possible overflow to normal)
    Vec4ui vv2 = tt2 & 0x7FF;
    // combine results
    Vec4i  bb1 = select(k1, 0x7C00, j1);                   // select INF if overflow
    Vec4i  bb2 = select(k2, 0x7C00, j2);
    Vec4i  dd1 = select(ii1, pp1, bb1);                    // select INF or NAN    
    Vec4i  dd2 = select(ii2, pp2, bb2);
    Vec4i  cc1 = select(uu1, vv1, dd1);                    // select if subnormal or zero or exponent underflow
    Vec4i  cc2 = select(uu2, vv2, dd2);
    // get sign bit
    Vec4i  sa1 = Vec4i(a1) >> 16;                          // extend sign bit to avoid saturation in pack instruction below
    Vec4i  sa2 = Vec4i(a2) >> 16;
    Vec4i  const smask = 0xFFFF8000;                       // extended sign mask
    Vec4i  sb1 = sa1 & smask;                              // isolate sign
    Vec4i  sb2 = sa2 & smask;
    Vec4i  sc1 = _mm_andnot_si128(smask, cc1);             // isolate exponent and mantissa
    Vec4i  sc2 = _mm_andnot_si128(smask, cc2);
    Vec4i  rr1 = sb1 | sc1;                                // combine with sign
    Vec4i  rr2 = sb2 | sc2;
    Vec4i  rc  = _mm_packs_epi32(rr1, rr2);                // pack into 16-bit words (words are sign extended so they will not saturate)
    return (__m128i)rc;                                    // return as Vec8h
}

#endif  // __F16C__


/*****************************************************************************
*
*          Operators for Vec8h
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec8h operator + (Vec8h const a, Vec8h const b) {
    return to_float16(to_float(a) + to_float(b));
}

// vector operator + : add vector and scalar
static inline Vec8h operator + (Vec8h const a, Float16 b) {
    return a + Vec8h(b);
}
static inline Vec8h operator + (Float16 a, Vec8h const b) {
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
    a = a + Float16(1.f); // 1.0f16 not supported by g++ version 12.1
    return a0;
}

// prefix operator ++
static inline Vec8h & operator ++ (Vec8h & a) {
    a = a +  Float16(1.f);
    return a;
}

// vector operator - : subtract element by element
static inline Vec8h operator - (Vec8h const a, Vec8h const b) {
    return to_float16(to_float(a) - to_float(b));
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
    return _mm_xor_si128(__m128i(a), _mm_set1_epi32(0x80008000));
}

// vector operator -= : subtract
static inline Vec8h & operator -= (Vec8h & a, Vec8h const b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec8h operator -- (Vec8h & a, int) {
    Vec8h a0 = a;
    a = a - Vec8h(Float16(1.f));
    return a0;
}

// prefix operator --
static inline Vec8h & operator -- (Vec8h & a) {
    a = a - Vec8h(Float16(1.f));
    return a;
}

// vector operator * : multiply element by element
static inline Vec8h operator * (Vec8h const a, Vec8h const b) {
    return to_float16(to_float(a) * to_float(b));
}

// vector operator * : multiply vector and scalar
static inline Vec8h operator * (Vec8h const a, Float16 b) {
    return a * Vec8h(b);
}
static inline Vec8h operator * (Float16 a, Vec8h const b) {
    return Vec8h(a) * b;
}

// vector operator *= : multiply
static inline Vec8h & operator *= (Vec8h & a, Vec8h const b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec8h operator / (Vec8h const a, Vec8h const b) {
    return to_float16(to_float(a) / to_float(b));
}

// vector operator / : divide vector and scalar
static inline Vec8h operator / (Vec8h const a, Float16 b) {
    return a / Vec8h(b);
}
static inline Vec8h operator / (Float16 a, Vec8h const b) {
    return Vec8h(a) / b;
}

// vector operator /= : divide
static inline Vec8h & operator /= (Vec8h & a, Vec8h const b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec8hb operator == (Vec8h const a, Vec8h const b) {
    return Vec8fb2hb(to_float(a) == to_float(b));
}

// vector operator != : returns true for elements for which a != b
static inline Vec8hb operator != (Vec8h const a, Vec8h const b) {
    return Vec8fb2hb(to_float(a) != to_float(b));
}

// vector operator < : returns true for elements for which a < b
static inline Vec8hb operator < (Vec8h const a, Vec8h const b) {
    return Vec8fb2hb(to_float(a) < to_float(b));
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec8hb operator <= (Vec8h const a, Vec8h const b) {
    return Vec8fb2hb(to_float(a) <= to_float(b));
}

// vector operator > : returns true for elements for which a > b
static inline Vec8hb operator > (Vec8h const a, Vec8h const b) {
    return Vec8fb2hb(to_float(a) > to_float(b));
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec8hb operator >= (Vec8h const a, Vec8h const b) {
    return Vec8fb2hb(to_float(a) >= to_float(b));
}

// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec8h operator & (Vec8h const a, Vec8h const b) {
    return _mm_and_si128(__m128i(a), __m128i(b));
}

// vector operator &= : bitwise and
static inline Vec8h & operator &= (Vec8h & a, Vec8h const b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec8h and Vec8hb
static inline Vec8h operator & (Vec8h const a, Vec8hb const b) {
#if INSTRSET >= 10  // compact boolean vector
    return _mm_maskz_mov_epi16(b, a);
#else               // broad boolean vector
    return _mm_and_si128(__m128i(a), __m128i(b));
#endif
}
static inline Vec8h operator & (Vec8hb const a, Vec8h const b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec8h operator | (Vec8h const a, Vec8h const b) {
    return _mm_or_si128(__m128i(a), __m128i(b));
}

// vector operator |= : bitwise or
static inline Vec8h & operator |= (Vec8h & a, Vec8h const b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8h operator ^ (Vec8h const a, Vec8h const b) {
    return _mm_xor_si128(__m128i(a), __m128i(b));
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
    return __m128i(select(Vec8sb(s), Vec8s(__m128i(a)), Vec8s(__m128i(b))));
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec8h if_add(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return a + (b & f);
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
static inline Vec8h if_sub(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return a - (b & f);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec8h if_mul(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return select(f, a*b, a);
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] / b[i]) : a[i]
static inline Vec8h if_div(Vec8hb const f, Vec8h const a, Vec8h const b) {
    return select(f, a/b, a);
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec8h(-0.0f16)) gives true, while Vec8h(-0.0f16) < Vec8h(0.0f16) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb sign_bit(Vec8h const a) {
    Vec8s t1 = __m128i(a);             // reinterpret as 16-bit integer
    Vec8s t2 = t1 >> 15;               // extend sign bit
    return t2 != 0;
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec8h sign_combine(Vec8h const a, Vec8h const b) {
    return a ^ (b & Vec8h(Float16(-0.0)));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb is_finite(Vec8h const a) {
    Vec8s b = __m128i(a);
    return (b & 0x7C00) != 0x7C00;
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb is_inf(Vec8h const a) {
    Vec8s b = __m128i(a);
    return (b & 0x7FFF) == 0x7C00;
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec8hb is_nan(Vec8h const a) {
    Vec8s b = __m128i(a);
    return (b & 0x7FFF) > 0x7C00;
}

// Function is_subnormal: gives true for elements that are subnormal
// false for finite numbers, zero, NAN and INF
static inline Vec8hb is_subnormal(Vec8h const a) {
    Vec8s b = __m128i(a);
    return (b & 0x7C00) == 0 && (b & 0x3FF) != 0;
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal
// false for finite numbers, NAN and INF
static inline Vec8hb is_zero_or_subnormal(Vec8h const a) {
    Vec8s b = __m128i(a);
    return (b & 0x7C00) == 0;
}

// Function infinite8h: returns a vector where all elements are +INF
static inline Vec8h infinite8h() {
    return Vec8h(_mm_set1_epi16(0x7C00));
}

// template for producing quiet NAN
template <>
Vec8h nan_vec<Vec8h>(uint32_t payload) {
    if constexpr (Vec8h::elementtype() == 15) {  // Float16
        return Vec8h(_mm_set1_epi16(0x7E00 | (payload & 0x01FF)));
    }
} 

// Function nan8h: returns a vector where all elements are NAN (quiet)
static inline Vec8h nan8h(int n = 0x10) {
    return nan_vec<Vec8h>(n);
}

// This function returns the code hidden in a NAN. The sign bit is ignored
static inline Vec8us nan_code(Vec8h const x) {
    Vec8us a = Vec8us(reinterpret_i(x));
    Vec8us const n = 0x3FF;
    return select(is_nan(x), a & n, Vec8us(0));
}

// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline Float16 horizontal_add(Vec8h const a) {
    return Float16(horizontal_add(to_float(a)));
}
// same, with high precision
static inline float horizontal_add_x(Vec8h const a) {
    return horizontal_add(to_float(a));
}

// function max: a > b ? a : b
static inline Vec8h max(Vec8h const a, Vec8h const b) {
    return to_float16(max(to_float(a), to_float(b)));
}

// function min: a < b ? a : b
static inline Vec8h min(Vec8h const a, Vec8h const b) {
    return to_float16(min(to_float(a), to_float(b)));
}
// NAN-safe versions of maximum and minimum are in vector_convert.h

// function abs: absolute value
static inline Vec8h abs(Vec8h const a) {
    return _mm_and_si128(a, _mm_set1_epi16(0x7FFF));
}

// function sqrt: square root
static inline Vec8h sqrt(Vec8h const a) {
    return to_float16(sqrt(to_float(a)));
}

// function square: a * a
static inline Vec8h square(Vec8h const a) {
    return a * a;
}

// The purpose of this template is to prevent implicit conversion of a float
// exponent to int when calling pow(vector, float) and vectormath_exp.h is not included
template <typename TT> static Vec8h pow(Vec8h const a, TT const n);  // = delete

// Raise floating point numbers to integer power n
// To do: Optimize pow<int>(Vec8h/Vec16h/Vec32h, n) to do calculations with float precision
template <>
inline Vec8h pow<int>(Vec8h const x0, int const n) {
    return to_float16(pow_template_i<Vec8f>(to_float(x0), n));
}

// allow conversion from unsigned int
template <>
inline Vec8h pow<uint32_t>(Vec8h const x0, uint32_t const n) {
    return to_float16(pow_template_i<Vec8f>(to_float(x0), (int)n));
}

// Raise floating point numbers to integer power n, where n is a compile-time constant:
// Template in vectorf128.h is used
//template <typename V, int n>
//static inline V pow_n(V const a);

// implement as function pow(vector, const_int)
template <int n>
static inline Vec8h pow(Vec8h const a, Const_int_t<n>) {
    return to_float16(pow_n<Vec8f, n>(to_float(a)));
}


static inline Vec8h round(Vec8h const a) {
    return to_float16(round(to_float(a)));
}

// function truncate: round towards zero. (result as float vector)
static inline Vec8h truncate(Vec8h const a) {
    return to_float16(truncate(to_float(a)));
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec8h floor(Vec8h const a) {
    return to_float16(floor(to_float(a)));
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec8h ceil(Vec8h const a) {
    return to_float16(ceil(to_float(a)));
}

// function roundi: round to nearest integer (even). (result as integer vector)
static inline Vec8s roundi(Vec8h const a) {
    return compress_saturated(roundi(to_float(a)));
}

// function truncatei: round towards zero. (result as integer vector)
static inline Vec8s truncatei(Vec8h const a) {
    //return compress(truncatei(to_float(a)));
    return compress_saturated(truncatei(to_float(a)));
}

// function to_float: convert integer vector to float vector
static inline Vec8h to_float16(Vec8s const a) {
    return to_float16(to_float(extend(a)));
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec8h to_float16(Vec8us const a) {
    return to_float16(to_float(extend(a)));
}

// Approximate math functions

// reciprocal (almost exact)
static inline Vec8h approx_recipr(Vec8h const a) {
    return to_float16(approx_recipr(to_float(a)));
}

// reciprocal squareroot (almost exact)
static inline Vec8h approx_rsqrt(Vec8h const a) {
    return to_float16(approx_rsqrt(to_float(a)));
}

// Fused multiply and add functions

// Multiply and add. a*b+c
static inline Vec8h mul_add(Vec8h const a, Vec8h const b, Vec8h const c) {
    return to_float16(mul_add(to_float(a),to_float(b),to_float(c)));
}

// Multiply and subtract. a*b-c
static inline Vec8h mul_sub(Vec8h const a, Vec8h const b, Vec8h const c) {
    return to_float16(mul_sub(to_float(a),to_float(b),to_float(c)));
}

// Multiply and inverse subtract
static inline Vec8h nmul_add(Vec8h const a, Vec8h const b, Vec8h const c) {
    return to_float16(nmul_add(to_float(a),to_float(b),to_float(c)));
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec8s exponent(Vec8h const a) {
    Vec8us t1 = __m128i(a);            // reinterpret as 16-bit integer
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
    Vec8us t1 = __m128i(a);   // reinterpret as 16-bit integer
    Vec8us t2 = Vec8us((t1 & 0x3FF) | 0x3C00); // set exponent to 0 + bias
    return __m128i(t2);
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
    return __m128i(t4);                // bit-cast to float
}

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
    return _mm_xor_si128(a, mask);
}


/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/
static inline __m128i reinterpret_h(__m128i const x) {
    return x;
}
/* Defined in vectorf128.h:
 __m128i reinterpret_i(__m128i const x)
 __m128  reinterpret_f(__m128i const x)
 __m128d reinterpret_d(__m128i const x)
*/


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
    return __m128i(permute8<i0, i1, i2, i3, i4, i5, i6, i7>(Vec8s(__m128i(a))));
}


/*****************************************************************************
*
*          Vector blend functions
*
*****************************************************************************/

// permute and blend Vec8h
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline Vec8h blend8(Vec8h const a, Vec8h const b) {
    return __m128i (blend8<i0, i1, i2, i3, i4, i5, i6, i7>(Vec8s(__m128i(a)), Vec8s(__m128i(b))));
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
    return __m128i(lookup8(index, Vec8s(__m128i(table))));
}

static inline Vec8h lookup16(Vec8s const index, Vec8h const table0, Vec8h const table1) {
    return __m128i(lookup16(index, Vec8s(__m128i(table0)), Vec8s(__m128i(table1))));
}

template <int n>
static inline Vec8h lookup(Vec8s const index, void const * table) {
    return __m128i(lookup<n>(index, (void const *)(table)));
}



/*****************************************************************************
*
*          256 bit vectors
*
*****************************************************************************/

#if MAX_VECTOR_SIZE >= 512


/*****************************************************************************
*
*          Vec16hb: Vector of 16 Booleans for use with Vec16h
*
*****************************************************************************/

#if INSTRSET >= 10
typedef Vec16b Vec16hb;   // compact boolean vector

#if MAX_VECTOR_SIZE >= 512
static inline Vec16hb Vec16fb2hb (Vec16fb const a) {
    return a;
}
#endif

#else

typedef Vec16sb Vec16hb;  // broad boolean vector

static inline Vec16hb Vec16fb2hb (Vec16fb const a) {
    // boolean vector needs compression from 32 bits to 16 bits per element
    Vec8fb lo = a.get_low();           // (cannot use _mm256_packs_epi32)
    Vec8fb hi = a.get_high();
    return Vec16hb(Vec8fb2hb(lo), Vec8fb2hb(hi));
}

#endif


/*****************************************************************************
*
*          Vec16h: Vector of 16 single precision floating point values
*
*****************************************************************************/

class Vec16h : public Vec16s {
public:
    // Default constructor:
    Vec16h() = default;
    // Constructor to broadcast the same value into all elements:
    Vec16h(Float16 f) : Vec16s(castfp162s(f)) {}
    Vec16h(float f) : Vec16s(castfp162s(Float16(f))) {}

    // Constructor to build from all elements:
    Vec16h(Float16 f0, Float16 f1, Float16 f2, Float16 f3, Float16 f4, Float16 f5, Float16 f6, Float16 f7,
    Float16 f8, Float16 f9, Float16 f10, Float16 f11, Float16 f12, Float16 f13, Float16 f14, Float16 f15) :
        Vec16s(castfp162s(f0), castfp162s(f1), castfp162s(f2), castfp162s(f3), castfp162s(f4), castfp162s(f5), castfp162s(f6), castfp162s(f7), 
            castfp162s(f8), castfp162s(f9), castfp162s(f10), castfp162s(f11), castfp162s(f12), castfp162s(f13), castfp162s(f14), castfp162s(f15)) {}

    // Constructor to build from two Vec8h:
    Vec16h(Vec8h const a0, Vec8h const a1) : Vec16s(Vec8s(a0), Vec8s(a1)) {};

#if INSTRSET >= 8
    // Constructor to convert from type __m256i used in intrinsics:
    Vec16h(__m256i const x) {
        ymm = x;
    }
    // Assignment operator to convert from type __m256i used in intrinsics:
    Vec16h & operator = (__m256i const x) {
        ymm = x;
        return *this;
    }
    // Type cast operator to convert to __m256i used in intrinsics
    operator __m256i() const {
        return ymm;
    }
#else
    // Constructor to convert from type Vec16s. This may cause undesired implicit conversions and ambiguities
    //Vec16h(Vec16s const x) : Vec16s(x) {}
#endif
    // Member function to load from array (unaligned)
    Vec16h & load(void const * p) {
        Vec16s::load(p);
        return *this;
    }
    // Member function to load from array, aligned by 32
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 32. In most cases there is no difference in speed between load and load_a
    Vec16h & load_a(void const * p) {
        Vec16s::load_a(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    // void store(void * p) const // inherited from Vec16s

    // Member function storing into array, aligned by 32
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 32.
    // void store_a(void * p) const // inherited from Vec16s 

    // Member function storing to aligned uncached memory (non-temporal store).
    // This may be more efficient than store_a when storing large blocks of memory if it 
    // is unlikely that the data will stay in the cache until it is read again.
    // Note: Will generate runtime error if p is not aligned by 32
    // void store_nt(void * p) const // inherited from Vec16s 

    // Partial load. Load n elements and set the rest to 0
    Vec16h & load_partial(int n, void const * p) {
        Vec16s::load_partial(n, p);
        return *this;
    }
    // Partial store. Store n elements
    // void store_partial(int n, void * p) const // inherited from Vec16s 

    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec16h & cutoff(int n) {
        Vec16s::cutoff(n);
        return *this;
    }
    // Member function to change a single element in vector
    Vec16h const insert(int index, Float16 a) {
        Vec16s::insert(index, castfp162s(a));
        return *this;
    }
    // Member function extract a single element from vector
    Float16 extract(int index) const {
        return casts2fp16(Vec16s::extract(index));
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    Float16 operator [] (int index) const {
        return extract(index);
    }
    Vec8h get_low() const {
        return __m128i(Vec16s::get_low());
    }
    Vec8h get_high() const {
        return __m128i(Vec16s::get_high());
    }
    static constexpr int size() {
        return 16;
    }
    static constexpr int elementtype() {
        return 15;
    }
};

/*****************************************************************************
*
*          conversions Vec16h <-> Vec16f
*
*****************************************************************************/
#if INSTRSET >= 9    // AVX512F instruction set has conversion instructions

// extend precision: Vec16h -> Vec16f
Vec16f to_float (Vec16h h) {
    return _mm512_cvtph_ps(h);
}

// reduce precision: Vec16f -> Vec16h
Vec16h to_float16 (Vec16f f) {
    return _mm512_cvtps_ph(f, 0);
}

#else

// extend precision: Vec16h -> Vec16f
Vec16f to_float (Vec16h h) {
    return Vec16f(to_float(h.get_low()), to_float(h.get_high()));
}

// reduce precision: Vec16f -> Vec16h
Vec16h to_float16 (Vec16f f) {
    return Vec16h(to_float16(f.get_low()), to_float16(f.get_high()));
}

#endif

/*****************************************************************************
*
*          Operators for Vec16h
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec16h operator + (Vec16h const a, Vec16h const b) {
    return to_float16(to_float(a) + to_float(b));
}


static inline Vec16h operator + (Float16 a, Vec16h const b) {
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
    a = a + Float16(1.f);
    return a0;
}

// prefix operator ++
static inline Vec16h & operator ++ (Vec16h & a) {
    a = a + Float16(1.f);
    return a;
}

// vector operator - : subtract element by element
static inline Vec16h operator - (Vec16h const a, Vec16h const b) {
    return to_float16(to_float(a) - to_float(b));
}

// vector operator - : subtract vector and scalar
static inline Vec16h operator - (Vec16h const a, Float16 b) {
    return a - Vec16h(b);
}
static inline Vec16h operator - (Float16 a, Vec16h const b) {
    return Vec16h(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec16h operator - (Vec16h const a) {
#if INSTRSET >= 8  // AVX2
    return _mm256_xor_si256(a, _mm256_set1_epi32(0x80008000));
#else
    return Vec16h(-a.get_low(), -a.get_high());
#endif
}

// vector operator -= : subtract
static inline Vec16h & operator -= (Vec16h & a, Vec16h const b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec16h operator -- (Vec16h & a, int) {
    Vec16h a0 = a;
    a = a - Vec16h(Float16(1.f));
    return a0;
}

// prefix operator --
static inline Vec16h & operator -- (Vec16h & a) {
    a = a - Vec16h(Float16(1.f));
    return a;
}

// vector operator * : multiply element by element
static inline Vec16h operator * (Vec16h const a, Vec16h const b) {
    return to_float16(to_float(a) * to_float(b));
}

// vector operator * : multiply vector and scalar
static inline Vec16h operator * (Vec16h const a, Float16 b) {
    return a * Vec16h(b);
}
static inline Vec16h operator * (Float16 a, Vec16h const b) {
    return Vec16h(a) * b;
}

// vector operator *= : multiply
static inline Vec16h & operator *= (Vec16h & a, Vec16h const b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec16h operator / (Vec16h const a, Vec16h const b) {
    return to_float16(to_float(a) / to_float(b));
}

// vector operator / : divide vector and scalar
static inline Vec16h operator / (Vec16h const a, Float16 b) {
    return a / Vec16h(b);
}
static inline Vec16h operator / (Float16 a, Vec16h const b) {
    return Vec16h(a) / b;
}

// vector operator /= : divide
static inline Vec16h & operator /= (Vec16h & a, Vec16h const b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec16hb operator == (Vec16h const a, Vec16h const b) {
    return Vec16fb2hb(to_float(a) == to_float(b));
}

// vector operator != : returns true for elements for which a != b
static inline Vec16hb operator != (Vec16h const a, Vec16h const b) {
    return Vec16fb2hb(to_float(a) != to_float(b));
}

// vector operator < : returns true for elements for which a < b
static inline Vec16hb operator < (Vec16h const a, Vec16h const b) {
    return Vec16fb2hb(to_float(a) < to_float(b));
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec16hb operator <= (Vec16h const a, Vec16h const b) {
    return Vec16fb2hb(to_float(a) <= to_float(b));
}

// vector operator > : returns true for elements for which a > b
static inline Vec16hb operator > (Vec16h const a, Vec16h const b) {
    return Vec16fb2hb(to_float(a) > to_float(b));
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec16hb operator >= (Vec16h const a, Vec16h const b) {
    return Vec16fb2hb(to_float(a) >= to_float(b));
}


// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec16h operator & (Vec16h const a, Vec16h const b) {
#if INSTRSET >= 8         
    return _mm256_and_si256(__m256i(a), __m256i(b));
#else
    return Vec16h(a.get_low() & b.get_low(), a.get_high() & b.get_high());
#endif
}

// vector operator &= : bitwise and
static inline Vec16h & operator &= (Vec16h & a, Vec16h const b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec16h and Vec16hb
static inline Vec16h operator & (Vec16h const a, Vec16hb const b) {
#if INSTRSET >= 10         
    return __m256i(_mm256_maskz_mov_epi16(b, __m256i(a)));
#elif INSTRSET >= 8
    return _mm256_and_si256(__m256i(a), __m256i(b));
#else
    return Vec16h(a.get_low() & b.get_low(), a.get_high() & b.get_high());
#endif
}
static inline Vec16h operator & (Vec16hb const a, Vec16h const b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec16h operator | (Vec16h const a, Vec16h const b) {
#if INSTRSET >= 8         
    return _mm256_or_si256(__m256i(a), __m256i(b));
#else
    return Vec16h(a.get_low() | b.get_low(), a.get_high() | b.get_high());
#endif
}

// vector operator |= : bitwise or
static inline Vec16h & operator |= (Vec16h & a, Vec16h const b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16h operator ^ (Vec16h const a, Vec16h const b) {
#if INSTRSET >= 8         
    return _mm256_xor_si256(__m256i(a), __m256i(b));
#else
    return Vec16h(a.get_low() ^ b.get_low(), a.get_high() ^ b.get_high());
#endif
}

// vector operator ^= : bitwise xor
static inline Vec16h & operator ^= (Vec16h & a, Vec16h const b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec16hb operator ! (Vec16h const a) {
    return a == Vec16h(Float16(0.0f));
}

/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/
#if INSTRSET >= 8
static inline __m256i reinterpret_h(__m256i const x) {
    return x;
}

#if defined(__GNUC__) && __GNUC__ <= 9 // GCC v. 9 is missing the _mm256_zextsi128_si256 intrinsic
static inline Vec16h extend_z(Vec8h a) {
    return Vec16h(a, Vec8h(Float16(0.f)));
}

#else
static inline Vec16h extend_z(Vec8h a) {
    return _mm256_zextsi128_si256(a);
}
#endif

#else // INSTRSET

static inline Vec16h reinterpret_h(Vec16s const x) {
    return Vec16h(Vec8h(x.get_low()), Vec8h(x.get_high()));
}

static inline Vec16s reinterpret_i(Vec16h const x) {
    return Vec16s(Vec8s(x.get_low()), Vec8s(x.get_high()));
}

static inline Vec16h extend_z(Vec8h a) {
    return Vec16h(a, Vec8h(0));
}

#endif  // INSTRSET


/*****************************************************************************
*
*          Functions for Vec16h
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec16h select(Vec16hb const s, Vec16h const a, Vec16h const b) {
#if INSTRSET >= 10
    return __m256i(_mm256_mask_mov_epi16(__m256i(b), s, __m256i(a)));
#elif INSTRSET >= 8
    return __m256i(select(Vec16sb(s), Vec16s(__m256i(a)), Vec16s(__m256i(b))));
#else
    return Vec16h(select(s.get_low(), a.get_low(), b.get_low()), select(s.get_high(), a.get_high(), b.get_high()));
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec16h if_add(Vec16hb const f, Vec16h const a, Vec16h const b) {
#if INSTRSET >= 8
    return a + (b & f);
#else
    return select(f, a+b, a);
#endif
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
static inline Vec16h if_sub(Vec16hb const f, Vec16h const a, Vec16h const b) {
#if INSTRSET >= 8
    return a - (b & f);
#else
    return select(f, a-b, a);
#endif
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec16h if_mul(Vec16hb const f, Vec16h const a, Vec16h const b) {
    return select(f, a*b, a);
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] / b[i]) : a[i]
static inline Vec16h if_div(Vec16hb const f, Vec16h const a, Vec16h const b) {
    return select(f, a/b, a);
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec16h(-0.0f16)) gives true, while Vec16h(-0.0f16) < Vec16h(0.0f16) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb sign_bit(Vec16h const a) {
    Vec16s t1 = reinterpret_i(a);                // reinterpret as 16-bit integer
    Vec16s t2 = t1 >> 15;                        // extend sign bit
    return t2 != 0;
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec16h sign_combine(Vec16h const a, Vec16h const b) {
    return a ^ (b & Vec16h(Float16(-0.0)));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb is_finite(Vec16h const a) {
    return (Vec16s(reinterpret_i(a)) & 0x7C00) != 0x7C00;
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb is_inf(Vec16h const a) {
    return (Vec16s(reinterpret_i(a)) & 0x7FFF) == 0x7C00;
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec16hb is_nan(Vec16h const a) {
    return (Vec16s(reinterpret_i(a)) & 0x7FFF) > 0x7C00;
}

// Function is_subnormal: gives true for elements that are subnormal
// false for finite numbers, zero, NAN and INF
static inline Vec16hb is_subnormal(Vec16h const a) {
    return (Vec16s(reinterpret_i(a)) & 0x7C00) == 0 && (Vec16s(reinterpret_i(a)) & 0x03FF) != 0;
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal
// false for finite numbers, NAN and INF
static inline Vec16hb is_zero_or_subnormal(Vec16h const a) {
    return (Vec16s(reinterpret_i(a)) & 0x7C00) == 0;
} 

// Function infinite16h: returns a vector where all elements are +INF
static inline Vec16h infinite16h() {
    return reinterpret_h(Vec16s(0x7C00));
}

// template for producing quiet NAN
template <>
Vec16h nan_vec<Vec16h>(uint32_t payload) {
    if constexpr (Vec16h::elementtype() == 15) {  // Float16
        return reinterpret_h(Vec16s(0x7E00 | (payload & 0x01FF)));
    }
} 

// Function nan16h: returns a vector where all elements are NAN (quiet)
static inline Vec16h nan16h(int n = 0x10) {
    return nan_vec<Vec16h>(n);
}

// This function returns the code hidden in a NAN. The sign bit is ignored
static inline Vec16us nan_code(Vec16h const x) {
    Vec16us a = Vec16us(reinterpret_i(x));
    Vec16us const n = 0x3FF;
    return select(is_nan(x), a & n, Vec16us(0));
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline Float16 horizontal_add(Vec16h const a) {
    return horizontal_add(a.get_low()+a.get_high());
}
// same, with high precision
static inline float horizontal_add_x(Vec16h const a) {
    return horizontal_add(to_float(a));
}

// function max: a > b ? a : b
static inline Vec16h max(Vec16h const a, Vec16h const b) {
    return to_float16(max(to_float(a), to_float(b)));
}

// function min: a < b ? a : b
static inline Vec16h min(Vec16h const a, Vec16h const b) {
    return to_float16(min(to_float(a), to_float(b)));
}
// NAN-safe versions of maximum and minimum are in vector_convert.h

// function abs: absolute value
static inline Vec16h abs(Vec16h const a) {
    return reinterpret_h(Vec16s(reinterpret_i(a)) & 0x7FFF);
}

// function sqrt: square root
static inline Vec16h sqrt(Vec16h const a) {
    return to_float16(sqrt(to_float(a)));
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
    return to_float16(round(to_float(a)));
}

// function truncate: round towards zero. (result as float vector)
static inline Vec16h truncate(Vec16h const a) {
    return to_float16(truncate(to_float(a)));
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec16h floor(Vec16h const a) {
    return to_float16(floor(to_float(a)));
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec16h ceil(Vec16h const a) {
    return to_float16(ceil(to_float(a)));
}


// function roundi: round to nearest integer (even). (result as integer vector)
static inline Vec16s roundi(Vec16h const a) {
    // Note: assume MXCSR control register is set to rounding
    return compress_saturated(roundi(to_float(a)));
}

// function truncatei: round towards zero. (result as integer vector)
static inline Vec16s truncatei(Vec16h const a) {
    return compress_saturated(truncatei(to_float(a)));
}

// function to_float: convert integer vector to float vector
static inline Vec16h to_float16(Vec16s const a) {
    return to_float16(to_float(extend(a)));
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec16h to_float16(Vec16us const a) {
    return to_float16(to_float(extend(a)));
}


// Approximate math functions

// reciprocal (almost exact)
static inline Vec16h approx_recipr(Vec16h const a) {
    return to_float16(approx_recipr(to_float(a)));
}

// reciprocal squareroot (almost exact)
static inline Vec16h approx_rsqrt(Vec16h const a) {
    return to_float16(approx_rsqrt(to_float(a)));
}

// Fused multiply and add functions

// Multiply and add. a*b+c
static inline Vec16h mul_add(Vec16h const a, Vec16h const b, Vec16h const c) {
    return to_float16(mul_add(to_float(a),to_float(b),to_float(c)));
}

// Multiply and subtract. a*b-c
static inline Vec16h mul_sub(Vec16h const a, Vec16h const b, Vec16h const c) {
    return to_float16(mul_sub(to_float(a),to_float(b),to_float(c)));
}

// Multiply and inverse subtract
static inline Vec16h nmul_add(Vec16h const a, Vec16h const b, Vec16h const c) {
    return to_float16(nmul_add(to_float(a),to_float(b),to_float(c)));
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec16s exponent(Vec16h const a) {
    Vec16us t1 = reinterpret_i(a);         // reinterpret as 16-bit integer
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
    Vec16us t1 = reinterpret_i(a);   // reinterpret as 16-bit integer
    Vec16us t2 = Vec16us((t1 & 0x3FF) | 0x3C00); // set exponent to 0 + bias
    return reinterpret_h(t2);
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
    return reinterpret_h(t4);           // reinterpret as float
}

// change signs on vectors Vec16h
// Each index i0 - i15 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline Vec16h change_sign(Vec16h const a) {
#if INSTRSET >= 8
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
    return Vec16h(_mm256_xor_si256(a, mask));     // flip sign bits
#else
    return Vec16h(change_sign<i0,i1,i2,i3,i4,i5,i6,i7>(a.get_low()), change_sign<i8,i9,i10,i11,i12,i13,i14,i15>(a.get_high()));
#endif
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
    return reinterpret_h (
    permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15> (
    Vec16s(reinterpret_i(a))));
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
    return reinterpret_h (
    blend16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15> (
    Vec16s(reinterpret_i(a)), Vec16s(reinterpret_i(b))));
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
    return reinterpret_h(lookup16(index, Vec16s(reinterpret_i(table))));
}

template <int n>
static inline Vec16h lookup(Vec16s const index, void const * table) {
    return reinterpret_h(lookup<n>(index, (void const *)(table)));
}

// prevent implicit type conversions
bool horizontal_and(Vec16h x) = delete;
bool horizontal_or(Vec16h x) = delete;

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

#if INSTRSET >= 10
typedef Vec32b Vec32hb;   // compact boolean vector

#else

typedef Vec32sb Vec32hb;  // broad boolean vector

#endif

 
/*****************************************************************************
*
*          Vec32h: Vector of 4 single precision floating point values
*
*****************************************************************************/

class Vec32h : public Vec32s {
public:
    // Default constructor:
    Vec32h() = default;
    // Constructor to broadcast the same value into all elements:
    Vec32h(Float16 f) : Vec32s(castfp162s(f)) {}   
    Vec32h(float f) : Vec32s(castfp162s(Float16(f))) {} 

    // Copy constructor
    Vec32h (Vec32h const &x) = default;

    // Copy assignment operator
    Vec32h & operator = (Vec32h const& x) = default;

    // Constructor to build from all elements:
    Vec32h(Float16 f0, Float16 f1, Float16 f2, Float16 f3, Float16 f4, Float16 f5, Float16 f6, Float16 f7,
    Float16 f8, Float16 f9, Float16 f10, Float16 f11, Float16 f12, Float16 f13, Float16 f14, Float16 f15,
    Float16 f16, Float16 f17, Float16 f18, Float16 f19, Float16 f20, Float16 f21, Float16 f22, Float16 f23,
    Float16 f24, Float16 f25, Float16 f26, Float16 f27, Float16 f28, Float16 f29, Float16 f30, Float16 f31) :
        Vec32s (castfp162s(f0), castfp162s(f1), castfp162s(f2), castfp162s(f3), castfp162s(f4), castfp162s(f5), castfp162s(f6), castfp162s(f7), 
            castfp162s(f8), castfp162s(f9), castfp162s(f10), castfp162s(f11), castfp162s(f12), castfp162s(f13), castfp162s(f14), castfp162s(f15),
            castfp162s(f16), castfp162s(f17), castfp162s(f18), castfp162s(f19), castfp162s(f20), castfp162s(f21), castfp162s(f22), castfp162s(f23),
            castfp162s(f24), castfp162s(f25), castfp162s(f26), castfp162s(f27), castfp162s(f28), castfp162s(f29), castfp162s(f30), castfp162s(f31))
    {}
    // Constructor to build from two Vec16h:
    Vec32h(Vec16h const a0, Vec16h const a1) : Vec32s(Vec16h(a0), Vec16h(a1)) {}

    // Constructor to convert from type __m512i used in intrinsics:
#if INSTRSET >= 10
    Vec32h(__m512i const x) {
        zmm = x;
    }
    // Assignment operator to convert from type __m256i used in intrinsics:
    Vec32h & operator = (__m512i const x) {
        zmm = x;
        return *this;
    }
    // Type cast operator to convert to __m256i used in intrinsics
    operator __m512i() const {
        return zmm;
    }
#else
    // Constructor to convert from type Vec32s. This may cause undesired implicit conversions and ambiguities
    // Vec32h(Vec32s const x) : Vec32s(x) {  }
#endif
    // Member function to load from array (unaligned)
    Vec32h & load(void const * p) {
        Vec32s::load(p);
        return *this;
    }
    // Member function to load from array, aligned by 64
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 64. In most cases there is no difference in speed between load and load_a
    Vec32h & load_a(void const * p) {
        Vec32s::load_a(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    // void store(void * p) const // inherited from Vec32s

    // Member function storing into array, aligned by 64
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 64.
    //void store_a(void * p) const // inherited from Vec32s

    // Member function storing to aligned uncached memory (non-temporal store).
    // This may be more efficient than store_a when storing large blocks of memory if it 
    // is unlikely that the data will stay in the cache until it is read again.
    // Note: Will generate runtime error if p is not aligned by 64
    // void store_nt(void * p) const // inherited from Vec32s

    // Partial load. Load n elements and set the rest to 0
    Vec32h & load_partial(int n, void const * p) {
        Vec32s::load_partial(n, p);
        return *this;
    }
    // Partial store. Store n elements
    // void store_partial(int n, void * p) const // inherited from Vec32s

    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec32h & cutoff(int n) {
        Vec32s::cutoff(n);
        return *this;
    }
    // Member function to change a single element in vector
    Vec32h const insert(int index, Float16 a) {
        Vec32s::insert(index, castfp162s(a));
        return *this;
    }
    // Member function extract a single element from vector
    Float16 extract(int index) const {
        return casts2fp16(Vec32s::extract(index));
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    Float16 operator [] (int index) const {
        return extract(index);
    }
    Vec16h get_low() const {
#if INSTRSET >= 8
        return __m256i(Vec32s::get_low());
#else
        return reinterpret_h(Vec32s::get_low());
#endif
    }

    Vec16h get_high() const {
#if INSTRSET >= 8
        return __m256i(Vec32s::get_high());
#else
        return reinterpret_h(Vec32s::get_high());
#endif
    }
    static constexpr int size() {
        return 32;
    }
    static constexpr int elementtype() {
        return 15;
    }
};


/*****************************************************************************
*
*          Operators for Vec32h
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec32h operator + (Vec32h const a, Vec32h const b) {
    return Vec32h(a.get_low() + b.get_low(), a.get_high() + b.get_high());
}

// vector operator + : add vector and scalar
static inline Vec32h operator + (Vec32h const a, Float16 b) {
    return a + Vec32h(b);
}
static inline Vec32h operator + (Float16 a, Vec32h const b) {
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
    a = a + Float16(1.f);
    return a0;
}

// prefix operator ++
static inline Vec32h & operator ++ (Vec32h & a) {
    a = a + Float16(1.f);
    return a;
}

// vector operator - : subtract element by element
static inline Vec32h operator - (Vec32h const a, Vec32h const b) {
    return Vec32h(a.get_low() - b.get_low(), a.get_high() - b.get_high());
}

// vector operator - : subtract vector and scalar
static inline Vec32h operator - (Vec32h const a, Float16 b) {
    return a - Vec32h(b);
}
static inline Vec32h operator - (Float16 a, Vec32h const b) {
    return Vec32h(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec32h operator - (Vec32h const a) {
#if INSTRSET >= 10  // AVX2
    return _mm512_xor_si512(a, _mm512_set1_epi32(0x80008000));
#else
    return Vec32h(-a.get_low(), -a.get_high());
#endif
}

// vector operator -= : subtract
static inline Vec32h & operator -= (Vec32h & a, Vec32h const b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec32h operator -- (Vec32h & a, int) {
    Vec32h a0 = a;
    a = a - Vec32h(Float16(1.f));
    return a0;
}

// prefix operator --
static inline Vec32h & operator -- (Vec32h & a) {
    a = a - Vec32h(Float16(1.f));
    return a;
}

// vector operator * : multiply element by element
static inline Vec32h operator * (Vec32h const a, Vec32h const b) {
    return Vec32h(a.get_low() * b.get_low(), a.get_high() * b.get_high());
}

// vector operator * : multiply vector and scalar
static inline Vec32h operator * (Vec32h const a, Float16 b) {
    return a * Vec32h(b);
}
static inline Vec32h operator * (Float16 a, Vec32h const b) {
    return Vec32h(a) * b;
}

// vector operator *= : multiply
static inline Vec32h & operator *= (Vec32h & a, Vec32h const b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec32h operator / (Vec32h const a, Vec32h const b) {
    return Vec32h(a.get_low() / b.get_low(), a.get_high() / b.get_high());
}

// vector operator / : divide vector and scalar
static inline Vec32h operator / (Vec32h const a, Float16 b) {
    return a / Vec32h(b);
}
static inline Vec32h operator / (Float16 a, Vec32h const b) {
    return Vec32h(a) / b;
}

// vector operator /= : divide
static inline Vec32h & operator /= (Vec32h & a, Vec32h const b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec32hb operator == (Vec32h const a, Vec32h const b) {
    return Vec32hb(a.get_low() == b.get_low(), a.get_high() == b.get_high());
}

// vector operator != : returns true for elements for which a != b
static inline Vec32hb operator != (Vec32h const a, Vec32h const b) {
    return Vec32hb(a.get_low() != b.get_low(), a.get_high() != b.get_high());
}

// vector operator < : returns true for elements for which a < b
static inline Vec32hb operator < (Vec32h const a, Vec32h const b) {
    return Vec32hb(a.get_low() < b.get_low(), a.get_high() < b.get_high());
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec32hb operator <= (Vec32h const a, Vec32h const b) {
    return Vec32hb(a.get_low() <= b.get_low(), a.get_high() <= b.get_high());
}

// vector operator > : returns true for elements for which a > b
static inline Vec32hb operator > (Vec32h const a, Vec32h const b) {
    return Vec32hb(a.get_low() > b.get_low(), a.get_high() > b.get_high());
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec32hb operator >= (Vec32h const a, Vec32h const b) {
    return Vec32hb(a.get_low() >= b.get_low(), a.get_high() >= b.get_high());
}


// Bitwise logical operators

// vector operator & : bitwise and
static inline Vec32h operator & (Vec32h const a, Vec32h const b) {
#if INSTRSET >= 10         
    return _mm512_and_si512(__m512i(a), __m512i(b));
#else
    return Vec32h(a.get_low() & b.get_low(), a.get_high() & b.get_high());
#endif
}

// vector operator &= : bitwise and
static inline Vec32h & operator &= (Vec32h & a, Vec32h const b) {
    a = a & b;
    return a;
}

// vector operator & : bitwise and of Vec32h and Vec32hb
static inline Vec32h operator & (Vec32h const a, Vec32hb const b) {
#if INSTRSET >= 10         
    return _mm512_maskz_mov_epi16(b, a);
#else
    return Vec32h(a.get_low() & b.get_low(), a.get_high() & b.get_high());
#endif
}
static inline Vec32h operator & (Vec32hb const a, Vec32h const b) {
    return b & a;
}

// vector operator | : bitwise or
static inline Vec32h operator | (Vec32h const a, Vec32h const b) {
#if INSTRSET >= 10         
    return _mm512_or_si512(__m512i(a), __m512i(b));
#else
    return Vec32h(a.get_low() | b.get_low(), a.get_high() | b.get_high());
#endif
}

// vector operator |= : bitwise or
static inline Vec32h & operator |= (Vec32h & a, Vec32h const b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec32h operator ^ (Vec32h const a, Vec32h const b) {
#if INSTRSET >= 10         
    return _mm512_xor_si512(__m512i(a), __m512i(b));
#else
    return Vec32h(a.get_low() ^ b.get_low(), a.get_high() ^ b.get_high());
#endif
}

// vector operator ^= : bitwise xor
static inline Vec32h & operator ^= (Vec32h & a, Vec32h const b) {
    a = a ^ b;
    return a;
}

// vector operator ! : logical not. Returns Boolean vector
static inline Vec32hb operator ! (Vec32h const a) {
    return a == Vec32h(Float16(0.f));
}


/*****************************************************************************
*
*          Functions for reinterpretation between vector types
*
*****************************************************************************/
#if INSTRSET >= 10
static inline __m512i reinterpret_h(__m512i const x) {
    return x;
}

#if defined(__GNUC__) && __GNUC__ <= 9 // GCC v. 9 is missing the _mm512_zextsi256_si512 intrinsic
static inline Vec32h extend_z(Vec16h a) {
    return Vec32h(a, Vec16h(0));
}
#else
static inline Vec32h extend_z(Vec16h a) {
    return _mm512_zextsi256_si512(a);
}
#endif
#else

static inline Vec32h reinterpret_h(Vec32s const x) {
    return Vec32h(Vec16h(reinterpret_h(x.get_low())), Vec16h(reinterpret_h(x.get_high())));
}

static inline Vec32s reinterpret_i(Vec32h const x) {
    return Vec32s(Vec16s(x.get_low()), Vec16s(x.get_high()));
}

static inline Vec32h extend_z(Vec16h a) {
    return Vec32h(a, Vec16h(Float16(0.f)));
}

#endif


/*****************************************************************************
*
*          Functions for Vec32h
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec32h select(Vec32hb const s, Vec32h const a, Vec32h const b) {
#if INSTRSET >= 10
    return __m512i(_mm512_mask_mov_epi16(__m512i(b), s, __m512i(a)));
#else
    return Vec32h(select(s.get_low(), a.get_low(), b.get_low()), select(s.get_high(), a.get_high(), b.get_high()));
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
static inline Vec32h if_add(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return a + (b & f);
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
static inline Vec32h if_sub(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return a - (b & f);
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
static inline Vec32h if_mul(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return select(f, a*b, a);
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] / b[i]) : a[i]
static inline Vec32h if_div(Vec32hb const f, Vec32h const a, Vec32h const b) {
    return select(f, a/b, a);
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(Vec32h(-0.0f16)) gives true, while Vec32h(-0.0f16) < Vec32h(0.0f16) gives false
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb sign_bit(Vec32h const a) {
    Vec32s t1 = reinterpret_i(a);          // reinterpret as 16-bit integer
    Vec32s t2 = t1 >> 15;                  // extend sign bit
    return t2 != 0;
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
static inline Vec32h sign_combine(Vec32h const a, Vec32h const b) {
    return a ^ (b & Vec32h(Float16(-0.0)));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb is_finite(Vec32h const a) {
    return (Vec32s(reinterpret_i(a)) & 0x7C00) != 0x7C00;
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb is_inf(Vec32h const a) {
    return (Vec32s(reinterpret_i(a)) & 0x7FFF) == 0x7C00;
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
static inline Vec32hb is_nan(Vec32h const a) {
    return (Vec32s(reinterpret_i(a)) & 0x7FFF) > 0x7C00;
}

// Function is_subnormal: gives true for elements that are subnormal
// false for finite numbers, zero, NAN and INF
static inline Vec32hb is_subnormal(Vec32h const a) {
    return (Vec32s(reinterpret_i(a)) & 0x7C00) == 0 && (Vec32s(reinterpret_i(a)) & 0x03FF) != 0;
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal
// false for finite numbers, NAN and INF
static inline Vec32hb is_zero_or_subnormal(Vec32h const a) {
    return (Vec32s(reinterpret_i(a)) & 0x7C00) == 0;
}

// Function infinite32h: returns a vector where all elements are +INF
static inline Vec32h infinite32h() {
    return reinterpret_h(Vec32s(0x7C00));
}

// template for producing quiet NAN
template <>
Vec32h nan_vec<Vec32h>(uint32_t payload) {
    if constexpr (Vec32h::elementtype() == 15) {  // Float16
        return reinterpret_h(Vec32s(0x7E00 | (payload & 0x01FF)));
    }
} 

// Function nan32h: returns a vector where all elements are NAN (quiet)
static inline Vec32h nan32h(int n = 0x10) {
    return nan_vec<Vec32h>(n);
}

// This function returns the code hidden in a NAN. The sign bit is ignored
static inline Vec32us nan_code(Vec32h const x) {
    Vec32us a = Vec32us(reinterpret_i(x));
    Vec32us const n = 0x3FF;
    return select(is_nan(x), a & n, Vec32us(0));
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline Float16 horizontal_add(Vec32h const a) {
    return horizontal_add(a.get_low()+a.get_high());
}
// same, with high precision
static inline float horizontal_add_x(Vec32h const a) {
    return horizontal_add_x(a.get_low()) + horizontal_add_x(a.get_high());
}

// function max: a > b ? a : b
static inline Vec32h max(Vec32h const a, Vec32h const b) {
        return Vec32h(max(a.get_low(), b.get_low()), max(a.get_high(), b.get_high()));
} 
// function min: a < b ? a : b
static inline Vec32h min(Vec32h const a, Vec32h const b) {
        return Vec32h(min(a.get_low(), b.get_low()), min(a.get_high(), b.get_high()));
}
// NAN-safe versions of maximum and minimum are in vector_convert.h

// function abs: absolute value
static inline Vec32h abs(Vec32h const a) {
    return reinterpret_h(Vec32s(reinterpret_i(a)) & 0x7FFF);
}

// function sqrt: square root
static inline Vec32h sqrt(Vec32h const a) {
    return Vec32h(sqrt(a.get_low()), sqrt(a.get_high()));
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
    return Vec32h(round(a.get_low()), round(a.get_high()));
}

// function truncate: round towards zero. (result as float vector)
static inline Vec32h truncate(Vec32h const a) {
    return Vec32h(truncate(a.get_low()), truncate(a.get_high()));
}

// function floor: round towards minus infinity. (result as float vector)
static inline Vec32h floor(Vec32h const a) {
    return Vec32h(floor(a.get_low()), floor(a.get_high()));
}

// function ceil: round towards plus infinity. (result as float vector)
static inline Vec32h ceil(Vec32h const a) {
    return Vec32h(ceil(a.get_low()), ceil(a.get_high()));
}

// function roundi: round to nearest integer (even). (result as integer vector)
static inline Vec32s roundi(Vec32h const a) {
    return Vec32s(roundi(a.get_low()), roundi(a.get_high()));
}

// function truncatei: round towards zero. (result as integer vector)
static inline Vec32s truncatei(Vec32h const a) {
    return Vec32s(truncatei(a.get_low()), truncatei(a.get_high()));
}

// function to_float: convert integer vector to float vector
static inline Vec32h to_float16(Vec32s const a) {
    return Vec32h(to_float16(a.get_low()), to_float16(a.get_high()));
}

// function to_float: convert unsigned integer vector to float vector
static inline Vec32h to_float16(Vec32us const a) {
    return Vec32h(to_float16(a.get_low()), to_float16(a.get_high()));
}

// Approximate math functions

// reciprocal (almost exact)
static inline Vec32h approx_recipr(Vec32h const a) {
    return Vec32h(approx_recipr(a.get_low()), approx_recipr(a.get_high()));
}

// reciprocal squareroot (almost exact)
static inline Vec32h approx_rsqrt(Vec32h const a) {
    return Vec32h(approx_rsqrt(a.get_low()), approx_rsqrt(a.get_high()));
}

// Fused multiply and add functions

// Multiply and add. a*b+c
static inline Vec32h mul_add(Vec32h const a, Vec32h const b, Vec32h const c) {
    return Vec32h(mul_add(a.get_low(), b.get_low(), c.get_low()), mul_add(a.get_high(), b.get_high(), c.get_high()));
}

// Multiply and subtract. a*b-c
static inline Vec32h mul_sub(Vec32h const a, Vec32h const b, Vec32h const c) {
    return Vec32h(mul_sub(a.get_low(), b.get_low(), c.get_low()), mul_sub(a.get_high(), b.get_high(), c.get_high()));
}

// Multiply and inverse subtract
static inline Vec32h nmul_add(Vec32h const a, Vec32h const b, Vec32h const c) {
    return Vec32h(nmul_add(a.get_low(), b.get_low(), c.get_low()), nmul_add(a.get_high(), b.get_high(), c.get_high()));
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
static inline Vec32s exponent(Vec32h const a) {
    Vec32us t1 = reinterpret_i(a);         // reinterpret as 16-bit integer
    Vec32us t2 = t1 << 1;                  // shift out sign bit
    Vec32us t3 = t2 >> 11;                 // shift down logical to position 0
    Vec32s  t4 = Vec32s(t3) - Vec32s(0x0F);// subtract bias from exponent
    return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
static inline Vec32h fraction(Vec32h const a) {
    Vec32us t1 = reinterpret_i(a);   // reinterpret as 16-bit integer
    Vec32us t2 = Vec32us((t1 & 0x3FF) | 0x3C00); // set exponent to 0 + bias
    return reinterpret_h(t2);
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  16 gives +INF
// n <= -15 gives 0.0f
// This function will never produce subnormals, and never raise exceptions
static inline Vec32h exp2(Vec32s const n) {
    Vec32s t1 = max(n, -15);            // limit to allowed range
    Vec32s t2 = min(t1, 16);
    Vec32s t3 = t2 + Vec32s(15);        // add bias
    Vec32s t4 = t3 << 10;               // put exponent into position 10
    return reinterpret_h(t4);           // reinterpret as float
}


// change signs on vectors Vec32h
// Each index i0 - i31 is 1 for changing sign on the corresponding element, 0 for no change
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15,
int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31 >
static inline Vec32h change_sign(Vec32h const a) {
    
#if INSTRSET >= 10
    if constexpr ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | i11 | i12 | i13 | i14 | i15) == 0) return a;
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
    return  _mm512_xor_si512(a, mask);     // flip sign bits
#else
    return Vec32h(change_sign<i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15>(a.get_low()), 
        change_sign<i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31>(a.get_high()));
#endif
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
    return reinterpret_h (
    permute32<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
    i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31 > (
    Vec32s(reinterpret_i(a))));
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
    return reinterpret_h (
    blend32<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
    i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31 > (
    Vec32s(reinterpret_i(a)), Vec32s(reinterpret_i(b))));
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
    return reinterpret_h(lookup32(index, Vec32s(reinterpret_i(table))));
}

template <int n>
static inline Vec32h lookup(Vec32s const index, void const * table) {
    return reinterpret_h(lookup<n>(index, (void const *)(table)));
}

// prevent implicit type conversions
bool horizontal_and(Vec32h x) = delete;
bool horizontal_or(Vec32h x) = delete;


#endif // MAX_VECTOR_SIZE >= 512


/*****************************************************************************
*
*          Mathematical functions
*
*****************************************************************************/

template <typename V>
static inline V vf_pow2n (V const n) {
    typedef decltype(roundi(n)) VI;
    const float pow2_23 =  8388608.0;            // 2^23
    const float bias = 127.0;                    // bias in exponent
    V a = n + (bias + pow2_23);                  // put n + bias in least significant bits
    VI b = reinterpret_i(a);                     // bit-cast to integer
    VI c = b << 23;                              // shift left 23 places to get into exponent field
    V d = reinterpret_f(c);                      // bit-cast back to float
    return d;
}

// Template for exp function, half precision
// The limit of abs(x) is defined by max_x below
// This function does not produce denormals
// Template parameters:
// VTYPE:  float vector type
// M1: 0 for exp, 1 for expm1
// BA: 0 for exp, 1 for 0.5*exp, 2 for pow(2,x), 10 for pow(10,x)

template<typename VTYPE, int M1, int BA>
static inline VTYPE exp_h(VTYPE const initial_x) { 
    // Taylor coefficients
    const float P0expf   =  1.f/2.f;
    const float P1expf   =  1.f/6.f;
    const float P2expf   =  1.f/24.f;
    VTYPE  x, r, x2, z, n2;                      // data vectors
    // maximum abs(x), value depends on BA, defined below
    // The lower limit of x is slightly more restrictive than the upper limit.
    // We are specifying the lower limit, except for BA = 1 because it is not used for negative x
    float max_x;
    if constexpr (BA <= 1) { // exp(x)
        //const float ln2f_hi  =  0.693359375f;
        //const float ln2f_lo  = -2.12194440e-4f;
        const float ln2f  =  0.69314718f;
        max_x = (BA == 0) ? 87.3f : 89.0f;
        x = initial_x;
        r = round(initial_x*float(1.44269504089f)); //VM_LOG2E
        x = nmul_add(r, VTYPE(ln2f), x);         //  x -= r * ln2f;
    }
    else if constexpr (BA == 2) {                // pow(2,x)
        max_x = 126.f;
        r = round(initial_x);
        x = initial_x - r;
        x = x * 0.69314718056f; // (float)VM_LN2;
    }
    else if constexpr (BA == 10) {               // pow(10,x)
        max_x = 37.9f;
        const float log10_2 = 0.30102999566f;   // log10(2)
        x = initial_x;
        r = round(initial_x*float(3.32192809489f)); // VM_LOG2E*VM_LN10
        x = nmul_add(r, VTYPE(log10_2), x);      //  x -= r * log10_2
        x = x * 2.30258509299f;  // (float)VM_LN10;
    }
    else  {  // undefined value of BA
        return 0.;
    }
    x2 = x * x;
    //z = polynomial_2(x,P0expf,P1expf,P2expf);
    z = mul_add(x2, P2expf, mul_add(x, P1expf, P0expf));
    z = mul_add(z, x2, x);                       // z *= x2;  z += x;
    if constexpr (BA == 1) r--;                  // 0.5 * exp(x)
    n2 = vf_pow2n(r);                            // multiply by power of 2
    if constexpr (M1 == 0) {                     // exp        
        z = (z + 1.0f) * n2;
    }
    else {                                       // expm1
        z = mul_add(z, n2, n2 - 1.0f);           //  z = z * n2 + (n2 - 1.0f);
#ifdef SIGNED_ZERO                               // pedantic preservation of signed zero
        z = select(initial_x == 0.f, initial_x, z);
#endif
    }
    // check for overflow
    auto inrange  = abs(initial_x) < max_x;      // boolean vector
    // check for INF and NAN
    inrange &= is_finite(initial_x);
    if (horizontal_and(inrange)) {               // fast normal path
        return z;
    }
    else {
        // overflow, underflow and NAN
        VTYPE inf = 1.e20f;                                // will overflow to INF
        r = select(sign_bit(initial_x), 0.f-(M1&1), inf);  // value in case of +/- overflow or INF
        z = select(inrange, z, r);                         // +/- underflow
        z = select(is_nan(initial_x), initial_x, z);       // NAN goes through
        return z;
    }
}


// Template for trigonometric functions
// Template parameters:
// VTYPE:  vector type
// SC:     1 = sin, 2 = cos, 3 = sincos, 4 = tan, 8 = multiply by pi
// Parameters:
// xx = input x (radians)
// cosret = return pointer (only if SC = 3)
template<typename VTYPE, int SC>
static inline VTYPE sincos_h(VTYPE * cosret, VTYPE const xx) {

    // define constants
    const float DP1F = 0.78515625f * 2.f;
    const float DP2F = 2.4187564849853515625E-4f * 2.f;
    const float DP3F = 3.77489497744594108E-8f * 2.f;

    const float P0sinf = -1.6666654611E-1f;
    const float P1sinf = 8.3321608736E-3f;

    const float P0cosf = 4.166664568298827E-2f;
    const float P1cosf = -1.388731625493765E-3f;

    const float pi     = 3.14159265358979323846f;// pi
    const float c2_pi  = float(2./3.14159265358979323846); // 2/pi

    typedef decltype(roundi(xx)) ITYPE;          // integer vector type
    typedef decltype(xx < xx) BVTYPE;            // boolean vector type

    VTYPE  xa, x, y, x2, s, c, sin1, cos1;       // data vectors
    ITYPE  q;                                    // integer vector
    BVTYPE swap;                                 // boolean vector

    xa = abs(xx);

    // Find quadrant
    if constexpr ((SC & 8) != 0) {
        y = round(xa * VTYPE(2.0f));
    }
    else {
        xa = select(xa > VTYPE(314.25f), VTYPE(0.f), xa); // avoid meaningless results for high x
        y = round(xa * c2_pi);                   // quadrant, as float
    }
    q = roundi(y);                               // quadrant, as integer
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
#if INSTRSET < 8
        x = ((xa - y * DP1F) - y * DP2F) - y * DP3F; // accuracy 2 ULP without FMA
#else
        x = nmul_add(y, DP2F + DP3F, nmul_add(y, DP1F, xa)); // accuracy 1 ULP with FMA
#endif
    }

    // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    x2 = x * x;
    s = mul_add(x2, P1sinf, P0sinf) * (x*x2) + x;
    c = mul_add(x2, P1cosf, P0cosf) * (x2*x2) + nmul_add(0.5f, x2, 1.0f); 
    // s = P0sinf * (x*x2) + x;  // 2 ULP error
    // c = P0cosf * (x2*x2) + nmul_add(0.5f, x2, 1.0f);  // 2 ULP error

    // swap sin and cos if odd quadrant
    swap = BVTYPE((q & 1) != 0);

    if constexpr ((SC & 5) != 0) {               // get sin
        sin1 = select(swap, c, s);
        ITYPE signsin = ((q << 30) ^ ITYPE(reinterpret_i(xx))); // sign
        sin1 = sign_combine(sin1, reinterpret_f(signsin));
    }
    if constexpr ((SC & 6) != 0) {               // get cos
        cos1 = select(swap, s, c);               // sign
        ITYPE signcos = ((q + 1) & 2) << 30;
        cos1 ^= reinterpret_f(signcos);
    }
    // select return
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

// Instantiations of templates

static inline Vec8h exp(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = exp_h<Vec8f, 0, 0>(xf);
    return to_float16(yf);
}

static inline Vec8h exp2(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = exp_h<Vec8f, 0, 2>(xf);
    return to_float16(yf);
}

static inline Vec8h exp10(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = exp_h<Vec8f, 0, 10>(xf);
    return to_float16(yf);
}

static inline Vec8h expm1(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = exp_h<Vec8f, 1, 0>(xf);
    return to_float16(yf);
}

static inline Vec8h sin(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = sincos_h<Vec8f, 1>(0, xf);
    return to_float16(yf);
}
static inline Vec8h cos(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = sincos_h<Vec8f, 2>(0, xf);
    return to_float16(yf);
}
static inline Vec8h sincos(Vec8h * cosret, Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f cf;  // cos return
    Vec8f yf = sincos_h<Vec8f, 3>(&cf, xf);
    if (cosret) *cosret = to_float16(cf);
    return to_float16(yf);
}
static inline Vec8h tan(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = sincos_h<Vec8f, 4>(0, xf);
    return to_float16(yf);
}

static inline Vec8h sinpi(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = sincos_h<Vec8f, 9>(0, xf);
    return to_float16(yf);
}
static inline Vec8h cospi(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = sincos_h<Vec8f, 10>(0, xf);
    return to_float16(yf);
}
static inline Vec8h sincospi(Vec8h * cosret, Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f cf;  // cos return
    Vec8f yf = sincos_h<Vec8f, 11>(&cf, xf);
    if (cosret) *cosret = to_float16(cf);
    return to_float16(yf);
}
static inline Vec8h tanpi(Vec8h const x) {
    Vec8f xf = to_float(x);
    Vec8f yf = sincos_h<Vec8f, 12>(0, xf);
    return to_float16(yf);
} 

#if MAX_VECTOR_SIZE >= 512

static inline Vec16h exp(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = exp_h<Vec16f, 0, 0>(xf);
    return to_float16(yf);
}

static inline Vec16h exp2(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = exp_h<Vec16f, 0, 2>(xf);
    return to_float16(yf);
}

static inline Vec16h exp10(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = exp_h<Vec16f, 0, 10>(xf);
    return to_float16(yf);
}

static inline Vec16h expm1(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = exp_h<Vec16f, 1, 0>(xf);
    return to_float16(yf);
}

static inline Vec16h sin(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = sincos_h<Vec16f, 1>(0, xf);
    return to_float16(yf);
}
static inline Vec16h cos(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = sincos_h<Vec16f, 2>(0, xf);
    return to_float16(yf);
}
static inline Vec16h sincos(Vec16h * cosret, Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f cf;  // cos return
    Vec16f yf = sincos_h<Vec16f, 3>(&cf, xf);
    if (cosret) *cosret = to_float16(cf);
    return to_float16(yf);
}
static inline Vec16h tan(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = sincos_h<Vec16f, 4>(0, xf);
    return to_float16(yf);
} 

static inline Vec16h sinpi(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = sincos_h<Vec16f, 9>(0, xf);
    return to_float16(yf);
}
static inline Vec16h cospi(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = sincos_h<Vec16f, 10>(0, xf);
    return to_float16(yf);
}
static inline Vec16h sincospi(Vec16h * cosret, Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f cf;  // cos return
    Vec16f yf = sincos_h<Vec16f, 11>(&cf, xf);
    if (cosret) *cosret = to_float16(cf);
    return to_float16(yf);
}
static inline Vec16h tanpi(Vec16h const x) {
    Vec16f xf = to_float(x);
    Vec16f yf = sincos_h<Vec16f, 12>(0, xf);
    return to_float16(yf);
} 

#endif  // MAX_VECTOR_SIZE >= 256 

#if MAX_VECTOR_SIZE >= 512

static inline Vec32h exp(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = exp_h<Vec16f, 0, 0>(xf_lo);
    Vec16f yf_hi = exp_h<Vec16f, 0, 0>(xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}

static inline Vec32h exp2(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = exp_h<Vec16f, 0, 2>(xf_lo);
    Vec16f yf_hi = exp_h<Vec16f, 0, 2>(xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}

static inline Vec32h exp10(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = exp_h<Vec16f, 0, 10>(xf_lo);
    Vec16f yf_hi = exp_h<Vec16f, 0, 10>(xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}

static inline Vec32h expm1(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = exp_h<Vec16f, 1, 0>(xf_lo);
    Vec16f yf_hi = exp_h<Vec16f, 1, 0>(xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}

static inline Vec32h sin(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = sincos_h<Vec16f, 1>(0, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 1>(0, xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}
static inline Vec32h cos(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = sincos_h<Vec16f, 2>(0, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 2>(0, xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}
static inline Vec32h sincos(Vec32h * cosret, Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f cf_lo, cf_hi;
    Vec16f yf_lo = sincos_h<Vec16f, 3>(&cf_lo, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 3>(&cf_hi, xf_hi);
    if (cosret) * cosret = Vec32h(to_float16(cf_lo), to_float16(cf_hi));
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}
static inline Vec32h tan(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = sincos_h<Vec16f, 4>(0, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 4>(0, xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}

static inline Vec32h sinpi(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = sincos_h<Vec16f, 9>(0, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 9>(0, xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}
static inline Vec32h cospi(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = sincos_h<Vec16f, 10>(0, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 10>(0, xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}
static inline Vec32h sincospi(Vec32h * cosret, Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f cf_lo, cf_hi;
    Vec16f yf_lo = sincos_h<Vec16f, 11>(&cf_lo, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 11>(&cf_hi, xf_hi);
    if (cosret) * cosret = Vec32h(to_float16(cf_lo), to_float16(cf_hi));
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
}
static inline Vec32h tanpi(Vec32h const x) {
    Vec16f xf_lo = to_float(x.get_low());
    Vec16f xf_hi = to_float(x.get_high());
    Vec16f yf_lo = sincos_h<Vec16f, 12>(0, xf_lo);
    Vec16f yf_hi = sincos_h<Vec16f, 12>(0, xf_hi);
    return Vec32h(to_float16(yf_lo), to_float16(yf_hi));
} 

#endif  // MAX_VECTOR_SIZE >= 512

#ifdef VCL_NAMESPACE
}
#endif

#endif // VECTORFP16_H
