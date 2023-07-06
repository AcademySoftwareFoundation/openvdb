//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Primary original authors:
//     Florian Kainz <kainz@ilm.com>
//     Rod Bogart <rgb@ilm.com>
//

#ifndef OPENVDB_MATH_HALF_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_HALF_HAS_BEEN_INCLUDED

/// @file half.h
/// The half type is a 16-bit floating number, compatible with the
/// IEEE 754-2008 binary16 type.
///
/// **Representation of a 32-bit float:**
///
/// We assume that a float, f, is an IEEE 754 single-precision
/// floating point number, whose bits are arranged as follows:
///
///     31 (msb)
///     |
///     | 30     23
///     | |      |
///     | |      | 22                    0 (lsb)
///     | |      | |                     |
///     X XXXXXXXX XXXXXXXXXXXXXXXXXXXXXXX
///
///     s e        m
///
/// S is the sign-bit, e is the exponent and m is the significand.
///
/// If e is between 1 and 254, f is a normalized number:
///
///             s    e-127
///     f = (-1)  * 2      * 1.m
///
/// If e is 0, and m is not zero, f is a denormalized number:
///
///             s    -126
///     f = (-1)  * 2      * 0.m
///
/// If e and m are both zero, f is zero:
///
///     f = 0.0
///
/// If e is 255, f is an "infinity" or "not a number" (NAN),
/// depending on whether m is zero or not.
///
/// Examples:
///
///     0 00000000 00000000000000000000000 = 0.0
///     0 01111110 00000000000000000000000 = 0.5
///     0 01111111 00000000000000000000000 = 1.0
///     0 10000000 00000000000000000000000 = 2.0
///     0 10000000 10000000000000000000000 = 3.0
///     1 10000101 11110000010000000000000 = -124.0625
///     0 11111111 00000000000000000000000 = +infinity
///     1 11111111 00000000000000000000000 = -infinity
///     0 11111111 10000000000000000000000 = NAN
///     1 11111111 11111111111111111111111 = NAN
///
/// **Representation of a 16-bit half:**
///
/// Here is the bit-layout for a half number, h:
///
///     15 (msb)
///     |
///     | 14  10
///     | |   |
///     | |   | 9        0 (lsb)
///     | |   | |        |
///     X XXXXX XXXXXXXXXX
///
///     s e     m
///
/// S is the sign-bit, e is the exponent and m is the significand.
///
/// If e is between 1 and 30, h is a normalized number:
///
///             s    e-15
///     h = (-1)  * 2     * 1.m
///
/// If e is 0, and m is not zero, h is a denormalized number:
///
///             S    -14
///     h = (-1)  * 2     * 0.m
///
/// If e and m are both zero, h is zero:
///
///     h = 0.0
///
/// If e is 31, h is an "infinity" or "not a number" (NAN),
/// depending on whether m is zero or not.
///
/// Examples:
///
///     0 00000 0000000000 = 0.0
///     0 01110 0000000000 = 0.5
///     0 01111 0000000000 = 1.0
///     0 10000 0000000000 = 2.0
///     0 10000 1000000000 = 3.0
///     1 10101 1111000001 = -124.0625
///     0 11111 0000000000 = +infinity
///     1 11111 0000000000 = -infinity
///     0 11111 1000000000 = NAN
///     1 11111 1111111111 = NAN
///
/// **Conversion via Lookup Table:**
///
/// Converting from half to float is performed by default using a
/// lookup table. There are only 65,536 different half numbers; each
/// of these numbers has been converted and stored in a table pointed
/// to by the ``imath_half_to_float_table`` pointer.
///
/// Prior to Imath v3.1, conversion from float to half was
/// accomplished with the help of an exponent look table, but this is
/// now replaced with explicit bit shifting.
///
/// **Conversion via Hardware:**
///
/// For Imath v3.1, the conversion routines have been extended to use
/// F16C SSE instructions whenever present and enabled by compiler
/// flags.
///
/// **Conversion via Bit-Shifting**
///
/// If F16C SSE instructions are not available, conversion can be
/// accomplished by a bit-shifting algorithm. For half-to-float
/// conversion, this is generally slower than the lookup table, but it
/// may be preferable when memory limits preclude storing of the
/// 65,536-entry lookup table.
///
/// The lookup table symbol is included in the compilation even if
/// ``IMATH_HALF_USE_LOOKUP_TABLE`` is false, because application code
/// using the exported ``half.h`` may choose to enable the use of the table.
///
/// An implementation can eliminate the table from compilation by
/// defining the ``IMATH_HALF_NO_LOOKUP_TABLE`` preprocessor symbol.
/// Simply add:
///
///     #define IMATH_HALF_NO_LOOKUP_TABLE
///
/// before including ``half.h``, or define the symbol on the compile
/// command line.
///
/// Furthermore, an implementation wishing to receive ``FE_OVERFLOW``
/// and ``FE_UNDERFLOW`` floating point exceptions when converting
/// float to half by the bit-shift algorithm can define the
/// preprocessor symbol ``IMATH_HALF_ENABLE_FP_EXCEPTIONS`` prior to
/// including ``half.h``:
///
///     #define IMATH_HALF_ENABLE_FP_EXCEPTIONS
///
/// **Conversion Performance Comparison:**
///
/// Testing on a Core i9, the timings are approximately:
///
/// half to float
/// - table: 0.71 ns / call
/// - no table: 1.06 ns / call
/// - f16c: 0.45 ns / call
///
/// float-to-half:
/// - original: 5.2 ns / call
/// - no exp table + opt: 1.27 ns / call
/// - f16c: 0.45 ns / call
///
/// **Note:** the timing above depends on the distribution of the
/// floats in question.
///

#include <openvdb/Platform.h>
#include <openvdb/version.h>
#include <iostream>

OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN

#if defined(_WIN32)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#elif defined(__F16C__)
#include <immintrin.h>
#endif

#include <stdint.h>
#include <stdio.h>

#ifdef IMATH_HALF_ENABLE_FP_EXCEPTIONS
#include <fenv.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {
namespace internal {

//-------------------------------------------------------------------------
// Limits
//
// Visual C++ will complain if VDBB_HALF_DENORM_MIN, VDB_HALF_NRM_MIN etc. are not float
// constants, but at least one other compiler (gcc 2.96) produces incorrect
// results if they are.
//-------------------------------------------------------------------------

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER

/// Smallest positive denormalized half
#define VDBB_HALF_DENORM_MIN 5.96046448e-08f
/// Smallest positive normalized half
#define VDB_HALF_NRM_MIN 6.10351562e-05f
/// Smallest positive normalized half
#define VDB_HALF_MIN 6.10351562e-05f
/// Largest positive half
#define VDB_HALF_MAX 65504.0f
/// Smallest positive e for which ``half(1.0 + e) != half(1.0)``
#define VDB_HALF_EPSILON 0.00097656f
#else
/// Smallest positive denormalized half
#define VDBB_HALF_DENORM_MIN 5.96046448e-08
/// Smallest positive normalized half
#define VDB_HALF_NRM_MIN 6.10351562e-05
/// Smallest positive normalized half
#define VDB_HALF_MIN 6.10351562e-05f
/// Largest positive half
#define VDB_HALF_MAX 65504.0
/// Smallest positive e for which ``half(1.0 + e) != half(1.0)``
#define VDB_HALF_EPSILON 0.00097656
#endif

/// Number of digits in mantissa (significand + hidden leading 1)
#define VDB_HALF_MANT_DIG 11
/// Number of base 10 digits that can be represented without change:
///
/// ``floor( (HALF_MANT_DIG - 1) * log10(2) ) => 3.01... -> 3``
#define VDB_HALF_DIG 3
/// Number of base-10 digits that are necessary to uniquely represent
/// all distinct values:
///
/// ``ceil(HALF_MANT_DIG * log10(2) + 1) => 4.31... -> 5``
#define VDB_HALF_DECIMAL_DIG 5
/// Base of the exponent
#define VDB_HALF_RADIX 2
/// Minimum negative integer such that ``HALF_RADIX`` raised to the power
/// of one less than that integer is a normalized half
#define VDBB_HALF_DENORM_MIN_EXP -13
/// Maximum positive integer such that ``HALF_RADIX`` raised to the power
/// of one less than that integer is a normalized half
#define VDB_HALF_MAX_EXP 16
/// Minimum positive integer such that 10 raised to that power is a
/// normalized half
#define VDBB_HALF_DENORM_MIN_10_EXP -4
/// Maximum positive integer such that 10 raised to that power is a
/// normalized half
#define VDB_HALF_MAX_10_EXP 4

/// a type for both C-only programs and C++ to use the same utilities
typedef union imath_half_uif
{
    uint32_t i;
    float f;
} imath_half_uif_t;

/// a type for both C-only programs and C++ to use the same utilities
typedef uint16_t imath_half_bits_t;

#if !defined(__cplusplus) && !defined(__CUDACC__)
/// if we're in a C-only context, alias the half bits type to half
typedef imath_half_bits_t half;
#endif

#if !defined(IMATH_HALF_NO_LOOKUP_TABLE)
#if defined(__cplusplus)
extern "C"
#else
extern
#endif
OPENVDB_API const imath_half_uif_t* imath_half_to_float_table;
#endif

///
/// Convert half to float
///

static inline float
imath_half_to_float (imath_half_bits_t h)
{
#if defined(__F16C__)
    // NB: The intel implementation does seem to treat NaN slightly
    // different than the original toFloat table does (i.e. where the
    // 1 bits are, meaning the signalling or not bits). This seems
    // benign, given that the original library didn't really deal with
    // signalling vs non-signalling NaNs
#    ifdef _MSC_VER
    /* msvc does not seem to have cvtsh_ss :( */
    return _mm_cvtss_f32 (_mm_cvtph_ps (_mm_set1_epi16 (h)));
#    else
    return _cvtsh_ss (h);
#    endif
#elif defined(IMATH_HALF_USE_LOOKUP_TABLE) && !defined(IMATH_HALF_NO_LOOKUP_TABLE)
    return imath_half_to_float_table[h].f;
#else
    imath_half_uif_t v;
    // this code would be clearer, although it does appear to be faster
    // (1.06 vs 1.08 ns/call) to avoid the constants and just do 4
    // shifts.
    //
    uint32_t hexpmant = ( (uint32_t)(h) << 17 ) >> 4;
    v.i = ((uint32_t)(h >> 15)) << 31;

    // the likely really does help if most of your numbers are "normal" half numbers
    if (OPENVDB_LIKELY ((hexpmant >= 0x00800000)))
    {
        v.i |= hexpmant;
        // either we are a normal number, in which case add in the bias difference
        // otherwise make sure all exponent bits are set
        if (OPENVDB_LIKELY ((hexpmant < 0x0f800000)))
            v.i += 0x38000000;
        else
            v.i |= 0x7f800000;
    }
    else if (hexpmant != 0)
    {
        // exponent is 0 because we're denormal, don't have to extract
        // the mantissa, can just use as is
        //
        //
        // other compilers may provide count-leading-zeros primitives,
        // but we need the community to inform us of the variants
        uint32_t lc;
#    if defined(_MSC_VER) && (_M_IX86 || _M_X64)
        lc = __lzcnt (hexpmant);
#    elif defined(__GNUC__) || defined(__clang__)
        lc = (uint32_t) __builtin_clz (hexpmant);
#    else
        lc = 0;
        while (0 == ((hexpmant << lc) & 0x80000000))
            ++lc;
#    endif
        lc -= 8;
        // so nominally we want to remove that extra bit we shifted
        // up, but we are going to add that bit back in, then subtract
        // from it with the 0x38800000 - (lc << 23)....
        //
        // by combining, this allows us to skip the & operation (and
        // remove a constant)
        //
        // hexpmant &= ~0x00800000;
        v.i |= 0x38800000;
        // lc is now x, where the desired exponent is then
        // -14 - lc
        // + 127 -> new exponent
        v.i |= (hexpmant << lc);
        v.i -= (lc << 23);
    }
    return v.f;
#endif
}

///
/// Convert half to float
///
/// Note: This only supports the "round to even" rounding mode, which
/// was the only mode supported by the original OpenEXR library
///

static inline imath_half_bits_t
imath_float_to_half (float f)
{
#if defined(__F16C__)
#    ifdef _MSC_VER
    // msvc does not seem to have cvtsh_ss :(
    return _mm_extract_epi16 (
        _mm_cvtps_ph (_mm_set_ss (f), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
        0);
#    else
    // preserve the fixed rounding mode to nearest
    return _cvtss_sh (f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#    endif
#else
    imath_half_uif_t v;
    imath_half_bits_t ret;
    uint32_t e, m, ui, r, shift;

    v.f = f;

    ui  = (v.i & ~0x80000000);
    ret = ((v.i >> 16) & 0x8000);

    // exponent large enough to result in a normal number, round and return
    if (ui >= 0x38800000)
    {
        // inf or nan
        if (OPENVDB_UNLIKELY (ui >= 0x7f800000))
        {
            ret |= 0x7c00;
            if (ui == 0x7f800000)
                return ret;
            m = (ui & 0x7fffff) >> 13;
            // make sure we have at least one bit after shift to preserve nan-ness
            return ret | (uint16_t)m | (uint16_t)(m == 0);
        }

        // too large, round to infinity
        if (OPENVDB_UNLIKELY (ui > 0x477fefff))
        {
#    ifdef IMATH_HALF_ENABLE_FP_EXCEPTIONS
            feraiseexcept (FE_OVERFLOW);
#    endif
            return ret | 0x7c00;
        }

        ui -= 0x38000000;
        ui = ((ui + 0x00000fff + ((ui >> 13) & 1)) >> 13);
        return ret | (uint16_t)ui;
    }

    // zero or flush to 0
    if (ui < 0x33000001)
    {
#    ifdef IMATH_HALF_ENABLE_FP_EXCEPTIONS
        if (ui == 0)
            return ret;
        feraiseexcept (FE_UNDERFLOW);
#    endif
        return ret;
    }

    // produce a denormalized half
    e     = (ui >> 23);
    shift = 0x7e - e;
    m     = 0x800000 | (ui & 0x7fffff);
    r     = m << (32 - shift);
    ret |= (m >> shift);
    if (r > 0x80000000 || (r == 0x80000000 && (ret & 0x1) != 0))
        ++ret;
    return ret;
#endif
}

////////////////////////////////////////

///
///
/// class half -- 16-bit floating point number
///
/// Type half can represent positive and negative numbers whose
/// magnitude is between roughly 6.1e-5 and 6.5e+4 with a relative
/// error of 9.8e-4; numbers smaller than 6.1e-5 can be represented
/// with an absolute error of 6.0e-8.  All integers from -2048 to
/// +2048 can be represented exactly.
///
/// Type half behaves (almost) like the built-in C++ floating point
/// types.  In arithmetic expressions, half, float and double can be
/// mixed freely.  Here are a few examples:
///
///     half a (3.5);
///     float b (a + sqrt (a));
///     a += b;
///     b += a;
///     b = a + 7;
///
/// Conversions from half to float are lossless; all half numbers
/// are exactly representable as floats.
///
/// Conversions from float to half may not preserve a float's value
/// exactly.  If a float is not representable as a half, then the
/// float value is rounded to the nearest representable half.  If a
/// float value is exactly in the middle between the two closest
/// representable half values, then the float value is rounded to
/// the closest half whose least significant bit is zero.
///
/// Overflows during float-to-half conversions cause arithmetic
/// exceptions.  An overflow occurs when the float value to be
/// converted is too large to be represented as a half, or if the
/// float value is an infinity or a NAN.
///
/// The implementation of type half makes the following assumptions
/// about the implementation of the built-in C++ types:
///
/// * float is an IEEE 754 single-precision number
/// * sizeof (float) == 4
/// * sizeof (unsigned int) == sizeof (float)
/// * alignof (unsigned int) == alignof (float)
/// * sizeof (uint16_t) == 2
///

class OPENVDB_API half
{
  public:
    /// A special tag that lets us initialize a half from the raw bits.
    enum OPENVDB_API FromBitsTag
    {
        FromBits
    };

    /// @{
    /// @name Constructors

    /// Default construction provides no initialization (hence it is
    /// not constexpr).
    half() noexcept = default;

    /// Construct from float
    half (float f) noexcept;

    /// Construct from bit-vector
    constexpr half (FromBitsTag, uint16_t bits) noexcept;

    /// Copy constructor
    constexpr half (const half&) noexcept = default;

    /// Move constructor
    constexpr half (half&&) noexcept = default;

    /// Destructor
    ~half() noexcept = default;

    /// @}

    /// Conversion to float
    operator float() const noexcept;

    /// @{
    /// @name Basic Algebra

    /// Unary minus
    constexpr half operator-() const noexcept;

    /// Assignment
    half& operator= (const half& h) noexcept = default;

    /// Move assignment
    half& operator= (half&& h) noexcept = default;

    /// Assignment from float
    half& operator= (float f) noexcept;

    /// Addition assignment
    half& operator+= (half h) noexcept;

    /// Addition assignment from float
    half& operator+= (float f) noexcept;

    /// Subtraction assignment
    half& operator-= (half h) noexcept;

    /// Subtraction assignment from float
    half& operator-= (float f) noexcept;

    /// Multiplication assignment
    half& operator*= (half h) noexcept;

    /// Multiplication assignment from float
    half& operator*= (float f) noexcept;

    /// Division assignment
    half& operator/= (half h) noexcept;

    /// Division assignment from float
    half& operator/= (float f) noexcept;

    /// @}

    /// Round to n-bit precision (n should be between 0 and 10).
    /// After rounding, the significand's 10-n least significant
    /// bits will be zero.
    constexpr half round (unsigned int n) const noexcept;

    /// @{
    /// @name Classification

    /// Return true if a normalized number, a denormalized number, or
    /// zero.
    constexpr bool isFinite() const noexcept;

    /// Return true if a normalized number.
    constexpr bool isNormalized() const noexcept;

    /// Return true if a denormalized number.
    constexpr bool isDenormalized() const noexcept;

    /// Return true if zero.
    constexpr bool isZero() const noexcept;

    /// Return true if NAN.
    constexpr bool isNan() const noexcept;

    /// Return true if a positive or a negative infinity
    constexpr bool isInfinity() const noexcept;

    /// Return true if the sign bit is set (negative)
    constexpr bool isNegative() const noexcept;

    /// @}

    /// @{
    /// @name Special values

    /// Return +infinity
    static constexpr half posInf() noexcept;

    /// Return -infinity
    static constexpr half negInf() noexcept;

    /// Returns a NAN with the bit pattern 0111111111111111
    static constexpr half qNan() noexcept;

    /// Return a NAN with the bit pattern 0111110111111111
    static constexpr half sNan() noexcept;

    /// @}

    /// @{
    /// @name Access to the internal representation

    /// Return the bit pattern
    constexpr uint16_t bits() const noexcept;

    /// Set the bit pattern
    constexpr void setBits (uint16_t bits) noexcept;

    /// @}

  public:
    static_assert (sizeof (float) == sizeof (uint32_t),
                   "Assumption about the size of floats correct");
    using uif = imath_half_uif;

  private:

    constexpr uint16_t mantissa() const noexcept;
    constexpr uint16_t exponent() const noexcept;

    uint16_t _h;
};

//----------------------------
// Half-from-float constructor
//----------------------------

inline half::half (float f) noexcept
    : _h (imath_float_to_half (f))
{
}

//------------------------------------------
// Half from raw bits constructor
//------------------------------------------

inline constexpr half::half (FromBitsTag, uint16_t bits) noexcept : _h (bits)
{}

//-------------------------
// Half-to-float conversion
//-------------------------

inline half::operator float() const noexcept
{
    return imath_half_to_float (_h);
}

//-------------------------
// Round to n-bit precision
//-------------------------

inline constexpr half
half::round (unsigned int n) const noexcept
{
    //
    // Parameter check.
    //

    if (n >= 10)
        return *this;

    //
    // Disassemble h into the sign, s,
    // and the combined exponent and significand, e.
    //

    uint16_t s = _h & 0x8000;
    uint16_t e = _h & 0x7fff;

    //
    // Round the exponent and significand to the nearest value
    // where ones occur only in the (10-n) most significant bits.
    // Note that the exponent adjusts automatically if rounding
    // up causes the significand to overflow.
    //

    e >>= 9 - n;
    e += e & 1;
    e <<= 9 - n;

    //
    // Check for exponent overflow.
    //

    if (e >= 0x7c00)
    {
        //
        // Overflow occurred -- truncate instead of rounding.
        //

        e = _h;
        e >>= 10 - n;
        e <<= 10 - n;
    }

    //
    // Put the original sign bit back.
    //

    half h (FromBits, s | e);

    return h;
}

//-----------------------
// Other inline functions
//-----------------------

inline constexpr half
half::operator-() const noexcept
{
    return half (FromBits, bits() ^ 0x8000);
}

inline half&
half::operator= (float f) noexcept
{
    *this = half (f);
    return *this;
}

inline half&
half::operator+= (half h) noexcept
{
    *this = half (float (*this) + float (h));
    return *this;
}

inline half&
half::operator+= (float f) noexcept
{
    *this = half (float (*this) + f);
    return *this;
}

inline half&
half::operator-= (half h) noexcept
{
    *this = half (float (*this) - float (h));
    return *this;
}

inline half&
half::operator-= (float f) noexcept
{
    *this = half (float (*this) - f);
    return *this;
}

inline half&
half::operator*= (half h) noexcept
{
    *this = half (float (*this) * float (h));
    return *this;
}

inline half&
half::operator*= (float f) noexcept
{
    *this = half (float (*this) * f);
    return *this;
}

inline half&
half::operator/= (half h) noexcept
{
    *this = half (float (*this) / float (h));
    return *this;
}

inline half&
half::operator/= (float f) noexcept
{
    *this = half (float (*this) / f);
    return *this;
}

inline constexpr uint16_t
half::mantissa() const noexcept
{
    return _h & 0x3ff;
}

inline constexpr uint16_t
half::exponent() const noexcept
{
    return (_h >> 10) & 0x001f;
}

inline constexpr bool
half::isFinite() const noexcept
{
    return exponent() < 31;
}

inline constexpr bool
half::isNormalized() const noexcept
{
    return exponent() > 0 && exponent() < 31;
}

inline constexpr bool
half::isDenormalized() const noexcept
{
    return exponent() == 0 && mantissa() != 0;
}

inline constexpr bool
half::isZero() const noexcept
{
    return (_h & 0x7fff) == 0;
}

inline constexpr bool
half::isNan() const noexcept
{
    return exponent() == 31 && mantissa() != 0;
}

inline constexpr bool
half::isInfinity() const noexcept
{
    return exponent() == 31 && mantissa() == 0;
}

inline constexpr bool
half::isNegative() const noexcept
{
    return (_h & 0x8000) != 0;
}

inline constexpr half
half::posInf() noexcept
{
    return half (FromBits, 0x7c00);
}

inline constexpr half
half::negInf() noexcept
{
    return half (FromBits, 0xfc00);
}

inline constexpr half
half::qNan() noexcept
{
    return half (FromBits, 0x7fff);
}

inline constexpr half
half::sNan() noexcept
{
    return half (FromBits, 0x7dff);
}

inline constexpr uint16_t
half::bits() const noexcept
{
    return _h;
}

inline constexpr void
half::setBits (uint16_t bits) noexcept
{
    _h = bits;
}

//----------
// Debugging
//----------

OPENVDB_API void printBits (std::ostream& os, half h);
OPENVDB_API void printBits (std::ostream& os, float f);
OPENVDB_API void printBits (char c[19], half h);
OPENVDB_API void printBits (char c[35], float f);

/// Output h to os, formatted as a float
OPENVDB_API std::ostream& operator<< (std::ostream& os, half h);

/// Input h from is
OPENVDB_API std::istream& operator>> (std::istream& is, half& h);

} // namespace internal
} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include <limits>

namespace std
{

template <> class numeric_limits<openvdb::math::internal::half>
{
public:
    static const bool is_specialized = true;

    static constexpr openvdb::math::internal::half min () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x0400); /*VDB_HALF_MIN*/
    }
    static constexpr openvdb::math::internal::half max () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x7bff); /*HALF_MAX*/
    }
    static constexpr openvdb::math::internal::half lowest ()
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0xfbff); /* -HALF_MAX */
    }

    static constexpr int  digits       = VDB_HALF_MANT_DIG;
    static constexpr int  digits10     = VDB_HALF_DIG;
    static constexpr int  max_digits10 = VDB_HALF_DECIMAL_DIG;
    static constexpr bool is_signed    = true;
    static constexpr bool is_integer   = false;
    static constexpr bool is_exact     = false;
    static constexpr int  radix        = VDB_HALF_RADIX;
    static constexpr openvdb::math::internal::half epsilon () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x1400); /*HALF_EPSILON*/
    }
    static constexpr openvdb::math::internal::half round_error () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x3800); /*0.5*/
    }

    static constexpr int min_exponent   = VDBB_HALF_DENORM_MIN_EXP;
    static constexpr int min_exponent10 = VDBB_HALF_DENORM_MIN_10_EXP;
    static constexpr int max_exponent   = VDB_HALF_MAX_EXP;
    static constexpr int max_exponent10 = VDB_HALF_MAX_10_EXP;

    static constexpr bool               has_infinity      = true;
    static constexpr bool               has_quiet_NaN     = true;
    static constexpr bool               has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm        = denorm_present;
    static constexpr bool               has_denorm_loss   = false;
    static constexpr openvdb::math::internal::half               infinity () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x7c00); /*half::posInf()*/
    }
    static constexpr openvdb::math::internal::half quiet_NaN () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x7fff); /*half::qNan()*/
    }
    static constexpr openvdb::math::internal::half signaling_NaN () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x7dff); /*half::sNan()*/
    }
    static constexpr openvdb::math::internal::half denorm_min () noexcept
    {
        return openvdb::math::internal::half (openvdb::math::internal::half::FromBits, 0x0001); /*VDBB_HALF_DENORM_MIN*/
    }

    static constexpr bool is_iec559  = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo  = false;

    static constexpr bool              traps           = true;
    static constexpr bool              tinyness_before = false;
    static constexpr float_round_style round_style     = round_to_nearest;
};

} // namespace std

OPENVDB_NO_TYPE_CONVERSION_WARNING_END

#endif // OPENVDB_MATH_HALF_HAS_BEEN_INCLUDED
