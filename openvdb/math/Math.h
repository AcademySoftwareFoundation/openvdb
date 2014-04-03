///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file Math.h
/// @brief General-purpose arithmetic and comparison routines, most of which
/// accept arbitrary value types (or at least arbitrary numeric value types)

#ifndef OPENVDB_MATH_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_HAS_BEEN_INCLUDED

#include <assert.h>
#include <algorithm> // for std::max()
#include <cmath>     // for floor(), ceil() and sqrt()
#include <math.h>    // for pow(), fabs() etc
#include <cstdlib>   // for srand(), abs(int)
#include <limits>    // for std::numeric_limits<Type>::max()
#include <string>
#include <boost/numeric/conversion/conversion_traits.hpp>
#include <boost/math/special_functions/cbrt.hpp>
#include <boost/random/mersenne_twister.hpp> // for boost::random::mt19937
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/version.hpp> // for BOOST_VERSION
#include <openvdb/Platform.h>
#include <openvdb/version.h>


// Compile pragmas

// Intel(r) compiler fires remark #1572: floating-point equality and inequality
// comparisons are unrealiable when == or != is used with floating point operands.
#if defined(__INTEL_COMPILER)
    #define OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN \
        _Pragma("warning (push)")    \
        _Pragma("warning (disable:1572)")
    #define OPENVDB_NO_FP_EQUALITY_WARNING_END \
        _Pragma("warning (pop)")
#else
    // For GCC, #pragma GCC diagnostic ignored "-Wfloat-equal"
    // isn't working until gcc 4.2+,
    // Trying
    // #pragma GCC system_header
    // creates other problems, most notably "warning: will never be executed"
    // in from templates, unsure of how to work around.
    // If necessary, could use integer based comparisons for equality
    #define OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    #define OPENVDB_NO_FP_EQUALITY_WARNING_END
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief Return the value of type T that corresponds to zero.
/// @note A zeroVal<T>() specialization must be defined for each @c ValueType T
/// that cannot be constructed using the form @c T(0).  For example, @c std::string(0)
/// treats 0 as @c NULL and throws a @c std::logic_error.
template<typename T> inline T zeroVal() { return T(0); }
/// Return the @c std::string value that corresponds to zero.
template<> inline std::string zeroVal<std::string>() { return ""; }
/// Return the @c bool value that corresponds to zero.
template<> inline bool zeroVal<bool>() { return false; }

/// @todo These won't be needed if we eliminate StringGrids.
//@{
/// @brief Needed to support the <tt>(zeroVal<ValueType>() + val)</tt> idiom
/// when @c ValueType is @c std::string
inline std::string operator+(const std::string& s, bool) { return s; }
inline std::string operator+(const std::string& s, int) { return s; }
inline std::string operator+(const std::string& s, float) { return s; }
inline std::string operator+(const std::string& s, double) { return s; }
//@}


namespace math {

/// @brief Return the unary negation of the given value.
/// @note A negative<T>() specialization must be defined for each ValueType T
/// for which unary negation is not defined.
template<typename T> inline T negative(const T& val) { return T(-val); }
/// Return the negation of the given boolean.
template<> inline bool negative(const bool& val) { return !val; }
/// Return the "negation" of the given string.
template<> inline std::string negative(const std::string& val) { return val; }


//@{
/// Tolerance for floating-point comparison
template<typename T> struct Tolerance { static T value() { return zeroVal<T>(); } };
template<> struct Tolerance<float>    { static float value() { return 1e-8f; } };
template<> struct Tolerance<double>   { static double value() { return 1e-15; } };
//@}

//@{
/// Delta for small floating-point offsets
template<typename T> struct Delta { static T value() { return zeroVal<T>(); } };
template<> struct Delta<float>    { static float value() { return  1e-5f; } };
template<> struct Delta<double>   { static double value() { return 1e-9; } };
//@}


// ==========> Random Values <==================

/// @brief Simple generator of random numbers over the range [0, 1)
/// @details Thread-safe as long as each thread has its own Rand01 instance
template<typename FloatType = double, typename EngineType = boost::mt19937>
class Rand01
{
private:
    EngineType mEngine;
    boost::uniform_01<FloatType> mRand;

public:
    typedef FloatType ValueType;

    /// @brief Initialize the generator.
    /// @param seed  seed value for the random number generator
    Rand01(unsigned int seed): mEngine(static_cast<typename EngineType::result_type>(seed)) {}

    /// Return a uniformly distributed random number in the range [0, 1).
    FloatType operator()() { return mRand(mEngine); }
};

typedef Rand01<double, boost::mt19937> Random01;


/// @brief Simple random integer generator
/// @details Thread-safe as long as each thread has its own RandInt instance
template<typename EngineType = boost::mt19937>
class RandInt
{
private:
#if BOOST_VERSION >= 104700
    typedef boost::random::uniform_int_distribution<int> Distr;
#else
    typedef boost::uniform_int<int> Distr;
#endif
    EngineType mEngine;
    Distr mRand;

public:
    /// @brief Initialize the generator.
    /// @param seed       seed value for the random number generator
    /// @param imin,imax  generate integers that are uniformly distributed over [imin, imax]
    RandInt(unsigned int seed, int imin, int imax):
        mEngine(static_cast<typename EngineType::result_type>(seed)),
        mRand(std::min(imin, imax), std::max(imin, imax))
    {}

    /// Change the range over which integers are distributed to [imin, imax].
    void setRange(int imin, int imax) { mRand = Distr(std::min(imin, imax), std::max(imin, imax)); }

    /// Return a randomly-generated integer in the current range.
    int operator()() { return mRand(mEngine); }
    /// @brief Return a randomly-generated integer in the new range [imin, imax],
    /// without changing the current range.
    int operator()(int imin, int imax)
    {
        const int lo = std::min(imin, imax), hi = std::max(imin, imax);
#if BOOST_VERSION >= 104700
        return mRand(mEngine, Distr::param_type(lo, hi));
#else
        return Distr(lo, hi)(mEngine);
#endif
    }
};

typedef RandInt<boost::mt19937> RandomInt;


// ==========> Clamp <==================

/// Return @a x clamped to [@a min, @a max]
template<typename Type>
inline Type
Clamp(Type x, Type min, Type max)
{
    assert(min<max);
    return x > min ? x < max ? x : max : min;
}


/// Return @a x clamped to [0, 1]
template<typename Type>
inline Type
Clamp01(Type x) { return x > Type(0) ? x < Type(1) ? x : Type(1) : Type(0); }


/// Return @c true if @a x is outside [0,1]
template<typename Type>
inline bool
ClampTest01(Type &x)
{
    if (x >= Type(0) && x <= Type(1)) return false;
    x = x < Type(0) ? Type(0) : Type(1);
    return true;
}


/// @brief Return 0 if @a x < @a min, 1 if @a x > @a max or else @f$(3-2t)t^2@f$,
/// where @f$t = (x-min)/(max-min)@f$.
template<typename Type>
inline Type
SmoothUnitStep(Type x, Type min, Type max)
{
    assert(min < max);
    const Type t = (x-min)/(max-min);
    return t > 0 ? t < 1 ? (3-2*t)*t*t : Type(1) : Type(0);
}


// ==========> Absolute Value <==================


//@{
/// Return the absolute value of the given quantity.
inline int32_t Abs(int32_t i) { return abs(i); }
inline int64_t Abs(int64_t i)
{
#ifdef _MSC_VER
    return (i < int64_t(0) ? -i : i);
#else
    return labs(i);
#endif
}
inline float Abs(float x) { return fabs(x); }
inline double Abs(double x) { return fabs(x); }
inline long double Abs(long double x) { return fabsl(x); }
inline uint32_t Abs(uint32_t i) { return i; }
inline uint64_t Abs(uint64_t i) { return i; }
// On OSX size_t and uint64_t are different types
#if defined(__APPLE__) || defined(MACOSX)
inline size_t Abs(size_t i) { return i; }
#endif
//@}


////////////////////////////////////////


// ==========> Value Comparison <==================


/// Return @c true if @a x is exactly equal to zero.
template<typename Type>
inline bool
isZero(const Type& x)
{
    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    return x == zeroVal<Type>();
    OPENVDB_NO_FP_EQUALITY_WARNING_END
}


/// @brief Return @c true if @a x is equal to zero to within
/// the default floating-point comparison tolerance.
template<typename Type>
inline bool
isApproxZero(const Type& x)
{
    const Type tolerance = Type(zeroVal<Type>() + Tolerance<Type>::value());
    return x < tolerance && x > -tolerance;
}

/// Return @c true if @a x is equal to zero to within the given tolerance.
template<typename Type>
inline bool
isApproxZero(const Type& x, const Type& tolerance)
{
    return x < tolerance && x > -tolerance;
}


/// Return @c true if @a x is less than zero.
template<typename Type>
inline bool
isNegative(const Type& x) { return x < zeroVal<Type>(); }

/// Return @c false, since @c bool values are never less than zero.
template<> inline bool isNegative<bool>(const bool&) { return false; }


/// @brief Return @c true if @a a is equal to @a b to within
/// the default floating-point comparison tolerance.
template<typename Type>
inline bool
isApproxEqual(const Type& a, const Type& b)
{
    const Type tolerance = Type(zeroVal<Type>() + Tolerance<Type>::value());
    return !(Abs(a - b) > tolerance);
}


/// Return @c true if @a a is equal to @a b to within the given tolerance.
template<typename Type>
inline bool
isApproxEqual(const Type& a, const Type& b, const Type& tolerance)
{
    return !(Abs(a - b) > tolerance);
}

#define OPENVDB_EXACT_IS_APPROX_EQUAL(T) \
    template<> inline bool isApproxEqual<T>(const T& a, const T& b) { return a == b; } \
    template<> inline bool isApproxEqual<T>(const T& a, const T& b, const T&) { return a == b; } \
    /**/

OPENVDB_EXACT_IS_APPROX_EQUAL(bool)
OPENVDB_EXACT_IS_APPROX_EQUAL(std::string)


/// @brief Return @c true if @a a is larger than @a b to within
/// the given tolerance, i.e., if @a b - @a a < @a tolerance.
template<typename Type>
inline bool
isApproxLarger(const Type& a, const Type& b, const Type& tolerance)
{
    return (b - a < tolerance);
}


/// @brief Return @c true if @a a is exactly equal to @a b.
template<typename T0, typename T1>
inline bool
isExactlyEqual(const T0& a, const T1& b)
{
    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    return a == b;
    OPENVDB_NO_FP_EQUALITY_WARNING_END
}


template<typename Type>
inline bool
isRelOrApproxEqual(const Type& a, const Type& b, const Type& absTol, const Type& relTol)
{
    // First check to see if we are inside the absolute tolerance
    // Necessary for numbers close to 0
    if (!(Abs(a - b) > absTol)) return true;

    // Next check to see if we are inside the relative tolerance
    // to handle large numbers that aren't within the abs tolerance
    // but could be the closest floating point representation
    double relError;
    if (Abs(b) > Abs(a)) {
        relError = Abs((a - b) / b);
    } else {
        relError = Abs((a - b) / a);
    }
    return (relError <= relTol);
}

template<>
inline bool
isRelOrApproxEqual(const bool& a, const bool& b, const bool&, const bool&)
{
    return (a == b);
}


// Avoid strict aliasing issues by using type punning
// http://cellperformance.beyond3d.com/articles/2006/06/understanding-strict-aliasing.html
// Using "casting through a union(2)"
inline int32_t
floatToInt32(const float aFloatValue)
{
    union FloatOrInt32 { float floatValue; int32_t int32Value; };
    const FloatOrInt32* foi = reinterpret_cast<const FloatOrInt32*>(&aFloatValue);
    return foi->int32Value;
}


inline int64_t
doubleToInt64(const double aDoubleValue)
{
    union DoubleOrInt64 { double doubleValue; int64_t int64Value; };
    const DoubleOrInt64* dol = reinterpret_cast<const DoubleOrInt64*>(&aDoubleValue);
    return dol->int64Value;
}


// aUnitsInLastPlace is the allowed difference between the least significant digits
// of the numbers' floating point representation
// Please read refernce paper before trying to use isUlpsEqual
// http://www.cygnus-software.com/papers/comparingFloats/comparingFloats.htm
inline bool
isUlpsEqual(const double aLeft, const double aRight, const int64_t aUnitsInLastPlace)
{
    int64_t longLeft = doubleToInt64(aLeft);
    // Because of 2's complement, must restore lexicographical order
    if (longLeft < 0) {
        longLeft = INT64_C(0x8000000000000000) - longLeft;
    }

    int64_t longRight = doubleToInt64(aRight);
    // Because of 2's complement, must restore lexicographical order
    if (longRight < 0) {
        longRight = INT64_C(0x8000000000000000) - longRight;
    }

    int64_t difference = labs(longLeft - longRight);
    return (difference <= aUnitsInLastPlace);
}

inline bool
isUlpsEqual(const float aLeft, const float aRight, const int32_t aUnitsInLastPlace)
{
    int32_t intLeft = floatToInt32(aLeft);
    // Because of 2's complement, must restore lexicographical order
    if (intLeft < 0) {
        intLeft = 0x80000000 - intLeft;
    }

    int32_t intRight = floatToInt32(aRight);
    // Because of 2's complement, must restore lexicographical order
    if (intRight < 0) {
        intRight = 0x80000000 - intRight;
    }

    int32_t difference = abs(intLeft - intRight);
    return (difference <= aUnitsInLastPlace);
}


////////////////////////////////////////


// ==========> Pow <==================

/// Return @f$ x^2 @f$.
template<typename Type>
inline Type Pow2(Type x) { return x*x; }

/// Return @f$ x^3 @f$.
template<typename Type>
inline Type Pow3(Type x) { return x*x*x; }

/// Return @f$ x^4 @f$.
template<typename Type>
inline Type Pow4(Type x) { return Pow2(Pow2(x)); }

/// Return @f$ x^n @f$.
template<typename Type>
Type
Pow(Type x, int n)
{
    Type ans = 1;
    if (n < 0) {
        n = -n;
        x = Type(1)/x;
    }
    while (n--) ans *= x;
    return ans;
}

//@{
/// Return @f$ b^e @f$.
inline float
Pow(float b, float e)
{
    assert( b >= 0.0f && "Pow(float,float): base is negative" );
    return powf(b,e);
}

inline double
Pow(double b, double e)
{
    assert( b >= 0.0 && "Pow(double,double): base is negative" );
    return pow(b,e);
}
//@}


// ==========> Max <==================

/// Return the maximum of two values
template<typename Type>
inline const Type&
Max(const Type& a, const Type& b)
{
    return std::max(a,b) ;
}

/// Return the maximum of three values
template<typename Type>
inline const Type&
Max(const Type& a, const Type& b, const Type& c)
{
    return std::max( std::max(a,b), c ) ;
}

/// Return the maximum of four values
template<typename Type>
inline const Type&
Max(const Type& a, const Type& b, const Type& c, const Type& d)
{
    return std::max(std::max(a,b), std::max(c,d));
}

/// Return the maximum of five values
template<typename Type>
inline const Type&
Max(const Type& a, const Type& b, const Type& c, const Type& d, const Type& e)
{
    return std::max(std::max(a,b), Max(c,d,e));
}

/// Return the maximum of six values
template<typename Type>
inline const Type&
Max(const Type& a, const Type& b, const Type& c, const Type& d, const Type& e, const Type& f)
{
    return std::max(Max(a,b,c), Max(d,e,f));
}

/// Return the maximum of seven values
template<typename Type>
inline const Type&
Max(const Type& a, const Type& b, const Type& c, const Type& d,
    const Type& e, const Type& f, const Type& g)
{
    return std::max(Max(a,b,c,d), Max(e,f,g));
}

/// Return the maximum of eight values
template<typename Type>
inline const Type&
Max(const Type& a, const Type& b, const Type& c, const Type& d,
    const Type& e, const Type& f, const Type& g, const Type& h)
{
    return std::max(Max(a,b,c,d), Max(e,f,g,h));
}


// ==========> Min <==================

/// Return the minimum of two values
template<typename Type>
inline const Type&
Min(const Type& a, const Type& b) { return std::min(a, b); }

/// Return the minimum of three values
template<typename Type>
inline const Type&
Min(const Type& a, const Type& b, const Type& c) { return std::min(std::min(a, b), c); }

/// Return the minimum of four values
template<typename Type>
inline const Type&
Min(const Type& a, const Type& b, const Type& c, const Type& d)
{
    return std::min(std::min(a, b), std::min(c, d));
}

/// Return the minimum of five values
template<typename Type>
inline const Type&
Min(const Type& a, const Type& b, const Type& c, const Type& d, const Type& e)
{
    return std::min(std::min(a,b), Min(c,d,e));
}

/// Return the minimum of six values
template<typename Type>
inline const Type&
Min(const Type& a, const Type& b, const Type& c, const Type& d, const Type& e, const Type& f)
{
    return std::min(Min(a,b,c), Min(d,e,f));
}

/// Return the minimum of seven values
template<typename Type>
inline const Type&
Min(const Type& a, const Type& b, const Type& c, const Type& d,
    const Type& e, const Type& f, const Type& g)
{
    return std::min(Min(a,b,c,d), Min(e,f,g));
}

/// Return the minimum of eight values
template<typename Type>
inline const Type&
Min(const Type& a, const Type& b, const Type& c, const Type& d,
    const Type& e, const Type& f, const Type& g, const Type& h)
{
    return std::min(Min(a,b,c,d), Min(e,f,g,h));
}


// ============> Exp <==================

/// Return @f$ e^x @f$.
template<typename Type>
inline Type Exp(const Type& x) { return std::exp(x); }


////////////////////////////////////////


/// Return the sign of the given value as an integer (either -1, 0 or 1).
template <typename Type>
inline int Sign(const Type &x) { return (zeroVal<Type>() < x) - (x < zeroVal<Type>()); }


/// @brief Return @c true if @a a and @a b have different signs.
/// @note Zero is considered a positive number.
template <typename Type>
inline bool
SignChange(const Type& a, const Type& b)
{
    return ( (a<zeroVal<Type>()) ^ (b<zeroVal<Type>()) );
}


/// @brief Return @c true if the interval [@a a, @a b] includes zero,
/// i.e., if either @a a or @a b is zero or if they have different signs.
template <typename Type>
inline bool
ZeroCrossing(const Type& a, const Type& b)
{
    return a * b <= zeroVal<Type>();
}


//@{
/// Return the square root of a floating-point value.
inline float Sqrt(float x) { return sqrtf(x); }
inline double Sqrt(double x) { return sqrt(x); }
inline long double Sqrt(long double x) { return sqrtl(x); }
//@}


//@{
/// Return the cube root of a floating-point value.
inline float Cbrt(float x) { return boost::math::cbrt(x); }
inline double Cbrt(double x) { return boost::math::cbrt(x); }
inline long double Cbrt(long double x) { return boost::math::cbrt(x); }
//@}


//@{
/// Return the remainder of @a x / @a y.
inline int Mod(int x, int y) { return (x % y); }
inline float Mod(float x, float y) { return fmodf(x,y); }
inline double Mod(double x, double y) { return fmod(x,y); }
inline long double Mod(long double x, long double y) { return fmodl(x,y); }
template<typename Type> inline Type Remainder(Type x, Type y) { return Mod(x,y); }
//@}


//@{
/// Return @a x rounded up to the nearest integer.
inline float RoundUp(float x) { return ceilf(x); }
inline double RoundUp(double x) { return ceil(x); }
inline long double RoundUp(long double x) { return ceill(x); }
//@}
/// Return @a x rounded up to the nearest multiple of @a base.
template<typename Type>
inline Type
RoundUp(Type x, Type base)
{
    Type remainder = Remainder(x, base);
    return remainder ? x-remainder+base : x;
}


//@{
/// Return @a x rounded down to the nearest integer.
inline float RoundDown(float x) { return floorf(x); }
inline double RoundDown(double x) { return floor(x); }
inline long double RoundDown(long double x) { return floorl(x); }
template <typename Type> inline Type Round(Type x) { return RoundDown(x+0.5); }
//@}
/// Return @a x rounded down to the nearest multiple of @a base.
template<typename Type>
inline Type
RoundDown(Type x, Type base)
{
    Type remainder = Remainder(x, base);
    return remainder ? x-remainder : x;
}


/// Return the integer part of @a x.
template<typename Type>
inline Type
IntegerPart(Type x)
{
    return (x > 0 ? RoundDown(x) : RoundUp(x));
}

/// Return the fractional part of @a x.
template<typename Type>
inline Type FractionalPart(Type x) { return Mod(x,Type(1)); }


//@{
/// Return the floor of @a x.
inline int Floor(float x) { return (int)RoundDown(x); }
inline int Floor(double x) { return (int)RoundDown(x); }
inline int Floor(long double x) { return (int)RoundDown(x); }
//@}


//@{
/// Return the ceiling of @a x.
inline int Ceil(float x) { return (int)RoundUp(x); }
inline int Ceil(double x) { return (int)RoundUp(x); }
inline int Ceil(long double x) { return (int)RoundUp(x); }
//@}


/// Return @a x if it is greater in magnitude than @a delta.  Otherwise, return zero.
template<typename Type>
inline Type Chop(Type x, Type delta) { return (Abs(x) < delta ? zeroVal<Type>() : x); }


/// Return @a x truncated to the given number of decimal digits.
template<typename Type>
inline Type
Truncate(Type x, unsigned int digits)
{
    Type tenth = Pow(10,digits);
    return RoundDown(x*tenth+0.5)/tenth;
}


////////////////////////////////////////


/// Return the inverse of @a x.
template<typename Type>
inline Type
Inv(Type x)
{
    assert(x);
    return Type(1)/x;
}


enum Axis {
    X_AXIS = 0,
    Y_AXIS = 1,
    Z_AXIS = 2
};

// enum values are consistent with their historical mx analogs.
enum RotationOrder {
    XYZ_ROTATION = 0,
    XZY_ROTATION,
    YXZ_ROTATION,
    YZX_ROTATION,
    ZXY_ROTATION,
    ZYX_ROTATION,
    XZX_ROTATION,
    ZXZ_ROTATION
};


template <typename S, typename T>
struct promote {
    typedef typename boost::numeric::conversion_traits<S, T>::supertype type;
};


/// @brief Return the index [0,1,2] of the smallest value in a 3D vector.
/// @note This methods assumes operator[] exists and avoids branching.
/// @details If two components of the input vector are equal and smaller then the
/// third component, the largest index of the two is always returned.
/// If all three vector components are equal the largest index, i.e. 2, is
/// returned. In other words the return value corresponds to the largest index
/// of the of the smallest vector components.
template<typename Vec3T>
size_t
MinIndex(const Vec3T& v)
{
    static const size_t hashTable[8] = { 2, 1, 9, 1, 2, 9, 0, 0 };//9 is a dummy value
    const size_t hashKey =
        ((v[0] < v[1]) << 2) + ((v[0] < v[2]) << 1) + (v[1] < v[2]);// ?*4+?*2+?*1
    return hashTable[hashKey];
}


/// @brief Return the index [0,1,2] of the largest value in a 3D vector.
/// @note This methods assumes operator[] exists and avoids branching.
/// @details If two components of the input vector are equal and larger then the
/// third component, the largest index of the two is always returned.
/// If all three vector components are equal the largest index, i.e. 2, is
/// returned. In other words the return value corresponds to the largest index
/// of the of the largest vector components.
template<typename Vec3T>
size_t
MaxIndex(const Vec3T& v)
{
    static const size_t hashTable[8] = { 2, 1, 9, 1, 2, 9, 0, 0 };//9 is a dummy value
    const size_t hashKey =
        ((v[0] > v[1]) << 2) + ((v[0] > v[2]) << 1) + (v[1] > v[2]);// ?*4+?*2+?*1
    return hashTable[hashKey];
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_MATH_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
