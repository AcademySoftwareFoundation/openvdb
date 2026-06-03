// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   Math.h

    \author Ken Museth

    \date  January 8, 2020

    \brief Math functions and classes

*/

#ifndef NANOVDB_MATH_MATH_H_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_MATH_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Util.h>// for __hostdev__ and lots of other utility functions

#include <type_traits>// for std::is_floating_point (matches long double in addition to float / double)

#if defined(__CUDA_ARCH__)
#include <cuda/std/limits>// for ::cuda::std::numeric_limits
#endif

namespace nanovdb {// =================================================================

namespace math {// =============================================================

// ----------------------------> Various math functions <-------------------------------------

//@{
/// @brief Pi constant taken from Boost to match old behaviour
template<typename T>
inline __hostdev__ constexpr T pi() noexcept
{
    return 3.141592653589793238462643383279502884e+00;
}
template<>
inline __hostdev__ constexpr float pi() noexcept
{
    return 3.141592653589793238462643383279502884e+00F;
}
template<>
inline __hostdev__ constexpr double pi() noexcept
{
    return 3.141592653589793238462643383279502884e+00;
}
template<>
inline __hostdev__ constexpr long double pi() noexcept
{
    return 3.141592653589793238462643383279502884e+00L;
}
//@}

//@{
/// @brief Per-type tolerance used by approximate floating-point comparisons.
/// @details Returned by @c Tolerance<T>::value() and consumed by @c isApproxZero.
/// Specialized for @c float, @c double, and @c long double. Instantiating on
/// any other @c T is an "incomplete type" compile error by design — callers
/// must opt in by adding a specialization.
template<typename T>
struct Tolerance;
template<>
struct Tolerance<float>
{
    __hostdev__ [[nodiscard]] static constexpr float value() noexcept { return 1e-8f; }
};
template<>
struct Tolerance<double>
{
    __hostdev__ [[nodiscard]] static constexpr double value() noexcept { return 1e-15; }
};
template<>
struct Tolerance<long double>
{
    __hostdev__ [[nodiscard]] static constexpr long double value() noexcept { return 1e-18L; }
};
//@}

//@{
/// @brief Per-type small floating-point offset, useful for nudging boundaries
/// off-edge in geometric tests.
/// @details Returned by @c Delta<T>::value(). Specialized for @c float,
/// @c double, and @c long double; instantiating on any other @c T is a
/// compile error by design.
template<typename T>
struct Delta;
template<>
struct Delta<float>
{
    __hostdev__ [[nodiscard]] static constexpr float value() noexcept { return 1e-5f; }
};
template<>
struct Delta<double>
{
    __hostdev__ [[nodiscard]] static constexpr double value() noexcept { return 1e-9; }
};
template<>
struct Delta<long double>
{
    __hostdev__ [[nodiscard]] static constexpr long double value() noexcept { return 1e-12L; }
};
//@}

//@{
/// @brief Per-type largest representable value (mirrors @c std::numeric_limits::max).
/// @details Returned by @c Maximum<T>::value(). The host fallback (`#else`
/// branch) forwards to @c std::numeric_limits so any @c T with a
/// specialization there works; the CUDA-device branch uses
/// @c ::cuda::std::numeric_limits; the HIP branch hard-codes a handful of
/// common types because @c std::numeric_limits is unavailable on the device
/// in some HIP toolchains.
template<typename T>
struct Maximum;
#if defined(__CUDA_ARCH__)
template<typename T>
struct Maximum
{
    __hostdev__ [[nodiscard]] static constexpr T value() noexcept { return ::cuda::std::numeric_limits<T>::max(); }
};
#elif defined(__HIP__)
template<>
struct Maximum<int>
{
    __hostdev__ [[nodiscard]] static constexpr int value() noexcept { return 2147483647; }
};
template<>
struct Maximum<uint32_t>
{
    __hostdev__ [[nodiscard]] static constexpr uint32_t value() noexcept { return 4294967295u; }
};
template<>
struct Maximum<float>
{
    __hostdev__ [[nodiscard]] static constexpr float value() noexcept { return 1e+38f; }
};
template<>
struct Maximum<double>
{
    __hostdev__ [[nodiscard]] static constexpr double value() noexcept { return 1e+308; }
};
#else
template<typename T>
struct Maximum
{
    [[nodiscard]] static constexpr T value() noexcept { return std::numeric_limits<T>::max(); }
};
#endif
//@}

/// @brief Return @c true if @c |x| is below @c Tolerance<Type>::value().
template<typename Type>
__hostdev__ [[nodiscard]] inline constexpr bool isApproxZero(const Type& x) noexcept
{
    return !(x > Tolerance<Type>::value()) && !(x < -Tolerance<Type>::value());
}

/// @brief Same-type minimum (the primary template).
template<typename Type>
__hostdev__ [[nodiscard]] inline constexpr Type Min(Type a, Type b) noexcept
{
    return (a < b) ? a : b;
}
// Mixed integer-type tiebreakers: the templated `Min<Type>` requires both
// args to deduce to the same `Type`, so calls like `Min(uint32_t, size_t)`
// would otherwise fall through to the float/double overloads with equal
// implicit-conversion cost — ambiguous. These integer overloads win.
__hostdev__ [[nodiscard]] inline constexpr int32_t Min(int32_t a, int32_t b) noexcept
{
    return a < b ? a : b;
}
__hostdev__ [[nodiscard]] inline constexpr uint32_t Min(uint32_t a, uint32_t b) noexcept
{
    return a < b ? a : b;
}
__hostdev__ [[nodiscard]] inline float Min(float a, float b) noexcept
{
    return fminf(a, b);
}
__hostdev__ [[nodiscard]] inline double Min(double a, double b) noexcept
{
    return fmin(a, b);
}
/// @brief Same-type maximum (the primary template).
template<typename Type>
__hostdev__ [[nodiscard]] inline constexpr Type Max(Type a, Type b) noexcept
{
    return (a > b) ? a : b;
}
// See Min above: these integer overloads disambiguate mixed-int calls.
__hostdev__ [[nodiscard]] inline constexpr int32_t Max(int32_t a, int32_t b) noexcept
{
    return a > b ? a : b;
}
__hostdev__ [[nodiscard]] inline constexpr uint32_t Max(uint32_t a, uint32_t b) noexcept
{
    return a > b ? a : b;
}
__hostdev__ [[nodiscard]] inline float Max(float a, float b) noexcept
{
    return fmaxf(a, b);
}
__hostdev__ [[nodiscard]] inline double Max(double a, double b) noexcept
{
    return fmax(a, b);
}
//@{
/// @brief Clamp @c x to the closed interval [@c a, @c b].
__hostdev__ [[nodiscard]] inline float Clamp(float x, float a, float b) noexcept
{
    return Max(Min(x, b), a);
}
__hostdev__ [[nodiscard]] inline double Clamp(double x, double a, double b) noexcept
{
    return Max(Min(x, b), a);
}
//@}

//@{
/// @brief Fractional part of @c x, i.e. @c x @c - @c floor(x). Always non-negative.
__hostdev__ [[nodiscard]] inline float Fract(float x) noexcept
{
    return x - floorf(x);
}
__hostdev__ [[nodiscard]] inline double Fract(double x) noexcept
{
    return x - floor(x);
}
//@}

//@{
/// @brief Floor of @c x as a 32-bit signed integer (truncation toward -inf).
__hostdev__ [[nodiscard]] inline int32_t Floor(float x) noexcept
{
    return int32_t(floorf(x));
}
__hostdev__ [[nodiscard]] inline int32_t Floor(double x) noexcept
{
    return int32_t(floor(x));
}
//@}

//@{
/// @brief Ceil of @c x as a 32-bit signed integer (rounding toward +inf).
__hostdev__ [[nodiscard]] inline int32_t Ceil(float x) noexcept
{
    return int32_t(ceilf(x));
}
__hostdev__ [[nodiscard]] inline int32_t Ceil(double x) noexcept
{
    return int32_t(ceil(x));
}
//@}

/// @brief Return @c x * @c x.
template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Pow2(T x) noexcept
{
    return x * x;
}

/// @brief Return @c x * @c x * @c x.
template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Pow3(T x) noexcept
{
    return x * x * x;
}

/// @brief Return @c x raised to the fourth power.
template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Pow4(T x) noexcept
{
    return Pow2(x * x);
}
/// @brief Absolute value (generic primary template, branchless for arithmetic types).
/// @note The @c float / @c double / @c int specializations below are not
template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Abs(T x) noexcept
{
    return x < 0 ? -x : x;
}

template<>
__hostdev__ [[nodiscard]] inline float Abs(float x) noexcept
{
    return fabsf(x);
}

template<>
__hostdev__ [[nodiscard]] inline double Abs(double x) noexcept
{
    return fabs(x);
}

template<>
__hostdev__ [[nodiscard]] inline int Abs(int x) noexcept
{
    return abs(x);
}

/// @brief Round each component of @c xyz to its closest integer coordinate.
/// @details Forward declaration of the primary template — there is no
/// definition here. Callers resolve to one of the @c float / @c double
/// overloads below, both of which use the @c floor(x+0.5) rule
/// (round-half-toward-+inf), so a @c float and @c double input with the
/// same value yield the same integer coordinate.
template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ [[nodiscard]] inline CoordT Round(const Vec3T<RealT>& xyz) noexcept;

/// @brief Round each component to its closest integer (round-half-toward-+inf)
/// using @c floor(x+0.5). Same rule is applied to both single and double
/// precision so float and double inputs yield the same integer coords.
template<typename CoordT, template<typename> class Vec3T>
__hostdev__ [[nodiscard]] inline CoordT Round(const Vec3T<float>& xyz) noexcept
{
    return CoordT(int32_t(floorf(xyz[0] + 0.5f)),
                  int32_t(floorf(xyz[1] + 0.5f)),
                  int32_t(floorf(xyz[2] + 0.5f)));
}

/// @brief Double-precision variant of @c Round — see the @c float overload above.
template<typename CoordT, template<typename> class Vec3T>
__hostdev__ [[nodiscard]] inline CoordT Round(const Vec3T<double>& xyz) noexcept
{
    return CoordT(int32_t(floor(xyz[0] + 0.5)),
                  int32_t(floor(xyz[1] + 0.5)),
                  int32_t(floor(xyz[2] + 0.5)));
}

/// @brief Round each component of @c xyz down (toward -inf) into a @c CoordT.
template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ [[nodiscard]] inline CoordT RoundDown(const Vec3T<RealT>& xyz) noexcept
{
    return CoordT(Floor(xyz[0]), Floor(xyz[1]), Floor(xyz[2]));
}

//@{
/// Return the square root of a floating-point value.
__hostdev__ [[nodiscard]] inline float Sqrt(float x) noexcept
{
    return sqrtf(x);
}
__hostdev__ [[nodiscard]] inline double Sqrt(double x) noexcept
{
    return sqrt(x);
}
//@}

/// Return the sign of the given value as an integer (either -1, 0 or 1).
template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Sign(const T& x) noexcept
{
    return ((T(0) < x) ? T(1) : T(0)) - ((x < T(0)) ? T(1) : T(0));
}

/// @brief Return the component index (0, 1, or 2) of the smallest element of @c v.
/// @note Ties resolve to the lowest index (i.e. x beats y, y beats z).
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr int MinIndex(const Vec3T& v) noexcept
{
#if 0
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] < v[1]) << 2) + ((v[0] < v[2]) << 1) + (v[1] < v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
#else
    if (v[0] < v[1] && v[0] < v[2])
        return 0;
    if (v[1] < v[2])
        return 1;
    else
        return 2;
#endif
}

/// @brief Return the component index (0, 1, or 2) of the largest element of @c v.
/// @note Ties resolve to the lowest index (i.e. x beats y, y beats z).
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr int MaxIndex(const Vec3T& v) noexcept
{
#if 0
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] > v[1]) << 2) + ((v[0] > v[2]) << 1) + (v[1] > v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
#else
    if (v[0] > v[1] && v[0] > v[2])
        return 0;
    if (v[1] > v[2])
        return 1;
    else
        return 2;
#endif
}

/// @brief round up byteSize to the nearest wordSize, e.g. to align to machine word: AlignUp<sizeof(size_t)(n)
///
/// @details both wordSize and byteSize are in byte units
template<uint64_t wordSize>
__hostdev__ [[nodiscard]] inline constexpr uint64_t AlignUp(uint64_t byteCount) noexcept
{
    const uint64_t r = byteCount % wordSize;
    return r ? byteCount - r + wordSize : byteCount;
}

// ------------------------------> Coord <--------------------------------------

// forward declaration so we can define Coord::asVec3s and Coord::asVec3d
template<typename>
class Vec3;

/// @brief Signed (i, j, k) 32-bit integer coordinate class, similar to openvdb::math::Coord
class Coord
{
    int32_t mVec[3]; // private member data - three signed index coordinates
public:
    using ValueType = int32_t;
    using IndexType = uint32_t;

    /// @brief Initialize all coordinates to zero.
    __hostdev__ constexpr Coord() noexcept
        : mVec{0, 0, 0}
    {
    }

    /// @brief Initializes all coordinates to the given signed integer.
    __hostdev__ explicit constexpr Coord(ValueType n) noexcept
        : mVec{n, n, n}
    {
    }

    /// @brief Initializes coordinate to the given signed integers.
    __hostdev__ constexpr Coord(ValueType i, ValueType j, ValueType k) noexcept
        : mVec{i, j, k}
    {
    }

    /// @brief Read three signed integers from @a ptr (no bounds check).
    __hostdev__ constexpr Coord(ValueType* ptr) noexcept
        : mVec{ptr[0], ptr[1], ptr[2]}
    {
    }

    //@{
    /// @brief Named component accessors (x = mVec[0], y = mVec[1], z = mVec[2]).
    __hostdev__ constexpr int32_t x() const noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t y() const noexcept { return mVec[1]; }
    __hostdev__ constexpr int32_t z() const noexcept { return mVec[2]; }

    __hostdev__ constexpr int32_t& x() noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t& y() noexcept { return mVec[1]; }
    __hostdev__ constexpr int32_t& z() noexcept { return mVec[2]; }
    //@}

    /// @brief Largest representable @c Coord (all components = INT32_MAX).
    __hostdev__ [[nodiscard]] static constexpr Coord max() noexcept { return Coord(int32_t((1u << 31) - 1)); }

    /// @brief Smallest representable @c Coord (all components = INT32_MIN).
    __hostdev__ [[nodiscard]] static constexpr Coord min() noexcept { return Coord(-int32_t((1u << 31) - 1) - 1); }

    /// @brief Byte size of a @c Coord (always 12 on a 32-bit @c int32_t platform).
    __hostdev__ [[nodiscard]] static constexpr size_t memUsage() noexcept { return sizeof(Coord); }

    /// @brief Return a const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ constexpr const ValueType& operator[](IndexType i) const noexcept { NANOVDB_ASSERT(i < 3); return mVec[i]; }

    /// @brief Return a non-const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ constexpr ValueType& operator[](IndexType i) noexcept { NANOVDB_ASSERT(i < 3); return mVec[i]; }

    /// @brief Assignment operator that works with openvdb::Coord
    template<typename CoordT>
    __hostdev__ constexpr Coord& operator=(const CoordT& other) noexcept
    {
        static_assert(sizeof(Coord) == sizeof(CoordT), "Mis-matched sizeof");
        mVec[0] = other[0];
        mVec[1] = other[1];
        mVec[2] = other[2];
        return *this;
    }

    /// @brief Return a new instance with coordinates masked by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord operator&(IndexType n) const noexcept { return Coord(mVec[0] & n, mVec[1] & n, mVec[2] & n); }

    /// @brief Return a new instance with coordinates left-shifted by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord operator<<(IndexType n) const noexcept { return Coord(mVec[0] << n, mVec[1] << n, mVec[2] << n); }

    /// @brief Return a new instance with coordinates right-shifted by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord operator>>(IndexType n) const noexcept { return Coord(mVec[0] >> n, mVec[1] >> n, mVec[2] >> n); }

    /// @brief Return true if this Coord is lexicographically less than the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator<(const Coord& rhs) const noexcept
    {
        return mVec[0] < rhs[0] ? true
             : mVec[0] > rhs[0] ? false
             : mVec[1] < rhs[1] ? true
             : mVec[1] > rhs[1] ? false
             : mVec[2] < rhs[2] ? true : false;
    }

    /// @brief Return true if this Coord is lexicographically less or equal to the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator<=(const Coord& rhs) const noexcept
    {
        return mVec[0] < rhs[0] ? true
             : mVec[0] > rhs[0] ? false
             : mVec[1] < rhs[1] ? true
             : mVec[1] > rhs[1] ? false
             : mVec[2] <=rhs[2] ? true : false;
    }

    /// @brief Return true if this Coord is lexicographically greater than the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>(const Coord& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true
             : mVec[1] < rhs[1] ? false
             : mVec[2] > rhs[2] ? true : false;
    }

    /// @brief Return true if this Coord is lexicographically greater or equal to the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>=(const Coord& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true
             : mVec[1] < rhs[1] ? false
             : mVec[2] >=rhs[2] ? true : false;
    }

    /// @brief Return true iff every component matches @a rhs.
    __hostdev__ [[nodiscard]] constexpr bool   operator==(const Coord& rhs) const noexcept { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    /// @brief Return true iff any component differs from @a rhs.
    __hostdev__ [[nodiscard]] constexpr bool   operator!=(const Coord& rhs) const noexcept { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2]; }
    /// @brief In-place component-wise bitwise AND with the given mask.
    __hostdev__ constexpr Coord& operator&=(int n) noexcept
    {
        mVec[0] &= n;
        mVec[1] &= n;
        mVec[2] &= n;
        return *this;
    }
    /// @brief In-place left-shift of every component by @a n bits.
    __hostdev__ constexpr Coord& operator<<=(uint32_t n) noexcept
    {
        mVec[0] <<= n;
        mVec[1] <<= n;
        mVec[2] <<= n;
        return *this;
    }
    /// @brief In-place right-shift of every component by @a n bits.
    __hostdev__ constexpr Coord& operator>>=(uint32_t n) noexcept
    {
        mVec[0] >>= n;
        mVec[1] >>= n;
        mVec[2] >>= n;
        return *this;
    }
    /// @brief Add a scalar to every component in place.
    __hostdev__ constexpr Coord& operator+=(int n) noexcept
    {
        mVec[0] += n;
        mVec[1] += n;
        mVec[2] += n;
        return *this;
    }
    /// @brief Component-wise sum of two coordinates.
    __hostdev__ [[nodiscard]] constexpr Coord  operator+(const Coord& rhs) const noexcept { return Coord(mVec[0] + rhs[0], mVec[1] + rhs[1], mVec[2] + rhs[2]); }
    /// @brief Component-wise difference of two coordinates.
    __hostdev__ [[nodiscard]] constexpr Coord  operator-(const Coord& rhs) const noexcept { return Coord(mVec[0] - rhs[0], mVec[1] - rhs[1], mVec[2] - rhs[2]); }
    /// @brief Component-wise negation.
    __hostdev__ [[nodiscard]] constexpr Coord  operator-() const noexcept { return Coord(-mVec[0], -mVec[1], -mVec[2]); }
    /// @brief In-place component-wise addition with another coordinate.
    __hostdev__ constexpr Coord& operator+=(const Coord& rhs) noexcept
    {
        mVec[0] += rhs[0];
        mVec[1] += rhs[1];
        mVec[2] += rhs[2];
        return *this;
    }
    /// @brief In-place component-wise subtraction with another coordinate.
    __hostdev__ constexpr Coord& operator-=(const Coord& rhs) noexcept
    {
        mVec[0] -= rhs[0];
        mVec[1] -= rhs[1];
        mVec[2] -= rhs[2];
        return *this;
    }

    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ constexpr Coord& minComponent(const Coord& other) noexcept
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        if (other[2] < mVec[2])
            mVec[2] = other[2];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ constexpr Coord& maxComponent(const Coord& other) noexcept
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        if (other[2] > mVec[2])
            mVec[2] = other[2];
        return *this;
    }
#if defined(__CUDACC__) // the following functions only run on the GPU!
    /// @brief Device-only @c atomicMin component-wise.
    __device__ inline Coord& minComponentAtomic(const Coord& other) noexcept
    {
        atomicMin(&mVec[0], other[0]);
        atomicMin(&mVec[1], other[1]);
        atomicMin(&mVec[2], other[2]);
        return *this;
    }
    /// @brief Device-only @c atomicMax component-wise.
    __device__ inline Coord& maxComponentAtomic(const Coord& other) noexcept
    {
        atomicMax(&mVec[0], other[0]);
        atomicMax(&mVec[1], other[1]);
        atomicMax(&mVec[2], other[2]);
        return *this;
    }
#endif

    /// @brief Return a new @c Coord offset component-by-component by @a (dx, dy, dz).
    __hostdev__ [[nodiscard]] constexpr Coord offsetBy(ValueType dx, ValueType dy, ValueType dz) const noexcept
    {
        return Coord(mVec[0] + dx, mVec[1] + dy, mVec[2] + dz);
    }

    /// @brief Return a new @c Coord offset by the same scalar @a n in every component.
    __hostdev__ [[nodiscard]] constexpr Coord offsetBy(ValueType n) const noexcept { return this->offsetBy(n, n, n); }

    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    __hostdev__ [[nodiscard]] static inline constexpr bool lessThan(const Coord& a, const Coord& b) noexcept
    {
        return (a[0] < b[0] || a[1] < b[1] || a[2] < b[2]);
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    template<typename Vec3T>
    __hostdev__ [[nodiscard]] static Coord Floor(const Vec3T& xyz) noexcept { return Coord(math::Floor(xyz[0]), math::Floor(xyz[1]), math::Floor(xyz[2])); }

    /// @brief Return a hash key derived from the existing coordinates.
    /// @details The hash function is originally taken from the SIGGRAPH paper:
    ///          "VDB: High-resolution sparse volumes with dynamic topology"
    ///          and the prime numbers are modified based on the ACM Transactions on Graphics paper:
    ///          "Real-time 3D reconstruction at scale using voxel hashing" (the second number had a typo!)
    template<int Log2N = 3 + 4 + 5>
    __hostdev__ [[nodiscard]] constexpr uint32_t hash() const noexcept { return ((1 << Log2N) - 1) & (mVec[0] * 73856093 ^ mVec[1] * 19349669 ^ mVec[2] * 83492791); }

    /// @brief Return the octant of this Coord
    __hostdev__ [[nodiscard]] constexpr uint8_t octant() const noexcept { return (uint8_t(bool(mVec[0] & (1u << 31)))) |
                                                (uint8_t(bool(mVec[1] & (1u << 31))) << 1) |
                                                (uint8_t(bool(mVec[2] & (1u << 31))) << 2); }

    /// @brief Return a single precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec3<float> asVec3s() const noexcept;

    /// @brief Return a double precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec3<double> asVec3d() const noexcept;

    /// @brief Identity (Coord is already integer); provided so generic code
    /// can call @c .round() uniformly on @c Coord and @c Vec3<T>.
    __hostdev__ [[nodiscard]] inline constexpr Coord round() const noexcept { return *this; }
}; // Coord class


/// @brief Type alias for Coord so we have a consistent naming convention
using Coord3 = Coord;


template <typename T>
class Vec2;

/// @brief Signed (i, j) 32-bit integer coordinate class, similar to openvdb::math::Coord
class Coord2
{
    int32_t mVec[2]; // private member data - three signed index coordinates
public:
    using ValueType = int32_t;
    using IndexType = uint32_t;

    /// @brief Initialize all coordinates to zero.
    __hostdev__ constexpr Coord2() noexcept
        : mVec{0, 0}
    {
    }

    /// @brief Initializes all coordinates to the given signed integer.
    __hostdev__ explicit constexpr Coord2(ValueType n) noexcept
        : mVec{n, n}
    {
    }

    /// @brief Initializes coordinate to the given signed integers.
    __hostdev__ constexpr Coord2(ValueType i, ValueType j) noexcept
        : mVec{i, j}
    {
    }

    /// @brief Read two signed integers from @a ptr (no bounds check).
    __hostdev__ constexpr Coord2(ValueType* ptr) noexcept
        : mVec{ptr[0], ptr[1]}
    {
    }

    //@{
    /// @brief Named component accessors (x = mVec[0], y = mVec[1]).
    __hostdev__ constexpr int32_t x() const noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t y() const noexcept { return mVec[1]; }

    __hostdev__ constexpr int32_t& x() noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t& y() noexcept { return mVec[1]; }
    //@}

    /// @brief Largest representable @c Coord2 (all components = INT32_MAX).
    __hostdev__ [[nodiscard]] static constexpr Coord2 max() noexcept { return Coord2(int32_t((1u << 31) - 1)); }

    /// @brief Smallest representable @c Coord2 (all components = INT32_MIN).
    __hostdev__ [[nodiscard]] static constexpr Coord2 min() noexcept { return Coord2(-int32_t((1u << 31) - 1) - 1); }

    /// @brief Byte size of a @c Coord2 (always 8 on a 32-bit @c int32_t platform).
    __hostdev__ [[nodiscard]] static constexpr size_t memUsage() noexcept { return sizeof(Coord2); }

    /// @brief Return a const reference to the given Coord component.
    /// @warning The argument is assumed to be 0 or 1,
    __hostdev__ constexpr const ValueType& operator[](IndexType i) const noexcept { NANOVDB_ASSERT(i < 2); return mVec[i]; }

    /// @brief Return a non-const reference to the given Coord component.
    /// @warning The argument is assumed to be 0 or 1.
    __hostdev__ constexpr ValueType& operator[](IndexType i) noexcept { NANOVDB_ASSERT(i < 2); return mVec[i]; }

    /// @brief Assignment operator that works with openvdb::Coord
    template<typename CoordT>
    __hostdev__ constexpr Coord2& operator=(const CoordT& other) noexcept
    {
        static_assert(sizeof(Coord2) == sizeof(CoordT), "Mis-matched sizeof");
        mVec[0] = other[0];
        mVec[1] = other[1];
        return *this;
    }

    /// @brief Return a new instance with coordinates masked by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord2 operator&(IndexType n) const noexcept { return Coord2(mVec[0] & n, mVec[1] & n); }

    /// @brief Return a new instance with coordinates left-shifted by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord2 operator<<(IndexType n) const noexcept { return Coord2(mVec[0] << n, mVec[1] << n); }

    /// @brief Return a new instance with coordinates right-shifted by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord2 operator>>(IndexType n) const noexcept { return Coord2(mVec[0] >> n, mVec[1] >> n); }

    /// @brief Return true if this Coord is lexicographically less than the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator<(const Coord2& rhs) const noexcept
    {
        return mVec[0] < rhs[0] ? true
             : mVec[0] > rhs[0] ? false
             : mVec[1] < rhs[1] ? true : false;
    }

    /// @brief Return true if this Coord is lexicographically less or equal to the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator<=(const Coord2& rhs) const noexcept
    {
        return mVec[0] < rhs[0] ? true
             : mVec[0] > rhs[0] ? false
             : mVec[1] <= rhs[1] ? true : false;
    }

    /// @brief Return true if this Coord is lexicographically greater than the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>(const Coord2& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true : false;
    }

    /// @brief Return true if this Coord is lexicographically greater or equal to the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>=(const Coord2& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] >= rhs[1] ? true : false;
    }

    /// @brief Return true iff every component matches @a rhs.
    __hostdev__ [[nodiscard]] constexpr bool   operator==(const Coord2& rhs) const noexcept { return mVec[0] == rhs[0] && mVec[1] == rhs[1]; }
    /// @brief Return true iff any component differs from @a rhs.
    __hostdev__ [[nodiscard]] constexpr bool   operator!=(const Coord2& rhs) const noexcept { return mVec[0] != rhs[0] || mVec[1] != rhs[1]; }
    /// @brief In-place component-wise bitwise AND with the given mask.
    __hostdev__ constexpr Coord2& operator&=(int n) noexcept
    {
        mVec[0] &= n;
        mVec[1] &= n;
        return *this;
    }
    /// @brief In-place left-shift of every component by @a n bits.
    __hostdev__ constexpr Coord2& operator<<=(uint32_t n) noexcept
    {
        mVec[0] <<= n;
        mVec[1] <<= n;
        return *this;
    }
    /// @brief In-place right-shift of every component by @a n bits.
    __hostdev__ constexpr Coord2& operator>>=(uint32_t n) noexcept
    {
        mVec[0] >>= n;
        mVec[1] >>= n;
        return *this;
    }
    /// @brief Add a scalar to every component in place.
    __hostdev__ constexpr Coord2& operator+=(int n) noexcept
    {
        mVec[0] += n;
        mVec[1] += n;
        return *this;
    }
    /// @brief Component-wise sum of two coordinates.
    __hostdev__ [[nodiscard]] constexpr Coord2  operator+(const Coord2& rhs) const noexcept { return Coord2(mVec[0] + rhs[0], mVec[1] + rhs[1]); }
    /// @brief Component-wise difference of two coordinates.
    __hostdev__ [[nodiscard]] constexpr Coord2  operator-(const Coord2& rhs) const noexcept { return Coord2(mVec[0] - rhs[0], mVec[1] - rhs[1]); }
    /// @brief Component-wise negation.
    __hostdev__ [[nodiscard]] constexpr Coord2  operator-() const noexcept { return Coord2(-mVec[0], -mVec[1]); }
    /// @brief In-place component-wise addition with another coordinate.
    __hostdev__ constexpr Coord2& operator+=(const Coord2& rhs) noexcept
    {
        mVec[0] += rhs[0];
        mVec[1] += rhs[1];
        return *this;
    }
    /// @brief In-place component-wise subtraction with another coordinate.
    __hostdev__ constexpr Coord2& operator-=(const Coord2& rhs) noexcept
    {
        mVec[0] -= rhs[0];
        mVec[1] -= rhs[1];
        return *this;
    }

    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ constexpr Coord2& minComponent(const Coord2& other) noexcept
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ constexpr Coord2& maxComponent(const Coord2& other) noexcept
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        return *this;
    }
#if defined(__CUDACC__) // the following functions only run on the GPU!
    /// @brief Device-only @c atomicMin component-wise.
    __device__ inline Coord2& minComponentAtomic(const Coord2& other) noexcept
    {
        atomicMin(&mVec[0], other[0]);
        atomicMin(&mVec[1], other[1]);
        return *this;
    }
    /// @brief Device-only @c atomicMax component-wise.
    __device__ inline Coord2& maxComponentAtomic(const Coord2& other) noexcept
    {
        atomicMax(&mVec[0], other[0]);
        atomicMax(&mVec[1], other[1]);
        return *this;
    }
#endif

    /// @brief Return a new @c Coord2 offset component-by-component by @a (dx, dy).
    __hostdev__ [[nodiscard]] constexpr Coord2 offsetBy(ValueType dx, ValueType dy) const noexcept
    {
        return Coord2(mVec[0] + dx, mVec[1] + dy);
    }

    /// @brief Return a new @c Coord2 offset by the same scalar @a n in both components.
    __hostdev__ [[nodiscard]] constexpr Coord2 offsetBy(ValueType n) const noexcept { return this->offsetBy(n, n); }

    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    __hostdev__ [[nodiscard]] static inline constexpr bool lessThan(const Coord2& a, const Coord2& b) noexcept
    {
        return (a[0] < b[0] || a[1] < b[1]);
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    template<typename Vec2T>
    __hostdev__ [[nodiscard]] static Coord2 Floor(const Vec2T& xy) noexcept { return Coord2(math::Floor(xy[0]), math::Floor(xy[1])); }

    /// @brief Return a single precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec2<float> asVec2s() const noexcept;

    /// @brief Return a double precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec2<double> asVec2d() const noexcept;

    /// @brief Identity (Coord2 is already integer); provided so generic code
    /// can call @c .round() uniformly on @c Coord2 and @c Vec2<T>.
    __hostdev__ [[nodiscard]] inline constexpr Coord2 round() const noexcept { return *this; }
}; // Coord2 class

// ----------------------------> VecBase <-----------------------------------

/// @brief Base class for fixed-size vectors. Provides shared element-wise
/// arithmetic, scalar arithmetic, equality, length/dot reductions,
/// component min/max, and integer-coord rounding. Derived classes provide
/// constructors and any dimension-specific extras (e.g. cross/outer on Vec3).
template<typename T, int N>
class VecBase
{
protected:
    T mVec[N];

public:
    using ValueType = T;
    static constexpr int SIZE = N;

    /// @brief Default-construct (mVec is left uninitialized for trivially-default-constructible @c T).
    VecBase() noexcept = default;

protected:
    /// @brief Variadic component-init ctor — used only by derived ctors
    /// (`Vec2(T x, T y) : Base(x, y) {}`). The member-init list directly
    /// initializes @c mVec, avoiding the default-construct-then-assign that
    /// a body-assignment ctor would impose on a non-fundamental @c T.
    template<typename... Args>
    __hostdev__ explicit constexpr VecBase(Args... args) noexcept : mVec{T(args)...}
    {
        static_assert(sizeof...(Args) == N, "VecBase: wrong number of constructor arguments");
    }

public:
    /// @brief Indexed element access. Asserts 0 <= i < N in debug builds.
    __hostdev__ constexpr const T& operator[](int i) const noexcept { NANOVDB_ASSERT(i >= 0 && i < N); return mVec[i]; }
    __hostdev__ constexpr T&       operator[](int i) noexcept       { NANOVDB_ASSERT(i >= 0 && i < N); return mVec[i]; }

    __hostdev__ constexpr const T& x() const noexcept { return mVec[0]; }
    __hostdev__ constexpr       T& x() noexcept       { return mVec[0]; }
    __hostdev__ constexpr const T& y() const noexcept { static_assert(N >= 2, "VecBase::y() requires N >= 2"); return mVec[1]; }
    __hostdev__ constexpr       T& y() noexcept       { static_assert(N >= 2, "VecBase::y() requires N >= 2"); return mVec[1]; }
    __hostdev__ constexpr const T& z() const noexcept { static_assert(N >= 3, "VecBase::z() requires N >= 3"); return mVec[2]; }
    __hostdev__ constexpr       T& z() noexcept       { static_assert(N >= 3, "VecBase::z() requires N >= 3"); return mVec[2]; }
    __hostdev__ constexpr const T& w() const noexcept { static_assert(N >= 4, "VecBase::w() requires N >= 4"); return mVec[3]; }
    __hostdev__ constexpr       T& w() noexcept       { static_assert(N >= 4, "VecBase::w() requires N >= 4"); return mVec[3]; }

    /// @brief raw pointer to the underlying N-element storage
    __hostdev__ constexpr T*       asPointer() noexcept       { return mVec; }
    __hostdev__ constexpr const T* asPointer() const noexcept { return mVec; }

    // ---- generic element-wise helpers (taking/returning Derived) ----

    /// @brief Return @c *this + @a rhs as a @c Derived.
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived plus(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] + rhs[i];
        return out;
    }
    /// @brief Return @c *this - @a rhs as a @c Derived.
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived minus(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] - rhs[i];
        return out;
    }
    /// @brief Return component-wise @c *this * @a rhs as a @c Derived (Hadamard product).
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived mul(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] * rhs[i];
        return out;
    }
    /// @brief Return component-wise @c *this / @a rhs as a @c Derived.
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived div(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] / rhs[i];
        return out;
    }
    /// @brief Return @c -(*this) as a @c Derived.
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived negate() const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = -mVec[i];
        return out;
    }
    /// @brief Return @c s * (*this) as a @c Derived (scalar broadcast).
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived scale(const T& s) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] * s;
        return out;
    }
    /// @brief Return @c (*this) / @a s element-wise as a @c Derived. Uses per-element
    /// division (correct for integer @c T, unlike multiplying by 1/s).
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived divideBy(const T& s) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] / s;
        return out;
    }
    /// @brief Return a unit-length copy (@c *this divided by @c length()) as a @c Derived.
    /// Const, non-mutating counterpart of the derived @c normalize(). Not @c constexpr —
    /// @c length() calls @c Sqrt.
    template<typename Derived>
    __hostdev__ [[nodiscard]] Derived normalized() const noexcept {
        return this->template divideBy<Derived>(this->length());
    }

    // ---- in-place compound assignment helpers ----

    /// @brief Component-wise add @a rhs into @c *this and return @c *this.
    __hostdev__ constexpr VecBase& addAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] += rhs.mVec[i];
        return *this;
    }
    /// @brief Component-wise subtract @a rhs from @c *this and return @c *this.
    __hostdev__ constexpr VecBase& subAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] -= rhs.mVec[i];
        return *this;
    }
    /// @brief Component-wise multiply @c *this by @a rhs (Hadamard product) and return @c *this.
    __hostdev__ constexpr VecBase& mulAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] *= rhs.mVec[i];
        return *this;
    }
    /// @brief Component-wise divide @c *this by @a rhs and return @c *this.
    __hostdev__ constexpr VecBase& divAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] /= rhs.mVec[i];
        return *this;
    }
    /// @brief Multiply every component by scalar @a s in place; return @c *this.
    __hostdev__ constexpr VecBase& scaleAssign(const T& s) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] *= s;
        return *this;
    }
    /// @brief Divide every component by scalar @a s in place (per-element, integer-safe);
    /// return @c *this.
    __hostdev__ constexpr VecBase& divideAssignScalar(const T& s) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] /= s;
        return *this;
    }

    /// @brief Return @c true iff every component compares equal to @a rhs.
    __hostdev__ [[nodiscard]] constexpr bool equals(const VecBase& rhs) const noexcept {
        for (int i = 0; i < N; ++i) if (mVec[i] != rhs.mVec[i]) return false;
        return true;
    }

    // ---- reductions ----

    /// @brief dot product. @a V must have @c operator[] valid for 0..N-1.
    template<typename V>
    __hostdev__ [[nodiscard]] constexpr T dot(const V& v) const noexcept {
        T s = T(0);
        for (int i = 0; i < N; ++i) s += mVec[i] * v[i];
        return s;
    }
    /// @brief Squared L2 length (sum of squared components). Constexpr, no sqrt.
    __hostdev__ [[nodiscard]] constexpr T lengthSqr() const noexcept {
        T s = T(0);
        for (int i = 0; i < N; ++i) s += mVec[i] * mVec[i];
        return s;
    }
    /// @brief L2 length (Euclidean norm). Not @c constexpr — calls @c Sqrt.
    __hostdev__ [[nodiscard]] T length() const noexcept { return Sqrt(this->lengthSqr()); }

    /// @brief return the smallest of the N components
    __hostdev__ [[nodiscard]] constexpr T smallestComponent() const noexcept {
        T m = mVec[0];
        for (int i = 1; i < N; ++i) if (mVec[i] < m) m = mVec[i];
        return m;
    }
    /// @brief return the largest of the N components
    __hostdev__ [[nodiscard]] constexpr T largestComponent() const noexcept {
        T m = mVec[0];
        for (int i = 1; i < N; ++i) if (mVec[i] > m) m = mVec[i];
        return m;
    }

    // ---- component-wise (mutating) min/max ----

    /// @brief Component-wise take the @c min of @c *this and @a other (in place); return @c *this.
    template<typename V>
    __hostdev__ constexpr VecBase& mergeMin(const V& other) noexcept {
        for (int i = 0; i < N; ++i) if (other[i] < mVec[i]) mVec[i] = other[i];
        return *this;
    }
    /// @brief Component-wise take the @c max of @c *this and @a other (in place); return @c *this.
    template<typename V>
    __hostdev__ constexpr VecBase& mergeMax(const V& other) noexcept {
        for (int i = 0; i < N; ++i) if (other[i] > mVec[i]) mVec[i] = other[i];
        return *this;
    }

    // ---- integer rounding (toward -inf / +inf / nearest) ----
    //

    /// @brief floor-rounded components into the @c Result type (whose @c [i]
    /// must accept int32_t). For integer @c T the value is passed through.
    /// @note Only the integer-@c T specialization is usable as a constant
    /// expression; the floating-point branch calls @c math::Floor, which
    /// isn't constexpr until C++23.
    template<typename Result>
    __hostdev__ [[nodiscard]] constexpr Result floorAs() const noexcept {
        Result r{};
        if constexpr (std::is_floating_point<T>::value) {
            for (int i = 0; i < N; ++i) r[i] = math::Floor(mVec[i]);
        } else {
            for (int i = 0; i < N; ++i) r[i] = static_cast<int32_t>(mVec[i]);
        }
        return r;
    }
    /// @brief ceil-rounded components into the @c Result type (whose @c [i]
    /// must accept int32_t). For integer @c T the value is passed through.
    /// @note Only the integer-@c T specialization is usable as a constant
    /// expression; the floating-point branch calls @c math::Ceil, which
    /// isn't constexpr until C++23.
    template<typename Result>
    __hostdev__ [[nodiscard]] constexpr Result ceilAs() const noexcept {
        Result r{};
        if constexpr (std::is_floating_point<T>::value) {
            for (int i = 0; i < N; ++i) r[i] = math::Ceil(mVec[i]);
        } else {
            for (int i = 0; i < N; ++i) r[i] = static_cast<int32_t>(mVec[i]);
        }
        return r;
    }
    /// @brief nearest-integer rounding using floor(x + 0.5) for floating @c T
    /// (round-half-toward-positive-infinity). Unifies behaviour between
    /// float, double, and long double; pass-through for integer @c T.
    /// @note See @c floorAs — only the integer-@c T branch is constexpr-usable.
    template<typename Result>
    __hostdev__ [[nodiscard]] constexpr Result roundAs() const noexcept {
        Result r{};
        if constexpr (std::is_floating_point<T>::value) {
            const T half = T(0.5);
            for (int i = 0; i < N; ++i) r[i] = math::Floor(mVec[i] + half);
        } else {
            for (int i = 0; i < N; ++i) r[i] = static_cast<int32_t>(mVec[i]);
        }
        return r;
    }
}; // VecBase<T, N>

// ----------------------------> Vec2 <--------------------------------------

/// @brief A simple vector class with two components, similar to openvdb::math::Vec2
///
/// Aligned to 2*alignof(T) so the whole class fits in one SIMD-friendly
/// chunk (e.g. 8 bytes for Vec2<float>, 16 bytes for Vec2<double>)
template<typename T>
class alignas(alignof(T) * 2) Vec2 final : public VecBase<T, 2>
{
    using Base = VecBase<T, 2>;

public:
    using ValueType = T;
    static constexpr int size = 2; // openvdb::math::Tuple-compat alias of SIZE

    /// @brief Default-construct (components are left uninitialized for fundamental @c T).
    Vec2() noexcept = default;
    /// @brief Broadcast: set both components to @a x.
    __hostdev__ explicit constexpr Vec2(T x) noexcept            : Base(x, x) {}
    /// @brief Component-wise construction.
    __hostdev__ constexpr Vec2(T x, T y) noexcept                : Base(x, y) {}

    /// @brief Cross-template converting ctor (e.g. from @c openvdb::Vec2). Implicit
    /// to preserve foreign-type interop; same-class ctor below is @c explicit.
    template<template<class> class Vec2T, class T2>
    __hostdev__ explicit constexpr Vec2(const Vec2T<T2>& v) noexcept : Base(v[0], v[1])
    {
        static_assert(Vec2T<T2>::size == 2, "expected Vec2T::size==2!");
    }
    /// @brief Explicit cross-precision conversion within nanovdb (e.g. @c Vec2d → @c Vec2f).
    template<typename T2>
    __hostdev__ explicit constexpr Vec2(const Vec2<T2>& v) noexcept : Base(v[0], v[1]) {}
    /// @brief Construct from a 2D integer coordinate.
    __hostdev__ explicit constexpr Vec2(const Coord2& ijk) noexcept : Base(ijk[0], ijk[1]) {}

    /// @brief Assign from any 2-component vector type (foreign or nanovdb).
    template<template<class> class Vec2T, class T2>
    __hostdev__ constexpr Vec2& operator=(const Vec2T<T2>& rhs) noexcept {
        static_assert(Vec2T<T2>::size == 2, "expected Vec2T::size==2!");
        this->mVec[0] = rhs[0]; this->mVec[1] = rhs[1];
        return *this;
    }

    // ---- element-wise (Vec & Vec) ----
    /// @brief Component-wise negation.
    __hostdev__ [[nodiscard]] constexpr Vec2  operator-() const noexcept            { return Base::template negate<Vec2>(); }
    /// @brief Component-wise sum.
    __hostdev__ [[nodiscard]] constexpr Vec2  operator+(const Vec2& v) const noexcept { return Base::template plus<Vec2>(v); }
    /// @brief Component-wise difference.
    __hostdev__ [[nodiscard]] constexpr Vec2  operator-(const Vec2& v) const noexcept { return Base::template minus<Vec2>(v); }
    /// @brief Component-wise (Hadamard) product.
    __hostdev__ [[nodiscard]] constexpr Vec2  operator*(const Vec2& v) const noexcept { return Base::template mul<Vec2>(v); }
    /// @brief Component-wise division.
    __hostdev__ [[nodiscard]] constexpr Vec2  operator/(const Vec2& v) const noexcept { return Base::template div<Vec2>(v); }
    /// @brief In-place component-wise addition.
    __hostdev__ constexpr Vec2& operator+=(const Vec2& v) noexcept    { Base::addAssign(v); return *this; }
    /// @brief In-place component-wise subtraction.
    __hostdev__ constexpr Vec2& operator-=(const Vec2& v) noexcept    { Base::subAssign(v); return *this; }

    // ---- mixed Vec2 / Coord2 ----
    /// @brief Add an integer @c Coord2 to this vector (component-wise).
    __hostdev__ [[nodiscard]] constexpr Vec2  operator+(const Coord2& ijk) const noexcept { return Vec2(this->mVec[0] + ijk[0], this->mVec[1] + ijk[1]); }
    /// @brief Subtract an integer @c Coord2 from this vector (component-wise).
    __hostdev__ [[nodiscard]] constexpr Vec2  operator-(const Coord2& ijk) const noexcept { return Vec2(this->mVec[0] - ijk[0], this->mVec[1] - ijk[1]); }
    /// @brief In-place component-wise addition of an integer @c Coord2.
    __hostdev__ constexpr Vec2& operator+=(const Coord2& ijk) noexcept {
        this->mVec[0] += T(ijk[0]); this->mVec[1] += T(ijk[1]);
        return *this;
    }
    /// @brief In-place component-wise subtraction of an integer @c Coord2.
    __hostdev__ constexpr Vec2& operator-=(const Coord2& ijk) noexcept {
        this->mVec[0] -= T(ijk[0]); this->mVec[1] -= T(ijk[1]);
        return *this;
    }

    // ---- scalar ----
    /// @brief Component-wise multiply by scalar @a s.
    __hostdev__ [[nodiscard]] constexpr Vec2  operator*(const T& s) const noexcept  { return Base::template scale<Vec2>(s); }
    /// @brief Component-wise divide by scalar @a s (integer-safe).
    __hostdev__ [[nodiscard]] constexpr Vec2  operator/(const T& s) const noexcept  { return Base::template divideBy<Vec2>(s); }
    /// @brief In-place component-wise multiply by scalar @a s.
    __hostdev__ constexpr Vec2& operator*=(const T& s) noexcept       { Base::scaleAssign(s); return *this; }
    /// @brief In-place component-wise divide by scalar @a s.
    __hostdev__ constexpr Vec2& operator/=(const T& s) noexcept       { Base::divideAssignScalar(s); return *this; }
    /// @brief Normalize in place (divide by @c length()). Not @c constexpr — calls @c std::sqrt.
    __hostdev__ Vec2& normalize() noexcept                  { return (*this) /= this->length(); }
    /// @brief Return a normalized (unit-length) copy; const counterpart of @c normalize().
    __hostdev__ [[nodiscard]] Vec2 normalized() const noexcept {
        return Base::template normalized<Vec2>();
    }

    // ---- equality ----
    /// @brief Component-wise equality.
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Vec2& rhs) const noexcept { return Base::equals(rhs); }
    /// @brief Component-wise inequality.
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Vec2& rhs) const noexcept { return !Base::equals(rhs); }

    // ---- component-wise min/max ----
    /// @brief Take the component-wise minimum of @c *this and @a other in place.
    __hostdev__ constexpr Vec2& minComponent(const Vec2& other) noexcept { Base::mergeMin(other); return *this; }
    /// @brief Take the component-wise maximum of @c *this and @a other in place.
    __hostdev__ constexpr Vec2& maxComponent(const Vec2& other) noexcept { Base::mergeMax(other); return *this; }

    /// @brief Return the smallest vector component
    __hostdev__ [[nodiscard]] constexpr ValueType min() const noexcept { return Base::smallestComponent(); }
    /// @brief Return the largest vector component
    __hostdev__ [[nodiscard]] constexpr ValueType max() const noexcept { return Base::largestComponent(); }

    /// @brief Round each component down (toward negative infinity)
    /// @return integer Coord2
    /// @note Only constexpr for integer @c T (floorAs uses non-constexpr math::Floor for floating point).
    __hostdev__ [[nodiscard]] constexpr Coord2 floor() const noexcept { return Base::template floorAs<Coord2>(); }
    /// @brief Round each component up (toward positive infinity)
    /// @return integer Coord2
    /// @note Only constexpr for integer @c T (ceilAs uses non-constexpr math::Ceil for floating point).
    __hostdev__ [[nodiscard]] constexpr Coord2 ceil()  const noexcept { return Base::template ceilAs<Coord2>(); }
    /// @brief Round each component to its closest integer value
    /// @return integer Coord2
    /// @note Only constexpr for integer @c T (roundAs uses non-constexpr math::Floor for floating point).
    __hostdev__ [[nodiscard]] constexpr Coord2 round() const noexcept { return Base::template roundAs<Coord2>(); }

    // ---- scalar * Vec / scalar / Vec (hidden friends — found only via ADL on Vec2,
    // never participate in unrelated namespace-scope overload sets) ----

    /// @brief Scalar-on-the-left multiplication (hidden friend).
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec2 operator*(T1 scalar, const Vec2& vec) noexcept
    {
        return Vec2(scalar * vec[0], scalar * vec[1]);
    }
    /// @brief Scalar-on-the-left division (hidden friend).
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec2 operator/(T1 scalar, const Vec2& vec) noexcept
    {
        return Vec2(scalar / vec[0], scalar / vec[1]);
    }
}; // Vec2<T>

/// @brief Return a single precision floating-point vector of this coordinate
__hostdev__ [[nodiscard]] inline constexpr Vec2<float> Coord2::asVec2s() const noexcept
{
    return Vec2<float>(float(mVec[0]), float(mVec[1]));
}

/// @brief Return a double precision floating-point vector of this coordinate
__hostdev__ [[nodiscard]] inline constexpr Vec2<double> Coord2::asVec2d() const noexcept
{
    return Vec2<double>(double(mVec[0]), double(mVec[1]));
}


/// @brief Base class for fixed-size matrices. Provides shared element-wise
/// arithmetic, scalar arithmetic, equality, transpose, mat*mat, and mat*vec.
/// Storage is a flat row-major array @c mData[ROWS * COLS]. Derived classes
/// supply constructors plus any dimension-specific extras (e.g. @c inverse()
/// on @c Mat2).
template<typename T, int ROWS, int COLS>
class MatBase {
protected:
    T mData[ROWS * COLS];  // 1D array storage

public:
    using ValueType = T;

    /// @brief Compile-time row count.
    [[nodiscard]] static constexpr int rows() noexcept { return ROWS; }
    /// @brief Compile-time column count.
    [[nodiscard]] static constexpr int cols() noexcept { return COLS; }
    /// @brief Compile-time element count, i.e. @c ROWS * @c COLS.
    [[nodiscard]] static constexpr int size() noexcept { return ROWS * COLS; }

    /// @brief Default-construct (entries are left uninitialized for trivially-default-constructible @c T).
    MatBase() noexcept = default;

    /// @brief Read @c ROWS*COLS elements from @a array in row-major order, converting each
    /// element from @c S to @c T via @c static_cast.
    template<typename S>
    __hostdev__ constexpr MatBase(S* array) noexcept {
        for (int i = 0; i < size(); ++i) {
            mData[i] = static_cast<T>(array[i]);
        }
    }

protected:
    /// @brief Variadic component-init ctor — used only by derived ctors
    /// (`Mat2(T a, T b, T c, T d) : Base(a, b, c, d) {}`). The member-init
    /// list directly initializes @c mData, avoiding the default-construct-
    /// then-assign that a body-assignment ctor would impose on a
    /// non-fundamental @c T.
    template<typename... Args>
    __hostdev__ explicit constexpr MatBase(Args... args) noexcept : mData{T(args)...}
    {
        static_assert(sizeof...(Args) == ROWS * COLS, "MatBase: wrong number of constructor arguments");
    }

public:

    /// @brief 2D row access. Returns a pointer to the start of @a row; the caller
    /// indexes into it with the column. Asserts @c 0 <= @a row < @c ROWS in debug
    /// builds; column index is the caller's responsibility (use @c 0 <= col < @c COLS).
    __hostdev__ constexpr T* operator[](int row) noexcept {
        NANOVDB_ASSERT(row >= 0 && row < ROWS);
        return &mData[row * COLS];
    }
    /// @brief Const overload of @c operator[](int); see non-const variant.
    __hostdev__ constexpr const T* operator[](int row) const noexcept {
        NANOVDB_ASSERT(row >= 0 && row < ROWS);
        return &mData[row * COLS];
    }

    /// @brief return a raw pointer to the underlying 1D storage (row-major)
    __hostdev__ constexpr T*       data() noexcept       { return mData; }
    /// @brief return a const raw pointer to the underlying 1D storage (row-major)
    __hostdev__ constexpr const T* data() const noexcept { return mData; }

    // ---- generic element-wise helpers ----

    /// @brief return @c *this + @a rhs as a @c Derived
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived plus(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < size(); ++i) out.data()[i] = mData[i] + rhs.data()[i];
        return out;
    }

    /// @brief return @c *this - @a rhs as a @c Derived
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived minus(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < size(); ++i) out.data()[i] = mData[i] - rhs.data()[i];
        return out;
    }

    /// @brief return -(*this) as a @c Derived
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived negate() const noexcept {
        Derived out{};
        for (int i = 0; i < size(); ++i) out.data()[i] = -mData[i];
        return out;
    }

    /// @brief return @a s * (*this) as a @c Derived
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived scale(const T& s) const noexcept {
        Derived out{};
        for (int i = 0; i < size(); ++i) out.data()[i] = mData[i] * s;
        return out;
    }

    /// @brief Element-wise add @a rhs into @c *this and return @c *this.
    __hostdev__ constexpr MatBase& addAssign(const MatBase& rhs) noexcept {
        for (int i = 0; i < size(); ++i) mData[i] += rhs.mData[i];
        return *this;
    }
    /// @brief Element-wise subtract @a rhs from @c *this and return @c *this.
    __hostdev__ constexpr MatBase& subAssign(const MatBase& rhs) noexcept {
        for (int i = 0; i < size(); ++i) mData[i] -= rhs.mData[i];
        return *this;
    }
    /// @brief Multiply every element by scalar @a s in place; return @c *this.
    __hostdev__ constexpr MatBase& scaleAssign(const T& s) noexcept {
        for (int i = 0; i < size(); ++i) mData[i] *= s;
        return *this;
    }

    /// @brief return (*this) / @a s element-wise as a @c Derived. Uses
    /// per-element division (correct for integer @c T, unlike multiplying by 1/s).
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived divideBy(const T& s) const noexcept {
        Derived out{};
        for (int i = 0; i < size(); ++i) out.data()[i] = mData[i] / s;
        return out;
    }
    /// @brief Divide every element by scalar @a s in place; return @c *this.
    __hostdev__ constexpr MatBase& divideAssignScalar(const T& s) noexcept {
        for (int i = 0; i < size(); ++i) mData[i] /= s;
        return *this;
    }

    /// @brief Return @c true iff every element compares equal to @a rhs.
    __hostdev__ [[nodiscard]] constexpr bool equals(const MatBase& rhs) const noexcept {
        for (int i = 0; i < size(); ++i) if (mData[i] != rhs.mData[i]) return false;
        return true;
    }

    // ---- generic transpose ----

    /// @brief return the transpose of @c *this as the @c Result type.
    /// @tparam Result a matrix type whose dimensions are (COLS, ROWS)
    template<typename Result>
    __hostdev__ [[nodiscard]] constexpr Result transposeAs() const noexcept {
        static_assert(Result::rows() == COLS && Result::cols() == ROWS,
                      "transposeAs: result dims must be (COLS, ROWS)");
        Result r{};
        for (int i = 0; i < ROWS; ++i)
            for (int j = 0; j < COLS; ++j)
                r[j][i] = mData[i * COLS + j];
        return r;
    }

    // ---- generic matrix * matrix ----

    /// @brief return (*this) * @a rhs as the @c Result type.
    /// @tparam Result a matrix type whose dimensions are (ROWS, Rhs::cols())
    /// @tparam Rhs    a matrix type whose row count equals @c COLS
    template<typename Result, typename Rhs>
    __hostdev__ [[nodiscard]] constexpr Result multiply(const Rhs& rhs) const noexcept {
        static_assert(COLS == Rhs::rows(), "multiply: lhs.cols must equal rhs.rows");
        static_assert(Result::rows() == ROWS && Result::cols() == Rhs::cols(),
                      "multiply: result dims mismatch");
        Result r{};
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < Rhs::cols(); ++j) {
                T sum = T(0);
                for (int k = 0; k < COLS; ++k)
                    sum += mData[i * COLS + k] * rhs[k][j];
                r[i][j] = sum;
            }
        }
        return r;
    }

    // ---- generic matrix * vector ----

    /// @brief return (*this) * @a v as the @c VecResult type.
    /// @tparam VecResult a vector type whose @c SIZE equals @c ROWS
    /// @tparam VecRhs    a vector type whose @c SIZE equals @c COLS
    template<typename VecResult, typename VecRhs>
    __hostdev__ [[nodiscard]] constexpr VecResult multiplyVec(const VecRhs& v) const noexcept {
        static_assert(VecRhs::SIZE == COLS && VecResult::SIZE == ROWS,
                      "multiplyVec: dim mismatch");
        VecResult r{};
        for (int i = 0; i < ROWS; ++i) {
            T sum = T(0);
            for (int k = 0; k < COLS; ++k) sum += mData[i * COLS + k] * v[k];
            r[i] = sum;
        }
        return r;
    }
};

// Forward declarations
template<typename T> class Mat2;
template<typename T> class Mat2x3;
template<typename T> class Mat3x2;
template<typename T> class Mat3;
template<typename T> class Mat4;
template<typename T> class Vec4;



/// @brief 2x2 row-major matrix.
/// @details Aligned to 4*alignof(T) — @c Mat2 stores 4 elements (2x2), which is
/// already a power-of-2 multiple of @c alignof(T), so this is free in size
/// and gives SIMD-friendly placement (16 bytes for @c Mat2<float>, 32 bytes
/// for @c Mat2<double>).
template <typename T>
class alignas(alignof(T) * 4) Mat2 final : public MatBase<T, 2, 2> {
    using Base = MatBase<T, 2, 2>;
public:
    /// @brief Default-construct (entries left uninitialized for fundamental @c T).
    Mat2() noexcept = default;
    /// @brief Constructor given individual array elements, the ordering is in row major form:
    /** @verbatim
        a b
        c d
        @endverbatim */
    __hostdev__ constexpr Mat2(T a, T b, T c, T d) noexcept : Base(a, b, c, d) {}

    /// @brief Constructor given array of elements, the ordering is in row major form
    template<typename Source>
    __hostdev__ constexpr Mat2(Source* array) noexcept : Base(array) {}

    // ---- element-wise ----
    __hostdev__ [[nodiscard]] constexpr Mat2  operator-() const noexcept                  { return this->template negate<Mat2>(); }
    __hostdev__ [[nodiscard]] constexpr Mat2  operator+(const Mat2& m) const noexcept     { return this->template plus<Mat2>(m); }
    __hostdev__ [[nodiscard]] constexpr Mat2  operator-(const Mat2& m) const noexcept     { return this->template minus<Mat2>(m); }
    __hostdev__ constexpr Mat2& operator+=(const Mat2& m) noexcept          { Base::addAssign(m); return *this; }
    __hostdev__ constexpr Mat2& operator-=(const Mat2& m) noexcept          { Base::subAssign(m); return *this; }

    // ---- matrix * matrix / matrix * vector ----
    __hostdev__ [[nodiscard]] constexpr Mat2     operator*(const Mat2& m) const noexcept     { return this->template multiply<Mat2, Mat2>(m); }
    __hostdev__ [[nodiscard]] constexpr Vec2<T>  operator*(const Vec2<T>& v) const noexcept  { return this->template multiplyVec<Vec2<T>, Vec2<T>>(v); }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Mat2  operator*(const T& s) const noexcept        { return this->template scale<Mat2>(s); }
    __hostdev__ [[nodiscard]] constexpr Mat2  operator/(const T& s) const noexcept        { return this->template divideBy<Mat2>(s); }
    __hostdev__ constexpr Mat2& operator*=(const T& s) noexcept             { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Mat2& operator/=(const T& s) noexcept             { Base::divideAssignScalar(s); return *this; }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Mat2& m) const noexcept     { return Base::equals(m); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Mat2& m) const noexcept     { return !Base::equals(m); }

    /// @brief returns transpose of this
    __hostdev__ [[nodiscard]] constexpr Mat2 transpose() const noexcept { return this->template transposeAs<Mat2>(); }

    /// @brief returns inverse of this
    __hostdev__ [[nodiscard]] constexpr Mat2<T> inverse() const noexcept {
        T det = (*this)[0][0] * (*this)[1][1] - (*this)[0][1] * (*this)[1][0];
        if (isApproxZero(det)) {
            return Mat2<T>(T(0), T(0), T(0), T(0));
        }
        T invDet = T(1) / det;
        return Mat2<T>((*this)[1][1] * invDet, -(*this)[0][1] * invDet,
                      -(*this)[1][0] * invDet,  (*this)[0][0] * invDet);
    }

    /// @brief scalar * Mat (hidden friend — found only via ADL on Mat2)
    __hostdev__ [[nodiscard]] friend constexpr Mat2 operator*(const T& s, const Mat2& m) noexcept {
        return m.template scale<Mat2>(s);
    }
};

/// @brief 2x3 row-major matrix.
/// @details Intentionally NOT @c alignas-elevated: its byte size
/// (6*sizeof(T)) is not a power-of-2 multiple of @c alignof(T), so any
/// @c alignas(N > alignof(T)) would force tail padding and break
/// packed-array layout plus on-disk format compatibility.
template <typename T>
class Mat2x3 final : public MatBase<T, 2, 3> {
    using Base = MatBase<T, 2, 3>;
public:
    /// @brief Default-construct (entries left uninitialized for fundamental @c T).
    Mat2x3() noexcept = default;
    /// @brief Constructor given individual array elements, the ordering is in row major form:
    /** @verbatim
        a b c
        d e f
        @endverbatim */
    __hostdev__ constexpr Mat2x3(T a, T b, T c, T d, T e, T f) noexcept : Base(a, b, c, d, e, f) {}

    /// @brief Constructor given array of elements, the ordering is in row major form
    template<typename Source>
    __hostdev__ constexpr Mat2x3(Source* array) noexcept : Base(array) {}


    // ---- element-wise ----
    __hostdev__ [[nodiscard]] constexpr Mat2x3  operator-() const noexcept                  { return this->template negate<Mat2x3>(); }
    __hostdev__ [[nodiscard]] constexpr Mat2x3  operator+(const Mat2x3& m) const noexcept   { return this->template plus<Mat2x3>(m); }
    __hostdev__ [[nodiscard]] constexpr Mat2x3  operator-(const Mat2x3& m) const noexcept   { return this->template minus<Mat2x3>(m); }
    __hostdev__ constexpr Mat2x3& operator+=(const Mat2x3& m) noexcept        { Base::addAssign(m); return *this; }
    __hostdev__ constexpr Mat2x3& operator-=(const Mat2x3& m) noexcept        { Base::subAssign(m); return *this; }

    // ---- matrix * vector ----
    __hostdev__ [[nodiscard]] constexpr Vec2<T> operator*(const Vec3<T>& v) const noexcept  { return this->template multiplyVec<Vec2<T>, Vec3<T>>(v); }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Mat2x3  operator*(const T& s) const noexcept        { return this->template scale<Mat2x3>(s); }
    __hostdev__ [[nodiscard]] constexpr Mat2x3  operator/(const T& s) const noexcept        { return this->template divideBy<Mat2x3>(s); }
    __hostdev__ constexpr Mat2x3& operator*=(const T& s) noexcept             { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Mat2x3& operator/=(const T& s) noexcept             { Base::divideAssignScalar(s); return *this; }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Mat2x3& m) const noexcept     { return Base::equals(m); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Mat2x3& m) const noexcept     { return !Base::equals(m); }

    /// @brief returns transpose of this
    __hostdev__ [[nodiscard]] constexpr Mat3x2<T> transpose() const noexcept { return this->template transposeAs<Mat3x2<T>>(); }

    /// @brief scalar * Mat (hidden friend — found only via ADL on Mat2x3)
    __hostdev__ [[nodiscard]] friend constexpr Mat2x3 operator*(const T& s, const Mat2x3& m) noexcept {
        return m.template scale<Mat2x3>(s);
    }
};

/// @brief 3x2 row-major matrix.
/// @details Intentionally NOT @c alignas-elevated: its byte size
/// (6*sizeof(T)) is not a power-of-2 multiple of @c alignof(T), so any
/// @c alignas(N > alignof(T)) would force tail padding and break
/// packed-array layout plus on-disk format compatibility.
template <typename T>
class Mat3x2 final : public MatBase<T, 3, 2> {
    using Base = MatBase<T, 3, 2>;
public:
    /// @brief Default-construct (entries left uninitialized for fundamental @c T).
    Mat3x2() noexcept = default;

    /// @brief Constructor given individual array elements, the ordering is in row major form:
    /** @verbatim
        a b
        c d
        e f
        @endverbatim */
    __hostdev__ constexpr Mat3x2(T a, T b, T c, T d, T e, T f) noexcept : Base(a, b, c, d, e, f) {}

    /// @brief Constructor given array of elements, the ordering is in row major form
    template<typename Source>
    __hostdev__ constexpr Mat3x2(Source *a) noexcept : Base(a) {}

    // ---- element-wise ----
    __hostdev__ [[nodiscard]] constexpr Mat3x2  operator-() const noexcept                  { return this->template negate<Mat3x2>(); }
    __hostdev__ [[nodiscard]] constexpr Mat3x2  operator+(const Mat3x2& m) const noexcept   { return this->template plus<Mat3x2>(m); }
    __hostdev__ [[nodiscard]] constexpr Mat3x2  operator-(const Mat3x2& m) const noexcept   { return this->template minus<Mat3x2>(m); }
    __hostdev__ constexpr Mat3x2& operator+=(const Mat3x2& m) noexcept        { Base::addAssign(m); return *this; }
    __hostdev__ constexpr Mat3x2& operator-=(const Mat3x2& m) noexcept        { Base::subAssign(m); return *this; }

    // ---- matrix * vector ----
    __hostdev__ [[nodiscard]] constexpr Vec3<T> operator*(const Vec2<T>& v) const noexcept  { return this->template multiplyVec<Vec3<T>, Vec2<T>>(v); }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Mat3x2  operator*(const T& s) const noexcept        { return this->template scale<Mat3x2>(s); }
    __hostdev__ [[nodiscard]] constexpr Mat3x2  operator/(const T& s) const noexcept        { return this->template divideBy<Mat3x2>(s); }
    __hostdev__ constexpr Mat3x2& operator*=(const T& s) noexcept             { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Mat3x2& operator/=(const T& s) noexcept             { Base::divideAssignScalar(s); return *this; }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Mat3x2& m) const noexcept     { return Base::equals(m); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Mat3x2& m) const noexcept     { return !Base::equals(m); }

    /// @brief returns transpose of this
    __hostdev__ [[nodiscard]] constexpr Mat2x3<T> transpose() const noexcept { return this->template transposeAs<Mat2x3<T>>(); }

    /// @brief scalar * Mat (hidden friend — found only via ADL on Mat3x2)
    __hostdev__ [[nodiscard]] friend constexpr Mat3x2 operator*(const T& s, const Mat3x2& m) noexcept {
        return m.template scale<Mat3x2>(s);
    }
};

/// @brief 3x3 row-major matrix.
/// @details Intentionally NOT @c alignas-elevated: its byte size
/// (9*sizeof(T)) is not a power-of-2 multiple of @c alignof(T), so any
/// @c alignas(N > alignof(T)) would force tail padding and break
/// packed-array layout plus on-disk format compatibility.
template <typename T>
class Mat3 final : public MatBase<T, 3, 3> {
    using Base = MatBase<T, 3, 3>;
public:
    /// @brief Default-construct (entries left uninitialized for fundamental @c T).
    Mat3() noexcept = default;

    /// @brief Constructor given individual array elements, the ordering is in row major form:
    /** @verbatim
        a b c
        d e f
        g h i
        @endverbatim */
    __hostdev__ constexpr Mat3(T a, T b, T c,
                               T d, T e, T f,
                               T g, T h, T i) noexcept : Base(a, b, c, d, e, f, g, h, i) {}

    /// @brief Constructor given array of elements, the ordering is in row major form
    template<typename Source>
    __hostdev__ constexpr Mat3(Source *a) noexcept : Base(a) {}


    // ---- element-wise ----
    __hostdev__ [[nodiscard]] constexpr Mat3  operator-() const noexcept                  { return this->template negate<Mat3>(); }
    __hostdev__ [[nodiscard]] constexpr Mat3  operator+(const Mat3& m) const noexcept     { return this->template plus<Mat3>(m); }
    __hostdev__ [[nodiscard]] constexpr Mat3  operator-(const Mat3& m) const noexcept     { return this->template minus<Mat3>(m); }
    __hostdev__ constexpr Mat3& operator+=(const Mat3& m) noexcept          { Base::addAssign(m); return *this; }
    __hostdev__ constexpr Mat3& operator-=(const Mat3& m) noexcept          { Base::subAssign(m); return *this; }

    // ---- matrix * matrix / matrix * vector ----
    __hostdev__ [[nodiscard]] constexpr Mat3    operator*(const Mat3& m) const noexcept     { return this->template multiply<Mat3, Mat3>(m); }
    __hostdev__ [[nodiscard]] constexpr Vec3<T> operator*(const Vec3<T>& v) const noexcept  { return this->template multiplyVec<Vec3<T>, Vec3<T>>(v); }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Mat3  operator*(const T& s) const noexcept        { return this->template scale<Mat3>(s); }
    __hostdev__ [[nodiscard]] constexpr Mat3  operator/(const T& s) const noexcept        { return this->template divideBy<Mat3>(s); }
    __hostdev__ constexpr Mat3& operator*=(const T& s) noexcept             { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Mat3& operator/=(const T& s) noexcept             { Base::divideAssignScalar(s); return *this; }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Mat3& m) const noexcept     { return Base::equals(m); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Mat3& m) const noexcept     { return !Base::equals(m); }

    /// @brief returns transpose of this
    __hostdev__ [[nodiscard]] constexpr Mat3 transpose() const noexcept { return this->template transposeAs<Mat3>(); }

    /// @brief scalar * Mat (hidden friend — found only via ADL on Mat3)
    __hostdev__ [[nodiscard]] friend constexpr Mat3 operator*(const T& s, const Mat3& m) noexcept {
        return m.template scale<Mat3>(s);
    }
};

/// @brief 4x4 row-major matrix.
/// @details Aligned to 16*alignof(T) — @c Mat4 stores 16 elements (4x4), which is
/// already a power-of-2 multiple of @c alignof(T), so the alignment is free
/// in size. Whole-matrix alignment (64 bytes for @c Mat4<float>, 128 bytes
/// for @c Mat4<double>) is heavy compared to row-alignment (4*alignof(T))
/// but matches the per-class "align to full size" rule used for @c Vec2 /
/// @c Vec4 / @c Mat2 above and lets a @c Mat4<float> load with a single
/// AVX-512 instruction.
template <typename T>
class alignas(alignof(T) * 16) Mat4 final : public MatBase<T, 4, 4> {
    using Base = MatBase<T, 4, 4>;
public:
    /// @brief Default-construct (entries left uninitialized for fundamental @c T).
    Mat4() noexcept = default;

    /// @brief Constructor given individual array elements, the ordering is in row major form:
    /** @verbatim
        a b c d
        e f g h
        i j k l
        m n o p
        @endverbatim */
    __hostdev__ constexpr Mat4(T a, T b, T c, T d,
                               T e, T f, T g, T h,
                               T i, T j, T k, T l,
                               T m, T n, T o, T p) noexcept
        : Base(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) {}

    /// @brief Constructor given array of elements, the ordering is in row major form
    template<typename Source>
    __hostdev__ constexpr Mat4(Source *a) noexcept : Base(a) {}

    // ---- element-wise ----
    __hostdev__ [[nodiscard]] constexpr Mat4  operator-() const noexcept                  { return this->template negate<Mat4>(); }
    __hostdev__ [[nodiscard]] constexpr Mat4  operator+(const Mat4& m) const noexcept     { return this->template plus<Mat4>(m); }
    __hostdev__ [[nodiscard]] constexpr Mat4  operator-(const Mat4& m) const noexcept     { return this->template minus<Mat4>(m); }
    __hostdev__ constexpr Mat4& operator+=(const Mat4& m) noexcept          { Base::addAssign(m); return *this; }
    __hostdev__ constexpr Mat4& operator-=(const Mat4& m) noexcept          { Base::subAssign(m); return *this; }

    // ---- matrix * matrix / matrix * vector ----
    __hostdev__ [[nodiscard]] constexpr Mat4    operator*(const Mat4& m) const noexcept     { return this->template multiply<Mat4, Mat4>(m); }
    __hostdev__ [[nodiscard]] constexpr Vec4<T> operator*(const Vec4<T>& v) const noexcept  { return this->template multiplyVec<Vec4<T>, Vec4<T>>(v); }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Mat4  operator*(const T& s) const noexcept        { return this->template scale<Mat4>(s); }
    __hostdev__ [[nodiscard]] constexpr Mat4  operator/(const T& s) const noexcept        { return this->template divideBy<Mat4>(s); }
    __hostdev__ constexpr Mat4& operator*=(const T& s) noexcept             { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Mat4& operator/=(const T& s) noexcept             { Base::divideAssignScalar(s); return *this; }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Mat4& m) const noexcept     { return Base::equals(m); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Mat4& m) const noexcept     { return !Base::equals(m); }

    /// @brief returns transpose of this
    __hostdev__ [[nodiscard]] constexpr Mat4 transpose() const noexcept { return this->template transposeAs<Mat4>(); }

    /// @brief scalar * Mat (hidden friend — found only via ADL on Mat4)
    __hostdev__ [[nodiscard]] friend constexpr Mat4 operator*(const T& s, const Mat4& m) noexcept {
        return m.template scale<Mat4>(s);
    }
};


/// @brief Multiply a 2x3 matrix by a 3x2 matrix, result is a 2x2 matrix
template<typename T>
__hostdev__ [[nodiscard]] constexpr Mat2<T> operator*(const Mat2x3<T>& lhs, const Mat3x2<T>& rhs) noexcept {
    return lhs.template multiply<Mat2<T>, Mat3x2<T>>(rhs);
}
/// @brief Multiply a 2x3 matrix by a 3x3 matrix, result is a 2x3 matrix
template<typename T>
__hostdev__ [[nodiscard]] constexpr Mat2x3<T> operator*(const Mat2x3<T>& lhs, const Mat3<T>& rhs) noexcept {
    return lhs.template multiply<Mat2x3<T>, Mat3<T>>(rhs);
}
/// @brief Multiply a 3x2 matrix by a 2x2 matrix, result is a 3x2 matrix
template<typename T>
__hostdev__ [[nodiscard]] constexpr Mat3x2<T> operator*(const Mat3x2<T>& lhs, const Mat2<T>& rhs) noexcept {
    return lhs.template multiply<Mat3x2<T>, Mat2<T>>(rhs);
}
/// @brief Multiply a 2x2 matrix by a 2x3 matrix, result is a 2x3 matrix
template<typename T>
__hostdev__ [[nodiscard]] constexpr Mat2x3<T> operator*(const Mat2<T>& lhs, const Mat2x3<T>& rhs) noexcept {
    return lhs.template multiply<Mat2x3<T>, Mat2x3<T>>(rhs);
}
/// @brief Multiply a 3x2 matrix by a 2x3 matrix, result is a 3x3 matrix
template<typename T>
__hostdev__ [[nodiscard]] constexpr Mat3<T> operator*(const Mat3x2<T>& lhs, const Mat2x3<T>& rhs) noexcept {
    return lhs.template multiply<Mat3<T>, Mat2x3<T>>(rhs);
}
/// @brief Multiply a 3x3 matrix by a 3x2 matrix, result is a 3x2 matrix
template<typename T>
__hostdev__ [[nodiscard]] constexpr Mat3x2<T> operator*(const Mat3<T>& lhs, const Mat3x2<T>& rhs) noexcept {
    return lhs.template multiply<Mat3x2<T>, Mat3x2<T>>(rhs);
}
// ----------------------------> Vec3 <--------------------------------------

/// @brief A simple vector class with three components, similar to openvdb::math::Vec3
///
/// Vec3 is intentionally NOT alignas-elevated: its byte size
/// (3*sizeof(T)) is not a power-of-2 multiple of alignof(T), so any
/// alignas(N > alignof(T)) would force tail padding and break
/// packed-array layout plus on-disk format compatibility.
template<typename T>
class Vec3 final : public VecBase<T, 3>
{
    using Base = VecBase<T, 3>;

public:
    using ValueType = T;
    static constexpr int size = 3; // openvdb::math::Tuple-compat alias of SIZE

    /// @brief Default-construct (components are left uninitialized for fundamental @c T).
    Vec3() noexcept = default;
    /// @brief Broadcast: set all three components to @a x.
    __hostdev__ explicit constexpr Vec3(T x) noexcept            : Base(x, x, x) {}
    /// @brief Component-wise construction.
    __hostdev__ constexpr Vec3(T x, T y, T z) noexcept           : Base(x, y, z) {}

    /// @brief Cross-template converting ctor (e.g. from @c openvdb::Vec3). Implicit
    /// to preserve foreign-type interop; same-class ctor below is @c explicit.
    template<template<class> class Vec3T, class T2>
    __hostdev__ explicit constexpr Vec3(const Vec3T<T2>& v) noexcept : Base(v[0], v[1], v[2])
    {
        static_assert(Vec3T<T2>::size == 3, "expected Vec3T::size==3!");
    }
    /// @brief Explicit cross-precision conversion within nanovdb (e.g. @c Vec3d → @c Vec3f).
    template<typename T2>
    __hostdev__ explicit constexpr Vec3(const Vec3<T2>& v) noexcept : Base(v[0], v[1], v[2]) {}
    /// @brief Construct from a 3D integer coordinate.
    __hostdev__ explicit constexpr Vec3(const Coord& ijk) noexcept : Base(ijk[0], ijk[1], ijk[2]) {}

    /// @brief Assign from any 3-component vector type (foreign or nanovdb).
    template<template<class> class Vec3T, class T2>
    __hostdev__ constexpr Vec3& operator=(const Vec3T<T2>& rhs) noexcept {
        static_assert(Vec3T<T2>::size == 3, "expected Vec3T::size==3!");
        this->mVec[0] = rhs[0]; this->mVec[1] = rhs[1]; this->mVec[2] = rhs[2];
        return *this;
    }

    // ---- element-wise (Vec & Vec) ----
    /// @brief Component-wise negation.
    __hostdev__ [[nodiscard]] constexpr Vec3  operator-() const noexcept             { return Base::template negate<Vec3>(); }
    /// @brief Component-wise sum.
    __hostdev__ [[nodiscard]] constexpr Vec3  operator+(const Vec3& v) const noexcept { return Base::template plus<Vec3>(v); }
    /// @brief Component-wise difference.
    __hostdev__ [[nodiscard]] constexpr Vec3  operator-(const Vec3& v) const noexcept { return Base::template minus<Vec3>(v); }
    /// @brief Component-wise (Hadamard) product.
    __hostdev__ [[nodiscard]] constexpr Vec3  operator*(const Vec3& v) const noexcept { return Base::template mul<Vec3>(v); }
    /// @brief Component-wise division.
    __hostdev__ [[nodiscard]] constexpr Vec3  operator/(const Vec3& v) const noexcept { return Base::template div<Vec3>(v); }
    /// @brief In-place component-wise addition.
    __hostdev__ constexpr Vec3& operator+=(const Vec3& v) noexcept     { Base::addAssign(v); return *this; }
    /// @brief In-place component-wise subtraction.
    __hostdev__ constexpr Vec3& operator-=(const Vec3& v) noexcept     { Base::subAssign(v); return *this; }

    // ---- mixed Vec3 / Coord (3D) ----
    /// @brief Add an integer @c Coord to this vector (component-wise).
    __hostdev__ [[nodiscard]] constexpr Vec3  operator+(const Coord& ijk) const noexcept { return Vec3(this->mVec[0] + ijk[0], this->mVec[1] + ijk[1], this->mVec[2] + ijk[2]); }
    /// @brief Subtract an integer @c Coord from this vector (component-wise).
    __hostdev__ [[nodiscard]] constexpr Vec3  operator-(const Coord& ijk) const noexcept { return Vec3(this->mVec[0] - ijk[0], this->mVec[1] - ijk[1], this->mVec[2] - ijk[2]); }
    /// @brief In-place component-wise addition of an integer @c Coord.
    __hostdev__ constexpr Vec3& operator+=(const Coord& ijk) noexcept {
        this->mVec[0] += T(ijk[0]); this->mVec[1] += T(ijk[1]); this->mVec[2] += T(ijk[2]);
        return *this;
    }
    /// @brief In-place component-wise subtraction of an integer @c Coord.
    __hostdev__ constexpr Vec3& operator-=(const Coord& ijk) noexcept {
        this->mVec[0] -= T(ijk[0]); this->mVec[1] -= T(ijk[1]); this->mVec[2] -= T(ijk[2]);
        return *this;
    }

    // ---- scalar ----
    /// @brief Component-wise multiply by scalar @a s.
    __hostdev__ [[nodiscard]] constexpr Vec3  operator*(const T& s) const noexcept   { return Base::template scale<Vec3>(s); }
    /// @brief Component-wise divide by scalar @a s (integer-safe).
    __hostdev__ [[nodiscard]] constexpr Vec3  operator/(const T& s) const noexcept   { return Base::template divideBy<Vec3>(s); }
    /// @brief In-place component-wise multiply by scalar @a s.
    __hostdev__ constexpr Vec3& operator*=(const T& s) noexcept        { Base::scaleAssign(s); return *this; }
    /// @brief In-place component-wise divide by scalar @a s.
    __hostdev__ constexpr Vec3& operator/=(const T& s) noexcept        { Base::divideAssignScalar(s); return *this; }
    /// @brief Normalize in place (divide by @c length()). Not @c constexpr — calls @c std::sqrt.
    __hostdev__ Vec3& normalize() noexcept                   { return (*this) /= this->length(); }
    /// @brief Return a normalized (unit-length) copy; const counterpart of @c normalize().
    __hostdev__ [[nodiscard]] Vec3 normalized() const noexcept {
        return Base::template normalized<Vec3>();
    }

    // ---- equality ----
    /// @brief Component-wise equality.
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Vec3& rhs) const noexcept { return Base::equals(rhs); }
    /// @brief Component-wise inequality.
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Vec3& rhs) const noexcept { return !Base::equals(rhs); }

    // ---- component-wise min/max ----
    /// @brief Take the component-wise minimum of @c *this and @a other in place.
    __hostdev__ constexpr Vec3& minComponent(const Vec3& other) noexcept { Base::mergeMin(other); return *this; }
    /// @brief Take the component-wise maximum of @c *this and @a other in place.
    __hostdev__ constexpr Vec3& maxComponent(const Vec3& other) noexcept { Base::mergeMax(other); return *this; }

    /// @brief Return the smallest vector component
    __hostdev__ [[nodiscard]] constexpr ValueType min() const noexcept { return Base::smallestComponent(); }
    /// @brief Return the largest vector component
    __hostdev__ [[nodiscard]] constexpr ValueType max() const noexcept { return Base::largestComponent(); }

    /// @brief Round each component down (toward negative infinity)
    /// @return integer Coord
    /// @note Only constexpr for integer @c T (floorAs uses non-constexpr math::Floor for floating point).
    __hostdev__ [[nodiscard]] constexpr Coord floor() const noexcept { return Base::template floorAs<Coord>(); }
    /// @brief Round each component up (toward positive infinity)
    /// @return integer Coord
    /// @note Only constexpr for integer @c T (ceilAs uses non-constexpr math::Ceil for floating point).
    __hostdev__ [[nodiscard]] constexpr Coord ceil()  const noexcept { return Base::template ceilAs<Coord>(); }
    /// @brief Round each component to its closest integer value
    /// @return integer Coord
    /// @note Only constexpr for integer @c T (roundAs uses non-constexpr math::Floor for floating point).
    __hostdev__ [[nodiscard]] constexpr Coord round() const noexcept { return Base::template roundAs<Coord>(); }

    // ---- 3D-specific ----

    /// @brief cross product with another 3-vector
    template<typename Vec3T>
    __hostdev__ [[nodiscard]] constexpr Vec3 cross(const Vec3T& v) const noexcept {
        return Vec3(this->mVec[1] * v[2] - this->mVec[2] * v[1],
                    this->mVec[2] * v[0] - this->mVec[0] * v[2],
                    this->mVec[0] * v[1] - this->mVec[1] * v[0]);
    }

    /// @brief Outer product of a 3x1 vector and a 1x3 vector, result is a 3x3 matrix
    template<typename Vec3T>
    __hostdev__ [[nodiscard]] constexpr Mat3<ValueType> outer(const Vec3T& v) const noexcept {
        return Mat3<ValueType>(this->mVec[0] * v[0], this->mVec[0] * v[1], this->mVec[0] * v[2],
                               this->mVec[1] * v[0], this->mVec[1] * v[1], this->mVec[1] * v[2],
                               this->mVec[2] * v[0], this->mVec[2] * v[1], this->mVec[2] * v[2]);
    }

    // ---- scalar * Vec / scalar / Vec (hidden friends — found only via ADL on Vec3,
    // never participate in unrelated namespace-scope overload sets) ----

    /// @brief Scalar-on-the-left multiplication (hidden friend).
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec3 operator*(T1 scalar, const Vec3& vec) noexcept
    {
        return Vec3(scalar * vec[0], scalar * vec[1], scalar * vec[2]);
    }
    /// @brief Scalar-on-the-left division (hidden friend).
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec3 operator/(T1 scalar, const Vec3& vec) noexcept
    {
        return Vec3(scalar / vec[0], scalar / vec[1], scalar / vec[2]);
    }
}; // Vec3<T>

/// @brief Return a single precision floating-point vector of this coordinate
__hostdev__ [[nodiscard]] inline constexpr Vec3<float> Coord::asVec3s() const noexcept
{
    return Vec3<float>(float(mVec[0]), float(mVec[1]), float(mVec[2]));
}

/// @brief Return a double precision floating-point vector of this coordinate
__hostdev__ [[nodiscard]] inline constexpr Vec3<double> Coord::asVec3d() const noexcept
{
    return Vec3<double>(double(mVec[0]), double(mVec[1]), double(mVec[2]));
}

// ----------------------------> Vec4 <--------------------------------------

/// @brief A simple vector class with four components, similar to openvdb::math::Vec4
///
/// Aligned to 4*alignof(T) so the whole class fits in one SIMD register
/// (16 bytes for Vec4<float>, 32 bytes for Vec4<double>), without any
/// tail padding because the byte size is already a power-of-2 multiple
/// of alignof(T).
template<typename T>
class alignas(alignof(T) * 4) Vec4 final : public VecBase<T, 4>
{
    using Base = VecBase<T, 4>;

public:
    using ValueType = T;
    static constexpr int size = 4; // openvdb::math::Tuple-compat alias of SIZE

    /// @brief Default-construct (components are left uninitialized for fundamental @c T).
    Vec4() noexcept = default;
    /// @brief Broadcast: set all four components to @a x.
    __hostdev__ explicit constexpr Vec4(T x) noexcept            : Base(x, x, x, x) {}
    /// @brief Component-wise construction.
    __hostdev__ constexpr Vec4(T x, T y, T z, T w) noexcept      : Base(x, y, z, w) {}

    /// @brief Explicit cross-precision conversion within nanovdb (e.g. @c Vec4d → @c Vec4f).
    template<typename T2>
    __hostdev__ explicit constexpr Vec4(const Vec4<T2>& v) noexcept : Base(v[0], v[1], v[2], v[3]) {}
    /// @brief Cross-template converting ctor (e.g. from @c openvdb::Vec4). Implicit
    /// to preserve foreign-type interop.
    template<template<class> class Vec4T, class T2>
    __hostdev__ explicit constexpr Vec4(const Vec4T<T2>& v) noexcept : Base(v[0], v[1], v[2], v[3])
    {
        static_assert(Vec4T<T2>::size == 4, "expected Vec4T::size==4!");
    }
    /// @brief Assign from any 4-component vector type (foreign or nanovdb).
    template<template<class> class Vec4T, class T2>
    __hostdev__ constexpr Vec4& operator=(const Vec4T<T2>& rhs) noexcept {
        static_assert(Vec4T<T2>::size == 4, "expected Vec4T::size==4!");
        this->mVec[0] = rhs[0]; this->mVec[1] = rhs[1]; this->mVec[2] = rhs[2]; this->mVec[3] = rhs[3];
        return *this;
    }

    // ---- element-wise (Vec & Vec) ----
    /// @brief Component-wise negation.
    __hostdev__ [[nodiscard]] constexpr Vec4  operator-() const noexcept             { return Base::template negate<Vec4>(); }
    /// @brief Component-wise sum.
    __hostdev__ [[nodiscard]] constexpr Vec4  operator+(const Vec4& v) const noexcept { return Base::template plus<Vec4>(v); }
    /// @brief Component-wise difference.
    __hostdev__ [[nodiscard]] constexpr Vec4  operator-(const Vec4& v) const noexcept { return Base::template minus<Vec4>(v); }
    /// @brief Component-wise (Hadamard) product.
    __hostdev__ [[nodiscard]] constexpr Vec4  operator*(const Vec4& v) const noexcept { return Base::template mul<Vec4>(v); }
    /// @brief Component-wise division.
    __hostdev__ [[nodiscard]] constexpr Vec4  operator/(const Vec4& v) const noexcept { return Base::template div<Vec4>(v); }
    /// @brief In-place component-wise addition.
    __hostdev__ constexpr Vec4& operator+=(const Vec4& v) noexcept     { Base::addAssign(v); return *this; }
    /// @brief In-place component-wise subtraction.
    __hostdev__ constexpr Vec4& operator-=(const Vec4& v) noexcept     { Base::subAssign(v); return *this; }

    // ---- scalar ----
    /// @brief Component-wise multiply by scalar @a s.
    __hostdev__ [[nodiscard]] constexpr Vec4  operator*(const T& s) const noexcept   { return Base::template scale<Vec4>(s); }
    /// @brief Component-wise divide by scalar @a s (integer-safe).
    __hostdev__ [[nodiscard]] constexpr Vec4  operator/(const T& s) const noexcept   { return Base::template divideBy<Vec4>(s); }
    /// @brief In-place component-wise multiply by scalar @a s.
    __hostdev__ constexpr Vec4& operator*=(const T& s) noexcept        { Base::scaleAssign(s); return *this; }
    /// @brief In-place component-wise divide by scalar @a s.
    __hostdev__ constexpr Vec4& operator/=(const T& s) noexcept        { Base::divideAssignScalar(s); return *this; }
    /// @brief Normalize in place (divide by @c length()). Not @c constexpr — calls @c std::sqrt.
    __hostdev__ Vec4& normalize() noexcept                   { return (*this) /= this->length(); }
    /// @brief Return a normalized (unit-length) copy; const counterpart of @c normalize().
    __hostdev__ [[nodiscard]] Vec4 normalized() const noexcept {
        return Base::template normalized<Vec4>();
    }

    // ---- equality ----
    /// @brief Component-wise equality.
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Vec4& rhs) const noexcept { return Base::equals(rhs); }
    /// @brief Component-wise inequality.
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Vec4& rhs) const noexcept { return !Base::equals(rhs); }

    // ---- component-wise min/max ----
    /// @brief Take the component-wise minimum of @c *this and @a other in place.
    __hostdev__ constexpr Vec4& minComponent(const Vec4& other) noexcept { Base::mergeMin(other); return *this; }
    /// @brief Take the component-wise maximum of @c *this and @a other in place.
    __hostdev__ constexpr Vec4& maxComponent(const Vec4& other) noexcept { Base::mergeMax(other); return *this; }

    /// @brief Return the smallest vector component
    __hostdev__ [[nodiscard]] constexpr ValueType min() const noexcept { return Base::smallestComponent(); }
    /// @brief Return the largest vector component
    __hostdev__ [[nodiscard]] constexpr ValueType max() const noexcept { return Base::largestComponent(); }

    /// @brief Round each component down (toward negative infinity)
    /// @return Vec4<int32_t> (NanoVDB has no Coord4)
    /// @note Only constexpr for integer @c T (floorAs uses non-constexpr math::Floor for floating point).
    __hostdev__ [[nodiscard]] constexpr Vec4<int32_t> floor() const noexcept { return Base::template floorAs<Vec4<int32_t>>(); }
    /// @brief Round each component up (toward positive infinity)
    /// @return Vec4<int32_t>
    /// @note Only constexpr for integer @c T (ceilAs uses non-constexpr math::Ceil for floating point).
    __hostdev__ [[nodiscard]] constexpr Vec4<int32_t> ceil()  const noexcept { return Base::template ceilAs<Vec4<int32_t>>(); }
    /// @brief Round each component to its closest integer value
    /// @return Vec4<int32_t>
    /// @note Only constexpr for integer @c T (roundAs uses non-constexpr math::Floor for floating point).
    __hostdev__ [[nodiscard]] constexpr Vec4<int32_t> round() const noexcept { return Base::template roundAs<Vec4<int32_t>>(); }

    // ---- scalar * Vec / scalar / Vec (hidden friends — found only via ADL on Vec4,
    // never participate in unrelated namespace-scope overload sets) ----

    /// @brief Scalar-on-the-left multiplication (hidden friend).
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec4 operator*(T1 scalar, const Vec4& vec) noexcept
    {
        return Vec4(scalar * vec[0], scalar * vec[1], scalar * vec[2], scalar * vec[3]);
    }
    /// @brief Scalar-on-the-left division (hidden friend).
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec4 operator/(T1 scalar, const Vec4& vec) noexcept
    {
        return Vec4(scalar / vec[0], scalar / vec[1], scalar / vec[2], scalar / vec[3]);
    }
}; // Vec4<T>
// ----------------------------> matMult <--------------------------------------
//
// All six matMult / matMultT overloads were originally written with
// fma / fmaf for the single-rounding precision benefit. Those stdlib
// functions are not constexpr in C++17, which transitively blocked
// Map::applyMap and BBox<CoordT,false>::transform<Map> from being
// constexpr. Switching to plain `a * b + c` form gives back the
// constexpr-eligibility (worth ~1 ulp of rounding accuracy in the
// worst case, well below NanoVDB's geometric precision) and the
// device-side codegen is unchanged in practice — nvcc contracts
// `a * b + c` back into a hardware FMA by default (-fmad=true).
//
// (C++23's constexpr fma will eventually obviate this trade-off.)

/// @brief Multiply a 3x3 matrix and a 3d vector using 32bit floating point arithmetics
/// @note This corresponds to a linear mapping, e.g. scaling, rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the matrix
/// @return result of matrix-vector multiplication, i.e. mat x xyz
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMult(const float* mat, const Vec3T& xyz) noexcept
{
    const float x = static_cast<float>(xyz[0]);
    const float y = static_cast<float>(xyz[1]);
    const float z = static_cast<float>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[1] + z * mat[2],
                 x * mat[3] + y * mat[4] + z * mat[5],
                 x * mat[6] + y * mat[7] + z * mat[8]);
}

/// @brief Multiply a 3x3 matrix and a 3d vector using 64bit floating point arithmetics
/// @note This corresponds to a linear mapping, e.g. scaling, rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the matrix
/// @return result of matrix-vector multiplication, i.e. mat x xyz
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMult(const double* mat, const Vec3T& xyz) noexcept
{
    const double x = static_cast<double>(xyz[0]);
    const double y = static_cast<double>(xyz[1]);
    const double z = static_cast<double>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[1] + z * mat[2],
                 x * mat[3] + y * mat[4] + z * mat[5],
                 x * mat[6] + y * mat[7] + z * mat[8]);
}

/// @brief Multiply a 3x3 matrix to a 3d vector and add another 3d vector using 32bit floating point arithmetics
/// @note This corresponds to an affine transformation, i.e a linear mapping followed by a translation. e.g. scale/rotation and translation
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param vec 3d vector to be added AFTER the matrix multiplication
/// @param xyz input vector to be multiplied by the matrix and a translated by @c vec
/// @return result of affine transformation, i.e. (mat x xyz) + vec
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMult(const float* mat, const float* vec, const Vec3T& xyz) noexcept
{
    const float x = static_cast<float>(xyz[0]);
    const float y = static_cast<float>(xyz[1]);
    const float z = static_cast<float>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[1] + z * mat[2] + vec[0],
                 x * mat[3] + y * mat[4] + z * mat[5] + vec[1],
                 x * mat[6] + y * mat[7] + z * mat[8] + vec[2]);
}

/// @brief Multiply a 3x3 matrix to a 3d vector and add another 3d vector using 64bit floating point arithmetics
/// @note This corresponds to an affine transformation, i.e a linear mapping followed by a translation. e.g. scale/rotation and translation
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param vec 3d vector to be added AFTER the matrix multiplication
/// @param xyz input vector to be multiplied by the matrix and a translated by @c vec
/// @return result of affine transformation, i.e. (mat x xyz) + vec
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMult(const double* mat, const double* vec, const Vec3T& xyz) noexcept
{
    const double x = static_cast<double>(xyz[0]);
    const double y = static_cast<double>(xyz[1]);
    const double z = static_cast<double>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[1] + z * mat[2] + vec[0],
                 x * mat[3] + y * mat[4] + z * mat[5] + vec[1],
                 x * mat[6] + y * mat[7] + z * mat[8] + vec[2]);
}

/// @brief Multiply the transposed of a 3x3 matrix and a 3d vector using 32bit floating point arithmetics
/// @note This corresponds to an inverse linear mapping, e.g. inverse scaling, inverse rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the transposed matrix
/// @return result of matrix-vector multiplication, i.e. mat^T x xyz
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMultT(const float* mat, const Vec3T& xyz) noexcept
{
    const float x = static_cast<float>(xyz[0]);
    const float y = static_cast<float>(xyz[1]);
    const float z = static_cast<float>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[3] + z * mat[6],
                 x * mat[1] + y * mat[4] + z * mat[7],
                 x * mat[2] + y * mat[5] + z * mat[8]);
}

/// @brief Multiply the transposed of a 3x3 matrix and a 3d vector using 64bit floating point arithmetics
/// @note This corresponds to an inverse linear mapping, e.g. inverse scaling, inverse rotation etc.
/// @tparam Vec3T Template type of the input and output 3d vectors
/// @param mat pointer to an array of floats with the 3x3 matrix
/// @param xyz input vector to be multiplied by the transposed matrix
/// @return result of matrix-vector multiplication, i.e. mat^T x xyz
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMultT(const double* mat, const Vec3T& xyz) noexcept
{
    const double x = static_cast<double>(xyz[0]);
    const double y = static_cast<double>(xyz[1]);
    const double z = static_cast<double>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[3] + z * mat[6],
                 x * mat[1] + y * mat[4] + z * mat[7],
                 x * mat[2] + y * mat[5] + z * mat[8]);
}

/// @brief Multiply the transpose of a 3x3 matrix by @a xyz and add @a vec (32-bit floats).
/// @note Corresponds to an inverse affine transform: @c (mat^T x xyz) + @c vec.
/// @tparam Vec3T Template type of the input and output 3d vectors
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMultT(const float* mat, const float* vec, const Vec3T& xyz) noexcept
{
    const float x = static_cast<float>(xyz[0]);
    const float y = static_cast<float>(xyz[1]);
    const float z = static_cast<float>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[3] + z * mat[6] + vec[0],
                 x * mat[1] + y * mat[4] + z * mat[7] + vec[1],
                 x * mat[2] + y * mat[5] + z * mat[8] + vec[2]);
}

/// @brief Multiply the transpose of a 3x3 matrix by @a xyz and add @a vec (64-bit floats).
/// @note Corresponds to an inverse affine transform: @c (mat^T x xyz) + @c vec.
/// @tparam Vec3T Template type of the input and output 3d vectors
template<typename Vec3T>
__hostdev__ [[nodiscard]] inline constexpr Vec3T matMultT(const double* mat, const double* vec, const Vec3T& xyz) noexcept
{
    const double x = static_cast<double>(xyz[0]);
    const double y = static_cast<double>(xyz[1]);
    const double z = static_cast<double>(xyz[2]);
    return Vec3T(x * mat[0] + y * mat[3] + z * mat[6] + vec[0],
                 x * mat[1] + y * mat[4] + z * mat[7] + vec[1],
                 x * mat[2] + y * mat[5] + z * mat[8] + vec[2]);
}

// ----------------------------> BBox <-------------------------------------

/// @brief Common base for floating-point and integer bounding boxes; not
/// constructible directly (only as a base of @c BBox).
/// @details Stores the closed-segment endpoints @c mCoord[0] (min corner) and
/// @c mCoord[1] (max corner). Whether the box is "min-inclusive max-inclusive"
/// or "min-inclusive max-exclusive" depends on the derived @c BBox
/// specialization (integer vs floating point).
template<typename Vec3T>
struct BaseBBox
{
    Vec3T                    mCoord[2];
    /// @brief Equality on the two corner points.
    __hostdev__ [[nodiscard]] constexpr bool         operator==(const BaseBBox& rhs) const noexcept { return mCoord[0] == rhs.mCoord[0] && mCoord[1] == rhs.mCoord[1]; };
    /// @brief Inequality on the two corner points.
    __hostdev__ [[nodiscard]] constexpr bool         operator!=(const BaseBBox& rhs) const noexcept { return mCoord[0] != rhs.mCoord[0] || mCoord[1] != rhs.mCoord[1]; };
    /// @brief Indexed corner access: @a i = 0 returns min, @a i = 1 returns max.
    __hostdev__ constexpr const Vec3T& operator[](int i) const noexcept { NANOVDB_ASSERT(i >= 0 && i < 2); return mCoord[i]; }
    /// @brief Mutable variant of @c operator[].
    __hostdev__ constexpr Vec3T&       operator[](int i) noexcept { NANOVDB_ASSERT(i >= 0 && i < 2); return mCoord[i]; }
    /// @brief Mutable accessor for the min corner.
    __hostdev__ constexpr Vec3T&       min() noexcept { return mCoord[0]; }
    /// @brief Mutable accessor for the max corner.
    __hostdev__ constexpr Vec3T&       max() noexcept { return mCoord[1]; }
    /// @brief Const accessor for the min corner.
    __hostdev__ constexpr const Vec3T& min() const noexcept { return mCoord[0]; }
    /// @brief Const accessor for the max corner.
    __hostdev__ constexpr const Vec3T& max() const noexcept { return mCoord[1]; }
    /// @brief Translate (rigid-shift) both corners by @a xyz; return @c *this.
    __hostdev__ constexpr BaseBBox&    translate(const Vec3T& xyz) noexcept
    {
        mCoord[0] += xyz;
        mCoord[1] += xyz;
        return *this;
    }
    /// @brief Expand this bounding box to enclose point @c xyz.
    __hostdev__ constexpr BaseBBox& expand(const Vec3T& xyz) noexcept
    {
        mCoord[0].minComponent(xyz);
        mCoord[1].maxComponent(xyz);
        return *this;
    }

    /// @brief Expand this bounding box to enclose the given bounding box.
    __hostdev__ constexpr BaseBBox& expand(const BaseBBox& bbox) noexcept
    {
        mCoord[0].minComponent(bbox[0]);
        mCoord[1].maxComponent(bbox[1]);
        return *this;
    }

    /// @brief Intersect this bounding box with the given bounding box.
    __hostdev__ constexpr BaseBBox& intersect(const BaseBBox& bbox) noexcept
    {
        mCoord[0].maxComponent(bbox[0]);
        mCoord[1].minComponent(bbox[1]);
        return *this;
    }

    /// @brief Return @c true iff @a xyz lies in the closed box (each component
    /// satisfies @c min[i] <= @c xyz[i] <= @c max[i]).
    __hostdev__ [[nodiscard]] constexpr bool isInside(const Vec3T& xyz) const noexcept
    {
        if (xyz[0] < mCoord[0][0] || xyz[1] < mCoord[0][1] || xyz[2] < mCoord[0][2])
            return false;
        if (xyz[0] > mCoord[1][0] || xyz[1] > mCoord[1][1] || xyz[2] > mCoord[1][2])
            return false;
        return true;
    }

protected:
    /// @brief Default-construct (leaves corners uninitialized — derived classes
    /// supply a meaningful default).
    __hostdev__ constexpr BaseBBox() noexcept {}
    /// @brief Construct from @a min and @a max corners directly.
    __hostdev__ constexpr BaseBBox(const Vec3T& min, const Vec3T& max) noexcept
        : mCoord{min, max}
    {
    }
}; // BaseBBox

template<typename Vec3T, bool = util::is_floating_point<typename Vec3T::ValueType>::value>
struct BBox;

/// @brief Partial template specialization for floating point coordinate types.
///
/// @note Min is inclusive and max is exclusive. If min = max the dimension of
///       the bounding box is zero and therefore it is also empty.
template<typename Vec3T>
struct BBox<Vec3T, true> : public BaseBBox<Vec3T>
{
    using Vec3Type = Vec3T;
    using ValueType = typename Vec3T::ValueType;
    static_assert(util::is_floating_point<ValueType>::value, "Expected a floating point coordinate type");
    using BaseT = BaseBBox<Vec3T>;
    using BaseT::mCoord;
    /// @brief Default construction sets BBox to an empty bbox
    __hostdev__ constexpr BBox() noexcept
        : BaseT(Vec3T( Maximum<typename Vec3T::ValueType>::value()),
                Vec3T(-Maximum<typename Vec3T::ValueType>::value()))
    {
    }
    /// @brief Construct from explicit @a min / @a max corners.
    __hostdev__ constexpr BBox(const Vec3T& min, const Vec3T& max) noexcept
        : BaseT(min, max)
    {
    }
    /// @brief Convert an integer @c Coord box into this floating-point box.
    /// @note @c max is exclusive in the floating-point convention, so the
    /// integer @c max corner is incremented by 1 before conversion.
    __hostdev__ constexpr BBox(const Coord& min, const Coord& max) noexcept
        : BaseT(Vec3T(ValueType(min[0]), ValueType(min[1]), ValueType(min[2])),
                Vec3T(ValueType(max[0] + 1), ValueType(max[1] + 1), ValueType(max[2] + 1)))
    {
    }
    /// @brief Return a cube-shaped @c BBox with min corner @a min and edge length @a dim.
    __hostdev__ [[nodiscard]] static constexpr BBox createCube(const Coord& min, typename Coord::ValueType dim) noexcept
    {
        return BBox(min, min.offsetBy(dim));
    }

    /// @brief Construct from a @c BaseBBox<Coord>; delegates to the @c (min, @c max) ctor.
    __hostdev__ constexpr BBox(const BaseBBox<Coord>& bbox) noexcept
        : BBox(bbox[0], bbox[1])
    {
    }
    /// @brief Return @c true if this bounding box is empty (max <= min in any axis).
    __hostdev__ [[nodiscard]] constexpr bool  empty() const noexcept { return mCoord[0][0] >= mCoord[1][0] ||
                                             mCoord[0][1] >= mCoord[1][1] ||
                                             mCoord[0][2] >= mCoord[1][2]; }
    /// @brief Convert to bool: @c true iff this bounding box has positive volume.
    __hostdev__ [[nodiscard]] constexpr operator bool() const noexcept { return mCoord[0][0] < mCoord[1][0] &&
                                               mCoord[0][1] < mCoord[1][1] &&
                                               mCoord[0][2] < mCoord[1][2]; }
    /// @brief Return @c max - @c min when non-empty, else the zero vector.
    __hostdev__ [[nodiscard]] constexpr Vec3T dim() const noexcept { return *this ? this->max() - this->min() : Vec3T(0); }
    /// @brief Return @c true iff @a p is strictly inside the box (open interval — min is exclusive, max is exclusive).
    __hostdev__ [[nodiscard]] constexpr bool  isInside(const Vec3T& p) const noexcept
    {
        return p[0] > mCoord[0][0] && p[1] > mCoord[0][1] && p[2] > mCoord[0][2] &&
               p[0] < mCoord[1][0] && p[1] < mCoord[1][1] && p[2] < mCoord[1][2];
    }

}; // BBox<Vec3T, true>

/// @brief Partial template specialization for integer coordinate types
///
/// @note Both min and max are INCLUDED in the bbox so dim = max - min + 1. So,
///       if min = max the bounding box contains exactly one point and dim = 1!
template<typename CoordT>
struct BBox<CoordT, false> : public BaseBBox<CoordT>
{
    static_assert(util::is_same<int, typename CoordT::ValueType>::value, "Expected \"int\" coordinate type");
    using BaseT = BaseBBox<CoordT>;
    using BaseT::mCoord;
    /// @brief Iterator over the domain covered by a BBox
    /// @details z is the fastest-moving coordinate.
    class Iterator
    {
        const BBox& mBBox;
        CoordT      mPos;

    public:
        /// @brief Construct an iterator positioned at @c b.min().
        __hostdev__ constexpr Iterator(const BBox& b) noexcept
            : mBBox(b)
            , mPos(b.min())
        {
        }
        /// @brief Construct an iterator positioned at the given @a p inside @a b.
        __hostdev__ constexpr Iterator(const BBox& b, const Coord& p) noexcept
            : mBBox(b)
            , mPos(p)
        {
        }
        /// @brief Pre-increment: advance to the next coordinate in row-major z-fastest order.
        __hostdev__ constexpr Iterator& operator++() noexcept
        {
            if (mPos[2] < mBBox[1][2]) { // this is the most common case
                ++mPos[2];// increment z
            } else if (mPos[1] < mBBox[1][1]) {
                mPos[2] = mBBox[0][2];// reset z
                ++mPos[1];// increment y
            } else if (mPos[0] <= mBBox[1][0]) {
                mPos[2] = mBBox[0][2];// reset z
                mPos[1] = mBBox[0][1];// reset y
                ++mPos[0];// increment x
            }
            return *this;
        }
        /// @brief Post-increment: advance to the next coordinate; return a copy of the previous state.
        __hostdev__ constexpr Iterator operator++(int) noexcept
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        /// @brief Iterator equality (asserts both iterators belong to the same @c BBox).
        __hostdev__ [[nodiscard]] constexpr bool operator==(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos == rhs.mPos;
        }
        /// @brief Iterator inequality (asserts both iterators belong to the same @c BBox).
        __hostdev__ [[nodiscard]] constexpr bool operator!=(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos != rhs.mPos;
        }
        /// @brief Lexicographic less-than over the underlying position.
        __hostdev__ [[nodiscard]] constexpr bool operator<(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos < rhs.mPos;
        }
        /// @brief Lexicographic less-than-or-equal over the underlying position.
        __hostdev__ [[nodiscard]] constexpr bool operator<=(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos <= rhs.mPos;
        }
        /// @brief Return @c true if the iterator still points to a valid coordinate.
        __hostdev__ [[nodiscard]] constexpr operator bool() const noexcept { return mPos <= mBBox[1]; }
        /// @brief Dereference to the current coordinate.
        __hostdev__ [[nodiscard]] constexpr const CoordT& operator*() const noexcept { return mPos; }
    }; // Iterator
    /// @brief Iterator positioned at the min corner.
    __hostdev__ [[nodiscard]] constexpr Iterator begin() const noexcept { return Iterator{*this}; }
    /// @brief One-past-the-end iterator (max corner shifted by one in @c x).
    __hostdev__ [[nodiscard]] constexpr Iterator end()   const noexcept { return Iterator{*this, CoordT(mCoord[1][0]+1, mCoord[0][1], mCoord[0][2])}; }
    /// @brief Default construct an empty bbox (@c min = @c CoordT::max(), @c max = @c CoordT::min()).
    __hostdev__ constexpr BBox() noexcept
        : BaseT(CoordT::max(), CoordT::min())
    {
    }
    /// @brief Construct from explicit @a min / @a max integer corners.
    __hostdev__ constexpr BBox(const CoordT& min, const CoordT& max) noexcept
        : BaseT(min, max)
    {
    }

    /// @brief Splitting constructor (used by parallel range-based iteration).
    /// @details Halves the box along its longest axis: @c other keeps the upper
    /// half, the newly constructed @c BBox is the lower half. The @c SplitT tag
    /// type is ignored; it exists so this overload doesn't collide with the
    /// copy constructor.
    template<typename SplitT>
    __hostdev__ constexpr BBox(BBox& other, const SplitT&) noexcept
        : BaseT(other.mCoord[0], other.mCoord[1])
    {
        NANOVDB_ASSERT(this->is_divisible());
        const int n = MaxIndex(this->dim());
        mCoord[1][n] = (mCoord[0][n] + mCoord[1][n]) >> 1;
        other.mCoord[0][n] = mCoord[1][n] + 1;
    }

    /// @brief Return a cube-shaped @c BBox with min corner @a min and edge length @a dim.
    /// @note Subtracts 1 from @a dim because the integer convention is min/max inclusive.
    __hostdev__ [[nodiscard]] static constexpr BBox createCube(const CoordT& min, typename CoordT::ValueType dim) noexcept
    {
        return BBox(min, min.offsetBy(dim - 1));
    }

    /// @brief Return a cube-shaped @c BBox spanning @a min..@a max in every axis.
    __hostdev__ [[nodiscard]] static constexpr BBox createCube(typename CoordT::ValueType min, typename CoordT::ValueType max) noexcept
    {
        return BBox(CoordT(min), CoordT(max));
    }

    /// @brief Return @c true iff the box has more than one cell along every axis.
    __hostdev__ [[nodiscard]] constexpr bool is_divisible() const noexcept { return mCoord[0][0] < mCoord[1][0] &&
                                                   mCoord[0][1] < mCoord[1][1] &&
                                                   mCoord[0][2] < mCoord[1][2]; }
    /// @brief Return true if this bounding box is empty, e.g. uninitialized
    __hostdev__ [[nodiscard]] constexpr bool     empty() const noexcept { return mCoord[0][0] > mCoord[1][0] ||
                                                mCoord[0][1] > mCoord[1][1] ||
                                                mCoord[0][2] > mCoord[1][2]; }
    /// @brief Convert this BBox to boolean true if it is not empty
    __hostdev__ [[nodiscard]] constexpr operator bool() const noexcept { return mCoord[0][0] <= mCoord[1][0] &&
                                               mCoord[0][1] <= mCoord[1][1] &&
                                               mCoord[0][2] <= mCoord[1][2]; }
    /// @brief Return the per-axis extent (max - min + 1 when non-empty, else 0).
    __hostdev__ [[nodiscard]] constexpr CoordT   dim() const noexcept { return *this ? this->max() - this->min() + Coord(1) : Coord(0); }
    /// @brief Return the number of integer cells in the box, i.e. @c dim().x * @c dim().y * @c dim().z.
    __hostdev__ [[nodiscard]] constexpr uint64_t volume() const noexcept
    {
        auto d = this->dim();
        return uint64_t(d[0]) * uint64_t(d[1]) * uint64_t(d[2]);
    }
    /// @brief Return @c true iff the integer coordinate @a p lies inside the closed box.
    __hostdev__ [[nodiscard]] constexpr bool isInside(const CoordT& p) const noexcept { return !(CoordT::lessThan(p, this->min()) || CoordT::lessThan(this->max(), p)); }
    /// @brief Return @c true if the given bounding box is inside this bounding box.
    __hostdev__ [[nodiscard]] constexpr bool isInside(const BBox& b) const noexcept
    {
        return !(CoordT::lessThan(b.min(), this->min()) || CoordT::lessThan(this->max(), b.max()));
    }

    /// @brief Return @c true if the given bounding box overlaps with this bounding box.
    __hostdev__ [[nodiscard]] constexpr bool hasOverlap(const BBox& b) const noexcept
    {
        return !(CoordT::lessThan(this->max(), b.min()) || CoordT::lessThan(b.max(), this->min()));
    }

    /// @warning This converts a CoordBBox into a floating-point bounding box which implies that max += 1 !
    template<typename RealT = double>
    __hostdev__ [[nodiscard]] constexpr BBox<Vec3<RealT>> asReal() const noexcept
    {
        static_assert(util::is_floating_point<RealT>::value, "CoordBBox::asReal: Expected a floating point coordinate");
        return BBox<Vec3<RealT>>(Vec3<RealT>(RealT(mCoord[0][0]), RealT(mCoord[0][1]), RealT(mCoord[0][2])),
                                 Vec3<RealT>(RealT(mCoord[1][0] + 1), RealT(mCoord[1][1] + 1), RealT(mCoord[1][2] + 1)));
    }
    /// @brief Return a new instance that is expanded by the specified padding.
    __hostdev__ [[nodiscard]] constexpr BBox expandBy(typename CoordT::ValueType padding) const noexcept
    {
        return BBox(mCoord[0].offsetBy(-padding), mCoord[1].offsetBy(padding));
    }

    /// @brief Transform this coordinate bounding box by the specified map.
    /// @param map mapping of index to world coordinates
    /// @return world bounding box
    template<typename Map>
    __hostdev__ [[nodiscard]] constexpr auto transform(const Map& map) const noexcept
    {
        using Vec3T = Vec3<double>;
        const Vec3T tmp = map.applyMap(Vec3T(mCoord[0][0], mCoord[0][1], mCoord[0][2]));
        BBox<Vec3T> bbox(tmp, tmp);// return value
        bbox.expand(map.applyMap(Vec3T(mCoord[0][0], mCoord[0][1], mCoord[1][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[0][0], mCoord[1][1], mCoord[0][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[0][1], mCoord[0][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[1][1], mCoord[0][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[0][1], mCoord[1][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[0][0], mCoord[1][1], mCoord[1][2])));
        bbox.expand(map.applyMap(Vec3T(mCoord[1][0], mCoord[1][1], mCoord[1][2])));
        return bbox;
    }

#if defined(__CUDACC__) // the following functions only run on the GPU!
    /// @brief Device-only: atomically expand the box to enclose the integer point @a ijk.
    __device__ inline BBox& expandAtomic(const CoordT& ijk) noexcept
    {
        mCoord[0].minComponentAtomic(ijk);
        mCoord[1].maxComponentAtomic(ijk);
        return *this;
    }
    /// @brief Device-only: atomically expand the box to enclose another @a bbox.
    __device__ inline BBox& expandAtomic(const BBox& bbox) noexcept
    {
        mCoord[0].minComponentAtomic(bbox[0]);
        mCoord[1].maxComponentAtomic(bbox[1]);
        return *this;
    }
    /// @brief Device-only: atomically intersect this box with @a bbox.
    __device__ inline BBox& intersectAtomic(const BBox& bbox) noexcept
    {
        mCoord[0].maxComponentAtomic(bbox[0]);
        mCoord[1].minComponentAtomic(bbox[1]);
        return *this;
    }
#endif
}; // BBox<CoordT, false>

// --------------------------> Rgba8 <------------------------------------

/// @brief 8-bit red, green, blue, alpha packed into 32 bit unsigned int
class Rgba8
{
    union
    {
        uint8_t  c[4];   // 4 integer color channels of red, green, blue and alpha components.
        uint32_t packed; // 32 bit packed representation
    } mData;

public:
    static const int SIZE = 4;
    using ValueType = uint8_t;

    /// @brief Default copy constructor
    Rgba8(const Rgba8&) noexcept = default;

    /// @brief Default move constructor
    Rgba8(Rgba8&&) noexcept = default;

    /// @brief Default move assignment operator
    /// @return non-const reference to this instance
    Rgba8&      operator=(Rgba8&&) noexcept = default;

    /// @brief Default copy assignment operator
    /// @return non-const reference to this instance
    Rgba8&      operator=(const Rgba8&) noexcept = default;

    /// @brief Default ctor initializes all channels to zero
    __hostdev__ constexpr Rgba8() noexcept
        : mData{{0, 0, 0, 0}}
    {
        static_assert(sizeof(uint32_t) == sizeof(Rgba8), "Unexpected sizeof");
    }

    /// @brief integer r,g,b,a ctor where alpha channel defaults to opaque
    /// @note all values should be in the range 0u to 255u
    __hostdev__ constexpr Rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255u) noexcept
        : mData{{r, g, b, a}}
    {
    }

    /// @brief Ctor where all channels are initialized to the same value.
    /// @note value should be in the range 0u to 255u
    explicit __hostdev__ constexpr Rgba8(uint8_t v) noexcept
        : mData{{v, v, v, v}}
    {
    }

    /// @brief floating-point r,g,b,a ctor where alpha channel defaults to opaque
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ constexpr Rgba8(float r, float g, float b, float a = 1.0f) noexcept
        : mData{{static_cast<uint8_t>(0.5f + r * 255.0f), // round floats to nearest integers
                 static_cast<uint8_t>(0.5f + g * 255.0f), // double {{}} is needed due to union
                 static_cast<uint8_t>(0.5f + b * 255.0f),
                 static_cast<uint8_t>(0.5f + a * 255.0f)}}
    {
    }

    /// @brief Construct from a @c Vec3<float> with rgb components (alpha set to opaque).
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ constexpr Rgba8(const Vec3<float>& rgb) noexcept
        : Rgba8(rgb[0], rgb[1], rgb[2])
    {
    }

    /// @brief Construct from a @c Vec4<float> with rgba components.
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ constexpr Rgba8(const Vec4<float>& rgba) noexcept
        : Rgba8(rgba[0], rgba[1], rgba[2], rgba[3])
    {
    }

    /// @brief Compare on the 32-bit packed integer representation (lexicographic by byte).
    __hostdev__ [[nodiscard]] bool  operator< (const Rgba8& rhs) const noexcept { return mData.packed < rhs.mData.packed; }
    /// @brief Equality on the 32-bit packed integer representation.
    __hostdev__ [[nodiscard]] bool  operator==(const Rgba8& rhs) const noexcept { return mData.packed == rhs.mData.packed; }
    /// @brief Return the squared L2 length of the rgb channels in the [0, 1] range
    /// (alpha is ignored). @c 0.0000153787005f is @c 1/255^2.
    __hostdev__ [[nodiscard]] constexpr float lengthSqr() const noexcept
    {
        return 0.0000153787005f * (float(mData.c[0]) * mData.c[0] +
                                   float(mData.c[1]) * mData.c[1] +
                                   float(mData.c[2]) * mData.c[2]); //1/255^2
    }
    /// @brief L2 length of the rgb channels (alpha ignored). Not @c constexpr — calls @c sqrtf.
    __hostdev__ [[nodiscard]] float           length() const noexcept { return sqrtf(this->lengthSqr()); }
    /// @brief Return the @a n'th color channel as a float in the range 0 to 1.
    __hostdev__ [[nodiscard]] constexpr float           asFloat(int n) const noexcept { return 0.003921569f*float(mData.c[n]); }// divide by 255
    /// @brief Indexed channel access. Asserts @c 0 <= @a n < 4 in debug builds.
    __hostdev__ constexpr const uint8_t&  operator[](int n) const noexcept { NANOVDB_ASSERT(n >= 0 && n < 4); return mData.c[n]; }
    /// @brief Mutable variant of @c operator[].
    __hostdev__ constexpr uint8_t&        operator[](int n) noexcept { NANOVDB_ASSERT(n >= 0 && n < 4); return mData.c[n]; }
    /// @brief Const access to the 32-bit packed integer representation.
    __hostdev__ const uint32_t& packed() const noexcept { return mData.packed; }
    /// @brief Mutable access to the 32-bit packed integer representation.
    __hostdev__ uint32_t&       packed() noexcept { return mData.packed; }
    //@{
    /// @brief Named channel accessors (r, g, b, a). Both const and non-const variants.
    __hostdev__ constexpr const uint8_t&  r() const noexcept { return mData.c[0]; }
    __hostdev__ constexpr const uint8_t&  g() const noexcept { return mData.c[1]; }
    __hostdev__ constexpr const uint8_t&  b() const noexcept { return mData.c[2]; }
    __hostdev__ constexpr const uint8_t&  a() const noexcept { return mData.c[3]; }
    __hostdev__ constexpr uint8_t&        r() noexcept { return mData.c[0]; }
    __hostdev__ constexpr uint8_t&        g() noexcept { return mData.c[1]; }
    __hostdev__ constexpr uint8_t&        b() noexcept { return mData.c[2]; }
    __hostdev__ constexpr uint8_t&        a() noexcept { return mData.c[3]; }
    //@}
    /// @brief Implicit conversion to a @c Vec3<float> with channels in [0, 1] (alpha dropped).
    __hostdev__ [[nodiscard]] constexpr           operator Vec3<float>() const noexcept {
        return Vec3<float>(this->asFloat(0), this->asFloat(1), this->asFloat(2));
    }
    /// @brief Implicit conversion to a @c Vec4<float> with rgba channels in [0, 1].
    __hostdev__ [[nodiscard]] constexpr           operator Vec4<float>() const noexcept {
        return Vec4<float>(this->asFloat(0), this->asFloat(1), this->asFloat(2), this->asFloat(3));
    }
}; // Rgba8

using Vec3d  = Vec3<double>;
using Vec3f  = Vec3<float>;
using Vec3i  = Vec3<int32_t>;
using Vec3u  = Vec3<uint32_t>;
using Vec3u8 = Vec3<uint8_t>;
using Vec3u16 = Vec3<uint16_t>;

using Vec4R  = Vec4<double>;
using Vec4d  = Vec4<double>;
using Vec4f  = Vec4<float>;
using Vec4i  = Vec4<int>;

}// namespace math ===============================================================

using Rgba8 [[deprecated("Use math::Rgba8 instead.")]] = math::Rgba8;
using math::Coord;

using Vec3d = math::Vec3<double>;
using Vec3f = math::Vec3<float>;
using Vec3i = math::Vec3<int32_t>;
using Vec3u = math::Vec3<uint32_t>;
using Vec3u8 = math::Vec3<uint8_t>;
using Vec3u16 = math::Vec3<uint16_t>;

using Vec4R = math::Vec4<double>;
using Vec4d = math::Vec4<double>;
using Vec4f = math::Vec4<float>;
using Vec4i = math::Vec4<int>;

using CoordBBox = math::BBox<Coord>;
using Vec3dBBox = math::BBox<Vec3d>;
using BBoxR [[deprecated("Use Vec3dBBox instead.")]] = math::BBox<Vec3d>;

} // namespace nanovdb ===================================================================

#endif // end of NANOVDB_MATH_MATH_H_HAS_BEEN_INCLUDED
