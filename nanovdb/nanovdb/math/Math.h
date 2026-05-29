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
/// Tolerance for floating-point comparison
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
/// Delta for small floating-point offsets
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
/// Maximum floating-point values
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

template<typename Type>
__hostdev__ [[nodiscard]] inline constexpr bool isApproxZero(const Type& x) noexcept
{
    return !(x > Tolerance<Type>::value()) && !(x < -Tolerance<Type>::value());
}

template<typename Type>
__hostdev__ [[nodiscard]] inline constexpr Type Min(Type a, Type b) noexcept
{
    return (a < b) ? a : b;
}
// Not constexpr — fminf/fmin aren't constexpr until C++23.
__hostdev__ [[nodiscard]] inline float Min(float a, float b) noexcept
{
    return fminf(a, b);
}
// Not constexpr — fminf/fmin aren't constexpr until C++23.
__hostdev__ [[nodiscard]] inline double Min(double a, double b) noexcept
{
    return fmin(a, b);
}
template<typename Type>
__hostdev__ [[nodiscard]] inline constexpr Type Max(Type a, Type b) noexcept
{
    return (a > b) ? a : b;
}
// Not constexpr — fmaxf/fmax aren't constexpr until C++23.
__hostdev__ [[nodiscard]] inline float Max(float a, float b) noexcept
{
    return fmaxf(a, b);
}
// Not constexpr — fmaxf/fmax aren't constexpr until C++23.
__hostdev__ [[nodiscard]] inline double Max(double a, double b) noexcept
{
    return fmax(a, b);
}
// Not constexpr — depends on non-constexpr float/double Min/Max overloads.
__hostdev__ [[nodiscard]] inline float Clamp(float x, float a, float b) noexcept
{
    return Max(Min(x, b), a);
}
// Not constexpr — depends on non-constexpr float/double Min/Max overloads.
__hostdev__ [[nodiscard]] inline double Clamp(double x, double a, double b) noexcept
{
    return Max(Min(x, b), a);
}

__hostdev__ [[nodiscard]] inline float Fract(float x) noexcept
{
    return x - floorf(x);
}
__hostdev__ [[nodiscard]] inline double Fract(double x) noexcept
{
    return x - floor(x);
}

__hostdev__ [[nodiscard]] inline int32_t Floor(float x) noexcept
{
    return int32_t(floorf(x));
}
__hostdev__ [[nodiscard]] inline int32_t Floor(double x) noexcept
{
    return int32_t(floor(x));
}

__hostdev__ [[nodiscard]] inline int32_t Ceil(float x) noexcept
{
    return int32_t(ceilf(x));
}
__hostdev__ [[nodiscard]] inline int32_t Ceil(double x) noexcept
{
    return int32_t(ceil(x));
}

template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Pow2(T x) noexcept
{
    return x * x;
}

template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Pow3(T x) noexcept
{
    return x * x * x;
}

template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Pow4(T x) noexcept
{
    return Pow2(x * x);
}
template<typename T>
__hostdev__ [[nodiscard]] inline constexpr T Abs(T x) noexcept
{
    return x < 0 ? -x : x;
}

// Not constexpr — fabsf isn't constexpr until C++23.
template<>
__hostdev__ [[nodiscard]] inline float Abs(float x) noexcept
{
    return fabsf(x);
}

// Not constexpr — fabs isn't constexpr until C++23.
template<>
__hostdev__ [[nodiscard]] inline double Abs(double x) noexcept
{
    return fabs(x);
}

// Not constexpr — std::abs(int) isn't constexpr until C++23.
template<>
__hostdev__ [[nodiscard]] inline int Abs(int x) noexcept
{
    return abs(x);
}

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

template<typename CoordT, template<typename> class Vec3T>
__hostdev__ [[nodiscard]] inline CoordT Round(const Vec3T<double>& xyz) noexcept
{
    return CoordT(int32_t(floor(xyz[0] + 0.5)),
                  int32_t(floor(xyz[1] + 0.5)),
                  int32_t(floor(xyz[2] + 0.5)));
}

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

    __hostdev__ constexpr Coord(ValueType* ptr) noexcept
        : mVec{ptr[0], ptr[1], ptr[2]}
    {
    }

    __hostdev__ constexpr int32_t x() const noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t y() const noexcept { return mVec[1]; }
    __hostdev__ constexpr int32_t z() const noexcept { return mVec[2]; }

    __hostdev__ constexpr int32_t& x() noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t& y() noexcept { return mVec[1]; }
    __hostdev__ constexpr int32_t& z() noexcept { return mVec[2]; }

    __hostdev__ [[nodiscard]] static constexpr Coord max() noexcept { return Coord(int32_t((1u << 31) - 1)); }

    __hostdev__ [[nodiscard]] static constexpr Coord min() noexcept { return Coord(-int32_t((1u << 31) - 1) - 1); }

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

    // @brief Return a new instance with coordinates left-shifted by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord operator<<(IndexType n) const noexcept { return Coord(mVec[0] << n, mVec[1] << n, mVec[2] << n); }

    // @brief Return a new instance with coordinates right-shifted by the given unsigned integer.
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

    // @brief Return true if this Coord is lexicographically greater than the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>(const Coord& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true
             : mVec[1] < rhs[1] ? false
             : mVec[2] > rhs[2] ? true : false;
    }

    // @brief Return true if this Coord is lexicographically greater or equal to the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>=(const Coord& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true
             : mVec[1] < rhs[1] ? false
             : mVec[2] >=rhs[2] ? true : false;
    }

    // @brief Return true if the Coord components are identical.
    __hostdev__ [[nodiscard]] constexpr bool   operator==(const Coord& rhs) const noexcept { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    __hostdev__ [[nodiscard]] constexpr bool   operator!=(const Coord& rhs) const noexcept { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2]; }
    __hostdev__ constexpr Coord& operator&=(int n) noexcept
    {
        mVec[0] &= n;
        mVec[1] &= n;
        mVec[2] &= n;
        return *this;
    }
    __hostdev__ constexpr Coord& operator<<=(uint32_t n) noexcept
    {
        mVec[0] <<= n;
        mVec[1] <<= n;
        mVec[2] <<= n;
        return *this;
    }
    __hostdev__ constexpr Coord& operator>>=(uint32_t n) noexcept
    {
        mVec[0] >>= n;
        mVec[1] >>= n;
        mVec[2] >>= n;
        return *this;
    }
    __hostdev__ constexpr Coord& operator+=(int n) noexcept
    {
        mVec[0] += n;
        mVec[1] += n;
        mVec[2] += n;
        return *this;
    }
    __hostdev__ [[nodiscard]] constexpr Coord  operator+(const Coord& rhs) const noexcept { return Coord(mVec[0] + rhs[0], mVec[1] + rhs[1], mVec[2] + rhs[2]); }
    __hostdev__ [[nodiscard]] constexpr Coord  operator-(const Coord& rhs) const noexcept { return Coord(mVec[0] - rhs[0], mVec[1] - rhs[1], mVec[2] - rhs[2]); }
    __hostdev__ [[nodiscard]] constexpr Coord  operator-() const noexcept { return Coord(-mVec[0], -mVec[1], -mVec[2]); }
    __hostdev__ constexpr Coord& operator+=(const Coord& rhs) noexcept
    {
        mVec[0] += rhs[0];
        mVec[1] += rhs[1];
        mVec[2] += rhs[2];
        return *this;
    }
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
    __device__ inline Coord& minComponentAtomic(const Coord& other) noexcept
    {
        atomicMin(&mVec[0], other[0]);
        atomicMin(&mVec[1], other[1]);
        atomicMin(&mVec[2], other[2]);
        return *this;
    }
    __device__ inline Coord& maxComponentAtomic(const Coord& other) noexcept
    {
        atomicMax(&mVec[0], other[0]);
        atomicMax(&mVec[1], other[1]);
        atomicMax(&mVec[2], other[2]);
        return *this;
    }
#endif

    __hostdev__ [[nodiscard]] constexpr Coord offsetBy(ValueType dx, ValueType dy, ValueType dz) const noexcept
    {
        return Coord(mVec[0] + dx, mVec[1] + dy, mVec[2] + dz);
    }

    __hostdev__ [[nodiscard]] constexpr Coord offsetBy(ValueType n) const noexcept { return this->offsetBy(n, n, n); }

    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    __hostdev__ [[nodiscard]] static inline constexpr bool lessThan(const Coord& a, const Coord& b) noexcept
    {
        return (a[0] < b[0] || a[1] < b[1] || a[2] < b[2]);
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    // Not constexpr — math::Floor uses floorf/floor which aren't constexpr until C++23.
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
    //__hostdev__ size_t octant() const { return (uint32_t(mVec[0])>>31) | ((uint32_t(mVec[1])>>31)<<1) | ((uint32_t(mVec[2])>>31)<<2); }
    __hostdev__ [[nodiscard]] constexpr uint8_t octant() const noexcept { return (uint8_t(bool(mVec[0] & (1u << 31)))) |
                                                (uint8_t(bool(mVec[1] & (1u << 31))) << 1) |
                                                (uint8_t(bool(mVec[2] & (1u << 31))) << 2); }

    /// @brief Return a single precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec3<float> asVec3s() const noexcept;

    /// @brief Return a double precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec3<double> asVec3d() const noexcept;

    // returns a copy of itself, so it mimics the behaviour of Vec3<T>::round()
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

    __hostdev__ constexpr Coord2(ValueType* ptr) noexcept
        : mVec{ptr[0], ptr[1]}
    {
    }

    __hostdev__ constexpr int32_t x() const noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t y() const noexcept { return mVec[1]; }

    __hostdev__ constexpr int32_t& x() noexcept { return mVec[0]; }
    __hostdev__ constexpr int32_t& y() noexcept { return mVec[1]; }

    __hostdev__ [[nodiscard]] static constexpr Coord2 max() noexcept { return Coord2(int32_t((1u << 31) - 1)); }

    __hostdev__ [[nodiscard]] static constexpr Coord2 min() noexcept { return Coord2(-int32_t((1u << 31) - 1) - 1); }

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

    // @brief Return a new instance with coordinates left-shifted by the given unsigned integer.
    __hostdev__ [[nodiscard]] constexpr Coord2 operator<<(IndexType n) const noexcept { return Coord2(mVec[0] << n, mVec[1] << n); }

    // @brief Return a new instance with coordinates right-shifted by the given unsigned integer.
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

    // @brief Return true if this Coord is lexicographically greater than the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>(const Coord2& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] > rhs[1] ? true : false;
    }

    // @brief Return true if this Coord is lexicographically greater or equal to the given Coord.
    __hostdev__ [[nodiscard]] constexpr bool operator>=(const Coord2& rhs) const noexcept
    {
        return mVec[0] > rhs[0] ? true
             : mVec[0] < rhs[0] ? false
             : mVec[1] >= rhs[1] ? true : false;
    }

    // @brief Return true if the Coord components are identical.
    __hostdev__ [[nodiscard]] constexpr bool   operator==(const Coord2& rhs) const noexcept { return mVec[0] == rhs[0] && mVec[1] == rhs[1]; }
    __hostdev__ [[nodiscard]] constexpr bool   operator!=(const Coord2& rhs) const noexcept { return mVec[0] != rhs[0] || mVec[1] != rhs[1]; }
    __hostdev__ constexpr Coord2& operator&=(int n) noexcept
    {
        mVec[0] &= n;
        mVec[1] &= n;
        return *this;
    }
    __hostdev__ constexpr Coord2& operator<<=(uint32_t n) noexcept
    {
        mVec[0] <<= n;
        mVec[1] <<= n;
        return *this;
    }
    __hostdev__ constexpr Coord2& operator>>=(uint32_t n) noexcept
    {
        mVec[0] >>= n;
        mVec[1] >>= n;
        return *this;
    }
    __hostdev__ constexpr Coord2& operator+=(int n) noexcept
    {
        mVec[0] += n;
        mVec[1] += n;
        return *this;
    }
    __hostdev__ [[nodiscard]] constexpr Coord2  operator+(const Coord2& rhs) const noexcept { return Coord2(mVec[0] + rhs[0], mVec[1] + rhs[1]); }
    __hostdev__ [[nodiscard]] constexpr Coord2  operator-(const Coord2& rhs) const noexcept { return Coord2(mVec[0] - rhs[0], mVec[1] - rhs[1]); }
    __hostdev__ [[nodiscard]] constexpr Coord2  operator-() const noexcept { return Coord2(-mVec[0], -mVec[1]); }
    __hostdev__ constexpr Coord2& operator+=(const Coord2& rhs) noexcept
    {
        mVec[0] += rhs[0];
        mVec[1] += rhs[1];
        return *this;
    }
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
    __device__ inline Coord2& minComponentAtomic(const Coord2& other) noexcept
    {
        atomicMin(&mVec[0], other[0]);
        atomicMin(&mVec[1], other[1]);
        return *this;
    }
    __device__ inline Coord2& maxComponentAtomic(const Coord2& other) noexcept
    {
        atomicMax(&mVec[0], other[0]);
        atomicMax(&mVec[1], other[1]);
        return *this;
    }
#endif

    __hostdev__ [[nodiscard]] constexpr Coord2 offsetBy(ValueType dx, ValueType dy) const noexcept
    {
        return Coord2(mVec[0] + dx, mVec[1] + dy);
    }

    __hostdev__ [[nodiscard]] constexpr Coord2 offsetBy(ValueType n) const noexcept { return this->offsetBy(n, n); }

    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    __hostdev__ [[nodiscard]] static inline constexpr bool lessThan(const Coord2& a, const Coord2& b) noexcept
    {
        return (a[0] < b[0] || a[1] < b[1]);
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    // Not constexpr — math::Floor uses floorf/floor which aren't constexpr until C++23.
    template<typename Vec2T>
    __hostdev__ [[nodiscard]] static Coord2 Floor(const Vec2T& xy) noexcept { return Coord2(math::Floor(xy[0]), math::Floor(xy[1])); }

    /// @brief Return a single precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec2<float> asVec2s() const noexcept;

    /// @brief Return a double precision floating-point vector of this coordinate
    __hostdev__ [[nodiscard]] inline constexpr Vec2<double> asVec2d() const noexcept;

    // returns a copy of itself, so it mimics the behaviour of Vec3<T>::round()
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

    /// @brief Named component accessors. Available based on dimensionality:
    /// x() requires N >= 1, y() requires N >= 2, z() requires N >= 3, w()
    /// requires N >= 4. A call on a Vec too small for the requested component
    /// triggers a clear compile-time static_assert at the call site.
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

    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived plus(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] + rhs[i];
        return out;
    }
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived minus(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] - rhs[i];
        return out;
    }
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived mul(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] * rhs[i];
        return out;
    }
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived div(const Derived& rhs) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] / rhs[i];
        return out;
    }
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived negate() const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = -mVec[i];
        return out;
    }
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived scale(const T& s) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] * s;
        return out;
    }
    /// @brief return (*this) / @a s element-wise as a @c Derived. Uses per-element
    /// division (correct for integer @c T, unlike multiplying by 1/s).
    template<typename Derived>
    __hostdev__ [[nodiscard]] constexpr Derived divideBy(const T& s) const noexcept {
        Derived out{};
        for (int i = 0; i < N; ++i) out[i] = mVec[i] / s;
        return out;
    }

    // ---- in-place compound assignment helpers ----

    __hostdev__ constexpr VecBase& addAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] += rhs.mVec[i];
        return *this;
    }
    __hostdev__ constexpr VecBase& subAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] -= rhs.mVec[i];
        return *this;
    }
    __hostdev__ constexpr VecBase& mulAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] *= rhs.mVec[i];
        return *this;
    }
    __hostdev__ constexpr VecBase& divAssign(const VecBase& rhs) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] /= rhs.mVec[i];
        return *this;
    }
    __hostdev__ constexpr VecBase& scaleAssign(const T& s) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] *= s;
        return *this;
    }
    __hostdev__ constexpr VecBase& divideAssignScalar(const T& s) noexcept {
        for (int i = 0; i < N; ++i) mVec[i] /= s;
        return *this;
    }

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
    __hostdev__ [[nodiscard]] constexpr T lengthSqr() const noexcept {
        T s = T(0);
        for (int i = 0; i < N; ++i) s += mVec[i] * mVec[i];
        return s;
    }
    // Not constexpr — std::sqrt isn't constexpr until C++26.
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

    template<typename V>
    __hostdev__ constexpr VecBase& mergeMin(const V& other) noexcept {
        for (int i = 0; i < N; ++i) if (other[i] < mVec[i]) mVec[i] = other[i];
        return *this;
    }
    template<typename V>
    __hostdev__ constexpr VecBase& mergeMax(const V& other) noexcept {
        for (int i = 0; i < N; ++i) if (other[i] > mVec[i]) mVec[i] = other[i];
        return *this;
    }

    // ---- integer rounding (toward -inf / +inf / nearest) ----
    //
    // We test against `std::is_floating_point<T>` rather than
    // `util::is_floating_point<T>` because the latter only matches `float`
    // and `double` whereas the former (correctly) also matches
    // `long double`. Without the broader predicate, `Vec<long double>`
    // would silently truncate via `static_cast<int32_t>` instead of using
    // the floor/ceil/round rounding rules. The inner `math::Floor` /
    // `math::Ceil` only have float/double overloads, so `long double`
    // implicitly converts to `double` at the call site — matching the
    // pre-refactor behaviour of `Vec*::round()` which routed `long
    // double` through the same `Floor(x + 0.5)` path.

    /// @brief floor-rounded components into the @c Result type (whose @c [i]
    /// must accept int32_t). For integer @c T the value is passed through.
    /// @note Only the integer-@c T specialization is usable as a constant
    /// expression; the floating-point branch calls @c math::Floor /
    /// @c math::Ceil, which aren't constexpr until C++23.
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
/// chunk (e.g. 8 bytes for Vec2<float>, 16 bytes for Vec2<double>),
/// without any tail padding because the byte size is already a power-of-2
/// multiple of alignof(T).
template<typename T>
class alignas(alignof(T) * 2) Vec2 final : public VecBase<T, 2>
{
    using Base = VecBase<T, 2>;

public:
    using ValueType = T;
    static constexpr int size = 2; // openvdb::math::Tuple-compat alias of SIZE

    Vec2() noexcept = default;
    __hostdev__ explicit constexpr Vec2(T x) noexcept            : Base(x, x) {}
    __hostdev__ constexpr Vec2(T x, T y) noexcept                : Base(x, y) {}

    template<template<class> class Vec2T, class T2>
    __hostdev__ explicit constexpr Vec2(const Vec2T<T2>& v) noexcept : Base(v[0], v[1])
    {
        static_assert(Vec2T<T2>::size == 2, "expected Vec2T::size==2!");
    }
    template<typename T2>
    __hostdev__ explicit constexpr Vec2(const Vec2<T2>& v) noexcept : Base(v[0], v[1]) {}
    __hostdev__ explicit constexpr Vec2(const Coord2& ijk) noexcept : Base(ijk[0], ijk[1]) {}

    template<template<class> class Vec2T, class T2>
    __hostdev__ constexpr Vec2& operator=(const Vec2T<T2>& rhs) noexcept {
        static_assert(Vec2T<T2>::size == 2, "expected Vec2T::size==2!");
        this->mVec[0] = rhs[0]; this->mVec[1] = rhs[1];
        return *this;
    }

    // ---- element-wise (Vec & Vec) ----
    __hostdev__ [[nodiscard]] constexpr Vec2  operator-() const noexcept            { return Base::template negate<Vec2>(); }
    __hostdev__ [[nodiscard]] constexpr Vec2  operator+(const Vec2& v) const noexcept { return Base::template plus<Vec2>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec2  operator-(const Vec2& v) const noexcept { return Base::template minus<Vec2>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec2  operator*(const Vec2& v) const noexcept { return Base::template mul<Vec2>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec2  operator/(const Vec2& v) const noexcept { return Base::template div<Vec2>(v); }
    __hostdev__ constexpr Vec2& operator+=(const Vec2& v) noexcept    { Base::addAssign(v); return *this; }
    __hostdev__ constexpr Vec2& operator-=(const Vec2& v) noexcept    { Base::subAssign(v); return *this; }

    // ---- mixed Vec2 / Coord2 ----
    __hostdev__ [[nodiscard]] constexpr Vec2  operator+(const Coord2& ijk) const noexcept { return Vec2(this->mVec[0] + ijk[0], this->mVec[1] + ijk[1]); }
    __hostdev__ [[nodiscard]] constexpr Vec2  operator-(const Coord2& ijk) const noexcept { return Vec2(this->mVec[0] - ijk[0], this->mVec[1] - ijk[1]); }
    __hostdev__ constexpr Vec2& operator+=(const Coord2& ijk) noexcept {
        this->mVec[0] += T(ijk[0]); this->mVec[1] += T(ijk[1]);
        return *this;
    }
    __hostdev__ constexpr Vec2& operator-=(const Coord2& ijk) noexcept {
        this->mVec[0] -= T(ijk[0]); this->mVec[1] -= T(ijk[1]);
        return *this;
    }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Vec2  operator*(const T& s) const noexcept  { return Base::template scale<Vec2>(s); }
    __hostdev__ [[nodiscard]] constexpr Vec2  operator/(const T& s) const noexcept  { return Base::template divideBy<Vec2>(s); }
    __hostdev__ constexpr Vec2& operator*=(const T& s) noexcept       { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Vec2& operator/=(const T& s) noexcept       { Base::divideAssignScalar(s); return *this; }
    // Not constexpr — depends on length() which calls std::sqrt.
    __hostdev__ Vec2& normalize() noexcept                  { return (*this) /= this->length(); }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Vec2& rhs) const noexcept { return Base::equals(rhs); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Vec2& rhs) const noexcept { return !Base::equals(rhs); }

    // ---- component-wise min/max ----
    __hostdev__ constexpr Vec2& minComponent(const Vec2& other) noexcept { Base::mergeMin(other); return *this; }
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
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec2 operator*(T1 scalar, const Vec2& vec) noexcept
    {
        return Vec2(scalar * vec[0], scalar * vec[1]);
    }
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


// Matrix base class
template<typename T, int ROWS, int COLS>
class MatBase {
protected:
    T mData[ROWS * COLS];  // 1D array storage

public:
    using ValueType = T;

    [[nodiscard]] static constexpr int rows() noexcept { return ROWS; }
    [[nodiscard]] static constexpr int cols() noexcept { return COLS; }
    [[nodiscard]] static constexpr int size() noexcept { return ROWS * COLS; }

    MatBase() noexcept = default;

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

    // 2D array access. Returns a row pointer; the caller is responsible
    // for the column bound (0 <= col < COLS).
    __hostdev__ constexpr T* operator[](int row) noexcept {
        NANOVDB_ASSERT(row >= 0 && row < ROWS);
        return &mData[row * COLS];
    }
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

    __hostdev__ constexpr MatBase& addAssign(const MatBase& rhs) noexcept {
        for (int i = 0; i < size(); ++i) mData[i] += rhs.mData[i];
        return *this;
    }
    __hostdev__ constexpr MatBase& subAssign(const MatBase& rhs) noexcept {
        for (int i = 0; i < size(); ++i) mData[i] -= rhs.mData[i];
        return *this;
    }
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
    __hostdev__ constexpr MatBase& divideAssignScalar(const T& s) noexcept {
        for (int i = 0; i < size(); ++i) mData[i] /= s;
        return *this;
    }

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



// Aligned to 4*alignof(T) — Mat2 stores 4 elements (2x2), which is already
// a power-of-2 multiple of alignof(T), so this is free in size and gives
// SIMD-friendly placement (16 bytes for Mat2<float>, 32 bytes for double).
template <typename T>
class alignas(alignof(T) * 4) Mat2 final : public MatBase<T, 2, 2> {
    using Base = MatBase<T, 2, 2>;
public:
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

// Mat2x3 is intentionally NOT alignas-elevated: its byte size
// (6*sizeof(T)) is not a power-of-2 multiple of alignof(T), so any
// alignas(N > alignof(T)) would force tail padding and break
// packed-array layout plus on-disk format compatibility.
template <typename T>
class Mat2x3 final : public MatBase<T, 2, 3> {
    using Base = MatBase<T, 2, 3>;
public:
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

// Mat3x2 is intentionally NOT alignas-elevated: its byte size
// (6*sizeof(T)) is not a power-of-2 multiple of alignof(T), so any
// alignas(N > alignof(T)) would force tail padding and break
// packed-array layout plus on-disk format compatibility.
template <typename T>
class Mat3x2 final : public MatBase<T, 3, 2> {
    using Base = MatBase<T, 3, 2>;
public:
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

// Mat3 is intentionally NOT alignas-elevated: its byte size
// (9*sizeof(T)) is not a power-of-2 multiple of alignof(T), so any
// alignas(N > alignof(T)) would force tail padding and break
// packed-array layout plus on-disk format compatibility.
template <typename T>
class Mat3 final : public MatBase<T, 3, 3> {
    using Base = MatBase<T, 3, 3>;
public:
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

// Aligned to 16*alignof(T) — Mat4 stores 16 elements (4x4), which is
// already a power-of-2 multiple of alignof(T), so the alignment is free
// in size. Whole-matrix alignment (64 bytes for Mat4<float>, 128 bytes
// for Mat4<double>) is heavy compared to row-alignment (4*alignof(T))
// but matches the per-class "align to full size" rule used for Vec2 /
// Vec4 / Mat2 above and lets a Mat4<float> load with a single AVX-512
// instruction. Drop to alignas(alignof(T) * 4) here if the over-aligned
// constraint causes friction with downstream allocators.
template <typename T>
class alignas(alignof(T) * 16) Mat4 final : public MatBase<T, 4, 4> {
    using Base = MatBase<T, 4, 4>;
public:
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
/// packed-array layout plus on-disk format compatibility. An opt-in
/// over-aligned Vec3 wrapper (e.g. for CUDA-coalesced loads) is best
/// added as a separate type rather than changing Vec3 itself.
template<typename T>
class Vec3 final : public VecBase<T, 3>
{
    using Base = VecBase<T, 3>;

public:
    using ValueType = T;
    static constexpr int size = 3; // openvdb::math::Tuple-compat alias of SIZE

    Vec3() noexcept = default;
    __hostdev__ explicit constexpr Vec3(T x) noexcept            : Base(x, x, x) {}
    __hostdev__ constexpr Vec3(T x, T y, T z) noexcept           : Base(x, y, z) {}

    template<template<class> class Vec3T, class T2>
    __hostdev__ explicit constexpr Vec3(const Vec3T<T2>& v) noexcept : Base(v[0], v[1], v[2])
    {
        static_assert(Vec3T<T2>::size == 3, "expected Vec3T::size==3!");
    }
    template<typename T2>
    __hostdev__ explicit constexpr Vec3(const Vec3<T2>& v) noexcept : Base(v[0], v[1], v[2]) {}
    __hostdev__ explicit constexpr Vec3(const Coord& ijk) noexcept : Base(ijk[0], ijk[1], ijk[2]) {}

    template<template<class> class Vec3T, class T2>
    __hostdev__ constexpr Vec3& operator=(const Vec3T<T2>& rhs) noexcept {
        static_assert(Vec3T<T2>::size == 3, "expected Vec3T::size==3!");
        this->mVec[0] = rhs[0]; this->mVec[1] = rhs[1]; this->mVec[2] = rhs[2];
        return *this;
    }

    // ---- element-wise (Vec & Vec) ----
    __hostdev__ [[nodiscard]] constexpr Vec3  operator-() const noexcept             { return Base::template negate<Vec3>(); }
    __hostdev__ [[nodiscard]] constexpr Vec3  operator+(const Vec3& v) const noexcept { return Base::template plus<Vec3>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec3  operator-(const Vec3& v) const noexcept { return Base::template minus<Vec3>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec3  operator*(const Vec3& v) const noexcept { return Base::template mul<Vec3>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec3  operator/(const Vec3& v) const noexcept { return Base::template div<Vec3>(v); }
    __hostdev__ constexpr Vec3& operator+=(const Vec3& v) noexcept     { Base::addAssign(v); return *this; }
    __hostdev__ constexpr Vec3& operator-=(const Vec3& v) noexcept     { Base::subAssign(v); return *this; }

    // ---- mixed Vec3 / Coord (3D) ----
    __hostdev__ [[nodiscard]] constexpr Vec3  operator+(const Coord& ijk) const noexcept { return Vec3(this->mVec[0] + ijk[0], this->mVec[1] + ijk[1], this->mVec[2] + ijk[2]); }
    __hostdev__ [[nodiscard]] constexpr Vec3  operator-(const Coord& ijk) const noexcept { return Vec3(this->mVec[0] - ijk[0], this->mVec[1] - ijk[1], this->mVec[2] - ijk[2]); }
    __hostdev__ constexpr Vec3& operator+=(const Coord& ijk) noexcept {
        this->mVec[0] += T(ijk[0]); this->mVec[1] += T(ijk[1]); this->mVec[2] += T(ijk[2]);
        return *this;
    }
    __hostdev__ constexpr Vec3& operator-=(const Coord& ijk) noexcept {
        this->mVec[0] -= T(ijk[0]); this->mVec[1] -= T(ijk[1]); this->mVec[2] -= T(ijk[2]);
        return *this;
    }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Vec3  operator*(const T& s) const noexcept   { return Base::template scale<Vec3>(s); }
    __hostdev__ [[nodiscard]] constexpr Vec3  operator/(const T& s) const noexcept   { return Base::template divideBy<Vec3>(s); }
    __hostdev__ constexpr Vec3& operator*=(const T& s) noexcept        { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Vec3& operator/=(const T& s) noexcept        { Base::divideAssignScalar(s); return *this; }
    // Not constexpr — depends on length() which calls std::sqrt.
    __hostdev__ Vec3& normalize() noexcept                   { return (*this) /= this->length(); }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Vec3& rhs) const noexcept { return Base::equals(rhs); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Vec3& rhs) const noexcept { return !Base::equals(rhs); }

    // ---- component-wise min/max ----
    __hostdev__ constexpr Vec3& minComponent(const Vec3& other) noexcept { Base::mergeMin(other); return *this; }
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
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec3 operator*(T1 scalar, const Vec3& vec) noexcept
    {
        return Vec3(scalar * vec[0], scalar * vec[1], scalar * vec[2]);
    }
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

    Vec4() noexcept = default;
    __hostdev__ explicit constexpr Vec4(T x) noexcept            : Base(x, x, x, x) {}
    __hostdev__ constexpr Vec4(T x, T y, T z, T w) noexcept      : Base(x, y, z, w) {}

    template<typename T2>
    __hostdev__ explicit constexpr Vec4(const Vec4<T2>& v) noexcept : Base(v[0], v[1], v[2], v[3]) {}
    template<template<class> class Vec4T, class T2>
    __hostdev__ explicit constexpr Vec4(const Vec4T<T2>& v) noexcept : Base(v[0], v[1], v[2], v[3])
    {
        static_assert(Vec4T<T2>::size == 4, "expected Vec4T::size==4!");
    }
    template<template<class> class Vec4T, class T2>
    __hostdev__ constexpr Vec4& operator=(const Vec4T<T2>& rhs) noexcept {
        static_assert(Vec4T<T2>::size == 4, "expected Vec4T::size==4!");
        this->mVec[0] = rhs[0]; this->mVec[1] = rhs[1]; this->mVec[2] = rhs[2]; this->mVec[3] = rhs[3];
        return *this;
    }

    // ---- element-wise (Vec & Vec) ----
    __hostdev__ [[nodiscard]] constexpr Vec4  operator-() const noexcept             { return Base::template negate<Vec4>(); }
    __hostdev__ [[nodiscard]] constexpr Vec4  operator+(const Vec4& v) const noexcept { return Base::template plus<Vec4>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec4  operator-(const Vec4& v) const noexcept { return Base::template minus<Vec4>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec4  operator*(const Vec4& v) const noexcept { return Base::template mul<Vec4>(v); }
    __hostdev__ [[nodiscard]] constexpr Vec4  operator/(const Vec4& v) const noexcept { return Base::template div<Vec4>(v); }
    __hostdev__ constexpr Vec4& operator+=(const Vec4& v) noexcept     { Base::addAssign(v); return *this; }
    __hostdev__ constexpr Vec4& operator-=(const Vec4& v) noexcept     { Base::subAssign(v); return *this; }

    // ---- scalar ----
    __hostdev__ [[nodiscard]] constexpr Vec4  operator*(const T& s) const noexcept   { return Base::template scale<Vec4>(s); }
    __hostdev__ [[nodiscard]] constexpr Vec4  operator/(const T& s) const noexcept   { return Base::template divideBy<Vec4>(s); }
    __hostdev__ constexpr Vec4& operator*=(const T& s) noexcept        { Base::scaleAssign(s); return *this; }
    __hostdev__ constexpr Vec4& operator/=(const T& s) noexcept        { Base::divideAssignScalar(s); return *this; }
    // Not constexpr — depends on length() which calls std::sqrt.
    __hostdev__ Vec4& normalize() noexcept                   { return (*this) /= this->length(); }

    // ---- equality ----
    __hostdev__ [[nodiscard]] constexpr bool operator==(const Vec4& rhs) const noexcept { return Base::equals(rhs); }
    __hostdev__ [[nodiscard]] constexpr bool operator!=(const Vec4& rhs) const noexcept { return !Base::equals(rhs); }

    // ---- component-wise min/max ----
    __hostdev__ constexpr Vec4& minComponent(const Vec4& other) noexcept { Base::mergeMin(other); return *this; }
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
    template<typename T1>
    __hostdev__ [[nodiscard]] friend constexpr Vec4 operator*(T1 scalar, const Vec4& vec) noexcept
    {
        return Vec4(scalar * vec[0], scalar * vec[1], scalar * vec[2], scalar * vec[3]);
    }
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

// Base-class for static polymorphism (cannot be constructed directly)
template<typename Vec3T>
struct BaseBBox
{
    Vec3T                    mCoord[2];
    __hostdev__ [[nodiscard]] constexpr bool         operator==(const BaseBBox& rhs) const noexcept { return mCoord[0] == rhs.mCoord[0] && mCoord[1] == rhs.mCoord[1]; };
    __hostdev__ [[nodiscard]] constexpr bool         operator!=(const BaseBBox& rhs) const noexcept { return mCoord[0] != rhs.mCoord[0] || mCoord[1] != rhs.mCoord[1]; };
    __hostdev__ constexpr const Vec3T& operator[](int i) const noexcept { NANOVDB_ASSERT(i >= 0 && i < 2); return mCoord[i]; }
    __hostdev__ constexpr Vec3T&       operator[](int i) noexcept { NANOVDB_ASSERT(i >= 0 && i < 2); return mCoord[i]; }
    __hostdev__ constexpr Vec3T&       min() noexcept { return mCoord[0]; }
    __hostdev__ constexpr Vec3T&       max() noexcept { return mCoord[1]; }
    __hostdev__ constexpr const Vec3T& min() const noexcept { return mCoord[0]; }
    __hostdev__ constexpr const Vec3T& max() const noexcept { return mCoord[1]; }
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

//    __hostdev__ BaseBBox expandBy(typename Vec3T::ValueType padding) const
//    {
//        return BaseBBox(mCoord[0].offsetBy(-padding),mCoord[1].offsetBy(padding));
//    }
    __hostdev__ [[nodiscard]] constexpr bool isInside(const Vec3T& xyz) const noexcept
    {
        if (xyz[0] < mCoord[0][0] || xyz[1] < mCoord[0][1] || xyz[2] < mCoord[0][2])
            return false;
        if (xyz[0] > mCoord[1][0] || xyz[1] > mCoord[1][1] || xyz[2] > mCoord[1][2])
            return false;
        return true;
    }

protected:
    __hostdev__ constexpr BaseBBox() noexcept {}
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
    __hostdev__ constexpr BBox(const Vec3T& min, const Vec3T& max) noexcept
        : BaseT(min, max)
    {
    }
    __hostdev__ constexpr BBox(const Coord& min, const Coord& max) noexcept
        : BaseT(Vec3T(ValueType(min[0]), ValueType(min[1]), ValueType(min[2])),
                Vec3T(ValueType(max[0] + 1), ValueType(max[1] + 1), ValueType(max[2] + 1)))
    {
    }
    __hostdev__ [[nodiscard]] static constexpr BBox createCube(const Coord& min, typename Coord::ValueType dim) noexcept
    {
        return BBox(min, min.offsetBy(dim));
    }

    __hostdev__ constexpr BBox(const BaseBBox<Coord>& bbox) noexcept
        : BBox(bbox[0], bbox[1])
    {
    }
    __hostdev__ [[nodiscard]] constexpr bool  empty() const noexcept { return mCoord[0][0] >= mCoord[1][0] ||
                                             mCoord[0][1] >= mCoord[1][1] ||
                                             mCoord[0][2] >= mCoord[1][2]; }
    __hostdev__ [[nodiscard]] constexpr operator bool() const noexcept { return mCoord[0][0] < mCoord[1][0] &&
                                               mCoord[0][1] < mCoord[1][1] &&
                                               mCoord[0][2] < mCoord[1][2]; }
    __hostdev__ [[nodiscard]] constexpr Vec3T dim() const noexcept { return *this ? this->max() - this->min() : Vec3T(0); }
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
        __hostdev__ constexpr Iterator(const BBox& b) noexcept
            : mBBox(b)
            , mPos(b.min())
        {
        }
        __hostdev__ constexpr Iterator(const BBox& b, const Coord& p) noexcept
            : mBBox(b)
            , mPos(p)
        {
        }
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
        __hostdev__ constexpr Iterator operator++(int) noexcept
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        __hostdev__ [[nodiscard]] constexpr bool operator==(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos == rhs.mPos;
        }
        __hostdev__ [[nodiscard]] constexpr bool operator!=(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos != rhs.mPos;
        }
        __hostdev__ [[nodiscard]] constexpr bool operator<(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos < rhs.mPos;
        }
        __hostdev__ [[nodiscard]] constexpr bool operator<=(const Iterator& rhs) const noexcept
        {
            NANOVDB_ASSERT(mBBox == rhs.mBBox);
            return mPos <= rhs.mPos;
        }
        /// @brief Return @c true if the iterator still points to a valid coordinate.
        __hostdev__ [[nodiscard]] constexpr operator bool() const noexcept { return mPos <= mBBox[1]; }
        __hostdev__ [[nodiscard]] constexpr const CoordT& operator*() const noexcept { return mPos; }
    }; // Iterator
    __hostdev__ [[nodiscard]] constexpr Iterator begin() const noexcept { return Iterator{*this}; }
    __hostdev__ [[nodiscard]] constexpr Iterator end()   const noexcept { return Iterator{*this, CoordT(mCoord[1][0]+1, mCoord[0][1], mCoord[0][2])}; }
    __hostdev__ constexpr BBox() noexcept
        : BaseT(CoordT::max(), CoordT::min())
    {
    }
    __hostdev__ constexpr BBox(const CoordT& min, const CoordT& max) noexcept
        : BaseT(min, max)
    {
    }

    template<typename SplitT>
    __hostdev__ constexpr BBox(BBox& other, const SplitT&) noexcept
        : BaseT(other.mCoord[0], other.mCoord[1])
    {
        NANOVDB_ASSERT(this->is_divisible());
        const int n = MaxIndex(this->dim());
        mCoord[1][n] = (mCoord[0][n] + mCoord[1][n]) >> 1;
        other.mCoord[0][n] = mCoord[1][n] + 1;
    }

    __hostdev__ [[nodiscard]] static constexpr BBox createCube(const CoordT& min, typename CoordT::ValueType dim) noexcept
    {
        return BBox(min, min.offsetBy(dim - 1));
    }

    __hostdev__ [[nodiscard]] static constexpr BBox createCube(typename CoordT::ValueType min, typename CoordT::ValueType max) noexcept
    {
        return BBox(CoordT(min), CoordT(max));
    }

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
    __hostdev__ [[nodiscard]] constexpr CoordT   dim() const noexcept { return *this ? this->max() - this->min() + Coord(1) : Coord(0); }
    __hostdev__ [[nodiscard]] constexpr uint64_t volume() const noexcept
    {
        auto d = this->dim();
        return uint64_t(d[0]) * uint64_t(d[1]) * uint64_t(d[2]);
    }
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

    /// @brief  @brief transform this coordinate bounding box by the specified map
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
    __device__ inline BBox& expandAtomic(const CoordT& ijk) noexcept
    {
        mCoord[0].minComponentAtomic(ijk);
        mCoord[1].maxComponentAtomic(ijk);
        return *this;
    }
    __device__ inline BBox& expandAtomic(const BBox& bbox) noexcept
    {
        mCoord[0].minComponentAtomic(bbox[0]);
        mCoord[1].maxComponentAtomic(bbox[1]);
        return *this;
    }
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

    /// @brief  @brief ctor where all channels are initialized to the same value
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

    /// @brief Vec3f r,g,b ctor (alpha channel it set to 1)
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ constexpr Rgba8(const Vec3<float>& rgb) noexcept
        : Rgba8(rgb[0], rgb[1], rgb[2])
    {
    }

    /// @brief Vec4f r,g,b,a ctor
    /// @note all values should be in the range 0.0f to 1.0f
    __hostdev__ constexpr Rgba8(const Vec4<float>& rgba) noexcept
        : Rgba8(rgba[0], rgba[1], rgba[2], rgba[3])
    {
    }

    __hostdev__ [[nodiscard]] bool  operator< (const Rgba8& rhs) const noexcept { return mData.packed < rhs.mData.packed; }
    __hostdev__ [[nodiscard]] bool  operator==(const Rgba8& rhs) const noexcept { return mData.packed == rhs.mData.packed; }
    __hostdev__ [[nodiscard]] constexpr float lengthSqr() const noexcept
    {
        return 0.0000153787005f * (float(mData.c[0]) * mData.c[0] +
                                   float(mData.c[1]) * mData.c[1] +
                                   float(mData.c[2]) * mData.c[2]); //1/255^2
    }
    __hostdev__ [[nodiscard]] float           length() const noexcept { return sqrtf(this->lengthSqr()); }
    /// @brief return n'th color channel as a float in the range 0 to 1
    __hostdev__ [[nodiscard]] constexpr float           asFloat(int n) const noexcept { return 0.003921569f*float(mData.c[n]); }// divide by 255
    __hostdev__ constexpr const uint8_t&  operator[](int n) const noexcept { NANOVDB_ASSERT(n >= 0 && n < 4); return mData.c[n]; }
    __hostdev__ constexpr uint8_t&        operator[](int n) noexcept { NANOVDB_ASSERT(n >= 0 && n < 4); return mData.c[n]; }
    __hostdev__ const uint32_t& packed() const noexcept { return mData.packed; }
    __hostdev__ uint32_t&       packed() noexcept { return mData.packed; }
    __hostdev__ constexpr const uint8_t&  r() const noexcept { return mData.c[0]; }
    __hostdev__ constexpr const uint8_t&  g() const noexcept { return mData.c[1]; }
    __hostdev__ constexpr const uint8_t&  b() const noexcept { return mData.c[2]; }
    __hostdev__ constexpr const uint8_t&  a() const noexcept { return mData.c[3]; }
    __hostdev__ constexpr uint8_t&        r() noexcept { return mData.c[0]; }
    __hostdev__ constexpr uint8_t&        g() noexcept { return mData.c[1]; }
    __hostdev__ constexpr uint8_t&        b() noexcept { return mData.c[2]; }
    __hostdev__ constexpr uint8_t&        a() noexcept { return mData.c[3]; }
    __hostdev__ [[nodiscard]] constexpr           operator Vec3<float>() const noexcept {
        return Vec3<float>(this->asFloat(0), this->asFloat(1), this->asFloat(2));
    }
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
