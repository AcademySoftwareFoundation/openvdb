// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   NanoVDB.h

    \author Ken Museth

    \date  January 8, 2020

    \brief Implements a light-weight self-contained VDB data-structure in a
           single file! In other words, this is a significantly watered-down
           version of the OpenVDB implementation, with few dependencies - so
           a one-stop-shop for a minimalistic VDB data structure that run on
           most platforms!

    \note It is important to note that NanoVDB (by design) is a read-only
          sparse GPU (and CPU) friendly data structure intended for applications
          like rendering and collision detection. As such it obviously lacks
          a lot of the functionalities and features of OpenVDB grids. NanoVDB
          is essentially a compact linearized (or serialized) representation of
          an openvdb tree with getValue methods only. For best performance use
          the ReadAccessor::getValue method as opposed to the Tree::getValue
          method. Note that since a ReadAccessor caches previous access patterns
          it is by design not thread-safe, so use one instantiate per thread
          (it is very lightweight). Also, it is not safe to copy accessors between
          the GPU and CPU! In fact, client code should only interface
          with the API of the Grid class (all other nodes of the NanoVDB data
          structure can safely be ignored by most client codes)!


    \warning NanoVDB grids can only be constructed from with tools like openToNanoVDB
             or the GridBuilder. This explains why none of the grid nodes defined below 
             have public constructors or destructors.

    \details Please see the following paper for more details on the data structure:
          K. Museth, “VDB: High-Resolution Sparse Volumes with Dynamic Topology”,
          ACM Transactions on Graphics 32(3), 2013, which can be found here:
          http://www.museth.org/Ken/Publications_files/Museth_TOG13.pdf


    Overview: This file implements the following fundamental class that when combined
          forms the backbone of the VDB tree data structure:

          Vec3 - a 3D vector
          Mask - a bitmask essential to the non-root tree nodes
          Map  - an affine coordinate transformation
          Grid - contains a Tree and a ma for world<->index transformations. Use
                 this class as the main API with client code!
          Tree - contains a RootNode and getValue methods that should only be used for debugging
          RootNode - the top-level node of the VDB data structure
          InternalNode - the internal nodes of the VDB data structure
          LeafNode - the lowest level tree nodes that encode voxel values and state
          ReadAccessor - implements accelerated random access operations

    Semantics: A VDB data structure encodes values and (binary) states associated with
          signed integer coordinates. Values encoded at the leaf node level are
          denoted voxel values, and values associated with other tree nodes are referred
          to as tile values, which by design cover a larger coordinate index domain.
*/

#ifndef NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED

#define NANOVDB_MAGIC_NUMBER 0x304244566f6e614eUL// "NanoVDB0" in hex - little endian (uint64_t)

#define NANOVDB_MAJOR_VERSION_NUMBER 21
#define NANOVDB_MINOR_VERSION_NUMBER 0

// This replaces a Coord key at the root level with a single uint64_t
#define USE_SINGLE_ROOT_KEY

// This replaces three levels of Coord keys in the ReadAccessor with one Coord
#define USE_SINGLE_ACCESSOR_KEY

#define NANOVDB_DATA_ALIGNMENT 32

// Helper macros for defining memory alignment
#if !defined(NANOVDB_ALIGN)

#if defined(__CUDACC__)
#define NANOVDB_ALIGN(n) __align__(n)
#else
#if __cplusplus >= 201103L
#define NANOVDB_ALIGN(n) alignas(n)
#else /* !(__cplusplus >= 201103L)*/
#if defined(__GNUC__)
#define NANOVDB_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define NANOVDB_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition of NANOVDB_ALIGN for your host compiler!"
#endif
#endif
#endif

#endif // !defined(NANOVDB_ALIGN)

#ifdef __CUDACC_RTC__

typedef signed char         int8_t;
typedef short               int16_t;
typedef int                 int32_t;
typedef long long           int64_t;
typedef unsigned char       uint8_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

#else

#include <stdlib.h> //    for abs in clang7
#include <stdint.h> //    for types like int32_t etc
#include <stddef.h> //    for size_t type
#include <cassert> //     for assert
#include <cmath> //       for sqrt and fma
#include <limits> //      for numeric_limits 
#endif

#ifdef __CUDACC__
// Only define __hostdev__ when using NVIDIA CUDA compiler
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif

namespace nanovdb {

// Implementation of std::is_same
template <typename T1, typename T2>
struct is_same
{
    static const bool value = false;
};

template <typename T>
struct is_same<T, T>
{
    static const bool value = true;
};

template <typename T>
struct is_floating_point
{
    static const bool value = is_same<T,float>::value || is_same<T,double>::value;
};

// Dummy type for a voxel with a binary mask value, e.g. the active state
class ValueMask
{
};

/// @brief List of types that are currently supported by NanoVDB
///
/// @note To expand on this list do:
///       1) Add the new type between Unknown and End in the enum below
///       2) Add the new type to Serializer::processGrid that maps openvdb types to GridType
///       3) Verify that the Deserializer::ConvertTrait works correctly with the new type
///       4) Add the new type to GridHandle::map that maps nanovdb types to GridType
///       5) Optionally add the new type to mapToStr in cmd/nanovdb_print.cpp
enum class GridType : uint32_t { Unknown = 0,
                                 Float = 1,
                                 Double = 2,
                                 Int16 = 3,
                                 Int32 = 4,
                                 Int64 = 5,
                                 Vec3f = 6,
                                 Vec3d = 7,
                                 Mask = 8,
                                 FP16 = 9,
                                 UInt32 = 10,
                                 End = 11 };

/// @brief Classes (defined in OpenVDB) that are currently supported by NanoVDB
enum class GridClass : uint32_t { Unknown = 0,
                                  LevelSet = 1,
                                  FogVolume = 2,
                                  Staggered = 3,
                                  PointIndex = 4,
                                  PointData = 5,
                                  End = 6 };

/// @brief Blind-data Classes that are currently supported by NanoVDB
enum class GridBlindDataClass : uint32_t { Unknown = 0,
                                           IndexArray = 1,
                                           AttributeArray = 2,
                                           End = 3 };

/// @brief Blind-data Semantics that are currently understood by NanoVDB
enum class GridBlindDataSemantic : uint32_t { Unknown = 0,
                                              PointPosition = 1,
                                              PointColor = 2,
                                              PointNormal = 3,
                                              PointRadius = 4,
                                              PointVelocity = 5,
                                              PointId = 6,
                                              End = 7 };

// ----------------------------> Various math functions <-------------------------------------

//@{
/// Tolerance for floating-point comparison
template<typename T>
struct Tolerance;
template<>
struct Tolerance<float>
{
    __hostdev__ static float value() { return 1e-8f; }
};
template<>
struct Tolerance<double>
{
    __hostdev__ static double value() { return 1e-15; }
};
//@}

//@{
/// Delta for small floating-point offsets
template<typename T>
struct Delta;
template<>
struct Delta<float>
{
    __hostdev__ static float value() { return 1e-5f; }
};
template<>
struct Delta<double>
{
    __hostdev__ static double value() { return 1e-9; }
};
//@}

//@{
/// Maximum floating-point values
template<typename T>
struct Maximum;
#ifdef __CUDA_ARCH__
template<>
struct Maximum<int>
{
    __hostdev__ static int value() { return 2147483647; }
};
template<>
struct Maximum<float>
{
    __hostdev__ static float value() { return 1e+38f; }
};
template<>
struct Maximum<double>
{
    __hostdev__ static double value() { return 1e+308; }
};
#else
template<typename T>
struct Maximum
{
    static T value() { return std::numeric_limits<T>::max(); }
};
#endif
//@}

template<typename Type>
__hostdev__ inline bool isApproxZero(const Type& x)
{
    return !(x > Tolerance<Type>::value()) && !(x < -Tolerance<Type>::value());
}

template<typename Type>
__hostdev__ inline Type Min(Type a, Type b)
{
    return (a < b) ? a : b;
}
__hostdev__ inline int32_t Min(int32_t a, int32_t b)
{
    return int32_t(fminf(float(a), float(b)));
}
__hostdev__ inline uint32_t Min(uint32_t a, uint32_t b)
{
    return uint32_t(fminf(float(a), float(b)));
}
__hostdev__ inline float Min(float a, float b)
{
    return fminf(a, b);
}
__hostdev__ inline double Min(double a, double b)
{
    return fmin(a, b);
}
template<typename Type>
__hostdev__ inline Type Max(Type a, Type b)
{
    return (a > b) ? a : b;
}

__hostdev__ inline int32_t Max(int32_t a, int32_t b)
{
    return int32_t(fmaxf(float(a), float(b)));
}
__hostdev__ inline uint32_t Max(uint32_t a, uint32_t b)
{
    return uint32_t(fmaxf(float(a), float(b)));
}
__hostdev__ inline float Max(float a, float b)
{
    return fmaxf(a, b);
}
__hostdev__ inline double Max(double a, double b)
{
    return fmax(a, b);
}
__hostdev__ inline float Clamp(float x, float a, float b)
{
    return Max(Min(x, b), a);
}
__hostdev__ inline double Clamp(double x, double a, double b)
{
    return Max(Min(x, b), a);
}

__hostdev__ inline float Fract(float x)
{
    return x - floorf(x);
}
__hostdev__ inline double Fract(double x)
{
    return x - floor(x);
}

__hostdev__ inline int32_t Floor(float x)
{
    return int32_t(floorf(x));
}
__hostdev__ inline int32_t Floor(double x)
{
    return int32_t(floor(x));
}

__hostdev__ inline int32_t Ceil(float x)
{
    return int32_t(ceilf(x));
}
__hostdev__ inline int32_t Ceil(double x)
{
    return int32_t(ceil(x));
}

template <typename T>
__hostdev__ inline T Pow2(T x)
{
    return x * x;
}

template <typename T>
__hostdev__ inline T Abs(T x)
{
    return x < 0 ? -x : x;
}

template<>
__hostdev__ inline float Abs(float x)
{
    return fabs(x);
}

template<>
__hostdev__ inline double Abs(double x)
{
    return fabs(x);
}

template<>
__hostdev__ inline int Abs(int x)
{
    return abs(x);
}

template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<RealT>& xyz);

template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<float>& xyz)
{
    return CoordT(int32_t(rintf(xyz[0])), int32_t(rintf(xyz[1])), int32_t(rintf(xyz[2])));
    //return CoordT(int32_t(roundf(xyz[0])), int32_t(roundf(xyz[1])), int32_t(roundf(xyz[2])) );
    //return CoordT(int32_t(floorf(xyz[0] + 0.5f)), int32_t(floorf(xyz[1] + 0.5f)), int32_t(floorf(xyz[2] + 0.5f)));
}

template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<double>& xyz)
{
    return CoordT(int32_t(floor(xyz[0] + 0.5)), int32_t(floor(xyz[1] + 0.5)), int32_t(floor(xyz[2] + 0.5)));
}

template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ inline CoordT RoundDown(const Vec3T<RealT>& xyz)
{
    return CoordT(Floor(xyz[0]), Floor(xyz[1]), Floor(xyz[2]));
}

//@{
/// Return the square root of a floating-point value.
inline __hostdev__ float Sqrt(float x)
{
    return sqrtf(x);
}
inline __hostdev__ double Sqrt(double x)
{
    return sqrt(x);
}
//@}

template<typename Vec3T>
__hostdev__ inline int MinIndex(const Vec3T& v)
{
#if 1
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] < v[1]) << 2) + ((v[0] < v[2]) << 1) + (v[1] < v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
#else
    if (v[0] < v[1] && v[0] < v[2]) return 0;
    if (v[1] < v[2]) return 1;
    else return 2;
#endif
}

template<typename Vec3T>
__hostdev__ inline int MaxIndex(const Vec3T& v)
{
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] > v[1]) << 2) + ((v[0] > v[2]) << 1) + (v[1] > v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
}

// round up byteSize to the nearest wordSize. E.g. to align to machine word: AlignUp<sizeof(size_t)(n)
template<uint64_t wordSize>
__hostdev__ inline uint64_t AlignUp(uint64_t byteCount)
{
    const uint64_t r = byteCount % wordSize;
    return r ? byteCount - r + wordSize : byteCount;
}

// ------------------------------> Coord <--------------------------------------

/// @brief Signed (i, j, k) 32-bit integer coordinate class, simular to openvdb::math::Coord
class Coord
{
    int32_t mVec[3]; // private member data - three signed index coordinates
public:
    using ValueType = int32_t;
    using IndexType = uint32_t;

    /// @brief Initialize all coordinates to zero.
    __hostdev__ Coord()
        : mVec{0, 0, 0}
    {
    }

    /// @brief Initializes all coordinates to the given signed integer.
    __hostdev__ explicit Coord(ValueType n)
        : mVec{n, n, n}
    {
    }

    /// @brief Initializes coordinate to the given signed integers.
    __hostdev__ Coord(ValueType i, ValueType j, ValueType k)
        : mVec{i, j, k}
    {
    }

    __hostdev__ int32_t x() const { return mVec[0]; }
    __hostdev__ int32_t y() const { return mVec[1]; }
    __hostdev__ int32_t z() const { return mVec[2]; }

    __hostdev__ int32_t& x() { return mVec[0]; }
    __hostdev__ int32_t& y() { return mVec[1]; }
    __hostdev__ int32_t& z() { return mVec[2]; }

    __hostdev__ static Coord max() { return Coord(int32_t((1u << 31) - 1)); }

    __hostdev__ static Coord min() { return Coord(-int32_t((1u << 31) - 1)-1); }

    __hostdev__ static size_t memUsage() { return sizeof(Coord); }

    /// @brief Return a const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ const ValueType& operator[](IndexType i) const { return mVec[i]; }

    /// @brief Return a non-const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ ValueType& operator[](IndexType i) { return mVec[i]; }

    /// @brief Return a new instance with coordinates masked by the given unsigned integer.
    __hostdev__ Coord operator&(IndexType n) const { return Coord(mVec[0] & n, mVec[1] & n, mVec[2] & n); }

    // @brief Return a new instance with coordinates left-shifted by the given unsigned integer.
    __hostdev__ Coord operator<<(IndexType n) const { return Coord(mVec[0] << n, mVec[1] << n, mVec[2] << n); }

    // @brief Return a new instance with coordinates right-shifted by the given unsigned integer.
    __hostdev__ Coord operator>>(IndexType n) const { return Coord(mVec[0] >> n, mVec[1] >> n, mVec[2] >> n); }

    /// @brief Return true is this Coord is Lexicographiclly less than the given Coord.
    __hostdev__ bool operator<(const Coord& rhs) const
    {
        return mVec[0] < rhs[0] ? true : mVec[0] > rhs[0] ? false : mVec[1] < rhs[1] ? true : mVec[1] > rhs[1] ? false : mVec[2] < rhs[2] ? true : false;
    }

    // @brief Return true if the Coord components are identical.
    __hostdev__ bool operator==(const Coord& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    __hostdev__ bool operator!=(const Coord& rhs) const { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2]; }
    __hostdev__ Coord& operator&=(int n)
    {
        mVec[0] &= n;
        mVec[1] &= n;
        mVec[2] &= n;
        return *this;
    }
    __hostdev__ Coord& operator<<=(uint32_t n)
    {
        mVec[0] <<= n;
        mVec[1] <<= n;
        mVec[2] <<= n;
        return *this;
    }
    __hostdev__ Coord& operator+=(int n)
    {
        mVec[0] += n;
        mVec[1] += n;
        mVec[2] += n;
        return *this;
    }
    __hostdev__ Coord operator+(const Coord& rhs) const { return Coord(mVec[0] + rhs[0], mVec[1] + rhs[1], mVec[2] + rhs[2]); }
    __hostdev__ Coord operator-(const Coord& rhs) const { return Coord(mVec[0] - rhs[0], mVec[1] - rhs[1], mVec[2] - rhs[2]); }
    __hostdev__ Coord& operator+=(const Coord& rhs)
    {
        mVec[0] += rhs[0];
        mVec[1] += rhs[1];
        mVec[2] += rhs[2];
        return *this;
    }

    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ Coord& minComponent(const Coord& other)
    {
        if (other[0] < mVec[0]) mVec[0] = other[0];
        if (other[1] < mVec[1]) mVec[1] = other[1];
        if (other[2] < mVec[2]) mVec[2] = other[2];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ Coord& maxComponent(const Coord& other)
    {
        if (other[0] > mVec[0]) mVec[0] = other[0];
        if (other[1] > mVec[1]) mVec[1] = other[1];
        if (other[2] > mVec[2]) mVec[2] = other[2];
        return *this;
    }

    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    __hostdev__ static inline bool lessThan(const Coord& a, const Coord& b)
    {
        return (a[0] < b[0] || a[1] < b[1] || a[2] < b[2]);
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    template<typename Vec3T>
    __hostdev__ static Coord Floor(const Vec3T& xyz) { return Coord(nanovdb::Floor(xyz[0]), nanovdb::Floor(xyz[1]), nanovdb::Floor(xyz[2])); }

    /// @brief Return a hash key derived from the existing coordinates.
    /// @details For details on this hash function please see the VDB paper.
    template<int Log2N = 3 + 4 + 5>
    __hostdev__ uint32_t hash() const { return ((1 << Log2N) - 1) & (mVec[0] * 73856093 ^ mVec[1] * 19349663 ^ mVec[2] * 83492791); }

    /// @brief Return the octant of this Coord
    //__hostdev__ size_t octant() const { return (uint32_t(mVec[0])>>31) | ((uint32_t(mVec[1])>>31)<<1) | ((uint32_t(mVec[2])>>31)<<2); }
    __hostdev__ uint8_t octant() const { return (uint8_t(bool(mVec[0] & (1u << 31)))) |
                                                (uint8_t(bool(mVec[1] & (1u << 31))) << 1) |
                                                (uint8_t(bool(mVec[2] & (1u << 31))) << 2); }
}; // Coord class

// ----------------------------> Vec3R <--------------------------------------

/// @brief A simple vector class with three double components, simular to openvdb::math::Vec3<double>
template<typename T>
class Vec3
{
    static_assert(is_floating_point<T>::value,"Vec3: expected a floating point value");
    T mVec[3];

public:
    static const int SIZE = 3;
    using ValueType = T;
    Vec3() = default;
    __hostdev__ explicit Vec3(T x)
        : mVec{x, x, x}
    {
    }
    __hostdev__ Vec3(T x, T y, T z)
        : mVec{x, y, z}
    {
    }
    template<typename T2>
    __hostdev__ explicit Vec3(const Vec3<T2>& v)
        : mVec{T(v[0]), T(v[1]), T(v[2])}
    {
    }
    __hostdev__ explicit Vec3(const Coord& ijk)
        : mVec{T(ijk[0]), T(ijk[1]), T(ijk[2])}
    {
    }
    __hostdev__ bool operator==(const Vec3& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    template<typename Vec3T>
    __hostdev__ Vec3& operator=(const Vec3T& rhs)
    {
        mVec[0] = rhs[0];
        mVec[1] = rhs[1];
        mVec[2] = rhs[2];
        return *this;
    }
    __hostdev__ const T& operator[](int i) const { return mVec[i]; }
    __hostdev__ T& operator[](int i) { return mVec[i]; }
    template<typename Vec3T>
    __hostdev__ T dot(const Vec3T& v) const { return mVec[0] * v[0] + mVec[1] * v[1] + mVec[2] * v[2]; }
    template<typename Vec3T>
    __hostdev__ Vec3 cross(const Vec3T& v) const
    {
        return Vec3(mVec[1] * v[2] - mVec[2] * v[1],
                    mVec[2] * v[0] - mVec[0] * v[2],
                    mVec[0] * v[1] - mVec[1] * v[0]);
    }
    __hostdev__ T lengthSqr() const
    {
        return mVec[0] * mVec[0] + mVec[1] * mVec[1] + mVec[2] * mVec[2]; // 5 flops
    }
    __hostdev__ T    length() const { return Sqrt(this->lengthSqr()); }
    __hostdev__ Vec3 operator-() const { return Vec3(-mVec[0], -mVec[1], -mVec[2]); }
    __hostdev__ Vec3 operator*(const Vec3& v) const { return Vec3(mVec[0] * v[0], mVec[1] * v[1], mVec[2] * v[2]); }
    __hostdev__ Vec3 operator/(const Vec3& v) const { return Vec3(mVec[0] / v[0], mVec[1] / v[1], mVec[2] / v[2]); }
    __hostdev__ Vec3 operator+(const Vec3& v) const { return Vec3(mVec[0] + v[0], mVec[1] + v[1], mVec[2] + v[2]); }
    __hostdev__ Vec3 operator-(const Vec3& v) const { return Vec3(mVec[0] - v[0], mVec[1] - v[1], mVec[2] - v[2]); }
    __hostdev__ Vec3 operator*(const T& s) const { return Vec3(s * mVec[0], s * mVec[1], s * mVec[2]); }
    __hostdev__ Vec3 operator/(const T& s) const { return Vec3(mVec[0] / s, mVec[1] / s, mVec[2] / s); }
    __hostdev__ Vec3& operator+=(const Vec3& v)
    {
        mVec[0] += v[0];
        mVec[1] += v[1];
        mVec[2] += v[2];
        return *this;
    }
    __hostdev__ Vec3& operator-=(const Vec3& v)
    {
        mVec[0] -= v[0];
        mVec[1] -= v[1];
        mVec[2] -= v[2];
        return *this;
    }
    __hostdev__ Vec3& operator*=(const T& s)
    {
        mVec[0] *= s;
        mVec[1] *= s;
        mVec[2] *= s;
        return *this;
    }
    __hostdev__ Vec3& operator/=(const T& s)
    {
        mVec[0] /= s;
        mVec[1] /= s;
        mVec[2] /= s;
        return *this;
    }
    __hostdev__ Vec3& normalize()
    {
        (*this) *= T(1) / this->length();
        return *this;
    }
    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ void minComponent(const Vec3& other)
    {
        if (other[0] < mVec[0]) mVec[0] = other[0];
        if (other[1] < mVec[1]) mVec[1] = other[1];
        if (other[2] < mVec[2]) mVec[2] = other[2];
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ void maxComponent(const Vec3& other)
    {
        if (other[0] > mVec[0]) mVec[0] = other[0];
        if (other[1] > mVec[1]) mVec[1] = other[1];
        if (other[2] > mVec[2]) mVec[2] = other[2];
    }
    /// @brief Retun the smallest vector component
    __hostdev__ ValueType min() const
    {
        return mVec[0] < mVec[1] ? (mVec[0] < mVec[2] ? mVec[0] : mVec[2]) :
                                   (mVec[1] < mVec[2] ? mVec[1] : mVec[2]) ;
    }
    /// @brief Retun the largest vector component
    __hostdev__ ValueType max() const
    {
        return mVec[0] > mVec[1] ? (mVec[0] > mVec[2] ? mVec[0] : mVec[2]) :
                                   (mVec[1] > mVec[2] ? mVec[1] : mVec[2]) ;
    }
    __hostdev__ Coord floor() const { return Coord(Floor(mVec[0]),Floor(mVec[1]),Floor(mVec[2])); }
    __hostdev__ Coord  ceil() const { return Coord( Ceil(mVec[0]), Ceil(mVec[1]), Ceil(mVec[2])); }
    __hostdev__ Coord round() const { return Coord(Floor(mVec[0]+0.5),Floor(mVec[1]+0.5),Floor(mVec[2]+0.5)); }
}; // Vec3<T>

template<typename T1, typename T2>
inline __hostdev__ Vec3<T2> operator*(T1 scalar, const Vec3<T2>& vec)
{
    return Vec3<T2>(scalar * vec[0], scalar * vec[1], scalar * vec[2]);
}
template<typename T1, typename T2>
inline __hostdev__ Vec3<T2> operator/(T1 scalar, const Vec3<T2>& vec)
{
    return Vec3<T2>(scalar / vec[0], scalar / vec[1], scalar / vec[2]);
}

using Vec3R = Vec3<double>;
using Vec3d = Vec3<double>;
using Vec3f = Vec3<float>;

// ----------------------------> matMult <--------------------------------------

template<typename Vec3T>
inline __hostdev__ Vec3T matMult(const float* mat, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[1], xyz[2] * mat[2])),
                 fmaf(xyz[0], mat[3], fmaf(xyz[1], mat[4], xyz[2] * mat[5])),
                 fmaf(xyz[0], mat[6], fmaf(xyz[1], mat[7], xyz[2] * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
inline __hostdev__ Vec3T matMult(const double* mat, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[1], static_cast<double>(xyz[2]) * mat[2])),
                 fma(static_cast<double>(xyz[0]), mat[3], fma(static_cast<double>(xyz[1]), mat[4], static_cast<double>(xyz[2]) * mat[5])),
                 fma(static_cast<double>(xyz[0]), mat[6], fma(static_cast<double>(xyz[1]), mat[7], static_cast<double>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
inline __hostdev__ Vec3T matMult(const float* mat, const float* vec, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[1], fmaf(xyz[2], mat[2], vec[0]))),
                 fmaf(xyz[0], mat[3], fmaf(xyz[1], mat[4], fmaf(xyz[2], mat[5], vec[1]))),
                 fmaf(xyz[0], mat[6], fmaf(xyz[1], mat[7], fmaf(xyz[2], mat[8], vec[2])))); // 9 fmaf = 9 flops
}

template<typename Vec3T>
inline __hostdev__ Vec3T matMult(const double* mat, const double* vec, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[1], fma(static_cast<double>(xyz[2]), mat[2], vec[0]))),
                 fma(static_cast<double>(xyz[0]), mat[3], fma(static_cast<double>(xyz[1]), mat[4], fma(static_cast<double>(xyz[2]), mat[5], vec[1]))),
                 fma(static_cast<double>(xyz[0]), mat[6], fma(static_cast<double>(xyz[1]), mat[7], fma(static_cast<double>(xyz[2]), mat[8], vec[2])))); // 9 fma = 9 flops
}

// matMultT: Multiply with the transpose:

template<typename Vec3T>
inline __hostdev__ Vec3T matMultT(const float* mat, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[3], xyz[2] * mat[6])),
                 fmaf(xyz[0], mat[1], fmaf(xyz[1], mat[4], xyz[2] * mat[7])),
                 fmaf(xyz[0], mat[2], fmaf(xyz[1], mat[5], xyz[2] * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
inline __hostdev__ Vec3T matMultT(const double* mat, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[3], static_cast<double>(xyz[2]) * mat[6])),
                 fma(static_cast<double>(xyz[0]), mat[1], fma(static_cast<double>(xyz[1]), mat[4], static_cast<double>(xyz[2]) * mat[7])),
                 fma(static_cast<double>(xyz[0]), mat[2], fma(static_cast<double>(xyz[1]), mat[5], static_cast<double>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
inline __hostdev__ Vec3T matMultT(const float* mat, const float* vec, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[3], fmaf(xyz[2], mat[6], vec[0]))),
                 fmaf(xyz[0], mat[1], fmaf(xyz[1], mat[4], fmaf(xyz[2], mat[7], vec[1]))),
                 fmaf(xyz[0], mat[2], fmaf(xyz[1], mat[5], fmaf(xyz[2], mat[8], vec[2])))); // 9 fmaf = 9 flops
}

template<typename Vec3T>
inline __hostdev__ Vec3T matMultT(const double* mat, const double* vec, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[3], fma(static_cast<double>(xyz[2]), mat[6], vec[0]))),
                 fma(static_cast<double>(xyz[0]), mat[1], fma(static_cast<double>(xyz[1]), mat[4], fma(static_cast<double>(xyz[2]), mat[7], vec[1]))),
                 fma(static_cast<double>(xyz[0]), mat[2], fma(static_cast<double>(xyz[1]), mat[5], fma(static_cast<double>(xyz[2]), mat[8], vec[2])))); // 9 fma = 9 flops
}


// ----------------------------> BBox <-------------------------------------

// Base-class for static polymorphism (cannot be constructed directly)
template<typename Vec3T>
struct BaseBBox
{
    Vec3T mCoord[2];
    __hostdev__ bool  operator==(const BaseBBox& rhs) const { return mCoord[0] == rhs.mCoord[0] && mCoord[1] == rhs.mCoord[1]; };
    __hostdev__ const Vec3T& operator[](int i) const { return mCoord[i]; }
    __hostdev__ Vec3T& operator[](int i) { return mCoord[i]; }
    __hostdev__ Vec3T& min() { return mCoord[0]; }
    __hostdev__ Vec3T& max() { return mCoord[1]; }
    __hostdev__ const Vec3T& min() const { return mCoord[0]; }
    __hostdev__ const Vec3T& max() const { return mCoord[1]; }
    __hostdev__ void translate(const Vec3T &xyz) { mCoord[0] += xyz; mCoord[1] += xyz; }
    // @brief Expand this bounding box to enclose point (i, j, k).
    __hostdev__ void expand(const Vec3T& xyz)
    {
        mCoord[0].minComponent(xyz);
        mCoord[1].maxComponent(xyz);
    }
    __hostdev__ bool isInside(const Vec3T& xyz)
    {
        if (xyz[0] < mCoord[0][0] || xyz[1] < mCoord[0][1] || xyz[2] < mCoord[0][2])
            return false;
        if (xyz[0] > mCoord[1][0] || xyz[1] > mCoord[1][1] || xyz[2] > mCoord[1][2])
            return false;
        return true;
    }
protected:
    __hostdev__ BaseBBox() {}
    __hostdev__ BaseBBox(const Vec3T& min, const Vec3T& max) : mCoord{min, max} {}
};// BaseBBox

template<typename Vec3T, bool = is_floating_point<typename Vec3T::ValueType>::value>
struct BBox;

/// @brief Partial template specialization for floating point coordinate types.
///
/// @note Min is inclusive and max is exclusive. If min = max the dimension of
///       bounding box is is zero and therefore it is also empty.
template<typename Vec3T>
struct BBox<Vec3T, true> : public BaseBBox<Vec3T>
{
    using BaseT = BaseBBox<Vec3T>;
    using BaseT::mCoord;
    __hostdev__ BBox() : BaseT(Vec3T( Maximum<typename Vec3T::ValueType>::value()),
                   Vec3T(-Maximum<typename Vec3T::ValueType>::value())) {};
    __hostdev__ BBox(const Vec3T& min, const Vec3T& max) : BaseT(min, max) {}
    __hostdev__ bool empty() const { return mCoord[0][0] >= mCoord[1][0] ||
                                            mCoord[0][1] >= mCoord[1][1] ||
                                            mCoord[0][2] >= mCoord[1][2]; }
    __hostdev__ operator bool() const { return !this->empty(); }
    __hostdev__ Vec3T dim() const { return this->empty() ? Vec3T(0) : this->max() - this->min(); }
    __hostdev__ bool isInside(const Vec3T &p) const {
        return p[0] > mCoord[0][0] && p[1] > mCoord[0][1] && p[2] > mCoord[0][2] &&
               p[0] < mCoord[1][0] && p[1] < mCoord[1][1] && p[2] < mCoord[1][2];
    }
};

/// @brief Partial template specialization for integer coordinate types
///
/// @note Both min and max are INCLUDED in the bbox so dim = max - min + 1. So,
///       if min = max the bounding box contains exactly one point and dim = 1!
template<typename CoordT>
struct BBox<CoordT, false> : public BaseBBox<CoordT>
{
    static_assert(is_same<int, typename CoordT::ValueType>::value, "Expected \"int\" coordinate type");
    using BaseT = BaseBBox<CoordT>;
    using BaseT::mCoord;
    /// @brief Iterator over the domain covered by a BBox
    /// @details z is the fastest-moving coordinate.
    class Iterator {
        const BBox &mBBox;
        CoordT mPos;
    public:
        __hostdev__ Iterator(const BBox& b): mBBox(b), mPos(b.min()) {}
        __hostdev__ Iterator& operator++() {
            if (mPos[2] < mBBox[1][2]) { ++mPos[2]; } // this is the most common case
            else if (mPos[1] <  mBBox[1][1]) { mPos[2] = mBBox[0][2]; ++mPos[1]; }
            else if (mPos[0] <= mBBox[1][0]) { mPos[2] = mBBox[0][2]; mPos[1] = mBBox[0][1]; ++mPos[0]; }
            return *this;
        }
        __hostdev__ Iterator operator++(int) {auto tmp = *this; ++(*this); return tmp; }
        /// @brief Return @c true if the iterator still points to a valid coordinate.
        __hostdev__ operator bool() const { return mPos[0] <= mBBox[1][0]; }
        __hostdev__ const CoordT& operator*() const { return mPos; }
    };// Iterator
    __hostdev__ Iterator begin() const { return Iterator{*this}; }
    __hostdev__ BBox() : BaseT(CoordT::max(), CoordT::min()) {}
    __hostdev__ BBox(const CoordT& min, const CoordT& max) : BaseT(min, max) {}
    template <typename SplitT>
    __hostdev__ BBox(BBox &other, const SplitT&) : BaseT(other.mCoord[0], other.mCoord[1]) {
        assert(this->is_divisible());
        const int n = MaxIndex(this->dim());
        mCoord[1][n] = (mCoord[0][n] + mCoord[1][n]) >> 1;
        other.mCoord[0][n] = mCoord[1][n] + 1;
    }
    __hostdev__ bool is_divisible() const { return mCoord[0][0] < mCoord[1][0] && 
                                                   mCoord[0][1] < mCoord[1][1] && 
                                                   mCoord[0][2] < mCoord[1][2]; }
    __hostdev__ bool empty() const { return mCoord[0][0] > mCoord[1][0] ||
                                            mCoord[0][1] > mCoord[1][1] ||
                                            mCoord[0][2] > mCoord[1][2]; }
    __hostdev__ operator bool() const { return !this->empty(); }
    __hostdev__ CoordT dim() const {return this->empty() ? Coord(0) : this->max() - this->min() + Coord(1);}
    __hostdev__ bool isInside(const CoordT &p) const { return !(CoordT::lessThan(p,this->min()) || CoordT::lessThan(this->max(),p)); }
    __hostdev__ bool isInside(const BBox& b) const 
    {
        return !(CoordT::lessThan(b.min(),this->min()) || CoordT::lessThan(this->max(),b.max()));
    }
    template <typename RealT>
    __hostdev__  BBox<Vec3<RealT>> asReal() const {
        static_assert(is_floating_point<RealT>::value, "Expected a floating point coordinate");
        return BBox<Vec3<RealT>>(Vec3<RealT>(RealT(mCoord[0][0]    ), RealT(mCoord[0][1]    ), RealT(mCoord[0][2])   ),
                                 Vec3<RealT>(RealT(mCoord[1][0] + 1), RealT(mCoord[1][1] + 1), RealT(mCoord[1][2] + 1)));
    }
};

using CoordBBox = BBox<Coord>;

// ----------------------------> Mask <--------------------------------------

/// @brief Bit-mask to encode active states and facilitate sequnetial iterators
/// and a fast codec for I/O compression.
template<uint32_t LOG2DIM>
class Mask
{
    static constexpr uint32_t SIZE = 1U << (3 * LOG2DIM); // Number of bits in mask
    static constexpr uint32_t WORD_COUNT = SIZE >> 6; // Number of 64 bit words
    uint64_t                  mWords[WORD_COUNT];

public:
    /// @brief Return the memory footprint in bytes of this Mask
    __hostdev__ static size_t memUsage() { return sizeof(Mask); }

    /// @brief Return the number of bit available in this Mask
    __hostdev__ static uint32_t bitCount() { return SIZE; }

    /// @brief Return the number of machine words use by this Mask
    __hostdev__ static uint32_t wordCount() { return WORD_COUNT; }

    __hostdev__ uint32_t countOn() const
    {
        uint32_t sum = 0, n = WORD_COUNT;
        for (const uint64_t* w = mWords; n--; ++w) sum += CountOn(*w);
        return sum;
    }

    class Iterator
    {
    public:
      __hostdev__ Iterator() : mPos(Mask::SIZE), mParent(nullptr) {}
      __hostdev__ Iterator(uint32_t pos, const Mask* parent) : mPos(pos), mParent(parent) {}
      //__hostdev__ bool operator!=(const Iterator &iter) const {return mPos != iter.mPos;}
      Iterator& operator=(const Iterator&) = default;
      __hostdev__ uint32_t operator*() const { return mPos; }
      //__hostdev__ uint32_t pos() const { return mPos; }
      __hostdev__ operator bool() const { return mPos != Mask::SIZE;}
      __hostdev__ Iterator& operator++() { mPos = mParent->findNextOn(mPos+1); return *this; }
    private:
      uint32_t     mPos;
      const Mask*  mParent;
    }; // Memeber class MaskIterator

    /// @brief Initialize all bits to zero.
    __hostdev__ Mask()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = 0;
    }
    __hostdev__ Mask(bool on)
    {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = v;
    }

    /// @brief Copy constructor
    __hostdev__ Mask(const Mask& other)
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = other.mWords[i];
    }

    /// @brief Assignment operator from another Mask type
    __hostdev__ Mask& operator=(const Mask& other)
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = other.mWords[i];
        return *this;
    }

    __hostdev__ Iterator beginOn() const { return Iterator(this->findFirstOn(), this); }
    //__hostdev__ Iterator end() const { return Iterator(SIZE, this); }

    /// @brief Return true of the given bit is set.
    __hostdev__ bool isOn(uint32_t n) const { return 0 != (mWords[n >> 6] & (uint64_t(1) << (n & 63))); }

    __hostdev__ bool isOn() const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) if (mWords[i] != ~uint64_t(0)) return false;
        return true;
    }

    __hostdev__ bool isOff() const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) if (mWords[i] != uint64_t(0)) return false;
        return true;
    }

    /// @brief Set the given bit on.
    __hostdev__ void setOn( uint32_t n) { mWords[n >> 6] |=   uint64_t(1) << (n & 63);  }
    __hostdev__ void setOff(uint32_t n) { mWords[n >> 6] &= ~(uint64_t(1) << (n & 63)); }

    __hostdev__ void set(uint32_t n, bool On) { On ? this->setOn(n) : this->setOff(n); }

    /// @brief Set all bits off
    __hostdev__ void setOff()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = 0;
    }

    /// @brief Set all bits off
    __hostdev__ void set(bool on)
    {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = v;
    }
    /// brief Toggle the state of all bits in the mask
    __hostdev__ void toggle()
    {
        uint32_t n = WORD_COUNT;
        for (auto* w = mWords; n--; ++w) *w = ~*w;
    }
private:
    __hostdev__ static inline uint32_t CountOn(uint64_t v)
    {
        v = v - ((v >> 1) & uint64_t(0x5555555555555555));
        v = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
        return (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) * uint64_t(0x101010101010101)) >> 56;
    }
    __hostdev__ static inline uint32_t FindLowestOn(uint64_t v)
    {
      static const unsigned char DeBruijn[64] = {
          0,   1,  2, 53,  3,  7, 54, 27, 4,  38, 41,  8, 34, 55, 48, 28,
          62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
          63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
          51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
      };
      return DeBruijn[uint64_t((v & -static_cast<int64_t>(v)) * uint64_t(0x022FDD63CC95386D)) >> 58];
    }
    __hostdev__ uint32_t findFirstOn() const
    {
        uint32_t n = 0;
        const uint64_t* w = mWords;
        for (; n<WORD_COUNT && !*w; ++w, ++n) ;
        return n==WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(*w);
    }
    __hostdev__ uint32_t findNextOn(uint32_t start) const
    {
        uint32_t n = start >> 6;//initiate
        if (n >= WORD_COUNT) return SIZE; // check for out of bounds
        uint32_t m = start & 63;
        uint64_t b = mWords[n];
        if (b & (uint64_t(1) << m)) return start;//simpel case: start is on
        b &= ~uint64_t(0) << m;// mask out lower bits
        while(!b && ++n<WORD_COUNT) b = mWords[n];// find next none-zero word
        return (!b ? SIZE : (n << 6) + FindLowestOn(b));//catch last word=0
    }
}; // Mask class

// ----------------------------> Map <--------------------------------------

/// @brief Defined an affine transform and its inverse represented as a 3x3 matrix and a vec3 translation
struct Map
{
    float  mMatF[9]; // 9*4 bytes
    float  mInvMatF[9]; // 9*4 bytes
    float  mVecF[3]; // 3*4 bytes
    float  mTaperF; // 4 bytes, taper value
    double mMatD[9];
    double mInvMatD[9];
    double mVecD[3];
    double mTaperD;

    // This method can only be called on the host to initialize the member data
    template<typename Mat4T>
    __hostdev__ void set(const Mat4T& mat, const Mat4T& invMat, double taper);

    template<typename Vec3T>
    __hostdev__ Vec3T applyMap(const Vec3T& xyz) const { return matMult(mMatD, mVecD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyMapF(const Vec3T& xyz) const { return matMult(mMatF, mVecF, xyz); }

    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobian(const Vec3T& xyz) const { return matMult(mMatD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobianF(const Vec3T& xyz) const { return matMult(mMatF, xyz); }

    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMap(const Vec3T& xyz) const
    {
        return matMult(mInvMatD, Vec3T(xyz[0] - mVecD[0], xyz[1] - mVecD[1], xyz[2] - mVecD[2]));
    }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMapF(const Vec3T& xyz) const
    {
        return matMult(mInvMatF, Vec3T(xyz[0] - mVecF[0], xyz[1] - mVecF[1], xyz[2] - mVecF[2]));
    }

    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobian(const Vec3T& xyz) const { return matMult(mInvMatD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobianF(const Vec3T& xyz) const { return matMult(mInvMatF, xyz); }

    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& xyz) const { return matMultT(mInvMatD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& xyz) const { return matMultT(mInvMatF, xyz); }
}; // Map

template<typename Mat4T>
void Map::set(const Mat4T& mat, const Mat4T& invMat, double taper)
{
    float  *mf = mMatF, *vf = mVecF;
    float  *mif = mInvMatF;
    double *md = mMatD, *vd = mVecD;
    double *mid = mInvMatD;
    mTaperF = static_cast<float>(taper);
    mTaperD = taper;
    for (int i = 0; i < 3; ++i) {
        *vd++ = mat[3][i]; //translation
        *vf++ = static_cast<float>(mat[3][i]);
        for (int j = 0; j < 3; ++j) {
            *md++ = mat[j][i]; //transposed
            *mid++ = invMat[j][i];
            *mf++ = static_cast<float>(mat[j][i]);
            *mif++ = static_cast<float>(invMat[j][i]);
        }
    }
}

// ----------------------------> GridBlindMetaData <--------------------------------------

struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) GridBlindMetaData
{
    static const int      MaxNameSize = 256;
    int64_t               mByteOffset; // byte offset to the blind data, relative to the GridData.
    uint64_t              mElementCount; // number of elements, e.g. point count
    uint32_t              mFlags; // flags
    GridBlindDataSemantic mSemantic; // semantic meaning of the data.
    GridBlindDataClass    mDataClass; // 4 bytes
    GridType              mDataType; // 4 bytes
    char                  mName[MaxNameSize];
}; // GridBlindMetaData

// ----------------------------> Grid <--------------------------------------

/*
    The following class and comment is for internal use only

    Memory layout:

    Grid ->       39 x double                          (world bbox and affine transformation)
    Tree -> Root  3 x ValueType + int32_t + N x Tiles  (background,min,max,tileCount + tileCount x Tiles)

    N2 upper InternalNodes each with 2 bit masks, N2 tiles, and min/max values

    N1 lower InternalNodes each with 2 bit masks, N1 tiles, and min/max values

    N0 LeafNodes each with a bit mask, N0 ValueTypes and min/max
*/

/// @brief Stuct with all the member data of the Grid (useful during serialization of an openvdb grid)
///
/// @note The transform is assumed to be affine (s0 linear!) and have uniform scale! So frustrum transforms
///       and non-uniform scaling is not supported (primarily because they complicate ray-tracing in index space)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!

struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) GridData
{
    static const int MaxNameSize = 256;
    uint64_t         mMagic; // 8 byte magic to validate it is valid grid data.
    char             mGridName[MaxNameSize];
    BBox<Vec3R>      mWorldBBox; // floating-point AABB of active values in WORLD SPACE (2 x 3 doubles)
    Map              mMap; // affine transformation between index and world space in both single and double precision
    double           mUniformScale; // size of a voxel in world units
    GridClass        mGridClass; // 2 bytes
    GridType         mGridType; //  2 bytes
    uint32_t         mBlindDataCount; // count of GridBlindMetaData structures that follow this grid (after the gridname).
    uint64_t         memUsage() const { return sizeof(*this) + mBlindDataCount * sizeof(GridBlindMetaData); }
    //__hostdev__ const char* gridName() const { return mGridName; }
    // Affine transformations based on double precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMap(const Vec3T& xyz) const { return mMap.applyMap(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMap(const Vec3T& xyz) const { return mMap.applyInverseMap(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobian(const Vec3T& xyz) const { return mMap.applyJacobian(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobian(const Vec3T& xyz) const { return mMap.applyInverseJacobian(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& xyz) const { return mMap.applyIJT(xyz); }
    // Affine transformations based on single precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMapF(const Vec3T& xyz) const { return mMap.applyMapF(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMapF(const Vec3T& xyz) const { return mMap.applyInverseMapF(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobianF(const Vec3T& xyz) const { return mMap.applyJacobianF(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobianF(const Vec3T& xyz) const { return mMap.applyInverseJacobianF(xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& xyz) const { return mMap.applyIJTF(xyz); }

    /// @brief Return a const pointer to the blind meta data
    __hostdev__ const GridBlindMetaData* metaPtr() const
    {
        return reinterpret_cast<const GridBlindMetaData*>(this + 1);
    }

    // @brief Return a const void pointer to the tree
    __hostdev__ const void* treePtr() const { return this->metaPtr() + mBlindDataCount; }

    /// @brief Returns a const reference to the blindMetaData at the specified linear offset.
    ///
    /// @warning The linear offset is assumed to be in the valid range
    __hostdev__ const GridBlindMetaData& blindMetaData(uint32_t n) const
    {
        assert(n < mBlindDataCount);
        return *(this->metaPtr() + n);
    }

}; // GridData

// Forward decleration of accelerated random access class
template<typename>
class ReadAccessor;

/// @brief Highest level of the data structure. Contains a tree and a world->index
///        transform (that currenrtly only supports uniform scaling and translation).
///
/// @note This the API of this class to interface with client code
template<typename TreeT>
class Grid : private GridData
{
public:
    using TreeType = TreeT;
    using DataType = GridData;
    using ValueType = typename TreeT::ValueType;
    using CoordType = typename TreeT::CoordType;
    using AccessorType = ReadAccessor<typename TreeT::RootType>;

    /// @brief Disallow constructions, copy and assignment
    ///
    /// @note Only a Serializer, defined elsewhere, can instantiate this class
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    ~Grid() = delete;

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief return memory usage in bytes for the class (note this computes room for blindMetaData structures.)
    __hostdev__ static uint64_t memUsage(uint64_t blindDataCount = 0)
    {
        return sizeof(GridData) + blindDataCount * sizeof(GridBlindMetaData);
    }

    /// @brief Return a const reference to the tree
    __hostdev__ const TreeT& tree() const { return *reinterpret_cast<const TreeT*>(this->treePtr()); }

    /// @brief Return a new instance of a ReadAccessor used to access values in this grid
    __hostdev__ AccessorType getAccessor() const { return ReadAccessor<typename TreeT::RootType>(this->tree().root()); }

    /// @brief Return a const reference to the size of a (uniform) voxel in world units
    __hostdev__ const double& voxelSize() const { return DataType::mUniformScale; }

    /// @brief Return a const reference to the Map for this grid
    __hostdev__ const Map& map() const { return DataType::mMap; }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndex(const Vec3T& xyz) const { return this->applyInverseMap(xyz); }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorld(const Vec3T& xyz) const { return this->applyMap(xyz); }

    // assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDir(const Vec3T& dir) const { return this->applyInverseJacobian(dir); }
    //__hostdev__ Vec3T indexToWorldDir( const Vec3T &dir ) const { return this->applyJacobian(dir); }

    // assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDir(const Vec3T& dir) const { return this->applyJacobian(dir); }
    //__hostdev__ Vec3T worldToIndexDir( const Vec3T &dir ) const { return this->applyInverseJacobian(dir); }

    // Inverse jacobian map, suitable for gradients.
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& dir) const { return this->applyIJT(dir); }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexF(const Vec3T& xyz) const { return this->applyInverseMapF(xyz); }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldF(const Vec3T& xyz) const { return this->applyMapF(xyz); }

    // assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDirF(const Vec3T& dir) const { return this->applyInverseJacobianF(dir); }
    //__hostdev__ Vec3T indexToWorldDirF( const Vec3T &dir ) const { return this->applyJacobianF(dir); }

    // assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDirF(const Vec3T& dir) const { return this->applyJacobianF(dir); }
    //__hostdev__ Vec3T worldToIndexDirF( const Vec3T &dir ) const { return this->applyInverseJacobianF(dir); }

    // Inverse jacobian map, suitable for gradients.
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& dir) const { return this->applyIJTF(dir); }


    /// @brief Computes a AABB of active values in world space
    __hostdev__ const BBox<Vec3R>& worldBBox() const { return DataType::mWorldBBox; }

    /// @brief Computes a AABB of active values in index space
    __hostdev__ const BBox<CoordType>& indexBBox() const { return this->tree().bbox(); }

    /// @brief Return the total number of active voxels in this tree.
    __hostdev__ const uint64_t& activeVoxelCount() const { return this->tree().activeVoxelCount(); }

    /// @brief Methods related to the classification of this grid
    __hostdev__ bool      isValid() const { return DataType::mMagic == NANOVDB_MAGIC_NUMBER; }
    __hostdev__ const GridType&  gridType() const { return DataType::mGridType; }
    __hostdev__ const GridClass& gridClass() const { return DataType::mGridClass; }
    __hostdev__ bool      isLevelSet() const { return DataType::mGridClass == GridClass::LevelSet; }
    __hostdev__ bool      isFogVolume() const { return DataType::mGridClass == GridClass::FogVolume; }
    __hostdev__ bool      isStaggered() const { return DataType::mGridClass == GridClass::Staggered; }
    __hostdev__ bool      isPointIndex() const { return DataType::mGridClass == GridClass::PointIndex; }
    __hostdev__ bool      isPointData() const { return DataType::mGridClass == GridClass::PointData; }
    __hostdev__ bool      isUnknown() const { return DataType::mGridClass == GridClass::Unknown; }

    /// @brief Return a c-string with the name of this grid
    __hostdev__ const char* gridName() const { return DataType::mGridName; }

    /// @brief Return the count of blind-data encoded in this grid
    __hostdev__ int blindDataCount() const { return DataType::mBlindDataCount; }

    /// @brief Return the index of the blind data with specified semantic if found, otherwise -1.
    __hostdev__ int findBlindDataForSemantic(GridBlindDataSemantic semantic) const;

    /// @brief Returns a const pointer to the blindData at the specified linear offset.
    ///
    /// @warning Point might be NULL and the linear offset is assumed to be in the valid range
    __hostdev__ const void* blindData(uint32_t n) const
    {
        if (DataType::mBlindDataCount == 0)
            return nullptr;
        assert(n < DataType::mBlindDataCount);
        return reinterpret_cast<const char*>(this) + this->blindMetaData(n).mByteOffset;
    }

    __hostdev__ const GridBlindMetaData& blindMetaData(int n) const { return DataType::blindMetaData(n); }

private:
    static_assert(sizeof(GridData) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(GridData) is misaligned");
}; // Class Grid

template<typename TreeT>
int Grid<TreeT>::findBlindDataForSemantic(GridBlindDataSemantic semantic) const
{
    for (uint32_t i = 0, n = blindDataCount(); i < n; ++i)
        if (blindMetaData(i).mSemantic == semantic)
            return int(i);
    return -1;
}

// ----------------------------> Tree <--------------------------------------

template<int ROOT_LEVEL>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) TreeData
{
    uint64_t mBytes[ROOT_LEVEL + 1]; // byte offsets to nodes of type: leaf, lower internal, upper internal, and root
    uint32_t mCount[ROOT_LEVEL + 1]; // total number of nodes of type: leaf, lower internal, upper internal, and root
};

/// @brief Struct to derive node type from its level in a given tree
template<typename TreeT, int LEVEL>
struct Node;

// Partial template specialization of above Node struct
template<typename T>
struct Node<T, 0>
{
    using type = typename T::LeafNodeType;
};
template<typename T>
struct Node<T, 1>
{
    using type = typename T::RootType::ChildNodeType::ChildNodeType;
};
template<typename T>
struct Node<T, 2>
{
    using type = typename T::RootType::ChildNodeType;
};
template<typename T>
struct Node<T, 3>
{
    using type = typename T::RootType;
};

/// @brief VDB Tree, which is a thin wrapper around a RootNode.
template<typename RootT>
class Tree : private TreeData<RootT::LEVEL>
{
public:
    using DataType = TreeData<RootT::LEVEL>;
    using RootType = RootT;
    using LeafNodeType = typename RootT::LeafNodeType;
    using ValueType = typename RootT::ValueType;
    using CoordType = typename RootT::CoordType;
    using AccessorType = ReadAccessor<RootT>;
    template<int LEVEL>
    using NodeType = typename Node<Tree, LEVEL>::type;
    static_assert(is_same<NodeType<0>, LeafNodeType>::value, "NodeType<0> error");
    static_assert(is_same<NodeType<3>, RootType>::value, "NodeType<3> error");

    /// @brief This class cannot be constructed or deleted
    Tree() = delete;
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    ~Tree() = delete;

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief return memory usage in bytes for the class
    __hostdev__ static uint64_t memUsage() { return sizeof(DataType); }

    __hostdev__ const RootT& root() const { return *reinterpret_cast<const RootT*>(reinterpret_cast<const uint8_t*>(this) + DataType::mBytes[RootT::LEVEL]); }

    __hostdev__ AccessorType getAccessor() const { return ReadAccessor<RootT>(this->root()); }

    /// @brief Return the value of the given voxel (regardless of state or location in the tree.)
    __hostdev__ const ValueType& getValue(const CoordType& ijk) const { return this->root().getValue(ijk); }

    /// @brief Return the active state of the given voxel (regardless of state or location in the tree.)
    __hostdev__ bool isActive(const CoordType& ijk) const { return this->root().isActive(ijk); }

    /// @brief Combines the previous two methods in a single call
    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const { return this->root().probeValue(ijk, v); }

    /// @brief Return a const reference to the background value.
    __hostdev__ const ValueType& background() const { return this->root().background(); }

    /// @brief Sets the extrema values of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ void extrema(ValueType& min, ValueType& max) const;

    /// @brief Return a const reference to the index bounding box of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ const BBox<CoordType>& bbox() const { return this->root().bbox(); }

    /// @brief Return the total number of active voxels in this tree.
    __hostdev__ const uint64_t& activeVoxelCount() const { return this->root().activeVoxelCount(); }

    template<typename NodeT>
    __hostdev__ uint32_t nodeCount() const { return DataType::mCount[NodeT::LEVEL]; }

    __hostdev__ uint32_t nodeCount(int level) const { return DataType::mCount[level]; }

    template<typename NodeT>
    __hostdev__ const NodeT* getNode(uint32_t i) const;

    template<int LEVEL>
    __hostdev__ const NodeType<LEVEL>* getNode(uint32_t i) const;

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(TreeData) is misaligned");

}; // Tree class

template<typename RootT>
void Tree<RootT>::extrema(ValueType& min, ValueType& max) const
{
    min = this->root().valueMin();
    max = this->root().valueMax();
}

template<typename RootT>
template<typename NodeT>
const NodeT* Tree<RootT>::getNode(uint32_t i) const
{
    static_assert(is_same<NodeType<NodeT::LEVEL>, NodeT>::value, "Tree::getNode: unvalid node type");
    assert(i < DataType::mCount[NodeT::LEVEL]);
    return reinterpret_cast<const NodeT*>(reinterpret_cast<const uint8_t*>(this) + DataType::mBytes[NodeT::LEVEL]) + i;
}

template<typename RootT>
template<int LEVEL>
const typename Node<Tree<RootT>, LEVEL>::type* Tree<RootT>::getNode(uint32_t i) const
{
    assert(i < DataType::mCount[LEVEL]);
    return reinterpret_cast<const NodeType<LEVEL>*>(reinterpret_cast<const uint8_t*>(this) + DataType::mBytes[LEVEL]) + i;
}

// --------------------------> RootNode <------------------------------------

/// @brief Stuct with all the member data of the RootNode (useful during serialization of an openvdb RootNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ChildT>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) RootData
{
    using ValueT = typename ChildT::ValueType;
    using CoordT = typename ChildT::CoordType;
    /// @brief Return a key based on the coordinates of a voxel
#ifdef USE_SINGLE_ROOT_KEY
    using KeyT = uint64_t;
    __hostdev__ static KeyT CoordToKey(const CoordT& ijk)
    {
        static_assert(32 - ChildT::TOTAL <= 21, "Cannot use 64 bit root keys");
        return (KeyT(uint32_t(ijk[2]) >> ChildT::TOTAL)) | // lower 21 bits
               (KeyT(uint32_t(ijk[1]) >> ChildT::TOTAL) << 21) | // middle 21 bits
               (KeyT(uint32_t(ijk[0]) >> ChildT::TOTAL) << 42); // upper 21 bits
    }
#else
    using KeyT = CoordT;
    __hostdev__ static KeyT CoordToKey(const CoordT& ijk) { return ijk & ~ChildT::MASK; }
#endif
    BBox<CoordT> mBBox; // AABB if active values in index space, 2*3*4 = 24 bytes
    uint64_t mActiveVoxelCount; // total number of active voxels in the root and all its child nodes, 8 bytes
    uint32_t mTileCount, _padding[3]; // number of tiles and child pointers in the root node, 4 bytes
    ValueT   mBackground, mValueMin, mValueMax; // background value, i.e. value of any unset voxel + min/max

    struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) Tile
    {
        __hostdev__ void setChild(const CoordT& k, int32_t n)
        {
            key = CoordToKey(k);
            childID = n;
        }
        __hostdev__ void setValue(const CoordT& k, bool s, ValueT v)
        {
            key = CoordToKey(k);
            state = s;
            value = v;
            childID = -1;
        }
        KeyT    key;
        ValueT  value; // value of tile (i.e. no child node)
        int32_t childID; // negative values indicate no child node, i.e. this is a value tile
        uint8_t state; // state of tile value
    }; // Tile

    /// @brief Returns a non-const reference to the tile at the specified linear offset.
    ///
    /// @warning The linear offset is assumed to be in the valid range
    __hostdev__ Tile& tile(uint32_t n) const
    {
        assert(n < mTileCount);
        return *(reinterpret_cast<Tile*>(const_cast<RootData*>(this) + 1) + n);
    }

    /// @brief Returns a const reference to the child node in the specified tile.
    ///
    /// @warning A child node is assumed to exist in the specified tile
    __hostdev__ const ChildT& child(const Tile& tile) const
    {
        assert(tile.childID >= 0 && tile.childID < int32_t(ChildT::SIZE));
        return *(reinterpret_cast<const ChildT*>(reinterpret_cast<const Tile*>(this + 1) + mTileCount) + tile.childID);
    }

    /// @brief This class cannot be constructed or deleted
    RootData() = delete;
    RootData(const RootData&) = delete;
    RootData& operator=(const RootData&) = delete;
    ~RootData() = delete;
}; // RootData

/// @brief Top-most node of the VDB tree structure.
template<typename ChildT>
class RootNode : private RootData<ChildT>
{
public:
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    using DataType = RootData<ChildT>;
    using LeafNodeType = typename ChildT::LeafNodeType;
    using ChildNodeType = ChildT;
    using ValueType = typename ChildT::ValueType;
    using CoordType = typename ChildT::CoordType;
    using Tile = typename DataType::Tile;

    /// @brief This class cannot be constructed or deleted
    RootNode() = delete;
    RootNode(const RootNode&) = delete;
    RootNode& operator=(const RootNode&) = delete;
    ~RootNode() = delete;

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return a const reference to the index bounding box of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ const BBox<CoordType>& bbox() const { return DataType::mBBox; }

     /// @brief Return the total number of active voxels in the root and all its child nodes.
    __hostdev__ const uint64_t& activeVoxelCount() const { return DataType::mActiveVoxelCount; }

    /// @brief Return a const reference to the background value, i.e. the value associated with
    ///        any coordinate location that has not been set explicitly.
    __hostdev__ const ValueType& background() const { return DataType::mBackground; }

    /// @brief Return a const reference to the minimum active value encoded in this root node and any of its child nodes
    __hostdev__ const ValueType& valueMin() const { return DataType::mValueMin; }

    /// @brief Return a const reference to the maximum active value encoded in this root node and any of its child nodes
    __hostdev__ const ValueType& valueMax() const { return DataType::mValueMax; }

    /// @brief Return the number of tiles encoded in this root node
    __hostdev__ const uint32_t& tileCount() const { return DataType::mTileCount; }

    /// @brief Return the expected memory footprint in bytes with the specified number of tiles
    __hostdev__ static uint64_t memUsage(uint32_t _tileCount) { return sizeof(RootNode) + _tileCount * sizeof(Tile); }

    /// @brief Return the actual memory footprint of this root node
    __hostdev__ uint64_t memUsage() const { return sizeof(RootNode) + DataType::mTileCount * sizeof(Tile); }

    /// @brief Return the value of the given voxel
    __hostdev__ const ValueType& getValue(const CoordType& ijk) const
    {
        const Tile* tile = this->findTile(ijk);
        return tile ? (tile->childID < 0 ? tile->value : this->child(*tile).getValue(ijk)) : DataType::mBackground;
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
        const Tile* tile = this->findTile(ijk);
        return tile ? (tile->childID < 0 ? tile->state : this->child(*tile).isActive(ijk)) : false;
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->childID < 0) {
                v = tile->value;
                return tile->state;
            } else {
                return this->child(*tile).probeValue(ijk, v);
            }
        }
        v = DataType::mBackground;
        return false;
    }

    __hostdev__ const LeafNodeType* probeLeaf(const CoordType& ijk) const
    {
        const Tile* tile = this->findTile(ijk);
        if (tile == nullptr || tile->childID < 0)
            return nullptr;
        return this->child(*tile).probeLeaf(ijk);
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(RootData) is misaligned");
    static_assert(sizeof(typename DataType::Tile) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(RootData::Tile) is misaligned");

    template<typename>
    friend class ReadAccessor;
    template<typename>
    friend class Tree;

    /// @brief Private method to find a Tile of this root node by means of binary-search. This is obviously
    ///        much slower then direct lookup into a linear array (as in the other nodes) which is exactly
    ///        why it is important to use the ReadAccessor which amortizes this overhead by node caching and
    ///        inverse tree traversal!
    __hostdev__ const Tile* findTile(const CoordType& ijk) const
    {
        // binary-search of pre-sorted elements
        int32_t     low = 0, high = DataType::mTileCount; //low is inclusive and high is exclusive
        const Tile* tiles = reinterpret_cast<const Tile*>(this + 1);
        const auto  key = DataType::CoordToKey(ijk);
#if 1 //switch between linear and binary seach
        for (int i = low; i < high; i++) {
            const Tile* tile = &tiles[i];
            if (tile->key == key)
                return tile;
        }
#else
        while (low != high) {
            int         mid = low + ((high - low) >> 1);
            const Tile* tile = &tiles[mid];
            if (tile->key == key) {
                return tile;
            } else if (tile->key < key) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
#endif
        return nullptr;
    }

    /// @brief Private method to retun a voxel value and update a ReadAccessor
    template<typename AccT>
    __hostdev__ const ValueType& getValueAndCache(const CoordType& ijk, const AccT& acc) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->childID < 0)
                return tile->value;
            const ChildT& child = this->child(*tile);
            acc.insert(ijk, &child);
            return child.getValueAndCache(ijk, acc);
        }
        return DataType::mBackground;
    }

    template<typename AccT>
    __hostdev__ bool isActiveAndCache(const CoordType& ijk, const AccT& acc) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->childID < 0)
                return tile->state;
            const ChildT& child = this->child(*tile);
            acc.insert(ijk, &child);
            return child.isActiveAndCache(ijk, acc);
        }
        return false;
    }

    template<typename AccT>
    __hostdev__ bool probeValueAndCache(const CoordType& ijk, ValueType& v, const AccT& acc) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->childID < 0) {
                v = tile->value;
                return tile->state;
            }
            const ChildT& child = this->child(*tile);
            acc.insert(ijk, &child);
            return child.probeValueAndCache(ijk, v, acc);
        }
        v = DataType::mBackground;
        return false;
    }

    template<typename AccT>
    __hostdev__ const LeafNodeType* probeLeafAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const Tile* tile = this->findTile(ijk);
        if (tile == nullptr || tile->childID < 0)
            return nullptr;
        const ChildT& child = this->child(*tile);
        acc.insert(ijk, &child);
        return child.probeLeafAndCache(ijk, acc);
    }

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordType& ijk, const RayT& ray, const AccT& acc) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->childID < 0)
                return 1 << ChildT::TOTAL; //tile value
            const ChildT& child = this->child(*tile);
            acc.insert(ijk, &child);
            return child.getDimAndCache(ijk, ray, acc);
        }
        return ChildNodeType::dim(); // background
    }

}; // RootNode class

// After the RootNode the memory layout is assumbed to be the sorted Tiles

// --------------------------> InternalNode <------------------------------------

/// @brief Stuct with all the member data of the InternalNode (useful during serialization of an openvdb InternalNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ChildT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) InternalData
{
    using ValueT = typename ChildT::ValueType;
    using CoordT = typename ChildT::CoordType;
    using MaskT = typename ChildT::template MaskType<LOG2DIM>;

    union Tile
    {
        ValueT   value;
        uint32_t childID;

        /// @brief This class cannot be constructed or deleted
        Tile() = delete;
        Tile(const Tile&) = delete;
        Tile& operator=(const Tile&) = delete;
        ~Tile() = delete;
    }; // if ValueType is a float this has a footprint of only 4 bytes!

    MaskT        mValueMask, mChildMask; // typically 16^3/8=512 or 32^3/8=4096 bytes
    Tile         mTable[1u << (3 * LOG2DIM)]; // typically 16^3*4=16384 or 32^3*4=262144 bytes
    ValueT       mValueMin, mValueMax; // typically 8 bytes
    BBox<CoordT> mBBox; // 2*3*4 = 24 bytes
    int32_t      mOffset; // number of node offsets till first tile
    uint32_t     mFlags; // (due to word alignment 32 bit mFlags is free on 64 bit OS) 8 bytes

    /// @brief Returns a const pointer to the child node at the specifed linear offset.
    __hostdev__ const ChildT* child(uint32_t n) const
    {
        assert(mChildMask.isOn(n));
        return reinterpret_cast<const ChildT*>(this + mOffset) + mTable[n].childID;
    }

    /// @brief This class cannot be constructed or deleted
    InternalData() = delete;
    InternalData(const InternalData&) = delete;
    InternalData& operator=(const InternalData&) = delete;
    ~InternalData() = delete;
}; // InternalData

/// @brief Interal nodes of a VDB treedim(),
template<typename ChildT, uint32_t Log2Dim = ChildT::LOG2DIM + 1>
class InternalNode : private InternalData<ChildT, Log2Dim>
{
public:
    static constexpr uint32_t LOG2DIM = Log2Dim;
    static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; //dimension in index space
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); //number of tile values (or child pointers)
    static constexpr uint32_t MASK = (1u << TOTAL) - 1u;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf

    using LeafNodeType = typename ChildT::LeafNodeType;
    using ChildNodeType = ChildT;
    using ValueType = typename ChildT::ValueType;
    using CoordType = typename ChildT::CoordType;
    template<uint32_t LOG2>
    using MaskType = typename ChildT::template MaskType<LOG2>;
    using DataType = InternalData<ChildT, Log2Dim>;

    /// @brief This class cannot be constructed or deleted
    InternalNode() = delete;
    InternalNode(const InternalNode&) = delete;
    InternalNode& operator=(const InternalNode&) = delete;
    ~InternalNode() = delete;

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return the dimnetion, in voxel units, of this internal node (typically 8*16 or 8*16*32)
    __hostdev__ static uint32_t dim() { return 1u << TOTAL; }

    /// @brief return memory usage in bytes for the class
    __hostdev__ static size_t memUsage() { return sizeof(DataType); }

    /// @brief Return a const reference to the bit mask of active voxels in this internal node
    __hostdev__ const MaskType<LOG2DIM>& valueMask() const { return DataType::mValueMask; }

    /// @brief Return a const reference to the bit mask of child nodes in this internal node
    __hostdev__ const MaskType<LOG2DIM>& childMask() const { return DataType::mChildMask; }

    /// @brief Return the origin in index space of this leaf node
    __hostdev__ CoordType origin() const { return DataType::mBBox.min() & ~MASK; }

    /// @brief Return a const reference to the minimum active value encoded in this internal node and any of its child nodes
    __hostdev__ const ValueType& valueMin() const { return DataType::mValueMin; }

    /// @brief Return a const reference to the maximum active value encoded in this internal node and any of its child nodes
    __hostdev__ const ValueType& valueMax() const { return DataType::mValueMax; }

    /// @brief Return a const reference to the bounding box in index space of active values in this internal node and any of its child nodes
    __hostdev__ const BBox<CoordType>& bbox() const { return DataType::mBBox; }

    /// @brief Return the value of the given voxel
    __hostdev__ const ValueType& getValue(const CoordType& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        return DataType::mChildMask.isOn(n) ? this->child(n)->getValue(ijk) : DataType::mTable[n].value;
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        return DataType::mChildMask.isOn(n) ? this->child(n)->isActive(ijk) : DataType::mValueMask.isOn(n);
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (DataType::mChildMask.isOn(n))
            return this->child(n)->probeValue(ijk, v);
        v = DataType::mTable[n].value;
        return DataType::mValueMask.isOn(n);
    }

    __hostdev__ const LeafNodeType* probeLeaf(const CoordType& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (DataType::mChildMask.isOn(n))
            return this->child(n)->probeLeaf(ijk);
        return nullptr;
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(InternalData) is misaligned");

    template<typename>
    friend class ReadAccessor;
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;

    /// @brief Return the linear offset corresponding to the given coordinate
    __hostdev__ static uint32_t CoordToOffset(const CoordType& ijk)
    {
        return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) +
               (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) +
               ((ijk[2] & MASK) >> ChildT::TOTAL);
    }

    /// @biref Private read access method used by the ReadAccessor
    template<typename AccT>
    __hostdev__ const ValueType& getValueAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n))
            return DataType::mTable[n].value;
        const ChildT* child = this->child(n);
        acc.insert(ijk, child);
        return child->getValueAndCache(ijk, acc);
    }

    template<typename AccT>
    __hostdev__ bool isActiveAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n))
            return DataType::mValueMask.isOn(n);
        const ChildT* child = this->child(n);
        acc.insert(ijk, child);
        return child->isActiveAndCache(ijk, acc);
    }

    template<typename AccT>
    __hostdev__ bool probeValueAndCache(const CoordType& ijk, ValueType& v, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n)) {
            v = DataType::mTable[n].value;
            return DataType::mValueMask.isOn(n);
        }
        const ChildT* child = this->child(n);
        acc.insert(ijk, child);
        return child->probeValueAndCache(ijk, v, acc);
    }

    template<typename AccT>
    __hostdev__ const LeafNodeType* probeLeafAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n))
            return nullptr;
        const ChildT* child = this->child(n);
        acc.insert(ijk, child);
        return child->probeLeafAndCache(ijk, acc);
    }

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordType& ijk, const RayT& ray, const AccT& acc) const
    {
        if (DataType::mFlags & uint32_t(1))
            this->dim(); //ship this node if first bit is set
        //if (DataType::mFlags) ChildNodeType::dim();
        //if (DataType::mFlags || !ray.intersects( this->bbox() )) return 1<<TOTAL;
        //auto bbox = this->bbox();
        //bbox.min() += -1;
        //bbox.max() += 1;
        //if (!ray.intersects( bbox )) return 1<<TOTAL;

        const uint32_t n = CoordToOffset(ijk);
        if (DataType::mChildMask.isOn(n)) {
            const ChildT* child = this->child(n);
            acc.insert(ijk, child);
            return child->getDimAndCache(ijk, ray, acc);
        }
        return ChildNodeType::dim(); // tile value
    }

}; // InternalNode class

// --------------------------> LeafNode <------------------------------------

/// @brief Stuct with all the member data of the LeafNode (useful during serialization of an openvdb LeafNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ValueT, typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData
{
    using ValueType = ValueT;
    MaskT<LOG2DIM>    mValueMask; // typically 8 x uint64_t = 64 bytes
    ValueType         mValues[1u << 3 * LOG2DIM]; // typically 512 * 4 bytes = 2048 bytes
    ValueType         mValueMin, mValueMax; // typically 2 * 4 bytes = 8 bytes
    CoordT            mBBoxMin; // 12 bytes
    uint8_t           mBBoxDif[3], mFlags; // 4 bytes (due to word alignment the 8 bit mFlags is free on 64 bit OS)
    __hostdev__ const ValueType* values() const { return mValues; }
    __hostdev__ const ValueType& value(uint32_t i) const { return mValues[i]; }
    __hostdev__ const ValueType& valueMin() const { return mValueMin; }
    __hostdev__ const ValueType& valueMax() const { return mValueMax; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData

// Partial template specialization of LeafData with ValueMask
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<ValueMask, CoordT, MaskT, LOG2DIM>
{
    using ValueType = bool;
    static const ValueType mDummy;
    MaskT<LOG2DIM>         mValueMask; // typically 8 x uint64_t = 64 bytes
    CoordT                 mBBoxMin; // 12 bytes
    uint8_t                mBBoxDif[3], mFlags; // 4 bytes (due to word alligment the 8 bit mFlags is free on 64 bit OS)
    __hostdev__ const ValueType* values() const { return nullptr; }
    __hostdev__ const ValueType& value(uint32_t) const { return mDummy; }
    __hostdev__ const ValueType& valueMin() const { return mDummy; }
    __hostdev__ const ValueType& valueMax() const { return mDummy; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<ValueMask>

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
const bool LeafData<ValueMask, CoordT, MaskT, LOG2DIM>::mDummy = 1;

/// @brief Leaf nodes of the VDB tree. (defaults to 8x8x8 = 512 voxels)
template<typename ValueT,
         typename CoordT = Coord,
         template<uint32_t> class MaskT = Mask,
         uint32_t Log2Dim = 3>
class LeafNode : private LeafData<ValueT, CoordT, MaskT, Log2Dim>
{
public:
    static constexpr uint32_t LOG2DIM = Log2Dim;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr uint32_t MASK = (1u << LOG2DIM) - 1u; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf

    struct ChildNodeType
    {
        __hostdev__ static uint32_t dim() { return 1u; }
    }; // Voxel
    using LeafNodeType = LeafNode<ValueT, CoordT, MaskT, LOG2DIM>;
    using DataType = LeafData<ValueT, CoordT, MaskT, Log2Dim>;
    using ValueType = typename DataType::ValueType;
    using CoordType = CoordT;
    template<uint32_t LOG2>
    using MaskType = MaskT<LOG2>;

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return a const reference to the bit mask of active voxels in this leaf node
    __hostdev__ const MaskType<LOG2DIM>& valueMask() const { return DataType::mValueMask; }

    /// @brief Return a const pointer to the c-style array of voxe values of this leaf node
    __hostdev__ const ValueType* voxels() const { return DataType::values(); }

    /// @brief Return a const reference to the minimum active value encoded in this leaf node
    __hostdev__ const ValueType& valueMin() const { return DataType::valueMin(); }

    /// @brief Return a const reference to the maximum active value encoded in this leaf node
    __hostdev__ const ValueType& valueMax() const { return DataType::valueMax(); }

    __hostdev__ uint8_t flags() const { return DataType::mFlags; }

    /// @brief Return the origin in index space of this leaf node
    __hostdev__ CoordT origin() const { return DataType::mBBoxMin & ~MASK; }

    __hostdev__ static CoordT OffsetToLocalCoord(uint32_t n)
    {
        assert(n < SIZE);
        const uint32_t m = n & ((1<<2*LOG2DIM)-1);
        return CoordT(n >> 2*LOG2DIM, m >> LOG2DIM, m & MASK);
    }

    __hostdev__ CoordT offsetToGlobalCoord(uint32_t n) const
    {
        return OffsetToLocalCoord(n) + this->origin();
    }

    /// @brief Return the dimension, in index space, of this leaf node (typically 8 as for openvdb leaf nodes!)
    __hostdev__ static uint32_t dim() { return 1u << LOG2DIM; }

    /// @brief Return the bounding box in index space of active values in this leaf node
    __hostdev__ BBox<CoordT> bbox() const
    {
        BBox<CoordT> bbox(DataType::mBBoxMin, DataType::mBBoxMin);
        bbox.max()[0] += DataType::mBBoxDif[0];
        bbox.max()[1] += DataType::mBBoxDif[1];
        bbox.max()[2] += DataType::mBBoxDif[2];
        return bbox;
    }

    /// @brief Return the total number of voxels (e.g. values) encoded in this leaf node
    __hostdev__ static uint32_t voxelCount() { return 1u << (3 * LOG2DIM); }

    /// @brief return memory usage in bytes for the class
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafNodeType); }

    /// @brief This class cannot be constructed or deleted
    LeafNode() = delete;
    LeafNode(const LeafNode&) = delete;
    LeafNode& operator=(const LeafNode&) = delete;
    ~LeafNode() = delete;

    /// @brief Return the voxel value at the given offset.
    __hostdev__ const ValueType& getValue(uint32_t offset) const { return DataType::value(offset); }

    /// @brief Return the voxel value at the given coordinate.
    __hostdev__ const ValueType& getValue(const CoordT& ijk) const { return DataType::value(CoordToOffset(ijk)); }

    /// @brief Return @c true if the voxel value at the given coordinate is active.
    __hostdev__ bool isActive(const CoordT& ijk) const { return DataType::mValueMask.isOn(CoordToOffset(ijk)); }
    __hostdev__ bool isActive(uint32_t n) const { return DataType::mValueMask.isOn(n); }

    /// @brief Retun @c true if the voxel value at the given coordinate is active and updates @c v with the value.
    __hostdev__ bool probeValue(const CoordT& ijk, ValueType& v) const
    {
        const uint32_t n = CoordToOffset(ijk);
        v = DataType::value(n);
        return DataType::mValueMask.isOn(n);
    }

    __hostdev__ const LeafNode* probeLeaf(const CoordT&) const { return this; }

    /// @brief Return the linear offset corresponding to the given coordinate
    __hostdev__ static uint32_t CoordToOffset(const CoordT& ijk)
    {
        return ((ijk[0] & MASK) << (2 * LOG2DIM)) + ((ijk[1] & MASK) << LOG2DIM) + (ijk[2] & MASK);
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(LeafData) is misaligned");

    template<typename>
    friend class ReadAccessor;
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;

    /// @brief Private method to retun a voxel value and update a (dummy) ReadAccessor
    template<typename AccT>
    __hostdev__ const ValueType& getValueAndCache(const CoordT& ijk, const AccT&) const { return this->getValue(ijk); }

    template<typename AccT>
    __hostdev__ bool isActiveAndCache(const CoordT& ijk, const AccT&) const { return this->isActive(ijk); }

    template<typename AccT>
    __hostdev__ bool probeValueAndCache(const CoordT& ijk, ValueType& v, const AccT&) const { return this->probeValue(ijk, v); }

    template<typename AccT>
    __hostdev__ const LeafNode* probeLeafAndCache(const CoordT&, const AccT&) const { return this; }

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordT&, const RayT& /*ray*/, const AccT&) const
    {
        if (DataType::mFlags & uint8_t(1))
            return this->dim(); // skip this node if first bit is set
        //if (!ray.intersects( this->bbox() )) return 1<<TOTAL;
        return ChildNodeType::dim();
    }

}; // LeafNode class

// --------------------------> ReadAccessor <------------------------------------

/// @brief A read-only value acessor with three levels of node caching. This allows for
///        inverse tree traversal during lookup, which is on average significantly faster
///        then the calling the equivalent method on the tree (i.e. top-down traversal).
///
/// @note  By virture of the fact that a value accessor accelerates random access operations
///        by re-using cached access patterns, this access should be reused for multiple access
///        operations. In other words, never create an instace of this calls for a single
///        acccess only. In generate avoid single access operations with this accessor, and
///        if that's not possible call the corresponding method on the tree instead.
///
/// @warning Since this ReadAccessor internally cached raw pointers to the nodes of the tree
///          structure, it is not save to copy between host and device, or even share among
///          multiple threads on the same host or device. However, it's light-weight so simple
///          instantiate one per thread (on the host and/or device).
///
/// @details Used to accelerated random access into a VDB tree. Provides on
/// average O(1) random access operations by means of inverse tree traversal,
/// which amortizes the non-const time complexity of the root node.

template<typename RootT>
class ReadAccessor
{
public:
    using ValueType = typename RootT::ValueType;
    using CoordType = typename RootT::CoordType;
    using CoordValueType = typename RootT::CoordType::ValueType;
    using NodeT3 = const RootT; //                    root node
    using NodeT2 = typename NodeT3::ChildNodeType; // upper internal node
    using NodeT1 = typename NodeT2::ChildNodeType; // lower internal node
    using NodeT0 = typename NodeT1::ChildNodeType; // Leaf node

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
#ifdef USE_SINGLE_ACCESSOR_KEY
        : mKey(CoordType::max())
#else
        : mKeys
    {
        CoordType::max(), CoordType::max(), CoordType::max()
    }
#endif
        , mNode{nullptr, nullptr, nullptr, &root}
    {
    }

    __hostdev__ const RootT& root() const { return *(NodeT3*)mNode[3]; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

    /// @brief Return a const point to the cached node of the specified type
    ///
    /// @warning The return value could be NULL.
    template<typename NodeT>
    __hostdev__ const NodeT* getNode() const
    {
        using T = typename Node<Tree<RootT>, NodeT::LEVEL>::type;
        static_assert(is_same<T, NodeT>::value, "ReadAccessor::getNode: Invalid node type");
        return reinterpret_cast<const T*>(mNode[NodeT::LEVEL]);
    }

#ifdef USE_SINGLE_ACCESSOR_KEY
    template<typename NodeT>
    __hostdev__ bool isCached(CoordValueType dirty) const
    {
        if (!mNode[NodeT::LEVEL])
            return false;
        if (dirty & int32_t(~NodeT::MASK)) {
            mNode[NodeT::LEVEL] = nullptr;
            return false;
        }
        return true;
    }

    __hostdev__ CoordValueType computeDirty(const CoordType& ijk) const
    {
        return (ijk[0] ^ mKey[0]) | (ijk[1] ^ mKey[1]) | (ijk[2] ^ mKey[2]);
    }
#else
    template<typename NodeT>
    __hostdev__ bool isCached(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][0] && (ijk[1] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][1] && (ijk[2] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][2];
    }
#endif

    __hostdev__ const ValueType& getValue(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<NodeT0>(dirty)) {
            return ((NodeT0*)mNode[0])->getValue(ijk);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->getValueAndCache(ijk, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->getValueAndCache(ijk, *this);
        }
        return ((NodeT3*)mNode[3])->getValueAndCache(ijk, *this);
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<NodeT0>(dirty)) {
            return ((NodeT0*)mNode[0])->isActive(ijk);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->isActiveAndCache(ijk, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->isActiveAndCache(ijk, *this);
        }
        return ((NodeT3*)mNode[3])->isActiveAndCache(ijk, *this);
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<NodeT0>(dirty)) {
            return ((NodeT0*)mNode[0])->probeValue(ijk, v);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->probeValueAndCache(ijk, v, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->probeValueAndCache(ijk, v, *this);
        }
        return ((NodeT3*)mNode[3])->probeValueAndCache(ijk, v, *this);
    }

    __hostdev__ const NodeT0* probeLeaf(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<NodeT0>(dirty)) {
            return ((NodeT0*)mNode[0]);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->probeLeafAndCache(ijk, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->probeLeafAndCache(ijk, *this);
        }
        return ((NodeT3*)mNode[3])->probeLeafAndCache(ijk, *this);
    }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<NodeT0>(dirty)) {
            return ((NodeT0*)mNode[0])->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->getDimAndCache(ijk, ray, *this);
        }
        return ((NodeT3*)mNode[3])->getDimAndCache(ijk, ray, *this);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief Inserts a leaf node and key pair into this ReadAccessor
    template<typename NodeT>
    __hostdev__ void insert(const CoordType& ijk, const NodeT* node) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        mKey = ijk;
#else
        mKeys[NodeT::LEVEL] = ijk & ~NodeT::MASK;
#endif
        mNode[NodeT::LEVEL] = node;
    }

    // All member data are mutable to allow for access methods to be const
#ifdef USE_SINGLE_ACCESSOR_KEY // 44 bytes total
    mutable CoordType mKey; // 3*4 = 12 bytes
#else // 68 bytes total
    mutable CoordType mKeys[3]; // 3*3*4 = 36 bytes
#endif
    mutable const void* mNode[4]; // 4*8 = 32 bytes
}; // ReadAccessor class

/// @brief Default configuration used in OpenVDB: Root->32^3->16^3->8^3
template<typename ValueT>
using NanoLeaf = LeafNode<ValueT, Coord, Mask, 3>;
template<typename ValueT>
using NanoNode1 = InternalNode<NanoLeaf<ValueT>, 4>;
template<typename ValueT>
using NanoNode2 = InternalNode<NanoNode1<ValueT>, 5>;
template<typename ValueT>
using NanoRoot = RootNode<NanoNode2<ValueT>>;
template<typename ValueT>
using NanoTree = Tree<NanoRoot<ValueT>>;
template<typename ValueT>
using NanoGrid = Grid<NanoTree<ValueT>>;

using FloatTree = NanoTree<float>;
using DoubleTree = NanoTree<double>;
using Int32Tree = NanoTree<int32_t>;
using UInt32Tree = NanoTree<uint32_t>;
using Int64Tree = NanoTree<int64_t>;
using Vec3fTree = NanoTree<Vec3f>;
using Vec3dTree = NanoTree<Vec3d>;
using MaskTree = NanoTree<ValueMask>;

using FloatGrid = Grid<FloatTree>;
using DoubleGrid = Grid<DoubleTree>;
using Int32Grid = Grid<Int32Tree>;
using UInt32Grid = Grid<UInt32Tree>;
using Int64Grid = Grid<Int64Tree>;
using Vec3fGrid = Grid<Vec3fTree>;
using Vec3dGrid = Grid<Vec3dTree>;
using MaskGrid = Grid<MaskTree>;

/// @brief This is a convenient class that allows for access to grid meta-data
///        that are independent of the value type of a grid. That is, this calls
///        can be used to get information about a grid without actually knowing
///        its ValueType.
class GridMetaData
{
    // We cast to a grid templated on a dummy ValueType which is safe becase we are very
    // careful only to call certain methods which are known to be invariant to the ValueType!
    // In other words, don't use this technique unless you are intimitly familiar with the
    // memory-layout of the data structure and the reasons for why certain methods are safe
    // to call and others are not!
    using GridT = NanoGrid<int>;
    __hostdev__ const GridT& grid() const { return *reinterpret_cast<const GridT*>( this ); }
public:
    __hostdev__ bool isValid() const { return this->grid().isValid(); }
    __hostdev__ const char* gridName() const { return this->grid().gridName(); }
    __hostdev__ GridType gridType() const { return this->grid().gridType(); }
    __hostdev__ GridClass gridClass() const { return this->grid().gridClass(); }
    __hostdev__ bool isLevelSet() const { return this->grid().isLevelSet(); }
    __hostdev__ bool isFogVolume() const { return this->grid().isFogVolume(); }
    __hostdev__ bool isPointIndex() const { return this->grid().isPointIndex(); }
    __hostdev__ bool isPointData() const { return this->grid().isPointData(); }
    __hostdev__ bool isStaggered() const { return this->grid().isStaggered(); }
    __hostdev__ bool isUnknown() const { return this->grid().isUnknown(); }
    __hostdev__ const Map& map() const { return this->grid().map(); }
    __hostdev__ const BBox<Vec3R> worldBBox() const { return this->grid().worldBBox(); }
    __hostdev__ const BBox<Coord>& indexBBox() const { return this->grid().indexBBox(); }
    __hostdev__ double voxelSize() const { return this->grid().voxelSize(); }
    __hostdev__ int blindDataCount() const { return this->grid().blindDataCount(); }
    __hostdev__ const GridBlindMetaData& blindMetaData(int n) const { return this->grid().blindMetaData(n); }
    __hostdev__ uint64_t activeVoxelCount() const { return this->grid().activeVoxelCount(); }
    __hostdev__ uint32_t nodeCount(uint32_t level) const { return this->grid().tree().nodeCount(level);}
};// GridMetaData

/// @brief Class to access points at a specefic voxel location
template<typename AttT>
class PointAccessor : public ReadAccessor<NanoRoot<uint32_t>>
{
    using AccT = ReadAccessor<NanoRoot<uint32_t>>;
    const UInt32Grid* mGrid;
    const AttT*       mData;

public:
    PointAccessor(const UInt32Grid& grid)
        : AccT(grid.tree().root())
        , mGrid(&grid)
        , mData(reinterpret_cast<const AttT*>(grid.blindData(0)))
    {
        assert(grid.gridType() == GridType::UInt32);
        assert((grid.gridClass() == GridClass::PointIndex && is_same<uint32_t, AttT>::value) ||
               (grid.gridClass() == GridClass::PointData && is_same<Vec3f, AttT>::value));
        assert(grid.blindDataCount() >= 1);
    }
    /// @brief Return the total number of point in the grid
    __hostdev__ uint64_t gridPoints(const AttT*& begin, const AttT*& end) const
    {
        const uint64_t count = mGrid->blindMetaData(0).mElementCount;
        begin = mData;
        end = begin + count;
        return count;
    }
    /// @brief Return the number of point in the leaf node containing
    __hostdev__ uint64_t leafPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        auto* leaf = this->probeLeaf(ijk);
        if (leaf == nullptr)
            return 0;
        begin = mData + leaf->valueMin();
        end = begin + leaf->valueMax();
        return leaf->valueMax();
    }

    /// @brief get iterators over offsets to points at a specefic voxel location
    __hostdev__ uint64_t voxelPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        auto* leaf = this->probeLeaf(ijk);
        if (leaf == nullptr)
            return 0;
        const uint32_t offset = NodeT0::CoordToOffset(ijk);
        if (leaf->isActive(offset)) {
            auto* p = mData + leaf->valueMin();
            begin = p + (offset == 0 ? 0 : leaf->getValue(offset - 1));
            end = p + leaf->getValue(offset);
            return end - begin;
        }
        return 0;
    }
}; // PointAccessor

} // namespace nanovdb

#endif // end of NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
