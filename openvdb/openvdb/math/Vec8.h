// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_MATH_VEC8_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_VEC8_HAS_BEEN_INCLUDED

#include <openvdb/Exceptions.h>
#include "Math.h"
#include "Tuple.h"
#include "Vec3.h"
#include <algorithm>
#include <cmath>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename T> class Mat3;

template<typename T>
class Vec8: public Tuple<8, T>
{
public:
    using value_type = T;
    using ValueType = T;

    /// Trivial constructor, the vector is NOT initialized
    /// @note destructor, copy constructor, assignment operator and
    ///   move constructor are left to be defined by the compiler (default)
    Vec8() = default;

    /// @brief Construct a vector all of whose components have the given value.
    explicit Vec8(T val) { this->mm[0] = this->mm[1] = this->mm[2] = this->mm[3] = this->mm[4] = this->mm[5] = this->mm[6] = this->mm[7] = val; }

    /// Constructor with four arguments, e.g.   Vec8f v(1,2,3,4,5,6,7,8);
    Vec8(T x, T y, T z, T w, T p, T q, T r, T s)
    {
        this->mm[0] = x;
        this->mm[1] = y;
        this->mm[2] = z;
        this->mm[3] = w;
        this->mm[4] = p;
        this->mm[5] = q;
        this->mm[6] = r;
        this->mm[7] = s;
    }

    /// Constructor with array argument, e.g.   float a[8]; Vec8f v(a);
    template <typename Source>
    Vec8(Source *a)
    {
        this->mm[0] = static_cast<T>(a[0]);
        this->mm[1] = static_cast<T>(a[1]);
        this->mm[2] = static_cast<T>(a[2]);
        this->mm[3] = static_cast<T>(a[3]);
        this->mm[4] = static_cast<T>(a[4]);
        this->mm[5] = static_cast<T>(a[5]);
        this->mm[6] = static_cast<T>(a[6]);
        this->mm[7] = static_cast<T>(a[7]);
    }

    /// Conversion constructor
    template<typename Source>
    explicit Vec8(const Tuple<8, Source> &v)
    {
        this->mm[0] = static_cast<T>(v[0]);
        this->mm[1] = static_cast<T>(v[1]);
        this->mm[2] = static_cast<T>(v[2]);
        this->mm[3] = static_cast<T>(v[3]);
        this->mm[4] = static_cast<T>(v[4]);
        this->mm[5] = static_cast<T>(v[5]);
        this->mm[6] = static_cast<T>(v[6]);
        this->mm[7] = static_cast<T>(v[7]);
    }

    /// @brief Construct a vector all of whose components have the given value,
    /// which may be of an arithmetic type different from this vector's value type.
    /// @details Type conversion warnings are suppressed.
    template<typename Other>
    explicit Vec8(Other val,
        typename std::enable_if<std::is_arithmetic<Other>::value, Conversion>::type = Conversion{})
    {
        this->mm[0] = this->mm[1] = this->mm[2] = this->mm[3] = static_cast<T>(val);
        this->mm[4] = this->mm[5] = this->mm[6] = this->mm[7] = static_cast<T>(val);
    }

    /// Reference to the component, e.g.   v.x() = 4.5f;
    T& x() { return this->mm[0]; }
    T& y() { return this->mm[1]; }
    T& z() { return this->mm[2]; }
    T& w() { return this->mm[3]; }
    T& p() { return this->mm[4]; }
    T& q() { return this->mm[5]; }
    T& r() { return this->mm[6]; }
    T& s() { return this->mm[7]; }

    /// Get the component, e.g.   float f = v.y();
    T x() const { return this->mm[0]; }
    T y() const { return this->mm[1]; }
    T z() const { return this->mm[2]; }
    T w() const { return this->mm[3]; }
    T p() const { return this->mm[4]; }
    T q() const { return this->mm[5]; }
    T r() const { return this->mm[6]; }
    T s() const { return this->mm[7]; }

    T* asPointer() { return this->mm; }
    const T* asPointer() const { return this->mm; }

    /// Alternative indexed reference to the elements
    T& operator()(int i) { return this->mm[i]; }

    /// Alternative indexed constant reference to the elements,
    T operator()(int i) const { return this->mm[i]; }

    /// Returns a Vec3 with the first three elements of the Vec8.
    Vec3<T> getVec3() const { return Vec3<T>(this->mm[0], this->mm[1], this->mm[2]); }

    /// "this" vector gets initialized to [x, y, z, w, p, q, r, s],
    /// calling v.init(); has same effect as calling v = Vec8::zero();
    const Vec8<T>& init(T x=0, T y=0, T z=0, T w=0, T p=0, T q=0, T r=0, T s=0)
    {
        this->mm[0] = x; this->mm[1] = y; this->mm[2] = z; this->mm[3] = w;
        this->mm[4] = p; this->mm[5] = q; this->mm[6] = r; this->mm[7] = s;
        return *this;
    }

    /// Set "this" vector to zero
    const Vec8<T>& setZero()
    {
        this->mm[0] = 0; this->mm[1] = 0; this->mm[2] = 0; this->mm[3] = 0;
        this->mm[4] = 0; this->mm[5] = 0; this->mm[6] = 0; this->mm[7] = 0;
        return *this;
    }

    /// Assignment operator
    template<typename Source>
    const Vec8<T>& operator=(const Vec8<Source> &v)
    {
        // note: don't static_cast because that suppresses warnings
        this->mm[0] = v[0];
        this->mm[1] = v[1];
        this->mm[2] = v[2];
        this->mm[3] = v[3];
        this->mm[4] = v[4];
        this->mm[5] = v[5];
        this->mm[6] = v[6];
        this->mm[7] = v[7];

        return *this;
    }

    /// Test if "this" vector is equivalent to vector v with tolerance
    /// of eps
    bool eq(const Vec8<T> &v, T eps = static_cast<T>(1.0e-8)) const
    {
        return isApproxEqual(this->mm[0], v.mm[0], eps) &&
            isApproxEqual(this->mm[1], v.mm[1], eps) &&
            isApproxEqual(this->mm[2], v.mm[2], eps) &&
            isApproxEqual(this->mm[3], v.mm[3], eps) &&
            isApproxEqual(this->mm[4], v.mm[4], eps) &&
            isApproxEqual(this->mm[5], v.mm[5], eps) &&
            isApproxEqual(this->mm[6], v.mm[6], eps) &&
            isApproxEqual(this->mm[7], v.mm[7], eps);
    }

    /// Negation operator, for e.g.   v1 = -v2;
    Vec8<T> operator-() const
    {
        return Vec8<T>(
            -this->mm[0],
            -this->mm[1],
            -this->mm[2],
            -this->mm[3]
            -this->mm[4],
            -this->mm[5],
            -this->mm[6],
            -this->mm[7]);
    }

    /// this = v1 + v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.add(v1,v);
    template <typename T0, typename T1>
    const Vec8<T>& add(const Vec8<T0> &v1, const Vec8<T1> &v2)
    {
        this->mm[0] = v1[0] + v2[0];
        this->mm[1] = v1[1] + v2[1];
        this->mm[2] = v1[2] + v2[2];
        this->mm[3] = v1[3] + v2[3];
        this->mm[4] = v1[4] + v2[4];
        this->mm[5] = v1[5] + v2[5];
        this->mm[6] = v1[6] + v2[6];
        this->mm[7] = v1[7] + v2[7];

        return *this;
    }


    /// this = v1 - v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.sub(v1,v);
    template <typename T0, typename T1>
    const Vec8<T>& sub(const Vec8<T0> &v1, const Vec8<T1> &v2)
    {
        this->mm[0] = v1[0] - v2[0];
        this->mm[1] = v1[1] - v2[1];
        this->mm[2] = v1[2] - v2[2];
        this->mm[3] = v1[3] - v2[3];
        this->mm[4] = v1[4] - v2[4];
        this->mm[5] = v1[5] - v2[5];
        this->mm[6] = v1[6] - v2[6];
        this->mm[7] = v1[7] - v2[7];

        return *this;
    }

    /// this =  scalar*v, v need not be a distinct object from "this",
    /// e.g. v.scale(1.5,v1);
    template <typename T0, typename T1>
    const Vec8<T>& scale(T0 scale, const Vec8<T1> &v)
    {
        this->mm[0] = scale * v[0];
        this->mm[1] = scale * v[1];
        this->mm[2] = scale * v[2];
        this->mm[3] = scale * v[3];
        this->mm[4] = scale * v[4];
        this->mm[5] = scale * v[5];
        this->mm[6] = scale * v[6];
        this->mm[7] = scale * v[7];

        return *this;
    }

    template <typename T0, typename T1>
    const Vec8<T> &div(T0 scalar, const Vec8<T1> &v)
    {
        this->mm[0] = v[0] / scalar;
        this->mm[1] = v[1] / scalar;
        this->mm[2] = v[2] / scalar;
        this->mm[3] = v[3] / scalar;
        this->mm[4] = v[4] / scalar;
        this->mm[5] = v[5] / scalar;
        this->mm[6] = v[6] / scalar;
        this->mm[7] = v[7] / scalar;

        return *this;
    }

    /// Dot product
    T dot(const Vec8<T> &v) const
    {
        return (this->mm[0]*v.mm[0] + this->mm[1]*v.mm[1]
            + this->mm[2]*v.mm[2] + this->mm[3]*v.mm[3]
            + this->mm[4]*v.mm[4] + this->mm[5]*v.mm[5]
            + this->mm[6]*v.mm[6] + this->mm[7]*v.mm[7]);
    }

    /// Length of the vector
    T length() const
    {
        return std::sqrt(
            this->mm[0]*this->mm[0] +
            this->mm[1]*this->mm[1] +
            this->mm[2]*this->mm[2] +
            this->mm[3]*this->mm[3] +
            this->mm[4]*this->mm[4] +
            this->mm[5]*this->mm[5] +
            this->mm[6]*this->mm[6] +
            this->mm[7]*this->mm[7]);
    }


    /// Squared length of the vector, much faster than length() as it
    /// does not involve square root
    T lengthSqr() const
    {
        return (this->mm[0]*this->mm[0] + this->mm[1]*this->mm[1]
            + this->mm[2]*this->mm[2] + this->mm[3]*this->mm[3]
            + this->mm[4]*this->mm[4] + this->mm[5]*this->mm[5]
            + this->mm[6]*this->mm[6] + this->mm[7]*this->mm[7]);
    }

    /// Return a reference to itself after the exponent has been
    /// applied to all the vector components.
    inline const Vec8<T>& exp()
    {
        this->mm[0] = std::exp(this->mm[0]);
        this->mm[1] = std::exp(this->mm[1]);
        this->mm[2] = std::exp(this->mm[2]);
        this->mm[3] = std::exp(this->mm[3]);
        this->mm[4] = std::exp(this->mm[4]);
        this->mm[5] = std::exp(this->mm[5]);
        this->mm[6] = std::exp(this->mm[6]);
        this->mm[7] = std::exp(this->mm[7]);
        return *this;
    }

    /// Return a reference to itself after log has been
    /// applied to all the vector components.
    inline const Vec8<T>& log()
    {
        this->mm[0] = std::log(this->mm[0]);
        this->mm[1] = std::log(this->mm[1]);
        this->mm[2] = std::log(this->mm[2]);
        this->mm[3] = std::log(this->mm[3]);
        this->mm[4] = std::log(this->mm[4]);
        this->mm[5] = std::log(this->mm[5]);
        this->mm[6] = std::log(this->mm[6]);
        this->mm[7] = std::log(this->mm[7]);
        return *this;
    }

    /// Return the sum of all the vector components.
    inline T sum() const
    {
        return this->mm[0] + this->mm[1] + this->mm[2] + this->mm[3]
        + this->mm[4] + this->mm[5] + this->mm[6] + this->mm[7];
    }

    /// Return the product of all the vector components.
    inline T product() const
    {
        return this->mm[0] * this->mm[1] * this->mm[2] * this->mm[3]
        * this->mm[4] * this->mm[5] * this->mm[6] * this->mm[7];
    }

    /// this = normalized this
    bool normalize(T eps = static_cast<T>(1.0e-8))
    {
        T d = length();
        if (isApproxEqual(d, T(0), eps)) {
            return false;
        }
        *this *= (T(1) / d);
        return true;
    }

    /// return normalized this, throws if null vector
    Vec8<T> unit(T eps=0) const
    {
        T d;
        return unit(eps, d);
    }

    /// return normalized this and length, throws if null vector
    Vec8<T> unit(T eps, T& len) const
    {
        len = length();
        if (isApproxEqual(len, T(0), eps)) {
            throw ArithmeticError("Normalizing null 8-vector");
        }
        return *this / len;
    }

    /// return normalized this, or (1, 0, 0, 0, 0, 0, 0, 0) if this is null vector
    Vec8<T> unitSafe() const
    {
        T l2 = lengthSqr();
        return l2 ? *this / static_cast<T>(sqrt(l2)) : Vec8<T>(1, 0, 0, 0, 0, 0, 0, 0);
    }

    /// Multiply each element of this vector by @a scalar.
    template <typename S>
    const Vec8<T> &operator*=(S scalar)
    {
        this->mm[0] *= scalar;
        this->mm[1] *= scalar;
        this->mm[2] *= scalar;
        this->mm[3] *= scalar;
        this->mm[4] *= scalar;
        this->mm[5] *= scalar;
        this->mm[6] *= scalar;
        this->mm[7] *= scalar;
        return *this;
    }

    /// Multiply each element of this vector by the corresponding element of the given vector.
    template <typename S>
    const Vec8<T> &operator*=(const Vec8<S> &v1)
    {
        this->mm[0] *= v1[0];
        this->mm[1] *= v1[1];
        this->mm[2] *= v1[2];
        this->mm[3] *= v1[3];
        this->mm[4] *= v1[4];
        this->mm[5] *= v1[5];
        this->mm[6] *= v1[6];
        this->mm[7] *= v1[7];

        return *this;
    }

    /// Divide each element of this vector by @a scalar.
    template <typename S>
    const Vec8<T> &operator/=(S scalar)
    {
        this->mm[0] /= scalar;
        this->mm[1] /= scalar;
        this->mm[2] /= scalar;
        this->mm[3] /= scalar;
        this->mm[4] /= scalar;
        this->mm[5] /= scalar;
        this->mm[6] /= scalar;
        this->mm[7] /= scalar;
        return *this;
    }

    /// Divide each element of this vector by the corresponding element of the given vector.
    template <typename S>
    const Vec8<T> &operator/=(const Vec8<S> &v1)
    {
        this->mm[0] /= v1[0];
        this->mm[1] /= v1[1];
        this->mm[2] /= v1[2];
        this->mm[3] /= v1[3];
        this->mm[4] /= v1[4];
        this->mm[5] /= v1[5];
        this->mm[6] /= v1[6];
        this->mm[7] /= v1[7];
        return *this;
    }

    /// Add @a scalar to each element of this vector.
    template <typename S>
    const Vec8<T> &operator+=(S scalar)
    {
        this->mm[0] += scalar;
        this->mm[1] += scalar;
        this->mm[2] += scalar;
        this->mm[3] += scalar;
        this->mm[4] += scalar;
        this->mm[5] += scalar;
        this->mm[6] += scalar;
        this->mm[7] += scalar;
        return *this;
    }

    /// Add each element of the given vector to the corresponding element of this vector.
    template <typename S>
    const Vec8<T> &operator+=(const Vec8<S> &v1)
    {
        this->mm[0] += v1[0];
        this->mm[1] += v1[1];
        this->mm[2] += v1[2];
        this->mm[3] += v1[3];
        this->mm[4] += v1[4];
        this->mm[5] += v1[5];
        this->mm[6] += v1[6];
        this->mm[7] += v1[7];
        return *this;
    }

    /// Subtract @a scalar from each element of this vector.
    template <typename S>
    const Vec8<T> &operator-=(S scalar)
    {
        this->mm[0] -= scalar;
        this->mm[1] -= scalar;
        this->mm[2] -= scalar;
        this->mm[3] -= scalar;
        this->mm[4] -= scalar;
        this->mm[5] -= scalar;
        this->mm[6] -= scalar;
        this->mm[7] -= scalar;
        return *this;
    }

    /// Subtract each element of the given vector from the corresponding element of this vector.
    template <typename S>
    const Vec8<T> &operator-=(const Vec8<S> &v1)
    {
        this->mm[0] -= v1[0];
        this->mm[1] -= v1[1];
        this->mm[2] -= v1[2];
        this->mm[3] -= v1[3];
        this->mm[4] -= v1[4];
        this->mm[5] -= v1[5];
        this->mm[6] -= v1[6];
        this->mm[7] -= v1[7];
        return *this;
    }

    // Number of cols, rows, elements
    static unsigned numRows() { return 1; }
    static unsigned numColumns()  { return 8; }
    static unsigned numElements()  { return 8; }

    /// Predefined constants, e.g.   Vec8f v = Vec8f::xNegAxis();
    static Vec8<T> zero() { return Vec8<T>(0, 0, 0, 0, 0, 0, 0, 0); }
    static Vec8<T> origin() { return Vec8<T>(0, 0, 0, 1, 1, 1, 1, 1); } // TODO: Is this correct?
    static Vec8<T> ones() { return Vec8<T>(1, 1, 1, 1, 1, 1, 1, 1); }
};

/// Equality operator, does exact floating point comparisons
template <typename T0, typename T1>
inline bool operator==(const Vec8<T0> &v0, const Vec8<T1> &v1)
{
    return
        isExactlyEqual(v0[0], v1[0]) &&
        isExactlyEqual(v0[1], v1[1]) &&
        isExactlyEqual(v0[2], v1[2]) &&
        isExactlyEqual(v0[3], v1[3]) &&
        isExactlyEqual(v0[4], v1[4]) &&
        isExactlyEqual(v0[5], v1[5]) &&
        isExactlyEqual(v0[6], v1[6]) &&
        isExactlyEqual(v0[7], v1[7]);
}

/// Inequality operator, does exact floating point comparisons
template <typename T0, typename T1>
inline bool operator!=(const Vec8<T0> &v0, const Vec8<T1> &v1) { return !(v0==v1); }

/// Multiply each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec8<typename promote<S, T>::type> operator*(S scalar, const Vec8<T> &v)
{ return v*scalar; }

/// Multiply each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec8<typename promote<S, T>::type> operator*(const Vec8<T> &v, S scalar)
{
    Vec8<typename promote<S, T>::type> result(v);
    result *= scalar;
    return result;
}

/// Multiply corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec8<typename promote<T0, T1>::type> operator*(const Vec8<T0> &v0, const Vec8<T1> &v1)
{
    Vec8<typename promote<T0, T1>::type> result(v0[0]*v1[0],
                                                v0[1]*v1[1],
                                                v0[2]*v1[2],
                                                v0[3]*v1[3],
                                                v0[4]*v1[4],
                                                v0[5]*v1[5],
                                                v0[6]*v1[6],
                                                v0[7]*v1[7]);
    return result;
}

/// Divide @a scalar by each element of the given vector and return the result.
template <typename S, typename T>
inline Vec8<typename promote<S, T>::type> operator/(S scalar, const Vec8<T> &v)
{
    return Vec8<typename promote<S, T>::type>(scalar/v[0],
                                              scalar/v[1],
                                              scalar/v[2],
                                              scalar/v[3],
                                              scalar/v[4],
                                              scalar/v[5],
                                              scalar/v[6],
                                              scalar/v[7]);
}

/// Divide each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec8<typename promote<S, T>::type> operator/(const Vec8<T> &v, S scalar)
{
    Vec8<typename promote<S, T>::type> result(v);
    result /= scalar;
    return result;
}

/// Divide corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec8<typename promote<T0, T1>::type> operator/(const Vec8<T0> &v0, const Vec8<T1> &v1)
{
    Vec8<typename promote<T0, T1>::type>
        result(v0[0]/v1[0], v0[1]/v1[1], v0[2]/v1[2], v0[3]/v1[3], v0[4]/v1[4], v0[5]/v1[5], v0[6]/v1[6], v0[7]/v1[7]);
    return result;
}

/// Add corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec8<typename promote<T0, T1>::type> operator+(const Vec8<T0> &v0, const Vec8<T1> &v1)
{
    Vec8<typename promote<T0, T1>::type> result(v0);
    result += v1;
    return result;
}

/// Add @a scalar to each element of the given vector and return the result.
template <typename S, typename T>
inline Vec8<typename promote<S, T>::type> operator+(const Vec8<T> &v, S scalar)
{
    Vec8<typename promote<S, T>::type> result(v);
    result += scalar;
    return result;
}

/// Subtract corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec8<typename promote<T0, T1>::type> operator-(const Vec8<T0> &v0, const Vec8<T1> &v1)
{
    Vec8<typename promote<T0, T1>::type> result(v0);
    result -= v1;
    return result;
}

/// Subtract @a scalar from each element of the given vector and return the result.
template <typename S, typename T>
inline Vec8<typename promote<S, T>::type> operator-(const Vec8<T> &v, S scalar)
{
    Vec8<typename promote<S, T>::type> result(v);
    result -= scalar;
    return result;
}

template <typename T>
inline bool
isApproxEqual(const Vec8<T>& a, const Vec8<T>& b)
{
    return a.eq(b);
}
template <typename T>
inline bool
isApproxEqual(const Vec8<T>& a, const Vec8<T>& b, const Vec8<T>& eps)
{
    return isApproxEqual(a[0], b[0], eps[0]) &&
           isApproxEqual(a[1], b[1], eps[1]) &&
           isApproxEqual(a[2], b[2], eps[2]) &&
           isApproxEqual(a[3], b[3], eps[3]) &&
           isApproxEqual(a[4], b[4], eps[4]) &&
           isApproxEqual(a[5], b[5], eps[5]) &&
           isApproxEqual(a[6], b[6], eps[6]) &&
           isApproxEqual(a[7], b[7], eps[7]);
}

template<typename T>
inline Vec8<T>
Abs(const Vec8<T>& v)
{
    return Vec8<T>(Abs(v[0]), Abs(v[1]), Abs(v[2]), Abs(v[3]), Abs(v[4]), Abs(v[5]), Abs(v[6]), Abs(v[7]));
}

/// @remark We are switching to a more explicit name because the semantics
/// are different from std::min/max. In that case, the function returns a
/// reference to one of the objects based on a comparator. Here, we must
/// fabricate a new object which might not match either of the inputs.

/// Return component-wise minimum of the two vectors.
template <typename T>
inline Vec8<T> minComponent(const Vec8<T> &v1, const Vec8<T> &v2)
{
    return Vec8<T>(
            std::min(v1.x(), v2.x()),
            std::min(v1.y(), v2.y()),
            std::min(v1.z(), v2.z()),
            std::min(v1.w(), v2.w()),
            std::min(v1.p(), v2.p()),
            std::min(v1.q(), v2.q()),
            std::min(v1.r(), v2.r()),
            std::min(v1.s(), v2.s()));
}

/// Return component-wise maximum of the two vectors.
template <typename T>
inline Vec8<T> maxComponent(const Vec8<T> &v1, const Vec8<T> &v2)
{
    return Vec8<T>(
            std::max(v1.x(), v2.x()),
            std::max(v1.y(), v2.y()),
            std::max(v1.z(), v2.z()),
            std::max(v1.w(), v2.w()),
            std::max(v1.p(), v2.p()),
            std::max(v1.q(), v2.q()),
            std::max(v1.r(), v2.r()),
            std::max(v1.s(), v2.s()));
}

/// @brief Return a vector with the exponent applied to each of
/// the components of the input vector.
template <typename T>
inline Vec8<T> Exp(Vec8<T> v) { return v.exp(); }

/// @brief Return a vector with log applied to each of
/// the components of the input vector.
template <typename T>
inline Vec8<T> Log(Vec8<T> v) { return v.log(); }

using Vec8i = Vec8<int32_t>;
using Vec8ui = Vec8<uint32_t>;
using Vec8s = Vec8<float>;
using Vec8d = Vec8<double>;

OPENVDB_IS_POD(Vec8i)
OPENVDB_IS_POD(Vec8ui)
OPENVDB_IS_POD(Vec8s)
OPENVDB_IS_POD(Vec8d)

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_VEC8_HAS_BEEN_INCLUDED
