///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#ifndef OPENVDB_MATH_VEC4_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_VEC4_HAS_BEEN_INCLUDED

#include <cmath>
#include <openvdb/Exceptions.h>
#include "Math.h"
#include "Tuple.h"
#include "Vec3.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename T> class Mat3;

template<typename T>
class Vec4: public Tuple<4, T>
{
public:
    typedef T value_type;
    typedef T ValueType;

    /// Trivial constructor, the vector is NOT initialized
    Vec4() {}

    /// Constructor with one argument, e.g.   Vec4f v(0);
    explicit Vec4(T val) { this->mm[0] = this->mm[1] = this->mm[2] = this->mm[3] = val; }

    /// Constructor with four arguments, e.g.   Vec4f v(1,2,3,4);
    Vec4(T x, T y, T z, T w)
    {
        this->mm[0] = x;
        this->mm[1] = y;
        this->mm[2] = z;
        this->mm[3] = w;
    }

    /// Constructor with array argument, e.g.   float a[4]; Vec4f v(a);
    template <typename Source>
    Vec4(Source *a)
    {
        this->mm[0] = a[0];
        this->mm[1] = a[1];
        this->mm[2] = a[2];
        this->mm[3] = a[3];
    }

    /// Conversion constructor
    template<typename Source>
    explicit Vec4(const Tuple<4, Source> &v)
    {
        this->mm[0] = static_cast<T>(v[0]);
        this->mm[1] = static_cast<T>(v[1]);
        this->mm[2] = static_cast<T>(v[2]);
        this->mm[3] = static_cast<T>(v[3]);
    }

    /// Reference to the component, e.g.   v.x() = 4.5f;
    T& x() { return this->mm[0]; }
    T& y() { return this->mm[1]; }
    T& z() { return this->mm[2]; }
    T& w() { return this->mm[3]; }

    /// Get the component, e.g.   float f = v.y();
    T x() const { return this->mm[0]; }
    T y() const { return this->mm[1]; }
    T z() const { return this->mm[2]; }
    T w() const { return this->mm[3]; }

    T* asPointer() { return this->mm; }
    const T* asPointer() const { return this->mm; }

    /// Alternative indexed reference to the elements
    T& operator()(int i) { return this->mm[i]; }

    /// Alternative indexed constant reference to the elements,
    T operator()(int i) const { return this->mm[i]; }

    /// Returns a Vec3 with the first three elements of the Vec4.
    Vec3<T> getVec3() const { return Vec3<T>(this->mm[0], this->mm[1], this->mm[2]); }

    /// "this" vector gets initialized to [x, y, z, w],
    /// calling v.init(); has same effect as calling v = Vec4::zero();
    const Vec4<T>& init(T x=0, T y=0, T z=0, T w=0)
    {
        this->mm[0] = x; this->mm[1] = y; this->mm[2] = z; this->mm[3] = w;
        return *this;
    }

    /// Set "this" vector to zero
    const Vec4<T>& setZero()
    {
        this->mm[0] = 0; this->mm[1] = 0; this->mm[2] = 0; this->mm[3] = 0;
        return *this;
    }

    /// Assignment operator
    template<typename Source>
    const Vec4<T>& operator=(const Vec4<Source> &v)
    {
        // note: don't static_cast because that suppresses warnings
        this->mm[0] = v[0];
        this->mm[1] = v[1];
        this->mm[2] = v[2];
        this->mm[3] = v[3];

        return *this;
    }

    /// Test if "this" vector is equivalent to vector v with tolerance
    /// of eps
    bool eq(const Vec4<T> &v, T eps=1.0e-8) const
    {
        return isApproxEqual(this->mm[0], v.mm[0], eps) &&
            isApproxEqual(this->mm[1], v.mm[1], eps) &&
            isApproxEqual(this->mm[2], v.mm[2], eps) &&
            isApproxEqual(this->mm[3], v.mm[3], eps);
    }

    /// Negation operator, for e.g.   v1 = -v2;
    Vec4<T> operator-() const
    {
        return Vec4<T>(
            -this->mm[0],
            -this->mm[1],
            -this->mm[2],
            -this->mm[3]);
    }

    /// this = v1 + v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.add(v1,v);
    template <typename T0, typename T1>
    const Vec4<T>& add(const Vec4<T0> &v1, const Vec4<T1> &v2)
    {
        this->mm[0] = v1[0] + v2[0];
        this->mm[1] = v1[1] + v2[1];
        this->mm[2] = v1[2] + v2[2];
        this->mm[3] = v1[3] + v2[3];

        return *this;
    }


    /// this = v1 - v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.sub(v1,v);
    template <typename T0, typename T1>
    const Vec4<T>& sub(const Vec4<T0> &v1, const Vec4<T1> &v2)
    {
        this->mm[0] = v1[0] - v2[0];
        this->mm[1] = v1[1] - v2[1];
        this->mm[2] = v1[2] - v2[2];
        this->mm[3] = v1[3] - v2[3];

        return *this;
    }

    /// this =  scalar*v, v need not be a distinct object from "this",
    /// e.g. v.scale(1.5,v1);
    template <typename T0, typename T1>
    const Vec4<T>& scale(T0 scale, const Vec4<T1> &v)
    {
        this->mm[0] = scale * v[0];
        this->mm[1] = scale * v[1];
        this->mm[2] = scale * v[2];
        this->mm[3] = scale * v[3];

        return *this;
    }

    template <typename T0, typename T1>
    const Vec4<T> &div(T0 scalar, const Vec4<T1> &v)
    {
        this->mm[0] = v[0] / scalar;
        this->mm[1] = v[1] / scalar;
        this->mm[2] = v[2] / scalar;
        this->mm[3] = v[3] / scalar;

        return *this;
    }

    /// Dot product
    T dot(const Vec4<T> &v) const
    {
        return (this->mm[0]*v.mm[0] + this->mm[1]*v.mm[1]
            + this->mm[2]*v.mm[2] + this->mm[3]*v.mm[3]);
    }

    /// Length of the vector
    T length() const
    {
        return sqrt(
            this->mm[0]*this->mm[0] +
            this->mm[1]*this->mm[1] +
            this->mm[2]*this->mm[2] +
            this->mm[3]*this->mm[3]);
    }


    /// Squared length of the vector, much faster than length() as it
    /// does not involve square root
    T lengthSqr() const
    {
        return (this->mm[0]*this->mm[0] + this->mm[1]*this->mm[1]
            + this->mm[2]*this->mm[2] + this->mm[3]*this->mm[3]);
    }

    /// Return a reference to itsef after the exponent has been
    /// applied to all the vector components.
    inline const Vec4<T>& exp()
    {
        this->mm[0] = std::exp(this->mm[0]);
        this->mm[1] = std::exp(this->mm[1]);
        this->mm[2] = std::exp(this->mm[2]);
        this->mm[3] = std::exp(this->mm[3]);
        return *this;
    }

    /// Return the sum of all the vector components.
    inline T sum() const
    {
        return this->mm[0] + this->mm[1] + this->mm[2] + this->mm[3];
    }


    /// this = normalized this
    bool normalize(T eps=1.0e-8)
    {
        T d = length();
        if (isApproxEqual(d, T(0), eps)) {
            return false;
        }
        *this *= (T(1) / d);
        return true;
    }

    /// return normalized this, throws if null vector
    Vec4<T> unit(T eps=0) const
    {
        T d;
        return unit(eps, d);
    }

    /// return normalized this and length, throws if null vector
    Vec4<T> unit(T eps, T& len) const
    {
        len = length();
        if (isApproxEqual(len, T(0), eps)) {
            throw ArithmeticError("Normalizing null 4-vector");
        }
        return *this / len;
    }

    /// Returns v, where \f$v_i *= scalar\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator*=(S scalar)
    {
        this->mm[0] *= scalar;
        this->mm[1] *= scalar;
        this->mm[2] *= scalar;
        this->mm[3] *= scalar;
        return *this;
    }

    /// Returns v0, where \f$v0_i *= v1_i\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator*=(const Vec4<S> &v1)
    {
        this->mm[0] *= v1[0];
        this->mm[1] *= v1[1];
        this->mm[2] *= v1[2];
        this->mm[3] *= v1[3];

        return *this;
    }

    /// Returns v, where \f$v_i /= scalar\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator/=(S scalar)
    {
        this->mm[0] /= scalar;
        this->mm[1] /= scalar;
        this->mm[2] /= scalar;
        this->mm[3] /= scalar;
        return *this;
    }

    /// Returns v0, where \f$v0_i /= v1_i\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator/=(const Vec4<S> &v1)
    {
        this->mm[0] /= v1[0];
        this->mm[1] /= v1[1];
        this->mm[2] /= v1[2];
        this->mm[3] /= v1[3];
        return *this;
    }

    /// Returns v, where \f$v_i += scalar\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator+=(S scalar)
    {
        this->mm[0] += scalar;
        this->mm[1] += scalar;
        this->mm[2] += scalar;
        this->mm[3] += scalar;
        return *this;
    }

    /// Returns v0, where \f$v0_i += v1_i\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator+=(const Vec4<S> &v1)
    {
        this->mm[0] += v1[0];
        this->mm[1] += v1[1];
        this->mm[2] += v1[2];
        this->mm[3] += v1[3];
        return *this;
    }

    /// Returns v, where \f$v_i += scalar\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator-=(S scalar)
    {
        this->mm[0] -= scalar;
        this->mm[1] -= scalar;
        this->mm[2] -= scalar;
        this->mm[3] -= scalar;
        return *this;
    }

    /// Returns v0, where \f$v0_i -= v1_i\f$ for \f$i \in [0, 3]\f$
    template <typename S>
    const Vec4<T> &operator-=(const Vec4<S> &v1)
    {
        this->mm[0] -= v1[0];
        this->mm[1] -= v1[1];
        this->mm[2] -= v1[2];
        this->mm[3] -= v1[3];
        return *this;
    }

    // Number of cols, rows, elements
    static unsigned numRows() { return 1; }
    static unsigned numColumns()  { return 4; }
    static unsigned numElements()  { return 4; }

    /// True if a Nan is present in vector
    bool isNan() const
    {
        return isnan(this->mm[0]) || isnan(this->mm[1])
            || isnan(this->mm[2]) || isnan(this->mm[3]);
    }

    /// True if an Inf is present in vector
    bool isInfinite() const
    {
        return isinf(this->mm[0]) || isinf(this->mm[1])
            || isinf(this->mm[2]) || isinf(this->mm[3]);
    }

    /// True if all no Nan or Inf values present
    bool isFinite() const
    {
        return finite(this->mm[0]) && finite(this->mm[1])
            && finite(this->mm[2]) && finite(this->mm[3]);
    }

    /// Predefined constants, e.g.   Vec4f v = Vec4f::xNegAxis();
    static Vec4<T> zero() { return Vec4<T>(0, 0, 0, 0); }
    static Vec4<T> origin() { return Vec4<T>(0, 0, 0, 1); }
};

/// Equality operator, does exact floating point comparisons
template <typename T0, typename T1>
inline bool operator==(const Vec4<T0> &v0, const Vec4<T1> &v1)
{
    return
        isExactlyEqual(v0[0], v1[0]) &&
        isExactlyEqual(v0[1], v1[1]) &&
        isExactlyEqual(v0[2], v1[2]) &&
        isExactlyEqual(v0[3], v1[3]);
}

/// Inequality operator, does exact floating point comparisons
template <typename T0, typename T1>
inline bool operator!=(const Vec4<T0> &v0, const Vec4<T1> &v1) { return !(v0==v1); }

/// Returns V, where \f$V_i = v_i * scalar\f$ for \f$i \in [0, 3]\f$
template <typename S, typename T>
inline Vec4<typename promote<S, T>::type> operator*(S scalar, const Vec4<T> &v)
{ return v*scalar; }

/// Returns V, where \f$V_i = v_i * scalar\f$ for \f$i \in [0, 3]\f$
template <typename S, typename T>
inline Vec4<typename promote<S, T>::type> operator*(const Vec4<T> &v, S scalar)
{
    Vec4<typename promote<S, T>::type> result(v);
    result *= scalar;
    return result;
}

/// Returns V, where \f$V_i = v0_i * v1_i\f$ for \f$i \in [0, 3]\f$
template <typename T0, typename T1>
inline Vec4<typename promote<T0, T1>::type> operator*(const Vec4<T0> &v0,
                                               const Vec4<T1> &v1)
{
    Vec4<typename promote<T0, T1>::type> result(v0[0]*v1[0],
                                                v0[1]*v1[1],
                                                v0[2]*v1[2],
                                                v0[3]*v1[3]);
    return result;
}

/// Returns V, where \f$V_i = scalar / v_i\f$ for \f$i \in [0, 3]\f$
template <typename S, typename T>
inline Vec4<typename promote<S, T>::type> operator/(S scalar, const Vec4<T> &v)
{
    return Vec4<typename promote<S, T>::type>(scalar/v[0],
                                              scalar/v[1],
                                              scalar/v[2],
                                              scalar/v[3]);
}

/// Returns V, where \f$V_i = v_i / scalar\f$ for \f$i \in [0, 3]\f$
template <typename S, typename T>
inline Vec4<typename promote<S, T>::type> operator/(const Vec4<T> &v, S scalar)
{
    Vec4<typename promote<S, T>::type> result(v);
    result /= scalar;
    return result;
}

/// Returns V, where \f$V_i = v0_i / v1_i\f$ for \f$i \in [0, 3]\f$
template <typename T0, typename T1>
inline Vec4<typename promote<T0, T1>::type> operator/(const Vec4<T0> &v0,
                                               const Vec4<T1> &v1)
{
    Vec4<typename promote<T0, T1>::type>
        result(v0[0]/v1[0], v0[1]/v1[1], v0[2]/v1[2], v0[3]/v1[3]);
    return result;
}

/// Returns V, where \f$V_i = v0_i + v1_i\f$ for \f$i \in [0, 3]\f$
template <typename T0, typename T1>
inline Vec4<typename promote<T0, T1>::type> operator+(const Vec4<T0> &v0, const Vec4<T1> &v1)
{
    Vec4<typename promote<T0, T1>::type> result(v0);
    result += v1;
    return result;
}

/// Returns V, where \f$V_i = v_i + scalar\f$ for \f$i \in [0, 3]\f$
template <typename S, typename T>
inline Vec4<typename promote<S, T>::type> operator+(const Vec4<T> &v, S scalar)
{
    Vec4<typename promote<S, T>::type> result(v);
    result += scalar;
    return result;
}

/// Returns V, where \f$V_i = v0_i - v1_i\f$ for \f$i \in [0, 3]\f$
template <typename T0, typename T1>
inline Vec4<typename promote<T0, T1>::type> operator-(const Vec4<T0> &v0, const Vec4<T1> &v1)
{
    Vec4<typename promote<T0, T1>::type> result(v0);
    result -= v1;
    return result;
}

/// Returns V, where \f$V_i = v_i - scalar\f$ for \f$i \in [0, 3]\f$
template <typename S, typename T>
inline Vec4<typename promote<S, T>::type> operator-(const Vec4<T> &v, S scalar)
{
    Vec4<typename promote<S, T>::type> result(v);
    result -= scalar;
    return result;
}

template <typename T>
inline bool
isApproxEqual(const Vec4<T>& a, const Vec4<T>& b)
{
    return a.eq(b);
}
template <typename T>
inline bool
isApproxEqual(const Vec4<T>& a, const Vec4<T>& b, const Vec4<T>& eps)
{
    return isApproxEqual(a[0], b[0], eps[0]) &&
           isApproxEqual(a[1], b[1], eps[1]) &&
           isApproxEqual(a[2], b[2], eps[2]) &&
           isApproxEqual(a[3], b[3], eps[3]);
}

template<typename T>
inline bool
isFinite(const Vec4<T>& v)
{
    return isFinite(v[0]) && isFinite(v[1]) && isFinite(v[2]) && isFinite(v[3]);
}

/// Return @c true if all components are exactly equal to zero.
template<typename T>
inline bool
isZero(const Vec4<T>& v)
{
    return isZero(v[0]) && isZero(v[1]) && isZero(v[2]) && isZero(v[3]);
}

template<typename T>
inline Vec4<T>
Abs(const Vec4<T>& v)
{
    return Vec4<T>(Abs(v[0]), Abs(v[1]), Abs(v[2]), Abs(v[3]));
}

/// @remark We are switching to a more explicit name because the semantics
/// are different from std::min/max. In that case, the function returns a
/// reference to one of the objects based on a comparator. Here, we must
/// fabricate a new object which might not match either of the inputs.

/// Return component-wise minimum of the two vectors.
template <typename T>
inline Vec4<T> minComponent(const Vec4<T> &v1, const Vec4<T> &v2)
{
    return Vec4<T>(
            std::min(v1.x(), v2.x()),
            std::min(v1.y(), v2.y()),
            std::min(v1.z(), v2.z()),
            std::min(v1.w(), v2.w()));
}

/// Return component-wise maximum of the two vectors.
template <typename T>
inline Vec4<T> maxComponent(const Vec4<T> &v1, const Vec4<T> &v2)
{
    return Vec4<T>(
            std::max(v1.x(), v2.x()),
            std::max(v1.y(), v2.y()),
            std::max(v1.z(), v2.z()),
            std::max(v1.w(), v2.w()));
}

/// @brief Return a vector with the exponent applied to each of
/// the components of the input vector.
template <typename T>
inline Vec4<T> Exp(Vec4<T> v) { return v.exp(); }

typedef Vec4<int32_t>   Vec4i;
typedef Vec4<uint32_t>  Vec4ui;
typedef Vec4<float>     Vec4s;
typedef Vec4<double>    Vec4d;

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_VEC4_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
