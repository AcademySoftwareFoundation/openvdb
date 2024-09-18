// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file Tuple.h
/// @author Ben Kwa

#ifndef OPENVDB_MATH_TUPLE_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_TUPLE_HAS_BEEN_INCLUDED

#include "Math.h"
#include <openvdb/util/Assert.h>
#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @brief Dummy class for tag dispatch of conversion constructors
struct Conversion {};


/// @class Tuple "Tuple.h"
/// A base class for homogenous tuple types
template<int SIZE, typename T>
class Tuple
{
public:
    using value_type = T;
    using ValueType = T;

    static const int size = SIZE;

    /// Trivial constructor, the Tuple is NOT initialized
    /// @note destructor, copy constructor, assignment operator and
    ///   move constructor are left to be defined by the compiler (default)
    Tuple() = default;

    /// @brief Conversion constructor.
    /// @details Tuples with different value types and different sizes can be
    /// interconverted using this member.  Converting from a larger tuple
    /// results in truncation; converting from a smaller tuple results in
    /// the extra data members being zeroed out.  This function assumes that
    /// the integer 0 is convertible to the tuple's value type.
    template <int src_size, typename src_valtype>
    explicit Tuple(Tuple<src_size, src_valtype> const &src) {
        enum { COPY_END = (SIZE < src_size ? SIZE : src_size) };

        for (int i = 0; i < COPY_END; ++i) {
            mm[i] = src[i];
        }
        for (int i = COPY_END; i < SIZE; ++i) {
            mm[i] = 0;
        }
    }

    // @brief  const access to an element in the tuple. The offset idx must be
    //   an integral type. A copy of the tuple data is returned.
    template <typename IdxT,
        typename std::enable_if<std::is_integral<IdxT>::value, bool>::type = true>
    T operator[](IdxT i) const {
        OPENVDB_ASSERT(i >= IdxT(0) && i < IdxT(SIZE));
        return mm[i];
    }

    // @brief  non-const access to an element in the tuple. The offset idx must be
    //   an integral type. A reference to the tuple data is returned.
    template <typename IdxT,
        typename std::enable_if<std::is_integral<IdxT>::value, bool>::type = true>
    T& operator[](IdxT i) {
        OPENVDB_ASSERT(i >= IdxT(0) && i < IdxT(SIZE));
        return mm[i];
    }

    // These exist solely to provide backwards compatibility with [] access of
    // non-integer types that were castable to 'int' (such as floating point type).
    // The above templates allow for any integer type to be used as an offset into
    // the tuple data.
    T operator[](int i) const { return this->template operator[]<int>(i); }
    T& operator[](int i) { return this->template operator[]<int>(i); }

    /// @name Compatibility
    /// These are mostly for backwards compatibility with functions that take
    /// old-style Vs (which are just arrays).
    //@{
    /// Copies this tuple into an array of a compatible type
    template <typename S>
    void toV(S *v) const {
        for (int i = 0; i < SIZE; ++i) {
            v[i] = mm[i];
        }
    }

    /// Exposes the internal array.  Be careful when using this function.
    value_type *asV() {
        return mm;
    }
    /// Exposes the internal array.  Be careful when using this function.
    value_type const *asV() const {
        return mm;
    }
    //@}  Compatibility

    /// @return string representation of Classname
    std::string str() const {
        std::ostringstream buffer;

        buffer << "[";

        // For each column
        for (unsigned j(0); j < SIZE; j++) {
            if (j) buffer << ", ";
            buffer << PrintCast(mm[j]);
        }

        buffer << "]";

        return buffer.str();
    }

    void write(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&mm), sizeof(T)*SIZE);
    }
    void read(std::istream& is) {
        is.read(reinterpret_cast<char*>(&mm), sizeof(T)*SIZE);
    }

    /// True if a Nan is present in this tuple
    bool isNan() const {
        for (int i = 0; i < SIZE; ++i) {
            if (math::isNan(mm[i])) return true;
        }
        return false;
    }

    /// True if an Inf is present in this tuple
    bool isInfinite() const {
        for (int i = 0; i < SIZE; ++i) {
            if (math::isInfinite(mm[i])) return true;
        }
        return false;
    }

    /// True if no Nan or Inf values are present
    bool isFinite() const {
        for (int i = 0; i < SIZE; ++i) {
            if (!math::isFinite(mm[i])) return false;
        }
        return true;
    }

    /// True if all elements are exactly zero
    bool isZero() const {
        for (int i = 0; i < SIZE; ++i) {
            if (!math::isZero(mm[i])) return false;
        }
        return true;
    }

protected:
    T mm[SIZE];
};


////////////////////////////////////////


/// @return true if t0 < t1, comparing components in order of significance.
template<int SIZE, typename T0, typename T1>
bool
operator<(const Tuple<SIZE, T0>& t0, const Tuple<SIZE, T1>& t1)
{
    for (int i = 0; i < SIZE-1; ++i) {
        if (!isExactlyEqual(t0[i], t1[i])) return t0[i] < t1[i];
    }
    return t0[SIZE-1] < t1[SIZE-1];
}


/// @return true if t0 > t1, comparing components in order of significance.
template<int SIZE, typename T0, typename T1>
bool
operator>(const Tuple<SIZE, T0>& t0, const Tuple<SIZE, T1>& t1)
{
    for (int i = 0; i < SIZE-1; ++i) {
        if (!isExactlyEqual(t0[i], t1[i])) return t0[i] > t1[i];
    }
    return t0[SIZE-1] > t1[SIZE-1];
}


////////////////////////////////////////


/// @return the absolute value of the given Tuple.
template<int SIZE, typename T>
Tuple<SIZE, T>
Abs(const Tuple<SIZE, T>& t)
{
    Tuple<SIZE, T> result;
    for (int i = 0; i < SIZE; ++i) result[i] = math::Abs(t[i]);
    return result;
}

/// Return @c true if a Nan is present in the tuple.
template<int SIZE, typename T>
inline bool isNan(const Tuple<SIZE, T>& t) { return t.isNan(); }

/// Return @c true if an Inf is present in the tuple.
template<int SIZE, typename T>
inline bool isInfinite(const Tuple<SIZE, T>& t) { return t.isInfinite(); }

/// Return @c true if no Nan or Inf values are present.
template<int SIZE, typename T>
inline bool isFinite(const Tuple<SIZE, T>& t) { return t.isFinite(); }

/// Return @c true if all elements are exactly equal to zero.
template<int SIZE, typename T>
inline bool isZero(const Tuple<SIZE, T>& t) { return t.isZero(); }

////////////////////////////////////////


/// Write a Tuple to an output stream
template <int SIZE, typename T>
std::ostream& operator<<(std::ostream& ostr, const Tuple<SIZE, T>& classname)
{
    ostr << classname.str();
    return ostr;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_TUPLE_HAS_BEEN_INCLUDED
