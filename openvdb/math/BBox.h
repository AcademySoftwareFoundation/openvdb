///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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

#ifndef OPENVDB_MATH_BBOX_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_BBOX_HAS_BEEN_INCLUDED

#include "Math.h" // for math::isApproxEqual() and math::Tolerance()
#include "Vec3.h"
#include <ostream>
#include <algorithm> // for min/max
#include <boost/type_traits/is_integral.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @brief Axis-aligned bounding box
template<typename Vec3T>
class BBox
{
public:
    typedef Vec3T Vec3Type;
    typedef Vec3T ValueType;
    typedef Vec3T VectorType;
    typedef typename Vec3Type::ValueType ElementType;

    /// @brief Default constructor creates an invalid BBox
    BBox();

    /// @brief Constructor based on a minimum and maximum point.
    BBox(const Vec3T& xyzMin, const Vec3T& xyzMax);

    /// @brief Constructor based on a minimum and maximum point.
    /// If sorted is false the points will be sorted by x,y,z component.
    BBox(const Vec3T& xyzMin, const Vec3T& xyzMax, bool sorted);

    /// @brief Contruct a cubical BBox from a minimum coordinate and a
    /// single edge length.
    /// @note inclusive for integral <tt>ElementType</tt>s
    BBox(const Vec3T& xyzMin, const ElementType& length);

    /// @brief Constructor based on a raw array of six points. If
    /// sorted is false the points will be sorted by x,y,z component.
    explicit BBox(const ElementType* xyz, bool sorted = true);

    /// @brief Copy constructor
    BBox(const BBox&) = default;

    /// @brief Assignment operator
    BBox& operator=(const BBox&) = default;

    /// @brief Sort the min/max by x,y,z component.
    void sort();

    /// @brief Return a const reference to the minimum point of the BBox
    const Vec3T& min() const { return mMin; }

    /// @brief Return a const reference to the maximum point of the BBox
    const Vec3T& max() const { return mMax; }

    /// @brief Return a non-const reference to the minimum point of the BBox
    Vec3T& min() { return mMin; }

    /// @brief Return a non-const reference to the maximum point of the BBox
    Vec3T& max() { return mMax; }

    /// @brief Return true if the two BBox'es are identical
    bool operator==(const BBox& rhs) const;

    /// @brief Return true if the two BBox'es are not identical
    bool operator!=(const BBox& rhs) const { return !(*this == rhs); }

    /// @brief Return true if the BBox is empty, i.e. has no
    /// (positive) volume.
    bool empty() const;

    /// @brief Return true if the BBox has a (positive) volume.
    bool hasVolume() const { return !this->empty(); }

    /// @brief Return true if the BBox is valid, i.e. as a (positive) volume.
    operator bool() const { return !this->empty(); }

    /// @brief Return true if the all components of mMin <= mMax,
    /// i.e. the volume is not negative.
    /// @note For floating point values a tolerance is used for this test.
    bool isSorted() const;

    /// @brief Return the center point of the BBox
    Vec3d getCenter() const;

    /// @brief Returns the extents of the BBox, i.e. the length per axis
    /// for floating points values or number of grids per axis points
    /// integral values.
    /// @note inclusive for integral <tt>ElementType</tt>s
    Vec3T extents() const;

    /// @brief Return the volume spanned by this BBox.
    ElementType volume() const { Vec3T e = this->extents(); return e[0] * e[1] * e[2]; }

    /// Return the index (0, 1 or 2) of the longest axis.
    size_t maxExtent() const { return MaxIndex(mMax - mMin); }

    /// Return the index (0, 1 or 2) of the shortest axis.
    size_t minExtent() const { return MinIndex(mMax - mMin); }

    /// Return @c true if point (x, y, z) is inside this bounding box.
    bool isInside(const Vec3T& xyz) const;

    /// Return @c true if the given bounding box is inside this bounding box.
    bool isInside(const BBox&) const;

    /// Return @c true if the given bounding box overlaps with this bounding box.
    bool hasOverlap(const BBox&) const;

    /// Pad this bounding box.
    void expand(ElementType padding);

    /// Expand this bounding box to enclose point (x, y, z).
    void expand(const Vec3T& xyz);

    /// Union this bounding box with the given bounding box.
    void expand(const BBox&);
    // @brief Union this bbox with the cubical bbox defined from xyzMin and
    // length
    /// @note inclusive for integral <tt>ElementType</tt>s
    void expand(const Vec3T& xyzMin, const ElementType& length);

    /// Translate this bounding box by \f$(t_x, t_y, t_z)\f$.
    void translate(const Vec3T& t);

    /// Apply a map to this bounding box
    template<typename MapType>
    BBox applyMap(const MapType& map) const;

     /// Apply the inverse of a map to this bounding box
    template<typename MapType>
    BBox applyInverseMap(const MapType& map) const;

    /// Unserialize this bounding box from the given stream.
    void read(std::istream& is) { mMin.read(is); mMax.read(is); }

    /// Serialize this bounding box to the given stream.
    void write(std::ostream& os) const { mMin.write(os); mMax.write(os); }

private:
    Vec3T mMin, mMax;
}; // class BBox


////////////////////////////////////////


template<typename Vec3T>
inline
BBox<Vec3T>::BBox():
    mMin( std::numeric_limits<ElementType>::max()),
    mMax(-std::numeric_limits<ElementType>::max())
{
}

template<typename Vec3T>
inline
BBox<Vec3T>::BBox(const Vec3T& xyzMin, const Vec3T& xyzMax):
    mMin(xyzMin), mMax(xyzMax)
{
}

template<typename Vec3T>
inline
BBox<Vec3T>::BBox(const Vec3T& xyzMin, const Vec3T& xyzMax, bool sorted):
    mMin(xyzMin), mMax(xyzMax)
{
    if (!sorted) this->sort();
}

template<typename Vec3T>
inline
BBox<Vec3T>::BBox(const Vec3T& xyzMin, const ElementType& length):
    mMin(xyzMin), mMax(xyzMin)
{
    // min and max are inclusive for integral ElementType
    const ElementType size = boost::is_integral<ElementType>::value ? length-1 : length;
    mMax[0] += size;
    mMax[1] += size;
    mMax[2] += size;
}

template<typename Vec3T>
inline
BBox<Vec3T>::BBox(const ElementType* xyz, bool sorted):
    mMin(xyz[0], xyz[1], xyz[2]),
    mMax(xyz[3], xyz[4], xyz[5])
{
    if (!sorted) this->sort();
}


////////////////////////////////////////


template<typename Vec3T>
inline bool
BBox<Vec3T>::empty() const
{
    if (boost::is_integral<ElementType>::value) {
        // min and max are inclusive for integral ElementType
        return (mMin[0] > mMax[0] || mMin[1] > mMax[1] || mMin[2] > mMax[2]);
    }
    return mMin[0] >= mMax[0] || mMin[1] >= mMax[1] || mMin[2] >= mMax[2];
}


template<typename Vec3T>
inline bool
BBox<Vec3T>::operator==(const BBox& rhs) const
{
    if (boost::is_integral<ElementType>::value) {
        return mMin == rhs.min() && mMax == rhs.max();
    } else {
        return math::isApproxEqual(mMin, rhs.min()) && math::isApproxEqual(mMax, rhs.max());
    }
}


template<typename Vec3T>
inline void
BBox<Vec3T>::sort()
{
    Vec3T tMin(mMin), tMax(mMax);
    for (int i = 0; i < 3; ++i) {
        mMin[i] = std::min(tMin[i], tMax[i]);
        mMax[i] = std::max(tMin[i], tMax[i]);
    }
}


template<typename Vec3T>
inline bool
BBox<Vec3T>::isSorted() const
{
    if (boost::is_integral<ElementType>::value) {
        return (mMin[0] <= mMax[0] && mMin[1] <= mMax[1] && mMin[2] <= mMax[2]);
    } else {
        ElementType t = math::Tolerance<ElementType>::value();
        return (mMin[0] < (mMax[0] + t) && mMin[1] < (mMax[1] + t) && mMin[2] < (mMax[2] + t));
    }
}


template<typename Vec3T>
inline Vec3d
BBox<Vec3T>::getCenter() const
{
    return (Vec3d(mMin.asPointer()) + Vec3d(mMax.asPointer())) * 0.5;
}


template<typename Vec3T>
inline Vec3T
BBox<Vec3T>::extents() const
{
    if (boost::is_integral<ElementType>::value) {
        return (mMax - mMin) + Vec3T(1, 1, 1);
    } else {
        return (mMax - mMin);
    }
}

////////////////////////////////////////


template<typename Vec3T>
inline bool
BBox<Vec3T>::isInside(const Vec3T& xyz) const
{
    if (boost::is_integral<ElementType>::value) {
        return xyz[0] >= mMin[0] && xyz[0] <= mMax[0] &&
               xyz[1] >= mMin[1] && xyz[1] <= mMax[1] &&
               xyz[2] >= mMin[2] && xyz[2] <= mMax[2];
    } else {
        ElementType t = math::Tolerance<ElementType>::value();
        return xyz[0] > (mMin[0]-t) && xyz[0] < (mMax[0]+t) &&
               xyz[1] > (mMin[1]-t) && xyz[1] < (mMax[1]+t) &&
               xyz[2] > (mMin[2]-t) && xyz[2] < (mMax[2]+t);
    }
}


template<typename Vec3T>
inline bool
BBox<Vec3T>::isInside(const BBox& b) const
{
    if (boost::is_integral<ElementType>::value) {
        return b.min()[0] >= mMin[0]  && b.max()[0] <= mMax[0] &&
               b.min()[1] >= mMin[1]  && b.max()[1] <= mMax[1] &&
               b.min()[2] >= mMin[2]  && b.max()[2] <= mMax[2];
    } else {
        ElementType t = math::Tolerance<ElementType>::value();
        return (b.min()[0]-t) > mMin[0]  && (b.max()[0]+t) < mMax[0] &&
               (b.min()[1]-t) > mMin[1]  && (b.max()[1]+t) < mMax[1] &&
               (b.min()[2]-t) > mMin[2]  && (b.max()[2]+t) < mMax[2];
    }
}


template<typename Vec3T>
inline bool
BBox<Vec3T>::hasOverlap(const BBox& b) const
{
    if (boost::is_integral<ElementType>::value) {
        return mMax[0] >= b.min()[0] && mMin[0] <= b.max()[0] &&
               mMax[1] >= b.min()[1] && mMin[1] <= b.max()[1] &&
               mMax[2] >= b.min()[2] && mMin[2] <= b.max()[2];
    } else {
        ElementType t = math::Tolerance<ElementType>::value();
        return mMax[0] > (b.min()[0]-t) && mMin[0] < (b.max()[0]+t) &&
               mMax[1] > (b.min()[1]-t) && mMin[1] < (b.max()[1]+t) &&
               mMax[2] > (b.min()[2]-t) && mMin[2] < (b.max()[2]+t);
    }
}


////////////////////////////////////////


template<typename Vec3T>
inline void
BBox<Vec3T>::expand(ElementType dx)
{
    dx = std::abs(dx);
    for (int i = 0; i < 3; ++i) {
        mMin[i] -= dx;
        mMax[i] += dx;
    }
}


template<typename Vec3T>
inline void
BBox<Vec3T>::expand(const Vec3T& xyz)
{
    for (int i = 0; i < 3; ++i) {
        mMin[i] = std::min(mMin[i], xyz[i]);
        mMax[i] = std::max(mMax[i], xyz[i]);
    }
}


template<typename Vec3T>
inline void
BBox<Vec3T>::expand(const BBox& b)
{
    for (int i = 0; i < 3; ++i) {
        mMin[i] = std::min(mMin[i], b.min()[i]);
        mMax[i] = std::max(mMax[i], b.max()[i]);
    }
}

template<typename Vec3T>
inline void
BBox<Vec3T>::expand(const Vec3T& xyzMin, const ElementType& length)
{
    const ElementType size = boost::is_integral<ElementType>::value ? length-1 : length;
    for (int i = 0; i < 3; ++i) {
        mMin[i] = std::min(mMin[i], xyzMin[i]);
        mMax[i] = std::max(mMax[i], xyzMin[i] + size);
    }
}


template<typename Vec3T>
inline void
BBox<Vec3T>::translate(const Vec3T& dx)
{
    mMin += dx;
    mMax += dx;
}

template<typename Vec3T>
template<typename MapType>
inline BBox<Vec3T>
BBox<Vec3T>::applyMap(const MapType& map) const
{
    typedef Vec3<double> Vec3R;
    BBox<Vec3T> bbox;
    bbox.expand(map.applyMap(Vec3R(mMin[0], mMin[1], mMin[2])));
    bbox.expand(map.applyMap(Vec3R(mMin[0], mMin[1], mMax[2])));
    bbox.expand(map.applyMap(Vec3R(mMin[0], mMax[1], mMin[2])));
    bbox.expand(map.applyMap(Vec3R(mMax[0], mMin[1], mMin[2])));
    bbox.expand(map.applyMap(Vec3R(mMax[0], mMax[1], mMin[2])));
    bbox.expand(map.applyMap(Vec3R(mMax[0], mMin[1], mMax[2])));
    bbox.expand(map.applyMap(Vec3R(mMin[0], mMax[1], mMax[2])));
    bbox.expand(map.applyMap(Vec3R(mMax[0], mMax[1], mMax[2])));
    return bbox;
}

template<typename Vec3T>
template<typename MapType>
inline BBox<Vec3T>
BBox<Vec3T>::applyInverseMap(const MapType& map) const
{
    typedef Vec3<double> Vec3R;
    BBox<Vec3T> bbox;
    bbox.expand(map.applyInverseMap(Vec3R(mMin[0], mMin[1], mMin[2])));
    bbox.expand(map.applyInverseMap(Vec3R(mMin[0], mMin[1], mMax[2])));
    bbox.expand(map.applyInverseMap(Vec3R(mMin[0], mMax[1], mMin[2])));
    bbox.expand(map.applyInverseMap(Vec3R(mMax[0], mMin[1], mMin[2])));
    bbox.expand(map.applyInverseMap(Vec3R(mMax[0], mMax[1], mMin[2])));
    bbox.expand(map.applyInverseMap(Vec3R(mMax[0], mMin[1], mMax[2])));
    bbox.expand(map.applyInverseMap(Vec3R(mMin[0], mMax[1], mMax[2])));
    bbox.expand(map.applyInverseMap(Vec3R(mMax[0], mMax[1], mMax[2])));
    return bbox;
}

////////////////////////////////////////


template<typename Vec3T>
inline std::ostream&
operator<<(std::ostream& os, const BBox<Vec3T>& b)
{
    os << b.min() << " -> " << b.max();
    return os;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_BBOX_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
