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

#ifndef OPENVDB_MATH_BBOX_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_BBOX_HAS_BEEN_INCLUDED

#include "Math.h" // for isApproxEqual() and tolerance()
#include "Vec3.h"
#include <ostream>
#include <algorithm> // for min/max
#include <boost/type_traits/is_integral.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @brief Axis-aligned bounding box
template<class _VectorType>
class BBox
{
public:
    typedef _VectorType VectorType;
    typedef _VectorType ValueType;
    typedef typename _VectorType::ValueType ElementType;

    BBox();
    BBox(const VectorType& xyzMin, const VectorType& xyzMax);
    BBox(const VectorType& xyzMin, const VectorType& xyzMax, bool sorted);
    /// @brief Contruct a cubical BBox from a minimum coordinate and a
    /// single edge length.
    /// @note inclusive for integral <tt>ElementType</tt>s
    BBox(const VectorType& xyzMin, const ElementType& length);
    explicit BBox(const ElementType* xyz, bool sorted = true);
    BBox(const BBox& other);

    void sort();

    const VectorType& min() const { return mMin; }
    const VectorType& max() const { return mMax; }

    VectorType& min() { return mMin; }
    VectorType& max() { return mMax; }

    bool operator==(const BBox& rhs) const;
    bool operator!=(const BBox& rhs) const { return !(*this == rhs); }

    bool empty() const;
    bool hasVolume() const { return !empty(); }
    operator bool() const { return !empty(); }

    bool isSorted() const;

    Vec3d getCenter() const;

    /// @note inclusive for integral <tt>ElementType</tt>s
    VectorType extents() const;

    ElementType volume() const { VectorType e = extents(); return e[0] * e[1] * e[2]; }

    /// Return the index (0, 1 or 2) of the longest axis.
    size_t maxExtent() const;

    /// Return @c true if point (x, y, z) is inside this bounding box.
    bool isInside(const VectorType& xyz) const;

    /// Return @c true if the given bounding box is inside this bounding box.
    bool isInside(const BBox&) const;

    /// Return @c true if the given bounding box overlaps with this bounding box.
    bool hasOverlap(const BBox&) const;

    /// Pad this bounding box.
    void expand(ElementType padding);
    /// Expand this bounding box to enclose point (x, y, z).
    void expand(const VectorType& xyz);
    /// Union this bounding box with the given bounding box.
    void expand(const BBox&);
    // @brief Union this bbox with the cubical bbox defined from xyzMin and
    // length
    /// @note inclusive for integral <tt>ElementType</tt>s
    void expand(const VectorType& xyzMin, const ElementType& length);
    /// Translate this bounding box by \f$(t_x, t_y, t_z)\f$.
    void translate(const VectorType& t);

    /// Unserialize this bounding box from the given stream.
    void read(std::istream& is) { mMin.read(is); mMax.read(is); }
    /// Serialize this bounding box to the given stream.
    void write(std::ostream& os) const { mMin.write(os); mMax.write(os); }

private:
    VectorType mMin, mMax;
}; // class BBox


////////////////////////////////////////


template<class VectorType>
inline
BBox<VectorType>::BBox():
    mMin(ElementType(0), ElementType(0), ElementType(0)),
    mMax(ElementType(0), ElementType(0), ElementType(0))
{
}

template<class VectorType>
inline
BBox<VectorType>::BBox(const VectorType& xyzMin, const VectorType& xyzMax):
    mMin(xyzMin), mMax(xyzMax)
{
}

template<class VectorType>
inline
BBox<VectorType>::BBox(const VectorType& xyzMin, const VectorType& xyzMax, bool sorted):
    mMin(xyzMin), mMax(xyzMax)
{
    if (!sorted) this->sort();
}

template<class VectorType>
inline
BBox<VectorType>::BBox(const VectorType& xyzMin, const ElementType& length):
    mMin(xyzMin), mMax(xyzMin)
{
    // min and max are inclusive for integral ElementType
    const ElementType size = boost::is_integral<ElementType>::value ? length-1 : length;
    mMax[0] += size;
    mMax[1] += size;
    mMax[2] += size;
}

template<class VectorType>
inline
BBox<VectorType>::BBox(const ElementType* xyz, bool sorted):
    mMin(xyz[0], xyz[1], xyz[2]),
    mMax(xyz[3], xyz[4], xyz[5])
{
    if (!sorted) this->sort();
}


template<class VectorType>
inline
BBox<VectorType>::BBox(const BBox& other):
    mMin(other.mMin), mMax(other.mMax)
{
}


////////////////////////////////////////


template<class VectorType>
inline bool
BBox<VectorType>::empty() const
{
    if (boost::is_integral<ElementType>::value) {
        // min and max are inclusive for integral ElementType
        return (mMin[0] > mMax[0] || mMin[1] > mMax[1] || mMin[2] > mMax[2]);
    }
    return mMin[0] >= mMax[0] || mMin[1] >= mMax[1] || mMin[2] >= mMax[2];
}


template<class VectorType>
inline bool
BBox<VectorType>::operator==(const BBox& rhs) const
{
    if (boost::is_integral<ElementType>::value) {
        return mMin == rhs.min() && mMax == rhs.max();
    } else {
        return math::isApproxEqual(mMin, rhs.min()) && math::isApproxEqual(mMax, rhs.max());
    }
}


template<class VectorType>
inline void
BBox<VectorType>::sort()
{
    VectorType tMin(mMin), tMax(mMax);
    for (size_t i = 0; i < 3; ++i) {
        mMin[i] = std::min(tMin[i], tMax[i]);
        mMax[i] = std::max(tMin[i], tMax[i]);
    }
}


template<class VectorType>
inline bool
BBox<VectorType>::isSorted() const
{
    if (boost::is_integral<ElementType>::value) {
        return (mMin[0] <= mMax[0] && mMin[1] <= mMax[1] && mMin[2] <= mMax[2]);
    } else {
        ElementType t = tolerance<ElementType>::value();
        return (mMin[0] < (mMax[0] + t) && mMin[1] < (mMax[1] + t) && mMin[2] < (mMax[2] + t));
    }
}


template<class VectorType>
inline Vec3d
BBox<VectorType>::getCenter() const
{
    return (Vec3d(mMin.asPointer()) + Vec3d(mMax.asPointer())) * 0.5;
}


template<class VectorType>
inline VectorType
BBox<VectorType>::extents() const
{
    if (boost::is_integral<ElementType>::value) {
        return (mMax - mMin) + VectorType(1, 1, 1);
    } else {
        return (mMax - mMin);
    }
}


template<class VectorType>
inline size_t
BBox<VectorType>::maxExtent() const
{
    VectorType e = extents();
    if (e[0] > e[1] && e[0] > e[2]) return 0;
    else if (e[1] > e[2]) return 1;
    return 2;
}


////////////////////////////////////////


template<class VectorType>
inline bool
BBox<VectorType>::isInside(const VectorType& xyz) const
{
    if (boost::is_integral<ElementType>::value) {
        return xyz[0] >= mMin[0] && xyz[0] <= mMax[0] &&
               xyz[1] >= mMin[1] && xyz[1] <= mMax[1] &&
               xyz[2] >= mMin[2] && xyz[2] <= mMax[2];
    } else {
        ElementType t = tolerance<ElementType>::value();
        return xyz[0] > (mMin[0]-t) && xyz[0] < (mMax[0]+t) &&
               xyz[1] > (mMin[1]-t) && xyz[1] < (mMax[1]+t) &&
               xyz[2] > (mMin[2]-t) && xyz[2] < (mMax[2]+t);
    }
}


template<class VectorType>
inline bool
BBox<VectorType>::isInside(const BBox& b) const
{
    if (boost::is_integral<ElementType>::value) {
        return b.min()[0] >= mMin[0]  && b.max()[0] <= mMax[0] &&
               b.min()[1] >= mMin[1]  && b.max()[1] <= mMax[1] &&
               b.min()[2] >= mMin[2]  && b.max()[2] <= mMax[2];
    } else {
        ElementType t = tolerance<ElementType>::value();
        return (b.min()[0]-t) > mMin[0]  && (b.max()[0]+t) < mMax[0] &&
               (b.min()[1]-t) > mMin[1]  && (b.max()[1]+t) < mMax[1] &&
               (b.min()[2]-t) > mMin[2]  && (b.max()[2]+t) < mMax[2];
    }
}


template<class VectorType>
inline bool
BBox<VectorType>::hasOverlap(const BBox& b) const
{
    if (boost::is_integral<ElementType>::value) {
        return mMax[0] >= b.min()[0] && mMin[0] <= b.max()[0] &&
               mMax[1] >= b.min()[1] && mMin[1] <= b.max()[1] &&
               mMax[2] >= b.min()[2] && mMin[2] <= b.max()[2];
    } else {
        ElementType t = tolerance<ElementType>::value();
        return mMax[0] > (b.min()[0]-t) && mMin[0] < (b.max()[0]+t) &&
               mMax[1] > (b.min()[1]-t) && mMin[1] < (b.max()[1]+t) &&
               mMax[2] > (b.min()[2]-t) && mMin[2] < (b.max()[2]+t);
    }
}


////////////////////////////////////////


template<class VectorType>
inline void
BBox<VectorType>::expand(ElementType dx)
{
    dx = std::abs(dx);
    for (size_t i = 0; i < 3; ++i) {
        mMin[i] -= dx;
        mMax[i] += dx;
    }
}


template<class VectorType>
inline void
BBox<VectorType>::expand(const VectorType& xyz)
{
    for (size_t i = 0; i < 3; ++i) {
        mMin[i] = std::min(mMin[i], xyz[i]);
        mMax[i] = std::max(mMax[i], xyz[i]);
    }
}


template<class VectorType>
inline void
BBox<VectorType>::expand(const BBox& b)
{
    for (size_t i = 0; i < 3; ++i) {
        mMin[i] = std::min(mMin[i], b.min()[i]);
        mMax[i] = std::max(mMax[i], b.max()[i]);
    }
}

template<class VectorType>
inline void
BBox<VectorType>::expand(const VectorType& xyzMin, const ElementType& length)
{
    const ElementType size = boost::is_integral<ElementType>::value ? length-1 : length;
    for (size_t i = 0; i < 3; ++i) {
        mMin[i] = std::min(mMin[i], xyzMin[i]);
        mMax[i] = std::max(mMax[i], xyzMin[i] + size);
    }
}


template<class VectorType>
inline void
BBox<VectorType>::translate(const VectorType& dx)
{
    mMin += dx;
    mMax += dx;
}


////////////////////////////////////////


template<class VectorType>
inline std::ostream&
operator<<(std::ostream& os, const BBox<VectorType>& b)
{
    os << b.min() << " -> " << b.max();
    return os;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_BBOX_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
