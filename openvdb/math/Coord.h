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

#ifndef OPENVDB_MATH_COORD_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_COORD_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include "Math.h"
#include "Vec3.h"

namespace tbb { class split; } // forward declaration


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @brief Signed (x, y, z) 32-bit integer coordinates
class Coord
{
public:
    typedef int32_t Int32;
    typedef uint32_t Index32;
    typedef Vec3<Int32> Vec3i;
    typedef Vec3<Index32> Vec3I;

    typedef Int32 ValueType;
    typedef std::numeric_limits<ValueType> Limits;

    Coord() { mVec[0] = mVec[1] = mVec[2] = 0; }
    explicit Coord(Int32 xyz) { mVec[0] = mVec[1] = mVec[2] = xyz; }
    Coord(Int32 x, Int32 y, Int32 z) { mVec[0] = x; mVec[1] = y; mVec[2] = z; }
    explicit Coord(const Vec3i& v) { mVec[0] = v[0]; mVec[1] = v[1]; mVec[2] = v[2]; }
    explicit Coord(const Vec3I& v)
    {
        mVec[0] = Int32(v[0]); mVec[1] = Int32(v[1]); mVec[2] = Int32(v[2]);
    }
    explicit Coord(const Int32* v) { mVec[0] = v[0]; mVec[1] = v[1]; mVec[2] = v[2]; }

    /// @brief Return the smallest possible coordinate
    static Coord min() { return Coord(Limits::min()); }

    /// @brief Return the largest possible coordinate
    static Coord max() { return Coord(Limits::max()); }

    /// @brief Return @a xyz rounded to the closest integer coordinates
    /// (cell centered conversion).
    template<typename T> static Coord round(const Vec3<T>& xyz)
    {
        return Coord(Int32(Round(xyz[0])), Int32(Round(xyz[1])), Int32(Round(xyz[2])));
    }
    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    template<typename T> static Coord floor(const Vec3<T>& xyz)
    {
        return Coord(Int32(Floor(xyz[0])), Int32(Floor(xyz[1])), Int32(Floor(xyz[2])));
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz+1 (node centered conversion).
    template<typename T> static Coord ceil(const Vec3<T>& xyz)
    {
        return Coord(Int32(Ceil(xyz[0])), Int32(Ceil(xyz[1])), Int32(Ceil(xyz[2])));
    }

    Coord& reset(Int32 x, Int32 y, Int32 z)
    {
        mVec[0] = x; mVec[1] = y; mVec[2] = z;
        return *this;
    }
    Coord& reset(Int32 xyz) { return this->reset(xyz, xyz, xyz); }

    Coord& setX(Int32 x) { mVec[0] = x; return *this; }
    Coord& setY(Int32 y) { mVec[1] = y; return *this; }
    Coord& setZ(Int32 z) { mVec[2] = z; return *this; }

    Coord& offset(Int32 dx, Int32 dy, Int32 dz)
    {
        mVec[0]+=dx; mVec[1]+=dy; mVec[2]+=dz;
        return *this;
    }
    Coord& offset(Int32 n) { return this->offset(n, n, n); }
    Coord offsetBy(Int32 dx, Int32 dy, Int32 dz) const
    {
        return Coord(mVec[0] + dx, mVec[1] + dy, mVec[2] + dz);
    }
    Coord offsetBy(Int32 n) const { return offsetBy(n, n, n); }

    Coord& operator+=(const Coord& rhs)
    {
        mVec[0] += rhs[0]; mVec[1] += rhs[1]; mVec[2] += rhs[2];
        return *this;
    }
    Coord& operator-=(const Coord& rhs)
    {
        mVec[0] -= rhs[0]; mVec[1] -= rhs[1]; mVec[2] -= rhs[2];
        return *this;
    }
    Coord operator+(const Coord& rhs) const
    {
        return Coord(mVec[0] + rhs[0], mVec[1] + rhs[1], mVec[2] + rhs[2]);
    }
    Coord operator-(const Coord& rhs) const
    {
        return Coord(mVec[0] - rhs[0], mVec[1] - rhs[1], mVec[2] - rhs[2]);
    }
    Coord operator-() const { return Coord(-mVec[0], -mVec[1], -mVec[2]); }

    Coord  operator>> (size_t n) const { return Coord(mVec[0]>>n, mVec[1]>>n, mVec[2]>>n); }
    Coord  operator<< (size_t n) const { return Coord(mVec[0]<<n, mVec[1]<<n, mVec[2]<<n); }
    Coord& operator<<=(size_t n) { mVec[0]<<=n; mVec[1]<<=n; mVec[2]<<=n; return *this; }
    Coord& operator>>=(size_t n) { mVec[0]>>=n; mVec[1]>>=n; mVec[2]>>=n; return *this; }
    Coord  operator&  (Int32 n) const { return Coord(mVec[0] & n, mVec[1] & n, mVec[2] & n); }
    Coord  operator|  (Int32 n) const { return Coord(mVec[0] | n, mVec[1] | n, mVec[2] | n); }
    Coord& operator&= (Int32 n) { mVec[0]&=n; mVec[1]&=n; mVec[2]&=n; return *this; }
    Coord& operator|= (Int32 n) { mVec[0]|=n; mVec[1]|=n; mVec[2]|=n; return *this; }

    Int32 x() const { return mVec[0]; }
    Int32 y() const { return mVec[1]; }
    Int32 z() const { return mVec[2]; }
    Int32 operator[](size_t i) const { assert(i < 3); return mVec[i]; }
    Int32& x() { return mVec[0]; }
    Int32& y() { return mVec[1]; }
    Int32& z() { return mVec[2]; }
    Int32& operator[](size_t i) { assert(i < 3); return mVec[i]; }

    const Int32* data() const { return mVec; }
    Int32* data() { return mVec; }
    const Int32* asPointer() const { return mVec; }
    Int32* asPointer() { return mVec; }
    Vec3d asVec3d() const { return Vec3d(double(mVec[0]), double(mVec[1]), double(mVec[2])); }
    Vec3s asVec3s() const { return Vec3s(float(mVec[0]), float(mVec[1]), float(mVec[2])); }
    Vec3i asVec3i() const { return Vec3i(mVec); }
    Vec3I asVec3I() const { return Vec3I(Index32(mVec[0]), Index32(mVec[1]), Index32(mVec[2])); }
    void asXYZ(Int32& x, Int32& y, Int32& z) const { x = mVec[0]; y = mVec[1]; z = mVec[2]; }

    bool operator==(const Coord& rhs) const
    {
        return (mVec[0] == rhs.mVec[0] && mVec[1] == rhs.mVec[1] && mVec[2] == rhs.mVec[2]);
    }
    bool operator!=(const Coord& rhs) const { return !(*this == rhs); }

    /// Lexicographic less than
    bool operator<(const Coord& rhs) const
    {
        return this->x() < rhs.x() ? true : this->x() > rhs.x() ? false
             : this->y() < rhs.y() ? true : this->y() > rhs.y() ? false
             : this->z() < rhs.z() ? true : false;
    }
    /// Lexicographic less than or equal to
    bool operator<=(const Coord& rhs) const
    {
        return this->x() < rhs.x() ? true : this->x() > rhs.x() ? false
             : this->y() < rhs.y() ? true : this->y() > rhs.y() ? false
             : this->z() <=rhs.z() ? true : false;
    }
    /// Lexicographic greater than
    bool operator>(const Coord& rhs) const { return !(*this <= rhs); }
    /// Lexicographic greater than or equal to
    bool operator>=(const Coord& rhs) const { return !(*this < rhs); }

    /// Perform a component-wise minimum with the other Coord.
    void minComponent(const Coord& other)
    {
        mVec[0] = std::min(mVec[0], other.mVec[0]);
        mVec[1] = std::min(mVec[1], other.mVec[1]);
        mVec[2] = std::min(mVec[2], other.mVec[2]);
    }

    /// Perform a component-wise maximum with the other Coord.
    void maxComponent(const Coord& other)
    {
        mVec[0] = std::max(mVec[0], other.mVec[0]);
        mVec[1] = std::max(mVec[1], other.mVec[1]);
        mVec[2] = std::max(mVec[2], other.mVec[2]);
    }

    /// Return the component-wise minimum of the two Coords.
    static inline Coord minComponent(const Coord& lhs, const Coord& rhs)
    {
        return Coord(std::min(lhs.x(), rhs.x()),
                     std::min(lhs.y(), rhs.y()),
                     std::min(lhs.z(), rhs.z()));
    }

    /// Return the component-wise maximum of the two Coords.
    static inline Coord maxComponent(const Coord& lhs, const Coord& rhs)
    {
        return Coord(std::max(lhs.x(), rhs.x()),
                     std::max(lhs.y(), rhs.y()),
                     std::max(lhs.z(), rhs.z()));
    }
    
    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    static inline bool lessThan(const Coord& a, const Coord& b)
    {
            return (a[0] < b[0] || a[1] < b[1] || a[2] < b[2]);
    }

    /// @brief Return the index (0, 1 or 2) with the smallest value.
    size_t minIndex() const { return MinIndex(mVec); }

    /// @brief Return the index (0, 1 or 2) with the largest value.
    size_t maxIndex() const { return MaxIndex(mVec); }

    void read(std::istream& is) { is.read(reinterpret_cast<char*>(mVec), sizeof(mVec)); }
    void write(std::ostream& os) const
    {
        os.write(reinterpret_cast<const char*>(mVec), sizeof(mVec));
    }

private:

    Int32 mVec[3];
}; // class Coord


////////////////////////////////////////


/// @brief Axis-aligned bounding box of signed integer coordinates
/// @note The range of the integer coordinates, [min, max], is inclusive.
/// Thus, a bounding box with min = max is not empty but rather encloses
/// a single coordinate.
class CoordBBox
{
public:
    typedef uint64_t         Index64;
    typedef Coord::ValueType ValueType;

    /// @brief Iterator over Coord domain covered by a CoordBBox
    ///
    /// @note If ZYX is true Z is the fastest moving coordinate, else
    /// it is the X coordinate, i.e. XYZ traversal
    template<bool ZYX>
    class Iterator {
    public:
        /// @brief C-tor from a bounding box
        Iterator(const CoordBBox &b) : mPos(b.min()), mMin(b.min()), mMax(b.max()) {}
        /// @brief Increments iterator to point to the next coordinate
        /// @note Stops a the last + 1 coordinate of the bounding box
        /// as defined by the template parameter.
        Iterator& operator++() {
            ZYX ? this->next<2,1,0>() : this->next<0,1,2>();
            return *this;
        }
        /// @brief Return true if the iterator still points to a valid coordinate
        operator bool() const {
            return ZYX ? mPos[0] <= mMax[0] : mPos[2] <= mMax[2];
        }
        /// @brief Return a const reference to the coordinate currently pointed to
        const Coord& operator*() const { return mPos; }
    private:
        template<size_t a, size_t b, size_t c>
        inline void next() {
            if ( mPos[a] < mMax[a] )  {//by far this is the most common case
                ++mPos[a];
            } else if ( mPos[b] < mMax[b] )  {
                mPos[a] = mMin[a];
                ++mPos[b];
            } else if ( mPos[c] <= mMax[c] ) {
                mPos[a] = mMin[a];
                mPos[b] = mMin[b];
                ++mPos[c];
            }
        }
        Coord mPos, mMin, mMax;
    };// CoordBBox::Iterator

    /// @brief The default constructor produces an empty bounding box.
    CoordBBox(): mMin(Coord::max()), mMax(Coord::min()) {}
    /// @brief Construct a bounding box with the given @a min and @a max bounds.
    CoordBBox(const Coord& min, const Coord& max): mMin(min), mMax(max) {}
    /// @brief Splitting constructor for use in TBB ranges
    /// @note The other bounding box is assumed to be divisible.
    CoordBBox(CoordBBox& other, const tbb::split&): mMin(other.mMin), mMax(other.mMax)
    {
        assert(this->is_divisible());
        const size_t n = this->maxExtent();
        mMax[n] = (mMin[n] + mMax[n]) >> 1;
        other.mMin[n] = mMax[n] + 1;
    }

    static CoordBBox createCube(const Coord& min, ValueType dim)
    {
        return CoordBBox(min, min.offsetBy(dim - 1));
    }

    /// Return an "infinite" bounding box, as defined by the Coord value range.
    static CoordBBox inf() { return CoordBBox(Coord::min(), Coord::max()); }

    const Coord& min() const { return mMin; }
    const Coord& max() const { return mMax; }

    Coord& min() { return mMin; }
    Coord& max() { return mMax; }

    void reset() { mMin = Coord::max(); mMax = Coord::min(); }
    void reset(const Coord& min, const Coord& max) { mMin = min; mMax = max; }
    void resetToCube(const Coord& min, ValueType dim) { mMin = min; mMax = min.offsetBy(dim - 1); }

    /// @note The start coordinate is inclusive.
    Coord getStart() const { return mMin; }
    /// @note The end coordinate is exclusive.
    Coord getEnd() const { return mMax.offsetBy(1); }

    bool operator==(const CoordBBox& rhs) const { return mMin == rhs.mMin && mMax == rhs.mMax; }
    bool operator!=(const CoordBBox& rhs) const { return !(*this == rhs); }

    bool empty() const { return (mMin[0] > mMax[0] || mMin[1] > mMax[1] || mMin[2] > mMax[2]); }
    //@{
    /// Return @c true if this bounding box is nonempty
    operator bool() const { return !this->empty(); }
    bool hasVolume() const { return !this->empty(); }
    //@}

    /// Return the floating-point position of the center of this bounding box.
    Vec3d getCenter() const { return 0.5 * Vec3d((mMin + mMax).asPointer()); }

    /// @brief Return the dimensions of the coordinates spanned by this bounding box.
    /// @note Since coordinates are inclusive, a bounding box with min = max
    /// has dimensions of (1, 1, 1).
    Coord dim() const { return mMax.offsetBy(1) - mMin; }
    /// @todo deprecate - use dim instead
    Coord extents() const { return this->dim(); }
    /// @brief Return the integer volume of coordinates spanned by this bounding box.
    /// @note Since coordinates are inclusive, a bounding box with min = max has volume one.
    Index64 volume() const
    {
        const Coord d = this->dim();
        return Index64(d[0]) * Index64(d[1]) * Index64(d[2]);
    }
    /// Return @c true if this bounding box can be subdivided [mainly for use by TBB].
    bool is_divisible() const { return mMin[0]<mMax[0] && mMin[1]<mMax[1] && mMin[2]<mMax[2]; }

    /// @brief Return the index (0, 1 or 2) of the shortest axis.
    size_t minExtent() const { return this->dim().minIndex(); }

    /// @brief Return the index (0, 1 or 2) of the longest axis.
    size_t maxExtent() const { return this->dim().maxIndex(); }

    /// Return @c true if point (x, y, z) is inside this bounding box.
    bool isInside(const Coord& xyz) const
    {
        return !(Coord::lessThan(xyz,mMin) || Coord::lessThan(mMax,xyz));
    }

    /// Return @c true if the given bounding box is inside this bounding box.
    bool isInside(const CoordBBox& b) const
    {
        return !(Coord::lessThan(b.mMin,mMin) || Coord::lessThan(mMax,b.mMax));
    }

    /// Return @c true if the given bounding box overlaps with this bounding box.
    bool hasOverlap(const CoordBBox& b) const
    {
        return !(Coord::lessThan(mMax,b.mMin) || Coord::lessThan(b.mMax,mMin));
    }

    /// Pad this bounding box with the specified padding.
    void expand(ValueType padding)
    {
        mMin.offset(-padding);
        mMax.offset( padding);
    }

    /// Return a new instance that is expanded by the specified padding.
    CoordBBox expandBy(ValueType padding) const
    {
        return CoordBBox(mMin.offsetBy(-padding),mMax.offsetBy(padding));
    }
    
    /// Expand this bounding box to enclose point (x, y, z).
    void expand(const Coord& xyz)
    {
        mMin.minComponent(xyz);
        mMax.maxComponent(xyz);
    }
    
    /// Union this bounding box with the given bounding box.
    void expand(const CoordBBox& bbox)
    {
          mMin.minComponent(bbox.min());
          mMax.maxComponent(bbox.max());
    }
    /// Intersect this bounding box with the given bounding box.
    void intersect(const CoordBBox& bbox)
    {
        mMin.maxComponent(bbox.min());
        mMax.minComponent(bbox.max());
    }
    /// @brief Union this bounding box with the cubical bounding box
    /// of the given size and with the given minimum coordinates.
    void expand(const Coord& min, Coord::ValueType dim)
    {
        mMin.minComponent(min);
        mMax.maxComponent(min.offsetBy(dim-1));
    }
    /// Translate this bounding box by @f$(t_x, t_y, t_z)@f$.
    void translate(const Coord& t) { mMin += t; mMax += t; }

    //@{
    /// @brief Bit-wise operations performed on both the min and max members
    CoordBBox  operator>> (size_t n) const { return CoordBBox(mMin>>n, mMax>>n); }
    CoordBBox  operator<< (size_t n) const { return CoordBBox(mMin<<n, mMax<<n); }
    CoordBBox& operator<<=(size_t n) { mMin <<= n; mMax <<= n; return *this; }
    CoordBBox& operator>>=(size_t n) { mMin >>= n; mMax >>= n; return *this; }
    CoordBBox  operator&  (Coord::Int32 n) const { return CoordBBox(mMin & n, mMax & n); }
    CoordBBox  operator|  (Coord::Int32 n) const { return CoordBBox(mMin | n, mMax | n); }
    CoordBBox& operator&= (Coord::Int32 n) { mMin &= n; mMax &= n; return *this; }
    CoordBBox& operator|= (Coord::Int32 n) { mMin |= n; mMax |= n; return *this; }
    //@}
     
    /// Unserialize this bounding box from the given stream.
    void read(std::istream& is) { mMin.read(is); mMax.read(is); }
    /// Serialize this bounding box to the given stream.
    void write(std::ostream& os) const { mMin.write(os); mMax.write(os); }

private:
    Coord mMin, mMax;
}; // class CoordBBox


////////////////////////////////////////


inline std::ostream& operator<<(std::ostream& os, const Coord& xyz)
{
    os << xyz.asVec3i(); return os;
}


//@{
/// Allow a Coord to be added to or subtracted from a Vec3.
template<typename T>
inline Vec3<typename promote<T, typename Coord::ValueType>::type>
operator+(const Vec3<T>& v0, const Coord& v1)
{
    Vec3<typename promote<T, typename Coord::ValueType>::type> result(v0);
    result[0] += v1[0];
    result[1] += v1[1];
    result[2] += v1[2];
    return result;
}

template<typename T>
inline Vec3<typename promote<T, typename Coord::ValueType>::type>
operator+(const Coord& v1, const Vec3<T>& v0)
{
    Vec3<typename promote<T, typename Coord::ValueType>::type> result(v0);
    result[0] += v1[0];
    result[1] += v1[1];
    result[2] += v1[2];
    return result;
}
//@}


//@{
/// Allow a Coord to be subtracted from a Vec3.
template <typename T>
inline Vec3<typename promote<T, Coord::ValueType>::type>
operator-(const Vec3<T>& v0, const Coord& v1)
{
    Vec3<typename promote<T, Coord::ValueType>::type> result(v0);
    result[0] -= v1[0];
    result[1] -= v1[1];
    result[2] -= v1[2];
    return result;
}

template <typename T>
inline Vec3<typename promote<T, Coord::ValueType>::type>
operator-(const Coord& v1, const Vec3<T>& v0)
{
    Vec3<typename promote<T, Coord::ValueType>::type> result(v0);
    result[0] -= v1[0];
    result[1] -= v1[1];
    result[2] -= v1[2];
    return -result;
}
//@}

inline std::ostream&
operator<<(std::ostream& os, const CoordBBox& b)
{
    os << b.min() << " -> " << b.max();
    return os;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_COORD_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
