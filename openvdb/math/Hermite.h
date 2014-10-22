///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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

#ifndef OPENVDB_MATH_HERMITE_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_HERMITE_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/version.h>
#include "QuantizedUnitVec.h"
#include "Math.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


// Forward declaration
class Hermite;


////////////////////////////////////////

// Utility methods


//@{
/// min and max operations done directly on the compressed data.
OPENVDB_API Hermite min(const Hermite&, const Hermite&);
OPENVDB_API Hermite max(const Hermite&, const Hermite&);
//@}


////////////////////////////////////////


/// @brief Quantized Hermite data object that stores compressed intersection
/// information (offsets and normlas) for the up-wind edges of a voxel. (Size 10 bytes)
class OPENVDB_API Hermite
{
public:

    Hermite();
    Hermite(const Hermite&);
    const Hermite& operator=(const Hermite&);

    /// clears all intersection data
    void clear();

    /// @return true if this Hermite objet has any edge intersection data.
    operator bool() const;

    /// equality operator
    inline bool operator==(const Hermite&) const;
    /// inequality operator
    bool operator!=(const Hermite& rhs) const { return !(*this == rhs); }

    /// unary negation operator, flips inside/outside state and normals.
    Hermite operator-() const;

    //@{
    /// @brief methods to compress and store edge data.
    /// @note @c offset is expected to be in the [0 to 1) range.
    template <typename T>
    void setX(T offset, const Vec3<T>&);

    template <typename T>
    void setY(T offset, const Vec3<T>&);

    template <typename T>
    void setZ(T offset, const Vec3<T>&);
    //@}

    /// @return true if the current Hermite object is classified
    // as being inside a contour.
    bool isInside() const { return MASK_SIGN & mData; }
    /// Set the inside/outside state to reflect if this Hermite object
    /// is located at a point in space that is inside/outside a contour.
    void setIsInside(bool);

    //@{
    /// @return true if this Hermite object has intersection data
    /// for the corresponding edge.
    bool hasOffsetX() const { return mXNormal; };
    bool hasOffsetY() const { return mYNormal; }
    bool hasOffsetZ() const { return MASK_ZFLAG & mData; }
    //@}

    //@{
    /// Edge offset greater-than comparisson operators
    /// @note is @c this offset > than @c other offset
    bool isGreaterX(const Hermite& other) const;
    bool isGreaterY(const Hermite& other) const;
    bool isGreaterZ(const Hermite& other) const;
    //@}

    //@{
    /// Edge offset less-than comparisson operators
    /// @note is @c this offset < than @c other offset
    bool isLessX(const Hermite& other) const;
    bool isLessY(const Hermite& other) const;
    bool isLessZ(const Hermite& other) const;
    //@}

    //@{
    /// @return uncompressed edge intersection offsets
    float getOffsetX() const;
    float getOffsetY() const;
    float getOffsetZ() const;
    //@}

    //@{
    /// @return uncompressed edge intersection normals
    Vec3s getNormalX() const { return QuantizedUnitVec::unpack(mXNormal); }
    Vec3s getNormalY() const { return QuantizedUnitVec::unpack(mYNormal); }
    Vec3s getNormalZ() const { return QuantizedUnitVec::unpack(mZNormal); }
    //@}

    //@{
    /// copy edge data from other Hermite object
    /// @note copies data in the compressed form
    void setX(const Hermite&);
    void setY(const Hermite&);
    void setZ(const Hermite&);
    //@}

    /// String representation.
    std::string str() const;

    /// Unserialize this transform from the given stream.
    void read(std::istream&);
    /// Serialize this transform to the given stream.
    void write(std::ostream&) const;

    //@{
    /// Operators required by OpenVDB.
    /// @note These methods don't perform meaningful operations on Hermite data.
    bool operator< (const Hermite&) const { return false; };
    bool operator> (const Hermite&) const { return false; };
    template<class T> Hermite operator+(const T&) const { return *this; }
    template<class T> Hermite operator-(const T&) const { return *this; }
    //@}

private:
    /// Helper function that quantizes a [0, 1) offset using 10-bits.
    template <typename T>
    static uint32_t quantizeOffset(T offset);

    /// Helper function that returns (signed) compressed-offsets,
    /// used by comparisson operators.
    static void getSignedOffsets(const Hermite& lhs, const Hermite& rhs,
        const uint32_t bitmask, int& lhsV, int& rhsV);


    // Bits masks
    // 10000000000000000000000000000000
    static const uint32_t MASK_SIGN  = 0x80000000;
    // 01000000000000000000000000000000
    static const uint32_t MASK_ZFLAG = 0x40000000;
    // 00111111111100000000000000000000
    static const uint32_t MASK_XSLOT = 0x3FF00000;
    // 00000000000011111111110000000000
    static const uint32_t MASK_YSLOT = 0x000FFC00;
    // 00000000000000000000001111111111
    static const uint32_t MASK_ZSLOT = 0x000003FF;
    // 00111111111111111111111111111111
    static const uint32_t MASK_SLOTS = 0x3FFFFFFF;


    uint16_t mXNormal, mYNormal, mZNormal;
    uint32_t mData;

}; // class Hermite


////////////////////////////////////////

//  output-stream insertion operator

inline std::ostream&
operator<<(std::ostream& ostr, const Hermite& rhs)
{
    ostr << rhs.str();
    return ostr;
}


////////////////////////////////////////

// construction and assignment

inline const Hermite&
Hermite::operator=(const Hermite& rhs)
{
    mData = rhs.mData;
    mXNormal = rhs.mXNormal;
    mYNormal = rhs.mYNormal;
    mZNormal = rhs.mZNormal;
    return *this;
}


inline void
Hermite::clear()
{
    mXNormal = 0;
    mYNormal = 0;
    mZNormal = 0;
    mData = 0;
}


////////////////////////////////////////

//  bool operator and equality

inline
Hermite::operator bool() const
{
    if (0 != (mXNormal | mYNormal)) return true;
    return hasOffsetZ();
}


inline bool
Hermite::operator==(const Hermite& rhs) const
{
    if(mXNormal != rhs.mXNormal) return false;
    if(mYNormal != rhs.mYNormal) return false;
    if(mZNormal != rhs.mZNormal) return false;
    return mData == rhs.mData;
}


////////////////////////////////////////

// unary negation operator

inline Hermite
Hermite::operator-() const
{
    Hermite ret(*this);
    QuantizedUnitVec::flipSignBits(ret.mXNormal);
    QuantizedUnitVec::flipSignBits(ret.mYNormal);
    QuantizedUnitVec::flipSignBits(ret.mZNormal);
    ret.mData = (~MASK_SIGN & ret.mData) | (MASK_SIGN & ~ret.mData);
    return ret;
}


////////////////////////////////////////

// Helper funcions

template <typename T>
inline uint32_t
Hermite::quantizeOffset(T offset)
{
    // the offset is expected to be normalized [0 to 1)
    assert(offset < 1.0);
    assert(offset > -1.0e-8);

    // quantize the offset using 10-bits. (higher bits are masked out)
    return uint32_t(1023 * offset) & MASK_ZSLOT;
}

inline void
Hermite::getSignedOffsets(const Hermite& lhs, const Hermite& rhs,
    const uint32_t bitmask, int& lhsV, int& rhsV)
{
    lhsV = bitmask & lhs.mData;
    rhsV = bitmask & rhs.mData;

    if(lhs.isInside()) lhsV = -lhsV;
    if(rhs.isInside()) rhsV = -rhsV;
}


////////////////////////////////////////

// compress and set edge data

template <typename T>
inline void
Hermite::setX(T offset, const Vec3<T>& n)
{
    mData &= ~MASK_XSLOT; // clear xslot
    mData |= quantizeOffset(offset) << 20;
    mXNormal = QuantizedUnitVec::pack(n);
}

template <typename T>
inline void
Hermite::setY(T offset, const Vec3<T>& n)
{
    mData &= ~MASK_YSLOT; // clear yslot
    mData |= quantizeOffset(offset) << 10;
    mYNormal = QuantizedUnitVec::pack(n);
}

template <typename T>
inline void
Hermite::setZ(T offset, const Vec3<T>& n)
{
    mData &= ~MASK_ZSLOT; // clear zslot
    mData |= MASK_ZFLAG | quantizeOffset(offset);
    mZNormal = QuantizedUnitVec::pack(n);
}


////////////////////////////////////////

// change inside/outside state

inline void
Hermite::setIsInside(bool isInside)
{
    mData &= ~MASK_SIGN; // clear sign-bit
    mData |= uint32_t(isInside) * MASK_SIGN;
}


////////////////////////////////////////

// Uncompress and return the edge intersection-offsets
// 0.000977517 = 1.0 / 1023

inline float
Hermite::getOffsetX() const
{
    return float(((mData >> 20) & MASK_ZSLOT) * 0.000977517);
}

inline float
Hermite::getOffsetY() const
{
    return float(((mData >> 10) & MASK_ZSLOT) * 0.000977517);
}

inline float
Hermite::getOffsetZ() const
{
    return float((mData & MASK_ZSLOT) * 0.000977517);
}


////////////////////////////////////////

// copy compressed edge data from other object

inline void
Hermite::setX(const Hermite& rhs)
{
    mData &= ~MASK_XSLOT; // clear xslot
    mData |= MASK_XSLOT & rhs.mData; // copy xbits from rhs
    mXNormal = rhs.mXNormal; // copy compressed normal

    // Flip the copied normal if the rhs object has
    // a different inside/outside state.
    if(hasOffsetX() && isInside() != rhs.isInside())
        QuantizedUnitVec::flipSignBits(mXNormal);
}

inline void
Hermite::setY(const Hermite& rhs)
{
    mData &= ~MASK_YSLOT;
    mData |= MASK_YSLOT & rhs.mData;
    mYNormal = rhs.mYNormal;

    if(hasOffsetY() && isInside() != rhs.isInside())
        QuantizedUnitVec::flipSignBits(mYNormal);
}

inline void
Hermite::setZ(const Hermite& rhs)
{
    mData &= ~MASK_ZSLOT;
    mData |= (MASK_ZFLAG | MASK_ZSLOT) & rhs.mData;
    mZNormal = rhs.mZNormal;
    if(hasOffsetZ() && isInside() != rhs.isInside())
        QuantizedUnitVec::flipSignBits(mZNormal);
}


////////////////////////////////////////

// edge comparison operators

inline bool
Hermite::isGreaterX(const Hermite& rhs) const
{
    int lhsV, rhsV;
    getSignedOffsets(*this, rhs, MASK_XSLOT, lhsV, rhsV);
    return lhsV > rhsV;
}

inline bool
Hermite::isGreaterY(const Hermite& rhs) const
{
    int lhsV, rhsV;
    getSignedOffsets(*this, rhs, MASK_YSLOT, lhsV, rhsV);
    return lhsV > rhsV;
}

inline bool
Hermite::isGreaterZ(const Hermite& rhs) const
{
    int lhsV, rhsV;
    getSignedOffsets(*this, rhs, MASK_ZSLOT, lhsV, rhsV);
    return lhsV > rhsV;
}

inline bool
Hermite::isLessX(const Hermite& rhs) const
{
    int lhsV, rhsV;
    getSignedOffsets(*this, rhs, MASK_XSLOT, lhsV, rhsV);
    return lhsV < rhsV;
}

inline bool
Hermite::isLessY(const Hermite& rhs) const
{
    int lhsV, rhsV;
    getSignedOffsets(*this, rhs, MASK_YSLOT, lhsV, rhsV);
    return lhsV < rhsV;
}

inline bool
Hermite::isLessZ(const Hermite& rhs) const
{
    int lhsV, rhsV;
    getSignedOffsets(*this, rhs, MASK_ZSLOT, lhsV, rhsV);
    return lhsV < rhsV;
}


////////////////////////////////////////


inline bool
isApproxEqual(const Hermite& lhs, const Hermite& rhs) { return lhs == rhs; }

inline bool
isApproxEqual(const Hermite& lhs, const Hermite& rhs, const Hermite& /*tolerance*/)
    { return isApproxEqual(lhs, rhs); }


} // namespace math


////////////////////////////////////////


template<> inline math::Hermite zeroVal<math::Hermite>() { return math::Hermite(); }


} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_HERMITE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
