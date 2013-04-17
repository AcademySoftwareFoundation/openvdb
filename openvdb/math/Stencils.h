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
//
/// @author Ken Museth
/// @file Stencils.h

#ifndef OPENVDB_MATH_STENCILS_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_STENCILS_HAS_BEEN_INCLUDED

#include <algorithm>
#include <vector>
#include <openvdb/math/Math.h>             // for Pow2, needed by WENO and  Gudonov
#include <openvdb/Types.h>                 // for Real
#include <openvdb/math/Coord.h>            // for Coord
#include <openvdb/math/FiniteDifference.h> // for WENO5 and GudonovsNormSqrd

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////


template<typename _GridType, typename StencilType>
class BaseStencil
{
public:
    typedef _GridType                     GridType;
    typedef typename GridType::TreeType   TreeType;
    typedef typename GridType::ValueType  ValueType;
    typedef std::vector<ValueType>        BufferType;
    typedef typename BufferType::iterator IterType;

    /// Initialize the stencil buffer with the values of voxel (x, y, z)
    /// and its neighbors.
    inline void moveTo(const Coord& ijk)
    {
        mCenter = ijk;
        mStencil[0] = mCache.getValue(ijk);
        static_cast<StencilType&>(*this).init(mCenter);
    }
    /// @brief Initialize the stencil buffer with the values of voxel
    /// (x, y, z) and its neighbors.
    ///
    /// @note This version is slightly faster than the one above, since
    /// the center voxel's value is read directly from the iterator.
    template<typename IterType>
    inline void moveTo(const IterType& iter)
    {
        mCenter = iter.getCoord();
        mStencil[0] = *iter;
        static_cast<StencilType&>(*this).init(mCenter);
    }

    /// @brief Return the value from the stencil buffer with linear
    /// offset pos.
    ///
    /// The default (@a pos = 0) corresponds to the center point of the stencil.
    inline ValueType getValue(unsigned int pos = 0) const
    {
        assert(pos < mStencil.size());
        return mStencil[pos];
    }

    /// Return the value at the specified location relative to the center of the stencil
    template<int i, int j, int k>
    const ValueType& getValue() const
    {
        return mStencil[static_cast<const StencilType&>(*this).template pos<i,j,k>()];
    }

    /// Set the value at the specified location relative to the center of the stencil
    template<int i, int j, int k>
    void setValue(const ValueType& value)
    {
        mStencil[static_cast<const StencilType&>(*this).template pos<i,j,k>()] = value;
    }

    /// Return the size of the stencil buffer.
    inline int size() { return mStencil.size(); }

    /// Return the median value of the current stencil.
    inline ValueType median() const
    {
        std::vector<ValueType> tmp(mStencil);//local copy
        assert(!tmp.empty());
        size_t midpoint = (tmp.size() - 1) >> 1;
        // Partially sort the vector until the median value is at the midpoint.
        std::nth_element(tmp.begin(), tmp.begin() + midpoint, tmp.end());
        return tmp[midpoint];
    }

    /// Return the mean value of the current stencil.
    inline ValueType mean() const
    {
        double sum = 0.0;
        for (int n=0, s=mStencil.size(); n<s; ++n) sum += mStencil[n];
        return ValueType(sum / mStencil.size());
    }

    /// Return the smallest value in the stencil buffer.
    inline ValueType min() const
    {
        IterType iter = std::min_element(mStencil.begin(), mStencil.end());
        return *iter;
    }

    /// Return the largest value in the stencil buffer.
    inline ValueType max() const
    {
        IterType iter = std::max_element(mStencil.begin(), mStencil.end());
        return *iter;
    }

    /// Return the coordinates of the center point of the stencil.
    inline const Coord& getCenterCoord() const { return mCenter; }

    /// Return the value at the center of the stencil
    inline const ValueType& getCenterValue() const
    {
        return this->getValue<0,0,0>();
    }

    /// Return true if the center of the stencil intersects the
    /// iso-contour specified by the isoValue
    inline bool intersects(const ValueType &isoValue = zeroVal<ValueType>()) const
    {
        const bool less = this->getValue< 0, 0, 0>() < isoValue;
        return (less  ^  (this->getValue<-1, 0, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 1, 0, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 0,-1, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 0, 1, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 0, 0,-1>() < isoValue)) ||
               (less  ^  (this->getValue< 0, 0, 1>() < isoValue))  ;
    }

protected:
    // Constructor is protected to prevent direct instantiation.
    BaseStencil(const GridType& grid, int size):
        mCache(grid.getConstAccessor()),
        mStencil(size)
    {
    }

    typename GridType::ConstAccessor mCache;
    BufferType                       mStencil;
    Coord                            mCenter;

}; // class BaseStencil


////////////////////////////////////////


namespace { // anonymous namespace for stencil-layout map

    // the seven point stencil
    template<int i, int j, int k> struct SevenPt {};
    template<> struct SevenPt< 0, 0, 0> { enum { idx = 0 }; };
    template<> struct SevenPt< 1, 0, 0> { enum { idx = 1 }; };
    template<> struct SevenPt< 0, 1, 0> { enum { idx = 2 }; };
    template<> struct SevenPt< 0, 0, 1> { enum { idx = 3 }; };
    template<> struct SevenPt<-1, 0, 0> { enum { idx = 4 }; };
    template<> struct SevenPt< 0,-1, 0> { enum { idx = 5 }; };
    template<> struct SevenPt< 0, 0,-1> { enum { idx = 6 }; };

}


template<typename GridType>
class SevenPointStencil: public BaseStencil<GridType, SevenPointStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, SevenPointStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType   BufferType;
    typedef typename GridType::ValueType    ValueType;
    typedef math::Vec3<ValueType>           Vec3Type;
    static const int SIZE = 7;

    SevenPointStencil(const GridType& grid): BaseType(grid, SIZE) {}

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    unsigned int pos() const { return SevenPt<i,j,k>::idx; }

private:
    inline void init(const Coord& ijk)
    {
        BaseType::template setValue< 0, 0, 0>(mCache.getValue(ijk));

        BaseType::template setValue<-1, 0, 0>(mCache.getValue(ijk.offsetBy(-1, 0, 0)));
        BaseType::template setValue< 1, 0, 0>(mCache.getValue(ijk.offsetBy( 1, 0, 0)));

        BaseType::template setValue< 0,-1, 0>(mCache.getValue(ijk.offsetBy( 0,-1, 0)));
        BaseType::template setValue< 0, 1, 0>(mCache.getValue(ijk.offsetBy( 0, 1, 0)));

        BaseType::template setValue< 0, 0,-1>(mCache.getValue(ijk.offsetBy( 0, 0,-1)));
        BaseType::template setValue< 0, 0, 1>(mCache.getValue(ijk.offsetBy( 0, 0, 1)));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
};


////////////////////////////////////////


namespace { // anonymous namespace for stencil-layout map

    // the dense point stencil
    template<int i, int j, int k> struct DensePt {};
    template<> struct DensePt< 0, 0, 0> { enum { idx = 0 }; };

    template<> struct DensePt< 1, 0, 0> { enum { idx = 1 }; };
    template<> struct DensePt< 0, 1, 0> { enum { idx = 2 }; };
    template<> struct DensePt< 0, 0, 1> { enum { idx = 3 }; };

    template<> struct DensePt<-1, 0, 0> { enum { idx = 4 }; };
    template<> struct DensePt< 0,-1, 0> { enum { idx = 5 }; };
    template<> struct DensePt< 0, 0,-1> { enum { idx = 6 }; };

    template<> struct DensePt<-1,-1, 0> { enum { idx = 7 }; };
    template<> struct DensePt< 0,-1,-1> { enum { idx = 8 }; };
    template<> struct DensePt<-1, 0,-1> { enum { idx = 9 }; };

    template<> struct DensePt< 1,-1, 0> { enum { idx = 10 }; };
    template<> struct DensePt< 0, 1,-1> { enum { idx = 11 }; };
    template<> struct DensePt<-1, 0, 1> { enum { idx = 12 }; };

    template<> struct DensePt<-1, 1, 0> { enum { idx = 13 }; };
    template<> struct DensePt< 0,-1, 1> { enum { idx = 14 }; };
    template<> struct DensePt< 1, 0,-1> { enum { idx = 15 }; };

    template<> struct DensePt< 1, 1, 0> { enum { idx = 16 }; };
    template<> struct DensePt< 0, 1, 1> { enum { idx = 17 }; };
    template<> struct DensePt< 1, 0, 1> { enum { idx = 18 }; };

}


template<typename GridType>
class SecondOrderDenseStencil: public BaseStencil<GridType, SecondOrderDenseStencil<GridType> >
{
public:
    typedef BaseStencil<GridType,SecondOrderDenseStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType   BufferType;
    typedef typename GridType::ValueType    ValueType;
    typedef math::Vec3<ValueType>           Vec3Type;

    static const int SIZE = 19;

    SecondOrderDenseStencil(const GridType& grid): BaseType(grid, SIZE) {}

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    unsigned int pos() const { return DensePt<i,j,k>::idx; }

private:
    inline void init(const Coord& ijk)
    {
        mStencil[DensePt< 0, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  0));

        mStencil[DensePt< 1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,  0,  0));
        mStencil[DensePt< 0, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  1,  0));
        mStencil[DensePt< 0, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  1));

        mStencil[DensePt<-1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,  0,  0));
        mStencil[DensePt< 0,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, -1,  0));
        mStencil[DensePt< 0, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,  0, -1));

        mStencil[DensePt<-1,-1, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, -1,  0));
        mStencil[DensePt< 1,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, -1,  0));
        mStencil[DensePt<-1, 1, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,  1,  0));
        mStencil[DensePt< 1, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,  1,  0));

        mStencil[DensePt<-1, 0,-1>::idx] = mCache.getValue(ijk.offsetBy(-1,  0, -1));
        mStencil[DensePt< 1, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 1,  0, -1));
        mStencil[DensePt<-1, 0, 1>::idx] = mCache.getValue(ijk.offsetBy(-1,  0,  1));
        mStencil[DensePt< 1, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 1,  0,  1));

        mStencil[DensePt< 0,-1,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, -1, -1));
        mStencil[DensePt< 0, 1,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,  1, -1));
        mStencil[DensePt< 0,-1, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, -1,  1));
        mStencil[DensePt< 0, 1, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,  1,  1));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
};


////////////////////////////////////////


namespace { // anonymous namespace for stencil-layout map

    // the dense point stencil
    template<int i, int j, int k> struct ThirteenPt {};
    template<> struct ThirteenPt< 0, 0, 0> { enum { idx = 0 }; };

    template<> struct ThirteenPt< 1, 0, 0> { enum { idx = 1 }; };
    template<> struct ThirteenPt< 0, 1, 0> { enum { idx = 2 }; };
    template<> struct ThirteenPt< 0, 0, 1> { enum { idx = 3 }; };

    template<> struct ThirteenPt<-1, 0, 0> { enum { idx = 4 }; };
    template<> struct ThirteenPt< 0,-1, 0> { enum { idx = 5 }; };
    template<> struct ThirteenPt< 0, 0,-1> { enum { idx = 6 }; };

    template<> struct ThirteenPt< 2, 0, 0> { enum { idx = 7 }; };
    template<> struct ThirteenPt< 0, 2, 0> { enum { idx = 8 }; };
    template<> struct ThirteenPt< 0, 0, 2> { enum { idx = 9 }; };

    template<> struct ThirteenPt<-2, 0, 0> { enum { idx = 10 }; };
    template<> struct ThirteenPt< 0,-2, 0> { enum { idx = 11 }; };
    template<> struct ThirteenPt< 0, 0,-2> { enum { idx = 12 }; };

}


template<typename GridType>
class ThirteenPointStencil: public BaseStencil<GridType, ThirteenPointStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, ThirteenPointStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType   BufferType;
    typedef typename GridType::ValueType    ValueType;
    typedef math::Vec3<ValueType>           Vec3Type;

    static const int SIZE = 13;

    ThirteenPointStencil(const GridType& grid): BaseType(grid, SIZE) {}

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    unsigned int pos() const { return ThirteenPt<i,j,k>::idx; }

private:
    inline void init(const Coord& ijk)
    {
        mStencil[ThirteenPt< 0, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  0));

        mStencil[ThirteenPt< 2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 2,  0,  0));
        mStencil[ThirteenPt< 1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,  0,  0));
        mStencil[ThirteenPt<-1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,  0,  0));
        mStencil[ThirteenPt<-2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-2,  0,  0));

        mStencil[ThirteenPt< 0, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  2,  0));
        mStencil[ThirteenPt< 0, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  1,  0));
        mStencil[ThirteenPt< 0,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, -1,  0));
        mStencil[ThirteenPt< 0,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, -2,  0));

        mStencil[ThirteenPt< 0, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  2));
        mStencil[ThirteenPt< 0, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  1));
        mStencil[ThirteenPt< 0, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,  0, -1));
        mStencil[ThirteenPt< 0, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 0,  0, -2));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
};


////////////////////////////////////////


namespace { // anonymous namespace for stencil-layout map

    // the 4th-order dense point stencil
    template<int i, int j, int k> struct FourthDensePt {};
    template<> struct FourthDensePt< 0, 0, 0> { enum { idx = 0 }; };

    template<> struct FourthDensePt<-2, 2, 0> { enum { idx = 1 }; };
    template<> struct FourthDensePt<-1, 2, 0> { enum { idx = 2 }; };
    template<> struct FourthDensePt< 0, 2, 0> { enum { idx = 3 }; };
    template<> struct FourthDensePt< 1, 2, 0> { enum { idx = 4 }; };
    template<> struct FourthDensePt< 2, 2, 0> { enum { idx = 5 }; };

    template<> struct FourthDensePt<-2, 1, 0> { enum { idx = 6 }; };
    template<> struct FourthDensePt<-1, 1, 0> { enum { idx = 7 }; };
    template<> struct FourthDensePt< 0, 1, 0> { enum { idx = 8 }; };
    template<> struct FourthDensePt< 1, 1, 0> { enum { idx = 9 }; };
    template<> struct FourthDensePt< 2, 1, 0> { enum { idx = 10 }; };

    template<> struct FourthDensePt<-2, 0, 0> { enum { idx = 11 }; };
    template<> struct FourthDensePt<-1, 0, 0> { enum { idx = 12 }; };
    template<> struct FourthDensePt< 1, 0, 0> { enum { idx = 13 }; };
    template<> struct FourthDensePt< 2, 0, 0> { enum { idx = 14 }; };

    template<> struct FourthDensePt<-2,-1, 0> { enum { idx = 15 }; };
    template<> struct FourthDensePt<-1,-1, 0> { enum { idx = 16 }; };
    template<> struct FourthDensePt< 0,-1, 0> { enum { idx = 17 }; };
    template<> struct FourthDensePt< 1,-1, 0> { enum { idx = 18 }; };
    template<> struct FourthDensePt< 2,-1, 0> { enum { idx = 19 }; };

    template<> struct FourthDensePt<-2,-2, 0> { enum { idx = 20 }; };
    template<> struct FourthDensePt<-1,-2, 0> { enum { idx = 21 }; };
    template<> struct FourthDensePt< 0,-2, 0> { enum { idx = 22 }; };
    template<> struct FourthDensePt< 1,-2, 0> { enum { idx = 23 }; };
    template<> struct FourthDensePt< 2,-2, 0> { enum { idx = 24 }; };


    template<> struct FourthDensePt<-2, 0, 2> { enum { idx = 25 }; };
    template<> struct FourthDensePt<-1, 0, 2> { enum { idx = 26 }; };
    template<> struct FourthDensePt< 0, 0, 2> { enum { idx = 27 }; };
    template<> struct FourthDensePt< 1, 0, 2> { enum { idx = 28 }; };
    template<> struct FourthDensePt< 2, 0, 2> { enum { idx = 29 }; };

    template<> struct FourthDensePt<-2, 0, 1> { enum { idx = 30 }; };
    template<> struct FourthDensePt<-1, 0, 1> { enum { idx = 31 }; };
    template<> struct FourthDensePt< 0, 0, 1> { enum { idx = 32 }; };
    template<> struct FourthDensePt< 1, 0, 1> { enum { idx = 33 }; };
    template<> struct FourthDensePt< 2, 0, 1> { enum { idx = 34 }; };

    template<> struct FourthDensePt<-2, 0,-1> { enum { idx = 35 }; };
    template<> struct FourthDensePt<-1, 0,-1> { enum { idx = 36 }; };
    template<> struct FourthDensePt< 0, 0,-1> { enum { idx = 37 }; };
    template<> struct FourthDensePt< 1, 0,-1> { enum { idx = 38 }; };
    template<> struct FourthDensePt< 2, 0,-1> { enum { idx = 39 }; };

    template<> struct FourthDensePt<-2, 0,-2> { enum { idx = 40 }; };
    template<> struct FourthDensePt<-1, 0,-2> { enum { idx = 41 }; };
    template<> struct FourthDensePt< 0, 0,-2> { enum { idx = 42 }; };
    template<> struct FourthDensePt< 1, 0,-2> { enum { idx = 43 }; };
    template<> struct FourthDensePt< 2, 0,-2> { enum { idx = 44 }; };


    template<> struct FourthDensePt< 0,-2, 2> { enum { idx = 45 }; };
    template<> struct FourthDensePt< 0,-1, 2> { enum { idx = 46 }; };
    template<> struct FourthDensePt< 0, 1, 2> { enum { idx = 47 }; };
    template<> struct FourthDensePt< 0, 2, 2> { enum { idx = 48 }; };

    template<> struct FourthDensePt< 0,-2, 1> { enum { idx = 49 }; };
    template<> struct FourthDensePt< 0,-1, 1> { enum { idx = 50 }; };
    template<> struct FourthDensePt< 0, 1, 1> { enum { idx = 51 }; };
    template<> struct FourthDensePt< 0, 2, 1> { enum { idx = 52 }; };

    template<> struct FourthDensePt< 0,-2,-1> { enum { idx = 53 }; };
    template<> struct FourthDensePt< 0,-1,-1> { enum { idx = 54 }; };
    template<> struct FourthDensePt< 0, 1,-1> { enum { idx = 55 }; };
    template<> struct FourthDensePt< 0, 2,-1> { enum { idx = 56 }; };

    template<> struct FourthDensePt< 0,-2,-2> { enum { idx = 57 }; };
    template<> struct FourthDensePt< 0,-1,-2> { enum { idx = 58 }; };
    template<> struct FourthDensePt< 0, 1,-2> { enum { idx = 59 }; };
    template<> struct FourthDensePt< 0, 2,-2> { enum { idx = 60 }; };

}


template<typename GridType>
class FourthOrderDenseStencil: public BaseStencil<GridType, FourthOrderDenseStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, FourthOrderDenseStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType   BufferType;
    typedef typename GridType::ValueType    ValueType;
    typedef math::Vec3<ValueType>           Vec3Type;

    static const int SIZE = 61;

    FourthOrderDenseStencil(const GridType& grid): BaseType(grid, SIZE) {}

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    unsigned int pos() const { return FourthDensePt<i,j,k>::idx; }

private:
    inline void init(const Coord& ijk)
    {
        mStencil[FourthDensePt< 0, 0, 0>::idx] = mCache.getValue(ijk);

        mStencil[FourthDensePt<-2, 2, 0>::idx] = mCache.getValue(ijk.offsetBy(-2, 2, 0));
        mStencil[FourthDensePt<-1, 2, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, 2, 0));
        mStencil[FourthDensePt< 0, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, 2, 0));
        mStencil[FourthDensePt< 1, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, 2, 0));
        mStencil[FourthDensePt< 2, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 2, 2, 0));

        mStencil[FourthDensePt<-2, 1, 0>::idx] = mCache.getValue(ijk.offsetBy(-2, 1, 0));
        mStencil[FourthDensePt<-1, 1, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, 1, 0));
        mStencil[FourthDensePt< 0, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, 1, 0));
        mStencil[FourthDensePt< 1, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, 1, 0));
        mStencil[FourthDensePt< 2, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 2, 1, 0));

        mStencil[FourthDensePt<-2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-2, 0, 0));
        mStencil[FourthDensePt<-1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, 0, 0));
        mStencil[FourthDensePt< 1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, 0, 0));
        mStencil[FourthDensePt< 2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 2, 0, 0));

        mStencil[FourthDensePt<-2,-1, 0>::idx] = mCache.getValue(ijk.offsetBy(-2,-1, 0));
        mStencil[FourthDensePt<-1,-1, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,-1, 0));
        mStencil[FourthDensePt< 0,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,-1, 0));
        mStencil[FourthDensePt< 1,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,-1, 0));
        mStencil[FourthDensePt< 2,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 2,-1, 0));

        mStencil[FourthDensePt<-2,-2, 0>::idx] = mCache.getValue(ijk.offsetBy(-2,-2, 0));
        mStencil[FourthDensePt<-1,-2, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,-2, 0));
        mStencil[FourthDensePt< 0,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,-2, 0));
        mStencil[FourthDensePt< 1,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,-2, 0));
        mStencil[FourthDensePt< 2,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 2,-2, 0));

        mStencil[FourthDensePt<-2, 0, 2>::idx] = mCache.getValue(ijk.offsetBy(-2, 0, 2));
        mStencil[FourthDensePt<-1, 0, 2>::idx] = mCache.getValue(ijk.offsetBy(-1, 0, 2));
        mStencil[FourthDensePt< 0, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 0, 0, 2));
        mStencil[FourthDensePt< 1, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 1, 0, 2));
        mStencil[FourthDensePt< 2, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 2, 0, 2));

        mStencil[FourthDensePt<-2, 0, 1>::idx] = mCache.getValue(ijk.offsetBy(-2, 0, 1));
        mStencil[FourthDensePt<-1, 0, 1>::idx] = mCache.getValue(ijk.offsetBy(-1, 0, 1));
        mStencil[FourthDensePt< 0, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, 0, 1));
        mStencil[FourthDensePt< 1, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 1, 0, 1));
        mStencil[FourthDensePt< 2, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 2, 0, 1));

        mStencil[FourthDensePt<-2, 0,-1>::idx] = mCache.getValue(ijk.offsetBy(-2, 0,-1));
        mStencil[FourthDensePt<-1, 0,-1>::idx] = mCache.getValue(ijk.offsetBy(-1, 0,-1));
        mStencil[FourthDensePt< 0, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, 0,-1));
        mStencil[FourthDensePt< 1, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 1, 0,-1));
        mStencil[FourthDensePt< 2, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 2, 0,-1));

        mStencil[FourthDensePt<-2, 0,-2>::idx] = mCache.getValue(ijk.offsetBy(-2, 0,-2));
        mStencil[FourthDensePt<-1, 0,-2>::idx] = mCache.getValue(ijk.offsetBy(-1, 0,-2));
        mStencil[FourthDensePt< 0, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 0, 0,-2));
        mStencil[FourthDensePt< 1, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 1, 0,-2));
        mStencil[FourthDensePt< 2, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 2, 0,-2));


        mStencil[FourthDensePt< 0,-2, 2>::idx] = mCache.getValue(ijk.offsetBy( 0,-2, 2));
        mStencil[FourthDensePt< 0,-1, 2>::idx] = mCache.getValue(ijk.offsetBy( 0,-1, 2));
        mStencil[FourthDensePt< 0, 1, 2>::idx] = mCache.getValue(ijk.offsetBy( 0, 1, 2));
        mStencil[FourthDensePt< 0, 2, 2>::idx] = mCache.getValue(ijk.offsetBy( 0, 2, 2));

        mStencil[FourthDensePt< 0,-2, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,-2, 1));
        mStencil[FourthDensePt< 0,-1, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,-1, 1));
        mStencil[FourthDensePt< 0, 1, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, 1, 1));
        mStencil[FourthDensePt< 0, 2, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, 2, 1));

        mStencil[FourthDensePt< 0,-2,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,-2,-1));
        mStencil[FourthDensePt< 0,-1,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,-1,-1));
        mStencil[FourthDensePt< 0, 1,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, 1,-1));
        mStencil[FourthDensePt< 0, 2,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, 2,-1));

        mStencil[FourthDensePt< 0,-2,-2>::idx] = mCache.getValue(ijk.offsetBy( 0,-2,-2));
        mStencil[FourthDensePt< 0,-1,-2>::idx] = mCache.getValue(ijk.offsetBy( 0,-1,-2));
        mStencil[FourthDensePt< 0, 1,-2>::idx] = mCache.getValue(ijk.offsetBy( 0, 1,-2));
        mStencil[FourthDensePt< 0, 2,-2>::idx] = mCache.getValue(ijk.offsetBy( 0, 2,-2));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
};


////////////////////////////////////////


namespace { // anonymous namespace for stencil-layout map

    // the dense point stencil
    template<int i, int j, int k> struct NineteenPt {};
    template<> struct NineteenPt< 0, 0, 0> { enum { idx = 0 }; };

    template<> struct NineteenPt< 1, 0, 0> { enum { idx = 1 }; };
    template<> struct NineteenPt< 0, 1, 0> { enum { idx = 2 }; };
    template<> struct NineteenPt< 0, 0, 1> { enum { idx = 3 }; };

    template<> struct NineteenPt<-1, 0, 0> { enum { idx = 4 }; };
    template<> struct NineteenPt< 0,-1, 0> { enum { idx = 5 }; };
    template<> struct NineteenPt< 0, 0,-1> { enum { idx = 6 }; };

    template<> struct NineteenPt< 2, 0, 0> { enum { idx = 7 }; };
    template<> struct NineteenPt< 0, 2, 0> { enum { idx = 8 }; };
    template<> struct NineteenPt< 0, 0, 2> { enum { idx = 9 }; };

    template<> struct NineteenPt<-2, 0, 0> { enum { idx = 10 }; };
    template<> struct NineteenPt< 0,-2, 0> { enum { idx = 11 }; };
    template<> struct NineteenPt< 0, 0,-2> { enum { idx = 12 }; };

    template<> struct NineteenPt< 3, 0, 0> { enum { idx = 13 }; };
    template<> struct NineteenPt< 0, 3, 0> { enum { idx = 14 }; };
    template<> struct NineteenPt< 0, 0, 3> { enum { idx = 15 }; };

    template<> struct NineteenPt<-3, 0, 0> { enum { idx = 16 }; };
    template<> struct NineteenPt< 0,-3, 0> { enum { idx = 17 }; };
    template<> struct NineteenPt< 0, 0,-3> { enum { idx = 18 }; };

}


template<typename GridType>
class NineteenPointStencil: public BaseStencil<GridType, NineteenPointStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, NineteenPointStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType   BufferType;
    typedef typename GridType::ValueType    ValueType;
    typedef math::Vec3<ValueType>           Vec3Type;

    static const int SIZE = 19;

    NineteenPointStencil(const GridType& grid): BaseType(grid, SIZE) {}

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    unsigned int pos() const { return NineteenPt<i,j,k>::idx; }

private:
    inline void init(const Coord& ijk)
    {
        mStencil[NineteenPt< 3, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 3,  0,  0));
        mStencil[NineteenPt< 2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 2,  0,  0));
        mStencil[NineteenPt< 1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,  0,  0));
        mStencil[NineteenPt<-1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,  0,  0));
        mStencil[NineteenPt<-2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-2,  0,  0));
        mStencil[NineteenPt<-3, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-3,  0,  0));

        mStencil[NineteenPt< 0, 3, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  3,  0));
        mStencil[NineteenPt< 0, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  2,  0));
        mStencil[NineteenPt< 0, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,  1,  0));
        mStencil[NineteenPt< 0,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, -1,  0));
        mStencil[NineteenPt< 0,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, -2,  0));
        mStencil[NineteenPt< 0,-3, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, -3,  0));


        mStencil[NineteenPt< 0, 0, 3>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  3));
        mStencil[NineteenPt< 0, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  2));
        mStencil[NineteenPt< 0, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,  0,  1));
        mStencil[NineteenPt< 0, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,  0, -1));
        mStencil[NineteenPt< 0, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 0,  0, -2));
        mStencil[NineteenPt< 0, 0,-3>::idx] = mCache.getValue(ijk.offsetBy( 0,  0, -3));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
};


////////////////////////////////////////


namespace { // anonymous namespace for stencil-layout map

    // the 4th-order dense point stencil
    template<int i, int j, int k> struct SixthDensePt { };
    template<> struct SixthDensePt< 0, 0, 0> { enum { idx = 0 }; };

    template<> struct SixthDensePt<-3, 3, 0> { enum { idx = 1 }; };
    template<> struct SixthDensePt<-2, 3, 0> { enum { idx = 2 }; };
    template<> struct SixthDensePt<-1, 3, 0> { enum { idx = 3 }; };
    template<> struct SixthDensePt< 0, 3, 0> { enum { idx = 4 }; };
    template<> struct SixthDensePt< 1, 3, 0> { enum { idx = 5 }; };
    template<> struct SixthDensePt< 2, 3, 0> { enum { idx = 6 }; };
    template<> struct SixthDensePt< 3, 3, 0> { enum { idx = 7 }; };

    template<> struct SixthDensePt<-3, 2, 0> { enum { idx = 8 }; };
    template<> struct SixthDensePt<-2, 2, 0> { enum { idx = 9 }; };
    template<> struct SixthDensePt<-1, 2, 0> { enum { idx = 10 }; };
    template<> struct SixthDensePt< 0, 2, 0> { enum { idx = 11 }; };
    template<> struct SixthDensePt< 1, 2, 0> { enum { idx = 12 }; };
    template<> struct SixthDensePt< 2, 2, 0> { enum { idx = 13 }; };
    template<> struct SixthDensePt< 3, 2, 0> { enum { idx = 14 }; };

    template<> struct SixthDensePt<-3, 1, 0> { enum { idx = 15 }; };
    template<> struct SixthDensePt<-2, 1, 0> { enum { idx = 16 }; };
    template<> struct SixthDensePt<-1, 1, 0> { enum { idx = 17 }; };
    template<> struct SixthDensePt< 0, 1, 0> { enum { idx = 18 }; };
    template<> struct SixthDensePt< 1, 1, 0> { enum { idx = 19 }; };
    template<> struct SixthDensePt< 2, 1, 0> { enum { idx = 20 }; };
    template<> struct SixthDensePt< 3, 1, 0> { enum { idx = 21 }; };

    template<> struct SixthDensePt<-3, 0, 0> { enum { idx = 22 }; };
    template<> struct SixthDensePt<-2, 0, 0> { enum { idx = 23 }; };
    template<> struct SixthDensePt<-1, 0, 0> { enum { idx = 24 }; };
    template<> struct SixthDensePt< 1, 0, 0> { enum { idx = 25 }; };
    template<> struct SixthDensePt< 2, 0, 0> { enum { idx = 26 }; };
    template<> struct SixthDensePt< 3, 0, 0> { enum { idx = 27 }; };


    template<> struct SixthDensePt<-3,-1, 0> { enum { idx = 28 }; };
    template<> struct SixthDensePt<-2,-1, 0> { enum { idx = 29 }; };
    template<> struct SixthDensePt<-1,-1, 0> { enum { idx = 30 }; };
    template<> struct SixthDensePt< 0,-1, 0> { enum { idx = 31 }; };
    template<> struct SixthDensePt< 1,-1, 0> { enum { idx = 32 }; };
    template<> struct SixthDensePt< 2,-1, 0> { enum { idx = 33 }; };
    template<> struct SixthDensePt< 3,-1, 0> { enum { idx = 34 }; };


    template<> struct SixthDensePt<-3,-2, 0> { enum { idx = 35 }; };
    template<> struct SixthDensePt<-2,-2, 0> { enum { idx = 36 }; };
    template<> struct SixthDensePt<-1,-2, 0> { enum { idx = 37 }; };
    template<> struct SixthDensePt< 0,-2, 0> { enum { idx = 38 }; };
    template<> struct SixthDensePt< 1,-2, 0> { enum { idx = 39 }; };
    template<> struct SixthDensePt< 2,-2, 0> { enum { idx = 40 }; };
    template<> struct SixthDensePt< 3,-2, 0> { enum { idx = 41 }; };


    template<> struct SixthDensePt<-3,-3, 0> { enum { idx = 42 }; };
    template<> struct SixthDensePt<-2,-3, 0> { enum { idx = 43 }; };
    template<> struct SixthDensePt<-1,-3, 0> { enum { idx = 44 }; };
    template<> struct SixthDensePt< 0,-3, 0> { enum { idx = 45 }; };
    template<> struct SixthDensePt< 1,-3, 0> { enum { idx = 46 }; };
    template<> struct SixthDensePt< 2,-3, 0> { enum { idx = 47 }; };
    template<> struct SixthDensePt< 3,-3, 0> { enum { idx = 48 }; };


    template<> struct SixthDensePt<-3, 0, 3> { enum { idx = 49 }; };
    template<> struct SixthDensePt<-2, 0, 3> { enum { idx = 50 }; };
    template<> struct SixthDensePt<-1, 0, 3> { enum { idx = 51 }; };
    template<> struct SixthDensePt< 0, 0, 3> { enum { idx = 52 }; };
    template<> struct SixthDensePt< 1, 0, 3> { enum { idx = 53 }; };
    template<> struct SixthDensePt< 2, 0, 3> { enum { idx = 54 }; };
    template<> struct SixthDensePt< 3, 0, 3> { enum { idx = 55 }; };


    template<> struct SixthDensePt<-3, 0, 2> { enum { idx = 56 }; };
    template<> struct SixthDensePt<-2, 0, 2> { enum { idx = 57 }; };
    template<> struct SixthDensePt<-1, 0, 2> { enum { idx = 58 }; };
    template<> struct SixthDensePt< 0, 0, 2> { enum { idx = 59 }; };
    template<> struct SixthDensePt< 1, 0, 2> { enum { idx = 60 }; };
    template<> struct SixthDensePt< 2, 0, 2> { enum { idx = 61 }; };
    template<> struct SixthDensePt< 3, 0, 2> { enum { idx = 62 }; };

    template<> struct SixthDensePt<-3, 0, 1> { enum { idx = 63 }; };
    template<> struct SixthDensePt<-2, 0, 1> { enum { idx = 64 }; };
    template<> struct SixthDensePt<-1, 0, 1> { enum { idx = 65 }; };
    template<> struct SixthDensePt< 0, 0, 1> { enum { idx = 66 }; };
    template<> struct SixthDensePt< 1, 0, 1> { enum { idx = 67 }; };
    template<> struct SixthDensePt< 2, 0, 1> { enum { idx = 68 }; };
    template<> struct SixthDensePt< 3, 0, 1> { enum { idx = 69 }; };


    template<> struct SixthDensePt<-3, 0,-1> { enum { idx = 70 }; };
    template<> struct SixthDensePt<-2, 0,-1> { enum { idx = 71 }; };
    template<> struct SixthDensePt<-1, 0,-1> { enum { idx = 72 }; };
    template<> struct SixthDensePt< 0, 0,-1> { enum { idx = 73 }; };
    template<> struct SixthDensePt< 1, 0,-1> { enum { idx = 74 }; };
    template<> struct SixthDensePt< 2, 0,-1> { enum { idx = 75 }; };
    template<> struct SixthDensePt< 3, 0,-1> { enum { idx = 76 }; };


    template<> struct SixthDensePt<-3, 0,-2> { enum { idx = 77 }; };
    template<> struct SixthDensePt<-2, 0,-2> { enum { idx = 78 }; };
    template<> struct SixthDensePt<-1, 0,-2> { enum { idx = 79 }; };
    template<> struct SixthDensePt< 0, 0,-2> { enum { idx = 80 }; };
    template<> struct SixthDensePt< 1, 0,-2> { enum { idx = 81 }; };
    template<> struct SixthDensePt< 2, 0,-2> { enum { idx = 82 }; };
    template<> struct SixthDensePt< 3, 0,-2> { enum { idx = 83 }; };


    template<> struct SixthDensePt<-3, 0,-3> { enum { idx = 84 }; };
    template<> struct SixthDensePt<-2, 0,-3> { enum { idx = 85 }; };
    template<> struct SixthDensePt<-1, 0,-3> { enum { idx = 86 }; };
    template<> struct SixthDensePt< 0, 0,-3> { enum { idx = 87 }; };
    template<> struct SixthDensePt< 1, 0,-3> { enum { idx = 88 }; };
    template<> struct SixthDensePt< 2, 0,-3> { enum { idx = 89 }; };
    template<> struct SixthDensePt< 3, 0,-3> { enum { idx = 90 }; };


    template<> struct SixthDensePt< 0,-3, 3> { enum { idx = 91 }; };
    template<> struct SixthDensePt< 0,-2, 3> { enum { idx = 92 }; };
    template<> struct SixthDensePt< 0,-1, 3> { enum { idx = 93 }; };
    template<> struct SixthDensePt< 0, 1, 3> { enum { idx = 94 }; };
    template<> struct SixthDensePt< 0, 2, 3> { enum { idx = 95 }; };
    template<> struct SixthDensePt< 0, 3, 3> { enum { idx = 96 }; };

    template<> struct SixthDensePt< 0,-3, 2> { enum { idx = 97 }; };
    template<> struct SixthDensePt< 0,-2, 2> { enum { idx = 98 }; };
    template<> struct SixthDensePt< 0,-1, 2> { enum { idx = 99 }; };
    template<> struct SixthDensePt< 0, 1, 2> { enum { idx = 100 }; };
    template<> struct SixthDensePt< 0, 2, 2> { enum { idx = 101 }; };
    template<> struct SixthDensePt< 0, 3, 2> { enum { idx = 102 }; };

    template<> struct SixthDensePt< 0,-3, 1> { enum { idx = 103 }; };
    template<> struct SixthDensePt< 0,-2, 1> { enum { idx = 104 }; };
    template<> struct SixthDensePt< 0,-1, 1> { enum { idx = 105 }; };
    template<> struct SixthDensePt< 0, 1, 1> { enum { idx = 106 }; };
    template<> struct SixthDensePt< 0, 2, 1> { enum { idx = 107 }; };
    template<> struct SixthDensePt< 0, 3, 1> { enum { idx = 108 }; };

    template<> struct SixthDensePt< 0,-3,-1> { enum { idx = 109 }; };
    template<> struct SixthDensePt< 0,-2,-1> { enum { idx = 110 }; };
    template<> struct SixthDensePt< 0,-1,-1> { enum { idx = 111 }; };
    template<> struct SixthDensePt< 0, 1,-1> { enum { idx = 112 }; };
    template<> struct SixthDensePt< 0, 2,-1> { enum { idx = 113 }; };
    template<> struct SixthDensePt< 0, 3,-1> { enum { idx = 114 }; };

    template<> struct SixthDensePt< 0,-3,-2> { enum { idx = 115 }; };
    template<> struct SixthDensePt< 0,-2,-2> { enum { idx = 116 }; };
    template<> struct SixthDensePt< 0,-1,-2> { enum { idx = 117 }; };
    template<> struct SixthDensePt< 0, 1,-2> { enum { idx = 118 }; };
    template<> struct SixthDensePt< 0, 2,-2> { enum { idx = 119 }; };
    template<> struct SixthDensePt< 0, 3,-2> { enum { idx = 120 }; };

    template<> struct SixthDensePt< 0,-3,-3> { enum { idx = 121 }; };
    template<> struct SixthDensePt< 0,-2,-3> { enum { idx = 122 }; };
    template<> struct SixthDensePt< 0,-1,-3> { enum { idx = 123 }; };
    template<> struct SixthDensePt< 0, 1,-3> { enum { idx = 124 }; };
    template<> struct SixthDensePt< 0, 2,-3> { enum { idx = 125 }; };
    template<> struct SixthDensePt< 0, 3,-3> { enum { idx = 126 }; };

}


template<typename GridType>
class SixthOrderDenseStencil: public BaseStencil<GridType, SixthOrderDenseStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, SixthOrderDenseStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType   BufferType;
    typedef typename GridType::ValueType    ValueType;
    typedef math::Vec3<ValueType>           Vec3Type;

    static const int SIZE = 127;

    SixthOrderDenseStencil(const GridType& grid): BaseType(grid, SIZE) {}

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    unsigned int pos() const { return SixthDensePt<i,j,k>::idx; }

private:
    inline void init(const Coord& ijk)
    {
        mStencil[SixthDensePt< 0, 0, 0>::idx] = mCache.getValue(ijk);

        mStencil[SixthDensePt<-3, 3, 0>::idx] = mCache.getValue(ijk.offsetBy(-3, 3, 0));
        mStencil[SixthDensePt<-2, 3, 0>::idx] = mCache.getValue(ijk.offsetBy(-2, 3, 0));
        mStencil[SixthDensePt<-1, 3, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, 3, 0));
        mStencil[SixthDensePt< 0, 3, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, 3, 0));
        mStencil[SixthDensePt< 1, 3, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, 3, 0));
        mStencil[SixthDensePt< 2, 3, 0>::idx] = mCache.getValue(ijk.offsetBy( 2, 3, 0));
        mStencil[SixthDensePt< 3, 3, 0>::idx] = mCache.getValue(ijk.offsetBy( 3, 3, 0));

        mStencil[SixthDensePt<-3, 2, 0>::idx] = mCache.getValue(ijk.offsetBy(-3, 2, 0));
        mStencil[SixthDensePt<-2, 2, 0>::idx] = mCache.getValue(ijk.offsetBy(-2, 2, 0));
        mStencil[SixthDensePt<-1, 2, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, 2, 0));
        mStencil[SixthDensePt< 0, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, 2, 0));
        mStencil[SixthDensePt< 1, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, 2, 0));
        mStencil[SixthDensePt< 2, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 2, 2, 0));
        mStencil[SixthDensePt< 3, 2, 0>::idx] = mCache.getValue(ijk.offsetBy( 3, 2, 0));

        mStencil[SixthDensePt<-3, 1, 0>::idx] = mCache.getValue(ijk.offsetBy(-3, 1, 0));
        mStencil[SixthDensePt<-2, 1, 0>::idx] = mCache.getValue(ijk.offsetBy(-2, 1, 0));
        mStencil[SixthDensePt<-1, 1, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, 1, 0));
        mStencil[SixthDensePt< 0, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0, 1, 0));
        mStencil[SixthDensePt< 1, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, 1, 0));
        mStencil[SixthDensePt< 2, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 2, 1, 0));
        mStencil[SixthDensePt< 3, 1, 0>::idx] = mCache.getValue(ijk.offsetBy( 3, 1, 0));

        mStencil[SixthDensePt<-3, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-3, 0, 0));
        mStencil[SixthDensePt<-2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-2, 0, 0));
        mStencil[SixthDensePt<-1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy(-1, 0, 0));
        mStencil[SixthDensePt< 1, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 1, 0, 0));
        mStencil[SixthDensePt< 2, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 2, 0, 0));
        mStencil[SixthDensePt< 3, 0, 0>::idx] = mCache.getValue(ijk.offsetBy( 3, 0, 0));

        mStencil[SixthDensePt<-3,-1, 0>::idx] = mCache.getValue(ijk.offsetBy(-3,-1, 0));
        mStencil[SixthDensePt<-2,-1, 0>::idx] = mCache.getValue(ijk.offsetBy(-2,-1, 0));
        mStencil[SixthDensePt<-1,-1, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,-1, 0));
        mStencil[SixthDensePt< 0,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,-1, 0));
        mStencil[SixthDensePt< 1,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,-1, 0));
        mStencil[SixthDensePt< 2,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 2,-1, 0));
        mStencil[SixthDensePt< 3,-1, 0>::idx] = mCache.getValue(ijk.offsetBy( 3,-1, 0));

        mStencil[SixthDensePt<-3,-2, 0>::idx] = mCache.getValue(ijk.offsetBy(-3,-2, 0));
        mStencil[SixthDensePt<-2,-2, 0>::idx] = mCache.getValue(ijk.offsetBy(-2,-2, 0));
        mStencil[SixthDensePt<-1,-2, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,-2, 0));
        mStencil[SixthDensePt< 0,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,-2, 0));
        mStencil[SixthDensePt< 1,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,-2, 0));
        mStencil[SixthDensePt< 2,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 2,-2, 0));
        mStencil[SixthDensePt< 3,-2, 0>::idx] = mCache.getValue(ijk.offsetBy( 3,-2, 0));

        mStencil[SixthDensePt<-3,-3, 0>::idx] = mCache.getValue(ijk.offsetBy(-3,-3, 0));
        mStencil[SixthDensePt<-2,-3, 0>::idx] = mCache.getValue(ijk.offsetBy(-2,-3, 0));
        mStencil[SixthDensePt<-1,-3, 0>::idx] = mCache.getValue(ijk.offsetBy(-1,-3, 0));
        mStencil[SixthDensePt< 0,-3, 0>::idx] = mCache.getValue(ijk.offsetBy( 0,-3, 0));
        mStencil[SixthDensePt< 1,-3, 0>::idx] = mCache.getValue(ijk.offsetBy( 1,-3, 0));
        mStencil[SixthDensePt< 2,-3, 0>::idx] = mCache.getValue(ijk.offsetBy( 2,-3, 0));
        mStencil[SixthDensePt< 3,-3, 0>::idx] = mCache.getValue(ijk.offsetBy( 3,-3, 0));

        mStencil[SixthDensePt<-3, 0, 3>::idx] = mCache.getValue(ijk.offsetBy(-3, 0, 3));
        mStencil[SixthDensePt<-2, 0, 3>::idx] = mCache.getValue(ijk.offsetBy(-2, 0, 3));
        mStencil[SixthDensePt<-1, 0, 3>::idx] = mCache.getValue(ijk.offsetBy(-1, 0, 3));
        mStencil[SixthDensePt< 0, 0, 3>::idx] = mCache.getValue(ijk.offsetBy( 0, 0, 3));
        mStencil[SixthDensePt< 1, 0, 3>::idx] = mCache.getValue(ijk.offsetBy( 1, 0, 3));
        mStencil[SixthDensePt< 2, 0, 3>::idx] = mCache.getValue(ijk.offsetBy( 2, 0, 3));
        mStencil[SixthDensePt< 3, 0, 3>::idx] = mCache.getValue(ijk.offsetBy( 3, 0, 3));

        mStencil[SixthDensePt<-3, 0, 2>::idx] = mCache.getValue(ijk.offsetBy(-3, 0, 2));
        mStencil[SixthDensePt<-2, 0, 2>::idx] = mCache.getValue(ijk.offsetBy(-2, 0, 2));
        mStencil[SixthDensePt<-1, 0, 2>::idx] = mCache.getValue(ijk.offsetBy(-1, 0, 2));
        mStencil[SixthDensePt< 0, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 0, 0, 2));
        mStencil[SixthDensePt< 1, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 1, 0, 2));
        mStencil[SixthDensePt< 2, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 2, 0, 2));
        mStencil[SixthDensePt< 3, 0, 2>::idx] = mCache.getValue(ijk.offsetBy( 3, 0, 2));

        mStencil[SixthDensePt<-3, 0, 1>::idx] = mCache.getValue(ijk.offsetBy(-3, 0, 1));
        mStencil[SixthDensePt<-2, 0, 1>::idx] = mCache.getValue(ijk.offsetBy(-2, 0, 1));
        mStencil[SixthDensePt<-1, 0, 1>::idx] = mCache.getValue(ijk.offsetBy(-1, 0, 1));
        mStencil[SixthDensePt< 0, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, 0, 1));
        mStencil[SixthDensePt< 1, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 1, 0, 1));
        mStencil[SixthDensePt< 2, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 2, 0, 1));
        mStencil[SixthDensePt< 3, 0, 1>::idx] = mCache.getValue(ijk.offsetBy( 3, 0, 1));

        mStencil[SixthDensePt<-3, 0,-1>::idx] = mCache.getValue(ijk.offsetBy(-3, 0,-1));
        mStencil[SixthDensePt<-2, 0,-1>::idx] = mCache.getValue(ijk.offsetBy(-2, 0,-1));
        mStencil[SixthDensePt<-1, 0,-1>::idx] = mCache.getValue(ijk.offsetBy(-1, 0,-1));
        mStencil[SixthDensePt< 0, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, 0,-1));
        mStencil[SixthDensePt< 1, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 1, 0,-1));
        mStencil[SixthDensePt< 2, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 2, 0,-1));
        mStencil[SixthDensePt< 3, 0,-1>::idx] = mCache.getValue(ijk.offsetBy( 3, 0,-1));

        mStencil[SixthDensePt<-3, 0,-2>::idx] = mCache.getValue(ijk.offsetBy(-3, 0,-2));
        mStencil[SixthDensePt<-2, 0,-2>::idx] = mCache.getValue(ijk.offsetBy(-2, 0,-2));
        mStencil[SixthDensePt<-1, 0,-2>::idx] = mCache.getValue(ijk.offsetBy(-1, 0,-2));
        mStencil[SixthDensePt< 0, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 0, 0,-2));
        mStencil[SixthDensePt< 1, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 1, 0,-2));
        mStencil[SixthDensePt< 2, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 2, 0,-2));
        mStencil[SixthDensePt< 3, 0,-2>::idx] = mCache.getValue(ijk.offsetBy( 3, 0,-2));

        mStencil[SixthDensePt<-3, 0,-3>::idx] = mCache.getValue(ijk.offsetBy(-3, 0,-3));
        mStencil[SixthDensePt<-2, 0,-3>::idx] = mCache.getValue(ijk.offsetBy(-2, 0,-3));
        mStencil[SixthDensePt<-1, 0,-3>::idx] = mCache.getValue(ijk.offsetBy(-1, 0,-3));
        mStencil[SixthDensePt< 0, 0,-3>::idx] = mCache.getValue(ijk.offsetBy( 0, 0,-3));
        mStencil[SixthDensePt< 1, 0,-3>::idx] = mCache.getValue(ijk.offsetBy( 1, 0,-3));
        mStencil[SixthDensePt< 2, 0,-3>::idx] = mCache.getValue(ijk.offsetBy( 2, 0,-3));
        mStencil[SixthDensePt< 3, 0,-3>::idx] = mCache.getValue(ijk.offsetBy( 3, 0,-3));

        mStencil[SixthDensePt< 0,-3, 3>::idx] = mCache.getValue(ijk.offsetBy( 0,-3, 3));
        mStencil[SixthDensePt< 0,-2, 3>::idx] = mCache.getValue(ijk.offsetBy( 0,-2, 3));
        mStencil[SixthDensePt< 0,-1, 3>::idx] = mCache.getValue(ijk.offsetBy( 0,-1, 3));
        mStencil[SixthDensePt< 0, 1, 3>::idx] = mCache.getValue(ijk.offsetBy( 0, 1, 3));
        mStencil[SixthDensePt< 0, 2, 3>::idx] = mCache.getValue(ijk.offsetBy( 0, 2, 3));
        mStencil[SixthDensePt< 0, 3, 3>::idx] = mCache.getValue(ijk.offsetBy( 0, 3, 3));

        mStencil[SixthDensePt< 0,-3, 2>::idx] = mCache.getValue(ijk.offsetBy( 0,-3, 2));
        mStencil[SixthDensePt< 0,-2, 2>::idx] = mCache.getValue(ijk.offsetBy( 0,-2, 2));
        mStencil[SixthDensePt< 0,-1, 2>::idx] = mCache.getValue(ijk.offsetBy( 0,-1, 2));
        mStencil[SixthDensePt< 0, 1, 2>::idx] = mCache.getValue(ijk.offsetBy( 0, 1, 2));
        mStencil[SixthDensePt< 0, 2, 2>::idx] = mCache.getValue(ijk.offsetBy( 0, 2, 2));
        mStencil[SixthDensePt< 0, 3, 2>::idx] = mCache.getValue(ijk.offsetBy( 0, 3, 2));

        mStencil[SixthDensePt< 0,-3, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,-3, 1));
        mStencil[SixthDensePt< 0,-2, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,-2, 1));
        mStencil[SixthDensePt< 0,-1, 1>::idx] = mCache.getValue(ijk.offsetBy( 0,-1, 1));
        mStencil[SixthDensePt< 0, 1, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, 1, 1));
        mStencil[SixthDensePt< 0, 2, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, 2, 1));
        mStencil[SixthDensePt< 0, 3, 1>::idx] = mCache.getValue(ijk.offsetBy( 0, 3, 1));

        mStencil[SixthDensePt< 0,-3,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,-3,-1));
        mStencil[SixthDensePt< 0,-2,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,-2,-1));
        mStencil[SixthDensePt< 0,-1,-1>::idx] = mCache.getValue(ijk.offsetBy( 0,-1,-1));
        mStencil[SixthDensePt< 0, 1,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, 1,-1));
        mStencil[SixthDensePt< 0, 2,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, 2,-1));
        mStencil[SixthDensePt< 0, 3,-1>::idx] = mCache.getValue(ijk.offsetBy( 0, 3,-1));

        mStencil[SixthDensePt< 0,-3,-2>::idx] = mCache.getValue(ijk.offsetBy( 0,-3,-2));
        mStencil[SixthDensePt< 0,-2,-2>::idx] = mCache.getValue(ijk.offsetBy( 0,-2,-2));
        mStencil[SixthDensePt< 0,-1,-2>::idx] = mCache.getValue(ijk.offsetBy( 0,-1,-2));
        mStencil[SixthDensePt< 0, 1,-2>::idx] = mCache.getValue(ijk.offsetBy( 0, 1,-2));
        mStencil[SixthDensePt< 0, 2,-2>::idx] = mCache.getValue(ijk.offsetBy( 0, 2,-2));
        mStencil[SixthDensePt< 0, 3,-2>::idx] = mCache.getValue(ijk.offsetBy( 0, 3,-2));

        mStencil[SixthDensePt< 0,-3,-3>::idx] = mCache.getValue(ijk.offsetBy( 0,-3,-3));
        mStencil[SixthDensePt< 0,-2,-3>::idx] = mCache.getValue(ijk.offsetBy( 0,-2,-3));
        mStencil[SixthDensePt< 0,-1,-3>::idx] = mCache.getValue(ijk.offsetBy( 0,-1,-3));
        mStencil[SixthDensePt< 0, 1,-3>::idx] = mCache.getValue(ijk.offsetBy( 0, 1,-3));
        mStencil[SixthDensePt< 0, 2,-3>::idx] = mCache.getValue(ijk.offsetBy( 0, 2,-3));
        mStencil[SixthDensePt< 0, 3,-3>::idx] = mCache.getValue(ijk.offsetBy( 0, 3,-3));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
};


//////////////////////////////////////////////////////////////////////


/// This is a simple 7-point nearest neighbor stencil that supports
/// gradient by second-order central differencing, first-order upwinding,
/// Laplacian, closest-point transform and zero-crossing test.
///
/// @note For optimal random access performance this class
/// includes its own grid accessor.
template<typename GridType>
class GradStencil: public BaseStencil<GridType, GradStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, GradStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType                 BufferType;
    typedef typename GridType::ValueType                  ValueType;
    typedef math::Vec3<ValueType>                         Vec3Type;

    static const int SIZE = 7;

    GradStencil(const GridType& grid):
        BaseType(grid, SIZE),
        mInv2Dx(ValueType(0.5 / grid.voxelSize()[0])),
        mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    GradStencil(const GridType& grid, Real dx):
        BaseType(grid, SIZE),
        mInv2Dx(ValueType(0.5 / dx)),
        mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    /// @brief Return the norm square of the single-sided upwind gradient
    /// (computed via Gudonov's scheme) at the previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline ValueType normSqGrad() const
    {
        return mInvDx2 * math::GudonovsNormSqrd(mStencil[0] > 0,
                                                mStencil[0] - mStencil[1],
                                                mStencil[2] - mStencil[0],
                                                mStencil[0] - mStencil[3],
                                                mStencil[4] - mStencil[0],
                                                mStencil[0] - mStencil[5],
                                                mStencil[6] - mStencil[0]);
    }

    /// @brief Return the gradient computed at the previously buffered
    /// location by second order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline Vec3Type gradient() const
    {
        return Vec3Type(mStencil[2] - mStencil[1],
                        mStencil[4] - mStencil[3],
                        mStencil[6] - mStencil[5])*mInv2Dx;
    }
    /// @brief Return the first-order upwind gradient corresponding to the direction V.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline Vec3Type gradient(const Vec3Type& V) const
    {
        return Vec3Type(V[0]>0 ? mStencil[0] - mStencil[1] : mStencil[2] - mStencil[0],
                        V[1]>0 ? mStencil[0] - mStencil[3] : mStencil[4] - mStencil[0],
                        V[2]>0 ? mStencil[0] - mStencil[5] : mStencil[6] - mStencil[0])*2*mInv2Dx;
    }

    /// Return the Laplacian computed at the previously buffered
    /// location by second-order central differencing.
    inline ValueType laplacian() const
    {
        return mInvDx2 * (mStencil[1] + mStencil[2] +
                          mStencil[3] + mStencil[4] +
                          mStencil[5] + mStencil[6] - 6*mStencil[0]);
    }

    /// Return @c true if the sign of the value at the center point of the stencil
    /// is different from the signs of any of its six nearest neighbors.
    inline bool zeroCrossing() const
    {
        const BufferType& v = mStencil;
        return (v[0]>0 ? (v[1]<0 || v[2]<0 || v[3]<0 || v[4]<0 || v[5]<0 || v[6]<0)
                       : (v[1]>0 || v[2]>0 || v[3]>0 || v[4]>0 || v[5]>0 || v[6]>0));
    }

    /// @brief Compute the closest-point transform to a level set.
    /// @return the closest point in index space to the surface
    /// from which the level set was derived.
    ///
    /// @note This method assumes that the grid represents a level set
    /// with distances in world units and a simple affine transfrom
    /// with uniform scaling.
    inline Vec3Type cpt()
    {
        const Coord& ijk = BaseType::getCenterCoord();
        const ValueType d = ValueType(mStencil[0] * 0.5 * mInvDx2); // distance in voxels / (2dx^2)
        return Vec3Type(ijk[0] - d*(mStencil[2] - mStencil[1]),
                        ijk[1] - d*(mStencil[4] - mStencil[3]),
                        ijk[2] - d*(mStencil[6] - mStencil[5]));
    }

private:
    inline void init(const Coord& ijk)
    {
        mStencil[1] = mCache.getValue(ijk.offsetBy(-1,  0,  0));
        mStencil[2] = mCache.getValue(ijk.offsetBy( 1,  0,  0));

        mStencil[3] = mCache.getValue(ijk.offsetBy( 0, -1,  0));
        mStencil[4] = mCache.getValue(ijk.offsetBy( 0,  1,  0));

        mStencil[5] = mCache.getValue(ijk.offsetBy( 0,  0, -1));
        mStencil[6] = mCache.getValue(ijk.offsetBy( 0,  0,  1));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
    const ValueType mInv2Dx, mInvDx2;
}; // class GradStencil


////////////////////////////////////////


/// @brief This is a special 19-point stencil that supports optimal fifth-order WENO
/// upwinding, second-order central differencing, Laplacian, and zero-crossing test.
///
/// @note For optimal random access performance this class
/// includes its own grid accessor.
template<typename GridType>
class WenoStencil: public BaseStencil<GridType, WenoStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, WenoStencil<GridType> > BaseType;
    typedef typename BaseType::BufferType                 BufferType;
    typedef typename GridType::ValueType                  ValueType;
    typedef math::Vec3<ValueType>                         Vec3Type;

    static const int SIZE = 19;

    WenoStencil(const GridType& grid):
        BaseType(grid, SIZE),
        mDx2(ValueType(math::Pow2(grid.voxelSize()[0]))),
        mInv2Dx(ValueType(0.5 / grid.voxelSize()[0])),
        mInvDx2(ValueType(1.0 / mDx2))
    {
    }

    WenoStencil(const GridType& grid, Real dx):
        BaseType(grid, SIZE),
        mDx2(ValueType(dx * dx)),
        mInv2Dx(ValueType(0.5 / dx)),
        mInvDx2(ValueType(1.0 / mDx2))
    {
    }

    /// @brief Return the norm-square of the WENO upwind gradient (computed via
    /// WENO upwinding and Gudonov's scheme) at the previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline ValueType normSqGrad() const
    {
        const BufferType& v = mStencil;
#ifdef DWA_OPENVDB
        // SSE optimized
        const simd::Float4
            v1(v[2]-v[1], v[ 8]-v[ 7], v[14]-v[13], 0),
            v2(v[3]-v[2], v[ 9]-v[ 8], v[15]-v[14], 0),
            v3(v[0]-v[3], v[ 0]-v[ 9], v[ 0]-v[15], 0),
            v4(v[4]-v[0], v[10]-v[ 0], v[16]-v[ 0], 0),
            v5(v[5]-v[4], v[11]-v[10], v[17]-v[16], 0),
            v6(v[6]-v[5], v[12]-v[11], v[18]-v[17], 0),
            dP_m = math::WENO5(v1, v2, v3, v4, v5, mDx2),
            dP_p = math::WENO5(v6, v5, v4, v3, v2, mDx2);

        return mInvDx2 * math::GudonovsNormSqrd(mStencil[0] > 0, dP_m, dP_p);
#else
        const Real
            dP_xm = math::WENO5(v[ 2]-v[ 1],v[ 3]-v[ 2],v[ 0]-v[ 3],v[ 4]-v[ 0],v[ 5]-v[ 4],mDx2),
            dP_xp = math::WENO5(v[ 6]-v[ 5],v[ 5]-v[ 4],v[ 4]-v[ 0],v[ 0]-v[ 3],v[ 3]-v[ 2],mDx2),
            dP_ym = math::WENO5(v[ 8]-v[ 7],v[ 9]-v[ 8],v[ 0]-v[ 9],v[10]-v[ 0],v[11]-v[10],mDx2),
            dP_yp = math::WENO5(v[12]-v[11],v[11]-v[10],v[10]-v[ 0],v[ 0]-v[ 9],v[ 9]-v[ 8],mDx2),
            dP_zm = math::WENO5(v[14]-v[13],v[15]-v[14],v[ 0]-v[15],v[16]-v[ 0],v[17]-v[16],mDx2),
            dP_zp = math::WENO5(v[18]-v[17],v[17]-v[16],v[16]-v[ 0],v[ 0]-v[15],v[15]-v[14],mDx2);
        return mInvDx2*math::GudonovsNormSqrd(v[0]>0,dP_xm,dP_xp,dP_ym,dP_yp,dP_zm,dP_zp);
#endif
    }

    /// Return the optimal fifth-order upwind gradient corresponding to the
    /// direction V.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline Vec3Type gradient(const Vec3Type& V) const
    {
        const BufferType& v = mStencil;
        return 2*mInv2Dx * Vec3Type(
            V[0]>0 ? math::WENO5(v[ 2]-v[ 1],v[ 3]-v[ 2],v[ 0]-v[ 3], v[ 4]-v[ 0],v[ 5]-v[ 4],mDx2)
                : math::WENO5(v[ 6]-v[ 5],v[ 5]-v[ 4],v[ 4]-v[ 0], v[ 0]-v[ 3],v[ 3]-v[ 2],mDx2),
            V[1]>0 ? math::WENO5(v[ 8]-v[ 7],v[ 9]-v[ 8],v[ 0]-v[ 9], v[10]-v[ 0],v[11]-v[10],mDx2)
                : math::WENO5(v[12]-v[11],v[11]-v[10],v[10]-v[ 0], v[ 0]-v[ 9],v[ 9]-v[ 8],mDx2),
            V[2]>0 ? math::WENO5(v[14]-v[13],v[15]-v[14],v[ 0]-v[15], v[16]-v[ 0],v[17]-v[16],mDx2)
                : math::WENO5(v[18]-v[17],v[17]-v[16],v[16]-v[ 0], v[ 0]-v[15],v[15]-v[14],mDx2));
    }
    /// Return the gradient computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline Vec3Type gradient() const
    {
        return mInv2Dx * Vec3Type(
            mStencil[ 4] - mStencil[ 3],
            mStencil[10] - mStencil[ 9],
            mStencil[16] - mStencil[15]);
    }

    /// Return the Laplacian computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline ValueType laplacian() const
    {
        return mInvDx2 * (
            mStencil[ 3] + mStencil[ 4] +
            mStencil[ 9] + mStencil[10] +
            mStencil[15] + mStencil[16] - 6*mStencil[0]);
    }

    /// Return @c true if the sign of the value at the center point of the stencil
    /// differs from the sign of any of its six nearest neighbors
    inline bool zeroCrossing() const
    {
        const BufferType& v = mStencil;
        return (v[ 0]>0 ? (v[ 3]<0 || v[ 4]<0 || v[ 9]<0 || v[10]<0 || v[15]<0 || v[16]<0)
                        : (v[ 3]>0 || v[ 4]>0 || v[ 9]>0 || v[10]>0 || v[15]>0 || v[16]>0));
    }

private:
    inline void init(const Coord& ijk)
    {
        mStencil[ 1] = mCache.getValue(ijk.offsetBy(-3,  0,  0));
        mStencil[ 2] = mCache.getValue(ijk.offsetBy(-2,  0,  0));
        mStencil[ 3] = mCache.getValue(ijk.offsetBy(-1,  0,  0));
        mStencil[ 4] = mCache.getValue(ijk.offsetBy( 1,  0,  0));
        mStencil[ 5] = mCache.getValue(ijk.offsetBy( 2,  0,  0));
        mStencil[ 6] = mCache.getValue(ijk.offsetBy( 3,  0,  0));

        mStencil[ 7] = mCache.getValue(ijk.offsetBy( 0, -3,  0));
        mStencil[ 8] = mCache.getValue(ijk.offsetBy( 0, -2,  0));
        mStencil[ 9] = mCache.getValue(ijk.offsetBy( 0, -1,  0));
        mStencil[10] = mCache.getValue(ijk.offsetBy( 0,  1,  0));
        mStencil[11] = mCache.getValue(ijk.offsetBy( 0,  2,  0));
        mStencil[12] = mCache.getValue(ijk.offsetBy( 0,  3,  0));

        mStencil[13] = mCache.getValue(ijk.offsetBy( 0,  0, -3));
        mStencil[14] = mCache.getValue(ijk.offsetBy( 0,  0, -2));
        mStencil[15] = mCache.getValue(ijk.offsetBy( 0,  0, -1));
        mStencil[16] = mCache.getValue(ijk.offsetBy( 0,  0,  1));
        mStencil[17] = mCache.getValue(ijk.offsetBy( 0,  0,  2));
        mStencil[18] = mCache.getValue(ijk.offsetBy( 0,  0,  3));
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
    const ValueType mDx2, mInv2Dx, mInvDx2;
}; // class WenoStencil


//////////////////////////////////////////////////////////////////////


template<typename GridType>
class CurvatureStencil: public BaseStencil<GridType, CurvatureStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, CurvatureStencil<GridType> > BaseType;
    typedef typename GridType::ValueType                       ValueType;

     static const int SIZE = 19;

    CurvatureStencil(const GridType& grid):
        BaseType(grid, SIZE),
        mInv2Dx(ValueType(0.5 / grid.voxelSize()[0])),
        mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    CurvatureStencil(const GridType& grid, Real dx):
        BaseType(grid, SIZE),
        mInv2Dx(ValueType(0.5 / dx)),
        mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    /// @brief Return the mean curvature at the previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline ValueType meanCurvature()
    {
        Real alpha, beta;
        this->meanCurvature(alpha, beta);
        return ValueType(alpha*mInv2Dx/math::Pow3(beta));
    }

    /// Return the mean curvature multiplied by the norm of the
    /// central-difference gradient. This method is very useful for
    /// mean-curvature flow of level sets!
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline ValueType meanCurvatureNormGrad()
    {
        Real alpha, beta;
        this->meanCurvature(alpha, beta);
        return ValueType(alpha*mInvDx2/(2*math::Pow2(beta)));
    }

    /// Return the Laplacian computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline ValueType laplacian() const
    {
        return mInvDx2 * (
            mStencil[1] + mStencil[2] +
            mStencil[3] + mStencil[4] +
            mStencil[5] + mStencil[6] - 6*mStencil[0]);
    }

    /// Return the gradient computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    inline math::Vec3<ValueType> gradient()
    {
        return math::Vec3<ValueType>(
            mStencil[2] - mStencil[1],
            mStencil[4] - mStencil[3],
            mStencil[6] - mStencil[5])*mInv2Dx;
    }

private:
    inline void init(const Coord &ijk)
    {
        mStencil[ 1] = mCache.getValue(ijk.offsetBy(-1,  0,  0));
        mStencil[ 2] = mCache.getValue(ijk.offsetBy( 1,  0,  0));

        mStencil[ 3] = mCache.getValue(ijk.offsetBy( 0, -1,  0));
        mStencil[ 4] = mCache.getValue(ijk.offsetBy( 0,  1,  0));

        mStencil[ 5] = mCache.getValue(ijk.offsetBy( 0,  0, -1));
        mStencil[ 6] = mCache.getValue(ijk.offsetBy( 0,  0,  1));

        mStencil[ 7] = mCache.getValue(ijk.offsetBy(-1, -1,  0));
        mStencil[ 8] = mCache.getValue(ijk.offsetBy( 1, -1,  0));
        mStencil[ 9] = mCache.getValue(ijk.offsetBy(-1,  1,  0));
        mStencil[10] = mCache.getValue(ijk.offsetBy( 1,  1,  0));

        mStencil[11] = mCache.getValue(ijk.offsetBy(-1,  0, -1));
        mStencil[12] = mCache.getValue(ijk.offsetBy( 1,  0, -1));
        mStencil[13] = mCache.getValue(ijk.offsetBy(-1,  0,  1));
        mStencil[14] = mCache.getValue(ijk.offsetBy( 1,  0,  1));

        mStencil[15] = mCache.getValue(ijk.offsetBy( 0, -1, -1));
        mStencil[16] = mCache.getValue(ijk.offsetBy( 0,  1, -1));
        mStencil[17] = mCache.getValue(ijk.offsetBy( 0, -1,  1));
        mStencil[18] = mCache.getValue(ijk.offsetBy( 0,  1,  1));
    }

    inline void meanCurvature(Real& alpha, Real& beta) const
    {
        // For performance all finite differences are unscaled wrt dx
        const Real
            Half(0.5), Quarter(0.25),
            Dx  = Half * (mStencil[2] - mStencil[1]), Dx2 = Dx * Dx, // * 1/dx
            Dy  = Half * (mStencil[4] - mStencil[3]), Dy2 = Dy * Dy, // * 1/dx
            Dz  = Half * (mStencil[6] - mStencil[5]), Dz2 = Dz * Dz, // * 1/dx
            Dxx = mStencil[2] - 2 * mStencil[0] + mStencil[1], // * 1/dx2
            Dyy = mStencil[4] - 2 * mStencil[0] + mStencil[3], // * 1/dx2
            Dzz = mStencil[6] - 2 * mStencil[0] + mStencil[5], // * 1/dx2
            Dxy = Quarter * (mStencil[10] - mStencil[ 8] + mStencil[7] - mStencil[ 9]), // * 1/dx2
            Dxz = Quarter * (mStencil[14] - mStencil[12] + mStencil[11] - mStencil[13]), // * 1/dx2
            Dyz = Quarter * (mStencil[18] - mStencil[16] + mStencil[15] - mStencil[17]); // * 1/dx2
        alpha = (Dx2*(Dyy+Dzz)+Dy2*(Dxx+Dzz)+Dz2*(Dxx+Dyy)-2*(Dx*(Dy*Dxy+Dz*Dxz)+Dy*Dz*Dyz));
        beta  = std::sqrt(Dx2 + Dy2 + Dz2); // * 1/dx
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
    const ValueType mInv2Dx, mInvDx2;
}; // class CurvatureStencil


//////////////////////////////////////////////////////////////////////


/// @brief Dense stencil of a given width
template<typename GridType>
class DenseStencil: public BaseStencil<GridType, DenseStencil<GridType> >
{
public:
    typedef BaseStencil<GridType, DenseStencil<GridType> > BaseType;
    typedef typename GridType::ValueType                   ValueType;

    DenseStencil(const GridType& grid, int halfWidth) :
        BaseType(grid, /*size=*/math::Pow3(2 * halfWidth + 1)),
        mHalfWidth(halfWidth)
    {
        //assert(halfWidth>0);//should this be allowed?
    }

private:
    /// Initialize the stencil buffer centered at (x, y, z).
    inline void init(const Coord& ijk)
    {
        for (int n=0, i=ijk[0]-mHalfWidth, ie = ijk[0]+mHalfWidth; i <= ie; ++i) {
            Coord sample_ijk(i,0,0);
            for (int j = ijk[1]-mHalfWidth, je = ijk[1]+mHalfWidth; j <= je; ++j) {
                sample_ijk.setY(j);
                for (int k = ijk[2]-mHalfWidth, ke = ijk[2] + mHalfWidth; k <= ke; ++k) {
                    sample_ijk.setZ(k);
                    mStencil[n++] = mCache.getValue(sample_ijk);
                }
            }
        }
    }

    template<typename, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mCache;
    using BaseType::mStencil;
    const int mHalfWidth;
};


} // end math namespace
} // namespace OPENVDB_VERSION_NAME
} // end openvdb namespace

#endif // OPENVDB_MATH_STENCILS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
