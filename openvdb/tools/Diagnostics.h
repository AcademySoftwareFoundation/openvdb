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

#ifndef OPENVDB_TOOLS_DIAGNOSTICS_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_DIAGNOSTICS_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/Operators.h>
#include <openvdb/tree/LeafManager.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <set>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/utility/enable_if.hpp>

/* TODO: Ken Museth

checkLevelSet
1) has level set class type
2) value type is floating point
3) has uniform scale
4) background value is positive and n*dx
5) active values in range between +-background
6) no active tiles
8) abs of inactive values = background
9) norm grad is close to one

checkDensity volume
1) has fog class tag
2) value type is a floating point
3) background = 0
4) all inactive values are zero
5) all active values are 0-1

*/


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


/// @brief  Threaded method to find unique inactive values.
///
/// @param grid         A VDB volume.
/// @param values       List of unique inactive values, returned by this method.
/// @param numValues    Number of values to look for.
/// @return @c false if the @a grid has more than @a numValues inactive values.
template<class GridType>
bool
uniqueInactiveValues(const GridType& grid,
    std::vector<typename GridType::ValueType>& values, size_t numValues);


////////////////////////////////////////////////////////////////////////////////

/// @brief Checks nan values
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckNan
{
    typedef typename VecTraits<typename GridT::ValueType>::ElementType ElementType;
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;

    /// @brief Default constructor
    CheckNan() {}

    /// Return true if the scalar value is nan
    inline bool operator()(const ElementType& v) const { return boost::math::isnan(v); }

    /// @brief This allow for vector values to be checked componentwise
    template <typename T>
    inline typename boost::enable_if_c<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const
    {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;//should unroll
        return false;
    }

    /// @brief Return true if the tile at the iterator location is nan
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the voxel at the iterator location is nan
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const { return "nan"; }

};// CheckNan

/// @brief Checks for infinite values, e.g. 1/0 or -1/0
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckInf
{
    typedef typename VecTraits<typename GridT::ValueType>::ElementType ElementType;
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;

    /// @brief Default constructor
    CheckInf() {}

    /// Return true if the value is infinite
    inline bool operator()(const ElementType& v) const { return boost::math::isinf(v); }

    /// Return true if any of the vector components are infinite.
    template <typename T> inline typename boost::enable_if_c<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const
    {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const { return "infinite"; }
};// CheckInf

/// @brief Checks for both NaN and inf values, i.e. any value that is not finite.
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckFinite
{
    typedef typename VecTraits<typename GridT::ValueType>::ElementType ElementType;
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;

    /// @brief Default constructor
    CheckFinite() {}

    /// Return true if the value is NOT finite, i.e. it's Nan or infinite
    inline bool operator()(const ElementType& v) const { return !boost::math::isfinite(v); }

    /// Return true if any of the vector components are Nan or infinite.
    template <typename T>
    inline typename boost::enable_if_c<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is Nan or infinite.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is Nan or infinite.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const { return "not finite"; }
};// CheckFinite

/// @brief Check that the magnitude of a value, a, is close to a fixed
/// magnitude, b, given a fixed tolerance c. That is | |a| - |b| | <= c
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOffCIter>
struct CheckMagnitude
{
    typedef typename VecTraits<typename GridT::ValueType>::ElementType ElementType;
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;

    /// @brief Default constructor
    CheckMagnitude(const ElementType& a,
                  const ElementType& t = math::Tolerance<ElementType>::value())
        : absVal(math::Abs(a)), tolVal(math::Abs(t))
    {
    }

    /// Return true if the magnitude of the value is not approximatly
    /// equal to totVal.
    inline bool operator()(const ElementType& v) const
    {
        return math::Abs(math::Abs(v) - absVal) > tolVal;
    }

    /// Return true if any of the vector components are infinite.
    template <typename T> inline typename boost::enable_if_c<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const
    {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "not equal to +/-"<<absVal<<" with a tolerance of "<<tolVal;
        return ss.str();
    }

    const ElementType absVal, tolVal;
};// CheckMagnitude
    
/// @brief Checks a value against a range
template <typename GridT,
          bool MinInclusive = true,//is min part of the range?
          bool MaxInclusive = true,//is max part of the range?
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckRange
{
    typedef typename VecTraits<typename GridT::ValueType>::ElementType ElementType;
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;

    // @brief Constructor taking a range to be tested against.
    CheckRange(const ElementType& _min, const ElementType& _max) : minVal(_min), maxVal(_max)
    {
    }

    /// Return true if the value is smaller then min or larger then max.
    inline bool operator()(const ElementType& v) const
    {
        return (MinInclusive ? v<minVal : v<=minVal) ||
               (MaxInclusive ? v>maxVal : v>=maxVal);
    }

    /// Return true if any of the vector components are out of range.
    template <typename T>
    inline typename boost::enable_if_c<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the voxel at the iterator location is out of range.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is out of range.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "outside the range "    << (MinInclusive ? "[" : "]")
           << minVal << "," << maxVal << (MaxInclusive ? "]" : "[");
        return ss.str();
    }

    const ElementType minVal, maxVal;
};// CheckRange

/// @brief Checks a value against a minimum
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckMin
{
    typedef typename VecTraits<typename GridT::ValueType>::ElementType ElementType;
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;

    // @brief Constructor taking a minimum to be tested against.
    CheckMin(const ElementType& _min) : minVal(_min) {}

    /// Return true if the value is smaller then min.
    inline bool operator()(const ElementType& v) const { return v<minVal; }

    /// Return true if any of the vector components are smaller then min.
    template <typename T>
    inline typename boost::enable_if_c<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the voxel at the iterator location is smaller then min.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is smaller then min.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "smaller then "<<minVal;
        return ss.str();
    }

    const ElementType minVal;
};// CheckMin

/// @brief Checks a value against a maximum
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckMax
{
    typedef typename VecTraits<typename GridT::ValueType>::ElementType ElementType;
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;

    /// @brief Constructor taking a maximum to be tested against.
    CheckMax(const ElementType& _max) : maxVal(_max) {}

    /// Return true if the value is larger then max.
    inline bool operator()(const ElementType& v) const { return v>maxVal; }

    /// Return true if any of the vector components are larger then max.
    template <typename T>
    inline typename boost::enable_if_c<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is larger then max.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the voxel at the iterator location is larger then max.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "larger then "<<maxVal;
        return ss.str();
    }

    const ElementType maxVal;
};// CheckMax

/// @brief Checks the norm of the gradient against a range
template<typename GridT,
         typename TreeIterT = typename GridT::ValueOnCIter,
         math::BiasedGradientScheme GradScheme = math::FIRST_BIAS>//math::WENO5_BIAS>
struct CheckNormGrad
{
    typedef typename GridT::ValueType ValueType;
    BOOST_STATIC_ASSERT(boost::is_floating_point<ValueType>::value);
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;
    typedef typename GridT::ConstAccessor AccT;

    /// @brief Constructor taking a grid and a range to be tested against.
    CheckNormGrad(const GridT&  grid, const ValueType& _min, const ValueType& _max)
        : acc(grid.getConstAccessor())
        , invdx2(ValueType(1.0/math::Pow2(grid.voxelSize()[0])))
        , minVal(_min)
        , maxVal(_max)
    {
        if ( !grid.hasUniformVoxels() ) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for CheckNormGrad to function");
        }
    }

    CheckNormGrad(const CheckNormGrad& other)
        : acc(other.acc.tree())
        , invdx2(other.invdx2)
        , minVal(other.minVal)
        , maxVal(other.maxVal)
    {
    }

    CheckNormGrad& operator=(const CheckNormGrad& other)
    {
        if (&other != this) {
            acc = AccT(other.acc.tree());
            invdx2 = other.invdx2;
            minVal = other.minVal;
            maxVal = other.maxVal;
        }
        return *this;
    }

    /// Return true if the value is smaller then min or larger then max.
    inline bool operator()(const ValueType& v) const { return v<minVal || v>maxVal; }

    /// @brief Return true if zero is outside the range.
    /// @note We assume that the norm of the gradient of a tile is always zero.
    inline bool operator()(const TreeIterT&) const { return (*this)(ValueType(0)); }

    /// @brief Return true if the norm of the gradient at a voxel
    /// location of the iterator is out of range.
    inline bool operator()(const VoxelIterT &iter) const
      {
          const Coord ijk = iter.getCoord();
          return (*this)(invdx2 * math::ISGradientNormSqrd<GradScheme>::result(acc, ijk));
      }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "outside the range ["<<minVal<<","<<maxVal<<"]";
        return ss.str();
    }

    AccT acc;
    const ValueType invdx2, minVal, maxVal;
};// CheckNormGrad

/// @brief Checks the divergence against a range
template<typename GridT,
         typename TreeIterT = typename GridT::ValueOnCIter,
         math::DScheme DiffScheme = math::CD_2ND>
struct CheckDivergence
{
    typedef typename GridT::ValueType ValueType;
    typedef typename VecTraits<ValueType>::ElementType ElementType;
    BOOST_STATIC_ASSERT(boost::is_floating_point<ElementType>::value);
    typedef TreeIterT TileIterT;
    typedef typename tree::IterTraits<typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>
    ::template NodeConverter<typename GridT::TreeType::LeafNodeType>::Type VoxelIterT;
    typedef typename GridT::ConstAccessor AccT;

    /// @brief Constructor taking a grid and a range to be tested against.
    CheckDivergence(const GridT&  grid,
                    const ValueType& _min,
                    const ValueType& _max)
        : acc(grid.getConstAccessor())
        , invdx(ValueType(1.0/grid.voxelSize()[0]))
        , minVal(_min)
        , maxVal(_max)
    {
        if ( !grid.hasUniformVoxels() ) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for CheckDivergence to function");
        }
    }
    /// Return true if the value is smaller then min or larger then max.
    inline bool operator()(const ElementType& v) const { return v<minVal || v>maxVal; }

    /// @brief Return true if zero is outside the range.
    /// @note We assume that the divergence of a tile is always zero.
    inline bool operator()(const TreeIterT&) const { return (*this)(ElementType(0)); }

    /// @brief Return true if the divergence at a voxel location of
    /// the iterator is out of range.
    inline bool operator()(const VoxelIterT &iter) const
      {
          const Coord ijk = iter.getCoord();
          return (*this)(invdx * math::ISDivergence<DiffScheme>::result(acc, ijk));
      }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "outside the range ["<<minVal<<","<<maxVal<<"]";
        return ss.str();
    }

    AccT acc;
    const ValueType invdx, minVal, maxVal;
};// CheckDivergence

/// @brief Performs multithreaded diagnostics of a grid
/// @note More documentation will be added soon!
template <typename GridT>
class Diagnose
{
  public:
    typedef typename GridT::template ValueConverter<bool>::Type  MaskType;

    Diagnose(const GridT& grid) : mGrid(&grid), mMask(new MaskType()), mCount(0)
      {
          mMask->setTransform(grid.transformPtr()->copy());
      }

    template <typename CheckT>
    std::string check(const CheckT& check,
                      bool updateMask = false,
                      bool checkVoxels = true,
                      bool checkTiles = true,
                      bool checkBackground = true)
    {
        typename MaskType::TreeType* mask = updateMask ? &(mMask->tree()) : NULL;
        CheckValues<CheckT> cc(mask, mGrid, check);
        std::ostringstream ss;
        if (checkBackground) ss << cc.checkBackground();
        if (checkTiles)      ss << cc.checkTiles();
        if (checkVoxels)     ss << cc.checkVoxels();
        mCount += cc.mCount;
        return ss.str();
    }

    /// @brief Return a boolean mask of all the values
    /// (i.e. tiles and/or voxels) that have failed one or
    /// more checks.
    typename MaskType::ConstPtr mask() const { return mMask; }

    /// @brief Return the number of values (i.e. background, tiles or
    /// voxels) that have failed one or more checks.
    Index64 valueCount() const { return mMask->activeVoxelCount(); }

    /// @brief Return total number of failed checks
    /// @note If only one check was performed and the mask was updated
    /// failureCount equals valueCount.
    Index64 failureCount() const { return mCount; }

private:
    const GridT*           mGrid;
    typename MaskType::Ptr mMask;
    Index64                mCount;

    /// @brief Private class that performs the multithreaded checks
    template <typename CheckT>
    struct CheckValues
    {
        typedef typename MaskType::TreeType MaskT;
        typedef typename GridT::TreeType::LeafNodeType LeafT;
        typedef typename tree::LeafManager<const typename GridT::TreeType> LeafManagerT;
        const bool      mOwnsMask;
        MaskT*          mMask;
        const GridT*    mGrid;
        const CheckT    mCheck;
        Index64         mCount;

        CheckValues(MaskT* mask, const GridT* grid, const CheckT& check)
            : mOwnsMask(false)
            , mMask(mask)
            , mGrid(grid)
            , mCheck(check)
            , mCount(0)
        {
        }
        CheckValues(CheckValues& other, tbb::split)
            : mOwnsMask(true)
            , mMask(other.mMask ? new MaskT() : NULL)
            , mGrid(other.mGrid)
            , mCheck(other.mCheck)
            , mCount(0)
        {
        }
        ~CheckValues() { if (mOwnsMask) delete mMask; }

        std::string checkBackground()
        {
            std::ostringstream ss;
            if (mCheck(mGrid->background())) {
                ++mCount;
                ss << "Background is " + mCheck.str() << std::endl;
            }
            return ss.str();
        }

        std::string checkTiles()
        {
            std::ostringstream ss;
            const Index64 n = mCount;
            typename CheckT::TileIterT i(mGrid->tree());
            for (i.setMaxDepth(GridT::TreeType::RootNodeType::LEVEL - 1); i; ++i) {
                if (mCheck(i)) {
                    ++mCount;
                    if (mMask) mMask->fill(i.getBoundingBox(), true, true);
                }
            }
            if (const Index64 m = mCount - n) {
                ss << m << (m==1?" tile is ":" tiles are ") + mCheck.str() << std::endl;
            }
            return ss.str();
        }

        std::string checkVoxels()
          {
              std::ostringstream ss;
              LeafManagerT leafs(mGrid->tree());
              const Index64 n = mCount;
              tbb::parallel_reduce(leafs.leafRange(), *this);
              if (const Index64 m = mCount - n) {
                  ss << m << (m==1?" voxel is ":" voxels are ") + mCheck.str() << std::endl;
              }
              return ss.str();
          }

        void operator()(const typename LeafManagerT::LeafRange& r)
          {
              typedef typename CheckT::VoxelIterT VoxelIterT;
              if (mMask) {
                  for (typename LeafManagerT::LeafRange::Iterator i=r.begin(); i; ++i) {
                      typename MaskT::LeafNodeType* maskLeaf = NULL;
                      for (VoxelIterT j = tree::IterTraits<LeafT, VoxelIterT>::begin(*i); j; ++j) {
                          if (mCheck(j)) {
                              ++mCount;
                              if (maskLeaf == NULL) maskLeaf = mMask->touchLeaf(j.getCoord());
                              maskLeaf->setValueOn(j.pos(), true);
                          }
                      }
                  }
              } else {
                  for (typename LeafManagerT::LeafRange::Iterator i=r.begin(); i; ++i) {
                      for (VoxelIterT j = tree::IterTraits<LeafT, VoxelIterT>::begin(*i); j; ++j) {
                          if (mCheck(j)) ++mCount;
                      }
                  }
              }
          }
        void join(const CheckValues& other)
        {
            if (mMask) mMask->merge(*(other.mMask), openvdb::MERGE_ACTIVE_STATES_AND_NODES);
            mCount += other.mCount;
        }
    };//End of private class CheckValues

};// End of public class Diagnose


////////////////////////////////////////////////////////////////////////////////

// Internal utility objects and implementation details


namespace diagnostics_internal {


template<typename TreeType>
class InactiveVoxelValues
{
public:
    typedef tree::LeafManager<TreeType> LeafArray;
    typedef typename TreeType::ValueType ValueType;
    typedef std::set<ValueType> SetType;

    InactiveVoxelValues(LeafArray&, size_t numValues);

    void runParallel();
    void runSerial();

    void getInactiveValues(SetType&) const;

    inline InactiveVoxelValues(const InactiveVoxelValues<TreeType>&, tbb::split);
    inline void operator()(const tbb::blocked_range<size_t>&);
    inline void join(const InactiveVoxelValues<TreeType>&);

private:
    LeafArray& mLeafArray;
    SetType mInactiveValues;
    size_t mNumValues;
};

template<typename TreeType>
InactiveVoxelValues<TreeType>::InactiveVoxelValues(LeafArray& leafs, size_t numValues)
    : mLeafArray(leafs)
    , mInactiveValues()
    , mNumValues(numValues)
{
}

template <typename TreeType>
inline
InactiveVoxelValues<TreeType>::InactiveVoxelValues(
    const InactiveVoxelValues<TreeType>& rhs, tbb::split)
    : mLeafArray(rhs.mLeafArray)
    , mInactiveValues()
    , mNumValues(rhs.mNumValues)
{
}

template<typename TreeType>
void
InactiveVoxelValues<TreeType>::runParallel()
{
    tbb::parallel_reduce(mLeafArray.getRange(), *this);
}


template<typename TreeType>
void
InactiveVoxelValues<TreeType>::runSerial()
{
    (*this)(mLeafArray.getRange());
}


template<typename TreeType>
inline void
InactiveVoxelValues<TreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    typename TreeType::LeafNodeType::ValueOffCIter iter;

    for (size_t n = range.begin(); n < range.end() && !tbb::task::self().is_cancelled(); ++n) {
        for (iter = mLeafArray.leaf(n).cbeginValueOff(); iter; ++iter) {
            mInactiveValues.insert(iter.getValue());
        }

        if (mInactiveValues.size() > mNumValues) {
            tbb::task::self().cancel_group_execution();
        }
    }
}

template<typename TreeType>
inline void
InactiveVoxelValues<TreeType>::join(const InactiveVoxelValues<TreeType>& rhs)
{
    mInactiveValues.insert(rhs.mInactiveValues.begin(), rhs.mInactiveValues.end());
}

template<typename TreeType>
inline void
InactiveVoxelValues<TreeType>::getInactiveValues(SetType& values) const
{
    values.insert(mInactiveValues.begin(), mInactiveValues.end());
}


////////////////////////////////////////


template<typename TreeType>
class InactiveTileValues
{
public:
    typedef tree::IteratorRange<typename TreeType::ValueOffCIter> IterRange;
    typedef typename TreeType::ValueType ValueType;
    typedef std::set<ValueType> SetType;

    InactiveTileValues(size_t numValues);

    void runParallel(IterRange&);
    void runSerial(IterRange&);

    void getInactiveValues(SetType&) const;

    inline InactiveTileValues(const InactiveTileValues<TreeType>&, tbb::split);
    inline void operator()(IterRange&);
    inline void join(const InactiveTileValues<TreeType>&);

private:
    SetType mInactiveValues;
    size_t mNumValues;
};


template<typename TreeType>
InactiveTileValues<TreeType>::InactiveTileValues(size_t numValues)
    : mInactiveValues()
    , mNumValues(numValues)
{
}

template <typename TreeType>
inline
InactiveTileValues<TreeType>::InactiveTileValues(
    const InactiveTileValues<TreeType>& rhs, tbb::split)
    : mInactiveValues()
    , mNumValues(rhs.mNumValues)
{
}

template<typename TreeType>
void
InactiveTileValues<TreeType>::runParallel(IterRange& range)
{
    tbb::parallel_reduce(range, *this);
}


template<typename TreeType>
void
InactiveTileValues<TreeType>::runSerial(IterRange& range)
{
    (*this)(range);
}


template<typename TreeType>
inline void
InactiveTileValues<TreeType>::operator()(IterRange& range)
{
    for (; range && !tbb::task::self().is_cancelled(); ++range) {
        typename TreeType::ValueOffCIter iter = range.iterator();
        for (; iter; ++iter) {
            mInactiveValues.insert(iter.getValue());
        }

        if (mInactiveValues.size() > mNumValues) {
            tbb::task::self().cancel_group_execution();
        }
    }
}

template<typename TreeType>
inline void
InactiveTileValues<TreeType>::join(const InactiveTileValues<TreeType>& rhs)
{
    mInactiveValues.insert(rhs.mInactiveValues.begin(), rhs.mInactiveValues.end());
}

template<typename TreeType>
inline void
InactiveTileValues<TreeType>::getInactiveValues(SetType& values) const
{
    values.insert(mInactiveValues.begin(), mInactiveValues.end());
}

} // namespace diagnostics_internal


////////////////////////////////////////


template<class GridType>
bool
uniqueInactiveValues(const GridType& grid,
    std::vector<typename GridType::ValueType>& values, size_t numValues)
{

    typedef typename GridType::TreeType TreeType;
    typedef typename GridType::ValueType ValueType;
    typedef std::set<ValueType> SetType;

    SetType uniqueValues;

    { // Check inactive voxels
        TreeType& tree = const_cast<TreeType&>(grid.tree());
        tree::LeafManager<TreeType> leafs(tree);
        diagnostics_internal::InactiveVoxelValues<TreeType> voxelOp(leafs, numValues);
        voxelOp.runParallel();
        voxelOp.getInactiveValues(uniqueValues);
    }

    // Check inactive tiles
    if (uniqueValues.size() <= numValues) {
        typename TreeType::ValueOffCIter iter(grid.tree());
        iter.setMaxDepth(TreeType::ValueAllIter::LEAF_DEPTH - 1);
        diagnostics_internal::InactiveTileValues<TreeType> tileOp(numValues);

        tree::IteratorRange<typename TreeType::ValueOffCIter> range(iter);
        tileOp.runParallel(range);

        tileOp.getInactiveValues(uniqueValues);
    }

    values.clear();
    values.reserve(uniqueValues.size());

    typename SetType::iterator it = uniqueValues.begin();
    for ( ; it != uniqueValues.end(); ++it) {
        values.push_back(*it);
    }

    return values.size() <= numValues;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_DIAGNOSTICS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
