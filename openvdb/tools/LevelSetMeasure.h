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
///
/// @file LevelSetMeasure.h

#ifndef OPENVDB_TOOLS_LEVELSETMEASURE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETMEASURE_HAS_BEEN_INCLUDED

#include <boost/math/constants/constants.hpp>//for Pi
#include <openvdb/math/Math.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {
    
/// @brief Smeared-out and continuous Dirac Delta function.
template<typename RealT>    
class DiracDelta    
{
public:
    DiracDelta(RealT eps) : mC(0.5/eps), mD(2*boost::math::constants::pi<RealT>()*mC), mE(eps) {}
    inline RealT operator()(RealT phi) const { return math::Abs(phi) > mE ? 0 : mC*(1+cos(mD*phi)); }
private:
    const RealT mC, mD, mE;
};


/// @brief Multi-threaded computation of surface area, volume and
/// average mean-curvature for narrow band level sets.
template<typename GridT,
         typename InterruptT = util::NullInterrupter>
class LevelSetMeasure
{
public:
    typedef GridT                                GridType;
    typedef typename GridType::TreeType          TreeType;
    typedef typename TreeType::ValueType         ValueType;
    typedef typename tree::LeafManager<const TreeType> ManagerType;
    typedef typename ManagerType::LeafRange      RangeType;
    
    BOOST_STATIC_ASSERT(boost::is_floating_point<ValueType>::value);

    /// @brief Main constructor from a grid
    /// @param grid The level set to be measured.
    /// @param interrupt Optional interrupter.
    /// @throw RuntimeError if the grid is not a level set.
    LevelSetMeasure(const GridType& grid, InterruptT* interrupt = NULL);

    LevelSetMeasure(ManagerType& leafs, Real Dx, InterruptT* interrupt);
    
    /// @brief Shallow copy constructor called by tbb::parallel_reduce() threads.
    /// @param other The other LevelSetMeasure from which to copy.
    /// @param dummy Dummy argument required by tbb.
    /// @warning Never call this method directly.
    LevelSetMeasure(const LevelSetMeasure& other, tbb::split dummy);

    /// @brief Re-initialize using the specified grid.
    void reinit(const GridType& grid);

    /// @brief Re-initialize using the specified LeafManager and voxelSize.
    void reinit(ManagerType& leafs, Real dx);
    
    /// @brief Destructor
    ~LevelSetMeasure() {}

     /// @return the grain-size used for multi-threading
    int  getGrainSize() const { return mGrainSize; }
    
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(int grainsize) { mGrainSize = grainsize; }

    /// @brief Compute the surface area and volume of the level
    /// set. Use the last argument to specify the result in world or
    /// voxel units.
    /// @note This method is faster (about 3x) then the measure method
    /// below that also computes the average mean-curvature.
    void measure(Real& area, Real& volume, bool useWorldUnits = true);

    /// @brief Compute the surface area, volume, and average
    /// mean-curvatue of the level set. Use the last argument to
    /// specify the result in world or voxel units.
    /// @note This method is slower (about 3x) then the measure method
    /// above that only computes the area and volume.
    void measure(Real& area, Real& volume, Real& avgMeanCurvature, bool useWorldUnits = true);

    /// @brief Used internally by tbb::parallel_reduce().
    /// @param range The range over which to perform multi-threading.
    /// @warning Never call this method directly!
    void operator()(const RangeType& range)
    {
        if (mTask) mTask(const_cast<LevelSetMeasure*>(this), range);
        else OPENVDB_THROW(ValueError, "task is undefined");
    }

    /// @brief Used internally by tbb::parallel_reduce().
    /// @warning Never call this method directly!
    void join(const LevelSetMeasure& other);

private:
    typedef typename GridT::ConstAccessor       AccT;
    typedef typename TreeType::LeafNodeType     LeafT;
    typedef typename LeafT::ValueOnCIter        VoxelCIterT;
    typedef typename ManagerType::BufferType    BufferT;
    typedef typename RangeType::Iterator        LeafIterT;
    
    AccT         mAcc;
    ManagerType* mLeafs;
    InterruptT*  mInterrupter;
    Real         mDx, mArea, mVol, mCurv;
    typename boost::function<void (LevelSetMeasure*, const RangeType&)> mTask;
    int          mGrainSize;

    // @brief Return false if the process was interrupted
    bool checkInterrupter();
        
    // Private cook method calling tbb::parallel_reduce
    void cook();
   
    // Private methods called by tbb::parallel_reduce threads
    void measure2( const RangeType& );

    // Private methods called by tbb::parallel_reduce threads
    void measure3( const RangeType& );

}; // end of LevelSetMeasure class


template<typename GridT, typename InterruptT>
LevelSetMeasure<GridT, InterruptT>::LevelSetMeasure(const GridT& grid, InterruptT* interrupt)
    : mAcc(grid.tree())
    , mLeafs(NULL)
    , mInterrupter(interrupt)
    , mDx(grid.voxelSize()[0])
    , mArea(0)
    , mVol(0)
    , mCurv(0)
    , mTask(0)
    , mGrainSize(1)
{
    if ( !grid.hasUniformVoxels() ) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for the LevelSetMeasure to function");
    }
    if ( grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
                      "LevelSetMeasure only supports level sets!\n"
                      "However, only level sets are guaranteed to work!\n"
                      "Hint: Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
    }
}

template<typename GridT, typename InterruptT>
LevelSetMeasure<GridT, InterruptT>::LevelSetMeasure(ManagerType& leafs, Real dx, InterruptT* interrupt)
    : mAcc(leafs.tree())
    , mLeafs(&leafs)
    , mInterrupter(interrupt)
    , mDx(dx)
    , mArea(0)
    , mVol(0)
    , mCurv(0)
    , mTask(0)
    , mGrainSize(1)
{
}
  

template<typename GridT, typename InterruptT>
LevelSetMeasure<GridT, InterruptT>::LevelSetMeasure(const LevelSetMeasure& other, tbb::split)
    : mAcc(other.mAcc)
    , mLeafs(other.mLeafs)
    , mInterrupter(other.mInterrupter)
    , mDx(other.mDx)
    , mArea(0)
    , mVol(0)
    , mCurv(0)
    , mTask(other.mTask)
    , mGrainSize(other.mGrainSize)
{
}

template<typename GridT, typename InterruptT>
void LevelSetMeasure<GridT, InterruptT>::reinit(const GridT& grid)
{
    if ( !grid.hasUniformVoxels() ) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for the LevelSetMeasure to function");
    }
    if ( grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
                      "LevelSetMeasure only supports level sets!\n"
                      "However, only level sets are guaranteed to work!\n"
                      "Hint: Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
    }
    mLeafs = NULL;
    mAcc = grid.getConstAccessor();
    mDx = grid.voxelSize()[0];
}

template<typename GridT, typename InterruptT>
void LevelSetMeasure<GridT, InterruptT>::reinit(ManagerType& leafs, Real dx)
{
    mLeafs = &leafs;
    mAcc = AccT(leafs.tree());
    mDx = dx;
}  

template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::join(const LevelSetMeasure& other)
{
    mArea += other.mArea;
    mVol  += other.mVol;
    mCurv += other.mCurv;
}

////////////////////////////////////////

template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::measure(Real& area, Real& volume, bool useWorldUnits)
{
    if (mInterrupter) mInterrupter->start("Measuring level set");
    mTask = boost::bind(&LevelSetMeasure::measure2, _1, _2);
    mArea = mVol = 0;
    this->cook();
    if (mInterrupter) mInterrupter->end();
    const Real dx = useWorldUnits ? mDx : 1.0;
    area   = dx * dx * mArea;
    volume = dx * dx * dx * mVol / 3.0;
}

template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::measure(Real& area, Real& volume, Real& avgMeanCurvature,
                                            bool useWorldUnits)
{
    if (mInterrupter) mInterrupter->start("Measuring level set");
    mTask = boost::bind(&LevelSetMeasure::measure3, _1, _2);
    mArea = mVol = mCurv = 0;
    this->cook();
    if (mInterrupter) mInterrupter->end();
    const Real dx = useWorldUnits ? mDx : 1.0;
    area   = dx * dx * mArea;
    volume = dx * dx * dx * mVol / 3.0;
    avgMeanCurvature = mCurv / area * dx;
}

///////////////////////// PRIVATE METHODS //////////////////////

template<typename GridT, typename InterruptT>
inline bool
LevelSetMeasure<GridT, InterruptT>::checkInterrupter()
{
    if (util::wasInterrupted(mInterrupter)) {
        tbb::task::self().cancel_group_execution();
        return false;
    }
    return true;
}

template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::cook()
{
    const bool newLeafs = mLeafs == NULL;
    if (newLeafs) mLeafs = new ManagerType(mAcc.tree());
    
    if (mGrainSize>0) {
        tbb::parallel_reduce(mLeafs->leafRange(mGrainSize), *this);
    } else {
        (*this)(mLeafs->leafRange());
    }

    if (newLeafs) {
        delete mLeafs;
        mLeafs = NULL;
    }
}
 
template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::measure2(const RangeType& range)
{
    typedef math::Vec3<ValueType> Vec3T;
    typedef math::ISGradient<math::CD_2ND> Grad;
    this->checkInterrupter();
    const Real invDx = 1.0/mDx;
    const DiracDelta<Real> DD(1.5);
    for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
        for (VoxelCIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const ValueType dd = DD(invDx * (*voxelIter));
            if (!math::isZero(dd)) {
                const Coord ijk =  voxelIter.getCoord();
                const Vec3T G = Grad::result(mAcc, ijk)*invDx;
                mArea += dd * G.dot(G);
                mVol  += dd * G.dot(ijk.asVec3d());
            }
        }
    }
}
 
template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::measure3(const RangeType& range)
{
    typedef math::Vec3<ValueType> Vec3T;
    typedef math::ISGradient<math::CD_2ND> Grad;
    typedef math::ISMeanCurvature<math::CD_SECOND, math::CD_2ND> Curv; 
    this->checkInterrupter();
    const Real invDx = 1.0/mDx;
    const DiracDelta<Real> DD(1.5);
    ValueType alpha, beta;
    for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
        for (VoxelCIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const ValueType dd = DD(invDx * (*voxelIter));
            if (!math::isZero(dd)) {
                const Coord ijk =  voxelIter.getCoord();
                const Vec3T G = Grad::result(mAcc, ijk)*invDx;
                const ValueType dA = dd * G.dot(G);
                mArea += dA;
                mVol  += dd * G.dot(ijk.asVec3d());
                Curv::result(mAcc, ijk, alpha, beta);
                mCurv += dA * alpha /(2*math::Pow2(beta))*invDx;
            }
        }
    }
}    
     
} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETMEASURE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
