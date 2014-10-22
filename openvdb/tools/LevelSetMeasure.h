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
//
/// @author Ken Museth
///
/// @file LevelSetMeasure.h

#ifndef OPENVDB_TOOLS_LEVELSETMEASURE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETMEASURE_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/math/constants/constants.hpp>//for Pi
#include <openvdb/math/Math.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/math/FiniteDifference.h>
#include <openvdb/math/Operators.h>
#include <openvdb/util/NullInterrupter.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Return the surface area of a narrow-band level set.
///
/// @param grid          a scalar, floating-point grid with one or more disjoint,
///                      closed isosurfaces at the given @a isovalue
/// @param useWorldSpace if true the area is computed in
///                      world space units, else in voxel units.
///
/// @throw TypeError if @a grid is not scalar or not floating-point or not a level set.
template<class GridType>
inline Real
levelSetArea(const GridType& grid, bool useWorldSpace = true);

/// @brief Return the volume of a narrow-band level set surface.
///
/// @param grid          a scalar, floating-point grid with one or more disjoint,
///                      closed isosurfaces at the given @a isovalue
/// @param useWorldSpace if true the volume is computed in
///                      world space units, else in voxel units.
///
/// @throw TypeError if @a grid is not scalar or not floating-point or not a level set.
template<class GridType>
inline Real
levelSetVolume(const GridType& grid, bool useWorldSpace = true);

/// @brief Compute the surface area and volume of a narrow-band level set.
///
/// @param grid          a scalar, floating-point grid with one or more disjoint,
///                      closed isosurfaces at the given @a isovalue
/// @param area          surface area of the level set
/// @param volume        volume of the level set surface
/// @param useWorldSpace if true the area and volume are computed in
///                      world space units, else in voxel units.
///
/// @throw TypeError if @a grid is not scalar or not floating-point or not a level set.
template<class GridType>
inline void
levelSetMeasure(const GridType& grid, Real& area, Real& volume, bool useWorldSpace = true);

/// @brief Compute the surface area and volume of a narrow-band level set.
///
/// @param grid          a scalar, floating-point grid with one or more disjoint,
///                      closed isosurfaces at the given @a isovalue
/// @param area          surface area of the level set
/// @param volume        volume of the level set surface
/// @param avgCurvature  average mean curvature of the level set surface
/// @param useWorldSpace if true the area, volume and curvature are computed in
///                      world space units, else in voxel units.
///
/// @throw TypeError if @a grid is not scalar or not floating-point or not a level set.
template<class GridType>
inline void
levelSetMeasure(const GridType& grid, Real& area, Real& volume, Real& avgCurvature,
                bool useWorldSpace = true);

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
///
/// @details To reduce the risk of round-off errors (primarily due to
/// catastrophic cancellation) and guarantee determinism during
/// multi-threading this class is implemented using parallel_for, and
/// delayed reduction of a sorted list.
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
    void operator()(const RangeType& range) const
    {
        if (mTask) mTask(const_cast<LevelSetMeasure*>(this), range);
        else OPENVDB_THROW(ValueError, "task is undefined");
    }

private:
    typedef typename GridT::ConstAccessor       AccT;
    typedef typename TreeType::LeafNodeType     LeafT;
    typedef typename LeafT::ValueOnCIter        VoxelCIterT;
    typedef typename ManagerType::BufferType    BufferT;
    typedef typename RangeType::Iterator        LeafIterT;

    AccT         mAcc;
    ManagerType* mLeafs;
    InterruptT*  mInterrupter;
    double       mDx;
    double*      mArray;
    typename boost::function<void (LevelSetMeasure*, const RangeType&)> mTask;
    int          mGrainSize;

    // @brief Return false if the process was interrupted
    bool checkInterrupter();

    // Private methods called by tbb::parallel_reduce threads
    void measure2( const RangeType& );

    // Private methods called by tbb::parallel_reduce threads
    void measure3( const RangeType& );

    inline double reduce(double* first, double scale)
    {
        double* last = first + mLeafs->leafCount();
        tbb::parallel_sort(first, last);//reduces catastrophic cancellation
        Real sum = 0.0;
        while(first != last) sum += *first++;
        return scale * sum;
    }

}; // end of LevelSetMeasure class


template<typename GridT, typename InterruptT>
inline
LevelSetMeasure<GridT, InterruptT>::LevelSetMeasure(const GridType& grid, InterruptT* interrupt)
    : mAcc(grid.tree())
    , mLeafs(NULL)
    , mInterrupter(interrupt)
    , mDx(grid.voxelSize()[0])
    , mArray(NULL)
    , mTask(0)
    , mGrainSize(1)
{
    if (!grid.hasUniformVoxels()) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for the LevelSetMeasure to function");
    }
    if (grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
            "LevelSetMeasure only supports level sets;"
            " try setting the grid class to \"level set\"");
    }
}


template<typename GridT, typename InterruptT>
inline
LevelSetMeasure<GridT, InterruptT>::LevelSetMeasure(
    ManagerType& leafs, Real dx, InterruptT* interrupt)
    : mAcc(leafs.tree())
    , mLeafs(&leafs)
    , mInterrupter(interrupt)
    , mDx(dx)
    , mArray(NULL)
    , mTask(0)
    , mGrainSize(1)
{
}

template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::reinit(const GridType& grid)
{
    if (!grid.hasUniformVoxels()) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for the LevelSetMeasure to function");
    }
    if (grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
            "LevelSetMeasure only supports level sets;"
            " try setting the grid class to \"level set\"");
    }
    mLeafs = NULL;
    mAcc = grid.getConstAccessor();
    mDx = grid.voxelSize()[0];
}


template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::reinit(ManagerType& leafs, Real dx)
{
    mLeafs = &leafs;
    mAcc = AccT(leafs.tree());
    mDx = dx;
}

////////////////////////////////////////


template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::measure(Real& area, Real& volume, bool useWorldUnits)
{
    if (mInterrupter) mInterrupter->start("Measuring level set");
    mTask = boost::bind(&LevelSetMeasure::measure2, _1, _2);

    const bool newLeafs = mLeafs == NULL;
    if (newLeafs) mLeafs = new ManagerType(mAcc.tree());
    const size_t leafCount = mLeafs->leafCount();
    if (leafCount == 0) {
        area = volume = 0;
        return;
    }
    mArray = new double[2*leafCount];

    if (mGrainSize>0) {
        tbb::parallel_for(mLeafs->leafRange(mGrainSize), *this);
    } else {
        (*this)(mLeafs->leafRange());
    }

    const double dx = useWorldUnits ? mDx : 1.0;
    area = this->reduce(mArray, math::Pow2(dx));
    volume = this->reduce(mArray + leafCount, math::Pow3(dx) / 3.0);

    if (newLeafs) {
        delete mLeafs;
        mLeafs = NULL;
    }
    delete [] mArray;

    if (mInterrupter) mInterrupter->end();
}


template<typename GridT, typename InterruptT>
inline void
LevelSetMeasure<GridT, InterruptT>::measure(Real& area, Real& volume, Real& avgMeanCurvature,
                                            bool useWorldUnits)
{
    if (mInterrupter) mInterrupter->start("Measuring level set");
    mTask = boost::bind(&LevelSetMeasure::measure3, _1, _2);

    const bool newLeafs = mLeafs == NULL;
    if (newLeafs) mLeafs = new ManagerType(mAcc.tree());
    const size_t leafCount = mLeafs->leafCount();
    if (leafCount == 0) {
        area = volume = avgMeanCurvature = 0;
        return;
    }
    mArray = new double[3*leafCount];

    if (mGrainSize>0) {
        tbb::parallel_for(mLeafs->leafRange(mGrainSize), *this);
    } else {
        (*this)(mLeafs->leafRange());
    }

    const double dx = useWorldUnits ? mDx : 1.0;
    area = this->reduce(mArray, math::Pow2(dx));
    volume = this->reduce(mArray + leafCount, math::Pow3(dx) / 3.0);
    avgMeanCurvature = this->reduce(mArray + 2*leafCount, dx/area);

    if (newLeafs) {
        delete mLeafs;
        mLeafs = NULL;
    }
    delete [] mArray;

    if (mInterrupter) mInterrupter->end();
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
LevelSetMeasure<GridT, InterruptT>::measure2(const RangeType& range)
{
    typedef math::Vec3<ValueType> Vec3T;
    typedef math::ISGradient<math::CD_2ND> Grad;
    this->checkInterrupter();
    const Real invDx = 1.0/mDx;
    const DiracDelta<Real> DD(1.5);
    const size_t leafCount = mLeafs->leafCount();
    for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
        Real sumA = 0, sumV = 0;//reduce risk of catastrophic cancellation
        for (VoxelCIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const Real dd = DD(invDx * (*voxelIter));
            if (dd > 0.0) {
                const Coord p = voxelIter.getCoord();
                const Vec3T g = invDx*Grad::result(mAcc, p);//voxel units
                sumA += dd * g.dot(g);
                sumV += dd * (g[0]*p[0]+g[1]*p[1]+g[2]*p[2]);
            }
        }
        double* v = mArray + leafIter.pos();
        *v = sumA;
        v += leafCount;
        *v = sumV;
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
    const size_t leafCount = mLeafs->leafCount();
    for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
        Real sumA = 0, sumV = 0, sumC = 0;//reduce risk of catastrophic cancellation
        for (VoxelCIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const Real dd = DD(invDx * (*voxelIter));
            if (dd > 0.0) {
                const Coord p = voxelIter.getCoord();
                const Vec3T g = invDx*Grad::result(mAcc, p);//voxel units
                const Real dA = dd * g.dot(g);
                sumA += dA;
                sumV += dd * (g[0]*p[0]+g[1]*p[1]+g[2]*p[2]);
                Curv::result(mAcc, p, alpha, beta);
                sumC += dA * alpha/(2*math::Pow2(beta))*invDx;
            }
        }
        double* v = mArray + leafIter.pos();
        *v = sumA;
        v += leafCount;
        *v = sumV;
        v += leafCount;
        *v = sumC;
    }
}

////////////////////////////////////////

template<class GridT>
inline typename boost::enable_if<boost::is_floating_point<typename GridT::ValueType>, Real>::type
doLevelSetArea(const GridT& grid, bool useWorldSpace)
{
    Real area, volume;
    LevelSetMeasure<GridT> m(grid);
    m.measure(area, volume, useWorldSpace);
    return area;
}

template<class GridT>
inline typename boost::disable_if<boost::is_floating_point<typename GridT::ValueType>, Real>::type
doLevelSetArea(const GridT&, bool)
{
    OPENVDB_THROW(TypeError,
        "level set area is supported only for scalar, floating-point grids");
}

template<class GridT>
inline Real
levelSetArea(const GridT& grid, bool useWorldSpace)
{
    return doLevelSetArea<GridT>(grid, useWorldSpace);
}

////////////////////////////////////////

template<class GridT>
inline typename boost::enable_if<boost::is_floating_point<typename GridT::ValueType>, Real>::type
doLevelSetVolume(const GridT& grid, bool useWorldSpace)
{
    Real area, volume;
    LevelSetMeasure<GridT> m(grid);
    m.measure(area, volume, useWorldSpace);
    return volume;
}

template<class GridT>
inline typename boost::disable_if<boost::is_floating_point<typename GridT::ValueType>, Real>::type
doLevelSetVolume(const GridT&, bool)
{
    OPENVDB_THROW(TypeError,
        "level set volume is supported only for scalar, floating-point grids");
}

template<class GridT>
inline Real
levelSetVolume(const GridT& grid, bool useWorldSpace)
{
    return doLevelSetVolume<GridT>(grid, useWorldSpace);
}

////////////////////////////////////////

template<class GridT>
inline typename boost::enable_if<boost::is_floating_point<typename GridT::ValueType> >::type
doLevelSetMeasure(const GridT& grid, Real& area, Real& volume, bool useWorldSpace)
{
    LevelSetMeasure<GridT> m(grid);
    m.measure(area, volume, useWorldSpace);
}

template<class GridT>
inline typename boost::disable_if<boost::is_floating_point<typename GridT::ValueType> >::type
doLevelSetMeasure(const GridT&, Real&, Real&, bool)
{
    OPENVDB_THROW(TypeError,
        "level set measure is supported only for scalar, floating-point grids");
}

template<class GridT>
inline void
levelSetMeasure(const GridT& grid, Real& area, Real& volume, bool useWorldSpace)
{
    doLevelSetMeasure<GridT>(grid, area, volume, useWorldSpace);
}

////////////////////////////////////////

template<class GridT>
inline typename boost::enable_if<boost::is_floating_point<typename GridT::ValueType> >::type
doLevelSetMeasure(const GridT& grid, Real& area, Real& volume, Real& avgCurvature,
                  bool useWorldSpace)
{
    LevelSetMeasure<GridT> m(grid);
    m.measure(area, volume, avgCurvature, useWorldSpace);
}

template<class GridT>
inline typename boost::disable_if<boost::is_floating_point<typename GridT::ValueType> >::type
doLevelSetMeasure(const GridT&, Real&, Real&, Real&, bool)
{
    OPENVDB_THROW(TypeError,
        "level set measure is supported only for scalar, floating-point grids");
}

template<class GridT>
inline void
levelSetMeasure(const GridT& grid, Real& area, Real& volume, Real& avgCurvature, bool useWorldSpace)
{
    doLevelSetMeasure<GridT>(grid, area, volume, avgCurvature, useWorldSpace);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETMEASURE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
