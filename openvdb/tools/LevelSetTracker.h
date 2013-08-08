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
/// @file LevelSetTracker.h
///
/// @brief Performs multi-threaded interface tracking of narrow band
/// level sets. This is the building-block for most level set
/// computations that involve dynamic topology, e.g. advection.

#ifndef OPENVDB_TOOLS_LEVEL_SET_TRACKER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_TRACKER_HAS_BEEN_INCLUDED

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/FiniteDifference.h>
#include <openvdb/math/Operators.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/math/Transform.h>
#include <openvdb/Grid.h>
#include <openvdb/util/NullInterrupter.h>
#include "Morphology.h"//for tools::dilateVoxels

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Performs multi-threaded interface tracking of narrow band level sets
template<typename GridT, typename InterruptT = util::NullInterrupter>
class LevelSetTracker
{
public:
    typedef GridT                                GridType;
    typedef typename GridT::TreeType             TreeType;
    typedef typename TreeType::LeafNodeType      LeafType;
    typedef typename TreeType::ValueType         ValueType;
    typedef typename tree::LeafManager<TreeType> LeafManagerType; // leafs + buffers
    typedef typename LeafManagerType::RangeType  RangeType;
    typedef typename LeafManagerType::BufferType BufferType;
    BOOST_STATIC_ASSERT(boost::is_floating_point<ValueType>::value);

    /// Main constructor
    LevelSetTracker(GridT& grid, InterruptT* interrupt = NULL);

    /// Shallow copy constructor called by tbb::parallel_for() threads during filtering
    LevelSetTracker(const LevelSetTracker& other);

    virtual ~LevelSetTracker() { if (mIsMaster) delete mLeafs; }

    /// Iterative normalization, i.e. solving the Eikonal equation
    void normalize();

    /// Track the level set interface, i.e. rebuild and normalize the
    /// narrow band of the level set.
    void track();

    /// Remove voxels that are outside the narrow band. (substep of track)
    void prune();

    /// @return the spatial finite difference scheme
    math::BiasedGradientScheme getSpatialScheme() const { return mSpatialScheme; }
    /// @brief Set the spatial finite difference scheme
    void setSpatialScheme(math::BiasedGradientScheme scheme) { mSpatialScheme = scheme; }

    /// @return the temporal integration scheme
    math::TemporalIntegrationScheme getTemporalScheme() const { return mTemporalScheme; }
    /// @brief Set the spatial finite difference scheme
    void setTemporalScheme(math::TemporalIntegrationScheme scheme) { mTemporalScheme = scheme; }

    /// @return The number of normalizations performed per track or
    /// normalize call.
    int  getNormCount() const { return mNormCount; }
    /// @brief Set the number of normalizations performed per track or
    /// normalize call.
    void setNormCount(int n) { mNormCount = n; }

    /// @return the grain-size used for multi-threading
    int  getGrainSize() const { return mGrainSize; }
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(int grainsize) { mGrainSize = grainsize; }

    ValueType voxelSize() const { return mDx; }

    void startInterrupter(const char* msg);
    void endInterrupter();
    /// @return false if the process was interrupted
    bool checkInterrupter();

    const GridType& grid() const { return *mGrid; }

    LeafManagerType& leafs() { return *mLeafs; }
    const LeafManagerType& leafs() const { return *mLeafs; }

    /// @brief Public functor called by tbb::parallel_for()
    /// @note Never call this method directly
    void operator()(const RangeType& r) const
    {
        if (mTask) mTask(const_cast<LevelSetTracker*>(this), r);
        else OPENVDB_THROW(ValueError, "task is undefined - call track(), etc");
    }
    
private:

    template<math::BiasedGradientScheme      SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    struct Normalizer
    {
        Normalizer(LevelSetTracker& tracker): mTracker(tracker), mTask(0) {}
        void normalize();
        void operator()(const RangeType& r) const {mTask(const_cast<Normalizer*>(this), r);}
        typedef typename boost::function<void (Normalizer*, const RangeType&)> FuncType;
        LevelSetTracker& mTracker;
        FuncType         mTask;
        void cook(int swapBuffer=0);
        void euler1(const RangeType& range, ValueType dt, Index resultBuffer);
        void euler2(const RangeType& range, ValueType dt, ValueType alpha,
                    Index phiBuffer, Index resultBuffer);
    }; // end of protected Normalizer class
    
    typedef typename boost::function<void (LevelSetTracker*, const RangeType&)> FuncType;

    void trim(const RangeType& r);

    template<math::BiasedGradientScheme SpatialScheme>
    void normalize1();

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    void normalize2();
     
    // Throughout the methods below mLeafs is always assumed to contain
    // a list of the current LeafNodes! The auxiliary buffers on the
    // other hand always have to be allocated locally, since some
    // methods need them and others don't!
    GridType*                       mGrid;
    LeafManagerType*                mLeafs;
    InterruptT*                     mInterrupter;
    const ValueType                 mDx;
    math::BiasedGradientScheme      mSpatialScheme;
    math::TemporalIntegrationScheme mTemporalScheme;
    int                             mNormCount;// Number of iteratations of normalization
    int                             mGrainSize;
    FuncType                        mTask;
    const bool                      mIsMaster;

    // disallow copy by assignment
    void operator=(const LevelSetTracker& other) {}

}; // end of LevelSetTracker class

template<typename GridT, typename InterruptT>
LevelSetTracker<GridT, InterruptT>::LevelSetTracker(GridT& grid, InterruptT* interrupt):
    mGrid(&grid),
    mLeafs(new LeafManagerType(grid.tree())),
    mInterrupter(interrupt),
    mDx(grid.voxelSize()[0]),
    mSpatialScheme(math::HJWENO5_BIAS),
    mTemporalScheme(math::TVD_RK1),
    mNormCount(static_cast<int>(LEVEL_SET_HALF_WIDTH)),
    mGrainSize(1),
    mTask(0),
    mIsMaster(true)// N.B.
{
    if ( !grid.hasUniformVoxels() ) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for the LevelSetTracker to function");
    }
    if ( grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
                      "LevelSetTracker only supports level sets!\n"
                      "However, only level sets are guaranteed to work!\n"
                      "Hint: Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
    }
}

template<typename GridT, typename InterruptT>
LevelSetTracker<GridT, InterruptT>::LevelSetTracker(const LevelSetTracker& other):
    mGrid(other.mGrid),
    mLeafs(other.mLeafs),
    mInterrupter(other.mInterrupter),
    mDx(other.mDx),
    mSpatialScheme(other.mSpatialScheme),
    mTemporalScheme(other.mTemporalScheme),
    mNormCount(other.mNormCount),
    mGrainSize(other.mGrainSize),
    mTask(other.mTask),
    mIsMaster(false)// N.B.
{
}

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::prune()
{
    this->startInterrupter("Pruning Level Set");
    // Prune voxels that are too far away from the zero-crossing
    mTask = boost::bind(&LevelSetTracker::trim, _1, _2);
    if (mGrainSize>0) {
        tbb::parallel_for(mLeafs->getRange(mGrainSize), *this);
    } else {
        (*this)(mLeafs->getRange());
    }

    // Remove inactive nodes from tree
    mGrid->tree().pruneLevelSet();

    // The tree topology has changes so rebuild the list of leafs
    mLeafs->rebuildLeafArray();
    this->endInterrupter();
}

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::track()
{
    // Dilate narrow-band (this also rebuilds the leaf array!)
    tools::dilateVoxels(*mLeafs);

    // Compute signed distances in dilated narrow-band
    this->normalize();

    // Remove voxels that are outside the narrow band
    this->prune();
}

template<typename GridT,  typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::startInterrupter(const char* msg)
{
    if (mInterrupter) mInterrupter->start(msg);
}

template<typename GridT,  typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::endInterrupter()
{
    if (mInterrupter) mInterrupter->end();
}

template<typename GridT,  typename InterruptT>
inline bool
LevelSetTracker<GridT, InterruptT>::checkInterrupter()
{
    if (util::wasInterrupted(mInterrupter)) {
        tbb::task::self().cancel_group_execution();
        return false;
    }
    return true;
}

/// Prunes away voxels that have moved outside the narrow band
template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::trim(const RangeType& range)
{
    typedef typename LeafType::ValueOnIter VoxelIterT;
    const_cast<LevelSetTracker*>(this)->checkInterrupter();
    const ValueType gamma = mGrid->background();
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        LeafType &leaf = mLeafs->leaf(n);
        for (VoxelIterT iter = leaf.beginValueOn(); iter; ++iter) {
            const ValueType val = *iter;
            if (val < -gamma)
                leaf.setValueOff(iter.pos(), -gamma);
            else if (val > gamma)
                leaf.setValueOff(iter.pos(),  gamma);
        }
    }
}

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::normalize()
{
    switch (mSpatialScheme) {
    case math::FIRST_BIAS:
        this->normalize1<math::FIRST_BIAS  >(); break;
    case math::SECOND_BIAS:
        this->normalize1<math::SECOND_BIAS >(); break;
    case math::THIRD_BIAS:
        this->normalize1<math::THIRD_BIAS  >(); break;
    case math::WENO5_BIAS:
        this->normalize1<math::WENO5_BIAS  >(); break;
    case math::HJWENO5_BIAS:
        this->normalize1<math::HJWENO5_BIAS>(); break;
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme>
inline void
LevelSetTracker<GridT, InterruptT>::normalize1()
{
    switch (mTemporalScheme) {
    case math::TVD_RK1:
        this->normalize2<SpatialScheme, math::TVD_RK1>(); break;
    case math::TVD_RK2:
        this->normalize2<SpatialScheme, math::TVD_RK2>(); break;
    case math::TVD_RK3:
        this->normalize2<SpatialScheme, math::TVD_RK3>(); break;
    default:
        OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetTracker<GridT, InterruptT>::normalize2()
{
    Normalizer<SpatialScheme, TemporalScheme> tmp(*this);
    tmp.normalize();
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetTracker<GridT,InterruptT>::Normalizer<SpatialScheme, TemporalScheme>::
normalize()
{
    /// Make sure we have enough temporal auxiliary buffers
    mTracker.mLeafs->rebuildAuxBuffers(TemporalScheme == math::TVD_RK3 ? 2 : 1);

    const ValueType dt = (TemporalScheme == math::TVD_RK1 ? ValueType(0.3) :
        TemporalScheme == math::TVD_RK2 ? ValueType(0.9) : ValueType(1.0))
        * ValueType(mTracker.voxelSize());

    for (int n=0, e=mTracker.getNormCount(); n < e; ++n) {

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        switch(TemporalScheme) {//switch is resolved at compile-time
        case math::TVD_RK1:
            //std::cerr << "1";
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(0) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = boost::bind(&Normalizer::euler1, _1, _2, dt, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(1);
            break;
        case math::TVD_RK2:
            //std::cerr << "2";
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = boost::bind(&Normalizer::euler1, _1, _2, dt, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(1);

            // Convex combine explict Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&Normalizer::euler2,
                _1, _2, dt, ValueType(0.5), /*phi=*/1, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)
            this->cook(1);
            break;
        case math::TVD_RK3:
            //std::cerr << "3";
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = boost::bind(&Normalizer::euler1, _1, _2, dt, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(1);

            // Convex combine explict Euler step: t2 = t0 + dt/2
            // Phi_t2(2) = 3/4 * Phi_t0(1) + 1/4 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&Normalizer::euler2,
                _1, _2, dt, ValueType(0.75), /*phi=*/1, /*result=*/2);
            // Cook and swap buffer 0 and 2 such that Phi_t2(0) and Phi_t1(2)
            this->cook(2);

            // Convex combine explict Euler step: t3 = t0 + dt
            // Phi_t3(2) = 1/3 * Phi_t0(1) + 2/3 * (Phi_t2(0) - dt * V.Grad_t2(0)
            mTask = boost::bind(&Normalizer::euler2,
                _1, _2, dt, ValueType(1.0/3.0), /*phi=*/1, /*result=*/2);
            // Cook and swap buffer 0 and 2 such that Phi_t3(0) and Phi_t2(2)
            this->cook(2);
            break;
        default:
            OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    mTracker.mLeafs->removeAuxBuffers();
}

/// Private method to perform the task (serial or threaded) and
/// subsequently swap the leaf buffers.
template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetTracker<GridT,InterruptT>::Normalizer<SpatialScheme, TemporalScheme>::
cook(int swapBuffer)
{
    mTracker.startInterrupter("Normalizing Level Set");

    if (mTracker.getGrainSize()>0) {
        tbb::parallel_for(mTracker.mLeafs->getRange(mTracker.getGrainSize()), *this);
    } else {
        (*this)(mTracker.mLeafs->getRange());
    }

    mTracker.mLeafs->swapLeafBuffer(swapBuffer, mTracker.getGrainSize()==0);

    mTracker.endInterrupter();
}


/// Perform normalization using one of the upwinding schemes
/// This currently supports only forward Euler time integration
/// and is not expected to work well with the higher-order spactial schemes
template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetTracker<GridT,InterruptT>::Normalizer<SpatialScheme, TemporalScheme>::
euler1(const RangeType &range, ValueType dt, Index resultBuffer)
{
    typedef math::BIAS_SCHEME<SpatialScheme>                             Scheme;
    typedef typename Scheme::template ISStencil<GridType>::StencilType   Stencil;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    mTracker.checkInterrupter();
    const ValueType one(1.0), invDx = one/mTracker.voxelSize();
    Stencil stencil(mTracker.grid());
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        BufferType& result = mTracker.mLeafs->getBuffer(n, resultBuffer);
        const LeafType& leaf = mTracker.mLeafs->leaf(n);
        for (VoxelIterT iter = leaf.cbeginValueOn(); iter; ++iter) {
            stencil.moveTo(iter);
            const ValueType normSqGradPhi =
                math::ISGradientNormSqrd<SpatialScheme>::result(stencil);
            const ValueType phi0 = stencil.getValue();
            const ValueType diff = math::Sqrt(normSqGradPhi)*invDx - one;
            const ValueType S = phi0 / (math::Sqrt(math::Pow2(phi0) + normSqGradPhi));
            result.setValue(iter.pos(), phi0 - dt * S * diff);
        }
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetTracker<GridT,InterruptT>::Normalizer<SpatialScheme, TemporalScheme>::
euler2(const RangeType& range, ValueType dt, ValueType alpha, Index phiBuffer, Index resultBuffer)
{
    typedef math::BIAS_SCHEME<SpatialScheme>                             Scheme;
    typedef typename Scheme::template ISStencil<GridType>::StencilType   Stencil;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    mTracker.checkInterrupter();
    const ValueType one(1.0), beta = one - alpha, invDx = one/mTracker.voxelSize();
    Stencil stencil(mTracker.grid());
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        const BufferType& phi = mTracker.mLeafs->getBuffer(n, phiBuffer);
        BufferType& result    = mTracker.mLeafs->getBuffer(n, resultBuffer);
        const LeafType& leaf  = mTracker.mLeafs->leaf(n);
        for (VoxelIterT iter  = leaf.cbeginValueOn(); iter; ++iter) {
            stencil.moveTo(iter);
            const ValueType normSqGradPhi =
                math::ISGradientNormSqrd<SpatialScheme>::result(stencil);
            const ValueType phi0 = stencil.getValue();
            const ValueType diff = math::Sqrt(normSqGradPhi)*invDx - one;
            const ValueType S = phi0 / (math::Sqrt(math::Pow2(phi0) + normSqGradPhi));
            result.setValue(iter.pos(), alpha*phi[iter.pos()] + beta*(phi0 - dt * S * diff));
        }
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_TRACKER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
