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
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include "ChangeBackground.h"// for changeLevelSetBackground
#include "Morphology.h"//for dilateVoxels
#include "Prune.h"// for pruneLevelSet

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
    typedef typename LeafManagerType::LeafRange  LeafRange;
    typedef typename LeafManagerType::BufferType BufferType;
    typedef typename TreeType::template ValueConverter<bool>::Type BoolMaskType;
    BOOST_STATIC_ASSERT(boost::is_floating_point<ValueType>::value);

    /// @brief Main constructor
    /// @throw RuntimeError if the grid is not a level set
    LevelSetTracker(GridT& grid, InterruptT* interrupt = NULL);

    /// @brief Shallow copy constructor called by tbb::parallel_for() threads during filtering
    LevelSetTracker(const LevelSetTracker& other);

    virtual ~LevelSetTracker() { if (mIsMaster) delete mLeafs; }

    /// @brief Iterative normalization, i.e. solving the Eikonal equation
    /// @note The mask it optional and by default it is ignored.
    template <typename MaskType>
    void normalize(const MaskType* mask);

    /// @brief Iterative normalization, i.e. solving the Eikonal equation
    void normalize() { this->normalize<BoolMaskType>(NULL); }

    /// @brief Track the level set interface, i.e. rebuild and normalize the
    /// narrow band of the level set.
    void track();

    /// @brief Remove voxels that are outside the narrow band. (substep of track)
    void prune();

    /// @brief Fast but approximate dilation of the narrow band
    /// @note This method works fine with low-order temporal and spatial schemes.
    void dilate(int iterations = 1);

    /// @brief Erodes the width of the narrow-band and update the background values
    /// @throw ValueError if @a iterations is larger then the current half-width.
    void erode(int iterations = 1);

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

private:

    // Private class to perform multi-threaded trimming of
    // voxels that are too far away from the zero-crossing.
    struct Trim
    {
        Trim(LevelSetTracker& tracker) : mTracker(tracker) {}
        void trim();
        void operator()(const RangeType& r) const;
        LevelSetTracker& mTracker;
    };// Trim

    // Private class to perform multi-threaded normalization
    template<math::BiasedGradientScheme      SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme,
             typename MaskT>
    struct Normalizer
    {
        typedef math::BIAS_SCHEME<SpatialScheme>                             SchemeT;
        typedef typename SchemeT::template ISStencil<GridType>::StencilType  StencilT;
        typedef typename MaskT::LeafNodeType MaskLeafT;
        typedef typename MaskLeafT::ValueOnCIter MaskIterT;
        typedef typename LeafType::ValueOnCIter VoxelIterT;
        Normalizer(LevelSetTracker& tracker, const MaskT* mask);
        void normalize();
        void operator()(const RangeType& r) const {mTask(const_cast<Normalizer*>(this), r);}
        void cook(int swapBuffer=0);
        template <int Nominator, int Denominator>
        void euler(const RangeType& range, Index phiBuffer, Index resultBuffer);
        inline void euler01(const RangeType& r)                   {this->euler<0,1>(r, 0, 1);}
        inline void euler12(const RangeType& r, Index n, Index m) {this->euler<1,2>(r, n, m);}
        inline void euler34(const RangeType& r, Index n, Index m) {this->euler<3,4>(r, n, m);}
        inline void euler13(const RangeType& r, Index n, Index m) {this->euler<1,3>(r, n, m);}
        template <int Nominator, int Denominator>
        void eval(StencilT& stencil, const BufferType& phi, BufferType& result, Index n) const;
        typedef typename boost::function<void (Normalizer*, const RangeType&)> FuncType;
        LevelSetTracker& mTracker;
        const MaskT*     mMask;
        const ValueType  mDt, mInvDx;
        FuncType         mTask;
    }; // Normalizer

    template<math::BiasedGradientScheme SpatialScheme, typename MaskT>
    void normalize1(const MaskT* mask);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme, typename MaskT>
    void normalize2(const MaskT* mask);

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
    const bool                      mIsMaster;

    // disallow copy by assignment
    void operator=(const LevelSetTracker& other) {}

}; // end of LevelSetTracker class

template<typename GridT, typename InterruptT>
LevelSetTracker<GridT, InterruptT>::
LevelSetTracker(GridT& grid, InterruptT* interrupt):
    mGrid(&grid),
    mLeafs(new LeafManagerType(grid.tree())),
    mInterrupter(interrupt),
    mDx(static_cast<ValueType>(grid.voxelSize()[0])),
    mSpatialScheme(math::HJWENO5_BIAS),
    mTemporalScheme(math::TVD_RK1),
    mNormCount(static_cast<int>(LEVEL_SET_HALF_WIDTH)),
    mGrainSize(1),
    mIsMaster(true)// N.B.
{
    if ( !grid.hasUniformVoxels() ) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for the LevelSetTracker to function");
    }
    if ( grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
            "LevelSetTracker expected a level set, got a grid of class \""
            + grid.gridClassToString(grid.getGridClass())
            + "\" [hint: Grid::setGridClass(openvdb::GRID_LEVEL_SET)]");
    }
}

template<typename GridT, typename InterruptT>
LevelSetTracker<GridT, InterruptT>::
LevelSetTracker(const LevelSetTracker& other):
    mGrid(other.mGrid),
    mLeafs(other.mLeafs),
    mInterrupter(other.mInterrupter),
    mDx(other.mDx),
    mSpatialScheme(other.mSpatialScheme),
    mTemporalScheme(other.mTemporalScheme),
    mNormCount(other.mNormCount),
    mGrainSize(other.mGrainSize),
    mIsMaster(false)// N.B.
{
}

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
prune()
{
    this->startInterrupter("Pruning Level Set");

    // Prune voxels that are too far away from the zero-crossing
    Trim t(*this);
    t.trim();

    // Remove inactive nodes from tree
    tools::pruneLevelSet(mGrid->tree());

    // The tree topology has changes so rebuild the list of leafs
    mLeafs->rebuildLeafArray();
    this->endInterrupter();
}

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
track()
{
    // Dilate narrow-band (this also rebuilds the leaf array!)
    tools::dilateVoxels(*mLeafs);

    // Compute signed distances in dilated narrow-band
    this->normalize();

    // Remove voxels that are outside the narrow band
    this->prune();
}

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
dilate(int iterations)
{
    if (mNormCount == 0) {
        for (int i=0; i < iterations; ++i) {
            tools::dilateVoxels(*mLeafs);
            mLeafs->rebuildLeafArray();
            tools::changeLevelSetBackground(leafs(), mDx + mGrid->background());
        }
    } else {
        for (int i=0; i < iterations; ++i) {
            BoolMaskType mask0(mGrid->tree(), false, TopologyCopy());
            tools::dilateVoxels(*mLeafs);
            mLeafs->rebuildLeafArray();
            tools::changeLevelSetBackground(leafs(), mDx + mGrid->background());
            BoolMaskType mask(mGrid->tree(), false, TopologyCopy());
            mask.topologyDifference(mask0);
            this->normalize(&mask);
        }
    }
}

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
erode(int iterations)
{
    tools::erodeVoxels(*mLeafs, iterations);
    mLeafs->rebuildLeafArray();
    const ValueType background = mGrid->background() - iterations*mDx;
    tools::changeLevelSetBackground(leafs(), background);
}

template<typename GridT,  typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
startInterrupter(const char* msg)
{
    if (mInterrupter) mInterrupter->start(msg);
}

template<typename GridT,  typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
endInterrupter()
{
    if (mInterrupter) mInterrupter->end();
}

template<typename GridT,  typename InterruptT>
inline bool
LevelSetTracker<GridT, InterruptT>::
checkInterrupter()
{
    if (util::wasInterrupted(mInterrupter)) {
        tbb::task::self().cancel_group_execution();
        return false;
    }
    return true;
}

template<typename GridT, typename InterruptT>
template<typename MaskT>
inline void
LevelSetTracker<GridT, InterruptT>::
normalize(const MaskT* mask)
{
    switch (mSpatialScheme) {
    case math::FIRST_BIAS:
        this->normalize1<math::FIRST_BIAS ,  MaskT>(mask); break;
    case math::SECOND_BIAS:
        this->normalize1<math::SECOND_BIAS,  MaskT>(mask); break;
    case math::THIRD_BIAS:
        this->normalize1<math::THIRD_BIAS,   MaskT>(mask); break;
    case math::WENO5_BIAS:
        this->normalize1<math::WENO5_BIAS,   MaskT>(mask); break;
    case math::HJWENO5_BIAS:
        this->normalize1<math::HJWENO5_BIAS, MaskT>(mask); break;
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme, typename MaskT>
inline void
LevelSetTracker<GridT, InterruptT>::
normalize1(const MaskT* mask)
{
    switch (mTemporalScheme) {
    case math::TVD_RK1:
        this->normalize2<SpatialScheme, math::TVD_RK1, MaskT>(mask); break;
    case math::TVD_RK2:
        this->normalize2<SpatialScheme, math::TVD_RK2, MaskT>(mask); break;
    case math::TVD_RK3:
        this->normalize2<SpatialScheme, math::TVD_RK3, MaskT>(mask); break;
    default:
        OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
inline void
LevelSetTracker<GridT, InterruptT>::
normalize2(const MaskT* mask)
{
    Normalizer<SpatialScheme, TemporalScheme, MaskT> tmp(*this, mask);
    tmp.normalize();
}

////////////////////////////////////////////////////////////////////////////

template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
Trim::trim()
{
    const int grainSize = mTracker.getGrainSize();
    if (grainSize>0) {
        tbb::parallel_for(mTracker.mLeafs->getRange(grainSize), *this);
    } else {
        (*this)(mTracker.mLeafs->getRange());
    }
}

/// Prunes away voxels that have moved outside the narrow band
template<typename GridT, typename InterruptT>
inline void
LevelSetTracker<GridT, InterruptT>::
Trim::operator()(const RangeType& range) const
{
    typedef typename LeafType::ValueOnIter VoxelIterT;
    mTracker.checkInterrupter();
    const ValueType gamma = mTracker.mGrid->background();
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        LeafType &leaf = mTracker.mLeafs->leaf(n);
        for (VoxelIterT iter = leaf.beginValueOn(); iter; ++iter) {
            const ValueType val = *iter;
            if (val <= -gamma)
                leaf.setValueOff(iter.pos(), -gamma);
            else if (val >= gamma)
                leaf.setValueOff(iter.pos(),  gamma);
        }
    }
}

////////////////////////////////////////////////////////////////////////////

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
inline
LevelSetTracker<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
Normalizer(LevelSetTracker& tracker, const MaskT* mask)
    : mTracker(tracker)
    , mMask(mask)
    , mDt(tracker.voxelSize()*(TemporalScheme == math::TVD_RK1 ? 0.3f :
                               TemporalScheme == math::TVD_RK2 ? 0.9f : 1.0f))
    , mInvDx(1.0f/tracker.voxelSize())
    , mTask(0)
{
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
inline void
LevelSetTracker<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
normalize()
{
    /// Make sure we have enough temporal auxiliary buffers
    mTracker.mLeafs->rebuildAuxBuffers(TemporalScheme == math::TVD_RK3 ? 2 : 1);

    for (int n=0, e=mTracker.getNormCount(); n < e; ++n) {

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        switch(TemporalScheme) {//switch is resolved at compile-time
        case math::TVD_RK1:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(0) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = boost::bind(&Normalizer::euler01, _1, _2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(1);
            break;
        case math::TVD_RK2:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = boost::bind(&Normalizer::euler01, _1, _2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(1);

            // Convex combine explict Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&Normalizer::euler12, _1, _2, /*phi=*/1, /*result=*/1);

            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)
            this->cook(1);
            break;
        case math::TVD_RK3:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = boost::bind(&Normalizer::euler01, _1, _2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(1);

            // Convex combine explict Euler step: t2 = t0 + dt/2
            // Phi_t2(2) = 3/4 * Phi_t0(1) + 1/4 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&Normalizer::euler34, _1, _2, /*phi=*/1, /*result=*/2);

            // Cook and swap buffer 0 and 2 such that Phi_t2(0) and Phi_t1(2)
            this->cook(2);

            // Convex combine explict Euler step: t3 = t0 + dt
            // Phi_t3(2) = 1/3 * Phi_t0(1) + 2/3 * (Phi_t2(0) - dt * V.Grad_t2(0)
            mTask = boost::bind(&Normalizer::euler13, _1, _2, /*phi=*/1, /*result=*/2);

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
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
inline void
LevelSetTracker<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
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

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
template <int Nominator, int Denominator>
inline void
LevelSetTracker<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
eval(StencilT& stencil, const BufferType& phi, BufferType& result, Index n) const
{
    typedef typename math::ISGradientNormSqrd<SpatialScheme> GradientT;
    static const ValueType alpha = ValueType(Nominator)/ValueType(Denominator);
    static const ValueType beta  = ValueType(1) - alpha;

    const ValueType normSqGradPhi = GradientT::result(stencil);
    const ValueType phi0 = stencil.getValue();
    ValueType v = phi0 / (math::Sqrt(math::Pow2(phi0) + normSqGradPhi));
    v = phi0 - mDt * v * (math::Sqrt(normSqGradPhi) * mInvDx - 1.0f);
    result.setValue(n, Nominator ? alpha * phi[n] + beta * v : v);
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
template <int Nominator, int Denominator>
inline void
LevelSetTracker<GridT,InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
euler(const RangeType& range, Index phiBuffer, Index resultBuffer)
{
    typedef typename LeafType::ValueOnCIter VoxelIterT;

    mTracker.checkInterrupter();

    StencilT stencil(mTracker.grid());

    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        const BufferType& phi = mTracker.mLeafs->getBuffer(n, phiBuffer);
        BufferType& result    = mTracker.mLeafs->getBuffer(n, resultBuffer);
        const LeafType& leaf  = mTracker.mLeafs->leaf(n);

        if (mMask == NULL) {
            for (VoxelIterT iter = leaf.cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                this->eval<Nominator,Denominator>(stencil, phi, result, iter.pos());
            }//loop over active voxels in the leaf of the level set
        } else if (const MaskLeafT* mask = mMask->probeLeaf(leaf.origin())) {
            for (MaskIterT iter  = mask->cbeginValueOn(); iter; ++iter) {
                const Index i = iter.pos();
                stencil.moveTo(iter.getCoord(), leaf.getValue(i));
                this->eval<Nominator,Denominator>(stencil, phi, result, i);
            }//loop over active voxels in the leaf of the mask
        }
    }//loop over leafs of the level set
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_TRACKER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
