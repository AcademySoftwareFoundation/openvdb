// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth
///
/// @file tools/LevelSetTrackerNew.h
///
/// @brief Performs multi-threaded interface tracking of narrow band
/// level sets. This is the building-block for most level set
/// computations that involve dynamic topology, e.g. advection.

#ifndef OPENVDB_TOOLS_LEVEL_SET_TRACKER_NEW_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_TRACKER_NEW_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/FiniteDifference.h>
#include <openvdb/math/Operators.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/math/Transform.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/thread/Threading.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/ChangeBackground.h>// for changeLevelSetBackground
#include <openvdb/tools/Morphology.h>//for dilateActiveValues
#include <openvdb/tools/Prune.h>// for pruneLevelSet

#include <tbb/parallel_for.h>

#include <functional>
#include <type_traits>

#include "Benchmark.h"
#include <nanovdb/NanoVDB.h>
#include "Stencils.h"

#define USE_NANOVDB_IMPLEMENTATION_FOR_NORMGRAD

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

namespace lstrack {

/// @brief How to handle voxels that fall outside the narrow band
/// @sa @link LevelSetTrackerNew::trimming() trimming@endlink,
///     @link LevelSetTrackerNew::setTrimming() setTrimming@endlink
enum class TrimMode {
    kNone,     ///< Leave out-of-band voxels intact
    kInterior, ///< Set out-of-band interior voxels to the background value
    kExterior, ///< Set out-of-band exterior voxels to the background value
    kAll       ///< Set all out-of-band voxels to the background value
};

} // namespace lstrack


/// @brief Performs multi-threaded interface tracking of narrow band level sets
template<typename GridT, typename InterruptT = util::NullInterrupter>
class LevelSetTrackerNew
{
public:
    using TrimMode = lstrack::TrimMode;

    using GridType = GridT;
    using TreeType = typename GridT::TreeType;
    using LeafType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;
    using LeafManagerType = typename tree::LeafManager<TreeType>; // leafs + buffers
    using LeafRange = typename LeafManagerType::LeafRange;
    using BufferType = typename LeafManagerType::BufferType;
    using MaskTreeType = typename TreeType::template ValueConverter<ValueMask>::Type;
    static_assert(std::is_floating_point<ValueType>::value,
        "LevelSetTrackerNew requires a level set grid with floating-point values");

    /// Lightweight struct that stores the state of the LevelSetTrackerNew
    struct State {
        State(math::BiasedGradientScheme s = math::HJWENO5_BIAS,
              math::TemporalIntegrationScheme t = math::TVD_RK1,
              int n = static_cast<int>(LEVEL_SET_HALF_WIDTH), int g = 1)
            : spatialScheme(s), temporalScheme(t), normCount(n), grainSize(g) {}
        math::BiasedGradientScheme      spatialScheme;
        math::TemporalIntegrationScheme temporalScheme;
        int                             normCount;// Number of iterations of normalization
        int                             grainSize;
    };

    /// @brief Main constructor
    /// @throw RuntimeError if the grid is not a level set
    LevelSetTrackerNew(GridT& grid, InterruptT* interrupt = nullptr);

    virtual ~LevelSetTrackerNew() { delete mLeafs; }

    /// @brief Iterative normalization, i.e. solving the Eikonal equation
    /// @note The mask it optional and by default it is ignored.
    template <typename MaskType>
    void normalize(const MaskType* mask);

    /// @brief Iterative normalization, i.e. solving the Eikonal equation
    void normalize() { this->normalize<MaskTreeType>(nullptr); }

    /// @brief Track the level set interface, i.e. rebuild and normalize the
    /// narrow band of the level set.
    void track();

    /// @brief Set voxels that are outside the narrow band to the background value
    /// (if trimming is enabled) and prune the grid.
    /// @details Pruning is done automatically as a step in tracking.
    /// @sa @link setTrimming() setTrimming@endlink, @link trimming() trimming@endlink
    void prune();

    /// @brief Fast but approximate dilation of the narrow band - one
    /// layer at a time. Normally we recommend using the resize method below
    /// which internally calls dilate (or erode) with the correct
    /// number of @a iterations to achieve the desired half voxel width
    /// of the narrow band (3 is recommended for most level set applications).
    ///
    /// @note Since many level set applications perform
    /// interface-tracking, which in turn rebuilds the narrow-band
    /// accurately, this dilate method can often be used with a
    /// single iterations of low-order re-normalization. This
    /// effectively allows very narrow bands to be created from points
    /// or polygons (e.g. with a half voxel width of 1), followed by a
    /// fast but approximate dilation (typically with a half voxel
    /// width of 3). This can be significantly faster than generating
    /// the final width of the narrow band from points or polygons.
    void dilate(int iterations = 1);

    /// @brief Erodes the width of the narrow-band and update the background values
    /// @throw ValueError if @a iterations is larger than the current half-width.
    void erode(int iterations = 1);

    /// @brief Resize the width of the narrow band, i.e. perform
    /// dilation and renormalization or erosion as required.
    bool resize(Index halfWidth = static_cast<Index>(LEVEL_SET_HALF_WIDTH));

    /// @brief Return the half width of the narrow band in floating-point voxel units.
    ValueType getHalfWidth() const { return mGrid->background()/mDx; }

    /// @brief Return the state of the tracker (see struct defined above)
    State getState() const { return mState; }

    /// @brief Set the state of the tracker (see struct defined above)
    void setState(const State& s) { mState = s; }

    /// @return the spatial finite difference scheme
    math::BiasedGradientScheme getSpatialScheme() const { return mState.spatialScheme; }

    /// @brief Set the spatial finite difference scheme
    void setSpatialScheme(math::BiasedGradientScheme s) { mState.spatialScheme = s; }

    /// @return the temporal integration scheme
    math::TemporalIntegrationScheme getTemporalScheme() const { return mState.temporalScheme; }

    /// @brief Set the spatial finite difference scheme
    void setTemporalScheme(math::TemporalIntegrationScheme s) { mState.temporalScheme = s;}

    /// @return The number of normalizations performed per track or
    /// normalize call.
    int getNormCount() const { return mState.normCount; }

    /// @brief Set the number of normalizations performed per track or
    /// normalize call.
    void setNormCount(int n) { mState.normCount = n; }

    /// @return the grain-size used for multi-threading
    int getGrainSize() const { return mState.grainSize; }

    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(int grainsize) { mState.grainSize = grainsize; }

    /// @brief Return the trimming mode for voxels outside the narrow band.
    /// @details Trimming is enabled by default and is applied automatically prior to pruning.
    /// @sa @link setTrimming() setTrimming@endlink, @link prune() prune@endlink
    TrimMode trimming() const { return mTrimMode; }
    /// @brief Specify whether to trim voxels outside the narrow band prior to pruning.
    /// @sa @link trimming() trimming@endlink, @link prune() prune@endlink
    void setTrimming(TrimMode mode) { mTrimMode = mode; }

    ValueType voxelSize() const { return mDx; }

    void startInterrupter(const char* msg);

    void endInterrupter();

    /// @return false if the process was interrupted
    bool checkInterrupter();

    const GridType& grid() const { return *mGrid; }

    LeafManagerType& leafs() { return *mLeafs; }

    const LeafManagerType& leafs() const { return *mLeafs; }

private:
    // disallow copy construction and copy by assignment!
    LevelSetTrackerNew(const LevelSetTrackerNew&);// not implemented
    LevelSetTrackerNew& operator=(const LevelSetTrackerNew&);// not implemented

    // Private class to perform multi-threaded trimming of
    // voxels that are too far away from the zero-crossing.
    template<TrimMode Trimming>
    struct Trim
    {
        Trim(LevelSetTrackerNew& tracker) : mTracker(tracker) {}
        void trim();
        void operator()(const LeafRange& r) const;
        LevelSetTrackerNew& mTracker;
    };// Trim

    // Private struct to perform multi-threaded normalization
    template<math::BiasedGradientScheme      SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme,
             typename MaskT>
    struct Normalizer
    {
        using SchemeT = math::BIAS_SCHEME<SpatialScheme>;
        using StencilT = typename SchemeT::template ISStencil<GridType>::StencilType;
        using MaskLeafT = typename MaskT::LeafNodeType;
        using MaskIterT = typename MaskLeafT::ValueOnCIter;
        using VoxelIterT = typename LeafType::ValueOnCIter;

        Normalizer(LevelSetTrackerNew& tracker, const MaskT* mask);
        void normalize();
        void operator()(const LeafRange& r) const {mTask(const_cast<Normalizer*>(this), r);}
        void cook(const char* msg, int swapBuffer=0);
        template <int Nominator, int Denominator>
        void euler(const LeafRange& range, Index phiBuffer, Index resultBuffer);
        inline void euler01(const LeafRange& r) {this->euler<0,1>(r, 0, 1);}
        inline void euler12(const LeafRange& r) {this->euler<1,2>(r, 1, 1);}
        inline void euler34(const LeafRange& r) {this->euler<3,4>(r, 1, 2);}
        inline void euler13(const LeafRange& r) {this->euler<1,3>(r, 1, 2);}
        template <int Nominator, int Denominator>
        void eval(StencilT& stencil, const ValueType* phi, ValueType* result, Index n) const;
        LevelSetTrackerNew& mTracker;
        const MaskT*     mMask;
        const ValueType  mDt, mInvDx;
        typename std::function<void (Normalizer*, const LeafRange&)> mTask;
    }; // Normalizer struct

    template<math::BiasedGradientScheme SpatialScheme, typename MaskT>
    void normalize1(const MaskT* mask);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme, typename MaskT>
    void normalize2(const MaskT* mask);

    // Throughout the methods below mLeafs is always assumed to contain
    // a list of the current LeafNodes! The auxiliary buffers on the
    // other hand always have to be allocated locally, since some
    // methods need them and others don't!
    GridType*        mGrid;
    LeafManagerType* mLeafs;
    InterruptT*      mInterrupter;
    const ValueType  mDx;
    State            mState;
    TrimMode         mTrimMode = TrimMode::kAll;
}; // end of LevelSetTrackerNew class

template<typename GridT, typename InterruptT>
LevelSetTrackerNew<GridT, InterruptT>::
LevelSetTrackerNew(GridT& grid, InterruptT* interrupt):
    mGrid(&grid),
    mLeafs(new LeafManagerType(grid.tree())),
    mInterrupter(interrupt),
    mDx(static_cast<ValueType>(grid.voxelSize()[0])),
    mState()
{
    if ( !grid.hasUniformVoxels() ) {
         OPENVDB_THROW(RuntimeError,
             "The transform must have uniform scale for the LevelSetTrackerNew to function");
    }
    if ( grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
            "LevelSetTrackerNew expected a level set, got a grid of class \""
            + grid.gridClassToString(grid.getGridClass())
            + "\" [hint: Grid::setGridClass(openvdb::GRID_LEVEL_SET)]");
    }
}

template<typename GridT, typename InterruptT>
void
LevelSetTrackerNew<GridT, InterruptT>::
prune()
{
    this->startInterrupter("Pruning Level Set");

    // Set voxels that are too far away from the zero crossing to the background value.
    switch (mTrimMode) {
        case TrimMode::kNone:     break;
        case TrimMode::kInterior: Trim<TrimMode::kInterior>(*this).trim(); break;
        case TrimMode::kExterior: Trim<TrimMode::kExterior>(*this).trim(); break;
        case TrimMode::kAll:      Trim<TrimMode::kAll>(*this).trim(); break;
    }

    // Remove inactive nodes from tree
    tools::pruneLevelSet(mGrid->tree());

    // The tree topology has changes so rebuild the list of leafs
    mLeafs->rebuildLeafArray();
    this->endInterrupter();
}

template<typename GridT, typename InterruptT>
void
LevelSetTrackerNew<GridT, InterruptT>::
track()
{
    openvdb::util::CpuTimer timer;

    Benchmark::printGridDiagnostics(Benchmark::getInstance().mHandle);

    // Dilate narrow-band

    timer.start("[LevelSetTrackerNew::track()] Dilating Active Values (NanoVDB)");
    if (Benchmark::getInstance().onCPU())
        Benchmark::dilateActiveValues<ExecutionPolicy::CPU>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, mGrid->background());
    else
        Benchmark::dilateActiveValues<ExecutionPolicy::CUDA>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, mGrid->background());
    timer.stop();
    
    timer.start("[LevelSetTrackerNew::track()] Normalizing Level Set (NanoVDB)");
    if (Benchmark::getInstance().onCPU())
        Benchmark::normalizeLevelSet<ExecutionPolicy::CPU>( Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, mGrid->background(),
            getNormCount(), voxelSize());
    else
        Benchmark::normalizeLevelSet<ExecutionPolicy::CUDA>( Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, mGrid->background(),
            getNormCount(), voxelSize());
    timer.stop();

    timer.start("[LevelSetTrackerNew::track()] Pruning narrow band (NanoVDB)");
    if (Benchmark::getInstance().onCPU())
        Benchmark::pruneNarrowBand<ExecutionPolicy::CPU>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, mGrid->background());
    else
        Benchmark::pruneNarrowBand<ExecutionPolicy::CUDA>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, mGrid->background());
    timer.stop();
}

template<typename GridT, typename InterruptT>
void
LevelSetTrackerNew<GridT, InterruptT>::
dilate(int iterations)
{
    if (this->getNormCount() == 0) {
        for (int i=0; i < iterations; ++i) {
            tools::dilateActiveValues( *mLeafs, 1, tools::NN_FACE, tools::IGNORE_TILES);
            tools::changeLevelSetBackground(this->leafs(), mDx + mGrid->background());
        }
    } else {
        for (int i=0; i < iterations; ++i) {
            MaskTreeType mask0(mGrid->tree(), false, TopologyCopy());
            tools::dilateActiveValues( *mLeafs, 1, tools::NN_FACE, tools::IGNORE_TILES);
            tools::changeLevelSetBackground(this->leafs(), mDx + mGrid->background());
            MaskTreeType mask(mGrid->tree(), false, TopologyCopy());
            mask.topologyDifference(mask0);
            this->normalize(&mask);
        }
    }
}

template<typename GridT, typename InterruptT>
void
LevelSetTrackerNew<GridT, InterruptT>::
erode(int iterations)
{
    tools::erodeActiveValues(*mLeafs, iterations, tools::NN_FACE, tools::IGNORE_TILES);
    tools::pruneLevelSet(mLeafs->tree());
    mLeafs->rebuildLeafArray();
    const ValueType background = mGrid->background() - ValueType(iterations) * mDx;
    tools::changeLevelSetBackground(this->leafs(), background);
}

template<typename GridT, typename InterruptT>
bool
LevelSetTrackerNew<GridT, InterruptT>::
resize(Index halfWidth)
{
    const int wOld = static_cast<int>(math::RoundDown(this->getHalfWidth()));
    const int wNew = static_cast<int>(halfWidth);
    if (wOld < wNew) {
        this->dilate(wNew - wOld);
    } else if (wOld > wNew) {
        this->erode(wOld - wNew);
    }
    return wOld != wNew;
}

template<typename GridT,  typename InterruptT>
inline void
LevelSetTrackerNew<GridT, InterruptT>::
startInterrupter(const char* msg)
{
    if (mInterrupter) mInterrupter->start(msg);
}

template<typename GridT,  typename InterruptT>
inline void
LevelSetTrackerNew<GridT, InterruptT>::
endInterrupter()
{
    if (mInterrupter) mInterrupter->end();
}

template<typename GridT,  typename InterruptT>
inline bool
LevelSetTrackerNew<GridT, InterruptT>::
checkInterrupter()
{
    if (util::wasInterrupted(mInterrupter)) {
        thread::cancelGroupExecution();
        return false;
    }
    return true;
}

template<typename GridT, typename InterruptT>
template<typename MaskT>
void
LevelSetTrackerNew<GridT, InterruptT>::
normalize(const MaskT* mask)
{
    switch (this->getSpatialScheme()) {
    case math::HJWENO5_BIAS:
        this->normalize1<math::HJWENO5_BIAS, MaskT>(mask); break;
    case math::UNKNOWN_BIAS:
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme, typename MaskT>
void
LevelSetTrackerNew<GridT, InterruptT>::
normalize1(const MaskT* mask)
{
    switch (this->getTemporalScheme()) {
    case math::TVD_RK1:
        this->normalize2<SpatialScheme, math::TVD_RK1, MaskT>(mask); break;
    case math::TVD_RK2:
        this->normalize2<SpatialScheme, math::TVD_RK2, MaskT>(mask); break;
    case math::TVD_RK3:
        this->normalize2<SpatialScheme, math::TVD_RK3, MaskT>(mask); break;
    case math::UNKNOWN_TIS:
    default:
        OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
void
LevelSetTrackerNew<GridT, InterruptT>::
normalize2(const MaskT* mask)
{
    Normalizer<SpatialScheme, TemporalScheme, MaskT> tmp(*this, mask);
    tmp.normalize();
}


////////////////////////////////////////////////////////////////////////////


template<typename GridT, typename InterruptT>
template<lstrack::TrimMode Trimming>
void
LevelSetTrackerNew<GridT, InterruptT>::Trim<Trimming>::trim()
{
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
    if (Trimming != TrimMode::kNone) {
        const int grainSize = mTracker.getGrainSize();
        const LeafRange range = mTracker.leafs().leafRange(grainSize);

        if (grainSize>0) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}


/// Trim away voxels that have moved outside the narrow band
template<typename GridT, typename InterruptT>
template<lstrack::TrimMode Trimming>
inline void
LevelSetTrackerNew<GridT, InterruptT>::Trim<Trimming>::operator()(const LeafRange& range) const
{
    mTracker.checkInterrupter();
    const ValueType gamma = mTracker.mGrid->background();

    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
    for (auto leafIter = range.begin(); leafIter; ++leafIter) {
        auto& leaf = *leafIter;
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            const auto val = *iter;
            switch (Trimming) { // resolved at compile time
                case TrimMode::kNone:
                    break;
                case TrimMode::kInterior:
                    if (val <= -gamma) { leaf.setValueOff(iter.pos(), -gamma); }
                    break;
                case TrimMode::kExterior:
                    if (val >= gamma) { leaf.setValueOff(iter.pos(), gamma); }
                    break;
                case TrimMode::kAll:
                    if (val <= -gamma) {
                        leaf.setValueOff(iter.pos(), -gamma);
                    } else if (val >= gamma) {
                        leaf.setValueOff(iter.pos(), gamma);
                    }
                    break;
            }
        }
    }
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}


////////////////////////////////////////////////////////////////////////////

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
inline
LevelSetTrackerNew<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
Normalizer(LevelSetTrackerNew& tracker, const MaskT* mask)
    : mTracker(tracker)
    , mMask(mask)
    , mDt(tracker.voxelSize()*(TemporalScheme == math::TVD_RK1 ? 0.3f :
                               TemporalScheme == math::TVD_RK2 ? 0.9f : 1.0f))
    , mInvDx(1.0f/tracker.voxelSize())
    , mTask(nullptr)
{
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
inline void
LevelSetTrackerNew<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
normalize()
{
    // openvdb::util::CpuTimer timer;
    namespace ph = std::placeholders;

    /// Make sure we have enough temporal auxiliary buffers
    // timer.start("[LevelSetTrackerNew::normalize()] rebuildAuxBuffers");
    mTracker.mLeafs->rebuildAuxBuffers(TemporalScheme == math::TVD_RK3 ? 2 : 1);

    for (int n=0, e=mTracker.getNormCount(); n < e; ++n) {

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        switch(TemporalScheme) {//switch is resolved at compile-time
        case math::TVD_RK1:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(0) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = std::bind(&Normalizer::euler01, ph::_1, ph::_2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook("Normalizing level set using TVD_RK1", 1);
            break;
        case math::TVD_RK2:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = std::bind(&Normalizer::euler01, ph::_1, ph::_2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            // timer.stop();
            // timer.start("[LevelSetTrackerNew::normalize()] euler01");
            this->cook("Normalizing level set using TVD_RK1 (step 1 of 2)", 1);

            // Convex combine explicit Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = std::bind(&Normalizer::euler12, ph::_1, ph::_2);

            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)

            // timer.restart("[LevelSetTrackerNew::normalize()] euler12");
            this->cook("Normalizing level set using TVD_RK1 (step 2 of 2)", 1);
            break;
        case math::TVD_RK3:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(1)
            mTask = std::bind(&Normalizer::euler01, ph::_1, ph::_2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook("Normalizing level set using TVD_RK3 (step 1 of 3)", 1);

            // Convex combine explicit Euler step: t2 = t0 + dt/2
            // Phi_t2(2) = 3/4 * Phi_t0(1) + 1/4 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = std::bind(&Normalizer::euler34, ph::_1, ph::_2);

            // Cook and swap buffer 0 and 2 such that Phi_t2(0) and Phi_t1(2)
            this->cook("Normalizing level set using TVD_RK3 (step 2 of 3)", 2);

            // Convex combine explicit Euler step: t3 = t0 + dt
            // Phi_t3(2) = 1/3 * Phi_t0(1) + 2/3 * (Phi_t2(0) - dt * V.Grad_t2(0)
            mTask = std::bind(&Normalizer::euler13, ph::_1, ph::_2);

            // Cook and swap buffer 0 and 2 such that Phi_t3(0) and Phi_t2(2)
            this->cook("Normalizing level set using TVD_RK3 (step 3 of 3)", 2);
            break;
        case math::UNKNOWN_TIS:
        default:
            OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    // timer.restart("[LevelSetTrackerNew::normalize()] removeAuxBuffers");
    mTracker.mLeafs->removeAuxBuffers();
    // timer.stop();
}

/// Private method to perform the task (serial or threaded) and
/// subsequently swap the leaf buffers.
template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
inline void
LevelSetTrackerNew<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
cook(const char* msg, int swapBuffer)
{
    mTracker.startInterrupter( msg );

    const int grainSize   = mTracker.getGrainSize();
    const LeafRange range = mTracker.leafs().leafRange(grainSize);

    grainSize>0 ? tbb::parallel_for(range, *this) : (*this)(range);

    mTracker.leafs().swapLeafBuffer(swapBuffer, grainSize==0);

    mTracker.endInterrupter();
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
template <int Nominator, int Denominator>
inline void
LevelSetTrackerNew<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
eval(StencilT& stencil, const ValueType* phi, ValueType* result, Index n) const
{
    using GradientT = typename math::ISGradientNormSqrd<SpatialScheme>;
    using NanoGradientT = typename nanovdb::math::WenoStencil<nanovdb::FloatGrid>; // For debugging purposes
    static const ValueType alpha = ValueType(Nominator)/ValueType(Denominator);
    static const ValueType beta  = ValueType(1) - alpha;

    ValueType normSqGradPhi = GradientT::result(stencil);

#ifdef USE_NANOVDB_IMPLEMENTATION_FOR_NORMGRAD
    // For debugging purposes
    ValueType nanoStencil[NanoGradientT::SIZE];
    nanoStencil[nanovdb::math::WenoPt< 0, 0, 0>::idx] = stencil.template getValue< 0, 0, 0>();
    nanoStencil[nanovdb::math::WenoPt< 1, 0, 0>::idx] = stencil.template getValue< 1, 0, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 1, 0>::idx] = stencil.template getValue< 0, 1, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 0, 1>::idx] = stencil.template getValue< 0, 0, 1>();
    nanoStencil[nanovdb::math::WenoPt<-1, 0, 0>::idx] = stencil.template getValue<-1, 0, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0,-1, 0>::idx] = stencil.template getValue< 0,-1, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 0,-1>::idx] = stencil.template getValue< 0, 0,-1>();
    nanoStencil[nanovdb::math::WenoPt< 2, 0, 0>::idx] = stencil.template getValue< 2, 0, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 2, 0>::idx] = stencil.template getValue< 0, 2, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 0, 2>::idx] = stencil.template getValue< 0, 0, 2>();
    nanoStencil[nanovdb::math::WenoPt<-2, 0, 0>::idx] = stencil.template getValue<-2, 0, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0,-2, 0>::idx] = stencil.template getValue< 0,-2, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 0,-2>::idx] = stencil.template getValue< 0, 0,-2>();
    nanoStencil[nanovdb::math::WenoPt< 3, 0, 0>::idx] = stencil.template getValue< 3, 0, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 3, 0>::idx] = stencil.template getValue< 0, 3, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 0, 3>::idx] = stencil.template getValue< 0, 0, 3>();
    nanoStencil[nanovdb::math::WenoPt<-3, 0, 0>::idx] = stencil.template getValue<-3, 0, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0,-3, 0>::idx] = stencil.template getValue< 0,-3, 0>();
    nanoStencil[nanovdb::math::WenoPt< 0, 0,-3>::idx] = stencil.template getValue< 0, 0,-3>();
    const ValueType normSqGradPhiNano = NanoGradientT::normSqGrad(nanoStencil, 1.0, 1.0);
    if (fabs(normSqGradPhi-normSqGradPhiNano)>2e-3){
        std::cout << "difference = " << fabs(normSqGradPhi-normSqGradPhiNano) << std::endl;
        throw std::runtime_error("Inconsistency in WENO calculation");
    }
    normSqGradPhi = normSqGradPhiNano;
#endif

    const ValueType phi0 = stencil.getValue();
    ValueType v = phi0 / ( math::Sqrt(math::Pow2(phi0) + normSqGradPhi) +
                           math::Tolerance<ValueType>::value() );
    v = phi0 - mDt * v * (math::Sqrt(normSqGradPhi) * mInvDx - 1.0f);
    result[n] = Nominator ? alpha * phi[n] + beta * v : v;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme      SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MaskT>
template <int Nominator, int Denominator>
inline void
LevelSetTrackerNew<GridT,InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
euler(const LeafRange& range, Index phiBuffer, Index resultBuffer)
{
    mTracker.checkInterrupter();

    StencilT stencil(mTracker.grid());

    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        const ValueType* phi = leafIter.buffer(phiBuffer).data();
        ValueType* result = leafIter.buffer(resultBuffer).data();
        if (mMask == nullptr) {
            for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                // if (!Benchmark::compareStencil(stencil, iter.getCoord(), 0)) throw std::runtime_error("[Benchmark::compareStencil] Inconsistent stencil found");
                this->eval<Nominator, Denominator>(stencil, phi, result, iter.pos());
            }//loop over active voxels in the leaf of the level set
        } else if (const MaskLeafT* mask = mMask->probeLeaf(leafIter->origin())) {
            throw std::runtime_error("Execution reached unintended branch in LevelSetTrackerNew::Normalizer::euler()");
            const ValueType* phi0 = leafIter->buffer().data();
            for (MaskIterT iter  = mask->cbeginValueOn(); iter; ++iter) {
                const Index i = iter.pos();
                stencil.moveTo(iter.getCoord(), phi0[i]);
                this->eval<Nominator, Denominator>(stencil, phi, result, i);
            }//loop over active voxels in the leaf of the mask
        }
    }//loop over leafs of the level set
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_LEVELSETTRACKERNEW
#include <openvdb/util/ExplicitInstantiation.h>
#endif

OPENVDB_INSTANTIATE_CLASS LevelSetTrackerNew<FloatGrid, util::NullInterrupter>;

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_TRACKER_NEW_HAS_BEEN_INCLUDED
