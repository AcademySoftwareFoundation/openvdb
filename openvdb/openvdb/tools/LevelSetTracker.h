// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth
///
/// @file tools/LevelSetTracker.h
///
/// @brief Performs multi-threaded interface tracking of narrow band
/// level sets. This is the building-block for most level set
/// computations that involve dynamic topology, e.g. advection.

#ifndef OPENVDB_TOOLS_LEVEL_SET_TRACKER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_TRACKER_HAS_BEEN_INCLUDED

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
#include "ChangeBackground.h"// for changeLevelSetBackground
#include "Morphology.h"//for dilateActiveValues
#include "Prune.h"// for pruneLevelSet

#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

namespace lstrack {

/// @brief How to handle voxels that fall outside the narrow band
/// @sa @link LevelSetTracker::trimming() trimming@endlink,
///     @link LevelSetTracker::setTrimming() setTrimming@endlink
enum class TrimMode {
    kNone,     ///< Leave out-of-band voxels intact
    kInterior, ///< Set out-of-band interior voxels to the background value
    kExterior, ///< Set out-of-band exterior voxels to the background value
    kAll       ///< Set all out-of-band voxels to the background value
};

namespace internal {

/// @brief Replaces an axis-aligned math:: star stencil in the tracker and advection kernels.
///
/// @details Reads the star from the current leaf and its six face neighbors, whose buffers
/// are cached once per leaf, instead of performing 6 * @a Radius ValueAccessor lookups per
/// voxel. It has the same value layout as the stencil it replaces and reads the same values.
/// Results can only differ in the last bits, where a compiler fuses multiply-adds differently.
///
/// @tparam GridT   the type of the grid to read from
/// @tparam Radius  the reach along each axis: 1 replaces math::SevenPointStencil,
///                 2 math::ThirteenPointStencil and 3 math::NineteenPointStencil
///
/// @note setLeaf() must be called for each leaf before moveTo() on any of its voxels.
/// @warning Leaves must be at least @a Radius voxels wide, which LeafStencil ensures.
template<typename GridT, int Radius>
class LeafStarStencil
{
public:
    using TreeType = typename GridT::TreeType;
    using ValueType = typename GridT::ValueType;
    using LeafType = typename TreeType::LeafNodeType;

    /// @param grid  the grid to read from, which must outlive this stencil
    /// @note Allocates the six tile proxy buffers on the heap, once per stencil, so that the
    /// stack footprint does not grow with the leaf size.
    explicit LeafStarStencil(const GridT& grid)
        : mAcc(grid.tree()), mTile(6 * LeafType::SIZE) {}

    /// Not copyable, because mBuf can point into this stencil's own mTile.
    LeafStarStencil(const LeafStarStencil&) = delete;
    LeafStarStencil& operator=(const LeafStarStencil&) = delete;

    /// @brief Cache the buffers of @a leaf and its six face neighbors.
    /// @param leaf  the leaf whose voxels the following moveTo() calls will visit
    /// @details A face neighbor that is not a leaf lies inside a tile or the background, so it
    /// is replaced by a proxy buffer filled with that constant value.
    void setLeaf(const LeafType& leaf)
    {
        // Face neighbors in the order -x, +x, -y, +y, -z, +z, stored in mBuf[1] to mBuf[6].
        const std::array<std::array<int, 3>, 6> faces = {{
            {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}}, {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}} }};
        const int dim = int(LeafType::DIM);

        mBuf[0] = leaf.buffer().data();
        for (size_t n = 0; n < faces.size(); ++n) {
            const Coord p = leaf.origin().offsetBy(
                faces[n][0] * dim, faces[n][1] * dim, faces[n][2] * dim);

            if (const LeafType* neighbor = mAcc.probeConstLeaf(p)) {
                mBuf[n + 1] = neighbor->buffer().data();
            } else {
                ValueType* tile = mTile.data() + n * LeafType::SIZE;
                std::fill(tile, tile + LeafType::SIZE, mAcc.getValue(p));
                mBuf[n + 1] = tile;
            }
        }
    }

    /// @brief Read the star around a voxel of the current leaf.
    /// @param iter  a value iterator of the leaf passed to setLeaf()
    template<typename IterT>
    void moveTo(const IterT& iter) { this->gather(iter.getCoord(), iter.pos(), *iter); }

    /// @brief Read the star around a voxel of the current leaf.
    /// @param ijk          the coordinates of a voxel in the leaf passed to setLeaf()
    /// @param centerValue  the value to use for the center of the star
    void moveTo(const Coord& ijk, const ValueType& centerValue)
    {
        this->gather(ijk, LeafType::coordToOffset(ijk), centerValue);
    }

    /// Return the value at the center of the star.
    const ValueType& getValue() const { return mValues[0]; }

    /// Return the value at offset (i, j, k) from the center, e.g. getValue<0, -2, 0>().
    template<int i, int j, int k>
    const ValueType& getValue() const { return mValues[pos<i, j, k>()]; }

    /// Return the coordinates of the voxel at the center of the star.
    const Coord& getCenterCoord() const { return mCenter; }

private:
    static const int SIZE = 1 + 6 * Radius;

    /// @brief Return the index of the value at offset (i, j, k) from the center.
    /// @note The offset must lie on a single axis, at most @a Radius from the center.
    template<int i, int j, int k>
    static constexpr unsigned pos()
    {
        static_assert((i != 0) + (j != 0) + (k != 0) <= 1,
            "LeafStarStencil: the offset must lie on a single axis");
        static_assert(-Radius <= i && i <= Radius && -Radius <= j && j <= Radius
            && -Radius <= k && k <= Radius, "LeafStarStencil: the offset must be at most Radius");
        if (i != 0) return slot(0, i);
        if (j != 0) return slot(1, j);
        if (k != 0) return slot(2, k);
        return 0; // the center
    }

    /// @brief Return the index in mValues of a neighbor, in the layout of the math:: stencils:
    /// the center at 0, then six slots per distance from 1 to Radius, in the order
    /// +x, +y, +z, -x, -y, -z.
    /// @param axis  0 for x, 1 for y, 2 for z
    /// @param d     the signed distance from the center, from -Radius to Radius, but not 0
    static constexpr unsigned slot(int axis, int d)
    {
        const int distance = d > 0 ? d : -d;
        const int direction = d > 0 ? axis : axis + 3; // 0 to 5 for +x, +y, +z, -x, -y, -z
        return unsigned(1 + 6 * (distance - 1) + direction);
    }

    /// @brief Read the 6 * Radius neighbors of a voxel of the current leaf into mValues.
    /// @details A neighbor outside the leaf is read from the face neighbor on that side, at
    /// the same position shifted by one leaf width.
    /// @param ijk          the coordinates of the voxel
    /// @param n            the linear offset of the voxel in the leaf
    /// @param centerValue  the value to store for the center of the star
    void gather(const Coord& ijk, Index n, const ValueType& centerValue)
    {
        const int dim = int(LeafType::DIM);
        const std::array<int, 3> stride = {{ dim * dim, dim, 1 }}; // offset step along x, y, z
        const Coord local = LeafType::offsetToLocalCoord(n);

        mCenter = ijk;
        mValues[0] = centerValue;
        for (int axis = 0; axis < 3; ++axis) {
            for (int d = -Radius; d <= Radius; ++d) {
                if (d == 0) continue;
                const int t = local[axis] + d; // neighbor coordinate along this axis
                int buffer = 0, shift = d;     // the current leaf

                if (t < 0) {
                    buffer = 1 + 2 * axis;     // the neighbor on the negative side
                    shift += dim;
                } else if (t >= dim) {
                    buffer = 2 + 2 * axis;     // the neighbor on the positive side
                    shift -= dim;
                }
                mValues[slot(axis, d)] = mBuf[buffer][int(n) + shift * stride[axis]];
            }
        }
    }

    tree::ValueAccessor<const TreeType> mAcc;
    std::array<const ValueType*, 7> mBuf; // the leaf, then its -x, +x, -y, +y, -z, +z neighbors
    std::vector<ValueType> mTile;
    std::array<ValueType, SIZE> mValues;
    Coord mCenter;
};

/// @brief Selects the stencil type that the tracker and advection kernels use.
/// @details @c Type is LeafStarStencil in place of math::SevenPointStencil (FIRST_BIAS),
/// math::ThirteenPointStencil (SECOND_BIAS) and math::NineteenPointStencil (THIRD_BIAS,
/// WENO5_BIAS, HJWENO5_BIAS) if leaves are at least 4 voxels wide, so that offsets of +-3 never
/// reach past the face neighbors, and @a StencilT otherwise.
/// setLeaf() prepares the selected stencil for a leaf, and does nothing for a math:: stencil.
template<typename GridT, typename StencilT, bool = (GridT::TreeType::LeafNodeType::DIM >= 4)>
struct LeafStencil
{
    using Type = StencilT;
    template<typename LeafT> static void setLeaf(Type&, const LeafT&) {}
};

/// The LeafStencil members for a LeafStarStencil of the given radius.
template<typename GridT, int Radius>
struct LeafStarStencilSelector
{
    using Type = LeafStarStencil<GridT, Radius>;
    template<typename LeafT> static void setLeaf(Type& s, const LeafT& leaf) { s.setLeaf(leaf); }
};

template<typename GridT>
struct LeafStencil<GridT, math::SevenPointStencil<GridT>, true>
    : LeafStarStencilSelector<GridT, 1> {};

template<typename GridT>
struct LeafStencil<GridT, math::ThirteenPointStencil<GridT>, true>
    : LeafStarStencilSelector<GridT, 2> {};

template<typename GridT>
struct LeafStencil<GridT, math::NineteenPointStencil<GridT>, true>
    : LeafStarStencilSelector<GridT, 3> {};

} // namespace internal

} // namespace lstrack


/// @brief Performs multi-threaded interface tracking of narrow band level sets
template<typename GridT, typename InterruptT = util::NullInterrupter>
class LevelSetTracker
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
        "LevelSetTracker requires a level set grid with floating-point values");

    /// Lightweight struct that stores the state of the LevelSetTracker
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
    LevelSetTracker(GridT& grid, InterruptT* interrupt = nullptr);

    virtual ~LevelSetTracker() { delete mLeafs; }

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
    LevelSetTracker(const LevelSetTracker&);// not implemented
    LevelSetTracker& operator=(const LevelSetTracker&);// not implemented

    // Private class to perform multi-threaded trimming of
    // voxels that are too far away from the zero-crossing.
    template<TrimMode Trimming>
    struct Trim
    {
        Trim(LevelSetTracker& tracker) : mTracker(tracker) {}
        void trim();
        void operator()(const LeafRange& r) const;
        LevelSetTracker& mTracker;
    };// Trim

    // Private struct to perform multi-threaded normalization
    template<math::BiasedGradientScheme      SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme,
             typename MaskT>
    struct Normalizer
    {
        using SchemeT = math::BIAS_SCHEME<SpatialScheme>;
        using LeafStencilT = lstrack::internal::LeafStencil<GridType,
            typename SchemeT::template ISStencil<GridType>::StencilType>;
        using StencilT = typename LeafStencilT::Type;
        using MaskLeafT = typename MaskT::LeafNodeType;
        using MaskIterT = typename MaskLeafT::ValueOnCIter;
        using VoxelIterT = typename LeafType::ValueOnCIter;

        Normalizer(LevelSetTracker& tracker, const MaskT* mask);
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
        LevelSetTracker& mTracker;
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
}; // end of LevelSetTracker class

template<typename GridT, typename InterruptT>
LevelSetTracker<GridT, InterruptT>::
LevelSetTracker(GridT& grid, InterruptT* interrupt):
    mGrid(&grid),
    mLeafs(new LeafManagerType(grid.tree())),
    mInterrupter(interrupt),
    mDx(static_cast<ValueType>(grid.voxelSize()[0])),
    mState()
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
void
LevelSetTracker<GridT, InterruptT>::
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
LevelSetTracker<GridT, InterruptT>::
track()
{
    // Dilate narrow-band (this also rebuilds the leaf array!)
    tools::dilateActiveValues( *mLeafs, 1, tools::NN_FACE, tools::IGNORE_TILES);

    // Compute signed distances in dilated narrow-band
    this->normalize();

    // Remove voxels that are outside the narrow band
    this->prune();
}

template<typename GridT, typename InterruptT>
void
LevelSetTracker<GridT, InterruptT>::
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
LevelSetTracker<GridT, InterruptT>::
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
LevelSetTracker<GridT, InterruptT>::
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
        thread::cancelGroupExecution();
        return false;
    }
    return true;
}

template<typename GridT, typename InterruptT>
template<typename MaskT>
void
LevelSetTracker<GridT, InterruptT>::
normalize(const MaskT* mask)
{
    switch (this->getSpatialScheme()) {
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
    case math::UNKNOWN_BIAS:
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme, typename MaskT>
void
LevelSetTracker<GridT, InterruptT>::
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
LevelSetTracker<GridT, InterruptT>::
normalize2(const MaskT* mask)
{
    Normalizer<SpatialScheme, TemporalScheme, MaskT> tmp(*this, mask);
    tmp.normalize();
}


////////////////////////////////////////////////////////////////////////////


template<typename GridT, typename InterruptT>
template<lstrack::TrimMode Trimming>
void
LevelSetTracker<GridT, InterruptT>::Trim<Trimming>::trim()
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
LevelSetTracker<GridT, InterruptT>::Trim<Trimming>::operator()(const LeafRange& range) const
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
LevelSetTracker<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
Normalizer(LevelSetTracker& tracker, const MaskT* mask)
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
LevelSetTracker<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
normalize()
{
    namespace ph = std::placeholders;

    /// Make sure we have enough temporal auxiliary buffers
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
            this->cook("Normalizing level set using TVD_RK1 (step 1 of 2)", 1);

            // Convex combine explicit Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = std::bind(&Normalizer::euler12, ph::_1, ph::_2);

            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)
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
LevelSetTracker<GridT, InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
eval(StencilT& stencil, const ValueType* phi, ValueType* result, Index n) const
{
    using GradientT = typename math::ISGradientNormSqrd<SpatialScheme>;
    static const ValueType alpha = ValueType(Nominator)/ValueType(Denominator);
    static const ValueType beta  = ValueType(1) - alpha;

    const ValueType normSqGradPhi = GradientT::result(stencil);
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
LevelSetTracker<GridT,InterruptT>::
Normalizer<SpatialScheme, TemporalScheme, MaskT>::
euler(const LeafRange& range, Index phiBuffer, Index resultBuffer)
{
    mTracker.checkInterrupter();

    StencilT stencil(mTracker.grid());

    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        LeafStencilT::setLeaf(stencil, *leafIter);
        const ValueType* phi = leafIter.buffer(phiBuffer).data();
        ValueType* result = leafIter.buffer(resultBuffer).data();
        if (mMask == nullptr) {
            for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                this->eval<Nominator, Denominator>(stencil, phi, result, iter.pos());
            }//loop over active voxels in the leaf of the level set
        } else if (const MaskLeafT* mask = mMask->probeLeaf(leafIter->origin())) {
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

#ifdef OPENVDB_INSTANTIATE_LEVELSETTRACKER
#include <openvdb/util/ExplicitInstantiation.h>
#endif

OPENVDB_INSTANTIATE_CLASS LevelSetTracker<FloatGrid, util::NullInterrupter>;
OPENVDB_INSTANTIATE_CLASS LevelSetTracker<DoubleGrid, util::NullInterrupter>;

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_TRACKER_HAS_BEEN_INCLUDED
