// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Ken Museth
///
/// @file tools/Filter.h
///
/// @brief  Filtering of VDB volumes. All operations can optionally be masked
///         with another grid that acts as an alpha-mask. By default, filtering
///         operations do not modify the topology of the input tree and thus do
///         not process active tiles. However Filter::setProcessTiles can be
///         used to process active tiles, densifying them on demand when necessary.

#ifndef OPENVDB_TOOLS_FILTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_FILTER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/NodeManager.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/util/Util.h>
#include <openvdb/util/Assert.h>
#include <openvdb/thread/Threading.h>
#include "Interpolation.h"

#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>

#include <algorithm> // for std::max()
#include <functional>
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Volume filtering (e.g., diffusion) with optional alpha masking
template<typename GridT,
         typename MaskT = typename GridT::template ValueConverter<float>::Type,
         typename InterruptT = util::NullInterrupter>
class Filter
{
public:
    using GridType = GridT;
    using MaskType = MaskT;
    using TreeType = typename GridType::TreeType;
    using LeafType = typename TreeType::LeafNodeType;
    using ValueType = typename GridType::ValueType;
    using AlphaType = typename MaskType::ValueType;
    using LeafManagerType = typename tree::LeafManager<TreeType>;
    using RangeType = typename LeafManagerType::LeafRange;
    using BufferType = typename LeafManagerType::BufferType;
    static_assert(std::is_floating_point<AlphaType>::value,
        "openvdb::tools::Filter requires a mask grid with floating-point values");

    /// Constructor
    /// @param grid Grid to be filtered.
    /// @param interrupt Optional interrupter.
    Filter(GridT& grid, InterruptT* interrupt = nullptr)
        : mGrid(&grid)
        , mTask(nullptr)
        , mInterrupter(interrupt)
        , mMask(nullptr)
        , mGrainSize(1)
        , mMinMask(0)
        , mMaxMask(1)
        , mInvertMask(false)
        , mTiles(false) {}

    /// @brief Shallow copy constructor called by tbb::parallel_for()
    /// threads during filtering.
    /// @param other The other Filter from which to copy.
    Filter(const Filter& other)
        : mGrid(other.mGrid)
        , mTask(other.mTask)
        , mInterrupter(other.mInterrupter)
        , mMask(other.mMask)
        , mGrainSize(other.mGrainSize)
        , mMinMask(other.mMinMask)
        , mMaxMask(other.mMaxMask)
        , mInvertMask(other.mInvertMask)
        , mTiles(other.mTiles) {}

    /// @return the grain-size used for multi-threading
    int  getGrainSize() const { return mGrainSize; }
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grain size of 0 or less disables multi-threading!
    void setGrainSize(int grainsize) { mGrainSize = grainsize; }

    /// @return whether active tiles are being processed
    bool getProcessTiles() const { return mTiles; }
    /// @brief Set whether active tiles should also be processed.
    /// @note If true, some tiles may become voxelized
    /// @warning If using with a mask, ensure that the mask topology matches the
    /// tile topology of the filter grid as tiles will not respect overlapping
    /// mask values at tree levels finer than themselves e.g. a leaf level tile
    /// will only use the corresponding tile ijk value in the mask grid
    void setProcessTiles(bool flag) { mTiles = flag; }

    /// @brief Return the minimum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    AlphaType minMask() const { return mMinMask; }
    /// @brief Return the maximum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    AlphaType maxMask() const { return mMaxMask; }
    /// @brief Define the range for the (optional) scalar mask.
    /// @param min Minimum value of the range.
    /// @param max Maximum value of the range.
    /// @details Mask values outside the range are clamped to zero or one, and
    /// values inside the range map smoothly to 0->1 (unless the mask is inverted).
    /// @throw ValueError if @a min is not smaller than @a max.
    void setMaskRange(AlphaType min, AlphaType max)
    {
        if (!(min < max)) OPENVDB_THROW(ValueError, "Invalid mask range (expects min < max)");
        mMinMask = min;
        mMaxMask = max;
    }

    /// @brief Return true if the mask is inverted, i.e. min->max in the
    /// original mask maps to 1->0 in the inverted alpha mask.
    bool isMaskInverted() const { return mInvertMask; }
    /// @brief Invert the optional mask, i.e. min->max in the original
    /// mask maps to 1->0 in the inverted alpha mask.
    void invertMask(bool invert=true) { mInvertMask = invert; }

    /// @brief One iteration of a fast separable mean-value (i.e. box) filter.
    /// @param width The width of the mean-value filter is 2*width+1 voxels.
    /// @param iterations Number of times the mean-value filter is applied.
    /// @param mask Optional alpha mask.
    void mean(int width = 1, int iterations = 1, const MaskType* mask = nullptr);

    /// @brief One iteration of a fast separable Gaussian filter.
    ///
    /// @note This is approximated as 4 iterations of a separable mean filter
    /// which typically leads an approximation that's better than 95%!
    /// @param width The width of the mean-value filter is 2*width+1 voxels.
    /// @param iterations Number of times the mean-value filter is applied.
    /// @param mask Optional alpha mask.
    void gaussian(int width = 1, int iterations = 1, const MaskType* mask = nullptr);

    /// @brief One iteration of a median-value filter
    ///
    /// @note This filter is not separable and is hence relatively slow!
    /// @param width The width of the mean-value filter is 2*width+1 voxels.
    /// @param iterations Number of times the mean-value filter is applied.
    /// @param mask Optional alpha mask.
    void median(int width = 1, int iterations = 1, const MaskType* mask = nullptr);

    /// Offsets (i.e. adds) a constant value to all active voxels.
    /// @param offset Offset in the same units as the grid.
    /// @param mask Optional alpha mask.
    void offset(ValueType offset, const MaskType* mask = nullptr);

    /// @brief Used internally by tbb::parallel_for()
    /// @param range Range of LeafNodes over which to multi-thread.
    ///
    /// @warning Never call this method directly!
    void operator()(const RangeType& range) const
    {
        if (mTask) mTask(const_cast<Filter*>(this), range);
        else OPENVDB_THROW(ValueError, "task is undefined - call median(), mean(), etc.");
    }

private:
    using LeafT = typename TreeType::LeafNodeType;
    using VoxelIterT = typename LeafT::ValueOnIter;
    using VoxelCIterT = typename LeafT::ValueOnCIter;
    using BufferT = typename tree::LeafManager<TreeType>::BufferType;
    using LeafIterT = typename RangeType::Iterator;
    using AlphaMaskT = tools::AlphaMask<GridT, MaskT>;

    void cook(LeafManagerType& leafs);

    template<size_t Axis>
    struct Avg {
        Avg(const GridT* grid, Int32 w): acc(grid->tree()), width(w), frac(1.f/float(2*w+1)) {}
        inline ValueType operator()(Coord xyz);
        typename GridT::ConstAccessor acc;
        const Int32 width;
        const float frac;
    };

    // Private filter methods called by tbb::parallel_for threads
    template <typename AvgT>
    void doBox(const RangeType& r, Int32 w);
    void doBoxX(const RangeType& r, Int32 w) { this->doBox<Avg<0> >(r,w); }
    void doBoxY(const RangeType& r, Int32 w) { this->doBox<Avg<1> >(r,w); }
    void doBoxZ(const RangeType& r, Int32 w) { this->doBox<Avg<2> >(r,w); }
    void doMedian(const RangeType&, int);
    void doOffset(const RangeType&, ValueType);
    /// @return true if the process was interrupted
    bool wasInterrupted();

    GridType*        mGrid;
    typename std::function<void (Filter*, const RangeType&)> mTask;
    InterruptT*      mInterrupter;
    const MaskType*  mMask;
    int              mGrainSize;
    AlphaType        mMinMask, mMaxMask;
    bool             mInvertMask;
    bool             mTiles;
}; // end of Filter class


////////////////////////////////////////

/// @cond OPENVDB_DOCS_INTERNAL

namespace filter_internal {

template<typename TreeT>
struct Voxelizer
{
    // NodeManager for processing internal/root node values
    // @note  Should not cache leaf nodes
    using NodeManagerT = tree::NodeManager<TreeT, TreeT::RootNodeType::LEVEL-1>;
    using MaskT = typename TreeT::template ValueConverter<ValueMask>::Type;

    Voxelizer(TreeT& tree, const bool allNeighbors, const size_t grainSize)
        : mVoxelTopology()
        , mManager(nullptr)
        , mGrainSize(grainSize)
        , mOp(tree, mVoxelTopology, allNeighbors ? 26 : 6) {}

    /// @brief  Convert tiles to leaf nodes that exist at a particular
    ///         voxel distance away
    /// @param width  distance in voxels to seach for tiles from each leaf
    /// @return  Returns how many search iterations were performed, which
    ///          also represents how many leaf node neighbors may have been
    ///          created. Returns 0 if the tree is already entirely voxelized
    int run(const int width)
    {
        if (!mOp.tree().hasActiveTiles()) return 0;
        this->init();
        int count = 0;
        for (int i = 0; i < width; i += int(TreeT::LeafNodeType::DIM), ++count) {
            if (i > 0) mManager->rebuild();
            mManager->foreachBottomUp(mOp, mGrainSize > 0, mGrainSize);
            mOp.tree().topologyUnion(mVoxelTopology);
        }
        return count;
    }

private:
    void init()
    {
        if (mManager) {
            mManager->rebuild();
        }
        else {
            // @note  We don't actually need the leaf topology here, just the
            // internal node structure so that we can generate leaf nodes in parallel
            mVoxelTopology.topologyUnion(mOp.tree());
            mManager.reset(new NodeManagerT(mOp.tree()));
        }
    }

    struct CreateVoxelMask
    {
        using LeafT = typename TreeT::LeafNodeType;
        using RootT = typename TreeT::RootNodeType;

        CreateVoxelMask(TreeT& tree, MaskT& mask, const size_t NN)
            : mTree(tree), mVoxelTopology(mask), mNeighbors(NN) {}

        TreeT& tree() { return mTree; }

        // do nothing for leaf nodes. They shouldn't even be cached as
        // part of the NodeManager used with this method.
        void operator()(const LeafT&) const { OPENVDB_ASSERT(false); }

        void operator()(const RootT& node) const
        {
            using ChildT = typename RootT::ChildNodeType;
            static constexpr Int32 CHILDDIM = Int32(ChildT::DIM);
            static constexpr Int32 LEAFDIM = Int32(LeafT::DIM);
            const Tester op(mTree, mNeighbors);

            auto step =
                [&](const Coord& ijk,
                    const size_t axis1,
                    const size_t axis2,
                    const auto& val)
            {
                Coord offset(0);
                Int32& a = offset[axis1];
                Int32& b = offset[axis2];
                for (a = 0; a < CHILDDIM; a+=LEAFDIM) {
                    for (b = 0; b < CHILDDIM; b+=LEAFDIM) {
                        const Coord childijk = ijk + offset;
                        if (op.test(childijk, val)) {
                            mVoxelTopology.touchLeaf(childijk);
                        }
                    }
                }

                offset.reset(CHILDDIM-1);
                for (a = 0; a < CHILDDIM; a+=LEAFDIM) {
                    for (b = 0; b < CHILDDIM; b+=LEAFDIM) {
                        const Coord childijk = ijk + offset;
                        if (op.test(childijk, val)) {
                            mVoxelTopology.touchLeaf(childijk);
                        }
                    }
                }
            };

            for (auto iter = node.cbeginValueOn(); iter; ++iter) {
                const Coord& ijk = iter.getCoord();
                // @todo step only needs to search if a given direction
                // depending on the face
                step(ijk, 0, 1, *iter);
                step(ijk, 0, 2, *iter);
                step(ijk, 1, 2, *iter);
            }
        }

        template<typename NodeT>
        void operator()(const NodeT& node) const
        {
            using ChildT = typename NodeT::ChildNodeType;
            static constexpr Int32 CHILDDIM = Int32(ChildT::DIM);
            static constexpr Int32 LEAFDIM = Int32(LeafT::DIM);

            static auto step =
                [](const Tester& op,
                    const Coord& ijk,
                    const size_t axis1,
                    const size_t axis2,
                    const auto& val,
                    std::vector<Coord>& coords)
            {
                Coord offset(0);
                Int32& a = offset[axis1];
                Int32& b = offset[axis2];
                for (a = 0; a < CHILDDIM; a+=LEAFDIM) {
                    for (b = 0; b < CHILDDIM; b+=LEAFDIM) {
                        const Coord childijk = ijk + offset;
                        if (op.test(childijk, val)) {
                            coords.emplace_back(childijk);
                        }
                    }
                }

                offset.reset(CHILDDIM-1);
                for (a = 0; a < CHILDDIM; a+=LEAFDIM) {
                    for (b = 0; b < CHILDDIM; b+=LEAFDIM) {
                        const Coord childijk = ijk + offset;
                        if (op.test(childijk, val)) {
                            coords.emplace_back(childijk);
                        }
                    }
                }
            };

            /// Two types of algorithms here
            ///   1) For the case where this node is the direct parent of leaf nodes
            ///   2) For all other node types
            ///
            /// In general, given a tile's ijk, search its faces/edges/corners for
            /// values which differ from its own or leaf level topology. When a
            /// difference is detected, mask topology is generated which can be used
            /// with topologyUnion to ensure valid voxel values exist in the source
            /// grid.
            ///
            /// This operator handles all internal node types. For example, for the
            /// lowest level internal node (which contains leaf nodes as children)
            /// each tile is at the leaf level (a single tile represents an 8x8x8
            /// node). CHILDDIM is this case will match the valid of LEAFDIM, as we
            /// only need to check each tiles immediate neighbors. For higher level
            /// internal nodes (and the root node) each child tile will have a
            /// significantly larger CHILDDIM than the grid's LEAFDIM. We
            /// consistently probe values along the LEAFDIM stride to ensure no
            /// changes are missed.

            if (CHILDDIM == LEAFDIM) {
                // If the current node is the parent of leaf nodes, search each
                // neighbor directly and use a flag buffer to test offsets in
                // this node which need converting to leaf level topology.
                // This is faster than the more general method which steps across
                // faces (unecessary due to CHILDDIM == LEAFDIM) and provides
                // a simpler way of tracking new topology

                std::vector<char> flags(NodeT::NUM_VALUES, char(0));
                tbb::parallel_for(tbb::blocked_range<size_t>(0, NodeT::NUM_VALUES),
                    [&](const tbb::blocked_range<size_t>& range) {
                    const Tester op(mTree, mNeighbors);
                    for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                        if (node.isValueMaskOn(Index(n))) {
                            // if index is a tile, search its neighbors
                            const Coord ijk = node.offsetToGlobalCoord(Index(n));
                            flags[n] = op.test(ijk, node.getValue(ijk));
                        }
                    }
                });

                // create leaf level topology in this internal node
                Index idx = 0;
                for (auto iter = flags.begin(); iter != flags.end(); ++iter, ++idx) {
                    if (*iter) mVoxelTopology.touchLeaf(node.offsetToGlobalCoord(idx));
                }
            }
            else {
                // If this is a higher level internal node, we only need to search its
                // face/edge/vertex neighbors for values which differ or leaf level
                // topology. When a difference is detected, store the coordinate which
                // needs to be voxelized.
                // @todo investigate better threaded impl

                tbb::concurrent_vector<Coord> nodes;
                tbb::parallel_for(tbb::blocked_range<size_t>(0, NodeT::NUM_VALUES),
                    [&](const tbb::blocked_range<size_t>& range)
                {
                    const Tester op(mTree, mNeighbors);
                    std::vector<Coord> coords;

                    for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                        if (!node.isValueMaskOn(Index(n))) continue;

                        const Coord ijk = node.offsetToGlobalCoord(Index(n));
                        const auto& val = node.getValue(ijk);
                        // @todo step only needs to search if a given direction
                        // depending on the face
                        step(op, ijk, 0, 1, val, coords);
                        step(op, ijk, 0, 2, val, coords);
                        step(op, ijk, 1, 2, val, coords);
                    }

                    if (!coords.empty()) {
                        std::copy(coords.begin(), coords.end(),
                            nodes.grow_by(coords.size()));
                    }
                });

                // create leaf level topology in this internal node
                // @note  nodes may contain duplicate coords
                for (const auto& coord : nodes) {
                    mVoxelTopology.touchLeaf(coord);
                }
            }
        }

    private:
        struct Tester
        {
            Tester(const TreeT& tree, const size_t NN)
                : mAcc(tree), mNeighbors(NN) {}

            inline bool test(const Coord& ijk,
                const typename TreeT::ValueType& val) const
            {
                static constexpr Int32 LEAFDIM = Int32(LeafT::DIM);
                const Coord* NN = util::COORD_OFFSETS;
                for (size_t i = 0; i < mNeighbors; ++i, ++NN) {
                    Coord neighbor(*NN);
                    neighbor.x() *= LEAFDIM;
                    neighbor.y() *= LEAFDIM;
                    neighbor.z() *= LEAFDIM;
                    neighbor += ijk;
                    // if a leaf exists, assume its buffer is not constant
                    if (mAcc.getValue(neighbor) != val ||
                        mAcc.probeConstLeaf(neighbor)) {
                        return true;
                    }
                }
                return false;
            }
        private:
            const tree::ValueAccessor<const TreeT> mAcc;
            const size_t mNeighbors;
        };

    private:
        TreeT& mTree;
        MaskT& mVoxelTopology;
        const size_t mNeighbors;
    };// CreateVoxelMask

private:
    MaskT mVoxelTopology;
    std::unique_ptr<NodeManagerT> mManager;
    const size_t mGrainSize;
    CreateVoxelMask mOp;
};

// Helper function for Filter::Avg::operator()
template<typename T> static inline void accum(T& sum, T addend) { sum += addend; }
// Overload for bool ValueType
inline void accum(bool& sum, bool addend) { sum = sum || addend; }

} // namespace filter_internal

/// @endcond

////////////////////////////////////////


template<typename GridT, typename MaskT, typename InterruptT>
template<size_t Axis>
inline typename GridT::ValueType
Filter<GridT, MaskT, InterruptT>::Avg<Axis>::operator()(Coord xyz)
{
    ValueType sum = zeroVal<ValueType>();
    Int32 &i = xyz[Axis], j = i + width;
    for (i -= width; i <= j; ++i) filter_internal::accum(sum, acc.getValue(xyz));
    if constexpr(std::is_same<ValueType, bool>::value) {
        return sum && frac > 0.0f;
    } else {
        OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
        ValueType value = static_cast<ValueType>(sum * frac);
        OPENVDB_NO_TYPE_CONVERSION_WARNING_END
        return value;
    }
}


////////////////////////////////////////


template<typename GridT, typename MaskT, typename InterruptT>
void
Filter<GridT, MaskT, InterruptT>::mean(int width, int iterations, const MaskType* mask)
{
    if (iterations <= 0) return;
    mMask = mask;
    const int w = std::max(1, width);
    const bool serial = mGrainSize == 0;

    if (mInterrupter) mInterrupter->start("Applying mean filter");

    std::unique_ptr<filter_internal::Voxelizer<TreeType>> voxelizer;
    if (this->getProcessTiles()) {
        // if performing multiple iterations, also search edge/vertex
        // neighbors for difference topology.
        const bool allNeighbors = iterations > 1;
        // If processing tiles, create a voxelizer and run a single
        // width based search for tiles that need to be voxelized
        voxelizer.reset(new filter_internal::Voxelizer<TreeType>
            (mGrid->tree(), allNeighbors, mGrainSize));
        if (!voxelizer->run(w)) voxelizer.reset();
    }

    LeafManagerType leafs(mGrid->tree(), 1, serial);

    int iter = 1; // num of leaf level neighbor based searches performed
    int dist = w; // kernel distance of the current iteration
    for (int i=0; i<iterations && !this->wasInterrupted(); ++i, dist+=w) {
        if (i > 0 && voxelizer) {
            // the total influence distance in voxels of this iteration
            // minus how far we've already accounted for
            const int remain = dist - iter * int(TreeType::LeafNodeType::DIM);
            if (remain > 0) {
                const int searches = voxelizer->run(remain);
                if (searches == 0) voxelizer.reset();
                else               leafs.rebuild(serial);
                iter += searches;
            }
        }

        mTask = std::bind(&Filter::doBoxX, std::placeholders::_1, std::placeholders::_2, w);
        this->cook(leafs);
        // note that the order of the YZ passes are flipped to maintain backwards-compatibility
        // with an indexing typo in the original logic
        mTask = std::bind(&Filter::doBoxZ, std::placeholders::_1, std::placeholders::_2, w);
        this->cook(leafs);
        mTask = std::bind(&Filter::doBoxY, std::placeholders::_1, std::placeholders::_2, w);
        this->cook(leafs);
    }

    if (mInterrupter) mInterrupter->end();
}


template<typename GridT, typename MaskT, typename InterruptT>
void
Filter<GridT, MaskT, InterruptT>::gaussian(int width, int iterations, const MaskType* mask)
{
    if (iterations <= 0) return;
    mMask = mask;
    const int w = std::max(1, width);
    const bool serial = mGrainSize == 0;

    if (mInterrupter) mInterrupter->start("Applying Gaussian filter");

    std::unique_ptr<filter_internal::Voxelizer<TreeType>> voxelizer;
    if (this->getProcessTiles()) {
        // if performing multiple iterations, also search edge/vertex
        // neighbors for difference topology.
        const bool allNeighbors = iterations > 1;
        // If processing tiles, create a voxelizer and run a single
        // width based search for tiles that need to be voxelized
        // @note  account for sub iteration due to gaussian filter
        voxelizer.reset(new filter_internal::Voxelizer<TreeType>
            (mGrid->tree(), allNeighbors, mGrainSize));
        if (!voxelizer->run(w*4)) voxelizer.reset();
    }

    LeafManagerType leafs(mGrid->tree(), 1, serial);

    int iter = 1; // num of leaf level neighbor based searches performed
    int dist = w*4; // kernel distance of the current iteration
    for (int i=0; i<iterations; ++i, dist+=(w*4)) {
        if (i > 0 && voxelizer) {
            // the total influence distance in voxels of this iteration
            // minus how far we've already accounted for
            const int remain = dist - iter * int(TreeType::LeafNodeType::DIM);
            if (remain > 0) {
                const int searches = voxelizer->run(remain);
                if (searches == 0) voxelizer.reset();
                else               leafs.rebuild(serial);
                iter += searches;
            }
        }

        for (int n=0; n<4 && !this->wasInterrupted(); ++n) {
            mTask = std::bind(&Filter::doBoxX, std::placeholders::_1, std::placeholders::_2, w);
            this->cook(leafs);
            // note that the order of the YZ passes are flipped to maintain backwards-compatibility
            // with an indexing typo in the original logic
            mTask = std::bind(&Filter::doBoxZ, std::placeholders::_1, std::placeholders::_2, w);
            this->cook(leafs);
            mTask = std::bind(&Filter::doBoxY, std::placeholders::_1, std::placeholders::_2, w);
            this->cook(leafs);
        }
    }

    if (mInterrupter) mInterrupter->end();
}


template<typename GridT, typename MaskT, typename InterruptT>
void
Filter<GridT, MaskT, InterruptT>::median(int width, int iterations, const MaskType* mask)
{
    if (iterations <= 0) return;
    mMask = mask;
    const int w = std::max(1, width);
    const bool serial = mGrainSize == 0;

    if (mInterrupter) mInterrupter->start("Applying median filter");

    std::unique_ptr<filter_internal::Voxelizer<TreeType>> voxelizer;
    if (this->getProcessTiles()) {
        // If processing tiles, create a voxelizer and run a single
        // width based search for tiles that need to be voxelized
        voxelizer.reset(new filter_internal::Voxelizer<TreeType>
            (mGrid->tree(), /*allNeighbors*/true, mGrainSize));
        if (!voxelizer->run(w)) voxelizer.reset();
    }

    LeafManagerType leafs(mGrid->tree(), 1, serial);

    mTask = std::bind(&Filter::doMedian, std::placeholders::_1, std::placeholders::_2, w);

    int iter = 1; // num of leaf level neighbor based searches performed
    int dist = w; // kernel distance of the current iteration
    for (int i=0; i<iterations && !this->wasInterrupted(); ++i, dist+=w) {
        if (i > 0 && voxelizer) {
            // the total influence distance in voxels of this iteration
            // minus how far we've already accounted for
            const int remain = dist - iter * int(TreeType::LeafNodeType::DIM);
            if (remain > 0) {
                const int searches = voxelizer->run(remain);
                if (searches == 0) voxelizer.reset();
                else               leafs.rebuild(serial);
                iter += searches;
            }
        }

        this->cook(leafs);
    }

    if (mInterrupter) mInterrupter->end();
}


template<typename GridT, typename MaskT, typename InterruptT>
void
Filter<GridT, MaskT, InterruptT>::offset(ValueType value, const MaskType* mask)
{
    mMask = mask;

    if (mInterrupter) mInterrupter->start("Applying offset");

    if (this->getProcessTiles()) {
        // Don't process leaf nodes with the node manager - we'll do them
        // separately to allow for cleaner branching
        using NodeManagerT = tree::NodeManager<TreeType, TreeType::RootNodeType::LEVEL-1>;
        NodeManagerT manager(mGrid->tree());

        if (mask) {
            manager.foreachBottomUp([&](auto& node) {
                this->wasInterrupted();
                AlphaMaskT alpha(*mGrid, *mMask, mMinMask, mMaxMask, mInvertMask);
                typename AlphaMaskT::FloatType a, b;
                for (auto iter = node.beginValueOn(); iter; ++iter) {
                    if (!alpha(iter.getCoord(), a, b)) continue;
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
                    iter.modifyValue([&](ValueType& v) { v += a*value; });
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_END
                }
            });
        }
        else {
            manager.foreachBottomUp([&](auto& node) {
                this->wasInterrupted();
                for (auto iter = node.beginValueOn(); iter; ++iter) {
                    iter.modifyValue([&](ValueType& v) { v += value; });
                }
            });
        }
    }

    LeafManagerType leafs(mGrid->tree(), 0, mGrainSize==0);
    mTask = std::bind(&Filter::doOffset, std::placeholders::_1, std::placeholders::_2, value);
    this->cook(leafs);

    if (mInterrupter) mInterrupter->end();
}


////////////////////////////////////////


/// Private method to perform the task (serial or threaded) and
/// subsequently swap the leaf buffers.
template<typename GridT, typename MaskT, typename InterruptT>
void
Filter<GridT, MaskT, InterruptT>::cook(LeafManagerType& leafs)
{
    if (mGrainSize>0) {
        tbb::parallel_for(leafs.leafRange(mGrainSize), *this);
    } else {
        (*this)(leafs.leafRange());
    }
    leafs.swapLeafBuffer(1, mGrainSize==0);
}


/// One dimensional convolution of a separable box filter
template<typename GridT, typename MaskT, typename InterruptT>
template <typename AvgT>
void
Filter<GridT, MaskT, InterruptT>::doBox(const RangeType& range, Int32 w)
{
    this->wasInterrupted();
    AvgT avg(mGrid, w);
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(*mGrid, *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                const Coord xyz = iter.getCoord();
                if (alpha(xyz, a, b)) {
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
                    const ValueType value(b*(*iter) + a*avg(xyz));
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_END
                    buffer.setValue(iter.pos(), value);
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                buffer.setValue(iter.pos(), avg(iter.getCoord()));
            }
        }
    }
}


/// Performs simple but slow median-value diffusion
template<typename GridT, typename MaskT, typename InterruptT>
void
Filter<GridT, MaskT, InterruptT>::doMedian(const RangeType& range, int width)
{
    this->wasInterrupted();
    typename math::DenseStencil<GridType> stencil(*mGrid, width);//creates local cache!
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(*mGrid, *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    stencil.moveTo(iter);
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
                    ValueType value(b*(*iter) + a*stencil.median());
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_END
                    buffer.setValue(iter.pos(), value);
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                buffer.setValue(iter.pos(), stencil.median());
            }
        }
    }
}


/// Offsets the values by a constant
template<typename GridT, typename MaskT, typename InterruptT>
void
Filter<GridT, MaskT, InterruptT>::doOffset(const RangeType& range, ValueType offset)
{
    this->wasInterrupted();
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(*mGrid, *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            for (VoxelIterT iter = leafIter->beginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
                    ValueType value(*iter + a*offset);
                    OPENVDB_NO_TYPE_CONVERSION_WARNING_END
                    iter.setValue(value);
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            for (VoxelIterT iter = leafIter->beginValueOn(); iter; ++iter) {
                iter.setValue(*iter + offset);
            }
        }
    }
}


template<typename GridT, typename MaskT, typename InterruptT>
inline bool
Filter<GridT, MaskT, InterruptT>::wasInterrupted()
{
    if (util::wasInterrupted(mInterrupter)) {
        thread::cancelGroupExecution();
        return true;
    }
    return false;
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_FILTER
#include <openvdb/util/ExplicitInstantiation.h>
#endif

OPENVDB_INSTANTIATE_CLASS Filter<FloatGrid, FloatGrid, util::NullInterrupter>;
OPENVDB_INSTANTIATE_CLASS Filter<DoubleGrid, FloatGrid, util::NullInterrupter>;

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_FILTER_HAS_BEEN_INCLUDED
