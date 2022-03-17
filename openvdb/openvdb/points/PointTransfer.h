// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Nick Avramoussis
///
/// @file PointTransfer.h
///
/// @brief  Framework methods for rasterizing PointDataGrid data to Trees.
///
/// @details  Provides a generic inherited interface for deriving transfer
///   schemes that represent how point data should be rasterized. The provided
///   components together support the transfer of multiple attributes to
///   arbitrary and multiple grid types. Target grids must have the same
///   transform, but this transform can differ from the source PointDataGrid
///   (multiple instantiations of rasterize() should instead be invoked to
///   transfer to grids of different transforms). Arbitrary attributes can be
///   accessed and transfered to arbitrary trees.
///

#ifndef OPENVEB_POINTS_TRANSFER_HAS_BEEN_INCLUDED
#define OPENVEB_POINTS_TRANSFER_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/thread/Threading.h>

#include <type_traits>
#include <tuple>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @par A transfer scheme must be configured to call the provided
///   rasterize methods. See below for an example or
///   PointRasterizeSDF.h/PointRasterizeTrilinear.h for implementations.
/// @code
/// struct Transfer
/// {
///     /// @return Returns the tree topology to loop over. This can be different
///     ///   from the destination tree i.e. This can act as a mask.
///     inline auto& topology();
///
///     /// @brief  The maximum lookup range of this transfer scheme in index
///     ///   space of the source points.
///     /// @details The return value represent how far away from the destination
///     ///   leaf node points should be accessed.
///     /// @param origin  The leaf origin of the topology being accessed
///     /// @param idx     The leaf index of the topology being accessed
///     inline Int32 range(const Coord& origin, size_t idx) const;
///
///     /// @brief  The initialize function, called on each leaf which has valid
///     ///   topology to write to.
///     /// @param origin  The leaf origin of the topology being accessed
///     /// @param idx     The leaf index of the topology being accessed
///     /// @param bounds  The active voxel bounds of the leaf
///     inline void initialize(const Coord& origin, size_t idx, const CoordBBox& bounds);
///
///     /// @brief  Run each time a point leaf is accessed. Typically this is
///     ///  where attribute handles can be constructed
///     /// @param leaf  The PointDataLeafNode which is being accessed.
///     /// @return  Return true to continue rasterization, false to early exit
///     ///   and skip the current leaf's contribution to the destination volume.
///     inline bool startPointLeaf(const PointDataTree::LeafNodeType& leaf);
///
///     /// @brief  The point stamp function. Each point which contributes to
///     ///  the current leaf will call this function exactly once.
///     /// @param ijk  The current voxel containing the point being rasterized.
///     ///   May be outside the destination leaf node depending on the range()
///     /// @param id   The point index being rasterized
///     /// @param bounds  The active bounds of the leaf node.
///     void rasterizePoint(const Coord& ijk,
///                     const Index id,
///                     const CoordBBox& bounds);
///
///     /// @brief  Run each time a point leaf is finished with.
///     /// @param leaf  The PointDataLeafNode which was being accessed.
///     /// @return  Return true to continue rasterization, false to early exit
///     ///   and stop rasterization to the destination leaf node.
///     inline bool endPointLeaf(const PointDataTree::LeafNodeType& leaf);
///
///     /// @brief  The finalization function for the given destination tree(s).
///     /// @param origin  The leaf origin of the topology being accessed
///     /// @param idx     The leaf index of the topology being accessed
///     /// @return  Return true to stop, false to recursively rasterize
///     inline bool finalize(const Coord& origin, size_t idx);
/// };
/// @endcode
///
///
/// Below is a full example using the native components.
///
/// @code
/// /// @brief Sum point distances into a target float tree
/// ///   Note: Using TransformTransfer to handle different index spaces, and
/// ///   VolumeTransfer for automatic buffer setup
/// struct MyTransfer :
///     public TransformTransfer,
///     public VolumeTransfer<FloatTree>
/// {
///     MyTransfer(FloatGrid& dest, const PointDataGrid& source)
///         : TransformTransfer(source.transform(), dest.transform())
///         , VolumeTransfer(dest.tree())
///         , mHandle(nullptr) {}
///
///     MyTransfer(const MyTransfer& other)
///         : TransformTransfer(other)
///         , VolumeTransfer(other)
///         , mHandle(nullptr) {}
///
///     /// @brief Range in index space of the source points
///     Int32 range(const Coord&, size_t) const { return Int32(1); }
///
///     /// @brief Every time we start a new point leaf, init the position array.
///     ///   Always return true as we don't skip any leaf nodes.
///     bool startPointLeaf(const PointDataTree::LeafNodeType& leaf)
///     {
///         mHandle.reset(new AttributeHandle<Vec3f>(leaf.constAttributeArray("P"));
///         return true;
///     }
///
///     /// @brief  For each point, compute its relative index space position in
///     ///   the destination tree and sum the length of its distance
///     void rasterizePoint(const Coord& ijk, const Index id, const CoordBBox& bounds)
///     {
///         Vec3d P = ijk.asVec3d() + Vec3d(this->mHandle->get(id));
///         P = this->transformSourceToTarget(P); // TransformTransfer::transformSourceToTarget
///         // for each active voxel, accumulate distance
///         const auto* mask = this->mask(); // VolumeTransfer::mask
///         for (auto& coord : bounds) {
///             const Index voxel = FloatTree::LeafNodeType::coordToOffset(coord);
///             if (!mask->isOn(voxel)) continue;
///             Vec3d dist = coord.asVec3d() - P;
///             this->buffer()[voxel] += dist.length(); // VolumeTransfer::buffer
///         }
///     }
///
///     /// @brief Return true for endPointLeaf() to continue, false for finalize() so
///     ///   we don't recurse.
///     bool endPointLeaf(const PointDataTree::LeafNodeType&) { return true; }
///     bool finalize(const Coord&, size_t) { return false; }
///
/// private:
///     std::unique_ptr<AttributeHandle<Vec3f>> mHandle;
/// };
/// @endcode


/// @brief Perform potentially complex rasterization from a user defined
///  transfer scheme.
/// @details The method works by looping over a single Tree topology, looking
///   up point data at a position relative to that topology and passing that
///   data to a transfer scheme TransferT.
/// @note  Each thread receives a copy of the transfer scheme object.
/// @param points       the point data grid to rasterize
/// @param transfer     the transfer scheme
/// @param filter       optional point filter
/// @param interrupter  optional interrupter
template <typename PointDataTreeOrGridT,
    typename TransferT,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
inline void
rasterize(const PointDataTreeOrGridT& points,
          TransferT& transfer,
          const FilterT& filter = NullFilter(),
          InterrupterT* interrupter = nullptr);


///////////////////////////////////////////////////

/// @brief  The TransformTransfer module should be used if the source transform
///   of the input points and the target transforms of the destination volumes
///   differ. The default rasterizer will skip index to world (and vice versa)
///   transformations unless a transfer scheme derives from a TransformTransfer.
struct TransformTransfer
{
    TransformTransfer(const math::Transform& st,
        const math::Transform& tt)
        : mSourceTransform(st)
        , mTargetTransform(tt) {}

    template <typename T>
    inline auto transformSourceToTarget(const T& value) const
    {
        const auto result = mSourceTransform.indexToWorld(value);
        return mTargetTransform.worldToIndex(result);
    }

    template <typename T>
    inline auto transformTargetToSource(const T& value) const
    {
        const auto result = mTargetTransform.indexToWorld(value);
        return mSourceTransform.worldToIndex(result);
    }

    const math::Transform& sourceTransform() const { return mSourceTransform; }
    const math::Transform& targetTransform() const { return mTargetTransform; }

private:
    const math::Transform& mSourceTransform;
    const math::Transform& mTargetTransform;
};

/// @brief  The VolumeTransfer module provides methods to automatically setup
///   and access destination buffers for multiple target volumes of arbitrary
///   types. Deriving from a VolumeTransfer ensures that the available
///   buffers correlate to the order of the provided tree arguments.
template <typename ...TreeTypes>
struct VolumeTransfer
{
    static const size_t Size = sizeof...(TreeTypes);
    using TreeTupleT = std::tuple<TreeTypes*...>;

    template <size_t Idx> using TreeType = typename std::tuple_element<Idx, std::tuple<TreeTypes...>>::type;
    template <size_t Idx> using ValueType = typename TreeType<Idx>::ValueType;
    template <typename T> struct TypeResolver { using Type = typename T::ValueType; };
    using NodeMaskT = typename TreeType<0>::LeafNodeType::NodeMaskType;

    VolumeTransfer(TreeTypes*... trees);

    VolumeTransfer(TreeTypes&... trees)
        : VolumeTransfer(&trees...) {}

    VolumeTransfer(const VolumeTransfer& other)
        : mTreeArray(other.mTreeArray)
        , mBuffers()
        , mMasks()
    {
        mBuffers.fill(nullptr);
        mMasks.fill(nullptr);
    }

    inline TreeType<0>& topology() { return *(std::get<0>(mTreeArray)); }

    inline void initialize(const Coord& origin, const size_t, const CoordBBox&);

    template <size_t Idx>
    inline ValueType<Idx>* buffer()
    {
        return static_cast<ValueType<Idx>*>(mBuffers[Idx]);
    }

    template <size_t Idx>
    inline const ValueType<Idx>* buffer() const
    {
        return static_cast<ValueType<Idx>*>(mBuffers[Idx]);
    }

    template <size_t Idx>
    inline NodeMaskT* mask() { return mMasks[Idx]; }
    inline NodeMaskT* mask(const size_t idx) { return mMasks[idx]; }

    template <size_t Idx>
    inline const NodeMaskT* mask() const { return mMasks[Idx]; }
    inline const NodeMaskT* mask(const size_t idx) const { return mMasks[idx]; }

    template <typename FunctorT>
    inline void foreach(const FunctorT& functor);

private:
    const TreeTupleT mTreeArray;
    std::array<void*, Size> mBuffers;
    std::array<NodeMaskT*, Size> mMasks;
};

/// @brief  VolumeTransfer specialization for a single target volume
/// @todo this specialization should avoid the probe
template <typename TreeT>
struct VolumeTransfer<TreeT>
{
    using TreeType = TreeT;
    using ValueType = typename TreeType::ValueType;
    using NodeMaskT = typename TreeType::LeafNodeType::NodeMaskType;

    static_assert(std::is_base_of<TreeBase, TreeType>::value,
        "One or more template arguments to VolumeTransfer "
        "are not a valid openvdb::Tree type.");

    VolumeTransfer(TreeType* tree)
        : mTree(tree)
        , mBuffer(nullptr)
        , mMask(nullptr) {
        assert(tree);
    }

    VolumeTransfer(TreeType& tree)
        : VolumeTransfer(&tree) {}

    VolumeTransfer(const VolumeTransfer& other)
        : mTree(other.mTree)
        , mBuffer(nullptr)
        , mMask(nullptr) {}

    inline TreeType& topology() { return *mTree; }

    inline void initialize(const Coord& origin, const size_t, const CoordBBox&)
    {
        assert(mTree);
        if (auto leaf = mTree->probeLeaf(origin)) {
            mBuffer = leaf->buffer().data();
            mMask = &(leaf->getValueMask());
        }
        else {
            mBuffer = nullptr;
            mMask = nullptr;
        }
    }

    inline ValueType* buffer() { return mBuffer; }
    inline const ValueType* buffer() const { return mBuffer; }
    inline NodeMaskT* mask() { return mMask; }
    inline const NodeMaskT* mask() const { return mMask; }

    // compatibility with multi tree containers
    template <size_t> inline ValueType* buffer() { return this->buffer(); }
    template <size_t> inline const ValueType* buffer() const { return this->buffer(); }
    template <size_t> inline NodeMaskT* mask() { return this->mask(); }
    template <size_t> inline const NodeMaskT* mask() const { return this->mask(); }

private:
    TreeType* const mTree;
    ValueType* mBuffer;
    NodeMaskT* mMask;
};

namespace transfer_internal
{
template<typename T, typename F, size_t... Is>
void foreach(T&& t, const F& func, std::integer_sequence<size_t, Is...>)
{
    auto init = { (func(std::get<Is>(t), Is), 0)... };
    (void)init;
}

template<typename T, typename F, size_t... Is>
void foreach(void** buffers, const F& func, std::integer_sequence<size_t, Is...>)
{
    int init[sizeof...(Is)] = {
        (func(static_cast<typename std::tuple_element<Is, T>::type*>
            (*(buffers + Is)), Is), 0)...
    };
}

template<typename T, template <typename> class R, typename F, size_t... Is>
void foreach(void** buffers, const F& func, std::integer_sequence<size_t, Is...>)
{
    int init[sizeof...(Is)] = {
        (func(static_cast<typename R<typename std::tuple_element<Is, T>::type>::Type*>
           (*(buffers + Is)), Is), 0)...
    };
}
}

template <typename ...TreeTypes>
VolumeTransfer<TreeTypes...>::VolumeTransfer(TreeTypes*... trees)
    : mTreeArray({ trees... })
    , mBuffers()
    , mMasks()
{
    transfer_internal::foreach(mTreeArray, [](auto&& tree, const size_t) {
        using TreeT = typename std::remove_pointer<typename std::decay<decltype(tree)>::type>::type;
        static_assert(std::is_base_of<TreeBase, TreeT>::value,
            "One or more template arguments to VolumeTransfer "
            "are not a valid openvdb::Tree type.");
        assert(tree);
    }, std::make_integer_sequence<size_t, Size>());

    mBuffers.fill(nullptr);
    mMasks.fill(nullptr);
}

template <typename ...TreeTypes>
inline void VolumeTransfer<TreeTypes...>::initialize(const Coord& origin, const size_t, const CoordBBox&)
{
    transfer_internal::foreach(mTreeArray,
        [&](auto&& tree, const size_t i) {
            assert(tree);
            if (auto leaf = tree->probeLeaf(origin)) {
                mBuffers[i] = static_cast<void*>(leaf->buffer().data());
                mMasks[i] = &(leaf->getValueMask());
            }
            else {
                mBuffers[i] = nullptr;
                mMasks[i] = nullptr;
            }
        }, std::make_integer_sequence<size_t, Size>());
}

template <typename ...TreeTypes>
template <typename FunctorT>
inline void VolumeTransfer<TreeTypes...>::foreach(const FunctorT& functor)
{
    transfer_internal::foreach<TreeTupleT, TypeResolver>(mBuffers.data(), functor,
        std::make_integer_sequence<size_t, Size>());
}

namespace transfer_internal
{
template <typename TransferT,
          typename TopologyT,
          typename PointFilterT = points::NullFilter,
          typename InterrupterT = util::NullInterrupter>
struct RasterizePoints
{
    using LeafManagerT = tree::LeafManager<TopologyT>;
    using LeafNodeT = typename LeafManagerT::LeafNodeType;

    static const Index DIM = TopologyT::LeafNodeType::DIM;
    static const Int32 DIM32 = static_cast<Int32>(DIM);
    static const Index LOG2DIM = TopologyT::LeafNodeType::LOG2DIM;

    RasterizePoints(const points::PointDataTree& tree,
                    const TransferT& transfer,
                    const PointFilterT& filter = PointFilterT(),
                    InterrupterT* interrupter = nullptr)
        : mPointAccessor(tree)
        , mTransfer(transfer)
        , mFilter(filter)
        , mInterrupter(interrupter) {}

    void operator()(LeafNodeT& leaf, const size_t idx) const
    {
        if (util::wasInterrupted(mInterrupter)) {
            thread::cancelGroupExecution();
            return;
        }

        const Coord& origin = leaf.origin();
        auto& mask = leaf.getValueMask();

        CoordBBox bounds;

        bool state;
        if (mask.isConstant(state)) {
            if (!state) return; // all inactive
            else bounds = leaf.getNodeBoundingBox();
        }
        else {
            // Use evalActiveBoundingBox over getNodeBoundingBox()
            // to get a better approximation
            leaf.evalActiveBoundingBox(bounds);
            assert(!bounds.empty());
        }

        mTransfer.initialize(origin, idx, bounds);

        CoordBBox search = bounds.expandBy(mTransfer.range(origin, idx));
        this->transform<>(search);

        // start the iteration from a leaf origin
        const Coord min = (search.min() & ~(DIM-1));
        const Coord& max = search.max();
        PointFilterT localFilter(mFilter);

        // loop over overlapping leaf nodes
        Coord leafOrigin;
        for (leafOrigin[0] = min[0]; leafOrigin[0] <= max[0]; leafOrigin[0]+=DIM32) {
            for (leafOrigin[1] = min[1]; leafOrigin[1] <= max[1]; leafOrigin[1]+=DIM32) {
                for (leafOrigin[2] = min[2]; leafOrigin[2] <= max[2]; leafOrigin[2]+=DIM32) {

                    // if no overlap, continue
                    CoordBBox pbox = CoordBBox::createCube(leafOrigin, DIM32);
                    pbox.intersect(search);
                    if (pbox.empty()) continue;

                    // if no points, continue
                    const auto* pointLeaf = mPointAccessor.probeConstLeaf(leafOrigin);
                    if (!pointLeaf) continue;
                    if (!mTransfer.startPointLeaf(*pointLeaf)) continue;
                    localFilter.reset(*pointLeaf);

                    // loop over point voxels which contribute to this leaf
                    const Coord& pmin(pbox.min());
                    const Coord& pmax(pbox.max());
                    for (Coord ijk = pmin; ijk.x() <= pmax.x(); ++ijk.x()) {
                        const Index i = ((ijk.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
                        for (ijk.y() = pmin.y(); ijk.y() <= pmax.y(); ++ijk.y()) {
                            const Index ij = i + ((ijk.y() & (DIM-1u)) << LOG2DIM);
                            for (ijk.z() = pmin.z(); ijk.z() <= pmax.z(); ++ijk.z()) {
                                // voxel should be in this points leaf
                                assert((ijk & ~(DIM-1u)) == leafOrigin);
                                const Index index = ij + /*k*/(ijk.z() & (DIM-1u));
                                const Index end = pointLeaf->getValue(index);
                                Index id = (index == 0) ? 0 : Index(pointLeaf->getValue(index - 1));
                                for (; id < end; ++id) {
                                    if (!localFilter.valid(&id)) continue;
                                    mTransfer.rasterizePoint(ijk, id, bounds);
                                } //point idx
                            }
                        }
                    } // outer point voxel

                    if (!mTransfer.endPointLeaf(*pointLeaf)) {
                        // rescurse if necessary
                        if (!mTransfer.finalize(origin, idx)) {
                            this->operator()(leaf, idx);
                        }
                        return;
                    }
                }
            }
        } // outer leaf node

        // rescurse if necessary
        if (!mTransfer.finalize(origin, idx)) {
            this->operator()(leaf, idx);
        }
    }

    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {
            (*this)(*leaf, leaf.pos());
        }
    }

private:

    template <typename EnableT = TransferT>
    typename std::enable_if<std::is_base_of<TransformTransfer, EnableT>::value>::type
    transform(CoordBBox& bounds) const
    {
        const TransformTransfer* transform =
            static_cast<TransformTransfer*>(&mTransfer);
        const BBoxd bbox(bounds.min().asVec3d(), bounds.max().asVec3d());
        bounds = transform->sourceTransform().worldToIndexCellCentered(
            transform->targetTransform().indexToWorld(bbox));
    }

    template <typename EnableT = TransferT>
    typename std::enable_if<!std::is_base_of<TransformTransfer, EnableT>::value>::type
    transform(CoordBBox&) const {}

private:
    const PointDataGrid::ConstAccessor mPointAccessor;
    mutable TransferT mTransfer;
    const PointFilterT& mFilter;
    InterrupterT* mInterrupter;
};

} // namespace transfer_internal

///////////////////////////////////////////////////
///////////////////////////////////////////////////

template <typename PointDataTreeOrGridT,
    typename TransferT,
    typename FilterT,
    typename InterrupterT>
inline void
rasterize(const PointDataTreeOrGridT& points,
          TransferT& transfer,
          const FilterT& filter,
          InterrupterT* interrupter)
{
    using PointTreeT = typename TreeAdapter<PointDataTreeOrGridT>::TreeType;
    static_assert(std::is_base_of<TreeBase, PointTreeT>::value,
        "Provided points to rasterize is not a derived TreeBase type.");

    const auto& tree = TreeAdapter<PointDataTreeOrGridT>::tree(points);

    auto& topology = transfer.topology();
    using TreeT = typename std::decay<decltype(topology)>::type;
    tree::LeafManager<TreeT> manager(topology);
    transfer_internal::RasterizePoints<TransferT, TreeT, FilterT, InterrupterT>
        raster(tree, transfer, filter, interrupter);
    manager.foreach(raster);
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif //OPENVEB_POINTS_TRANSFER_HAS_BEEN_INCLUDED
