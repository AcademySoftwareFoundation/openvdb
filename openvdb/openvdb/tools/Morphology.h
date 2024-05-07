// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file   Morphology.h
///
/// @authors Ken Museth, Nick Avramoussis
///
/// @brief  Implementation of morphological dilation and erosion.
///
/// @note   By design the morphological operations only change the
///         state of voxels, not their values. If one desires to
///         change the values of voxels that change state an efficient
///         technique is to construct a boolean mask by performing a
///         topology difference between the original and final grids.

#ifndef OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

#include "Activate.h" // backwards compatibility
#include "Prune.h"
#include "ValueTransformer.h"

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/util/Assert.h>

#include <tbb/task_arena.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Voxel topology of nearest neighbors
/// @details
/// <dl>
/// <dt><b>NN_FACE</b>
/// <dd>face adjacency (6 nearest neighbors, defined as all neighbor
/// voxels connected along one of the primary axes)
///
/// <dt><b>NN_FACE_EDGE</b>
/// <dd>face and edge adjacency (18 nearest neighbors, defined as all
/// neighbor voxels connected along either one or two of the primary axes)
///
/// <dt><b>NN_FACE_EDGE_VERTEX</b>
/// <dd>face, edge and vertex adjacency (26 nearest neighbors, defined
/// as all neighbor voxels connected along either one, two or all
/// three of the primary axes)
/// </dl>
enum NearestNeighbors { NN_FACE = 6, NN_FACE_EDGE = 18, NN_FACE_EDGE_VERTEX = 26 };

/// @brief Different policies when dilating trees with active tiles
/// @details
/// <dl>
/// <dt><b>IGNORE_TILES</b>
/// <dd>Active tiles are ignores. For dilation, only active voxels are
/// dilated. For erosion, active tiles still appear as neighboring
/// activity however will themselves not be eroded.
///
/// <dt><b>EXPAND_TILES</b>
/// <dd>For dilation and erosion, active tiles are voxelized (expanded),
/// dilated or eroded and left in their voxelized state irrespective of
/// their final state.
///
/// <dt><b>PRESERVE_TILES</b>
/// <dd>For dilation, active tiles remain unchanged but they still
/// contribute to the dilation as if they were active voxels. For
/// erosion, active tiles are only eroded should the erosion wavefront
/// reach them, otherwise they are left unchanged. Additionally, dense
/// or empty nodes with constant values are pruned.
/// </dl>
enum TilePolicy { IGNORE_TILES, EXPAND_TILES, PRESERVE_TILES };

/// @brief Topologically dilate all active values (i.e. both voxels
///   and tiles) in a tree using one of three nearest neighbor
///   connectivity patterns.
/// @details If the input is *not* a MaskTree OR if tiles are being
///   preserved, this algorithm will copy the input tree topology onto a
///   MaskTree, performs the dilation on the mask and copies the resulting
///   topology back. This algorithm guarantees topology preservation
///   (non-pruned leaf nodes will persists) EXCEPT for direct MaskTree
///   dilation. MaskTree dilation is optimised for performance and may
///   replace existing leaf nodes i.e. any held leaf node pointers may
///   become invalid. See the Morphology class for more granular control.
/// @note This method is fully multi-threaded and support active tiles,
///   however only the PRESERVE_TILES policy ensures a pruned topology.
///   The values of any voxels are unchanged.
///
/// @param tree          tree or leaf manager to be dilated. The leaf
///                      manager will be synchronized with the result.
/// @param iterations    number of iterations to apply the dilation
/// @param nn            connectivity pattern of the dilation: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
/// @param mode          Defined the policy for handling active tiles
///                      (see above for details)
/// @param threaded      Whether to multi-thread execution
template<typename TreeOrLeafManagerT>
void dilateActiveValues(TreeOrLeafManagerT& tree,
    const int iterations = 1,
    const NearestNeighbors nn = NN_FACE,
    const TilePolicy mode = PRESERVE_TILES,
    const bool threaded = true);

/// @brief Topologically erode all active values (i.e. both voxels
///   and tiles) in a tree using one of three nearest neighbor
///   connectivity patterns.
/// @details If tiles are being preserve, this algorithm will copy the input
///   tree topology onto a MaskTree, performs the erosion on the mask and
///   intersects the resulting topology back. This algorithm guarantees
///   topology preservation (non-pruned leaf nodes will persists). See the
///   Morphology class for more granular control.
/// @note This method is fully multi-threaded and support active tiles,
///   however only the PRESERVE_TILES policy ensures a pruned topology.
///   The values of any voxels are unchanged. Erosion by NN_FACE neighbors
///   is usually faster than other neighbor schemes. NN_FACE_EDGE and
///   NN_FACE_EDGE_VERTEX operate at comparable dilation speeds.
///
/// @param tree          tree or leaf manager to be eroded. The leaf
///                      manager will be synchronized with the result.
/// @param iterations    number of iterations to apply the erosion
/// @param nn            connectivity pattern of the erosion: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
/// @param mode          Defined the policy for handling active tiles
///                      (see above for details)
/// @param threaded      Whether to multi-thread execution
template<typename TreeOrLeafManagerT>
void erodeActiveValues(TreeOrLeafManagerT& tree,
    const int iterations = 1,
    const NearestNeighbors nn = NN_FACE,
    const TilePolicy mode = PRESERVE_TILES,
    const bool threaded = true);


////////////////////////////////////////


namespace morphology {

/// @brief  Dilation/Erosion operations over a Trees leaf level voxel topology.
template<typename TreeType>
class Morphology
{
public:
    using LeafType = typename TreeType::LeafNodeType;
    using MaskType = typename LeafType::NodeMaskType;
    using ValueType = typename TreeType::ValueType;
    using MaskTreeT = typename TreeType::template ValueConverter<ValueMask>::Type;
    using MaskLeafT = typename MaskTreeT::LeafNodeType;
    using AccessorType = tree::ValueAccessor<TreeType>;

    Morphology(TreeType& tree)
        : mManagerPtr(new tree::LeafManager<TreeType>(tree))
        , mManager(*mManagerPtr)
        , mThreaded(true) {}

    Morphology(tree::LeafManager<TreeType>& tree)
        : mManagerPtr(nullptr)
        , mManager(tree)
        , mThreaded(true) {}

    /// @brief  Return whether this class is using multi-threading.
    bool getThreaded() const { return mThreaded; }
    /// @brief Set whether to use multi-threading.
    /// @note  The grain size is not exposed
    inline void setThreaded(const bool threaded) { mThreaded = threaded; }

    /// @brief  Return a const reference to the leaf manager
    inline const tree::LeafManager<TreeType>& leafManager() const { return mManager; }

    /// @brief Topologically erode all voxels by the provided nearest neighbor
    ///    scheme and optionally collapse constant leaf nodes
    /// @details Inactive Tiles contribute to the erosion but active tiles are
    ///    not modified.
    /// @param iter Number of erosion iterations
    /// @param nn Connectivity pattern of the erosion
    /// @param prune Whether to collapse constant leaf nodes after the erosion
    void erodeVoxels(const size_t iter,
        const NearestNeighbors nn,
        const bool prune = false);

    /// @brief Topologically dilate all voxels by the provided nearest neighbor
    ///    scheme and optionally collapse constant leaf nodes
    /// @details Voxel values are unchanged and only leaf nodes are used to
    ///    propagate the dilation.
    /// @param iter Number of dilation iterations
    /// @param nn Connectivity pattern of the dilation
    /// @param prune Whether to collapse constant leaf nodes after the dilation
    /// @param preserveMaskLeafNodes When dilating mask trees, the default behaviour
    ///    chooses to steal the mask nodes rather than copy them. Although faster,
    ///    this means that leaf nodes may be re-allocated. Set this to true if you
    ///    need the original topology pointers to be preserved.
    void dilateVoxels(const size_t iter,
        const NearestNeighbors nn,
        const bool prune = false,
        const bool preserveMaskLeafNodes = false);


    /// @brief Copy the current node masks onto the provided vector. The vector
    ///    is resized if necessary.
    /// @param masks The vector of NodeMasks to copy onto
    void copyMasks(std::vector<MaskType>& masks) const
    {
        if (masks.size() < mManager.leafCount()) {
            masks.resize(mManager.leafCount());
        }

        if (this->getThreaded()) {
            // @note this is marginally faster than using leafRange or foreach
            tbb::parallel_for(mManager.getRange(),
                [&](const tbb::blocked_range<size_t>& r){
                for (size_t idx = r.begin(); idx < r.end(); ++idx)
                    masks[idx] = mManager.leaf(idx).getValueMask();
            });
        }
        else {
            for (size_t idx = 0; idx < mManager.leafCount(); ++idx) {
                masks[idx] = mManager.leaf(idx).getValueMask();
            }
        }
    }

public:
    /// @brief  Node Mask dilation/erosion operations for individual leaf nodes on
    ///   a given tree. The leaf node may optionally belong to a different tree
    ///   than the provided accessor, which will have the effect of dilating the
    ///   leaf node mask into a different tree, or eroding the node mask based
    ///   on corresponding neighbors in a different tree.
    struct NodeMaskOp
    {
        static const Int32 DIM = static_cast<Int32>(LeafType::DIM);
        static const Int32 LOG2DIM = static_cast<Int32>(LeafType::LOG2DIM);

        // Select the storage size based off the dimensions of the leaf node
        using Word = typename std::conditional<LOG2DIM == 3, uint8_t,
            typename std::conditional<LOG2DIM == 4, uint16_t,
            typename std::conditional<LOG2DIM == 5, uint32_t,
            typename std::conditional<LOG2DIM == 6, uint64_t,
                void>::type>::type>::type>::type;

        static_assert(!std::is_same<Word, void>::value,
            "Unsupported Node Dimension for node mask dilation/erosion");

        NodeMaskOp(AccessorType& accessor,
            const NearestNeighbors op)
            : mOrigin(nullptr)
            , mNeighbors(NodeMaskOp::ksize(op), nullptr)
            , mAccessor(&accessor)
            , mOnTile(true)
            , mOffTile(false)
            , mOp(op) {}

        /// @brief Dilate a single leaf node by the current spatial scheme
        ///        stored on the instance of this NodeMaskOp. Neighbor leaf
        ///        nodes are also updated.
        /// @details  Unlike erode, dilate is expected to be called in a
        ///           single threaded context as it will update the node masks
        ///           of neighboring leaf nodes as well as the provided leaf.
        /// @param  leaf  The leaf to dilate. The leaf's origin and value mask
        ///               are used to calculate the result of the dilation.
        inline void dilate(LeafType& leaf)
        {
            // copy the mask
            const MaskType mask = leaf.getValueMask();
            this->dilate(leaf, mask);
        }

        /// @brief Dilate a single leaf node by the current spatial scheme
        ///        stored on the instance of this NodeMaskOp. The provided
        ///        mask is used in place of the actual leaf's node mask and
        ///        applied to the leaf afterwards. Neighbor leaf nodes are
        ///        also updated.
        /// @details  Unlike erode, dilate is expected to be called in a
        ///           single threaded context as it will update the node masks
        ///           of neighboring leaf nodes as well as the provided leaf.
        /// @param  leaf  The leaf to dilate. The leaf's origin is used to
        ///               calculate the result of the dilation.
        /// @param  mask  The node mask to use in place of the current leaf
        ///               node mask.
        inline void dilate(LeafType& leaf, const MaskType& mask)
        {
            this->clear();
            mNeighbors[0] = &(leaf.getValueMask());
            this->setOrigin(leaf.origin());
            switch (mOp) {
                case NN_FACE_EDGE        : { this->dilate18(mask); return; }
                case NN_FACE_EDGE_VERTEX : { this->dilate26(mask); return; }
                case NN_FACE             : { this->dilate6(mask);  return; }
                default                  : {
                    OPENVDB_ASSERT(false && "Unknown op during dilation."); return;
                }
            }
        }

        /// @brief Erode a single leaf node by the current spatial scheme
        ///        stored on the instance of this NodeMaskOp.
        /// @details  Unlike dilate, this method updates the provided mask
        ///           and does not apply the result to the leaf node. The
        ///           leaf node is simply used to infer the position in the
        ///           tree to find it's neighbors. This allows erode to be
        ///           called from multiple threads
        /// @param  leaf  The leaf to erode. The leaf's origin is used to
        ///               calculate the result of the erosion.
        /// @return The eroded mask
        inline MaskType erode(const LeafType& leaf)
        {
            // copy the mask
            MaskType mask = leaf.getValueMask();
            this->erode(leaf, mask);
            return mask;
        }

        /// @brief Erode a single leaf node by the current spatial scheme
        ///        stored on the instance of this NodeMaskOp. The provided
        ///        mask is used in place of the actual leaf's node mask and
        ///        stores the erosion result.
        /// @details  Unlike dilate, this method updates the provided mask
        ///           and does not apply the result to the leaf node. The
        ///           leaf node is simply used to infer the position in the
        ///           tree to find it's neighbors.
        /// @param  leaf  The leaf to erode. The leaf's origin is used to
        ///               calculate the result of the erosion.
        /// @param  mask  The node mask to use in place of the current leaf
        ///               node mask.
        inline void erode(const LeafType& leaf, MaskType& mask)
        {
            this->clear();
            // @note leaf mask will not be modified through gather methods
            mNeighbors[0] = const_cast<MaskType*>(&leaf.getValueMask());
            this->setOrigin(leaf.origin());
            switch (mOp) {
                case NN_FACE_EDGE        : { this->erode18(mask); return; }
                case NN_FACE_EDGE_VERTEX : { this->erode26(mask); return; }
                case NN_FACE             : { this->erode6(mask);  return; }
                default                  : {
                    OPENVDB_ASSERT(false && "Unknown op during erosion."); return;
                }
            }
        }

    private:
        static size_t ksize(const NearestNeighbors op) {
            switch (op) {
                case NN_FACE_EDGE        : return 19;
                case NN_FACE_EDGE_VERTEX : return 27;
                case NN_FACE             : return 7;
                default                  : return 7;
            }
        }

        void dilate6(const MaskType& mask);
        void dilate18(const MaskType& mask);
        void dilate26(const MaskType& mask);
        void erode6(MaskType& mask);

        /// @note  Forward API for erosion of 18/26 trees is to use erodeActiveValues
        ///        which falls back to an inverse dilation
        /// @todo  It may still be worth investigating more optimal gathering
        ///        techniques
        inline void erode18(MaskType&) { OPENVDB_THROW(NotImplementedError, "erode18 is not implemented yet!"); }
        inline void erode26(MaskType&) { OPENVDB_THROW(NotImplementedError, "erode26 is not implemented yet!"); }

        inline void setOrigin(const Coord& origin) { mOrigin = &origin; }
        inline const Coord& getOrigin() const { return *mOrigin; }
        inline void clear() { std::fill(mNeighbors.begin(), mNeighbors.end(), nullptr); }

        inline void scatter(size_t n, int indx)
        {
            OPENVDB_ASSERT(n < mNeighbors.size());
            OPENVDB_ASSERT(mNeighbors[n]);
            mNeighbors[n]->template getWord<Word>(indx) |= mWord;

        }
        template<int DX, int DY, int DZ>
        inline void scatter(size_t n, int indx)
        {
            OPENVDB_ASSERT(n < mNeighbors.size());
            if (!mNeighbors[n]) {
                mNeighbors[n] = this->getNeighbor<DX,DY,DZ,true>();
            }
            OPENVDB_ASSERT(mNeighbors[n]);
            this->scatter(n, indx - (DIM - 1)*(DY + DX*DIM));
        }
        inline Word gather(size_t n, int indx)
        {
            OPENVDB_ASSERT(n < mNeighbors.size());
            return mNeighbors[n]->template getWord<Word>(indx);
        }
        template<int DX, int DY, int DZ>
        inline Word gather(size_t n, int indx)
        {
            OPENVDB_ASSERT(n < mNeighbors.size());
            if (!mNeighbors[n]) {
                mNeighbors[n] = this->getNeighbor<DX,DY,DZ,false>();
            }
            return this->gather(n, indx - (DIM -1)*(DY + DX*DIM));
        }

        void scatterFacesXY(int x, int y, int i1, int n, int i2);
        void scatterEdgesXY(int x, int y, int i1, int n, int i2);
        Word gatherFacesXY(int x, int y, int i1, int n, int i2);
        /// @note Currently unused
        Word gatherEdgesXY(int x, int y, int i1, int n, int i2);

        template<int DX, int DY, int DZ, bool Create>
        inline MaskType* getNeighbor()
        {
            const Coord xyz = mOrigin->offsetBy(DX*DIM, DY*DIM, DZ*DIM);
            auto* leaf = mAccessor->probeLeaf(xyz);
            if (leaf) return &(leaf->getValueMask());
            if (mAccessor->isValueOn(xyz)) return &mOnTile;
            if (!Create)                   return &mOffTile;
            leaf = mAccessor->touchLeaf(xyz);
            return &(leaf->getValueMask());
        }

    private:
        const Coord* mOrigin;
        std::vector<MaskType*> mNeighbors;
        AccessorType* const mAccessor;
        Word mWord;
        MaskType mOnTile, mOffTile;
        const NearestNeighbors mOp;
    };// NodeMaskOp

private:
    std::unique_ptr<tree::LeafManager<TreeType>> mManagerPtr;
    tree::LeafManager<TreeType>& mManager;
    bool mThreaded;
};// Morphology


template <typename TreeT>
typename std::enable_if<std::is_same<TreeT, typename TreeT::template ValueConverter<ValueMask>::Type>::value,
    typename TreeT::template ValueConverter<ValueMask>::Type*>::type
getMaskTree(TreeT& tree) { return &tree; }

template <typename TreeT>
typename std::enable_if<!std::is_same<TreeT, typename TreeT::template ValueConverter<ValueMask>::Type>::value,
    typename TreeT::template ValueConverter<ValueMask>::Type*>::type
getMaskTree(TreeT&) { return nullptr; }


template <typename TreeType>
void Morphology<TreeType>::erodeVoxels(const size_t iter,
    const NearestNeighbors nn,
    const bool prune)
{
    if (iter == 0) return;
    const size_t leafCount = mManager.leafCount();
    if (leafCount == 0) return;
    auto& tree = mManager.tree();

    // If the nearest neighbor mode is not FACE, fall back to an
    // inverse dilation scheme which executes over a mask topology
    if (nn != NN_FACE) {
        // This method 1) dilates the input topology, 2) reverse the node masks,
        // 3) performs a final dilation and 4) subtracts the result from the original
        // topology. A cache of the original leaf pointers is required which tracks
        // the original leaf nodes in a mask topology. These will need their
        // masks updated in the original tree. The first dilation may create new leaf
        // nodes in two instances. The first is where no topology existed before. The
        // second is where an active tile overlaps with dilated topology. These
        // tiles will be expanded to a dense leaf nodes by topologyUnion. We need
        // to make sure these tiles are properly turned off.

        MaskTreeT mask(tree, false, TopologyCopy());

        // Create a new morphology class to perform dilation over the mask
        tree::LeafManager<MaskTreeT> manager(mask);
        Morphology<MaskTreeT> m(manager);
        m.setThreaded(this->getThreaded());

        // perform a single dilation using the current scheme. Necessary to
        // create edge leaf nodes and compute the active wavefront. Note that
        // the cached array pointers will continue to be valid
        m.dilateVoxels(1, nn, /*prune=*/false);

        // compute the wavefront. If the leaf previously existed, compute the
        // xor activity result which is guaranteed to be equal to but slightly
        // faster than a subtraction
        auto computeWavefront = [&](const size_t idx) {
            auto& leaf = manager.leaf(idx);
            auto& nodemask = leaf.getValueMask();
            if (const auto* original = tree.probeConstLeaf(leaf.origin())) {
                nodemask ^= original->getValueMask();
            }
            else {
                // should never have a dense leaf if it didn't exist in the
                // original tree (it was previous possible when dilateVoxels()
                // called topologyUnion without the preservation of active
                // tiles)
                OPENVDB_ASSERT(!nodemask.isOn());
            }
        };

        if (this->getThreaded()) {
            tbb::parallel_for(manager.getRange(),
                [&](const tbb::blocked_range<size_t>& r){
                for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                    computeWavefront(idx);
                }
            });
        }
        else {
            for (size_t idx = 0; idx < manager.leafCount(); ++idx) {
                computeWavefront(idx);
            }
        }

        // perform the inverse dilation
        m.dilateVoxels(iter, nn, /*prune=*/false);

        // subtract the inverse dilation from the original node masks
        auto subtractTopology = [&](const size_t idx) {
            auto& leaf = mManager.leaf(idx);
            const auto* maskleaf = mask.probeConstLeaf(leaf.origin());
            OPENVDB_ASSERT(maskleaf);
            leaf.getValueMask() -= maskleaf->getValueMask();
        };

        if (this->getThreaded()) {
            tbb::parallel_for(mManager.getRange(),
                [&](const tbb::blocked_range<size_t>& r){
                for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                    subtractTopology(idx);
                }
            });
        }
        else {
            for (size_t idx = 0; idx < leafCount; ++idx) {
                subtractTopology(idx);
            }
        }
    }
    else {
        // NN_FACE erosion scheme

        // Save the value masks of all leaf nodes.
        std::vector<MaskType> nodeMasks;
        this->copyMasks(nodeMasks);

        if (this->getThreaded()) {
            const auto range = mManager.getRange();
            for (size_t i = 0; i < iter; ++i) {
                // For each leaf, in parallel, gather neighboring off values
                // and update the cached value mask
                tbb::parallel_for(range,
                    [&](const tbb::blocked_range<size_t>& r) {
                    AccessorType accessor(tree);
                    NodeMaskOp cache(accessor, nn);
                    for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                        const auto& leaf = mManager.leaf(idx);
                        if (leaf.isEmpty()) continue;
                        // original bit-mask of current leaf node
                        MaskType& newMask = nodeMasks[idx];
                        cache.erode(leaf, newMask);
                    }
                });

                // update the masks after all nodes have been eroded
                tbb::parallel_for(range,
                    [&](const tbb::blocked_range<size_t>& r){
                    for (size_t idx = r.begin(); idx < r.end(); ++idx)
                        mManager.leaf(idx).setValueMask(nodeMasks[idx]);
                });
            }
        }
        else {
            AccessorType accessor(tree);
            NodeMaskOp cache(accessor, nn);
            for (size_t i = 0; i < iter; ++i) {
                // For each leaf, in parallel, gather neighboring off values
                // and update the cached value mask
                for (size_t idx = 0; idx < leafCount; ++idx) {
                    const auto& leaf = mManager.leaf(idx);
                    if (leaf.isEmpty()) continue;
                    // original bit-mask of current leaf node
                    MaskType& newMask = nodeMasks[idx];
                    cache.erode(leaf, newMask);
                }

                for (size_t idx = 0; idx < leafCount; ++idx) {
                    mManager.leaf(idx).setValueMask(nodeMasks[idx]);
                }
            }
        }
    }

    // if prune, replace any inactive nodes
    if (prune) {
        tools::prune(mManager.tree(),
            zeroVal<typename TreeType::ValueType>(),
            this->getThreaded());
        mManager.rebuild(!this->getThreaded());
    }
}

template <typename TreeType>
void Morphology<TreeType>::dilateVoxels(const size_t iter,
    const NearestNeighbors nn,
    const bool prune,
    const bool preserveMaskLeafNodes)
{
    if (iter == 0) return;

    const bool threaded = this->getThreaded();

    // Actual dilation op. main implementation is single threaded. Note that this
    // is templated (auto-ed) as the threaded implemenation may call this with a
    // different value type to the source morphology class
    // @note  GCC 6.4.0 crashes trying to compile this lambda with [&] capture
    auto dilate = [iter, nn, threaded](auto& manager, const bool collapse) {

        using LeafManagerT = typename std::decay<decltype(manager)>::type;
        using TreeT = typename LeafManagerT::TreeType;
        using ValueT = typename TreeT::ValueType;
        using LeafT = typename TreeT::LeafNodeType;

        // this is only used for the impl of copyMasks
        Morphology<TreeT> m(manager);
        m.setThreaded(threaded);

        TreeT& tree = manager.tree();
        tree::ValueAccessor<TreeT> accessor(tree);

        // build cache objects
        typename Morphology<TreeT>::NodeMaskOp cache(accessor, nn);
        std::vector<MaskType> nodeMasks;
        std::vector<std::unique_ptr<LeafT>> nodes;
        const ValueT& bg = tree.background();
        const bool steal = iter > 1;

        for (size_t i = 0; i < iter; ++i) {
            if (i > 0) manager.rebuild(!threaded);
            // If the leaf count is zero, we can stop dilation
            const size_t leafCount = manager.leafCount();
            if (leafCount == 0) return;

            // Copy the masks. This only resizes if necessary. As we're stealing/replacing
            // dense nodes, it's possible we don't need to re-allocate the cache.
            m.copyMasks(nodeMasks);

            // For each node, dilate the mask into itself and neighboring leaf nodes.
            // If the node was originally dense (all active), steal/replace it so
            // subsequent iterations are faster
            manager.foreach([&](auto& leaf, const size_t idx) {
                // original bit-mask of current leaf node
                const MaskType& oldMask = nodeMasks[idx];
                const bool dense = oldMask.isOn();
                cache.dilate(leaf, oldMask);
                if (!dense) return;
                // This node does not need to be visited again - replace or steal
                if (collapse) {
                    // if collapse, replace this dense leaf with an active background tile
                    accessor.addTile(1, leaf.origin(), bg, true);
                }
                else if (steal) {
                    // otherwise, temporarily steal this node
                    nodes.emplace_back(
                        tree.template stealNode<LeafT>(leaf.origin(),
                            zeroVal<ValueT>(), true));
                }
            }, false);
        }

        if (nodes.empty()) return;
        // Add back all dense nodes
        for (auto& node : nodes) {
            accessor.addLeaf(node.release());
        }
    };

    //

    if (!threaded) {
        // single threaded dilation. If it's a mask tree we can collapse
        // nodes during the dilation, otherwise we must call prune afterwards
        constexpr bool isMask = std::is_same<TreeType, MaskTreeT>::value;
        dilate(mManager, isMask && prune);
        if (!isMask && prune) {
            tools::prune(mManager.tree(),
                zeroVal<typename TreeType::ValueType>(),
                threaded);
        }
    }
    else {
        // multi-threaded dilation

        // Steal or create mask nodes that represent the current leaf nodes.
        // If the input is a mask tree, optionally re-allocate the nodes if
        // preserveMaskLeafNodes is true. This ensures that leaf node
        // pointers are not changed in the source tree. Stealing the mask
        // nodes is significantly faster as it also avoids a post union.
        std::vector<MaskLeafT*> array;
        MaskTreeT* mask = getMaskTree(mManager.tree());

        if (!mask) {
            MaskTreeT topology;
            topology.topologyUnion(mManager.tree());
            array.reserve(mManager.leafCount());
            topology.stealNodes(array);
        }
        else if (preserveMaskLeafNodes) {
            mask = nullptr; // act as if theres no mask tree
            array.resize(mManager.leafCount());
            tbb::parallel_for(mManager.getRange(),
                [&](const tbb::blocked_range<size_t>& r){
                for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                    array[idx] = new MaskLeafT(mManager.leaf(idx));
                }
            });
        }
        else {
            array.reserve(mManager.leafCount());
            mask->stealNodes(array);
        }

        // @note this grain size is used for optimal threading
        const size_t numThreads = size_t(tbb::this_task_arena::max_concurrency());
        const size_t subTreeSize = math::Max(size_t(1), array.size()/(2*numThreads));

        // perform recursive dilation to sub trees
        tbb::enumerable_thread_specific<std::unique_ptr<MaskTreeT>> pool;
        MaskLeafT** start = array.data();
        tbb::parallel_for(tbb::blocked_range<MaskLeafT**>(start, start + array.size(), subTreeSize),
            [&](const tbb::blocked_range<MaskLeafT**>& range) {
                std::unique_ptr<MaskTreeT> mask(new MaskTreeT);
                for (MaskLeafT** it = range.begin(); it != range.end(); ++it) mask->addLeaf(*it);
                tree::LeafManager<MaskTreeT> manager(*mask, range.begin(), range.end());
                dilate(manager, prune);
                auto& subtree = pool.local();
                if (!subtree) subtree = std::move(mask);
                else          subtree->merge(*mask, MERGE_ACTIVE_STATES);
            });

        if (!pool.empty()) {
            auto piter = pool.begin();
            MaskTreeT& subtree = mask ? *mask : **piter++;
            for (; piter != pool.end(); ++piter) subtree.merge(**piter);
            // prune, ensures partially merged nodes that may have become
            // dense are converted to tiles
            if (prune) tools::prune(subtree, zeroVal<typename MaskTreeT::ValueType>(), threaded);
            // copy final topology onto dest. If mask exists, then this
            // has already been handled by the above subtree merges
            if (!mask) mManager.tree().topologyUnion(subtree, /*preserve-active-tiles*/true);
        }
    }

    // sync
    mManager.rebuild(!threaded);
}


template <typename TreeType>
inline void
Morphology<TreeType>::NodeMaskOp::erode6(MaskType& mask)
{
    for (int x = 0; x < DIM; ++x) {
        for (int y = 0, n = (x << LOG2DIM); y < DIM; ++y, ++n) {
            // Extract the portion of the original mask that corresponds to a row in z.
            if (Word& w = mask.template getWord<Word>(n)) {
                // erode in two z directions (this is first since it uses the original w)
                w = Word(w &
                    (Word(w<<1 | (this->template gather<0,0,-1>(1, n)>>(DIM-1))) &
                     Word(w>>1 | (this->template gather<0,0, 1>(2, n)<<(DIM-1)))));
                w = Word(w & this->gatherFacesXY(x, y, 0, n, 3));
            }
        }// loop over y
    }//loop over x
}

template <typename TreeType>
inline void
Morphology<TreeType>::NodeMaskOp::dilate6(const MaskType& mask)
{
    for (int x = 0; x < DIM; ++x ) {
        for (int y = 0, n = (x << LOG2DIM);
                 y < DIM; ++y, ++n) {
            // Extract the portion of the original mask that corresponds to a row in z.
            if (const Word w = mask.template getWord<Word>(n)) {
                // Dilate the current leaf in the +z and -z direction
                this->mWord = Word(w | (w>>1) | (w<<1));
                this->scatter(0, n);
                // Dilate into neighbor leaf in the -z direction
                if ( (this->mWord = Word(w<<(DIM-1))) ) {
                    this->template scatter< 0, 0,-1>(1, n);
                }
                // Dilate into neighbor leaf in the +z direction
                if ( (this->mWord = Word(w>>(DIM-1))) ) {
                    this->template scatter< 0, 0, 1>(2, n);
                }
                // Dilate in the xy-face directions relative to the center leaf
                this->mWord = w;
                this->scatterFacesXY(x, y, 0, n, 3);
            }
        }// loop over y
    }//loop over x
}

template <typename TreeType>
inline void
Morphology<TreeType>::NodeMaskOp::dilate18(const MaskType& mask)
{
    //origins of neighbor leaf nodes in the -z and +z directions
    const Coord origin = this->getOrigin();
    const Coord orig_mz = origin.offsetBy(0, 0, -DIM);
    const Coord orig_pz = origin.offsetBy(0, 0,  DIM);
    for (int x = 0; x < DIM; ++x ) {
        for (int y = 0, n = (x << LOG2DIM); y < DIM; ++y, ++n) {
            if (const Word w = mask.template getWord<Word>(n)) {
                {
                    this->mWord = Word(w | (w>>1) | (w<<1));
                    this->setOrigin(origin);
                    this->scatter(0, n);
                    this->scatterFacesXY(x, y, 0, n, 3);
                    this->mWord = w;
                    this->scatterEdgesXY(x, y, 0, n, 3);
                }
                if ( (this->mWord = Word(w<<(DIM-1))) ) {
                    this->setOrigin(origin);
                    this->template scatter< 0, 0,-1>(1, n);
                    this->setOrigin(orig_mz);
                    this->scatterFacesXY(x, y, 1, n, 11);
                }
                if ( (this->mWord = Word(w>>(DIM-1))) ) {
                    this->setOrigin(origin);
                    this->template scatter< 0, 0, 1>(2, n);
                    this->setOrigin(orig_pz);
                    this->scatterFacesXY(x, y, 2, n, 15);
                }
            }
        }// loop over y
    }//loop over x
}


template <typename TreeType>
inline void
Morphology<TreeType>::NodeMaskOp::dilate26(const MaskType& mask)
{
    //origins of neighbor leaf nodes in the -z and +z directions
    const Coord origin = this->getOrigin();
    const Coord orig_mz = origin.offsetBy(0, 0, -DIM);
    const Coord orig_pz = origin.offsetBy(0, 0,  DIM);
    for (int x = 0; x < DIM; ++x) {
        for (int y = 0, n = (x << LOG2DIM); y < DIM; ++y, ++n) {
            if (const Word w = mask.template getWord<Word>(n)) {
                {
                    this->mWord = Word(w | (w>>1) | (w<<1));
                    this->setOrigin(origin);
                    this->scatter(0, n);
                    this->scatterFacesXY(x, y, 0, n, 3);
                    this->scatterEdgesXY(x, y, 0, n, 3);
                }
                if ( (this->mWord = Word(w<<(DIM-1))) ) {
                    this->setOrigin(origin);
                    this->template scatter< 0, 0,-1>(1, n);
                    this->setOrigin(orig_mz);
                    this->scatterFacesXY(x, y, 1, n, 11);
                    this->scatterEdgesXY(x, y, 1, n, 11);
                }
                if ( (this->mWord = Word(w>>(DIM-1))) ) {
                    this->setOrigin(origin);
                    this->template scatter< 0, 0, 1>(2, n);
                    this->setOrigin(orig_pz);
                    this->scatterFacesXY(x, y, 2, n, 19);
                    this->scatterEdgesXY(x, y, 2, n, 19);
                }
            }
        }// loop over y
    }//loop over x
}

template<typename TreeType>
inline void
Morphology<TreeType>::NodeMaskOp::scatterFacesXY(int x, int y, int i1, int n, int i2)
{
    // dilate current leaf or neighbor in the -x direction
    if (x > 0) {
        this->scatter(i1, n-DIM);
    } else {
        this->template scatter<-1, 0, 0>(i2, n);
    }
    // dilate current leaf or neighbor in the +x direction
    if (x < DIM-1) {
        this->scatter(i1, n+DIM);
    } else {
        this->template scatter< 1, 0, 0>(i2+1, n);
    }
    // dilate current leaf or neighbor in the -y direction
    if (y > 0) {
        this->scatter(i1, n-1);
    } else {
        this->template scatter< 0,-1, 0>(i2+2, n);
    }
    // dilate current leaf or neighbor in the +y direction
    if (y < DIM-1) {
        this->scatter(i1, n+1);
    } else {
        this->template scatter< 0, 1, 0>(i2+3, n);
    }
}


template<typename TreeType>
inline void
Morphology<TreeType>::NodeMaskOp::scatterEdgesXY(int x, int y, int i1, int n, int i2)
{
    if (x > 0) {
        if (y > 0) {
            this->scatter(i1, n-DIM-1);
        } else {
            this->template scatter< 0,-1, 0>(i2+2, n-DIM);
        }
        if (y < DIM-1) {
            this->scatter(i1, n-DIM+1);
        } else {
            this->template scatter< 0, 1, 0>(i2+3, n-DIM);
        }
    } else {
        if (y < DIM-1) {
            this->template scatter<-1, 0, 0>(i2  , n+1);
        } else {
            this->template scatter<-1, 1, 0>(i2+7, n  );
        }
        if (y > 0) {
            this->template scatter<-1, 0, 0>(i2  , n-1);
        } else {
            this->template scatter<-1,-1, 0>(i2+4, n  );
        }
    }
    if (x < DIM-1) {
        if (y > 0) {
            this->scatter(i1, n+DIM-1);
        } else {
            this->template scatter< 0,-1, 0>(i2+2, n+DIM);
        }
        if (y < DIM-1) {
            this->scatter(i1, n+DIM+1);
        } else {
            this->template scatter< 0, 1, 0>(i2+3, n+DIM);
        }
    } else {
        if (y > 0) {
            this->template scatter< 1, 0, 0>(i2+1, n-1);
        } else {
            this->template scatter< 1,-1, 0>(i2+6, n  );
        }
        if (y < DIM-1) {
            this->template scatter< 1, 0, 0>(i2+1, n+1);
        } else {
            this->template scatter< 1, 1, 0>(i2+5, n  );
        }
    }
}


template<typename TreeType>
inline typename Morphology<TreeType>::NodeMaskOp::Word
Morphology<TreeType>::NodeMaskOp::gatherFacesXY(int x, int y, int i1, int n, int i2)
{
    // erode current leaf or neighbor in negative x-direction
    Word w = x > 0 ?
        this->gather(i1, n - DIM) :
        this->template gather<-1,0,0>(i2, n);

    // erode current leaf or neighbor in positive x-direction
    w = Word(w & (x < DIM - 1 ?
        this->gather(i1, n + DIM) :
        this->template gather<1,0,0>(i2 + 1, n)));

    // erode current leaf or neighbor in negative y-direction
    w = Word(w & (y > 0 ?
        this->gather(i1, n - 1) :
        this->template gather<0,-1,0>(i2 + 2, n)));

    // erode current leaf or neighbor in positive y-direction
    w = Word(w & (y < DIM - 1 ?
        this->gather(i1, n + 1) :
        this->template gather<0,1,0>(i2+3, n)));

    return w;
}


template<typename TreeType>
inline typename Morphology<TreeType>::NodeMaskOp::Word
Morphology<TreeType>::NodeMaskOp::gatherEdgesXY(int x, int y, int i1, int n, int i2)
{
    Word w = ~Word(0);

    if (x > 0) {
        w &= y > 0 ?          this->gather(i1, n-DIM-1) :
                              this->template gather< 0,-1, 0>(i2+2, n-DIM);
        w &= y < DIM-1 ? this->gather(i1, n-DIM+1) :
                              this->template gather< 0, 1, 0>(i2+3, n-DIM);
    } else {
        w &= y < DIM-1 ? this->template gather<-1, 0, 0>(i2  , n+1):
                              this->template gather<-1, 1, 0>(i2+7, n  );
        w &= y > 0 ?          this->template gather<-1, 0, 0>(i2  , n-1):
                              this->template gather<-1,-1, 0>(i2+4, n  );
    }
    if (x < DIM-1) {
        w &= y > 0 ?          this->gather(i1, n+DIM-1) :
                              this->template gather< 0,-1, 0>(i2+2, n+DIM);
        w &= y < DIM-1 ? this->gather(i1, n+DIM+1) :
                              this->template gather< 0, 1, 0>(i2+3, n+DIM);
    } else {
        w &= y > 0          ? this->template gather< 1, 0, 0>(i2+1, n-1):
                              this->template gather< 1,-1, 0>(i2+6, n  );
        w &= y < DIM-1 ? this->template gather< 1, 0, 0>(i2+1, n+1):
                              this->template gather< 1, 1, 0>(i2+5, n  );
    }

    return w;
}

} // namespace morphology


/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/// @cond OPENVDB_DOCS_INTERNAL

namespace morph_internal {
template <typename T> struct Adapter {
    using TreeType = T;
    static TreeType& get(T& tree) { return tree; }
    static void sync(T&) {} // no-op
};
template <typename T>
struct Adapter<openvdb::tree::LeafManager<T>> {
    using TreeType = T;
    static TreeType& get(openvdb::tree::LeafManager<T>& M) { return M.tree(); }
    static void sync(openvdb::tree::LeafManager<T>& M) { M.rebuild(); }
};
}

/// @endcond

template<typename TreeOrLeafManagerT>
void dilateActiveValues(TreeOrLeafManagerT& treeOrLeafM,
                   const int iterations,
                   const NearestNeighbors nn,
                   const TilePolicy mode,
                   const bool threaded)
{
    using AdapterT = morph_internal::Adapter<TreeOrLeafManagerT>;
    using TreeT = typename AdapterT::TreeType;
    using MaskT = typename TreeT::template ValueConverter<ValueMask>::Type;

    if (iterations <= 0) return;

    if (mode == IGNORE_TILES) {
        morphology::Morphology<TreeT> morph(treeOrLeafM);
        morph.setThreaded(threaded);
        // This will also sync the leaf manager
        morph.dilateVoxels(static_cast<size_t>(iterations), nn, /*prune=*/false);
        return;
    }

    // The following branching optimises from the different tree types
    // and TilePolicy combinations

    auto& tree = AdapterT::get(treeOrLeafM);

    // If the input is a mask tree, don't copy the topology - voxelize
    // it directly and let the morphology class directly steal/prune
    // its nodes
    constexpr bool isMask = std::is_same<TreeT, MaskT>::value;

    if (isMask || mode == EXPAND_TILES) {
        tree.voxelizeActiveTiles();
        AdapterT::sync(treeOrLeafM);
        morphology::Morphology<TreeT> morph(treeOrLeafM);
        morph.setThreaded(threaded);

        if (mode == PRESERVE_TILES) {
            morph.dilateVoxels(static_cast<size_t>(iterations), nn, /*prune=*/true);
        }
        else {
            OPENVDB_ASSERT(mode == EXPAND_TILES);
            morph.dilateVoxels(static_cast<size_t>(iterations), nn, /*prune=*/false);
        }
        return;
    }

    // If the tree TreeType being dilated is not a MaskTree, always copy
    // the topology over onto a MaskTree, perform the required dilation
    // and copy the final topology back. This technique avoids unnecessary
    // allocation with tile expansion and correctly preserves the tree
    // topology.
    //
    // Note that we also always use a mask if the tile policy is PRESERVE_TILES
    // due to the way the underlying dilation only works on voxels.
    // @todo Investigate tile based dilation
    OPENVDB_ASSERT(mode == PRESERVE_TILES);

    MaskT topology;
    topology.topologyUnion(tree);
    topology.voxelizeActiveTiles();

    morphology::Morphology<MaskT> morph(topology);
    morph.setThreaded(threaded);
    morph.dilateVoxels(static_cast<size_t>(iterations), nn, /*prune=*/true);

    tree.topologyUnion(topology, /*preserve-tiles*/true);
    topology.clear();

    // @note  this is necessary to match the behaviour of mask tree dilation
    //        where  source partial leaf nodes that become dense are also
    //        converted into tiles, not simply newly created dense nodes
    tools::prune(tree, zeroVal<typename TreeT::ValueType>(), threaded);
    AdapterT::sync(treeOrLeafM);
}


template<typename TreeOrLeafManagerT>
void erodeActiveValues(TreeOrLeafManagerT& treeOrLeafM,
                      const int iterations,
                      const NearestNeighbors nn,
                      const TilePolicy mode,
                      const bool threaded)
{
    using AdapterT = morph_internal::Adapter<TreeOrLeafManagerT>;
    using TreeT = typename AdapterT::TreeType;
    using MaskT = typename TreeT::template ValueConverter<ValueMask>::Type;

    if (iterations <= 0) return;

    // If the tile policiy is PRESERVE_TILES, peform the erosion on a
    // voxelized mask grid followed by a topology intersection such that
    // the original uneroded topology is preserved.
    if (mode == PRESERVE_TILES) {
        auto& tree = AdapterT::get(treeOrLeafM);
        MaskT topology;
        topology.topologyUnion(tree);
        topology.voxelizeActiveTiles();

        {
            morphology::Morphology<MaskT> morph(topology);
            morph.setThreaded(threaded);
            morph.erodeVoxels(static_cast<size_t>(iterations), nn, /*prune=*/false);
        }

        // prune to ensure topologyIntersection does not expand tiles
        // which have not been changed
        tools::prune(topology, zeroVal<typename MaskT::ValueType>(), threaded);
        tree.topologyIntersection(topology);
        AdapterT::sync(treeOrLeafM);
        return;
    }

    if (mode == EXPAND_TILES) {
        // if expanding, voxelize everything first if there are active tiles
        // @note  check first to avoid any unnecessary rebuilds
        auto& tree = AdapterT::get(treeOrLeafM);
        if (tree.hasActiveTiles()) {
            tree.voxelizeActiveTiles();
            AdapterT::sync(treeOrLeafM);
        }
    }

    // ignoring tiles. They won't be eroded
    morphology::Morphology<TreeT> morph(treeOrLeafM);
    morph.setThreaded(threaded);
    morph.erodeVoxels(static_cast<size_t>(iterations), nn, /*prune=*/false);
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_MORPHOLOGY
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    void dilateActiveValues(TreeT&, \
        const int, const NearestNeighbors, const TilePolicy, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void dilateActiveValues(tree::LeafManager<TreeT>&, \
        const int, const NearestNeighbors, const TilePolicy, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void erodeActiveValues(TreeT&, \
        const int, const NearestNeighbors, const TilePolicy, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void erodeActiveValues(tree::LeafManager<TreeT>&, \
        const int, const NearestNeighbors, const TilePolicy, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED
