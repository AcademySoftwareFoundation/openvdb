// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//

#ifndef OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/TreeIterator.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <functional>
#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

////////////////////////////////////////

enum NearestNeighbors { NN_FACE = 6, NN_FACE_EDGE = 18, NN_FACE_EDGE_VERTEX = 26 };

/// Mapping from a Log2Dim to a data type of size 2^Log2Dim bits
template<Index Log2Dim> struct DimToWord {};
template<> struct DimToWord<3> { using Type = uint8_t; };
template<> struct DimToWord<4> { using Type = uint16_t; };
template<> struct DimToWord<5> { using Type = uint32_t; };
template<> struct DimToWord<6> { using Type = uint64_t; };


////////////////////////////////////////

/// @brief  Dilation/Erosion operations over a Trees voxel topology.
template<typename TreeType>
class Morphology
{
public:
    using LeafType = typename TreeType::LeafNodeType;
    using MaskType = typename LeafType::NodeMaskType;
    using ValueType = typename TreeType::ValueType;
    using MaskTreeT = typename TreeType::template ValueConverter<ValueMask>::Type;
    using AccessorType = tree::ValueAccessor<TreeType>;

    Morphology(TreeType& tree)
        : mManagerPtr(new tree::LeafManager<TreeType>(tree))
        , mManager(*mManagerPtr)
        , mGrainSize(1)
        , mMask(nullptr) {}

    Morphology(tree::LeafManager<TreeType>& tree)
        : mManagerPtr(nullptr)
        , mManager(tree)
        , mGrainSize(1)
        , mMask(nullptr) {}

    /// @return the grain-size used for multi-threading
    size_t getGrainSize() const { return mGrainSize; }
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grain size of 0 or less disables multi-threading
    inline void setGrainSize(const size_t grain) { mGrainSize = grain; }

    inline const tree::LeafManager<TreeType>& manager() const { return mManager; }
    inline void setMask(const MaskTreeT& mask) { mMask = &mask; }

    /// @brief Topologically erode all voxels by the provided nearest
    ///        neighbour scheme and iterations.
    //// @details Voxel values are unchanged and only leaf nodes/voxels
    ///           are updated. Inactive Tiles contribute to the erosion
    ///           but active tiles are not modified. Consider voxelizing
    ///           the tree with tree.voxelizeActiveTiles if you need tiles
    ///           to be eroded.
    /// @note  The topology structure of the tree is unchanged. i.e. tiles are
    ///        left alone and leaf nodes are not collapsed/pruned. As this
    ///        method may create fully inactive leaf nodes, consider following
    ///        with a tools::pruneInactive if they are undesired.
    inline void erodeVoxels(const size_t iter,
        const NearestNeighbors nn) {
        this->erodeVoxels(iter, nn, nullptr);
    }

    /// @brief Topologically erode all voxels by the provided nearest
    ///        neighbour scheme and iterations and additionally collapse
    ///        leaf nodes which are fully inactive with an inactive tile with
    ///        the provided value.
    //// @details Inactive Tiles contribute to the erosion but active tiles
    ///           are not modified. Consider voxelizing the tree with
    ///           tree.voxelizeActiveTiles if you need tiles to be eroded.
    inline void erodeVoxels(const size_t iter,
        const NearestNeighbors nn,
        const ValueType& inactive) {
        this->erodeVoxels(iter, nn, &inactive);
    }

    /// @brief Topologically dilate all voxels by the provided nearest
    ///        neighbour scheme and iterations.
    //// @details Voxel values are unchanged and only leaf nodes/voxels
    ///           are used to propagate the dilation. Consider voxelizing
    ////          the tree with tree.voxelizeActiveTiles if you need tiles
    ///           to be dilated.
    /// @note  The topology structure of the tree is unchanged. i.e. tiles are
    ///        left alone and leaf nodes are not collapsed/pruned.
    inline void dilateVoxels(const size_t iter,
        const NearestNeighbors nn) {
        this->dilateVoxels(iter, nn, nullptr);
    }

    /// @brief Topologically dilate all voxels by the provided nearest
    ///        neighbour scheme and iterations and additionally collapse
    ///        leaf nodes which are fully active with an active tile with
    ///        the provided value.
    //// @details Voxel values are unchanged and only leaf nodes/voxels
    ///           are used to propagate the dilation. Consider voxelizing
    ////          the tree with tree.voxelizeActiveTiles if you need tiles
    ///           to be dilated.
    inline void dilateVoxels(const size_t iter,
        const NearestNeighbors nn,
        const ValueType& active) {
        this->dilateVoxels(iter, nn, &active);
    }

    inline void toggleMasks() {
        tbb::parallel_for(mManager.getRange(),
            [&](const tbb::blocked_range<size_t>& r){
            for (size_t idx = r.begin(); idx < r.end(); ++idx)
                mManager.leaf(idx).getValueMask().toggle();
        });
    }

private:

    void erodeVoxels(const size_t iter,
        const NearestNeighbors nn,
        const ValueType* inactive);

    void dilateVoxels(const size_t iter,
        const NearestNeighbors nn,
        const ValueType* active);

    void updateMasks(const std::vector<MaskType>& masks) {
        assert(masks.size() >= mManager.leafCount());
        // @note this is marginally faster than using leafRange
        // or foreach
        tbb::parallel_for(mManager.getRange(),
            [&](const tbb::blocked_range<size_t>& r){
            for (size_t idx = r.begin(); idx < r.end(); ++idx)
                mManager.leaf(idx).setValueMask(masks[idx]);
        });
    }

    void copyMasks(std::vector<MaskType>& masks) const {
        assert(masks.size() >= mManager.leafCount());
        // @note this is marginally faster than using leafRange
        // or foreach
        tbb::parallel_for(mManager.getRange(),
            [&](const tbb::blocked_range<size_t>& r){
            for (size_t idx = r.begin(); idx < r.end(); ++idx)
                masks[idx] = mManager.leaf(idx).getValueMask();
        });
    }

    struct NodeMaskOp
    {
        static const Int32 DIM = static_cast<Int32>(LeafType::DIM);
        static const Int32 LOG2DIM = static_cast<Int32>(LeafType::LOG2DIM);
        using Word = typename DimToWord<LOG2DIM>::Type;

        NodeMaskOp(AccessorType& accessor,
            const NearestNeighbors op,
            const tree::ValueAccessor<const MaskTreeT>* const mask = nullptr)
            : mOrigin(nullptr)
            , mNeighbours(NodeMaskOp::ksize(op), nullptr)
            , mAccessor(&accessor)
            , mOnTile(true)
            , mOffTile(false)
            , mOp(op)
            , mMask(mask) {}

        static size_t ksize(const NearestNeighbors op) {
            switch (op) {
                case NN_FACE_EDGE        : return 19;
                case NN_FACE_EDGE_VERTEX : return 27;
                case NN_FACE             : return 7;
                default                  : return 7;
            }
        }

        /// @brief Dilate a single leaf node by the current spatial scheme
        ///        stored on the instance of this NodeMaskOp. The provided
        ///        mask is used in place of the actual leaf's node mask and
        ///        applied to the leaf afterwards. Neighbour leaf nodes are
        ///        also updated.
        /// @details  Unlike erode, dilate is expected to be called in a
        ///           single threaded context as it will update the node masks
        ///           of neighbouring leaf nodes as well as the provided leaf.
        /// @param  leaf  The leaf to dilate. The leaf's origin and value mask
        ///               (if mask is not provided) are used to calcualte the
        ///               result of the dilation.
        /// @param  mask  An optional mask to use in place of the current leaf
        ///               node mask. If not provided, the provided leaf's value
        ///               mask is used.
        inline void dilate(LeafType& leaf, const MaskType& mask)
        {
            this->clear();
            mNeighbours[0] = &(leaf.getValueMask());
            this->setOrigin(leaf.origin());
            switch (mOp) {
                case NN_FACE_EDGE        : { this->dilate18(mask); return; }
                case NN_FACE_EDGE_VERTEX : { this->dilate26(mask); return; }
                case NN_FACE             : { this->dilate6(mask);  return; }
                default                  : { this->dilate6(mask);  return; }
            }
        }

        /// @brief Erode a single leaf node by the current spatial scheme
        ///        stored on the instance of this NodeMaskOp. The provided
        ///        mask is used in place of the actual leaf's node mask and
        ///        stores the erosion result.
        /// @details  Unlike dialte, this method updates the provided mask
        ///           and does not apply the result to the leaf node. The
        ///           leaf node is simply used to infer the position in the
        ///           tree to find it's neighbours.
        inline void erode(const LeafType& leaf, MaskType& mask)
        {
            this->clear();
            // @note leaf mask will not be modified through gather methods
            mNeighbours[0] = const_cast<MaskType*>(&leaf.getValueMask());
            this->setOrigin(leaf.origin());
            switch (mOp) {
                case NN_FACE_EDGE        : { this->erode18(mask); return; }
                case NN_FACE_EDGE_VERTEX : { this->erode26(mask); return; }
                case NN_FACE             : { this->erode6(mask);  return; }
                default                  : { this->erode6(mask);  return; }
            }
        }

    private:
        void dilate6(const MaskType& mask);
        void dilate18(const MaskType& mask);
        void dilate26(const MaskType& mask);
        void erode6(MaskType& mask);

        /// @note  Forward API for erosion of 18/26 trees is to use erodeVoxels
        ///        which falls back to an inverse dilation
        /// @todo  It may still be worth investigating more optimal gathering
        ///        techniques
        inline void erode18(MaskType&) { OPENVDB_THROW(NotImplementedError, "erode18 is not implemented yet!"); }
        inline void erode26(MaskType&) { OPENVDB_THROW(NotImplementedError, "erode26 is not implemented yet!"); }

        inline void setOrigin(const Coord& origin) { mOrigin = &origin; }
        inline const Coord& getOrigin() const { return *mOrigin; }
        inline void clear() { std::fill(mNeighbours.begin(), mNeighbours.end(), nullptr); }

        inline void scatter(size_t n, int indx)
        {
            assert(mMask || (!mMask && mNeighbours[n]));
            assert(n < mNeighbours.size());
            if (mNeighbours[n]) {
                mNeighbours[n]->template getWord<Word>(indx) |= mWord;
            }
        }
        template<int DX, int DY, int DZ>
        inline void scatter(size_t n, int indx)
        {
            assert(n < mNeighbours.size());
            if (!mNeighbours[n]) {
                mNeighbours[n] = this->getNeighbour<DX,DY,DZ>(true);
            }
            assert(mMask || (!mMask && mNeighbours[n]));
            if (mNeighbours[n]) {
                this->scatter(n, indx - (DIM - 1)*(DY + DX*DIM));
            }
        }
        inline Word gather(size_t n, int indx)
        {
            assert(n < mNeighbours.size());
            return mNeighbours[n]->template getWord<Word>(indx);
        }
        template<int DX, int DY, int DZ>
        inline Word gather(size_t n, int indx)
        {
            assert(n < mNeighbours.size());
            if (!mNeighbours[n]) {
                mNeighbours[n] = this->getNeighbour<DX,DY,DZ>(false);
            }
            return this->gather(n, indx - (DIM -1)*(DY + DX*DIM));
        }

        void scatterFacesXY(int x, int y, int i1, int n, int i2);

        void scatterEdgesXY(int x, int y, int i1, int n, int i2);

        Word gatherFacesXY(int x, int y, int i1, int n, int i2);

        Word gatherEdgesXY(int x, int y, int i1, int n, int i2);

        template<int DX, int DY, int DZ>
        inline MaskType* getNeighbour(const bool create = true)
        {
            const Coord xyz = mOrigin->offsetBy(DX*DIM, DY*DIM, DZ*DIM);
            auto* leaf = mAccessor->probeLeaf(xyz);
            if (leaf) return &(leaf->getValueMask());
            if (mAccessor->isValueOn(xyz)) return &mOnTile;
            if (!create)                   return &mOffTile;
            if (mMask && !mMask->isValueOn(xyz)) return nullptr;
            leaf = mAccessor->touchLeaf(xyz);
            return &(leaf->getValueMask());
        }

        const Coord* mOrigin;
        std::vector<MaskType*> mNeighbours;
        AccessorType* const mAccessor;
        Word mWord;
        MaskType mOnTile, mOffTile;
        const NearestNeighbors mOp;
        const tree::ValueAccessor<const MaskTreeT>* const mMask;
    };// NodeMaskOp

    std::unique_ptr<tree::LeafManager<TreeType>> mManagerPtr;
    tree::LeafManager<TreeType>& mManager;
    size_t mGrainSize;
    const MaskTreeT* mMask;
};// Morphology


template <typename TreeType>
void Morphology<TreeType>::erodeVoxels(const size_t iter,
    const NearestNeighbors nn,
    const ValueType* inactive)
{
    if (iter == 0) return;
    const size_t leafCount = mManager.leafCount();
    if (leafCount == 0) return;
    auto& tree = mManager.tree();

    // If the nearest neighbour mode is not FACE, fall back to an
    // inverse dilation scheme which executes over a mask topology
    if (nn != NN_FACE) {
        MaskTreeT mask;
        mask.topologyUnion(tree);

        // cache the leaf node pointers which will be updated
        std::vector<typename MaskTreeT::LeafNodeType*> array;
        array.reserve(leafCount);
        mask.getNodes(array);

        // Create a new morphology class to perform dilation over the mask
        tree::LeafManager<MaskTreeT> manager(mask,
            array.data(), array.data() + array.size());
        Morphology<MaskTreeT> m(manager);

        // perform a single dilation using the current scheme
        m.dilateVoxels(1, nn);

        // build a tiled topology which represents the acceptable area of dilation
        /*
        MaskTreeT tiles;
        tree::ValueAccessor<MaskTreeT> acc(tiles);
        manager.foreach([&acc](const auto& leaf, size_t) {
            acc.addTile(1, leaf.origin(), true, true);
        }, false);
        m.setMask(tiles);
        */

        // toggle all node masks
        m.toggleMasks();

        // perform the inverse dilation - do not create tiles
        m.dilateVoxels(iter + 1, nn);

        tbb::parallel_for(mManager.getRange(),
            [&](const tbb::blocked_range<size_t>& r){
            for (size_t idx = r.begin(); idx < r.end(); ++idx)
                mManager.leaf(idx).setValueMask(array[idx]->getValueMask());
        });
        return;
    }

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> nodeMasks(leafCount);
    this->copyMasks(nodeMasks);

    std::vector<std::unique_ptr<LeafType>> nodes;

    // Collapse nodes every 4th iteration. A fully dense (active) leaf would
    // become completely inactive on the 5th iteration. Choosing the 4th
    // assumes that some leaf nodes will predominately be a combination of
    // active and inactive states
    bool collapse = false;
    for (size_t i = 0; i < iter; ++i/*, collapse = (i%4 == 0)*/) {

        // For each leaf, in parallel, gather neighboring off values
        // and update the cached value mask
        tbb::parallel_for(mManager.getRange(),
            [&](const tbb::blocked_range<size_t>& r) {
            AccessorType accessor(tree);
            NodeMaskOp cache(accessor, nn, nullptr);
            for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                const auto& leaf = mManager.leaf(idx);
                if (leaf.isEmpty()) continue;
                if (mMask && mMask->isValueOff(leaf.origin())) return;
                // original bit-mask of current leaf node
                MaskType& newMask = nodeMasks[idx];
                cache.erode(leaf, newMask);
            }
        });

        this->updateMasks(nodeMasks);
        if (!inactive && !collapse) continue;

        AccessorType accessor(tree);
        // @todo use a node manager to parallelise over internal nodes
        mManager.foreach([&](auto& leaf, const size_t) {
            if (!leaf.isEmpty()) return;
            if (inactive) {
                accessor.addTile(1, leaf.origin(), *inactive, false);
            }
            else {
                nodes.emplace_back(
                    tree.template stealNode<LeafType>(leaf.origin(),
                        zeroVal<ValueType>(), false));
            }
        }, false);

        mManager.rebuildLeafArray();
        this->copyMasks(nodeMasks);
    }

    if (nodes.empty()) return;

    AccessorType accessor(tree);
    // Add back all dense nodes and rebuild the leaf manager
    for (auto& node : nodes) {
        accessor.addLeaf(node.release());
    }
    mManager.rebuildLeafArray();
}

template <typename TreeType>
void Morphology<TreeType>::dilateVoxels(const size_t iter,
    const NearestNeighbors nn,
    const ValueType* active)
{
    if (iter == 0) return;
    auto& tree = mManager.tree();

    if (this->getGrainSize() > 0) {
        using LeafT = typename MaskTreeT::LeafNodeType;

        // Build mask leaf nodes
        std::vector<LeafT*> array;
        array.resize(mManager.leafCount());
        tbb::parallel_for(mManager.getRange(),
            [&](const tbb::blocked_range<size_t>& r){
            for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                array[idx] = new LeafT(mManager.leaf(idx), false, TopologyCopy());
            }
        });

        // @todo this grain size is required for optimal threading - shouldn't be exposed
        const size_t numThreads = size_t(tbb::task_scheduler_init::default_num_threads());
        const size_t grainSize = math::Max(size_t(1), array.size()/(2*numThreads));

        // perform recursive dilation to sub trees - ensure the grain size
        // is set to zero for subsequent dilation
        tbb::enumerable_thread_specific<MaskTreeT> pool;
        LeafT** start = array.data();
        tbb::parallel_for(tbb::blocked_range<LeafT**>(start, start + array.size(), grainSize),
            [&](const tbb::blocked_range<LeafT**>& range) {
                MaskTreeT mask;
                for (LeafT** it = range.begin(); it != range.end(); ++it) mask.addLeaf(*it);
                tree::LeafManager<MaskTreeT> manager(mask, range.begin(), range.end());
                Morphology<MaskTreeT> m(manager);
                m.setGrainSize(0);
                m.dilateVoxels(iter, nn, true);
                pool.local().merge(mask, MERGE_ACTIVE_STATES);
            });

        auto iter = pool.begin();
        MaskTreeT& mask = *iter;
        ++iter;
        for (; iter != pool.end(); ++iter) mask.merge(*iter);
        tree.topologyUnion(mask);
    }
    else {
        AccessorType accessor(tree);
        std::unique_ptr<tree::ValueAccessor<const MaskTreeT>> mask;
        if (mMask) mask.reset(new tree::ValueAccessor<const MaskTreeT>(*mMask));

        // build cache objects
        NodeMaskOp cache(accessor, nn, mask.get());
        std::vector<MaskType> nodeMasks;
        std::vector<std::unique_ptr<LeafType>> nodes;

        for (size_t i = 0; i < iter; ++i) {
            // If the leaf count is zero, we can stop dilation. This will only
            // occur on the first iteration or if we're using a mask
            const size_t leafCount = mManager.leafCount();
            if (leafCount == 0) return;

            // Resize the node mask cache on each iteration. As we're stealing/replacing
            // dense nodes, it's possible we don't need to re-allocate the cache.
            if (nodeMasks.size() < leafCount) nodeMasks.resize(leafCount);
            this->copyMasks(nodeMasks);

            // For each node, dilate the mask into itself and neighbouring leaf nodes.
            // If the node was originally dense (all active), steal/replace it so
            // subsequent iterations are faster
            mManager.foreach([&](LeafType& leaf, const size_t idx) {
                // original bit-mask of current leaf node
                const MaskType& oldMask = nodeMasks[idx];
                const bool dense = oldMask.isOn();
                cache.dilate(leaf, oldMask);
                if (!dense) return;
                // This node does not need to be visited again - replace or steal
                if (active) {
                    // if an active value as been supplied, replace this dense
                    // leaf with an active tile
                    accessor.addTile(1, leaf.origin(), *active, true);
                }
                else {
                    // otherwise, temporarily steal this node
                    nodes.emplace_back(
                        tree.template stealNode<LeafType>(leaf.origin(),
                            zeroVal<ValueType>(), true));
                }
            }, false);

            mManager.rebuildLeafArray();
        }

        if (nodes.empty()) return;
        // Add back all dense nodes and rebuild the leaf manager
        for (auto& node : nodes) {
            accessor.addLeaf(node.release());
        }
        mManager.rebuildLeafArray();
    }
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


/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED
