// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tree/NeighborAccessor.h
///
/// @brief Read only access to one leaf-sized region of a tree and the 26
/// regions around it, for stencils, sampling and reads of aligned grids.
///
/// @details A NeighborAccessor is bound to one leaf-sized region of the tree.
/// Each region can be a leaf, a tile or the background. Reads take the position
/// of a voxel in the bound region, from a
/// leaf iterator or as a linear offset, so they need no coordinate and no
/// search of the tree. A read at an offset from that voxel may fall in one of
/// the 26 regions around the bound region, which are looked up on first use.
/// A read farther away is cached in one more region, for the last region read there.
///
/// Read another grid at the voxels of a leaf:
/// @code
/// tree::NeighborAccessor acc(densityTree, leaf.origin());
/// for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
///     sum += acc.getValue(iter);
/// }
/// @endcode
///
/// Read a stencil, with one accessor per thread that moves from leaf to leaf:
/// @code
/// tree::NeighborAccessor acc(tree);
/// for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
///     acc.moveTo(leafIter->origin());
///     for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
///         laplacian = acc.getValue<1,0,0>(iter) + acc.getValue<-1,0,0>(iter) + ...
///     }
/// }
/// @endcode
///
/// Sample at a position, with offsets from the voxel at @c base:
/// @code
/// acc.moveTo(base);
/// const Index offset = LeafNodeType::coordToOffset(base);
/// value = acc.getValue<0,0,0>(offset) * w0 + acc.getValue<1,0,0>(offset) * w1 + ...
/// @endcode
///
/// Read whole rows along z, with one voxel per SIMD lane:
/// @code
/// acc.moveTo(leaf.origin());
/// for (Index rowOffset = 0; rowOffset < LeafNodeType::SIZE; rowOffset += LeafNodeType::DIM) {
///     const auto row = acc.getRow<-1,0,0>(rowOffset);
///     const auto previous = acc.getRow<0,0,-int(LeafNodeType::DIM)>(rowOffset);
///     ... // load row.data if row.contiguous, otherwise broadcast row.data[0]
/// }
/// @endcode
///
/// Read the value masks of a leaf and of its face neighbors, as a dilation does.
/// The template arguments count leaf-sized regions, not voxels:
/// @code
/// acc.moveTo(leaf.origin());
/// const auto& center = acc.getValueMask();
/// const auto& below = acc.getValueMask<0,0,-1>();
/// @endcode
///
/// @par Contract
/// From construction (or the last call to clear()) to the last use, nothing
/// may change the topology of the tree or swap a leaf buffer. This includes
/// adding, removing, pruning or deleting nodes, and replacing a tile with a
/// child node. RegionAndOffset values, voxel active states and tile values may change,
/// except that the tile values of a bool or mask tree are read when a region
/// is looked up. A change to the active state of a tile is not seen by
/// getValueMask(), isValueOn() or probeValue(). Call clear() and then
/// moveTo() after any such change. Every leaf buffer must be allocated. A
/// NeighborAccessor is not thread safe, so use one per thread.

#ifndef OPENVDB_TREE_NEIGHBORACCESSOR_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_NEIGHBORACCESSOR_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Platform.h>
#include <openvdb/Types.h>
#include <openvdb/util/Assert.h>

#include <array>
#include <cstdint>
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// @cond OPENVDB_DOCS_INTERNAL

namespace neighbor_accessor_internal
{

// A leaf-sized region of a tree whose leaves store an array of values
template<typename LeafT> struct ValueData;

// A leaf-sized region of a bool or mask tree, whose leaves store one bit per value
template<typename LeafT> struct MaskData;

template<typename LeafT>
using ValueOrMaskData = std::conditional_t<std::is_same_v<typename LeafT::ValueType, bool>,
    MaskData<LeafT>, ValueData<LeafT>>;

// A region the size of a parent node of the leaves
template<typename NodeT> struct ParentRegion;

// The regions looked up on first use
template<typename LeafT, typename ParentT> struct Cache;

} // namespace neighbor_accessor_internal

/// @endcond


template<typename TreeT>
class NeighborAccessor
{
public:
    /// Remove constness, this is always a read-only accessor
    using TreeType = std::remove_const_t<TreeT>;
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using RootNodeType = typename TreeType::RootNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;

    /// @brief Bind to nothing. Call moveTo() before the first read.
    explicit NeighborAccessor(const TreeType& tree);

    /// Bind to the leaf-sized region that contains @a xyz
    NeighborAccessor(const TreeType& tree, const Coord& xyz);

    /// @brief Bind to the leaf-sized region that contains @a xyz.
    /// @details The region may be a leaf, a tile or the background. Moving to a
    /// new coordinate that still lives within the same region is a no-op.
    OPENVDB_FORCE_INLINE void moveTo(const Coord& xyz)
    {
        const Coord origin = xyz & ~Int32(LeafNodeType::DIM - 1);
        if (origin != mOrigin) this->bind(origin);
    }

    /// @brief Bind to the leaf that contains @a xyz and throw if not a leaf.
    /// @throw LookupError if the region is a tile or the background.
    void moveToLeaf(const Coord& xyz);

    /// @brief Return the value of the voxel at @a offsetOrIter in the bound region.
    /// @details @a offsetOrIter is a linear offset or a leaf value iterator. Enable asserts
    /// to check the validity of the offset, iterator or bound region.
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE const ValueType& getValue(const IndexOrIterT& offsetOrIter) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        return mCache.regions[Center].getValue(offset);
    }

    /// @brief Return the value of the voxel that is relative to @a offsetOrIter using a
    /// local offset of (@a DX, @a DY, @a DZ) in the bound region.
    /// @details @a offsetOrIter is a linear offset or a leaf value iterator. Enable asserts
    /// to check the validity of the offset, iterator or bound region.
    /// @note An offset of more than the leaf dimension in any axis will fall back to a tree
    /// or parent node lookup and will not be as performant.
    template<int DX, int DY, int DZ, typename IndexOrIterT>
    OPENVDB_FORCE_INLINE const ValueType& getValue(const IndexOrIterT& offsetOrIter) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        const RegionAndOffset r = this->template applyRelativeOffset<DX, DY, DZ>(offset);
        return r.region->getValue(r.offset);
    }

    /// @brief Return the value of the voxel that is relative to @a offsetOrIter using a
    /// local offset of (@a dx, @a dy, @a dz) in the bound region.
    /// @details For stencils with a width known at run time. @a offsetOrIter is a
    /// linear offset or a leaf value iterator. Enable asserts to check the validity
    /// of the offset, iterator or bound region.
    /// @note A voxel outside the 26 regions around the bound region is slower.
    /// It is read from one more cached region, which holds the last region read there and
    /// is filled from the parent node of the bound region or of a neighbor region.
    /// A voxel more than one parent node away fills it with a search from the root.
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE const ValueType&
    getValue(const IndexOrIterT& offsetOrIter, int dx, int dy, int dz) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        const RegionAndOffset r = this->applyRelativeOffset(offset, dx, dy, dz);
        return r.region->getValue(r.offset);
    }

    /// @brief Return true if the voxel at @a offsetOrIter in the bound region is active.
    /// @details @a offsetOrIter is a linear offset or a leaf value iterator. Enable asserts
    /// to check the validity of the offset, iterator or bound region. A voxel of a tile
    /// has the active state of the tile, and a voxel of the background is inactive.
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE bool isValueOn(const IndexOrIterT& offsetOrIter) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        return mCache.regions[Center].isValueOn(offset);
    }

    /// @brief Return true if the voxel that is relative to @a offsetOrIter using a
    /// local offset of (@a DX, @a DY, @a DZ) in the bound region is active.
    /// @details As for isValueOn(offsetOrIter) and getValue<DX, DY, DZ>(offsetOrIter).
    /// @note An offset of more than the leaf dimension in any axis will fall back to a tree
    /// or parent node lookup and will not be as performant.
    template<int DX, int DY, int DZ, typename IndexOrIterT>
    OPENVDB_FORCE_INLINE bool isValueOn(const IndexOrIterT& offsetOrIter) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        const RegionAndOffset r = this->template applyRelativeOffset<DX, DY, DZ>(offset);
        return r.region->isValueOn(r.offset);
    }

    /// @brief Return true if the voxel that is relative to @a offsetOrIter using a
    /// local offset of (@a dx, @a dy, @a dz) in the bound region is active.
    /// @details As for isValueOn(offsetOrIter) and getValue(offsetOrIter, dx, dy, dz).
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE bool isValueOn(const IndexOrIterT& offsetOrIter, int dx, int dy, int dz) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        const RegionAndOffset r = this->applyRelativeOffset(offset, dx, dy, dz);
        return r.region->isValueOn(r.offset);
    }

    /// @brief Set @a value to the value of the voxel at @a offsetOrIter in the
    /// bound region, and return true if that voxel is active.
    /// @details As for getValue(offsetOrIter) and isValueOn(offsetOrIter).
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE bool probeValue(const IndexOrIterT& offsetOrIter, ValueType& value) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        value = mCache.regions[Center].getValue(offset);
        return mCache.regions[Center].isValueOn(offset);
    }

    /// @brief Set @a value to the value of the voxel that is relative to @a offsetOrIter
    /// using a local offset of (@a DX, @a DY, @a DZ) in the bound region, and return true
    /// if that voxel is active.
    /// @details As for getValue<DX, DY, DZ>(offsetOrIter) and isValueOn<DX, DY, DZ>(offsetOrIter).
    /// @note An offset of more than the leaf dimension in any axis will fall back to a tree
    /// or parent node lookup and will not be as performant.
    template<int DX, int DY, int DZ, typename IndexOrIterT>
    OPENVDB_FORCE_INLINE bool probeValue(const IndexOrIterT& offsetOrIter, ValueType& value) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        const RegionAndOffset r = this->template applyRelativeOffset<DX, DY, DZ>(offset);
        value = r.region->getValue(r.offset);
        return r.region->isValueOn(r.offset);
    }

    /// @brief Set @a value to the value of the voxel that is relative to @a offsetOrIter
    /// using a local offset of (@a dx, @a dy, @a dz) in the bound region, and return true
    /// if that voxel is active.
    /// @details As for getValue(offsetOrIter, dx, dy, dz) and isValueOn(offsetOrIter, dx, dy, dz).
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE bool
    probeValue(const IndexOrIterT& offsetOrIter, int dx, int dy, int dz, ValueType& value) const
    {
        const Index offset = this->toOffset(offsetOrIter);
        const RegionAndOffset r = this->applyRelativeOffset(offset, dx, dy, dz);
        value = r.region->getValue(r.offset);
        return r.region->isValueOn(r.offset);
    }

    /// @brief Return the value of the voxel at @a offsetOrIter in the bound region,
    /// without the check for a tile.
    /// @details @a offsetOrIter is a linear offset or a leaf value iterator. Enable asserts
    /// to check the validity of the offset, iterator or bound region.
    /// @note Use getValue() for a safer alternative.
    /// @warning The bound region must be a leaf, as moveToLeaf() guarantees.
    /// Undefined behavior if it is a tile or the background.
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE const ValueType& getLeafValueUnsafe(const IndexOrIterT& offsetOrIter) const
    {
        OPENVDB_ASSERT(mCache.regions[Center].isLeaf());
        const Index offset = this->toOffset(offsetOrIter);
        return mCache.regions[Center].getLeafValue(offset);
    }

    /// @brief Return the value of the voxel that is relative to @a offsetOrIter using a
    /// local offset of (@a DX, @a DY, @a DZ) in the bound region, without the check for a tile.
    /// @details @a offsetOrIter is a linear offset or a leaf value iterator. Enable asserts
    /// to check the validity of the offset, iterator or bound region.
    /// @note For a mask or an operation that never reads outside the bound leaf. There is
    /// no test for a neighbor region, so this is as fast as reading the leaf directly.
    /// @warning The bound region must be a leaf, as moveToLeaf() guarantees, and the
    /// voxel must be in it. Undefined behavior otherwise.
    template<int DX, int DY, int DZ, typename IndexOrIterT>
    OPENVDB_FORCE_INLINE const ValueType& getLeafValueUnsafe(const IndexOrIterT& offsetOrIter) const
    {
        constexpr int dim = int(LeafNodeType::DIM);
        [[maybe_unused]] constexpr int log2dim = int(LeafNodeType::LOG2DIM);
        static_assert(DX > -dim && DX < dim && DY > -dim && DY < dim && DZ > -dim && DZ < dim,
            "the offset must be less than the leaf dimension in each axis");
        OPENVDB_ASSERT(mCache.regions[Center].isLeaf());
        const Index offset = this->toOffset(offsetOrIter);
        OPENVDB_ASSERT(
            int(offset >> (2 * log2dim)) + DX >= 0 && int(offset >> (2 * log2dim)) + DX < dim &&
            int((offset >> log2dim) & (dim - 1)) + DY >= 0 &&
            int((offset >> log2dim) & (dim - 1)) + DY < dim &&
            int(offset & (dim - 1)) + DZ >= 0 && int(offset & (dim - 1)) + DZ < dim);
        constexpr int step = DX * dim * dim + DY * dim + DZ;
        return mCache.regions[Center].getLeafValue(Index(int(offset) + step));
    }

    /// A row of LeafNodeType::DIM voxels along z, as returned by getRow()
    struct Row;

    /// @brief Return the row along z at (@a DX, @a DY, @a DZ) from the row that
    /// starts at linear offset @a rowOffset in the bound region.
    /// @details @a rowOffset is the offset of a voxel with z = 0. @a DZ is
    /// -DIM, 0 or DIM, so that the row lies in one region. A row of a leaf is
    /// DIM values in a row in memory, for one vector load.
    template<int DX, int DY, int DZ>
    OPENVDB_FORCE_INLINE Row getRow(Index rowOffset) const
    {
        constexpr int dim = int(LeafNodeType::DIM);
        constexpr int log2dim = int(LeafNodeType::LOG2DIM);
        static_assert(!std::is_same_v<ValueType, bool>,
            "rows need leaves that store an array of values");
        static_assert(DX >= -dim && DX <= dim && DY >= -dim && DY <= dim,
            "a row is at most one leaf away in x and y");
        static_assert(DZ == -dim || DZ == 0 || DZ == dim, "DZ is -DIM, 0 or DIM");
        OPENVDB_ASSERT(this->isBound() && rowOffset < Index(LeafNodeType::SIZE));
        OPENVDB_ASSERT((rowOffset & Index(dim - 1)) == 0);

        int x = int(rowOffset >> (2 * log2dim));
        int y = int((rowOffset >> log2dim) & (dim - 1));
        int regionIndex = Center;
        // The arithmetic shift gives -1, 0 or 1 for the region crossed in each axis
        if constexpr (DX != 0) { x += DX; regionIndex += (x >> log2dim) * 9; x &= dim - 1; }
        if constexpr (DY != 0) { y += DY; regionIndex += (y >> log2dim) * 3; y &= dim - 1; }
        if constexpr (DZ != 0) { regionIndex += DZ / dim; }
        constexpr bool neighbor = DX != 0 || DY != 0 || DZ != 0;
        const ValueOrMaskData& region =
            neighbor ? this->neighborRegion(regionIndex) : mCache.regions[Center];
        const Index start = Index((x << (2 * log2dim)) | (y << log2dim));
        return Row{region.data + (start & region.indexMask), region.isLeaf()};
    }

    /// @brief Return the value mask of the region at (@a I, @a J, @a K) from the bound region.
    /// @details @a I, @a J and @a K count leaf-sized regions, from -1 to 1, not
    /// voxels. A leaf gives its own mask, so later changes to it are seen. A
    /// tile or the background gives a full mask if it is active and an empty
    /// mask if not.
    template<int I = 0, int J = 0, int K = 0>
    OPENVDB_FORCE_INLINE const NodeMaskType& getValueMask() const
    {
        return *this->template regionAt<I, J, K>().mask;
    }

    /// @brief Return true if the region at (@a I, @a J, @a K) from the bound region is a leaf.
    /// @details @a I, @a J and @a K count leaf-sized regions, from -1 to 1, not voxels.
    template<int I = 0, int J = 0, int K = 0>
    OPENVDB_FORCE_INLINE bool isLeaf() const
    {
        return this->template regionAt<I, J, K>().isLeaf();
    }

    /// Return the origin of the bound region, or Coord::max() if there is none
    const Coord& origin() const { return mOrigin; }

    const TreeType& tree() const { return *mTree; }

    /// Forget the bound region and every region looked up. Call moveTo() before the next read.
    void clear();

private:
    // The node above the leaves, which is the root in a tree of two levels
    using ParentNodeType = typename RootNodeType::NodeChainType::template Get<1>;
    static constexpr bool ParentIsRoot = std::is_same_v<ParentNodeType, RootNodeType>;

    using ValueOrMaskData = neighbor_accessor_internal::ValueOrMaskData<LeafNodeType>;
    using ParentRegion = neighbor_accessor_internal::ParentRegion<ParentNodeType>;
    using Cache = neighbor_accessor_internal::Cache<LeafNodeType, ParentNodeType>;

    static constexpr int Center = 13;

    bool isBound() const { return (mCache.valid >> Center) & 1u; }

    // An integer is a linear offset. Anything else is taken as a leaf value iterator,
    // so that an iterator never converts to an offset through its operator bool.
    template<typename IndexOrIterT>
    OPENVDB_FORCE_INLINE Index toOffset(const IndexOrIterT& offsetOrIter) const
    {
        OPENVDB_ASSERT(this->isBound());
        if constexpr (std::is_integral_v<IndexOrIterT>) {
            OPENVDB_ASSERT(Index(offsetOrIter) < Index(LeafNodeType::SIZE));
            return Index(offsetOrIter);
        } else {
            using SourceLeafT = std::decay_t<decltype(offsetOrIter.parent())>;
            static_assert(SourceLeafT::LEVEL == 0, "NeighborAccessor needs a leaf value iterator");
            static_assert(SourceLeafT::LOG2DIM == LeafNodeType::LOG2DIM,
                "the iterator and the tree need the same leaf size");
            OPENVDB_ASSERT(offsetOrIter.parent().origin() == mOrigin);
            return offsetOrIter.pos();
        }
    }

    // A voxel, as its region and its linear offset in that region
    struct RegionAndOffset
    {
        const ValueOrMaskData* region;
        Index offset;
    };

    // The voxel at (DX, DY, DZ) from the voxel at @a offset in the bound region
    template<int DX, int DY, int DZ>
    OPENVDB_FORCE_INLINE RegionAndOffset applyRelativeOffset(Index offset) const
    {
        [[maybe_unused]] constexpr int dim = int(LeafNodeType::DIM);
        [[maybe_unused]] constexpr int log2dim = int(LeafNodeType::LOG2DIM);
        if constexpr (DX == 0 && DY == 0 && DZ == 0) {
            return RegionAndOffset{&mCache.regions[Center], offset};
        } else if constexpr (DX < -dim || DX > dim || DY < -dim || DY > dim ||
                DZ < -dim || DZ > dim) {
            // Some voxels may still be in a neighbor region, which only the run time test finds
            return this->applyRelativeOffset(offset, DX, DY, DZ);
        } else {
            int x = int(offset >> (2 * log2dim));
            int y = int((offset >> log2dim) & (dim - 1));
            int z = int(offset & (dim - 1));
            int regionIndex = Center;
            // The arithmetic shift gives -1, 0 or 1 for the region crossed in each axis
            if constexpr (DX != 0) { x += DX; regionIndex += (x >> log2dim) * 9; x &= dim - 1; }
            if constexpr (DY != 0) { y += DY; regionIndex += (y >> log2dim) * 3; y &= dim - 1; }
            if constexpr (DZ != 0) { z += DZ; regionIndex += (z >> log2dim); z &= dim - 1; }
            return RegionAndOffset{&this->neighborRegion(regionIndex),
                Index((x << (2 * log2dim)) | (y << log2dim) | z)};
        }
    }

    // The voxel at (@a dx, @a dy, @a dz) from the voxel at @a offset in the bound region
    OPENVDB_FORCE_INLINE RegionAndOffset
    applyRelativeOffset(Index offset, int dx, int dy, int dz) const
    {
        constexpr int dim = int(LeafNodeType::DIM);
        constexpr int log2dim = int(LeafNodeType::LOG2DIM);
        int x = int(offset >> (2 * log2dim)) + dx;
        int y = int((offset >> log2dim) & (dim - 1)) + dy;
        int z = int(offset & (dim - 1)) + dz;
        if (x < -dim || x >= 2 * dim || y < -dim || y >= 2 * dim || z < -dim || z >= 2 * dim) {
            const Coord xyz = mOrigin.offsetBy(x, y, z);
            return RegionAndOffset{&this->farRegion(xyz), LeafNodeType::coordToOffset(xyz)};
        }
        const int regionIndex = Center + (x >> log2dim) * 9 + (y >> log2dim) * 3 + (z >> log2dim);
        x &= dim - 1;
        y &= dim - 1;
        z &= dim - 1;
        return RegionAndOffset{&this->neighborRegion(regionIndex),
            Index((x << (2 * log2dim)) | (y << log2dim) | z)};
    }

    OPENVDB_FORCE_INLINE const ValueOrMaskData& neighborRegion(int regionIndex) const
    {
        if (!((mCache.valid >> regionIndex) & 1u)) this->fillNeighbor(regionIndex);
        return mCache.regions[regionIndex];
    }

    // The region that contains @a xyz, outside the 26 regions around the bound region
    OPENVDB_FORCE_INLINE const ValueOrMaskData& farRegion(const Coord& xyz) const
    {
        const Coord origin = xyz & ~Int32(LeafNodeType::DIM - 1);
        if (origin != mCache.farOrigin) this->fillFar(origin);
        return mCache.far;
    }

    template<int I, int J, int K>
    OPENVDB_FORCE_INLINE const ValueOrMaskData& regionAt() const
    {
        static_assert(I >= -1 && I <= 1 && J >= -1 && J <= 1 && K >= -1 && K <= 1,
            "I, J and K count leaf-sized regions, from -1 to 1");
        constexpr int regionIndex = Center + I * 9 + J * 3 + K;
        OPENVDB_ASSERT(this->isBound());
        if constexpr (regionIndex == Center) {
            return mCache.regions[Center];
        } else {
            return this->neighborRegion(regionIndex);
        }
    }

    // Bind to the region at @a origin and fill its cached data. Not inlined, since a
    // move to a new region is rare and moveTo() is inlined at every call.
    OPENVDB_NO_INLINE void bind(const Coord& origin);

    // Fill a neighbor region on its first use
    OPENVDB_NO_INLINE void fillNeighbor(int regionIndex) const;

    // Fill the far region from the region at @a origin
    OPENVDB_NO_INLINE void fillFar(const Coord& origin) const;

    // Return the parent region at @a origin, which searches the root if it is not
    // in entry @a parentIndex. A region stays cached until another region needs its
    // entry. The two neighbors across one axis share an entry, so a walk that reads
    // both of them searches the root again each time it moves from one face of a
    // parent region to the other.
    const ParentRegion& parentRegion(int parentIndex, const Coord& origin) const;

    // Fill @a region from the child at @a childIndex of @a parent, a leaf or a tile
    static void fillFromParent(
        ValueOrMaskData& region, const ParentRegion& parent, Index childIndex);

    // Fill @a region from the region at @a origin, with a search from the root
    void fillFromRoot(ValueOrMaskData& region, const Coord& origin) const;

    const TreeType* mTree;
    // Coord::max() has low bits set, so it never equals a region origin
    Coord mOrigin = Coord::max(); // origin of the bound region
    Index mChildIndex = 0;        // offset of the bound region in its parent node
    bool mInterior = false;       // all 26 neighbors are children of one parent node
    // Filled on first use by const reads
    mutable Cache mCache;
}; // class NeighborAccessor


////////////////////////////////////////


/// @cond OPENVDB_DOCS_INTERNAL

namespace neighbor_accessor_internal
{

// Function statics, so that they are built before the first use
template<typename NodeMaskT>
inline const NodeMaskT& tileMask(bool active)
{
    static const NodeMaskT fullMask(true);
    static const NodeMaskT emptyMask(false);
    return active ? fullMask : emptyMask;
}

// A read is one load and an "and", with no branch.
template<typename LeafT>
struct ValueData
{
    using ValueType = typename LeafT::ValueType;
    using NodeMaskType = typename LeafT::NodeMaskType;

    void setLeaf(const LeafT& leaf)
    {
        OPENVDB_ASSERT_MESSAGE(!leaf.buffer().empty(), "NeighborAccessor needs allocated leaf buffers");
        data = leaf.buffer().data();
        indexMask = Index(LeafT::SIZE) - 1;
        mask = &leaf.getValueMask();
    }

    // @a value is the tile value or the background in place, so a change to it is seen
    void setTile(const ValueType* value, bool active)
    {
        data = value;
        indexMask = 0;
        mask = &tileMask<NodeMaskType>(active);
    }

    bool isLeaf() const { return indexMask != 0; }
    const ValueType& getValue(Index offset) const { return data[offset & indexMask]; }
    bool isValueOn(Index offset) const { return mask->isOn(offset); }
    const ValueType& getLeafValue(Index offset) const { return data[offset]; }

    const ValueType* data = nullptr;    // leaf values, or the tile value
    Index indexMask = 0;                // SIZE - 1 for a leaf, 0 for a tile
    const NodeMaskType* mask = nullptr; // leaf value mask, or a full or empty mask for a tile
}; // struct ValueData

// A tile reads its value from a full or empty mask, so a read has no test for a tile.
template<typename LeafT>
struct MaskData
{
    using NodeMaskType = typename LeafT::NodeMaskType;

    void setLeaf(const LeafT& leaf)
    {
        data = &leaf.buffer().storage();
        mask = &leaf.getValueMask();
    }

    void setTile(const bool* value, bool active)
    {
        data = &tileMask<NodeMaskType>(*value);
        mask = &tileMask<NodeMaskType>(active);
    }

    // Only a leaf has a value mask of its own
    bool isLeaf() const
    {
        return mask != &tileMask<NodeMaskType>(true) && mask != &tileMask<NodeMaskType>(false);
    }

    const bool& getValue(Index offset) const
    {
        // Not the ternary operator, with which Visual C++ returns a reference to a temporary
        if (data->isOn(offset)) return LeafT::Buffer::sOn;
        return LeafT::Buffer::sOff;
    }

    const bool& getLeafValue(Index offset) const { return this->getValue(offset); }
    bool isValueOn(Index offset) const { return mask->isOn(offset); }

    const NodeMaskType* data = nullptr; // leaf values, or a full or empty mask for a tile value
    const NodeMaskType* mask = nullptr; // leaf value mask, or a full or empty mask for a tile
}; // struct MaskData

template<typename NodeT>
struct ParentRegion
{
    const NodeT* node = nullptr;                      // null for a tile or the background
    const typename NodeT::ValueType* value = nullptr; // the tile value or the background
    Coord origin = Coord::max();                      // Coord::max() never equals a region origin
    bool active = false;                              // the active state of that tile
}; // struct ParentRegion

template<typename LeafT, typename ParentT>
struct Cache
{
    std::array<ValueOrMaskData<LeafT>, 27> regions{}; // indexed (dx+1)*9 + (dy+1)*3 + (dz+1)
    uint32_t valid = 0;                         // one bit per filled region
    // Entry 0 is the parent region of the bound region. The others are next to it,
    // indexed by the axes crossed: 4 for x, 2 for y and 1 for z.
    std::array<ParentRegion<ParentT>, 8> parents{};
    // The last region read outside the 27 regions
    ValueOrMaskData<LeafT> far{};
    Coord farOrigin = Coord::max();
}; // struct Cache

} // namespace neighbor_accessor_internal

/// @endcond


////////////////////////////////////////


template<typename TreeT>
struct NeighborAccessor<TreeT>::Row
{
    /// The values of the row, or one value for the whole row
    const ValueType* data = nullptr;
    /// True if @c data holds a value for each voxel, false for a tile or the background
    bool contiguous = false;
}; // struct Row


template<typename TreeT>
inline
NeighborAccessor<TreeT>::NeighborAccessor(const TreeType& tree)
    : mTree(&tree)
{
}


template<typename TreeT>
inline
NeighborAccessor<TreeT>::NeighborAccessor(const TreeType& tree, const Coord& xyz)
    : mTree(&tree)
{
    this->moveTo(xyz);
}


template<typename TreeT>
inline void
NeighborAccessor<TreeT>::moveToLeaf(const Coord& xyz)
{
    this->moveTo(xyz);
    if (!mCache.regions[Center].isLeaf()) {
        OPENVDB_THROW(LookupError, "no leaf at " << mOrigin);
    }
}


template<typename TreeT>
inline void
NeighborAccessor<TreeT>::clear()
{
    *this = NeighborAccessor(*mTree);
}


// bind(), fillNeighbor() and fillFar() are not declared inline, so that an extern template
// declaration keeps them out of the code that includes this header.
template<typename TreeT>
void
NeighborAccessor<TreeT>::bind(const Coord& origin)
{
    mOrigin = origin;
    ValueOrMaskData& center = mCache.regions[Center];
    if constexpr (ParentIsRoot) {
        this->fillFromRoot(center, origin);
    } else {
        constexpr Index childDim = Index(1) << ParentNodeType::LOG2DIM;
        // So that a neighbor crosses into at most one other parent region in each axis
        static_assert(childDim >= 2, "NeighborAccessor needs parent nodes larger than leaves");
        const Coord parentOrigin = origin & ~Int32(ParentNodeType::DIM - 1);
        const ParentRegion& parent = this->parentRegion(0, parentOrigin);
        mChildIndex = ParentNodeType::coordToOffset(origin);
        // Away from the faces of its parent node, all 26 neighbors of a region are
        // children of the same node. Unsigned, so a child coordinate of 0 wraps and fails.
        const Index x = Index(origin[0] - parentOrigin[0]) >> LeafNodeType::TOTAL;
        const Index y = Index(origin[1] - parentOrigin[1]) >> LeafNodeType::TOTAL;
        const Index z = Index(origin[2] - parentOrigin[2]) >> LeafNodeType::TOTAL;
        mInterior = parent.node != nullptr &&
            x - 1 < childDim - 2 && y - 1 < childDim - 2 && z - 1 < childDim - 2;
        this->fillFromParent(center, parent, mChildIndex);
    }
    mCache.valid = 1u << Center;
    mCache.farOrigin = Coord::max();
}


template<typename TreeT>
void
NeighborAccessor<TreeT>::fillNeighbor(int regionIndex) const
{
    const int i = regionIndex / 9 - 1;
    const int j = (regionIndex / 3) % 3 - 1;
    const int k = regionIndex % 3 - 1;
    ValueOrMaskData& region = mCache.regions[regionIndex];
    if constexpr (ParentIsRoot) {
        constexpr int dim = int(LeafNodeType::DIM);
        this->fillFromRoot(region, mOrigin.offsetBy(i * dim, j * dim, k * dim));
    } else {
        constexpr int log2ChildDim = int(ParentNodeType::LOG2DIM);
        constexpr int childDim = 1 << log2ChildDim;
        if (mInterior) {
            const int childOffset = i * childDim * childDim + j * childDim + k;
            this->fillFromParent(region, mCache.parents[0], Index(int(mChildIndex) + childOffset));
        } else {
            int x = int(mChildIndex >> (2 * log2ChildDim)) + i;
            int y = int((mChildIndex >> log2ChildDim) & Index(childDim - 1)) + j;
            int z = int(mChildIndex & Index(childDim - 1)) + k;
            // The arithmetic shift gives -1, 0 or 1 for the parent region crossed in each axis
            const int crossX = x >> log2ChildDim;
            const int crossY = y >> log2ChildDim;
            const int crossZ = z >> log2ChildDim;
            constexpr int dim = int(ParentNodeType::DIM);
            const Coord parentOrigin =
                mCache.parents[0].origin.offsetBy(crossX * dim, crossY * dim, crossZ * dim);
            // One bit per axis crossed, since the bound region is on at most one face in each axis
            const int parentIndex = ((crossX & 1) << 2) | ((crossY & 1) << 1) | (crossZ & 1);
            x &= childDim - 1;
            y &= childDim - 1;
            z &= childDim - 1;
            const Index childIndex = Index((x << (2 * log2ChildDim)) | (y << log2ChildDim) | z);
            this->fillFromParent(region, this->parentRegion(parentIndex, parentOrigin), childIndex);
        }
    }
    mCache.valid |= 1u << regionIndex;
}


template<typename TreeT>
void
NeighborAccessor<TreeT>::fillFar(const Coord& origin) const
{
    mCache.farOrigin = origin;
    if constexpr (ParentIsRoot) {
        this->fillFromRoot(mCache.far, origin);
    } else {
        constexpr int log2ParentDim = int(ParentNodeType::TOTAL);
        const Coord parentOrigin = origin & ~Int32(ParentNodeType::DIM - 1);
        const Coord& boundParentOrigin = mCache.parents[0].origin;
        // 64 bits, since the far region may be at the other end of the coordinate range
        const Int64 crossX = (Int64(parentOrigin[0]) - boundParentOrigin[0]) >> log2ParentDim;
        const Int64 crossY = (Int64(parentOrigin[1]) - boundParentOrigin[1]) >> log2ParentDim;
        const Int64 crossZ = (Int64(parentOrigin[2]) - boundParentOrigin[2]) >> log2ParentDim;
        if (crossX < -1 || crossX > 1 || crossY < -1 || crossY > 1 || crossZ < -1 || crossZ > 1) {
            // The parent regions cached are at most one away in each axis
            this->fillFromRoot(mCache.far, origin);
        } else {
            // The same entry as a neighbor region that crosses the same axes
            const int parentIndex = int(((crossX & 1) << 2) | ((crossY & 1) << 1) | (crossZ & 1));
            this->fillFromParent(mCache.far, this->parentRegion(parentIndex, parentOrigin),
                ParentNodeType::coordToOffset(origin));
        }
    }
}


template<typename TreeT>
inline const typename NeighborAccessor<TreeT>::ParentRegion&
NeighborAccessor<TreeT>::parentRegion(int parentIndex, const Coord& origin) const
{
    ParentRegion& region = mCache.parents[parentIndex];
    if (region.origin != origin) {
        const RootNodeType& root = mTree->root();
        region = ParentRegion{};
        region.origin = origin;
        region.node = root.template probeConstNode<ParentNodeType>(origin);
        if (!region.node) {
            // One tile or the background covers all of the region
            region.value = &root.getValue(origin);
            region.active = root.isValueOn(origin);
        }
    }
    return region;
}


// The nodes return a reference to the tile value or the background in place
template<typename TreeT>
inline void
NeighborAccessor<TreeT>::fillFromParent(
    ValueOrMaskData& region, const ParentRegion& parent, Index childIndex)
{
    if (!parent.node) {
        region.setTile(parent.value, parent.active);
    } else if (parent.node->isChildMaskOn(childIndex)) {
        region.setLeaf(*parent.node->getChildUnsafe(childIndex));
    } else {
        region.setTile(
            &parent.node->getValueUnsafe(childIndex), parent.node->isValueOn(childIndex));
    }
}


template<typename TreeT>
inline void
NeighborAccessor<TreeT>::fillFromRoot(ValueOrMaskData& region, const Coord& origin) const
{
    const RootNodeType& root = mTree->root();
    if (const LeafNodeType* leaf = root.probeConstLeaf(origin)) {
        region.setLeaf(*leaf);
    } else {
        region.setTile(&root.getValue(origin), root.isValueOn(origin));
    }
}


////////////////////////////////////////


// The tree type is deduced and always const
template<typename TreeT>
NeighborAccessor(const TreeT&) -> NeighborAccessor<const TreeT>;
template<typename TreeT>
NeighborAccessor(const TreeT&, const Coord&) -> NeighborAccessor<const TreeT>;


/// @brief True if @a T is a NeighborAccessor that is not const, as for overloads
/// that move the accessor
template<typename T> struct IsNeighborAccessor: std::false_type {};
template<typename TreeT> struct IsNeighborAccessor<NeighborAccessor<TreeT>>: std::true_type {};

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NEIGHBORACCESSOR_HAS_BEEN_INCLUDED
