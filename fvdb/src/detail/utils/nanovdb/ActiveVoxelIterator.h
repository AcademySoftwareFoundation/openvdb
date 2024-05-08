#pragma once

#include "CustomAccessors.h"


namespace fvdb {

/*
 * Hack to extract true data type within a nanovdb::NanoTree.
 * We need this because nanovdb::NanoTree<nanovdb::ValueOnIndex> has a data type mapping to
 * nanovdb::ValueOnIndex (which is a dummy struct), but the actual data is uint64_t.
 *
 * FIXME: We should mvoe this inside ActiveVoxelIterator to not pollute the namespace
 */
template <typename TreeT>
struct ActiveVoxelIteratorDataTypeExtractor {
    using DataType = typename TreeT::DataType;
};
template <>
struct ActiveVoxelIteratorDataTypeExtractor<nanovdb::NanoTree<nanovdb::ValueOnIndex>> {
    using DataType = int64_t;
};
template <>
struct ActiveVoxelIteratorDataTypeExtractor<nanovdb::NanoTree<nanovdb::ValueOnIndexMask>> {
    using DataType = int64_t;
};


/*
 * Const iterator over voxels in a nanovdb Index grid

 * Usage:
 *  for (auto it = NvdbIndexGridVoxelIterator(tree); it.isValid(); it++) {
 *      nanovdb::Coord active_voxel = it->first;
 *      uint64_t data_offset = it->second;
 *  }
 *
 * Each iterated item returns a pair (ijk, offset) where
 *   ijk is the coordinate of the active voxel
 *   offset is the offset into the index grid
 */
template <typename GridType, int64_t Offset=0>
struct ActiveVoxelIterator {
    // Iterator traits from std::iterator.
    using TreeT = typename nanovdb::NanoTree<GridType>;
    using DataType = typename ActiveVoxelIteratorDataTypeExtractor<TreeT>::DataType;
    using value_type = std::pair<nanovdb::Coord, DataType>;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    using LeafT = typename TreeT::LeafNodeType;

    ActiveVoxelIterator() = delete;

    ActiveVoxelIterator(const nanovdb::NanoTree<GridType>& tree, bool ignoreMasked = false, int64_t baseOffset = 0) {
        mLeaves = tree.template getFirstNode<0>();
        mNumLeaves = tree.nodeCount(0);
        mCurrentLeaf = 0;
        mCurrentLeafOffset = 0;
        mIgnoreMasked = ignoreMasked;
        mBaseOffset = baseOffset;

        // Move iterator to the first active voxel (or to the end if the tree is empty)
        moveToNextActiveVoxel();
    };

    bool isValid() {
        return mLeaves != nullptr;
    }

    // Dereferencable.
    const value_type& operator*() const {
        return mCurrentVoxelAndValue;
    }

    const value_type* operator->() const {
        return (const value_type*) &mCurrentVoxelAndValue;
    }

    const ActiveVoxelIterator& operator++() {
        mCurrentLeafOffset += 1;
        moveToNextActiveVoxel();
        return *this;
    }

    ActiveVoxelIterator operator++(int) {
        ActiveVoxelIterator tmp = *this; ++(*this); return tmp;
    }

    // Equality / inequality.
    bool operator==(const ActiveVoxelIterator& rhs) {
        return mLeaves == rhs.mLeaves &&
               mCurrentLeaf == rhs.mCurrentLeaf &&
               mCurrentLeafOffset == rhs.mCurrentLeafOffset;
    }
    bool operator!=(const ActiveVoxelIterator& rhs) {
        return !(*this == rhs);
    }

private:
    void moveToNextActiveVoxel() {
        for (uint64_t li = mCurrentLeaf; li < mNumLeaves; li += 1) {
            const LeafT& leaf = mLeaves[li];
            for (uint32_t lo = mCurrentLeafOffset; lo < LeafT::NUM_VALUES; lo += 1) {
                const bool isActive = mIgnoreMasked ? leaf.isActive(lo) : leaf.template get<fvdb::ActiveOrUnmasked<GridType>>(lo);
                if (isActive) {
                    mCurrentVoxelAndValue = std::make_pair(
                        leaf.offsetToGlobalCoord(lo),
                        (int64_t) leaf.getValue(lo) + Offset + mBaseOffset);

                    mCurrentLeafOffset = lo;
                    mCurrentLeaf = li;
                    return;
                }
            }
            mCurrentLeafOffset = 0;
        }
        mLeaves = nullptr;
        return;
    }
    const LeafT* mLeaves = nullptr;
    uint32_t mCurrentLeafOffset = 0;
    uint64_t mCurrentLeaf = 0;
    uint64_t mNumLeaves = 0;
    value_type mCurrentVoxelAndValue;
    bool mIgnoreMasked = false;
    int64_t mBaseOffset = 0;
};


template <typename GridType>
struct ActiveVoxelIteratorIJKOnly {
    using TreeT = typename nanovdb::NanoTree<GridType>;
    using LeafT = typename nanovdb::NanoTree<GridType>::LeafNodeType;
    ActiveVoxelIteratorIJKOnly() = delete;
    ActiveVoxelIteratorIJKOnly(const TreeT& tree) {
        mLeaves = tree.template getFirstNode<0>();
        mNumLeaves = tree.nodeCount(0);
        mCurrentLeaf = 0;
        mCurrentLeafOffset = 0;
        moveToNextActiveVoxel();
    };

    bool isValid() {
        return mLeaves != nullptr;
    }

    // Dereferencable.
    const nanovdb::Coord& operator*() const {
        return mCurrentIjk;
    }

    const ActiveVoxelIteratorIJKOnly& operator++() {
        mCurrentLeafOffset += 1;
        moveToNextActiveVoxel();
        return *this;
    }

    ActiveVoxelIteratorIJKOnly operator++(int) {
        ActiveVoxelIteratorIJKOnly tmp = *this; ++(*this); return tmp;
    }

    // Equality / inequality.
    bool operator==(const ActiveVoxelIteratorIJKOnly& rhs) {
        return mLeaves == rhs.mLeaves &&
               mCurrentLeaf == rhs.mCurrentLeaf &&
               mCurrentLeafOffset == rhs.mCurrentLeafOffset;
    }
    bool operator!=(const ActiveVoxelIteratorIJKOnly& rhs) {
        return !(*this == rhs);
    }

private:
    void moveToNextActiveVoxel() {
        for (uint64_t li = mCurrentLeaf; li < mNumLeaves; li += 1) {
            const LeafT& leaf = mLeaves[li];
            for (uint32_t lo = mCurrentLeafOffset; lo < LeafT::NUM_VALUES; lo += 1) {
                if (leaf.isActive(lo)) {
                    mCurrentIjk = leaf.offsetToGlobalCoord(lo);
                    mCurrentLeafOffset = lo;
                    mCurrentLeaf = li;
                    return;
                }
            }
            mCurrentLeafOffset = 0;
        }
        mLeaves = nullptr;
    }
    const LeafT* mLeaves = nullptr;
    uint32_t mCurrentLeafOffset = 0;
    uint64_t mCurrentLeaf = 0;
    uint64_t mNumLeaves = 0;
    nanovdb::Coord mCurrentIjk;
};
} // namespace fvdb