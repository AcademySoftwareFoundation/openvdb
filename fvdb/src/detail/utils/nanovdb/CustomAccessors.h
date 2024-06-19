#pragma once

#include <nanovdb/NanoVDB.h>

namespace fvdb {

// NOTE: When getters/setters are called, you are guaranteed that ijk does not map to a child of that node!



///  @brief Get/Set operation which returns ot sets whether a voxel is unmasked (in the case of ValueOnIndexMask)
///         or active or not (in the case of ValueOnIndex)=
template <typename BuildType>
struct ActiveOrUnmasked;
template <>
struct ActiveOrUnmasked<nanovdb::ValueOnIndex>{
    using BuildT = nanovdb::ValueOnIndex;
    __hostdev__ static bool get(const nanovdb::NanoRoot<BuildT>&) {return false;}
    __hostdev__ static bool get(const typename nanovdb::NanoRoot<BuildT>::Tile& tile) { return (bool) tile.state; }
    __hostdev__ static bool get(const nanovdb::NanoUpper<BuildT>&node, uint32_t n) {return node.mValueMask.isOn(n);}
    __hostdev__ static bool get(const nanovdb::NanoLower<BuildT>&node, uint32_t n) {return node.mValueMask.isOn(n);}
    __hostdev__ static bool get(const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n) {return leaf.mValueMask.isOn(n);}
}; // ActiveOrUnmasked<BuildT>
template <>
struct ActiveOrUnmasked<nanovdb::ValueOnIndexMask>{
    using BuildT = nanovdb::ValueOnIndexMask;
    __hostdev__ static bool get(const nanovdb::NanoRoot<BuildT>&) {return false;}
    __hostdev__ static bool get(const typename nanovdb::NanoRoot<BuildT>::Tile&) {return false;}
    __hostdev__ static bool get(const nanovdb::NanoUpper<BuildT>&, uint32_t) {return false;}
    __hostdev__ static bool get(const nanovdb::NanoLower<BuildT>&, uint32_t) {return false;}
    __hostdev__ static bool get(const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n) {return leaf.mMask.isOn(n);}
    __hostdev__ static void set(nanovdb::NanoRoot<nanovdb::ValueOnIndexMask>&, uint32_t) {}
    __hostdev__ static void set(typename nanovdb::NanoRoot<nanovdb::ValueOnIndexMask>::Tile&, uint32_t) {}
    __hostdev__ static void set(nanovdb::NanoUpper<BuildT>&, uint32_t) {}
    __hostdev__ static void set(nanovdb::NanoLower<BuildT>&, uint32_t) {}
    __hostdev__ static void set(nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n) {leaf.mMask.setOn(n);}
}; // ActiveOrUnmasked<BuildT>


/// @brief Set operation to set the mask of a ValueOnIndexMask node to the given value
struct AtomicMaskedStateSetOnlyHost {
    using BuildT = nanovdb::ValueOnIndexMask;
    __hostdev__ static void set(nanovdb::NanoRoot<BuildT>&, bool) {}
    __hostdev__ static void set(typename nanovdb::NanoRoot<BuildT>::Tile&, bool) {}
    __hostdev__ static void set(nanovdb::NanoUpper<BuildT>&, uint32_t, bool) {}
    __hostdev__ static void set(nanovdb::NanoLower<BuildT>&, uint32_t, bool) {}
    __hostdev__ static void set(nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n, bool value) {leaf.mMask.set(n, value);}
};

#if defined(__CUDACC__)
struct AtomicMaskedStateSetOnlyDevice {
    using BuildT = nanovdb::ValueOnIndexMask;
    __device__ static void set(nanovdb::NanoRoot<BuildT>&, bool) {}
    __device__ static void set(typename nanovdb::NanoRoot<BuildT>::Tile&, bool) {}
    __device__ static void set(nanovdb::NanoUpper<BuildT>&, uint32_t, bool) {}
    __device__ static void set(nanovdb::NanoLower<BuildT>&, uint32_t, bool) {}
    __device__ static void set(nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n, bool value) {leaf.mMask.setAtomic(n, value);}
};
#endif

/// @brief Get/Set operation which returns the total number of unmasked voxels in a leaf node (in the case of ValueOnIndexMask)
///        and the total number of active voxels in a leaf node (in the case of ValueOnIndex)
template <typename BuildType>
struct TotalUnmaskedPerLeaf;
template <>
struct TotalUnmaskedPerLeaf<nanovdb::ValueOnIndexMask> {
    using BuildT = nanovdb::ValueOnIndexMask;
    __hostdev__ static uint32_t get(const nanovdb::NanoRoot<BuildT>&) {return 0;}
    __hostdev__ static uint32_t get(const typename nanovdb::NanoRoot<BuildT>::Tile&) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoUpper<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLower<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n) {return leaf.mMask.countOn();}
};
template <>
struct TotalUnmaskedPerLeaf<nanovdb::ValueOnIndex> {
    using BuildT = nanovdb::ValueOnIndex;
    __hostdev__ static uint32_t get(const nanovdb::NanoRoot<BuildT>&) {return 0;}
    __hostdev__ static uint32_t get(const typename nanovdb::NanoRoot<BuildT>::Tile&) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoUpper<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLower<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n) {return leaf.mValueMask.countOn();}
};


/// @brief Get/Set operation which returns the total number of unmasked voxels in a leaf node up to but excluding the n^th bit (in the case of ValueOnIndexMask)
///        and the total number of active voxels up to but excluding the n^th bit in a leaf node (in the case of ValueOnIndex)
template <typename BuildType>
struct UnmaskedPerLeaf;
template <>
struct UnmaskedPerLeaf<nanovdb::ValueOnIndexMask>{
    using BuildT = nanovdb::ValueOnIndexMask;
    __hostdev__ static uint32_t get(const nanovdb::NanoRoot<BuildT>&) {return 0;}
    __hostdev__ static uint32_t get(const typename nanovdb::NanoRoot<BuildT>::Tile&) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoUpper<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLower<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n) {return leaf.mMask.countOn(n);}
};
template <>
struct UnmaskedPerLeaf<nanovdb::ValueOnIndex>{
    using BuildT = nanovdb::ValueOnIndex;
    __hostdev__ static uint32_t get(const nanovdb::NanoRoot<BuildT>&) {return 0;}
    __hostdev__ static uint32_t get(const typename nanovdb::NanoRoot<BuildT>::Tile&) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoUpper<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLower<BuildT>&, uint32_t) {return 0;}
    __hostdev__ static uint32_t get(const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t n) {return leaf.mValueMask.countOn(n);}
};

} // namespace fvdb