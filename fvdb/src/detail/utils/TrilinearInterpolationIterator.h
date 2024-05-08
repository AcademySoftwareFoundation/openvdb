#pragma once

#include <nanovdb/NanoVDB.h>


namespace fvdb {
namespace detail {

/*
 * Convenience iterator to iterate over the 8 neighboring voxels
 * of a point to perform trilinear interpolation.
 *
 * Usage:
 *  for (auto it = TrilinearInterpolationIterator<T>; it.isValid(); it++) {
 *      nanovdb::Coord voxel = it->first;
 *      T trilinear_weight = it->second;
 *  }
 *
 * Each iterated item returns a pair (ijk, w_tril) where
 *   ijk is the coordinate of one of the neighboring voxel
 *   w_tril is the trilinear weight that the voxel ijk contributes
 */
template <typename ScalarT>
struct TrilinearInterpolationIterator {
    struct PairT {
        nanovdb::Coord first;
        ScalarT second;
    };

    template <typename Scalar, int N>
    struct ArrayT {
        Scalar mData[N];

        __hostdev__
        constexpr Scalar operator [] (int i) const {return mData[i];}
    };

    // Iterator traits, previously from std::iterator.
    using value_type = PairT;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    TrilinearInterpolationIterator() = delete;

    __hostdev__
    TrilinearInterpolationIterator(const nanovdb::math::Vec3<ScalarT> p) {

        mCount = 0;
        mVoxel = p.floor();
        const nanovdb::math::Vec3<ScalarT> uvw =
            p - nanovdb::math::Vec3<ScalarT>(mVoxel.x(), mVoxel.y(), mVoxel.z());

        const ScalarT ONE = ScalarT(1);
        mTrilinearWeights = {
            (ONE - uvw[0]) * (ONE - uvw[1]) * (ONE - uvw[2]),
            (ONE - uvw[0]) * (ONE - uvw[1]) * uvw[2],
            (ONE - uvw[0]) * uvw[1] * (ONE - uvw[2]),
            (ONE - uvw[0]) * uvw[1] * uvw[2],
            uvw[0] * (ONE - uvw[1]) * (ONE - uvw[2]),
            uvw[0] * (ONE - uvw[1]) * uvw[2],
            uvw[0] * uvw[1] * (ONE - uvw[2]),
            uvw[0] * uvw[1] * uvw[2],
        };

        mCoordAndWeight = {
            mVoxel, mTrilinearWeights[0]
        };
    }

    __hostdev__
    inline const TrilinearInterpolationIterator& operator++() {
        mCount += 1;
        if (mCount >= 8) {
            return *this;
        }
        const uint8_t di = (mCount & (1 << 2)) >> 2;
        const uint8_t dj = (mCount & (1 << 1)) >> 1;
        const uint8_t dk = mCount & 1;
        const nanovdb::Coord ijk = nanovdb::Coord(di, dj, dk) + mVoxel;
        const ScalarT weight = mTrilinearWeights[mCount];
        mCoordAndWeight = { ijk, weight };
        return *this;
    }

    __hostdev__
    TrilinearInterpolationIterator operator++(int) {
        TrilinearInterpolationIterator tmp = *this; ++(*this); return tmp;
    }

    // Dereferencable.
    __hostdev__
    inline constexpr const value_type& operator*() const {
        return mCoordAndWeight;
    }

    __hostdev__
    inline constexpr const value_type* operator->() const {
        return (const value_type*) &mCoordAndWeight;
    }

    // Equality / inequality.
    __hostdev__
    inline constexpr bool operator==(const TrilinearInterpolationIterator& rhs) const {
        return mVoxel == rhs.mVoxel && mCount == rhs.mCount;
    }

    __hostdev__
    inline constexpr bool operator!=(const TrilinearInterpolationIterator& rhs) const {
        return !(*this == rhs);
    }

    __hostdev__
    inline constexpr bool isValid() {
        return mCount < 8;
    }

private:
    int32_t mCount = 0;
    value_type mCoordAndWeight;
    nanovdb::Coord mVoxel;
    ArrayT<ScalarT, 8> mTrilinearWeights;
    // std::array<ScalarT, 8> mTrilinearWeights;
};

} // namespace detail
} // namespace fvdb