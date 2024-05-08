#pragma once

#include <nanovdb/NanoVDB.h>


namespace fvdb {
namespace detail {

template <typename ScalarT>
struct TrilinearInterpolationWithGradIterator {
    struct PairT {
        nanovdb::Coord first;
        nanovdb::math::Vec4<ScalarT> second;
    };

    template <typename Scalar, int N>
    struct ArrayT {
        nanovdb::math::Vec4<ScalarT> mData[N];

        __hostdev__
        constexpr nanovdb::math::Vec4<ScalarT> operator [] (int i) const {return mData[i];}
    };

    // Iterator traits, previously from std::iterator.
    using value_type = PairT;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    TrilinearInterpolationWithGradIterator() = delete;

    __hostdev__
    TrilinearInterpolationWithGradIterator(const nanovdb::math::Vec3<ScalarT> p) {

        mCount = 0;
        mVoxel = p.floor();
        const nanovdb::math::Vec3<ScalarT> uvw =
            p - nanovdb::math::Vec3<ScalarT>(mVoxel.x(), mVoxel.y(), mVoxel.z());

        const ScalarT ONE = ScalarT(1);
        mTrilinearWXYZ = {
            nanovdb::math::Vec4<ScalarT>((ONE - uvw[0]) * (ONE - uvw[1]) * (ONE - uvw[2]),
                                   -(ONE - uvw[1]) * (ONE - uvw[2]),
                                   -(ONE - uvw[0]) * (ONE - uvw[2]),
                                   -(ONE - uvw[0]) * (ONE - uvw[1])),
            nanovdb::math::Vec4<ScalarT>((ONE - uvw[0]) * (ONE - uvw[1]) * uvw[2],
                                   -(ONE - uvw[1]) * uvw[2],
                                   -(ONE - uvw[0]) * uvw[2],
                                   (ONE - uvw[0]) * (ONE - uvw[1])),
            nanovdb::math::Vec4<ScalarT>((ONE - uvw[0]) * uvw[1] * (ONE - uvw[2]),
                                   -uvw[1] * (ONE - uvw[2]),
                                   (ONE - uvw[0]) * (ONE - uvw[2]),
                                   -(ONE - uvw[0]) * uvw[1]),
            nanovdb::math::Vec4<ScalarT>((ONE - uvw[0]) * uvw[1] * uvw[2],
                                   -uvw[1] * uvw[2],
                                   (ONE - uvw[0]) * uvw[2],
                                   (ONE - uvw[0]) * uvw[1]),
            nanovdb::math::Vec4<ScalarT>(uvw[0] * (ONE - uvw[1]) * (ONE - uvw[2]),
                                   (ONE - uvw[1]) * (ONE - uvw[2]),
                                   -uvw[0] * (ONE - uvw[2]),
                                   -uvw[0] * (ONE - uvw[1])),
            nanovdb::math::Vec4<ScalarT>(uvw[0] * (ONE - uvw[1]) * uvw[2],
                                   (ONE - uvw[1]) * uvw[2],
                                   -uvw[0] * uvw[2],
                                   uvw[0] * (ONE - uvw[1])),
            nanovdb::math::Vec4<ScalarT>(uvw[0] * uvw[1] * (ONE - uvw[2]),
                                   uvw[1] * (ONE - uvw[2]),
                                   uvw[0] * (ONE - uvw[2]),
                                   -uvw[0] * uvw[1]),
            nanovdb::math::Vec4<ScalarT>(uvw[0] * uvw[1] * uvw[2],
                                   uvw[1] * uvw[2],
                                   uvw[0] * uvw[2],
                                   uvw[0] * uvw[1]),
        };

        mCoordAndWXYZ = {
                mVoxel, mTrilinearWXYZ[0]
        };
    }

    __hostdev__
    inline const TrilinearInterpolationWithGradIterator& operator++() {
        mCount += 1;
        if (mCount >= 8) {
            return *this;
        }
        const uint8_t di = (mCount & (1 << 2)) >> 2;
        const uint8_t dj = (mCount & (1 << 1)) >> 1;
        const uint8_t dk = mCount & 1;
        const nanovdb::Coord ijk = nanovdb::Coord(di, dj, dk) + mVoxel;
        mCoordAndWXYZ = {ijk, mTrilinearWXYZ[mCount] };
        return *this;
    }

    __hostdev__
    TrilinearInterpolationWithGradIterator operator++(int) {
        TrilinearInterpolationWithGradIterator tmp = *this; ++(*this); return tmp;
    }

    // Dereferencable.
    __hostdev__
    inline constexpr const value_type& operator*() const {
        return mCoordAndWXYZ;
    }

    __hostdev__
    inline constexpr const value_type* operator->() const {
        return (const value_type*) &mCoordAndWXYZ;
    }

    // Equality / inequality.
    __hostdev__
    inline constexpr bool operator==(const TrilinearInterpolationWithGradIterator& rhs) const {
        return mVoxel == rhs.mVoxel && mCount == rhs.mCount;
    }

    __hostdev__
    inline constexpr bool operator!=(const TrilinearInterpolationWithGradIterator& rhs) const {
        return !(*this == rhs);
    }

    __hostdev__
    inline constexpr bool isValid() {
        return mCount < 8;
    }

private:
    int32_t mCount = 0;
    value_type mCoordAndWXYZ;
    nanovdb::Coord mVoxel;
    ArrayT<ScalarT, 8> mTrilinearWXYZ;
};

} // namespace detail
} // namespace fvdb
