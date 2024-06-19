#pragma once

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {

template <typename ScalarT>
struct BezierInterpolationWithGradIterator {
    struct PairT {
        nanovdb::Coord first;
        nanovdb::math::Vec4<ScalarT> second;
    };

    using value_type = PairT;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    BezierInterpolationWithGradIterator() = delete;

    __hostdev__ BezierInterpolationWithGradIterator(const nanovdb::math::Vec3<ScalarT>& p) {
        mCount = 0;
        mVoxel = p.round();
        mUVW = p - nanovdb::math::Vec3<ScalarT>(mVoxel.x(), mVoxel.y(), mVoxel.z());
        updateCoordAndWeight();
    }

    __hostdev__ inline const BezierInterpolationWithGradIterator& operator++() {
        mCount += 1;
        if (!isValid()) { return *this; }
        updateCoordAndWeight();
        return *this;
    }

    __hostdev__
    BezierInterpolationWithGradIterator operator++(int) {
        BezierInterpolationWithGradIterator tmp = *this; ++(*this); return tmp;
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
    inline constexpr bool operator==(const BezierInterpolationWithGradIterator& rhs) const {
        return mVoxel == rhs.mVoxel && mCount == rhs.mCount;
    }

    __hostdev__
    inline constexpr bool operator!=(const BezierInterpolationWithGradIterator& rhs) const {
        return !(*this == rhs);
    }

    __hostdev__
    inline constexpr bool isValid() {
        return mCount < 27;
    }

protected:

    __hostdev__ inline ScalarT bezier(const ScalarT x) {
        bool r1 = x < -1.5, r2 = x < -0.5, r3 = x < 0.5, r4 = x < 1.5;
        if (!r1 && r2) {
            return (x + 1.5) * (x + 1.5);
        } else if (!r2 && r3) {
            return -2 * x * x + 1.5;
        } else if (!r3 && r4) {
            return (x - 1.5) * (x - 1.5);
        }
        return 0.0;
    }

    __hostdev__ inline ScalarT bezierGrad(const ScalarT x) {
        bool r1 = x < -1.5, r2 = x < -0.5, r3 = x < 0.5, r4 = x < 1.5;
        if (!r1 && r2) {
            return 2 * x + 3;
        } else if (!r2 && r3) {
            return -4 * x;
        } else if (!r3 && r4) {
            return 2 * x - 3;
        }
        return 0.0;
    }

    __hostdev__ inline void updateCoordAndWeight() {
        const int32_t dz = mCount % 3 - 1;
        const int32_t dy = (mCount / 3) % 3 - 1;
        const int32_t dx = mCount / 9 - 1;
        ScalarT bx = bezier(mUVW[0] - dx), by = bezier(mUVW[1] - dy), bz = bezier(mUVW[2] - dz);
        ScalarT dbx = bezierGrad(mUVW[0] - dx), dby = bezierGrad(mUVW[1] - dy), dbz = bezierGrad(mUVW[2] - dz);
        mCoordAndWXYZ = {nanovdb::Coord(dx, dy, dz) + mVoxel, nanovdb::math::Vec4<ScalarT>(
            bx * by * bz, dbx * by * bz, bx * dby * bz, bx * by * dbz)};
    }

    int32_t mCount = 0;
    value_type mCoordAndWXYZ;
    nanovdb::Coord mVoxel;
    nanovdb::math::Vec3<ScalarT> mUVW;
};

} // namespace detail
} // namespace fvdb