#pragma once

#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>
#include <c10/util/Half.h>
#include <ATen/OpMathType.h>

#include "CustomAccessors.h"

#include <iostream>

namespace nanovdb {

namespace math {

template<>
struct Delta<c10::Half>
{
    __hostdev__ static c10::Half value() { return c10::Half(1e-3f); }
};

} // namespace math

} // namespace nanovdb

namespace fvdb {

template <typename AccT, typename ScalarT>
struct HDDASegmentIterator {
public:
    using BuildT = typename AccT::BuildType;
    using MathType = at::opmath_type<ScalarT>;
    using RayT = nanovdb::math::Ray<ScalarT>;
    using RayTInternal = nanovdb::math::Ray<MathType>;
    using TimespanT = typename RayTInternal::TimeSpan;
    using CoordT = nanovdb::Coord;
    using HDDAT = nanovdb::math::HDDA<RayTInternal, nanovdb::Coord>;

    using value_type = TimespanT;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    HDDASegmentIterator() = delete;

    __hostdev__
    bool isValid() const {
        return mTimespan.valid(0.0);
    }

    __hostdev__
    const HDDASegmentIterator& operator++() {
        nextSegment();
        return *this;
    }

    __hostdev__
    HDDASegmentIterator operator++(int) {
        HDDASegmentIterator tmp = *this; ++(*this); return tmp;
    }

    __hostdev__
    HDDASegmentIterator(const RayT& rayVox, const AccT& acc, const bool ignoreMasked) : mAcc(acc) {
        mIgnoreMasked = ignoreMasked;
        mRay = RayTInternal(
            nanovdb::math::Vec3<MathType>(rayVox.eye()),
            nanovdb::math::Vec3<MathType>(rayVox.dir()),
	    static_cast<MathType>(rayVox.t0()),
	    static_cast<MathType>(rayVox.t1())
        );
        CoordT ijk = nanovdb::math::RoundDown<CoordT>(rayVox(mRay.t0() + nanovdb::math::Delta<ScalarT>::value()));
        mHdda.init(mRay, mAcc.getDim(ijk, mRay));
        nextSegment(); // Move to first segment
    }

    // Dereferencable.
    __hostdev__
    const value_type& operator*() const {
        return mTimespan;
    }

    __hostdev__
    const value_type* operator->() const {
        return (const value_type*) &mTimespan;
    }

private:

    __hostdev__
    bool nextSegment() {
        mTimespan.t0 = mRay.t1() + static_cast<ScalarT>(5.0);
        mTimespan.t1 = mRay.t1();
        do {
            // Coordinate of the current voxel
            const int dim = mAcc.getDim(mHdda.voxel(), mRay);

            // Set the level of HDDA
            if (mHdda.dim() != dim) {
                mRay.setMinTime(mHdda.time());
                mHdda.update(mRay, dim);
            }

            const bool isActive = mIgnoreMasked ? mAcc.isActive(mHdda.voxel()) : mAcc.template get<fvdb::ActiveOrUnmasked<BuildT>>(mHdda.voxel());
            if (isActive) {  // We're inside an active region
                if (!mTimespan.valid()) { // This is the first hit
                    mTimespan.t0 = mHdda.time();
                }
            } else {  // We're not in an active region
                if (mTimespan.valid()) {  // We were just in an active region
                    mTimespan.t1 = mHdda.time();
                    break;
                }
            }
        } while(mHdda.step());

        if (!mTimespan.valid(0.0)) {
            mTimespan.t1 = fminf(mRay.t1(), mHdda.time());

        }
        // We didn't hit anything, return
        return mTimespan.valid(0.0);
    }

    const AccT& mAcc;
    RayTInternal mRay;
    HDDAT mHdda;
    TimespanT mTimespan;
    bool mIgnoreMasked;
};


template <typename AccT, typename ScalarT>
struct HDDAVoxelIterator {
    using MathType = at::opmath_type<ScalarT>;
    struct PairT {
        nanovdb::Coord first;
        typename nanovdb::math::Ray<MathType>::TimeSpan second;
    };
    using BuildT = typename AccT::BuildType;
    using RayT = nanovdb::math::Ray<ScalarT>;
    using RayTInternal = nanovdb::math::Ray<MathType>;
    using TimespanT = typename RayTInternal::TimeSpan;
    using CoordT = nanovdb::Coord;
    using HDDAT = nanovdb::math::HDDA<RayTInternal, nanovdb::Coord>;

    using value_type = PairT;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    HDDAVoxelIterator() = delete;

    __hostdev__
    HDDAVoxelIterator(const RayT& rayVox, const AccT& acc) : mAcc(acc) {
        mRay = RayTInternal(
            nanovdb::math::Vec3<MathType>(rayVox.eye()),
            nanovdb::math::Vec3<MathType>(rayVox.dir()),
            static_cast<MathType>(rayVox.t0()),
            static_cast<MathType>(rayVox.t1())
        );

        CoordT ijk = mRay(mRay.t0() + nanovdb::math::Delta<ScalarT>::value()).floor();
        mHdda.init(mRay, mAcc.getDim(ijk, mRay));
        mIsValid = nextVoxel();
    }

    __hostdev__
    bool isValid() const {
        return mIsValid;
    }

    __hostdev__
    const value_type& operator*() const {
        return mData;
    }

    __hostdev__
    const value_type* operator->() const {
        return (const value_type*) &mData;
    }

    __hostdev__
    const HDDAVoxelIterator& operator++() {
        mIsValid = nextVoxel();
        return *this;
    }

    __hostdev__
    HDDAVoxelIterator operator++(int) {
        HDDAVoxelIterator tmp = *this; ++(*this); return tmp;
    }

private:

    __hostdev__
    bool nextVoxel() {
        do {
            // Coordinate of the current voxel
            const int dim = mAcc.getDim(mHdda.voxel(), mRay);

            // Set the level of HDDA
            if (mHdda.dim() != dim) {
                mRay.setMinTime(mHdda.time());
                mHdda.update(mRay, dim);
            }
            // NOTE: This will return true if a tile is active
            if (mAcc.template get<fvdb::ActiveOrUnmasked<BuildT>>(mHdda.voxel())) {  // We hit an active voxel, increment hdda and return
                mData = { mHdda.voxel(), TimespanT(mHdda.time(), mHdda.next()) };
                mHdda.step();
                return true;
            }
        } while(mHdda.step());

        // We didn't find any active voxels, return
        return false;
    }

    bool mIsValid = false;
    const AccT& mAcc;
    RayTInternal mRay;
    HDDAT mHdda;
    value_type mData;
};

} // namespace fvdb
