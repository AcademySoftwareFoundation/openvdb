// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
///
/// @author Greg Hurst
///
/// @file LevelSetDilatedMeshImpl.h
///
/// @brief Generate a narrow-band level set of a dilated surface mesh.
///
/// @note By definition a level set has a fixed narrow band width
/// (the half width is defined by LEVEL_SET_HALF_WIDTH in Types.h),
/// whereas an SDF can have a variable narrow band width.

#ifndef OPENVDB_TOOLS_LEVELSETDILATEDMESHIMPL_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETDILATEDMESHIMPL_HAS_BEEN_INCLUDED

#include "ConvexVoxelizer.h"

#include <openvdb/tools/LevelSetTubes.h>
#include <openvdb/tools/PointPartitioner.h>
#include <openvdb/tools/Prune.h>

#include <openvdb/Grid.h>
#include <openvdb/Types.h> // for ComputeTypeFor
#include <openvdb/math/Math.h>
#include <openvdb/util/NullInterrupter.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vector>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

namespace lvlset {

/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of an _open_ prism.
/// The only parts of the level set populated are along both normals of the triangle.
/// Negative background tiles that fit inside the closed dilated triangle are also populated.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
class OpenTriangularPrismVoxelizer
    : public ConvexVoxelizer<
          GridType,
          OpenTriangularPrismVoxelizer<GridType, InterruptT>,
          InterruptT>
{
    using GridPtr = typename GridType::Ptr;

    using BaseT = ConvexVoxelizer<
        GridType,
        OpenTriangularPrismVoxelizer<GridType, InterruptT>,
        InterruptT
    >;

    using BaseT::mXYData;
    using BaseT::tileCeil;

    using ValueT   = typename BaseT::ValueT;
    using ComputeT = typename ComputeTypeFor<ValueT>::type;
    using Vec3T    = typename BaseT::Vec3T;

public:

    friend class ConvexVoxelizer<
        GridType,
        OpenTriangularPrismVoxelizer<GridType, InterruptT>,
        InterruptT
    >;

    /// @brief Constructor
    ///
    /// @param grid scalar grid to populate the level set in
    /// @param threaded center of the sphere in world units
    /// @param interrupter pointer to optional interrupter. Use template
    /// argument util::NullInterrupter if no interruption is desired.
    ///
    /// @note The voxel size and half width are determined from the input grid,
    /// meaning the voxel size and background value need to be set prior to voxelization
    OpenTriangularPrismVoxelizer(GridPtr& grid,
        const bool& threaded = false,
        InterruptT* interrupter = nullptr)
    : BaseT(grid, threaded, interrupter)
    {
    }

    /// @brief Create an open prism
    ///
    /// @param pt1    point 1 of the triangle in world units
    /// @param pt2    point 2 of the triangle in world units
    /// @param pt3    point 3 of the triangle in world units
    /// @param radius    radius of the open prism in world units
    template<typename ScalarType>
    void
    operator()(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
               const math::Vec3<ScalarType>& pt3, const ScalarType& radius)
    {
        static_assert(openvdb::is_floating_point<ScalarType>::value);

        if (initialize(pt1, pt2, pt3, radius))
            BaseT::iterate();
    }

private:

    inline void
    setXYRangeData(const Index& step = 1)
    {
        const ComputeT &x1 = mPts[0].x(), &x2 = mPts[1].x(), &x3 = mPts[2].x(),
                       &x4 = mPts[3].x(), &x5 = mPts[4].x(), &x6 = mPts[5].x();

        const ComputeT xmin = math::Min(x1, x2, x3, x4, x5, x6);
        const ComputeT xmax = math::Max(x1, x2, x3, x4, x5, x6);
        mXYData.reset(xmin, xmax, step);

        // TODO add logic to ignore edges in the interior of the projection
        // TODO add logic that classifies each segment as being either on 'top' or 'bottom'

        setXYSegmentRangeData<0,1,0>(step);
        setXYSegmentRangeData<1,2,0>(step);
        setXYSegmentRangeData<2,0,0>(step);

        setXYSegmentRangeData<3,4,0>(step);
        setXYSegmentRangeData<4,5,0>(step);
        setXYSegmentRangeData<5,3,0>(step);

        setXYSegmentRangeData<0,3,0>(step);
        setXYSegmentRangeData<1,4,0>(step);
        setXYSegmentRangeData<2,5,0>(step);
    }

    template<Index i, Index j, int MinMax = 0>
    inline void
    setXYSegmentRangeData(const Index& step = 1)
    {
        const ComputeT &x1 = mPts[i].x(), &x2 = mPts[j].x();

        // nothing to do if segment does not span across more than on voxel in x
        // other segments will handle this segment's range
        if (tileCeil(x1, step) == tileCeil(x2, step))
            return;

        const ComputeT x_start = tileCeil(math::Min(x1, x2), step),
                       x_end = math::Max(x1, x2),
                       stepv = ValueT(step);

        for (ComputeT x = x_start; x <= x_end; x += stepv) {
            if constexpr (MinMax <= 0)
                mXYData.expandYMin(x, line2D<i,j>(x));
            if constexpr (MinMax >= 0)
                mXYData.expandYMax(x, line2D<i,j>(x));
        }
    }

    // simply offset distance to the center plane, we may assume any CPQ falls in inside the prism
    inline ComputeT
    signedDistance(const Vec3T& p) const
    {
        return math::Abs(mTriNrml.dot(p - mA)) - mRad;
    }

    // allows for tiles to poke outside of the open prism into the tubes
    // adaptation of udTriangle at https://iquilezles.org/articles/distfunctions/
    inline ComputeT
    tilePointSignedDistance(const Vec3T& p) const
    {
        const Vec3T pa = p - mA,
                    pb = p - mB,
                    pc = p - mC;

        const ComputeT udist =
            math::Sign(mBAXNrml.dot(pa)) +
            math::Sign(mCBXNrml.dot(pb)) +
            math::Sign(mACXNrml.dot(pc)) < 2
            ?
            math::Sqrt(math::Min(
                (mBA * math::Clamp01(mBANorm2.dot(pa)) - pa).lengthSqr(),
                (mCB * math::Clamp01(mCBNorm2.dot(pb)) - pb).lengthSqr(),
                (mAC * math::Clamp01(mACNorm2.dot(pc)) - pc).lengthSqr()
            ))
            :
            math::Abs(mTriNrml.dot(p - mA));

        return udist - mRad;
    }

    inline bool
    tileCanFit(const Index& dim) const
    {
        return mRad >= BaseT::halfWidth() + ComputeT(0.5) * (ComputeT(dim)-ComputeT(1));
    }

    std::function<bool(ComputeT&, ComputeT&, const ComputeT&, const ComputeT&)> prismBottomTop =
    [this](ComputeT& zb, ComputeT& zt, const ComputeT& x, const ComputeT& y)
    {
        zb = std::numeric_limits<ComputeT>::lowest();
        zt = std::numeric_limits<ComputeT>::max();

        // TODO with proper book keeping we can know apriori which 2 indexes will set zb & zt
        //      basically figure out a poor man's cylindrical decomposition...
        setPlaneBottomTop<0>(zb, zt, x, y);
        setPlaneBottomTop<1>(zb, zt, x, y);
        setPlaneBottomTop<2>(zb, zt, x, y);
        setPlaneBottomTop<3>(zb, zt, x, y);
        setPlaneBottomTop<4>(zb, zt, x, y);

        return true;
    };

    template<Index i>
    inline void
    setPlaneBottomTop(ComputeT& zb, ComputeT& zt, const ComputeT& x, const ComputeT& y) const
    {
        if (math::isApproxZero(mFaceNrmls[i].z()))
            return;

        const ComputeT z = mPlaneXCoeffs[i]*x + mPlaneYCoeffs[i]*y + mPlaneOffsets[i];

        if (mFaceNrmls[i].z() < 0) {
            if (zb < z)
                zb = z;
        } else {
            if (zt > z)
                zt = z;
        }
    }

    // world space points and radius inputs
    // initializes class members in index space
    template<typename ScalarType>
    inline bool
    initialize(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
               const math::Vec3<ScalarType>& pt3, const ScalarType& r)
    {
        const ComputeT vx = BaseT::voxelSize(),
                       hw = BaseT::halfWidth();

        mA = Vec3T(pt1)/vx;
        mB = Vec3T(pt2)/vx;
        mC = Vec3T(pt3)/vx;

        mRad = ValueT(r)/vx;

        mBA = mB-mA;
        mCB = mC-mB;
        mAC = mA-mC;

        mTriNrml = mBA.cross(mC-mA);

        mBAXNrml = mTriNrml.cross(mBA);
        mCBXNrml = mTriNrml.cross(mCB);
        mACXNrml = mTriNrml.cross(mAC);

        mBANorm2 = math::isApproxZero(mBA.lengthSqr()) ? mBA : mBA/mBA.lengthSqr();
        mCBNorm2 = math::isApproxZero(mCB.lengthSqr()) ? mCB : mCB/mCB.lengthSqr();
        mACNorm2 = math::isApproxZero(mAC.lengthSqr()) ? mAC : mAC/mAC.lengthSqr();

        const ComputeT len = mTriNrml.length();
        if (math::isApproxZero(len)) {
            return false; // nothing to voxelize, prism has no volume
        } else {
            mTriNrml /= len;
        }

        const ComputeT hwRad = mRad + hw;
        if (math::isApproxZero(hwRad) || hwRad < 0)
            return false; // nothing to voxelize, prism has no volume

        mPts = {
            mA + hwRad * mTriNrml, mB + hwRad * mTriNrml, mC + hwRad * mTriNrml,
            mA - hwRad * mTriNrml, mB - hwRad * mTriNrml, mC - hwRad * mTriNrml
        };

        // tri1, tri2, quad1, quad2, quad3
        mFaceNrmls = {
            mTriNrml,
            -mTriNrml,
            mTriNrml.cross(mA-mB).unitSafe(),
            mTriNrml.cross(mB-mC).unitSafe(),
            mTriNrml.cross(mC-mA).unitSafe()
        };

        {
            static const std::vector<Index> p_ind = {0, 3, 0, 1, 2};

            mPlaneXCoeffs.assign(5, ValueT(0));
            mPlaneYCoeffs.assign(5, ValueT(0));
            mPlaneOffsets.assign(5, ValueT(0));

            for (Index i = 0; i < 5; ++i) {
                if (!math::isApproxZero(mFaceNrmls[i].z())) {
                    const ComputeT cx = mFaceNrmls[i].x()/mFaceNrmls[i].z(),
                                   cy = mFaceNrmls[i].y()/mFaceNrmls[i].z();
                    const Vec3T p = mPts[p_ind[i]];
                    mPlaneXCoeffs[i] = -cx;
                    mPlaneYCoeffs[i] = -cy;
                    mPlaneOffsets[i] = p.x()*cx + p.y()*cy + p.z();
                }
            }
        }

        BaseT::bottomTop = prismBottomTop;

        return true;
    }

    // ------------ general utilities ------------

    template <Index i, Index j>
    ComputeT
    line2D(const ValueT& x) const
    {
        const ComputeT &x1 = mPts[i].x(), &y1 = mPts[i].y(),
                       &x2 = mPts[j].x(), &y2 = mPts[j].y();

        const ComputeT m = (y2-y1)/(x2-x1);

        return y1 + m * (x-x1);
    }

    // ------------ private members ------------

    Vec3T mA, mB, mC;
    ComputeT mRad;

    Vec3T mBA, mCB, mAC;
    Vec3T mBAXNrml, mCBXNrml, mACXNrml;
    Vec3T mBANorm2, mCBNorm2, mACNorm2;

    std::vector<Vec3T> mPts = std::vector<Vec3T>(6);

    Vec3T mTriNrml;
    std::vector<Vec3T> mFaceNrmls = std::vector<Vec3T>(5);

    std::vector<ComputeT> mPlaneXCoeffs = std::vector<ValueT>(5),
                          mPlaneYCoeffs = std::vector<ValueT>(5),
                          mPlaneOffsets = std::vector<ValueT>(5);

}; // class OpenTriangularPrismVoxelizer

/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of an _open_ wedge.
/// The only parts of the level set populated are within a sector of a capsule.
/// The sector is defined by the intersection of two half spaces.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
class OpenCapsuleWedgeVoxelizer
    : public ConvexVoxelizer<
          GridType,
          OpenCapsuleWedgeVoxelizer<GridType, InterruptT>,
          InterruptT>
{
    using GridPtr = typename GridType::Ptr;

    using BaseT = ConvexVoxelizer<
        GridType,
        OpenCapsuleWedgeVoxelizer<GridType, InterruptT>,
        InterruptT
    >;

    using BaseT::mXYData;
    using BaseT::tileCeil;

    using ValueT   = typename BaseT::ValueT;
    using ComputeT = typename ComputeTypeFor<ValueT>::type;
    using Vec3T    = typename BaseT::Vec3T;
    using Vec2T    = typename BaseT::Vec2T;

public:

    friend class ConvexVoxelizer<
        GridType,
        OpenCapsuleWedgeVoxelizer<GridType, InterruptT>,
        InterruptT
    >;

    /// @brief Constructor
    ///
    /// @param grid scalar grid to populate the level set in
    /// @param threaded center of the sphere in world units
    /// @param interrupter pointer to optional interrupter. Use template
    /// argument util::NullInterrupter if no interruption is desired.
    ///
    /// @note The voxel size and half width are determined from the input grid,
    /// meaning the voxel size and background value need to be set prior to voxelization
    OpenCapsuleWedgeVoxelizer(GridPtr& grid, const bool& threaded = false,
        InterruptT* interrupter = nullptr)
    : BaseT(grid, threaded, interrupter)
    {
    }

    /// @brief Create an open wedge
    ///
    /// @param pt1    first endpoint open wedge in world units
    /// @param pt2    second endpoint open wedge in world units
    /// @param radius    radius of the open prism in world units
    /// @param nrml1    normal of a half space the the capsule is clipped with to form the open wedge
    /// @param nrml2    normal of the other half space the the capsule is clipped with to form the open wedge
    ///
    /// @note The normal vectors @f$n @f$ point outward from the open wedge,
    /// and the clipping half space is defined by the set of points @f$p @f$ that satisfy @f$n . (p - pt1) \leq 0@f$.
    template<typename ScalarType>
    void
    operator()(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
        const ScalarType& radius, const math::Vec3<ScalarType>& nrml1,
        const math::Vec3<ScalarType>& nrml2)
    {
        static_assert(openvdb::is_floating_point<ScalarType>::value);

        if (initialize(pt1, pt2, radius, nrml1, nrml2))
            BaseT::iterate();
    }

private:

    // computes *approximate* xy-range data: the projected caps might contain over-inclusive values
    inline void
    setXYRangeData(const Index& step = 1)
    {
        const ComputeT stepv = ValueT(step);

        // degenerate
        if (mX1 - mORad > mX2 + mORad) {
            mXYData.clear();
            return;
        }

        // short circuit a vertical cylinder
        if (mIsVertical) {
            mXYData.reset(mX1 - mORad, mX1 + mORad, step);

            for (ComputeT x = tileCeil(mX1 - mORad, step); x <= mX1 + mORad; x += stepv)
                mXYData.expandYRange(x, circle1Bottom(x), circle1Top(x));

            intersectWithXYWedgeLines();
            return;
        }

        const ComputeT v = math::Min(mORad, mORad * math::Abs(mYdiff)/mXYNorm);

        const ComputeT a0 = mX1 - mORad,
                       a1 = mX1 - v,
                       a2 = mX1 + v,
                       a3 = mX2 - v,
                       a4 = mX2 + v,
                       a5 = mX2 + mORad;

        const ComputeT tc0 = tileCeil(a0, step),
                       tc1 = tileCeil(a1, step),
                       tc2 = tileCeil(a2, step),
                       tc3 = tileCeil(a3, step),
                       tc4 = tileCeil(a4, step);

        mXYData.reset(a0, a5, step);

        for (ComputeT x = tc0; x <= a1; x += stepv)
            mXYData.expandYRange(x, circle1Bottom(x), circle1Top(x));

        if (!math::isApproxZero(mXdiff)) {
            if (mY1 > mY2) {
                for (ComputeT x = tc1; x <= math::Min(a2, a3); x += stepv)
                    mXYData.expandYRange(x, lineBottom(x), circle1Top(x));
            } else {
                for (ComputeT x = tc1; x <= math::Min(a2, a3); x += stepv)
                    mXYData.expandYRange(x, circle1Bottom(x), lineTop(x));
            }
        }

        if (a2 < a3) {
            for (ComputeT x = tc2; x <= a3; x += stepv)
                mXYData.expandYRange(x, lineBottom(x), lineTop(x));
        } else {
            if (mY2 <= mY1) {
                for (ComputeT x = tc3; x <= a2; x += stepv)
                    mXYData.expandYRange(x, circle2Bottom(x), circle1Top(x));
            } else {
                for (ComputeT x = tc3; x <= a2; x += stepv)
                    mXYData.expandYRange(x, circle1Bottom(x), circle2Top(x));
            }
        }

        if (!math::isApproxZero(mXdiff)) {
            if (mY1 > mY2) {
                for (ComputeT x = math::Max(tc2, tc3); x <= a4; x += stepv)
                    mXYData.expandYRange(x, circle2Bottom(x), lineTop(x));
            } else {
                for (ComputeT x = math::Max(tc2, tc3); x <= a4; x += stepv)
                    mXYData.expandYRange(x, lineBottom(x), circle2Top(x));
            }
        }

        for (ComputeT x = tc4; x <= a5; x += stepv)
            mXYData.expandYRange(x, circle2Bottom(x), circle2Top(x));

        intersectWithXYStrip();
        intersectWithXYWedgeLines();
    }

    inline void
    intersectWithXYStrip()
    {
        // these strips are vertical when the capsule is
        if (mIsVertical)
            return;

        const Vec3T &pp1 = mPlanePts[0], &pp2 = mPlanePts[1];
        const ComputeT &vx = mV.x(), &vy = mV.y();

        Vec2T n = Vec2T(-vy, vx).unitSafe();
        Vec3T cvec = mORad * Vec3T(-vy, vx, ValueT(0)).unitSafe();

        if (math::isApproxZero(vy))
            cvec.y() = math::Abs(cvec.y());
        else if (vy > 0)
            cvec *= ValueT(-1);

        const Vec3T cpmin(mPt1 - cvec), cpmax(mPt1 + cvec);

        if (math::isApproxZero(mXdiff)) {
            const ComputeT px = mPt1.x(),
                           xmin = math::Min(px, pp1.x(), pp2.x()),
                           xmax = math::Max(px, pp1.x(), pp2.x());

            if (!inWedge(cpmin))
                intersectWithXYHalfSpace(n.x() < 0 ? n : -n, Vec2T(xmin, ValueT(0)));

            if (!inWedge(cpmax))
                intersectWithXYHalfSpace(n.x() > 0 ? n : -n, Vec2T(xmax, ValueT(0)));
        } else {
            const ComputeT m = mYdiff/mXdiff;
            const ComputeT y1 = mPt1.y() - m * mPt1.x(),
                           y2 = pp1.y() - m * pp1.x(),
                           y3 = pp2.y() - m * pp2.x();
            const ComputeT ymin = math::Min(y1, y2, y3),
                           ymax = math::Max(y1, y2, y3);

            if (!inWedge(vy <= 0 ? cpmin : cpmax))
                intersectWithXYHalfSpace(n.y() < 0 ? n : -n, Vec2T(ValueT(0), ymin));

            if (!inWedge(vy > 0 ? cpmin : cpmax))
                intersectWithXYHalfSpace(n.y() > 0 ? n : -n, Vec2T(ValueT(0), ymax));
        }
    }

    inline void
    intersectWithXYWedgeLines()
    {
        const Vec3T v(mORad * mV.unitSafe()),
                    p1(mPt1 - v),
                    p2(mPt2 + v);

        const Vec2T p1_2d(p1.x(), p1.y()), p2_2d(p2.x(), p2.y());

        Vec2T d(-mPlaneNrmls[0].x() - mPlaneNrmls[1].x(),
                -mPlaneNrmls[0].y() - mPlaneNrmls[1].y());

        Vec2T n0(-mDirVectors[0].y(), mDirVectors[0].x()),
              n1(-mDirVectors[1].y(), mDirVectors[1].x());

        if (n0.dot(d) > 0)
            n0 *= ValueT(-1);
        if (n1.dot(d) > 0)
            n1 *= ValueT(-1);

        if (!math::isApproxZero(n0.lengthSqr()))
            intersectWithXYHalfSpace(n0, n0.dot(p2_2d - p1_2d) < 0 ? p1_2d : p2_2d);

        if (!math::isApproxZero(n1.lengthSqr()))
            intersectWithXYHalfSpace(n1, n1.dot(p2_2d - p1_2d) < 0 ? p1_2d : p2_2d);
    }

    inline void
    intersectWithXYHalfSpace(const Vec2T& n, const Vec2T& p)
    {
        if (mXYData.size() == 0)
            return;

        if (math::isApproxZero(n.y())) {
            const ComputeT &px = p.x();
            if (n.x() < 0) {
                const Index m = mXYData.size();
                for (Index i = 0; i < m; ++i) {
                    const ComputeT x = mXYData.getX(i);

                    if (x < px) mXYData.clearYRange(x);
                    else break;
                }
            } else {
                Index i = mXYData.size()-1;
                while (true) {
                    const ComputeT x = mXYData.getX(i);

                    if (x > px) mXYData.clearYRange(x);
                    else break;

                    if (i != 0) --i;
                    else break;
                }
            }
        } else {
            const bool set_min = n.y() < 0;
            const Index m = mXYData.size();

            const ComputeT b = -n.x()/n.y();
            const ComputeT a = p.y() - b * p.x();

            ComputeT x, ymin, ymax;
            for (Index i = 0; i < m; ++i) {
                mXYData.XYData(x, ymin, ymax, i);
                const ComputeT yint = a + b * x;

                if (ymin <= yint && yint <= ymax) {
                    if (set_min) mXYData.setYMin(x, yint);
                    else mXYData.setYMax(x, yint);
                } else {
                    if (set_min ? yint > ymax : yint < ymin)
                        mXYData.clearYRange(x);
                }
            }
        }

        mXYData.trim();
    }

    // distance in index space
    inline ValueT
    signedDistance(const Vec3T& p) const
    {
        const Vec3T w = p - mPt1;
        const ComputeT dot = w.dot(mV);

        // carefully short circuit with a fuzzy tolerance, which avoids division by small mVLenSqr
        if (dot <= math::Tolerance<ComputeT>::value())
            return w.length() - mRad;

        if (dot >= mVLenSqr)
            return (p - mPt2).length() - mRad;

        const ComputeT t = w.dot(mV)/mVLenSqr;

        return (w - t * mV).length() - mRad;
    }

    inline bool
    tileCanFit(const Index& dim) const
    {
        return mRad >= BaseT::halfWidth() + ComputeT(0.5) * (ComputeT(dim)-ComputeT(1));
    }

    std::function<bool(ComputeT&, ComputeT&, const ComputeT&, const ComputeT&)> capsuleBottomTopVertical =
    [this](ComputeT& zb, ComputeT& zt, const ComputeT& x, const ComputeT& y)
    {
        zb = BaseT::sphereBottom(mX1, mY1, math::Min(mZ1, mZ2), mORad, x, y);
        zt = BaseT::sphereTop(mX2, mY2, math::Max(mZ1, mZ2), mORad, x, y);

        return math::isFinite(zb) && math::isFinite(zt);
    };

    std::function<bool(ComputeT&, ComputeT&, const ComputeT&, const ComputeT&)> capsuleBottomTop =
    [this](ComputeT& zb, ComputeT& zt, const ComputeT& x, const ComputeT& y)
    {
        ComputeT cylptb, cylptt;
        if (!infiniteCylinderBottomTop(cylptb, cylptt, x, y))
            return false;

        const ComputeT dotb = (Vec3T(x, y, cylptb) - mPt1).dot(mV);
        const ComputeT dott = (Vec3T(x, y, cylptt) - mPt1).dot(mV);

        if (dotb < 0)
            zb = sphere1Bottom(x, y);
        else if (dotb > mVLenSqr)
            zb = sphere2Bottom(x, y);
        else
            zb = cylptb;

        if (dott < 0)
            zt = sphere1Top(x, y);
        else if (dott > mVLenSqr)
            zt = sphere2Top(x, y);
        else
            zt = cylptt;

        if (!math::isFinite(zb) || !math::isFinite(zt))
            return false;

        intersectWedge<0,1>(zb, zt, x, y);
        intersectWedge<1,0>(zb, zt, x, y);

        return inWedge(x, y, ValueT(0.5)*(zb+zt));
    };

    template<Index i, Index j>
    inline void
    intersectWedge(ComputeT& zb, ComputeT& zt, const ComputeT& x, const ComputeT& y)
    {
        const Vec3T& n0 = mPlaneNrmls[i];

        if (math::isApproxZero(n0.z()))
            return;

        const ComputeT zp = mPlaneXCoeffs[i]*x + mPlaneYCoeffs[i]*y + mPlaneOffsets[i];

        if (zb <= zp && zp <= zt && inHalfSpace<j>(Vec3T(x, y, zp))) {
            if (n0.z() < 0)
                zb = zp;
            else
                zt = zp;
        }
    }

    inline bool
    inWedge(const ComputeT& x, const ComputeT& y, const ComputeT& z)
    {
        return inWedge(Vec3T(x, y, z));
    }

    inline bool
    inWedge(const Vec3T& pt)
    {
        return inHalfSpace<0>(pt) && inHalfSpace<1>(pt);
    }

    template<Index i>
    inline bool
    inHalfSpace(const Vec3T& pt)
    {
        // allow points within a fuzzy fractional (index space) distance to the halfspace
        // this ensures the seams between open wedges and open prisms are completely filled in
        // assumes mPlaneNrmls[i] is a unit vector
        static const ComputeT VOXFRAC = 0.125;

        return mPlaneNrmls[i].dot(pt-mPt1) <= VOXFRAC;
    }

    // assumes tube is not vertical!
    inline bool
    infiniteCylinderBottomTop(ComputeT& cylptb, ComputeT& cylptt,
        const ComputeT& x, const ComputeT& y) const
    {
        const Vec2T q(x, y);

        const Vec2T qproj = mPt12d + mV2d*((q - mPt12d).dot(mV2d))/mXYNorm2;

        const ComputeT t = mX1 != mX2 ? (qproj[0] - mX1)/mXdiff : (qproj[1] - mY1)/mYdiff;

        const Vec3T qproj3D = mPt1 + t * mV;

        const ComputeT d2 = (q - qproj).lengthSqr();

        // outside of cylinder's 2D projection
        if (mORad2 < d2)
            return false;

        const ComputeT h = math::Sqrt((mORad2 - d2) * mVLenSqr/mXYNorm2);

        cylptb = qproj3D[2] - h;
        cylptt = qproj3D[2] + h;

        return true;
    }

    inline ComputeT
    lineBottom(const ComputeT& x) const
    {
        return mY1 + (mYdiff*(x-mX1) - mORad * mXYNorm)/mXdiff;
    }

    inline ComputeT
    lineTop(const ComputeT& x) const
    {
        return mY1 + (mYdiff*(x-mX1) + mORad * mXYNorm)/mXdiff;
    }

    inline ComputeT
    circle1Bottom(const ComputeT& x) const
    {
        return BaseT::circleBottom(mX1, mY1, mORad, x);
    }

    inline ComputeT
    circle1Top(const ComputeT& x) const
    {
        return BaseT::circleTop(mX1, mY1, mORad, x);
    }

    inline ComputeT
    circle2Bottom(const ComputeT& x) const
    {
        return BaseT::circleBottom(mX2, mY2, mORad, x);
    }

    inline ComputeT
    circle2Top(const ComputeT& x) const
    {
        return BaseT::circleTop(mX2, mY2, mORad, x);
    }

    inline ComputeT
    sphere1Bottom(const ComputeT& x, const ComputeT& y) const
    {
        return BaseT::sphereBottom(mX1, mY1, mZ1, mORad, x, y);
    }

    inline ComputeT
    sphere1Top(const ComputeT& x, const ComputeT& y) const
    {
        return BaseT::sphereTop(mX1, mY1, mZ1, mORad, x, y);
    }

    inline ComputeT
    sphere2Bottom(const ComputeT& x, const ComputeT& y) const
    {
        return BaseT::sphereBottom(mX2, mY2, mZ2, mORad, x, y);
    }

    inline ComputeT
    sphere2Top(const ComputeT& x, const ComputeT& y) const
    {
        return BaseT::sphereTop(mX2, mY2, mZ2, mORad, x, y);
    }

    // world space points and radius inputs
    // initializes class members in index space
    template<typename ScalarType>
    inline bool
    initialize(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
        const ScalarType& r, const math::Vec3<ScalarType>& nrml1,
        const math::Vec3<ScalarType>& nrml2)
    {
        const ComputeT vx = BaseT::voxelSize(),
                       hw = BaseT::halfWidth();

        // forces x1 <= x2
        if (pt1[0] <= pt2[0]) {
            mPt1 = Vec3T(pt1)/vx;
            mPt2 = Vec3T(pt2)/vx;
        } else {
            mPt1 = Vec3T(pt2)/vx;
            mPt2 = Vec3T(pt1)/vx;
        }

        mRad = ComputeT(r)/vx;

        // padded radius used to populate the outer halfwidth of the sdf
        mORad  = mRad + hw;
        mORad2 = mORad * mORad;

        // tube has no volume
        if (math::isApproxZero(mORad) || mORad < 0)
            return false;

        mV = mPt2 - mPt1;
        mVLenSqr = mV.lengthSqr();

        // no direction to form the wedge on a sphere
        if (math::isApproxZero(mVLenSqr))
            return false;

        mX1 = mPt1[0]; mY1 = mPt1[1]; mZ1 = mPt1[2];
        mX2 = mPt2[0]; mY2 = mPt2[1]; mZ2 = mPt2[2];

        mXdiff = mX2 - mX1;
        mYdiff = mY2 - mY1;
        mZdiff = mZ2 - mZ1;

        mPt12d = Vec2T(mX1, mY1);
        mPt22d = Vec2T(mX2, mY2);
        mV2d = mPt22d - mPt12d;

        mXYNorm2 = math::Pow2(mXdiff) + math::Pow2(mYdiff);
        mXYNorm = math::Sqrt(mXYNorm2);
        mIsVertical = math::isApproxZero(mXYNorm);

        {
            const Vec3T n1 = Vec3T(nrml1), n2 = Vec3T(nrml2);

            // no direction to form the wedge
            if (math::isApproxZero(n1.lengthSqr()) || math::isApproxZero(n2.lengthSqr()))
                return false;

            mPlaneNrmls[0] = (n1 - n1.projection(mV)).unitSafe();
            mPlaneNrmls[1] = (n2 - n2.projection(mV)).unitSafe();

            // degenerate wedge
            if (approxAntiParallel(mPlaneNrmls[0], mPlaneNrmls[1]))
                return false;

            mDirVectors[0] = mORad * mV.cross(mPlaneNrmls[0]).unitSafe();
            mDirVectors[1] = mORad * mV.cross(mPlaneNrmls[1]).unitSafe();

            if (approxParallel(mPlaneNrmls[0], mPlaneNrmls[1])) {
                mDirVectors[1] = -mDirVectors[0];
            } else {
                if (mPlaneNrmls[1].dot(mDirVectors[0]) > 0)
                    mDirVectors[0] *= ComputeT(-1);
                if (mPlaneNrmls[0].dot(mDirVectors[1]) > 0)
                    mDirVectors[1] *= ComputeT(-1);
            }

            mPlanePts[0] = mPt1 + mDirVectors[0] + ComputeT(0.025) * mPlaneNrmls[0];
            mPlanePts[1] = mPt1 + mDirVectors[1] + ComputeT(0.025) * mPlaneNrmls[1];
        }

        {
            mPlaneXCoeffs.assign(2, ValueT(0));
            mPlaneYCoeffs.assign(2, ValueT(0));
            mPlaneOffsets.assign(2, ValueT(0));

            for (Index i = 0; i < 2; ++i) {
                if (!math::isApproxZero(mPlaneNrmls[i].z())) {
                    const ComputeT cx = mPlaneNrmls[i].x()/mPlaneNrmls[i].z(),
                                   cy = mPlaneNrmls[i].y()/mPlaneNrmls[i].z();
                    const Vec3T p = mPlanePts[i];
                    mPlaneXCoeffs[i] = -cx;
                    mPlaneYCoeffs[i] = -cy;
                    mPlaneOffsets[i] = p.x()*cx + p.y()*cy + p.z();
                }
            }
        }

        BaseT::bottomTop = mIsVertical ? capsuleBottomTopVertical : capsuleBottomTop;

        return true;
    }

    inline bool
    approxAntiParallel(const Vec3T& n1, const Vec3T& n2)
    {
        return approxParallel(n1, -n2);
    }

    inline bool
    approxParallel(const Vec3T& n1, const Vec3T& n2)
    {
        return n1.unitSafe().eq(n2.unitSafe());
    }

    // ------------ private members ------------

    // wedge data -- populated via initialize()

    Vec3T mPt1, mPt2, mV;

    Vec2T mPt12d, mPt22d, mV2d;

    ComputeT mORad, mORad2, mRad, mVLenSqr, mXdiff, mYdiff, mZdiff, mXYNorm, mXYNorm2;

    ComputeT mX1, mY1, mZ1, mX2, mY2, mZ2;

    bool mIsVertical;

    std::vector<Vec3T> mPlaneNrmls = std::vector<Vec3T>(2),
                       mDirVectors = std::vector<Vec3T>(2),
                       mPlanePts   = std::vector<Vec3T>(2);

    std::vector<ComputeT> mPlaneXCoeffs = std::vector<ValueT>(2),
                          mPlaneYCoeffs = std::vector<ValueT>(2),
                          mPlaneOffsets = std::vector<ValueT>(2);

}; // class OpenCapsuleWedgeVoxelizer


/// @brief Class representing the connectivity of edges in a triangle mesh,
/// where each edge is associated with the cells (triangles) sharing it.
/// Provides methods to retrieve adjacent cells,
/// vertex coordinates, normals, and other geometric properties.
template<typename ValueT>
class TriangleMeshEdgeConnectivity {

    static_assert(std::is_floating_point<ValueT>::value);

    using Vec3T = math::Vec3<ValueT>;

public:

    /// @brief Constructs the TriangleMeshEdgeConnectivity object with given coordinates and cell data.
    /// Populates edge-to-cell adjacency and computes cell normals.
    ///
    /// @param coords    Vector of vertex coordinates.
    /// @param cells    Vector of cell (triangle) indices.
    TriangleMeshEdgeConnectivity(const std::vector<Vec3T>& coords,
                                 const std::vector<Vec3I>& cells)
    : mCoords(coords), mCells(cells)
    {
        const Index n = Index(coords.size());

        mNormals.resize(cells.size());

        for (Index i = 0; i < cells.size(); ++i) {
            const Vec3I& cell = mCells[i];

            Edge edges[3] = {
                Edge(cell[0], cell[1]),
                Edge(cell[1], cell[2]),
                Edge(cell[2], cell[0])
            };

            for (const Edge& edge : edges) {
                mEdgeMap[edge].push_back(i);
            }

            if (cell[0] >= n || cell[1] >= n || cell[2] >= n)
                OPENVDB_THROW(ValueError, "out of bounds index");

            const Vec3T &p1 = mCoords[cell[0]],
                        &p2 = mCoords[cell[1]],
                        &p3 = mCoords[cell[2]];

            mNormals[i] = (p2 - p1).cross(p3 - p1).unitSafe();
        }

        for (auto& [edge, cells] : mEdgeMap)
            sortAdjacentCells(edge, cells);
    }

    /// @brief Retrieves the IDs of cells adjacent to an edge formed by two vertices.
    ///
    /// @param v1 First vertex index.
    /// @param v2 Second vertex index.
    /// @param cellIds Output vector to hold the IDs of adjacent cells.
    /// @return True if adjacent cells are found, false otherwise.
    bool
    getAdjacentCells(const Index& v1, const Index& v2, std::vector<Index>& cellIds) const
    {
        Edge edge(v1, v2);
        auto it = mEdgeMap.find(edge);
        if (it != mEdgeMap.end()) {
            cellIds = it->second;
            return true;
        }
        return false;
    }

    /// @brief Retrieves the 3D coordinate at a given index.
    /// @tparam T Any integral type (int, unsigned int, size_t, etc.)
    /// @param i Index of the vertex.
    /// @return The 3D coordinate as a constant reference to Vec3T.
    template <typename T>
    inline const Vec3T&
    getCoord(const T& i) const
    {
        static_assert(std::is_integral<T>::value, "Index must be an integral type");

        return mCoords[i];
    }

    /// @brief Retrieves the cell (triangle) at a given index.
    /// @tparam T Any integral type (int, unsigned int, size_t, etc.)
    /// @param i Index of the cell.
    /// @return Constant reference to the triangle's vertex indices.
    template <typename T>
    inline const Vec3I&
    getCell(const T& i) const
    {
        static_assert(std::is_integral<T>::value, "Index must be an integral type");

        return mCells[i];
    }

    /// @brief Retrieves the 3D coordinates of the vertices forming a
    /// primitive (triangle) at a given cell index.
    /// @tparam T Any integral type (int, unsigned int, size_t, etc.)
    /// @param i Index of the cell (triangle).
    /// @return A vector of three Vec3T representing the coordinates of the triangle's vertices.
    template <typename T>
    inline std::vector<Vec3T>
    getPrimitive(const T& i) const
    {
        static_assert(std::is_integral<T>::value, "Index must be an integral type");

        const Vec3I cell = mCells[i];

        return {mCoords[cell[0]], mCoords[cell[1]], mCoords[cell[2]]};
    }

    /// @brief Retrieves the unit normal vector of a cell (triangle) at a given index.
    /// @tparam T Any integral type (int, unsigned int, size_t, etc.)
    /// @param i Index of the cell.
    /// @return The normal vector of the triangle as a Vec3T.
    template <typename T>
    inline Vec3T
    getNormal(const T& i) const
    {
        static_assert(std::is_integral<T>::value, "Index must be an integral type");

        return mNormals[i];
    }

    /// @brief Retrieves the total number of coordinates in the mesh.
    ///
    /// @return The number of coordinates as an Index.
    inline Index64
    coordCount() const
    {
        return mCoords.size();
    }

    /// @brief Retrieves the total number of cells (triangles) in the mesh.
    ///
    /// @return The number of cells as an Index.
    inline Index64
    cellCount() const
    {
        return mCells.size();
    }

private:
    struct Edge {
        Index mV1, mV2;

        Edge(Index v1, Index v2)
        : mV1(std::min(v1, v2)), mV2(std::max(v1, v2))
        {
        }

        bool operator<(const Edge& e) const
        {
            return mV1 < e.mV1 || (mV1 == e.mV1 && mV2 < e.mV2);
        }
    };

    inline Vec3T
    centroid(Index cellIdx) const
    {
        const Vec3I cell = mCells[cellIdx];
        return (mCoords[cell[0]] + mCoords[cell[1]] + mCoords[cell[2]]) / 3.0;
    }

    inline bool
    onSameHalfPlane(const Vec3T &n, const Vec3T& p0, const Vec3T &p1, const Vec3T &p2)
    {
        return math::Abs(math::Sign(n.dot(p1-p0)) - math::Sign(n.dot(p2-p0))) != 2;
    }

    inline void
    sortAdjacentCells(const Edge& edge, std::vector<Index>& cells)
    {
        if (cells.size() <= 2) return;

        const Vec3I &base_cell = mCells[cells[0]];
        const Index offset = edge.mV1 + edge.mV2;

        const Index p1Ind = base_cell[0] + base_cell[1] + base_cell[2] - offset;

        const Vec3T &p1 = mCoords[p1Ind],
                    &n1 = mNormals[cells[0]];

        const Vec3T p0 = mCoords[edge.mV1];

        Vec3T bi_nrml = n1.cross(p0 - mCoords[edge.mV2]);
        if (bi_nrml.dot(p1 - p0) > 0)
            bi_nrml *= ValueT(-1);

        auto windingamount = [&](Index cellIdx)
        {
            if (cellIdx == 0) return 0.0f;

            const Vec3I &cell = mCells[cellIdx];
            const Index p2Ind = cell[0] + cell[1] + cell[2] - offset;

            const Vec3T &p2 = mCoords[p2Ind],
                        &n2 = mNormals[cellIdx];

            const ValueT cos_theta = math::Abs(n1.dot(n2));
            const int sgn = math::Sign(n1.dot(p2 - p1)),
                      sgn2 = math::Sign(bi_nrml.dot(p2 - p0));

            return sgn != 0
                ? (sgn == 1
                    ? ValueT(1) + ValueT(sgn2) * cos_theta
                    : ValueT(3) - ValueT(sgn2) * cos_theta
                  )
                : (onSameHalfPlane(bi_nrml, p0, p1, p2) ? ValueT(0) : ValueT(2));
        };

        std::sort(cells.begin(), cells.end(), [&](const Index& t1, const Index& t2) {
            return windingamount(t1) < windingamount(t2);
        });
    }

    // ------------ private members ------------

    const std::vector<Vec3T>& mCoords;
    const std::vector<Vec3I>& mCells;

    std::vector<Vec3T> mNormals;

    std::map<Edge, std::vector<Index>> mEdgeMap;

}; // class TriangleMeshEdgeConnectivity


/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of a dilated mesh (surface mesh dilated by a radius in all directions).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the mesh vertex and radius type
/// and must be a floating-point scalar.
template <typename GridType, typename ScalarType = float,
          typename InterruptT = util::NullInterrupter, bool PtPartition = true>
class DilatedMeshVoxelizer {

    using GridPtr = typename GridType::Ptr;
    using TreeT = typename GridType::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    
    using ValueT = typename GridType::ValueType;
    using ComputeT = typename ComputeTypeFor<ValueT>::type;

    using PartitionerT = tools::PointPartitioner<Index32, LeafT::LOG2DIM>;

    using PrismVoxelizer = OpenTriangularPrismVoxelizer<GridType, InterruptT>;
    using WedgeVoxelizer = OpenCapsuleWedgeVoxelizer<GridType, InterruptT>;

    using MeshConnectivity = TriangleMeshEdgeConnectivity<ScalarType>;

    using Vec3T = math::Vec3<ScalarType>;

    static_assert(openvdb::is_floating_point<ValueT>::value);
    static_assert(std::is_floating_point<ScalarType>::value);

public:

    /// @brief Constructor for constant radius
    ///
    /// @param vertices    vertices of the mesh in world units
    /// @param triangles    triangle indices indices in the mesh
    /// @param radius    radius of all faces in world units
    /// @param voxelSize    voxel size in world units
    /// @param halfWidth    half-width in voxel units
    /// @param interrupter    pointer to optional interrupter. Use template
    /// argument util::NullInterrupter if no interruption is desired.
    DilatedMeshVoxelizer(const std::vector<Vec3T>& vertices, const std::vector<Vec3I>& triangles,
        ScalarType radius, float voxelSize, float halfWidth, InterruptT* interrupter)
    : mMesh(std::make_shared<const MeshConnectivity>(MeshConnectivity(vertices, triangles)))
    , mVox(voxelSize), mHw(halfWidth), mRad(radius)
    , mInterrupter(interrupter)
    {
        initializeGrid();

        if constexpr (PtPartition)
            initializePartitioner();

        mPVoxelizer = std::make_shared<PrismVoxelizer>(mGrid, false);
        mWVoxelizer = std::make_shared<WedgeVoxelizer>(mGrid, false);
    }

    DilatedMeshVoxelizer(DilatedMeshVoxelizer& other, tbb::split)
    : mMesh(other.mMesh), mVox(other.mVox), mHw(other.mHw)
    , mRad(other.mRad), mInterrupter(other.mInterrupter)
    , mPtPartitioner(other.mPtPartitioner)
    {
        initializeGrid();

        mPVoxelizer = std::make_shared<PrismVoxelizer>(mGrid, false);
        mWVoxelizer = std::make_shared<WedgeVoxelizer>(mGrid, false);
    }

    void operator()(const tbb::blocked_range<size_t>& rng)
    {
        if (!checkInterrupter())
            return;

        if constexpr (PtPartition) {
            for (size_t i = rng.begin(); i < rng.end(); ++i)
                for (auto it = mPtPartitioner->indices(i); it; ++it)
                    voxelizeTriangle(*it);
        } else {
            for (size_t i = rng.begin(); i < rng.end(); ++i)
                voxelizeTriangle(i);
        }
    }

    void join(DilatedMeshVoxelizer& other)
    {
        tools::CsgUnionOp<TreeT> op(other.mGrid->tree(), Steal());
        tree::DynamicNodeManager<TreeT> nodeManager(mGrid->tree());
        nodeManager.foreachTopDown(op, true);

        other.mGrid = nullptr;
    }

    inline Index64 bucketSize() const { return mPtPartitioner->size(); }

    inline Index64 cellSize() const { return mMesh->cellCount(); }

    inline GridPtr getGrid() const { return mGrid; }

private:

    inline bool
    affinelyIndependent(const Vec3T& p1, const Vec3T& p2, const Vec3T& p3) const
    {
        const Vec3T n = (p2-p1).cross(p3-p1);
        return !math::isApproxZero(n.x())
            || !math::isApproxZero(n.y())
            || !math::isApproxZero(n.z());
    }

    inline void
    voxelizeTriangle(const size_t& i)
    {
        const Vec3I &cell = mMesh->getCell(i);
        const std::vector<Vec3T> pts = mMesh->getPrimitive(i);

        // degenerate triangle
        if (!affinelyIndependent(pts[0], pts[1], pts[2])) {
            voxelizeCapsule(pts[0], pts[1], pts[2]);
            return;
        }

        // prism
        (*mPVoxelizer)(pts[0], pts[1], pts[2], mRad);

        std::vector<Index> cellIds;
        Vec3T n1, n2;

        // wedges
        for (Index j = 0; j < 3; ++j) {
            const bool success = mMesh->getAdjacentCells(cell[j], cell[(j+1) % 3], cellIds);
            if (success && cellIds[0] == i) {
                if (findWedgeNormals(Index(i), j, cellIds, n1, n2))
                    (*mWVoxelizer)(pts[j], pts[(j+1) % 3], mRad, n1, n2);
            }
        }
    }

    inline void
    voxelizeCapsule(const Vec3T& p1, const Vec3T& p2, const Vec3T& p3)
    {
        lvlset::CapsuleVoxelizer<GridType, InterruptT> voxelizer(mGrid, false);

        ComputeT d1 = ComputeT((p2-p1).lengthSqr()),
                 d2 = ComputeT((p3-p2).lengthSqr()),
                 d3 = ComputeT((p1-p3).lengthSqr());

        ComputeT maxd = math::Max(d1, d2, d3);

        if (maxd == d1)
            voxelizer(p1, p2, mRad);
        else if (maxd == d2)
            voxelizer(p2, p3, mRad);
        else
            voxelizer(p3, p1, mRad);
    }

    inline bool
    findWedgeNormals(const Index& cellIdx, const Index& vIdx,
                     const std::vector<Index>& cellIds, Vec3T& n1, Vec3T& n2) const
    {
        if (cellIds.size() == 1)
            return findWedgeNormals1(cellIdx, vIdx, n1, n2);
        else if (cellIds.size() == 2)
            return findWedgeNormals2(cellIdx, vIdx, cellIds[1], n1, n2);
        else if (cellIds.size() > 2)
            return findWedgeNormals3(cellIdx, vIdx, cellIds, n1, n2);

        return false;
    }

    inline bool
    findWedgeNormals1(const Index& cellIdx, const Index& vIdx,
                      Vec3T& n1, Vec3T& n2) const
    {
        const Vec3I &cell = mMesh->getCell(cellIdx);
        const Vec3T &p1 = mMesh->getCoord(cell[vIdx]),
                    &p2 = mMesh->getCoord(cell[(vIdx+1) % 3]),
                    &p3 = mMesh->getCoord(cell[(vIdx+2) % 3]);

        const Vec3T &n = mMesh->getNormal(cellIdx);

        n1 = n.cross(p2-p1).unitSafe();
        if (n1.dot(p3-p1) < 0) n1 *= -1.0f;

        n2 = n1;

        return true;
    }

    inline bool
    findWedgeNormals2(const Index& cellIdx, const Index& vIdx,
                      const Index& cellIdx2, Vec3T& n1, Vec3T& n2) const
    {
        const Vec3I &cell  = mMesh->getCell(cellIdx),
                    &cell2 = mMesh->getCell(cellIdx2);

        const Index cIdx2 = cell2[0] + cell2[1] + cell2[2] - cell[vIdx] - cell[(vIdx+1) % 3];

        const Vec3T &p1 = mMesh->getCoord(cell[vIdx]),
                    &p2 = mMesh->getCoord(cell[(vIdx+1) % 3]),
                    &p3 = mMesh->getCoord(cell[(vIdx+2) % 3]),
                    &p4 = mMesh->getCoord(cIdx2);

        const Vec3T &nrml1 = mMesh->getNormal(cellIdx),
                    &nrml2 = mMesh->getNormal(cellIdx2);

        n1 = nrml1.cross(p2-p1).unitSafe(),
        n2 = nrml2.cross(p2-p1).unitSafe();

        if (n1.dot(p3-p1) < 0) n1 *= -1.0f;
        if (n2.dot(p4-p1) < 0) n2 *= -1.0f;

        return true;
    }

    inline bool
    findWedgeNormals3(const Index& cellIdx, const Index& vIdx,
                      const std::vector<Index>& cellIds, Vec3T& n1, Vec3T& n2) const
    {
        const Vec3I &cell  = mMesh->getCell(cellIdx);

        const Index64 n = cellIds.size();
        const Index offset = cell[vIdx] + cell[(vIdx+1) % 3];

        for (Index64 i = 0; i < n; ++i) {
            const Vec3I &cell0 = mMesh->getCell(cellIds[i]),
                        &cell1 = mMesh->getCell(cellIds[(i+1) % n]),
                        &cell2 = mMesh->getCell(cellIds[(i+2) % n]);

            const Index cIdx0 = cell0[0] + cell0[1] + cell0[2] - offset,
                        cIdx1 = cell1[0] + cell1[1] + cell1[2] - offset,
                        cIdx2 = cell2[0] + cell2[1] + cell2[2] - offset;

            const Vec3T &p0 = mMesh->getCoord(cIdx0),
                        &p1 = mMesh->getCoord(cIdx1),
                        &p2 = mMesh->getCoord(cIdx2);

            Vec3T nrml0 = mMesh->getNormal(cellIds[i]),
                  nrml1 = mMesh->getNormal(cellIds[(i+1) % n]);

            if (nrml0.dot(p1-p0) > 0) nrml0 *= ScalarType(-1);
            if (nrml1.dot(p0-p1) > 0) nrml1 *= ScalarType(-1);

            if (nrml0.dot(p2-p0) > 0 || nrml1.dot(p2-p1) > 0)
                continue;

            Index vIdxi;
            if (cell0[0] == cell[vIdx])
                vIdxi = cell0[1] == cell[(vIdx+1) % 3] ? 0 : 2;
            else if (cell0[1] == cell[vIdx])
                vIdxi = cell0[2] == cell[(vIdx+1) % 3] ? 1 : 0;
            else
                vIdxi = cell0[0] == cell[(vIdx+1) % 3] ? 2 : 1;

            return findWedgeNormals2(cellIds[i], vIdxi, cellIds[(i+1) % n], n1, n2);
        }

        return false;
    }

    inline void
    computeCentroids(std::vector<Vec3T>& centroids)
    {
        centroids.resize(mMesh->cellCount());

        tbb::parallel_for(tbb::blocked_range<size_t>(0, centroids.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const std::vector<Vec3T> prim = mMesh->getPrimitive(i);
                    centroids[i] = (prim[0] + prim[1] + prim[2]) / ScalarType(3);
                }
            });
    }

    inline void
    initializeGrid()
    {
        mGrid = createLevelSet<GridType>(mVox, mHw);
    }

    inline void
    initializePartitioner()
    {
        std::vector<Vec3T> centroids;
        computeCentroids(centroids);

        lvlset::PointArray<Vec3T> points(centroids);

        mPtPartitioner = std::make_shared<PartitionerT>();
        mPtPartitioner->construct(points, mGrid->transform());
    }

    inline bool
    checkInterrupter()
    {
        if (util::wasInterrupted(mInterrupter)) {
            openvdb::thread::cancelGroupExecution();
            return false;
        }
        return true;
    }

    // ------------ private members ------------

    std::shared_ptr<const MeshConnectivity> mMesh;

    const float mVox, mHw;

    const ScalarType mRad;

    InterruptT* mInterrupter;

    GridPtr mGrid;

    std::shared_ptr<PartitionerT> mPtPartitioner;

    std::shared_ptr<PrismVoxelizer> mPVoxelizer;
    std::shared_ptr<WedgeVoxelizer> mWVoxelizer;

}; // class DilatedMeshVoxelizer

} // namespace lvlset


// ------------ createLevelSetDilatedMesh ------------- //

template <typename GridType, typename ScalarType, typename InterruptT>
typename GridType::Ptr
createLevelSetDilatedMesh(
    const std::vector<math::Vec3<ScalarType>>& vertices, const std::vector<Vec3I>& triangles,
    ScalarType radius, float voxelSize, float halfWidth, InterruptT* interrupter)
{
    static_assert(std::is_floating_point<ScalarType>::value);

    using GridPtr = typename GridType::Ptr;
    using ValueT = typename GridType::ValueType;

    using Voxelizer = typename lvlset::DilatedMeshVoxelizer<GridType, ScalarType, InterruptT>;

    static_assert(openvdb::is_floating_point<ValueT>::value,
        "createLevelSetDilatedMesh must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    Voxelizer op(vertices, triangles, radius, voxelSize, halfWidth, interrupter);

    const tbb::blocked_range<size_t> triangleRange(0, op.bucketSize());
    tbb::parallel_reduce(triangleRange, op);

    GridPtr grid = op.getGrid();
    tools::pruneLevelSet(grid->tree());

    return grid;
}

template <typename GridType, typename ScalarType, typename InterruptT>
typename GridType::Ptr
createLevelSetDilatedMesh(
    const std::vector<math::Vec3<ScalarType>>& vertices, const std::vector<Vec4I>& quads,
    ScalarType radius, float voxelSize, float halfWidth, InterruptT* interrupter)
{
    static_assert(std::is_floating_point<ScalarType>::value);

    using ValueT = typename GridType::ValueType;

    static_assert(openvdb::is_floating_point<ValueT>::value,
        "createLevelSetDilatedMesh must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    const Index64 n = quads.size();
    std::vector<Vec3I> triangles(2*n);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&](const tbb::blocked_range<size_t>& r) {
            for (Index64 i = r.begin(); i != r.end(); ++i) {
                const Vec4I& q = quads[i];
                triangles[i]     = Vec3I(q.x(), q.y(), q.z());
                triangles[i + n] = Vec3I(q.x(), q.z(), q.w());
            }
        });

    return createLevelSetDilatedMesh<GridType, ScalarType, InterruptT>(
        vertices, triangles, radius, voxelSize, halfWidth, interrupter);
}

template <typename GridType, typename ScalarType, typename InterruptT>
typename GridType::Ptr
createLevelSetDilatedMesh(const std::vector<math::Vec3<ScalarType>>& vertices,
    const std::vector<Vec3I>& triangles, const std::vector<Vec4I>& quads,
    ScalarType radius, float voxelSize, float halfWidth, InterruptT* interrupter)
{
    static_assert(std::is_floating_point<ScalarType>::value);

    using ValueT = typename GridType::ValueType;

    static_assert(openvdb::is_floating_point<ValueT>::value,
        "createLevelSetDilatedMesh must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    if (quads.empty())
        return createLevelSetDilatedMesh<GridType, ScalarType, InterruptT>(
            vertices, triangles, radius, voxelSize, halfWidth, interrupter);

    const Index64 tn = triangles.size(), qn = quads.size();
    const Index64 qn2 = tn + qn;
    std::vector<Vec3I> tris(tn + 2*qn);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, tn),
        [&](const tbb::blocked_range<size_t>& r) {
            for (Index64 i = r.begin(); i != r.end(); ++i) {
                tris[i] = triangles[i];
            }
        });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, qn),
        [&](const tbb::blocked_range<size_t>& r) {
            for (Index64 i = r.begin(); i != r.end(); ++i) {
                const Vec4I& q = quads[i];
                tris[i + tn]  = Vec3I(q.x(), q.y(), q.z());
                tris[i + qn2] = Vec3I(q.x(), q.z(), q.w());
            }
        });

    return createLevelSetDilatedMesh<GridType, ScalarType, InterruptT>(
        vertices, tris, radius, voxelSize, halfWidth, interrupter);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETDILATEDMESHIMPL_HAS_BEEN_INCLUDED
