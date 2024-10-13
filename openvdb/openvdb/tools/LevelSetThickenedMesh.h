// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @author Greg Hurst
///
/// @file LevelSetThickenedMesh.h
///
/// @brief Generate a narrow-band level set of a thickened mesh.
///
/// @note By definition a level set has a fixed narrow band width
/// (the half width is defined by LEVEL_SET_HALF_WIDTH in Types.h),
/// whereas an SDF can have a variable narrow band width.

#ifndef OPENVDB_TOOLS_LEVELSETTHICKENEDMESH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETTHICKENEDMESH_HAS_BEEN_INCLUDED

#include "ConvexVoxelizer.h"
#include "LevelSetTubes.h"
#include "PointPartitioner.h"
#include "Prune.h"

#include <openvdb/math/Math.h>
#include <openvdb/util/NullInterrupter.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a thickened triangle mesh (thickened by a radius in all directions).
///
/// @param vertices    Vertices of the mesh in world units.
/// @param triangles    Triangle indices of the mesh.
/// @param radius    Radius of the sphere in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetThickenedMesh(
    const std::vector<Vec3s>& vertices, const std::vector<Vec3I>& triangles,
    float radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a thickened quad mesh (thickened by a radius in all directions).
///
/// @param vertices    Vertices of the mesh in world units.
/// @param quads    Quad indices of the mesh.
/// @param radius    Radius of the sphere in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetThickenedMesh(
    const std::vector<Vec3s>& vertices, const std::vector<Vec4I>& quads,
    float radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a thickened triangle & quad mesh (thickened by a radius in all directions).
///
/// @param vertices    Vertices of the mesh in world units.
/// @param triangles    Triangle indices of the mesh.
/// @param quads    Quad indices of the mesh.
/// @param radius    Radius of the sphere in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetThickenedMesh(const std::vector<Vec3s>& vertices,
    const std::vector<Vec3I>& triangles, const std::vector<Vec4I>& quads,
    float radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr);

namespace lvlset {

/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of an _open_ prism.
/// The only parts of the level set populated are along both normals of the triangle.
/// Negative background tiles that fit inside the closed dilated triangle are also populated.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridT, typename InterruptT = util::NullInterrupter>
class OpenTriangularPrismVoxelizer
    : public ConvexVoxelizer<
          GridT,
          OpenTriangularPrismVoxelizer<GridT, InterruptT>,
          InterruptT>
{
    using GridPtr = typename GridT::Ptr;
    using ValueT  = typename GridT::ValueType;

    // ------------ base class members ------------

    using BaseT = ConvexVoxelizer<
        GridT,
        OpenTriangularPrismVoxelizer<GridT, InterruptT>,
        InterruptT
    >;

    using BaseT::mXYData;
    using BaseT::tileCeil;

public:

    friend class ConvexVoxelizer<
        GridT,
        OpenTriangularPrismVoxelizer<GridT, InterruptT>,
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
    void
    operator()(const Vec3s& pt1, const Vec3s& pt2, const Vec3s& pt3, const float& radius)
    {
        if (initialize(pt1, pt2, pt3, radius))
            BaseT::iterate();
    }

private:

    inline void
    setXYRangeData(const Index& step = 1) override
    {
        const float &x1 = mPts[0].x(), &x2 = mPts[1].x(), &x3 = mPts[2].x(),
                    &x4 = mPts[3].x(), &x5 = mPts[4].x(), &x6 = mPts[5].x();

        const float &y1 = mPts[0].y(), &y2 = mPts[1].y(), &y3 = mPts[2].y();

        const float xmin = math::Min(x1, x2, x3, x4, x5, x6);
        const float xmax = math::Max(x1, x2, x3, x4, x5, x6);
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
        const float &x1 = mPts[i].x(), &x2 = mPts[j].x();

        // nothing to do if segment does not span across more than on voxel in x
        // other segments will handle this segment's range
        if (tileCeil(x1, step) == tileCeil(x2, step))
            return;

        const float x_start = tileCeil(math::Min(x1, x2), step);
        const float x_end = math::Max(x1, x2);

        for (float x = x_start; x <= x_end; x += step) {
            if constexpr (MinMax <= 0)
                mXYData.expandYMin(x, line2D<i,j>(x));
            if constexpr (MinMax >= 0)
                mXYData.expandYMax(x, line2D<i,j>(x));
        }
    }

    // simply offset distance to the center plane, we may assume any CPQ falls in inside the prism
    inline float
    signedDistance(const Vec3s& p) const
    {
        return math::Abs(mTriNrml.dot(p - mA)) - mRad;
    }

    // allows for tiles to poke outside of the open prism into the tubes
    // adaptation of udTriangle at https://iquilezles.org/articles/distfunctions/
    inline float
    tilePointSignedDistance(const Vec3s& p) const
    {
        const Vec3s pa = p - mA,
                    pb = p - mB,
                    pc = p - mC;

        const float udist =
            math::Sign(mBAXNrml.dot(pa)) +
            math::Sign(mCBXNrml.dot(pb)) +
            math::Sign(mACXNrml.dot(pc)) < 2.0f
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
    tileCanFit(const Index& dim) const override
    {
        return mRad >= BaseT::halfWidth() + 0.5f * (dim-1u);
    }

    std::function<bool(float&, float&, const float&, const float&)> prismBottomTop =
    [this](float& zb, float& zt, const float& x, const float& y)
    {
        zb = std::numeric_limits<float>::lowest();
        zt = std::numeric_limits<float>::max();

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
    setPlaneBottomTop(float& zb, float& zt, const float& x, const float& y) const
    {
        if (math::isApproxZero(mFaceNrmls[i].z()))
            return;

        const float z = mPlaneXCoeffs[i]*x + mPlaneYCoeffs[i]*y + mPlaneOffsets[i];

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
    inline bool
    initialize(const Vec3s& pt1, const Vec3s& pt2, const Vec3s& pt3, const float& r)
    {
        const float vx = BaseT::voxelSize(),
                    hw = BaseT::halfWidth();

        mA = pt1/vx;
        mB = pt2/vx;
        mC = pt3/vx;

        mRad = r/vx;
        if (math::isApproxZero(mRad) || mRad < 0)
            return false; // nothing to voxelize, prism has no volume

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

        const float len = mTriNrml.length();
        if (math::isApproxZero(len)) {
            return false; // nothing to voxelize, prism has no volume
        } else {
            mTriNrml /= len;
        }

        const float hwRad = mRad + hw;
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

            mPlaneXCoeffs.assign(5, 0.0f);
            mPlaneYCoeffs.assign(5, 0.0f);
            mPlaneOffsets.assign(5, 0.0f);

            for (Index i = 0; i < 5; ++i) {
                if (!math::isApproxZero(mFaceNrmls[i].z())) {
                    const float cx = mFaceNrmls[i].x()/mFaceNrmls[i].z(),
                                cy = mFaceNrmls[i].y()/mFaceNrmls[i].z();
                    const Vec3s p = mPts[p_ind[i]];
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
    float
    line2D(const float& x) const
    {
        const float &x1 = mPts[i].x(), &y1 = mPts[i].y(),
                    &x2 = mPts[j].x(), &y2 = mPts[j].y();

        const float m = (y2-y1)/(x2-x1);

        return y1 + m * (x-x1);
    }

    // ------------ private members ------------

    Vec3s mA, mB, mC;
    float mRad;

    Vec3s mBA, mCB, mAC;
    Vec3s mBAXNrml, mCBXNrml, mACXNrml;
    Vec3s mBANorm2, mCBNorm2, mACNorm2;

    std::vector<Vec3s> mPts = std::vector<Vec3s>(6);

    Vec3s mTriNrml;
    std::vector<Vec3s> mFaceNrmls = std::vector<Vec3s>(5);

    std::vector<float> mPlaneXCoeffs = std::vector<float>(5),
                  mPlaneYCoeffs = std::vector<float>(5),
                  mPlaneOffsets = std::vector<float>(5);

}; // class OpenTriangularPrismVoxelizer

/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of an _open_ wedge.
/// The only parts of the level set populated are within a sector of a capsule.
/// The sector is defined by the intersection of two half spaces.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridT, typename InterruptT = util::NullInterrupter>
class OpenTubeWedgeVoxelizer
    : public ConvexVoxelizer<
          GridT,
          OpenTubeWedgeVoxelizer<GridT, InterruptT>,
          InterruptT>
{
    using GridPtr = typename GridT::Ptr;
    using ValueT  = typename GridT::ValueType;

    // ------------ base class members ------------

    using BaseT = ConvexVoxelizer<
        GridT,
        OpenTubeWedgeVoxelizer<GridT, InterruptT>,
        InterruptT
    >;

    using BaseT::mXYData;
    using BaseT::tileCeil;

public:

    friend class ConvexVoxelizer<
        GridT,
        OpenTubeWedgeVoxelizer<GridT, InterruptT>,
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
    OpenTubeWedgeVoxelizer(GridPtr& grid, const bool& threaded = false,
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
    void
    operator()(const Vec3s& pt1, const Vec3s& pt2, const float& radius,
                    const Vec3s& nrml1, const Vec3s& nrml2)
    {
        if (initialize(pt1, pt2, radius, nrml1, nrml2))
            BaseT::iterate();
    }

private:

    // computes *approximate* xy-range data: the projected caps might contain over-inclusive values
    inline void
    setXYRangeData(const Index& step = 1) override
    {
        // short circuit a vertical cylinder
        if (mIsVertical) {
            mXYData.reset(mX1 - mORad, mX1 + mORad, step);

            for (float x = tileCeil(mX1 - mORad, step); x <= mX1 + mORad; x += step)
                mXYData.expandYRange(x, circle1Bottom(x), circle1Top(x));

            intersectWithXYWedgeLines();
            return;
        }

        const float v = math::Min(mORad, mORad * math::Abs(mYdiff)/mXYNorm);

        const float a0 = mX1 - mORad,
                    a1 = mX1 - v,
                    a2 = mX1 + v,
                    a3 = mX2 - v,
                    a4 = mX2 + v,
                    a5 = mX2 + mORad;

        const float tc0 = tileCeil(a0, step),
                    tc1 = tileCeil(a1, step),
                    tc2 = tileCeil(a2, step),
                    tc3 = tileCeil(a3, step),
                    tc4 = tileCeil(a4, step);

        mXYData.reset(a0, a5, step);

        for (float x = tc0; x <= a1; x += step)
            mXYData.expandYRange(x, circle1Bottom(x), circle1Top(x));

        if (!math::isApproxZero(mXdiff)) {
            if (mY1 > mY2) {
                for (float x = tc1; x <= math::Min(a2, a3); x += step)
                    mXYData.expandYRange(x, lineBottom(x), circle1Top(x));
            } else {
                for (float x = tc1; x <= math::Min(a2, a3); x += step)
                    mXYData.expandYRange(x, circle1Bottom(x), lineTop(x));
            }
        }

        if (a2 < a3) {
            for (float x = tc2; x <= a3; x += step)
                mXYData.expandYRange(x, lineBottom(x), lineTop(x));
        } else {
            if (mY2 <= mY1) {
                for (float x = tc3; x <= a2; x += step)
                    mXYData.expandYRange(x, circle2Bottom(x), circle1Top(x));
            } else {
                for (float x = tc3; x <= a2; x += step)
                    mXYData.expandYRange(x, circle1Bottom(x), circle2Top(x));
            }
        }

        if (!math::isApproxZero(mXdiff)) {
            if (mY1 > mY2) {
                for (float x = math::Max(tc2, tc3); x <= a4; x += step)
                    mXYData.expandYRange(x, circle2Bottom(x), lineTop(x));
            } else {
                for (float x = math::Max(tc2, tc3); x <= a4; x += step)
                    mXYData.expandYRange(x, lineBottom(x), circle2Top(x));
            }
        }

        for (float x = tc4; x <= a5; x += step)
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

        const Vec3s &pp1 = mPlanePts[0], &pp2 = mPlanePts[1];
        const float &vx = mV.x(), &vy = mV.y(), &vz = mV.z();

        Vec2s n = Vec2s(-vy, vx).unitSafe();
        Vec3s cvec = mORad * Vec3s(-vy, vx, 0.0f).unitSafe();

        if (math::isApproxZero(vy)) cvec.y() = math::Abs(cvec.y());
        else if (vy > 0.0f) cvec *= -1.0f;

        const Vec3s cpmin(mPt1 - cvec), cpmax(mPt1 + cvec);

        if (math::isApproxZero(mXdiff)) {
            const float px = mPt1.x(), py = mPt1.y();
            const float xmin = math::Min(px, pp1.x(), pp2.x()),
                        xmax = math::Max(px, pp1.x(), pp2.x());

            if (!inWedge(cpmin))
                intersectWithXYHalfSpace(n.x() < 0 ? n : -n, Vec2s(xmin, 0.0f));

            if (!inWedge(cpmax))
                intersectWithXYHalfSpace(n.x() > 0 ? n : -n, Vec2s(xmax, 0.0f));
        } else {
            const float m = mYdiff/mXdiff;
            const float y1 = mPt1.y() - m * mPt1.x(),
                        y2 = pp1.y() - m * pp1.x(),
                        y3 = pp2.y() - m * pp2.x();
            const float ymin = math::Min(y1, y2, y3),
                        ymax = math::Max(y1, y2, y3);

            if (!inWedge(vy <= 0.0f ? cpmin : cpmax))
                intersectWithXYHalfSpace(n.y() < 0 ? n : -n, Vec2s(0.0f, ymin));

            if (!inWedge(vy > 0.0f ? cpmin : cpmax))
                intersectWithXYHalfSpace(n.y() > 0 ? n : -n, Vec2s(0.0f, ymax));
        }
    }

    inline void
    intersectWithXYWedgeLines()
    {
        const Vec3s v(mORad * mV.unitSafe()),
                    p1(mPt1 - v),
                    p2(mPt2 + v);

        const Vec2s p1_2d(p1.x(), p1.y()), p2_2d(p2.x(), p2.y());

        Vec2s d(-mPlaneNrmls[0].x() - mPlaneNrmls[1].x(),
                -mPlaneNrmls[0].y() - mPlaneNrmls[1].y());

        Vec2s n0(-mDirVectors[0].y(), mDirVectors[0].x()),
              n1(-mDirVectors[1].y(), mDirVectors[1].x());

        if (n0.dot(d) > 0.0f)
            n0 *= -1;
        if (n1.dot(d) > 0.0f)
            n1 *= -1;

        if (!math::isApproxZero(n0.lengthSqr()))
            intersectWithXYHalfSpace(n0, n0.dot(p2_2d - p1_2d) < 0.0f ? p1_2d : p2_2d);

        if (!math::isApproxZero(n1.lengthSqr()))
            intersectWithXYHalfSpace(n1, n1.dot(p2_2d - p1_2d) < 0.0f ? p1_2d : p2_2d);
    }

    inline void
    intersectWithXYHalfSpace(const Vec2s& n, const Vec2s& p)
    {
        if (mXYData.size() == 0)
            return;

        if (math::isApproxZero(n.y())) {
            const float px = p.x();
            if (n.x() < 0) {
                const Index m = mXYData.size();
                for (Index i = 0; i < m; ++i) {
                    const float x = mXYData.getX(i);

                    if (x < px) mXYData.clearYRange(x);
                    else break;
                }
            } else {
                Index i = mXYData.size()-1;
                while (true) {
                    const float x = mXYData.getX(i);

                    if (x > px) mXYData.clearYRange(x);
                    else break;

                    if (i != 0) --i;
                    else break;
                }
            }
        } else {
            const bool set_min = n.y() < 0;
            const Index m = mXYData.size();

            const float b = -n.x()/n.y();
            const float a = p.y() - b * p.x();

            float x, ymin, ymax;
            for (Index i = 0; i < m; ++i) {
                mXYData.XYData(x, ymin, ymax, i);
                const float yint = a + b * x;

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
    inline float
    signedDistance(const Vec3s& p) const
    {
        const Vec3s w = p - mPt1;
        const float dot = w.dot(mV);

        // carefully short circuit with a fuzzy tolerance, which avoids division by small mVLenSqr
        if (dot <= math::Tolerance<float>::value())
            return w.length() - mRad;

        if (dot >= mVLenSqr)
            return (p - mPt2).length() - mRad;

        const float t = w.dot(mV)/mVLenSqr;

        return (w - t * mV).length() - mRad;
    }

    inline bool
    tileCanFit(const Index& dim) const override
    {
        return mRad >= BaseT::halfWidth() + 0.5f * (dim-1u);
    }

    std::function<bool(float&, float&, const float&, const float&)> tubeBottomTopVertical =
    [this](float& zb, float& zt, const float& x, const float& y)
    {
        zb = BaseT::sphereBottom(mX1, mY1, math::Min(mZ1, mZ2), mORad, x, y);
        zt = BaseT::sphereTop(mX2, mY2, math::Max(mZ1, mZ2), mORad, x, y);

        return std::isfinite(zb) && std::isfinite(zt);
    };

    std::function<bool(float&, float&, const float&, const float&)> tubeBottomTop =
    [this](float& zb, float& zt, const float& x, const float& y)
    {
        float cylptb, cylptt;
        if (!infiniteCylinderBottomTop(cylptb, cylptt, x, y))
            return false;

        const float dotb = (Vec3s(x, y, cylptb) - mPt1).dot(mV);
        const float dott = (Vec3s(x, y, cylptt) - mPt1).dot(mV);

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

        if (!std::isfinite(zb) || !std::isfinite(zt))
            return false;

        intersectWedge<0,1>(zb, zt, x, y);
        intersectWedge<1,0>(zb, zt, x, y);

        return inWedge(x, y, 0.5f*(zb+zt));
    };

    template<Index i, Index j>
    inline void
    intersectWedge(float& zb, float& zt, const float& x, const float& y)
    {
        const Vec3s& n0 = mPlaneNrmls[i];

        if (math::isApproxZero(n0.z()))
            return;

        const float zp = mPlaneXCoeffs[i]*x + mPlaneYCoeffs[i]*y + mPlaneOffsets[i];

        if (zb <= zp && zp <= zt && inHalfSpace<j>(Vec3s(x, y, zp))) {
            if (n0.z() < 0)
                zb = zp;
            else
                zt = zp;
        }
    }

    inline bool
    inWedge(const float& x, const float& y, const float& z)
    {
        return inWedge(Vec3s(x, y, z));
    }

    inline bool
    inWedge(const Vec3s& pt)
    {
        return inHalfSpace<0>(pt) && inHalfSpace<1>(pt);
    }

    template<Index i>
    inline bool
    inHalfSpace(const Vec3s& pt)
    {
        // allow points within a fuzzy fractional (index space) distance to the halfspace
        // this ensures the seams between open wedges and open prisms are completely filled in
        // assumes mPlaneNrmls[i] is a unit vector
        static const float VOXFRAC = 0.125f;

        return mPlaneNrmls[i].dot(pt-mPt1) <= VOXFRAC;
    }

    // assumes tube is not vertical!
    inline bool
    infiniteCylinderBottomTop(float& cylptb, float& cylptt,
        const float& x, const float& y) const
    {
        const Vec2s q(x, y);

        const Vec2s qproj = mPt12d + mV2d*((q - mPt12d).dot(mV2d))/mXYNorm2;

        const float t = mX1 != mX2 ? (qproj[0] - mX1)/mXdiff : (qproj[1] - mY1)/mYdiff;

        const Vec3s qproj3D = mPt1 + t * mV;

        const float d2 = (q - qproj).lengthSqr();

        // outside of cylinder's 2D projection
        if (mORad2 < d2)
            return false;

        const float h = math::Sqrt((mORad2 - d2) * mVLenSqr/mXYNorm2);

        cylptb = qproj3D[2] - h;
        cylptt = qproj3D[2] + h;

        return true;
    }

    inline float
    lineBottom(const float& x) const
    {
        return mY1 + (mYdiff*(x-mX1) - mORad * mXYNorm)/mXdiff;
    }

    inline float
    lineTop(const float& x) const
    {
        return mY1 + (mYdiff*(x-mX1) + mORad * mXYNorm)/mXdiff;
    }

    inline float
    circle1Bottom(const float& x) const
    {
        return BaseT::circleBottom(mX1, mY1, mORad, x);
    }

    inline float
    circle1Top(const float& x) const
    {
        return BaseT::circleTop(mX1, mY1, mORad, x);
    }

    inline float
    circle2Bottom(const float& x) const
    {
        return BaseT::circleBottom(mX2, mY2, mORad, x);
    }

    inline float
    circle2Top(const float& x) const
    {
        return BaseT::circleTop(mX2, mY2, mORad, x);
    }

    inline float
    sphere1Bottom(const float& x, const float& y) const
    {
        return BaseT::sphereBottom(mX1, mY1, mZ1, mORad, x, y);
    }

    inline float
    sphere1Top(const float& x, const float& y) const
    {
        return BaseT::sphereTop(mX1, mY1, mZ1, mORad, x, y);
    }

    inline float
    sphere2Bottom(const float& x, const float& y) const
    {
        return BaseT::sphereBottom(mX2, mY2, mZ2, mORad, x, y);
    }

    inline float
    sphere2Top(const float& x, const float& y) const
    {
        return BaseT::sphereTop(mX2, mY2, mZ2, mORad, x, y);
    }

    // world space points and radius inputs
    // initializes class members in index space
    inline bool
    initialize(const Vec3s& pt1, const Vec3s& pt2, const float& r,
               const Vec3s& n1, const Vec3s& n2)
    {
        const float vx = BaseT::voxelSize(),
                    hw = BaseT::halfWidth();

        // forces x1 <= x2
        if (pt1[0] <= pt2[0]) {
            mPt1 = pt1/vx;
            mPt2 = pt2/vx;
        } else {
            mPt1 = pt2/vx;
            mPt2 = pt1/vx;
        }

        mRad = r/vx;

        // tube has no volume
        if (math::isApproxZero(mRad))
            return false;

        // padded radius used to populate the outer halfwidth of the sdf
        mORad  = mRad + hw;
        mORad2 = mORad * mORad;

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

        mPt12d = Vec2s(mX1, mY1);
        mPt22d = Vec2s(mX2, mY2);
        mV2d = mPt22d - mPt12d;

        mXYNorm2 = math::Pow2(mXdiff) + math::Pow2(mYdiff);
        mXYNorm = math::Sqrt(mXYNorm2);
        mIsVertical = math::isApproxZero(mXYNorm);

        {
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
                if (mPlaneNrmls[1].dot(mDirVectors[0]) > 0.0f)
                    mDirVectors[0] *= -1.0f;
                if (mPlaneNrmls[0].dot(mDirVectors[1]) > 0.0f)
                    mDirVectors[1] *= -1.0f;
            }

            mPlanePts[0] = mPt1 + mDirVectors[0] + 0.025f * mPlaneNrmls[0];
            mPlanePts[1] = mPt1 + mDirVectors[1] + 0.025f * mPlaneNrmls[1];
        }

        {
            mPlaneXCoeffs.assign(2, 0.0f);
            mPlaneYCoeffs.assign(2, 0.0f);
            mPlaneOffsets.assign(2, 0.0f);

            for (Index i = 0; i < 2; ++i) {
                if (!math::isApproxZero(mPlaneNrmls[i].z())) {
                    const float cx = mPlaneNrmls[i].x()/mPlaneNrmls[i].z(),
                                cy = mPlaneNrmls[i].y()/mPlaneNrmls[i].z();
                    const Vec3s p = mPlanePts[i];
                    mPlaneXCoeffs[i] = -cx;
                    mPlaneYCoeffs[i] = -cy;
                    mPlaneOffsets[i] = p.x()*cx + p.y()*cy + p.z();
                }
            }
        }

        BaseT::bottomTop = mIsVertical ? tubeBottomTopVertical : tubeBottomTop;

        return true;
    }

    inline bool
    approxAntiParallel(const Vec3s& n1, const Vec3s& n2)
    {
        return approxParallel(n1, -n2);
    }

    inline bool
    approxParallel(const Vec3s& n1, const Vec3s& n2)
    {
        return n1.unitSafe().eq(n2.unitSafe());
    }

    // ------------ private members ------------

    // tube data -- populated via initialize()

    Vec3s mPt1, mPt2, mV;

    Vec2s mPt12d, mPt22d, mV2d;

    float mORad, mORad2, mRad, mVLenSqr, mXdiff, mYdiff, mZdiff, mXYNorm, mXYNorm2;

    float mX1, mY1, mZ1, mX2, mY2, mZ2;

    bool mIsVertical;

    std::vector<Vec3s> mPlaneNrmls = std::vector<Vec3s>(2),
                  mDirVectors = std::vector<Vec3s>(2),
                  mPlanePts   = std::vector<Vec3s>(2);

    std::vector<float> mPlaneXCoeffs = std::vector<float>(2),
                  mPlaneYCoeffs = std::vector<float>(2),
                  mPlaneOffsets = std::vector<float>(2);

}; // class OpenTubeWedgeVoxelizer


/// @brief Class representing the connectivity of edges in a triangle mesh,
/// where each edge is associated with the cells (triangles) sharing it.
/// Provides methods to retrieve adjacent cells,
/// vertex coordinates, normals, and other geometric properties.
class TriangleMeshEdgeConnectivity {

public:

    /// @brief Constructs the TriangleMeshEdgeConnectivity object with given coordinates and cell data.
    /// Populates edge-to-cell adjacency and computes cell normals.
    ///
    /// @param coords    Vector of vertex coordinates.
    /// @param cells    Vector of cell (triangle) indices.
    TriangleMeshEdgeConnectivity(const std::vector<Vec3s>& coords, const std::vector<Vec3I>& cells)
    : mCoords(coords), mCells(cells)
    {
        const Index n = coords.size();

        mNormals.resize(cells.size());

        for (Index i = 0; i < cells.size(); i++) {
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

            const Vec3s &p1 = mCoords[cell[0]],
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
    ///
    /// @param i Index of the vertex.
    /// @return The 3D coordinate as a constant reference to Vec3s.
    inline const Vec3s&
    getCoord(const Index& i) const
    {
        return mCoords[i];
    }

    /// @brief Retrieves the cell (triangle) at a given index.
    ///
    /// @param i Index of the cell.
    /// @return Constant reference to the triangle's vertex indices.
    inline const Vec3I&
    getCell(const Index& i) const
    {
        return mCells[i];
    }

    /// @brief Retrieves the 3D coordinates of the vertices forming a
    /// primitive (triangle) at a given cell index.
    ///
    /// @param i Index of the cell (triangle).
    /// @return A vector of three Vec3s representing the coordinates of the triangle's vertices.
    inline std::vector<Vec3s>
    getPrimitive(const Index& i) const
    {
        const Vec3I cell = mCells[i];
        return {mCoords[cell[0]], mCoords[cell[1]], mCoords[cell[2]]};
    }

    /// @brief Retrieves the unit normal vector of a cell (triangle) at a given index.
    ///
    /// @param i Index of the cell.
    /// @return The normal vector of the triangle as a Vec3s.
    inline Vec3s
    getNormal(const Index& i) const
    {
        return mNormals[i];
    }

    /// @brief Retrieves the total number of coordinates in the mesh.
    ///
    /// @return The number of coordinates as an Index.
    inline Index
    coordCount() const
    {
        return mCoords.size();
    }

    /// @brief Retrieves the total number of cells (triangles) in the mesh.
    ///
    /// @return The number of cells as an Index.
    inline Index
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

    inline Vec3s
    centroid(Index cellIdx) const
    {
        const Vec3I cell = mCells[cellIdx];
        return (mCoords[cell[0]] + mCoords[cell[1]] + mCoords[cell[2]]) / 3.0;
    }

    inline bool
    onSameHalfPlane(const Vec3s &n, const Vec3s& p0, const Vec3s &p1, const Vec3s &p2)
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

        const Vec3s &p1 = mCoords[p1Ind],
                    &n1 = mNormals[cells[0]];

        const Vec3s p0 = mCoords[edge.mV1];

        Vec3s bi_nrml = n1.cross(p0 - mCoords[edge.mV2]);
        if (bi_nrml.dot(p1 - p0) > 0) bi_nrml *= -1.0f;

        auto windingamount = [&](Index cellIdx)
        {
            if (cellIdx == 0) return 0.0f;

            const Vec3I &cell = mCells[cellIdx];
            const Index p2Ind = cell[0] + cell[1] + cell[2] - offset;

            const Vec3s &p2 = mCoords[p2Ind],
                        &n2 = mNormals[cellIdx];

            const float cos_theta = math::Abs(n1.dot(n2));
            const int sgn = math::Sign(n1.dot(p2 - p1)),
                      sgn2 = math::Sign(bi_nrml.dot(p2 - p0));

            return sgn != 0
                ? (sgn == 1 ? 1.0f + sgn2 * cos_theta : 3.0f - sgn2 * cos_theta)
                : (onSameHalfPlane(bi_nrml, p0, p1, p2) ? 0.0f : 2.0f);
        };

        std::sort(cells.begin(), cells.end(), [&](const Index& t1, const Index& t2) {
            return windingamount(t1) < windingamount(t2);
        });
    }

    // ------------ private members ------------

    const std::vector<Vec3s>& mCoords;
    const std::vector<Vec3I>& mCells;

    std::vector<Vec3s> mNormals;

    std::map<Edge, std::vector<Index>> mEdgeMap;

}; // class TriangleMeshEdgeConnectivity


/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of a thckened mesh (surface mesh thickened by a radius in all directions).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridT, typename InterruptT, bool PtPartition = true>
class ThickenedMeshVoxelizer {

    using GridPtr = typename GridT::Ptr;
    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;

    using PartitionerT = tools::PointPartitioner<Index32, LeafT::LOG2DIM>;

    using PrismVoxelizer = OpenTriangularPrismVoxelizer<GridT, InterruptT>;
    using WedgeVoxelizer = OpenTubeWedgeVoxelizer<GridT, InterruptT>;

    using MeshConnectivity = TriangleMeshEdgeConnectivity;

public:

    /// @brief Constructor for constant radius
    ///
    /// @param vertices    vertices of the mesh in world units
    /// @param triangles    triangle indices indices in the mesh
    /// @param radius    radius of all tubes in world units
    /// @param voxelSize    voxel size in world units
    /// @param background    background value in voxel units
    /// @param interrupter    pointer to optional interrupter. Use template
    /// argument util::NullInterrupter if no interruption is desired.
    /// @param grid    optional grid to populate into (grid need not be empty).
    ThickenedMeshVoxelizer(const std::vector<Vec3s>& vertices, const std::vector<Vec3I>& triangles,
        float radius, float voxelSize, float background,
        InterruptT* interrupter, GridPtr grid = nullptr)
    : mVox(voxelSize), mBg(background)
    , mMesh(std::make_shared<const MeshConnectivity>(MeshConnectivity(vertices, triangles)))
    , mRad(radius), mInterrupter(interrupter)
    {
        if (!grid)
            initializeGrid();
        else
            mGrid = grid;

        if constexpr (PtPartition)
            initializePartitioner();

        mPVoxelizer = std::make_shared<PrismVoxelizer>(mGrid, false);
        mWVoxelizer = std::make_shared<WedgeVoxelizer>(mGrid, false);
    }

    ThickenedMeshVoxelizer(ThickenedMeshVoxelizer& other, tbb::split)
    : mVox(other.mVox), mBg(other.mBg)
    , mPtPartitioner(other.mPtPartitioner), mMesh(other.mMesh)
    , mRad(other.mRad), mInterrupter(other.mInterrupter)
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

    void join(ThickenedMeshVoxelizer& other)
    {
        tools::CsgUnionOp<TreeT> op(other.mGrid->tree(), Steal());
        tree::DynamicNodeManager<TreeT> nodeManager(mGrid->tree());
        nodeManager.foreachTopDown(op, true);

        other.mGrid = nullptr;
    }

    inline Index bucketSize() const { return mPtPartitioner->size(); }

    inline Index cellSize() const { return mMesh->cellCount(); }

    inline GridPtr getGrid() const { return mGrid; }

private:

    inline bool
    affinelyIndependent(const Vec3s& p1, const Vec3s& p2, const Vec3s& p3) const
    {
        const Vec3s n = (p2-p1).cross(p3-p1);
        return !math::isApproxZero(n.x())
            || !math::isApproxZero(n.y())
            || !math::isApproxZero(n.z());
    }

    inline void
    voxelizeTriangle(const size_t& i)
    {
        const Vec3I &cell = mMesh->getCell(i);
        const std::vector<Vec3s> pts = mMesh->getPrimitive(i);

        if (!affinelyIndependent(pts[0], pts[1], pts[2])) {
            voxelizeTube(pts[0], pts[1], pts[2]);
            return;
        }

        // prism
        (*mPVoxelizer)(pts[0], pts[1], pts[2], mRad);

        std::vector<Index> cellIds;
        Vec3s n1, n2;

        // wedges
        for (Index j = 0; j < 3; ++j) {
            const bool success = mMesh->getAdjacentCells(cell[j], cell[(j+1) % 3], cellIds);
            if (success && cellIds[0] == i) {
                if (findWedgeNormals(i, j, cellIds, n1, n2))
                    (*mWVoxelizer)(pts[j], pts[(j+1) % 3], mRad, n1, n2);
            }
        }
    }

    inline void
    voxelizeTube(const Vec3s& p1, const Vec3s& p2, const Vec3s& p3)
    {
        lvlset::CapsuleVoxelizer<GridT, InterruptT> voxelizer(mGrid, false);

        float d1 = (p2-p1).lengthSqr(),
              d2 = (p3-p2).lengthSqr(),
              d3 = (p1-p3).lengthSqr();

        float maxd = math::Max(d1, d2, d3);

        if (maxd == d1)
            voxelizer(p1, p2, mRad);
        else if (maxd == d2)
            voxelizer(p2, p3, mRad);
        else
            voxelizer(p3, p1, mRad);
    }

    inline bool
    findWedgeNormals(const Index& cellIdx, const Index& vIdx,
                     const std::vector<Index>& cellIds, Vec3s& n1, Vec3s& n2) const
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
                      Vec3s& n1, Vec3s& n2) const
    {
        const Vec3I &cell = mMesh->getCell(cellIdx);
        const Vec3s &p1 = mMesh->getCoord(cell[vIdx]),
                    &p2 = mMesh->getCoord(cell[(vIdx+1) % 3]),
                    &p3 = mMesh->getCoord(cell[(vIdx+2) % 3]);

        const Vec3s &n = mMesh->getNormal(cellIdx);

        n1 = n.cross(p2-p1).unitSafe();
        if (n1.dot(p3-p1) < 0) n1 *= -1.0f;

        n2 = n1;

        return true;
    }

    inline bool
    findWedgeNormals2(const Index& cellIdx, const Index& vIdx,
                      const Index& cellIdx2, Vec3s& n1, Vec3s& n2) const
    {
        const Vec3I &cell  = mMesh->getCell(cellIdx),
                    &cell2 = mMesh->getCell(cellIdx2);

        const Index cIdx2 = cell2[0] + cell2[1] + cell2[2] - cell[vIdx] - cell[(vIdx+1) % 3];

        const Vec3s &p1 = mMesh->getCoord(cell[vIdx]),
                    &p2 = mMesh->getCoord(cell[(vIdx+1) % 3]),
                    &p3 = mMesh->getCoord(cell[(vIdx+2) % 3]),
                    &p4 = mMesh->getCoord(cIdx2);

        const Vec3s &nrml1 = mMesh->getNormal(cellIdx),
                    &nrml2 = mMesh->getNormal(cellIdx2);

        n1 = nrml1.cross(p2-p1).unitSafe(),
        n2 = nrml2.cross(p2-p1).unitSafe();

        if (n1.dot(p3-p1) < 0) n1 *= -1.0f;
        if (n2.dot(p4-p1) < 0) n2 *= -1.0f;

        return true;
    }

    inline bool
    findWedgeNormals3(const Index& cellIdx, const Index& vIdx,
                      const std::vector<Index>& cellIds, Vec3s& n1, Vec3s& n2) const
    {
        const Vec3I &cell  = mMesh->getCell(cellIdx);

        const Index n = cellIds.size(),
                    offset = cell[vIdx] + cell[(vIdx+1) % 3];

        for (Index i = 0; i < n; ++i) {
            const Vec3I &cell0 = mMesh->getCell(cellIds[i]),
                        &cell1 = mMesh->getCell(cellIds[(i+1) % n]),
                        &cell2 = mMesh->getCell(cellIds[(i+2) % n]);

            const Index cIdx0 = cell0[0] + cell0[1] + cell0[2] - offset,
                        cIdx1 = cell1[0] + cell1[1] + cell1[2] - offset,
                        cIdx2 = cell2[0] + cell2[1] + cell2[2] - offset;

            const Vec3s &p0 = mMesh->getCoord(cIdx0),
                        &p1 = mMesh->getCoord(cIdx1),
                        &p2 = mMesh->getCoord(cIdx2);

            Vec3s nrml0 = mMesh->getNormal(cellIds[i]),
                  nrml1 = mMesh->getNormal(cellIds[(i+1) % n]);

            if (nrml0.dot(p1-p0) > 0) nrml0 *= -1.0f;
            if (nrml1.dot(p0-p1) > 0) nrml1 *= -1.0f;

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
    computeCentroids(std::vector<Vec3s>& centroids)
    {
        centroids.resize(mMesh->cellCount());

        tbb::parallel_for(tbb::blocked_range<size_t>(0, centroids.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const std::vector<Vec3s> prim = mMesh->getPrimitive(i);
                    centroids[i] = (prim[0] + prim[1] + prim[2]) / 3.0f;
                }
            });
    }

    inline void
    initializeGrid()
    {
        math::Transform transform(*(math::Transform::createLinearTransform(mVox)));
        mGrid = GridPtr(new GridT(mBg));
        mGrid->setTransform(transform.copy());
        mGrid->setGridClass(GRID_LEVEL_SET);
    }

    inline void
    initializePartitioner()
    {
        std::vector<Vec3s> centroids;
        computeCentroids(centroids);

        lvlset::PointArray<Vec3s> points(centroids);

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

    const float mVox, mBg, mRad;

    std::shared_ptr<PartitionerT> mPtPartitioner;

    GridPtr mGrid;

    std::shared_ptr<PrismVoxelizer> mPVoxelizer;
    std::shared_ptr<WedgeVoxelizer> mWVoxelizer;

    InterruptT* mInterrupter;

}; // class ThickenedMeshVoxelizer

} // namespace lvlset


// ------------ createLevelSetThickenedMesh ------------- //

template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetThickenedMesh(
    const std::vector<Vec3s>& vertices, const std::vector<Vec3I>& triangles,
    float radius, float voxelSize, float halfWidth, InterruptT* interrupter)
{
    using GridPtr = typename GridType::Ptr;
    using ValueT = typename GridType::ValueType;

    using Voxelizer = typename lvlset::ThickenedMeshVoxelizer<GridType, InterruptT>;

    static_assert(std::is_floating_point<ValueT>::value,
        "createLevelSetThickenedMesh must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    Voxelizer op(vertices, triangles, radius, voxelSize, voxelSize * halfWidth, interrupter);

    const tbb::blocked_range<size_t> triangleRange(0, op.bucketSize());
    tbb::parallel_reduce(triangleRange, op);

    GridPtr grid = op.getGrid();
    tools::pruneLevelSet(grid->tree());

    return grid;
}

template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetThickenedMesh(
    const std::vector<Vec3s>& vertices, const std::vector<Vec4I>& quads,
    float radius, float voxelSize, float halfWidth, InterruptT* interrupter)
{
    using ValueT = typename GridType::ValueType;

    static_assert(std::is_floating_point<ValueT>::value,
        "createLevelSetThickenedMesh must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    const Index n = quads.size();
    std::vector<Vec3I> triangles(2*n);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const Vec4I& q = quads[i];
                triangles[i]     = Vec3I(q.x(), q.y(), q.z());
                triangles[i + n] = Vec3I(q.x(), q.z(), q.w());
            }
        });

    return createLevelSetThickenedMesh<GridType, InterruptT>(vertices, triangles, radius,
                                                             voxelSize, halfWidth, interrupter);
}

template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetThickenedMesh(const std::vector<Vec3s>& vertices,
    const std::vector<Vec3I>& triangles, const std::vector<Vec4I>& quads,
    float radius, float voxelSize, float halfWidth, InterruptT* interrupter)
{
    using ValueT = typename GridType::ValueType;

    static_assert(std::is_floating_point<ValueT>::value,
        "createLevelSetThickenedMesh must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    if (quads.empty())
        return createLevelSetThickenedMesh<GridType, InterruptT>(vertices, triangles, radius,
                                                                 voxelSize, halfWidth, interrupter);

    const Index tn = triangles.size(), qn = quads.size();
    const Index qn2 = tn + qn;
    std::vector<Vec3I> tris(tn + 2*qn);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, tn),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                tris[i] = triangles[i];
            }
        });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, qn),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const Vec4I& q = quads[i];
                tris[i + tn]  = Vec3I(q.x(), q.y(), q.z());
                tris[i + qn2] = Vec3I(q.x(), q.z(), q.w());
            }
        });

    return createLevelSetThickenedMesh<GridType, InterruptT>(vertices, tris, radius,
                                                             voxelSize, halfWidth, interrupter);
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_LEVELSETTHICKENEDMESH
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetThickenedMesh<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec3I>&, float, float, float, util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetThickenedMesh<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec4I>&, float, float, float, util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetThickenedMesh<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec3I>&, const std::vector<Vec4I>&, float, float, float, \
        util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETTHICKENEDMESH_HAS_BEEN_INCLUDED
