// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @author Greg Hurst
///
/// @file LevelSetTubes.h
///
/// @brief Generate a narrow-band level set of a capsule, tapered capsule, and tube complex.
///
/// @note By definition a level set has a fixed narrow band width
/// (the half width is defined by LEVEL_SET_HALF_WIDTH in Types.h),
/// whereas an SDF can have a variable narrow band width.

#ifndef OPENVDB_TOOLS_LEVELSETTUBES_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETTUBES_HAS_BEEN_INCLUDED

#include "ConvexVoxelizer.h"
#include "PointPartitioner.h"
#include "Prune.h"

#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/NullInterrupter.h>

#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a capsule (tube with constant radius and sphere caps).
///
/// @param pt1    First capsule endpoint in world units.
/// @param pt2    Second capsule endpoint in world units.
/// @param radius    Radius of the capsule in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius, float voxelSize,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr, bool threaded = true);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a capsule (tube with constant radius and sphere caps).
///
/// @param pt1    First capsule endpoint in world units.
/// @param pt2    Second capsule endpoint in world units.
/// @param radius    Radius of the capsule in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType>
typename GridType::Ptr
createLevelSetCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius, float voxelSize,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH), bool threaded = true);


/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tapered capsule (tube with sphere caps and different radii at both ends,
/// or equivalently the convex hull of two spheres with possibly different centers and radii).
///
/// @param pt1    First tapered capsule endpoint in world units.
/// @param pt2    Second tapered capsule endpoint in world units.
/// @param radius1    Radius of the tapered capsule at @c pt1 in world units.
/// @param radius2    Radius of the tapered capsule at @c pt2 in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetTaperedCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius1, float radius2,
    float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr, bool threaded = true);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tapered capsule (tube with sphere caps and different radii at both ends,
/// or equivalently the convex hull of two spheres with possibly different centers and radii).
///
/// @param pt1    First tapered capsule endpoint in world units.
/// @param pt2    Second tapered capsule endpoint in world units.
/// @param radius1    Radius of the tapered capsule at @c pt1 in world units.
/// @param radius2    Radius of the tapered capsule at @c pt2 in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType>
typename GridType::Ptr
createLevelSetTaperedCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius1, float radius2,
    float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH), bool threaded = true);

/// @brief Different policies when creating a tube complex with varying radii
/// @details
/// <dl>
/// <dt><b>TUBE_VERTEX_RADII</b>
/// <dd>Specify that the tube complex radii are per-vertex,
/// meaning each tube has different radii at its two endpoints
/// and the complex is a collection of tapered capsules.
///
/// <dt><b>TUBE_SEGMENT_RADII</b>
/// <dd>Specify that the tube complex radii are per-segment,
/// meaning each tube has a constant radius and the complex is a collection of capsules.
///
/// <dt><b>TUBE_AUTOMATIC</b>
/// <dd>Specify that the only valid setting is to be chosen,
/// defaulting to the per-vertex policy if both are valid.
/// </dl>
enum TubeRadiiPolicy { TUBE_AUTOMATIC = 0, TUBE_VERTEX_RADII, TUBE_SEGMENT_RADII };

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tube complex (a collection of capsules defined by endpoint coordinates and segment indices).
///
/// @param vertices    Endpoint vertices in the tube complex in world units.
/// @param segments    Segment indices in the tube complex.
/// @param radius    Radius of all tubes in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetTubeComplex(const std::vector<Vec3s>& vertices, const std::vector<Vec2I>& segments,
    float radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tube complex (a collection of tubes defined by endpoint coordinates, segment indices, and radii).
///
/// @param vertices    Endpoint vertices in the tube complex in world units.
/// @param segments    Segment indices in the tube complex.
/// @param radii    Radii specification for all tubes in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param TubeRadiiPolicy    Policies: per-segment, per-vertex, or automatic (default).
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note The automatic @c TubeRadiiPolicy chooses the valid per-segment or per-vertex policy, defaulting to per-vertex if both are valid.
template <typename GridType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetTubeComplex(const std::vector<Vec3s>& vertices, const std::vector<Vec2I>& segments,
    const std::vector<float>& radii, float voxelSize,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH), TubeRadiiPolicy radii_policy = TUBE_AUTOMATIC,
    InterruptT* interrupter = nullptr);


////////////////////////////////////////


namespace lvlset {

/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of a capsule.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
class CapsuleVoxelizer
    : public ConvexVoxelizer<
          GridType,
          CapsuleVoxelizer<GridType, InterruptT>,
          InterruptT>
{
    using GridPtr = typename GridType::Ptr;
    using ValueT  = typename GridType::ValueType;

    // ------------ base class members ------------

    using BaseT = ConvexVoxelizer<
        GridType,
        CapsuleVoxelizer<GridType, InterruptT>,
        InterruptT
    >;

    using BaseT::mXYData;
    using BaseT::tileCeil;

public:

    friend class ConvexVoxelizer<
        GridType,
        CapsuleVoxelizer<GridType, InterruptT>,
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
    CapsuleVoxelizer(GridPtr& grid, const bool& threaded = true,
                     InterruptT* interrupter = nullptr)
    : BaseT(grid, threaded, interrupter)
    {
    }

    /// @brief Create a capsule
    ///
    /// @param pt1    first endpoint of the capsule in world units
    /// @param pt2    second endpoint of the capsule in world units
    /// @param radius    radius of the capsule in world units
    void
    operator()(const Vec3s& pt1, const Vec3s& pt2, const float& r)
    {
        initialize(pt1, pt2, r);

        BaseT::iterate();
    }

private:

    // setXYRangeData inputs:
    //   step: step size in index space (default is 1)
    //
    // setXYRangeData class member inputs:
    //   mPt1, mPt2, mORad: a tube with points p1, p2 and radius r (padded for halfwidth)
    //
    // setXYRangeData class member outputs:
    //   mXs:  list of x ordinates to scan over
    //   mYbs: list of bottom y ordinates to start scanning with, for each x
    //   mYts: list of top y ordinates to stop scanning at, for each x
    //
    //
    // This routine projects the tube on the xy-plane, giving a stadium shape.
    // It detemines the x-scan range, and the y-range for each x-value.
    // The x-range is divided into intervals depending on if each y endpoint hits a circle or line.
    //
    // The x-range is partitioned by ordinates a0, a1, a2, a3, a4, a5
    //    and based on some cases listed below, some ai will be permuted.
    //
    // The x-range intervals have a few flavors depending on
    //    * if the stadium points right-down or right-up
    //    * the bottom circle ends before the top circle starts, or not
    //
    // 2D projection of capsule onto xy-plane, giving a stadium:
    //
    //   ∧ y                                    ********
    //   |                                   ***        ***
    //   |                                 *               *
    //   |                               / |                *
    //   |                             /   |                *
    //   |                           /     |                 *
    //   |                         /       |       p2        *
    //   |                       /         |      /  \   r   *
    //   |                     /           |    /      \    *|
    //   |                   /             |  /          \  *|
    //   |                 /               |/              * |
    //   |               /                /|             / | |
    //   |             /                /  |           /   | |
    //   |           /                /    |         /     | |
    //   |         *                /      |       /       | |
    //   |        *|              /        |     /         | |
    //   |        *|            /          |   /           | |
    //   |       * |          /            | /             | |
    //   |       * |        p1             |               | |
    //   |       * |                     / |               | |
    //   |       |*|                   /   |               | |
    //   |       |*|                 /     |               | |
    //   |       | |               *       |               | |
    //   |       | |***        *** |       |               | |
    //   |       | |   ********    |       |               | |
    //   |       | |               |       |               | |
    //   |       /  \              |       |               /  \
    //   |     a0    a1            a2      a3            a4    a5
    //   |                                                           x
    //   └----------------------------------------------------------->
    //
    // In this schematic, we have a0 < a1 < a2 < a3 < a4 < a5,
    //    but, for examples, it's possible for a2 and a3 to swap if the stadium is more vertical

    inline void
    setXYRangeData(const Index& step = 1) override
    {
        // short circuit a vertical cylinder
        if (mIsVertical) {
            mXYData.reset(mX1 - mORad, mX1 + mORad, step);

            for (float x = tileCeil(mX1 - mORad, step); x <= mX1 + mORad; x += step)
                mXYData.expandYRange(x, circle1Bottom(x), circle1Top(x));
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
        return mRad >= BaseT::halfWidth() + 0.70711f * (dim-1u);
    }

    // vertical capsule
    // for a given x,y pair, find the z-range of a tube
    //   z-range is bottom sphere cap to the top sphere cap in vertical case
    std::function<bool(float&, float&, const float&, const float&)> capsuleBottomTopVertical =
    [this](float& zb, float& zt, const float& x, const float& y)
    {
        zb = BaseT::sphereBottom(mX1, mY1, math::Min(mZ1, mZ2), mORad, x, y);
        zt = BaseT::sphereTop(mX2, mY2, math::Max(mZ1, mZ2), mORad, x, y);

        return std::isfinite(zb) && std::isfinite(zt);
    };

    // non vertical capsule
    // for a given x,y pair, find the z-range of a tube
    //   first find the z-range as if its an infinite cylinder
    //   then for each z-range endpoint, determine if it should be on a sphere cap
    std::function<bool(float&, float&, const float&, const float&)> capsuleBottomTop =
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

        return std::isfinite(zb) && std::isfinite(zt);
    };

    // assumes capsule is not vertical!
    inline bool
    infiniteCylinderBottomTop(float& cylptb, float& cylptt, const float& x, const float& y) const
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
    inline void initialize(const Vec3s& pt1, const Vec3s& pt2, const float& r)
    {
        const float vx = BaseT::voxelSize(),
                    hw = BaseT::halfWidth();

        if (pt1[0] <= pt2[0]) {
            mPt1 = pt1/vx;
            mPt2 = pt2/vx;
        } else {
            mPt1 = pt2/vx;
            mPt2 = pt1/vx;
        }

        mRad = r/vx;

        // padded radius used to populate the outer halfwidth of the sdf
        mORad  = mRad + hw;
        mORad2 = mORad * mORad;

        mV = mPt2 - mPt1;
        mVLenSqr = mV.lengthSqr();

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

        BaseT::bottomTop = mIsVertical ? capsuleBottomTopVertical : capsuleBottomTop;
    }

    // ------------ private members ------------

    // tube data -- populated via initialize()

    Vec3s mPt1, mPt2, mV;

    Vec2s mPt12d, mPt22d, mV2d;

    float mORad, mORad2, mRad, mVLenSqr, mXdiff, mYdiff, mZdiff, mXYNorm, mXYNorm2;

    float mX1, mY1, mZ1, mX2, mY2, mZ2;

    bool mIsVertical;

}; // class CapsuleVoxelizer


/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of a tapered capsule.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template <typename GridType, typename InterruptT = util::NullInterrupter>
class TaperedCapsuleVoxelizer
    : public ConvexVoxelizer<
          GridType,
          TaperedCapsuleVoxelizer<GridType, InterruptT>,
          InterruptT>
{
    using GridPtr = typename GridType::Ptr;
    using ValueT  = typename GridType::ValueType;

    // ------------ base class members ------------

    using BaseT = ConvexVoxelizer<
        GridType,
        TaperedCapsuleVoxelizer<GridType, InterruptT>,
        InterruptT
    >;

    using BaseT::mXYData;
    using BaseT::tileCeil;

public:

    friend class ConvexVoxelizer<
        GridType,
        TaperedCapsuleVoxelizer<GridType, InterruptT>,
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
    TaperedCapsuleVoxelizer(GridPtr& grid, const bool& threaded = true,
                       InterruptT* interrupter = nullptr)
    : BaseT(grid, threaded, interrupter)
    {
    }

    /// @brief Create a tapered capsule
    ///
    /// @param pt1    first endpoint of the tapered capsule in world units
    /// @param pt2    second endpoint of the tapered capsule in world units
    /// @param radius1    radius of the tapered capsule at @c pt1 in world units
    /// @param radius2    radius of the tapered capsule at @c pt2 in world units
    void
    operator()(const Vec3s& pt1, const Vec3s& pt2, const float& radius1, const float& radius2)
    {
        // fail on degenerate inputs for now

        // ball
        if ((pt1 - pt2).lengthSqr() <= math::Pow2(radius1 - radius2)) {
            OPENVDB_THROW(RuntimeError,
                "The tapered capsule is degenerate, in this case it is a ball. Consider using the CapsuleVoxelizer class instead.");
        }

        // tube
        if (math::Abs(radius1 - radius2) < 0.001f*BaseT::voxelSize()) {
            OPENVDB_THROW(RuntimeError,
                "The tapered capsule is degenerate, in this case it is a capsule. Consider using the CapsuleVoxelizer class instead.");
        }

        initialize(pt1, pt2, radius1, radius2);

        BaseT::iterate();
    }

private:

    inline void
    setXYRangeData(const Index& step = 1) override
    {
        // short circuit when one circle is in the other
        if (mXYNorm2 <= mRdiff2) {
            if (mX1 - mORad1 <= mX2 - mORad2) {
                mXYData.reset(mX1 - mORad1, mX1 + mORad1, step);
                for (float x = tileCeil(mX1 - mORad1, step); x <= mX1 + mORad1; x += step)
                    mXYData.expandYRange(x, circle1Bottom(x), circle1Top(x));
            } else {
                mXYData.reset(mX2 - mORad2, mX2 + mORad2, step);
                for (float x = tileCeil(mX2 - mORad2, step); x <= mX2 + mORad2; x += step)
                    mXYData.expandYRange(x, circle2Bottom(x), circle2Top(x));
            }
            return;
        }

        mXYData.reset(
            math::Min(mX1 - mORad1, mX2 - mORad2),
            math::Max(mX1 + mORad1, mX2 + mORad2),
            step
        );

        Vec2s p1t, p2t, p1b, p2b;
        const bool success = pullyPoints(p1t, p2t, p1b, p2b);

        if (success) {
            setLineXYData(p1t, p2t, step);
            setLineXYData(p1b, p2b, step);

            setCircleXYData(p1t, p1b, step, true);  // mPt1
            setCircleXYData(p2t, p2b, step, false); // mPt2
        }

        mXYData.trim();
    }

    // https://en.wikipedia.org/wiki/Belt_problem#Pulley_problem
    inline bool
    pullyPoints(Vec2s& p1t, Vec2s& p2t, Vec2s& p1b, Vec2s& p2b) const
    {
        const float diff = mXYNorm2 - mRdiff2;
        if (diff < 0.0f)
            return false;

        const float alpha = std::atan2(mYdiff, mXdiff),
                    theta = std::atan2(math::Sqrt(diff), mRdiff);

        const float sin1 = math::Sin(theta + alpha), sin2 = math::Sin(theta - alpha),
                    cos1 = math::Cos(theta + alpha), cos2 = math::Cos(theta - alpha);

        p1t.x() = mX1 + mORad1*cos1; p1t.y() = mY1 + mORad1*sin1;
        p2t.x() = mX2 + mORad2*cos1; p2t.y() = mY2 + mORad2*sin1;
        p2b.x() = mX2 + mORad2*cos2; p2b.y() = mY2 - mORad2*sin2;
        p1b.x() = mX1 + mORad1*cos2; p1b.y() = mY1 - mORad1*sin2;

        return true;
    }

    inline void
    setLineXYData(const Vec2s& q1, const Vec2s& q2, const float& step)
    {
        if (math::Abs(q1.x() - q2.x()) < math::Tolerance<float>::value()) {
            float x = tileCeil(q1.x(), step);
            if (q1.x() == x) {
                mXYData.expandYRange(x, q1.y());
                mXYData.expandYRange(x, q2.y());
            }
        } else {
            const bool q1_left = q1.x() < q2.x();
            const float &x1 = q1_left ? q1.x() : q2.x(),
                        &y1 = q1_left ? q1.y() : q2.y(),
                        &x2 = q1_left ? q2.x() : q1.x(),
                        &y2 = q1_left ? q2.y() : q1.y();

            float m = (y2 - y1)/(x2 - x1),
                  x = tileCeil(x1, step),
                  y = y1 + m * (x-x1),
                  delta = m * step;
            for (; x <= x2; x += step, y += delta)
                mXYData.expandYRange(x, y);
        }
    }

    inline void
    setCircleXYData(const Vec2s& q1, const Vec2s& q2,
        const float& step, const bool is_pt1)
    {
        const Vec3s &p1 = is_pt1 ? mPt1 : mPt2;
        const float &r1 = is_pt1 ? mORad1 : mORad2;

        const std::vector<float> xs = {
            tileCeil(p1.x() - r1, step),
            tileCeil(math::Min(q1.x(), q2.x()), step),
            tileCeil(math::Max(q1.x(), q2.x()), step),
            tileCeil(p1.x() + r1, step)
        };

        for (int i = 0; i < xs.size()-1; ++i) {
            setCircleHiXYData(xs[i], xs[i+1], step, is_pt1);
            setCircleLoXYData(xs[i], xs[i+1], step, is_pt1);
        }
    }

    inline void
    setCircleHiXYData(const float& x1, const float& x2,
        const float& step, const bool& is_pt1)
    {
        const float x_test = static_cast<float>(math::Floor(0.5f*(x1+x2)));

        if (is_pt1) {
            // if |x2-x1| is small, our test point might be too close to the pulley point
            if (math::Abs(x2-x1) < 5 || mXYData.getYMax(x_test) <= circle1Top(x_test)) {
                for (float x = x1; x < x2; x += step)
                    mXYData.expandYMax(x, circle1Top(x));
            }
        } else {
            if (math::Abs(x2-x1) < 5 || mXYData.getYMax(x_test) <= circle2Top(x_test)) {
                for (float x = x1; x < x2; x += step)
                    mXYData.expandYMax(x, circle2Top(x));
            }
        }
    }

    inline void
    setCircleLoXYData(const float& x1, const float& x2,
        const float& step, const bool& is_pt1)
    {
        const float x_test = static_cast<float>(math::Floor(0.5f*(x1+x2)));

        if (is_pt1) {
            // if |x2-x1| is small, our test point might be too close to the pulley point
            if (math::Abs(x2-x1) < 5 || mXYData.getYMin(x_test) >= circle1Bottom(x_test)) {
                for (float x = x1; x < x2; x += step)
                    mXYData.expandYMin(x, circle1Bottom(x));
            }
        } else {
            if (math::Abs(x2-x1) < 5 || mXYData.getYMin(x_test) >= circle2Bottom(x_test)) {
                for (float x = x1; x < x2; x += step)
                    mXYData.expandYMin(x, circle2Bottom(x));
            }
        }
    }

    // Round Cone: https://iquilezles.org/articles/distfunctions/
    // distance in index space
    inline float
    signedDistance(const Vec3s& p) const
    {
        const Vec3s  w  = p - mPt1;
        const ValueT y  = w.dot(mV),
                     z  = y - mVLenSqr,
                     x2 = (w*mVLenSqr - mV*y).lengthSqr(),
                     y2 = y*y*mVLenSqr,
                     z2 = z*z*mVLenSqr,
                     k  = mRdiff2*x2; // should multiply by sgn(mRdiff), but it's always positive

        if (math::Sign(z)*mA2*z2 >= k)
            return  math::Sqrt(x2 + z2)*mInvVLenSqr - mRad2;

        if (math::Sign(y)*mA2*y2 <= k)
            return  math::Sqrt(x2 + y2)*mInvVLenSqr - mRad1;

        return (math::Sqrt(x2*mA2*mInvVLenSqr) + y*mRdiff)*mInvVLenSqr - mRad1;
    }

    inline bool
    tileCanFit(const Index& dim) const override
    {
        return math::Max(mRad1, mRad2) >= BaseT::halfWidth() + 0.70711f * (dim-1u);
    }

    std::function<bool(float&, float&, const float&, const float&)> TaperedCapsuleBottomTop =
    [this](float& zb, float& zt, const float& x, const float& y)
    {
        const Vec2s q(x, y);

        const bool in_ball1 = (q - mPt12d).lengthSqr() <= mORad1Sqr,
                   in_ball2 = (q - mPt22d).lengthSqr() <= mORad2Sqr;

        if (in_ball1) {
            zt = sphere1Top(x, y);
            zb = 2.0f*mZ1 - zt;
        }

        if (in_ball2) {
            if (in_ball1) {
                const float zt2 = sphere2Top(x, y),
                            zb2 = 2.0f*mZ2 - zt2;

                zt = math::Max(zt, zt2);
                zb = math::Min(zb, zb2);
            } else {
                zt = sphere2Top(x, y);
                zb = 2.0f*mZ2 - zt;
            }
        }

        // attempt to short circuit when top and bottom hits are on sphere caps
        if (in_ball1 || in_ball2) {
            const float ht = mConeD.dot(Vec3s(x,y,zt) - mConeV);
            // top point is in one of the half spaces pointing away from the cone
            if (mH1 > ht || ht > mH2) {
                const float hb = mConeD.dot(Vec3s(x,y,zb) - mConeV);
                // bottom point is in one of the half spaces pointing away from the cone
                if (mH1 > hb || hb > mH2)
                    return true;
            }
        }

        float conezb, conezt;
        int cint_cnt;
        openConeFrustumBottomTop(conezb, conezt, cint_cnt, x, y);

        if (in_ball1 && in_ball2) {
            if (cint_cnt == 2) {
                zb = math::Min(zb, conezb);
                zt = math::Max(zt, conezt);
            } else if (cint_cnt == 1) {
                zb = math::Min(zb, conezb);
                zt = math::Max(zt, conezb);
            }

            // cint_cnt == 0 implies zb and zt are already set correctly
            return true;
        }

        if (cint_cnt == 2 || (!in_ball1 && !in_ball2)) {
            zb = conezb; zt = conezt;
            return cint_cnt == 2;
        }

        // we know at this point that in_ball1 ^ in_ball2

        // zt and zb have been assigned values already based on the ball their in
        if (cint_cnt == 0)
            return true;

        // cint_cnt == 1 and we're only in one ball

        zt = math::Max(zt, conezb);
        zb = math::Min(zb, conezb);

        return true;
    };

    // https://www.geometrictools.com/Documentation/IntersectionLineCone.pdf
    inline void
    openConeFrustumBottomTop(float& conezb, float& conezt, int& cint_cnt,
        const float& x, const float& y) const
    {
        cint_cnt = 0;
        const Vec3d p(x, y, mRayZ);
        const Vec3d diff = p - mConeV;

        const double ddotdiff = mConeD.dot(diff);

        const double c1 = mGamma * diff.z() - mConeD.z() * ddotdiff;
        const double c0 = ddotdiff * ddotdiff - mGamma * diff.lengthSqr();

        if (mC2 != 0.0f) {
            const double delta = c1*c1 - c0*mC2;
            if (delta >= 0.0f) {
                const double sqrt = math::Sqrt(delta);
                const double t1 = mC2Inv*(-c1 + sqrt);
                if (validFrustumRange(t1, ddotdiff)) {
                    cint_cnt++;
                    conezb = mRayZ - t1;
                }
                const double t2 = mC2Inv*(-c1 - sqrt);
                if (validFrustumRange(t2, ddotdiff)) {
                    cint_cnt++;
                    if (cint_cnt == 2 && t1 > t2)
                        conezt = mRayZ - t2;
                    else {
                        conezt = conezb;
                        conezb = mRayZ - t2;
                    }
                }
            }
        } else if (c1 != 0.0f) {
            const double t = -c0/(2.0f*c1);
            if (validFrustumRange(t, ddotdiff)) {
                cint_cnt = 1;
                conezb = mRayZ - t;
            }
        }

        // ignore the c2 == c1 == 0 case, where the ray is on the boundary of the cone
    }

    inline bool
    validFrustumRange(const double& t, const double& ddotdiff) const
    {
        const double h = ddotdiff - t * mConeD.z();

        return mH1 <= h && h <= mH2;
    }

    inline float
    circle1Bottom(const float& x) const
    {
        return BaseT::circleBottom(mX1, mY1, mORad1, x);
    }

    inline float
    circle1Top(const float& x) const
    {
        return BaseT::circleTop(mX1, mY1, mORad1, x);
    }

    inline float
    circle2Bottom(const float& x) const
    {
        return BaseT::circleBottom(mX2, mY2, mORad2, x);
    }

    inline float
    circle2Top(const float& x) const
    {
        return BaseT::circleTop(mX2, mY2, mORad2, x);
    }

    inline float
    sphere1Bottom(const float& x, const float& y) const
    {
        return BaseT::sphereBottom(mX1, mY1, mZ1, mORad1, x, y);
    }

    inline float
    sphere1Top(const float& x, const float& y) const
    {
        return BaseT::sphereTop(mX1, mY1, mZ1, mORad1, x, y);
    }

    inline float
    sphere2Bottom(const float& x, const float& y) const
    {
        return BaseT::sphereBottom(mX2, mY2, mZ2, mORad2, x, y);
    }

    inline float
    sphere2Top(const float& x, const float& y) const
    {
        return BaseT::sphereTop(mX2, mY2, mZ2, mORad2, x, y);
    }

    // world space points and radius inputs
    // initializes class members in index space
    inline void
    initialize(const Vec3s& pt1, const Vec3s& pt2, const float& r1, const float& r2)
    {
        const float vx = BaseT::voxelSize(),
                    hw = BaseT::halfWidth();

        // enforce mRad1 > mRad2
        if (r2 <= r1) {
            mPt1 = pt1/vx;
            mPt2 = pt2/vx;
            mRad1 = r1/vx;
            mRad2 = r2/vx;
        } else {
            mPt1 = pt2/vx;
            mPt2 = pt1/vx;
            mRad1 = r2/vx;
            mRad2 = r1/vx;
        }

        // padded radii used to populate the outer halfwidth of the sdf
        mORad1 = mRad1 + hw;
        mORad2 = mRad2 + hw;
        mORad1Sqr = mORad1 * mORad1;
        mORad2Sqr = mORad2 * mORad2;

        mV = mPt2 - mPt1;
        mVLenSqr = mV.lengthSqr();
        mInvVLenSqr = mVLenSqr != 0.0f ? 1.0f/mVLenSqr : 1.0f;

        mX1 = mPt1[0]; mY1 = mPt1[1]; mZ1 = mPt1[2];
        mX2 = mPt2[0]; mY2 = mPt2[1]; mZ2 = mPt2[2];

        mXdiff = mX2 - mX1;
        mYdiff = mY2 - mY1;
        mZdiff = mZ2 - mZ1;

        mPt12d = Vec2s(mX1, mY1);
        mPt22d = Vec2s(mX2, mY2);
        mV2d   = mPt22d - mPt12d;

        mXYNorm2  = math::Pow2(mXdiff) + math::Pow2(mYdiff);
        mXYNorm   = math::Sqrt(mXYNorm2);
        mIXYNorm2 = mXYNorm2 != 0.0f ? 1.0f/mXYNorm2 : 1.0f;

        // mRdiff is non negative
        mRdiff  = mRad1 - mRad2;
        mRdiff2 = mRdiff * mRdiff;
        mA2     = mVLenSqr - mRdiff2;

        // we assume
        //   alpha is solid angle of cone
        //   r1 != r2, since the object is not a capsule
        //   P > abs(r1-r2), since one ball is not contained in the other
        const double P = mV.length(),
                     csc = P/mRdiff,  // csc(alpha/2)
                     sin = mRdiff/P;  // sin(alpha/2)
        mGamma = 1.0 - mRdiff2/(P*P); // cos(alpha/2)^2
        mH1 = mORad2*(csc-sin);
        mH2 = mORad1*(csc-sin);

        mConeD = -((Vec3d)mV).unitSafe();
        mConeV = (Vec3d)mPt1 - (double)mORad1 * csc * mConeD;

        mRayZ  = math::Max(mZ1 + mORad1, mZ2 + mORad2) + 2.0;
        mC2 = math::Pow2(mConeD.z()) - mGamma;
        mC2Inv = mC2 != 0.0 ? 1.0/mC2 : 1.0;

        BaseT::bottomTop = TaperedCapsuleBottomTop;
    }

    // ------------ private members ------------

    // tapered capsule data -- populated via initialize()

    Vec3s mPt1, mPt2, mV;

    Vec3d mConeV, mConeD;

    Vec2s mPt12d, mPt22d, mV2d;

    float mORad1, mORad2, mORad1Sqr, mORad2Sqr, mRad1, mRad2, mVLenSqr, mInvVLenSqr,
          mXdiff, mYdiff, mZdiff, mXYNorm, mXYNorm2, mIXYNorm2, mRdiff, mRdiff2, mA2;

    double mRayZ, mGamma, mC2, mC2Inv, mH1, mH2;

    float mX1, mY1, mZ1, mX2, mY2, mZ2;

}; // class TaperedCapsuleVoxelizer


/// @brief Class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of a tube complex.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note Setting @c PerSegmentRadii to @c true gives a complex of capsules and a complex of tapered capsules otherwise.
template <typename GridType, typename InterruptT = util::NullInterrupter, bool PerSegmentRadii = true>
class TubeComplexVoxelizer {

    using GridPtr = typename GridType::Ptr;
    using TreeT = typename GridType::TreeType;
    using LeafT = typename TreeT::LeafNodeType;

    using PartitionerT = tools::PointPartitioner<Index32, LeafT::LOG2DIM>;

public:

    /// @brief Constructor for constant radius
    ///
    /// @param vertices    endpoint vertices in the tube complex in world units
    /// @param segments    segment indices in the tube complex
    /// @param radius    radius of all tubes in world units
    /// @param voxelSize    voxel size in world units
    /// @param background    background value in voxel units
    /// @param interrupter pointer to optional interrupter. Use template
    /// argument util::NullInterrupter if no interruption is desired.
    TubeComplexVoxelizer(const std::vector<Vec3s>& vertices, const std::vector<Vec2I>& segments,
                         float radius, float voxelSize, float background, InterruptT* interrupter)
    : mVox(voxelSize), mBg(background)
    , mCoords(vertices), mCells(segments)
    , mRad(radius), mRadii(mEmptyVector)
    , mInterrupter(interrupter)
    {
        initializeGrid();
        initializePartitioner();
    }

    /// @brief Constructor for varying radii
    ///
    /// @param vertices    endpoint vertices in the tube complex in world units
    /// @param segments    segment indices in the tube complex
    /// @param radii    radii specification for all tubes in world units
    /// @param voxelSize    voxel size in world units
    /// @param background    background value in voxel units
    /// @param interrupter    pointer to optional interrupter. Use template
    /// argument util::NullInterrupter if no interruption is desired.
    ///
    /// @note If @c PerSegmentRadii is set to @c true then @c segments and @c radii must have
    /// the same size. If @c PerSegmentRadii is set to @c false then @c vertices and @c radii
    /// must have the same size.
    TubeComplexVoxelizer(const std::vector<Vec3s>& vertices, const std::vector<Vec2I>& segments,
                         const std::vector<float>& radii, float voxelSize, float background,
                         InterruptT* interrupter)
    : mVox(voxelSize), mBg(background)
    , mCoords(vertices), mCells(segments)
    , mRadii(radii), mRad(0.0)
    , mInterrupter(interrupter)
    {
        if constexpr (PerSegmentRadii) {
            if (mCells.size() != mRadii.size())
                OPENVDB_THROW(RuntimeError,
                    "TubeComplexVoxelizer needs the same number of segments and radii");
        } else {
            if (mCoords.size() != mRadii.size())
                OPENVDB_THROW(RuntimeError,
                    "TubeComplexVoxelizer needs the same number of coordinates and radii");
        }

        initializeGrid();
        initializePartitioner();
    }

    TubeComplexVoxelizer(TubeComplexVoxelizer& other, tbb::split)
    : mVox(other.mVox), mBg(other.mBg)
    , mCoords(other.mCoords), mCells(other.mCells)
    , mRadii(other.mRadii), mRad(other.mRad)
    , mPtPartitioner(other.mPtPartitioner), mInterrupter(other.mInterrupter)
    {
        initializeGrid();
    }

    template<bool PSR = PerSegmentRadii>
    inline typename std::enable_if_t<PSR, void>
    operator()(const tbb::blocked_range<size_t>& rng)
    {
        if (!checkInterrupter())
            return;

        if (mRadii.size() == 0)
            constantRadiusVoxelize(rng);
        else
            perSegmentRadiusVoxelize(rng);
    }

    template<bool PSR = PerSegmentRadii>
    inline typename std::enable_if_t<!PSR, void>
    operator()(const tbb::blocked_range<size_t>& rng)
    {
        if (!checkInterrupter())
            return;

        if (mRadii.size() == 0)
            constantRadiusVoxelize(rng);
        else
            perVertexRadiusVoxelize(rng);
    }

    void join(TubeComplexVoxelizer& other)
    {
        tools::CsgUnionOp<TreeT> op(other.mGrid->tree(), Steal());
        tree::DynamicNodeManager<TreeT> nodeManager(mGrid->tree());
        nodeManager.foreachTopDown(op, true);

        other.mGrid = nullptr;
    }

    inline Index bucketSize() const { return mPtPartitioner->size(); }

    inline GridPtr getGrid() const { return mGrid; }

private:

    /// TODO increase performance by not creating parts of caps that overlap with other tubes:
    ///
    /// * Determine segment adjacency
    /// * Create _open_ cylinders and conical frustums
    /// * Create _open_ & _partial_ caps
    ///
    /// This should help speed up creation of complexes that contain
    /// a bunch of short segments that approximate a smooth curve.
    ///
    /// Idea is similar to _open_ prisms and _open_ wedges speeding up
    /// creation of thickened mesh level sets from finely triangulated meshes.

    inline void
    constantRadiusVoxelize(const tbb::blocked_range<size_t>& rng)
    {
        CapsuleVoxelizer<GridType, InterruptT> voxelizer(mGrid, false);

        for (size_t i = rng.begin(); i < rng.end(); ++i) {
            for (auto it = mPtPartitioner->indices(i); it; ++it) {
                const Index k = *it;
                const Vec2I& cell = mCells[k];
                voxelizer(mCoords[cell[0]], mCoords[cell[1]], mRad);
            }
        }
    }

    inline void
    perSegmentRadiusVoxelize(const tbb::blocked_range<size_t>& rng)
    {
        CapsuleVoxelizer<GridType, InterruptT> voxelizer(mGrid, false);

        for (size_t i = rng.begin(); i < rng.end(); ++i) {
            for (auto it = mPtPartitioner->indices(i); it; ++it) {
                const Index k = *it;
                const Vec2I& cell = mCells[k];
                voxelizer(mCoords[cell[0]], mCoords[cell[1]], mRadii[k]);
            }
        }
    }

    inline void
    perVertexRadiusVoxelize(const tbb::blocked_range<size_t>& rng)
    {
        TaperedCapsuleVoxelizer<GridType, InterruptT> rc_voxelizer(mGrid, false);

        CapsuleVoxelizer<GridType, InterruptT> c_voxelizer(mGrid, false);

        for (size_t i = rng.begin(); i < rng.end(); ++i) {
            for (auto it = mPtPartitioner->indices(i); it; ++it) {
                const Index k = *it;
                const Vec2I& cell = mCells[k];
                const Index32 &i = cell.x(), &j = cell.y();

                const Vec3s &pt1 = mCoords[i], &pt2 = mCoords[j];
                const float &r1 = mRadii[i], &r2 = mRadii[j];

                if ((pt1 - pt2).lengthSqr() <= math::Pow2(r1-r2)) { // ball
                    if (r1 >= r2)
                        c_voxelizer(pt1, pt1, r1);
                    else
                        c_voxelizer(pt2, pt2, r2);
                } else if (math::Abs(r1-r2) < 0.001f*mVox) { // tube
                    c_voxelizer(pt1, pt2, r1);
                } else {
                    rc_voxelizer(pt1, pt2, r1, r2);
                }
            }
        }
    }

    inline void
    computeCentroids(std::vector<Vec3s>& centroids)
    {
        const Index n = mCoords.size();

        centroids.resize(mCells.size());

        tbb::parallel_for(tbb::blocked_range<size_t>(0, centroids.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const Vec2I &cell = mCells[i];

                    if (cell[0] >= n || cell[1] >= n)
                        OPENVDB_THROW(ValueError, "out of bounds index");

                    centroids[i] = 0.5f * (mCoords[cell[0]] + mCoords[cell[1]]);
                }
            });
    }

    inline void
    initializeGrid()
    {
        math::Transform transform(*(math::Transform::createLinearTransform(mVox)));
        mGrid = GridPtr(new GridType(mBg));
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

    const std::vector<Vec3s>& mCoords;
    const std::vector<Vec2I>& mCells;
    const std::vector<float>& mRadii;

    inline static const std::vector<float> mEmptyVector = {};

    const float mVox, mBg, mRad;

    std::shared_ptr<PartitionerT> mPtPartitioner;

    GridPtr mGrid;

    InterruptT* mInterrupter;

}; // class TubeComplexVoxelizer

} // namespace lvlset


// ------------ createLevelSetTubeComplex ------------- //

// constant radius

template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetTubeComplex(const std::vector<Vec3s>& vertices, const std::vector<Vec2I>& segments,
    float radius, float voxelSize, float halfWidth, InterruptT* interrupter)
{
    using GridPtr = typename GridType::Ptr;
    using TreeT = typename GridType::TreeType;
    using ValueT = typename GridType::ValueType;

    using ComplexVoxelizer = typename lvlset::TubeComplexVoxelizer<GridType, InterruptT>;

    static_assert(std::is_floating_point<ValueT>::value,
        "createLevelSetTubeComplex must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    const float background = voxelSize * halfWidth;
    ComplexVoxelizer op(vertices, segments, radius, voxelSize, background, interrupter);

    const tbb::blocked_range<size_t> segmentRange(0, op.bucketSize());
    tbb::parallel_reduce(segmentRange, op);

    GridPtr tubegrid = op.getGrid();
    tools::pruneLevelSet(tubegrid->tree());

    return tubegrid;
}

// varying radii

template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetTubeComplex(const std::vector<Vec3s>& vertices, const std::vector<Vec2I>& segments,
    const std::vector<float>& radii, float voxelSize, float halfWidth,
    TubeRadiiPolicy radii_policy, InterruptT* interrupter)
{
    using GridPtr = typename GridType::Ptr;
    using ValueT = typename GridType::ValueType;

    using CapsuleComplexVoxelizer = typename lvlset::TubeComplexVoxelizer<GridType, InterruptT, true>;
    using TaperedCapsuleComplexVoxelizer = typename lvlset::TubeComplexVoxelizer<GridType, InterruptT, false>;

    static_assert(std::is_floating_point<ValueT>::value,
        "createLevelSetTubeComplex must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    switch(radii_policy) {
        case TUBE_AUTOMATIC : {
            if (vertices.size() != radii.size() && segments.size() != radii.size())
                OPENVDB_THROW(ValueError,
                    "createLevelSetTubeComplex requires equal number of vertices and radii, or segments and radii, with automatic radii policy.");
            break;
        }
        case TUBE_VERTEX_RADII : {
            if (vertices.size() != radii.size())
                OPENVDB_THROW(ValueError,
                    "createLevelSetTubeComplex requires equal number of vertices and radii with per-vertex radii policy.");
            break;
        }
        case TUBE_SEGMENT_RADII : {
            if (segments.size() != radii.size())
                OPENVDB_THROW(ValueError,
                    "createLevelSetTubeComplex requires equal number of segments and radii with per-segment radii policy.");
            break;
        }
        default:
            OPENVDB_THROW(ValueError, "Invalid tube radii policy.");
    }

    const float background = voxelSize * halfWidth;
    GridPtr tubegrid;

    if (vertices.size() == radii.size()) {
        TaperedCapsuleComplexVoxelizer op(vertices, segments, radii,
                                     voxelSize, background, interrupter);

        const tbb::blocked_range<size_t> segmentRange(0, op.bucketSize());
        tbb::parallel_reduce(segmentRange, op);

        tubegrid = op.getGrid();
    } else {
        CapsuleComplexVoxelizer op(vertices, segments, radii,
                                   voxelSize, background, interrupter);

        const tbb::blocked_range<size_t> segmentRange(0, op.bucketSize());
        tbb::parallel_reduce(segmentRange, op);

        tubegrid = op.getGrid();
    }

    tools::pruneLevelSet(tubegrid->tree());

    return tubegrid;
}


// ------------ createLevelSetCapsule ------------- //

template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius, float voxelSize,
    float halfWidth, InterruptT* interrupter, bool threaded)
{
    using GridPtr = typename GridType::Ptr;
    using ValueT = typename GridType::ValueType;

    using CapsuleVoxelizer = typename lvlset::CapsuleVoxelizer<GridType, InterruptT>;

    static_assert(std::is_floating_point<ValueT>::value,
        "createLevelSetCapsule must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    const ValueT background = voxelSize * halfWidth;
    math::Transform transform(*(math::Transform::createLinearTransform(voxelSize)));

    GridPtr grid(new GridType(background));
    grid->setTransform(transform.copy());
    grid->setGridClass(GRID_LEVEL_SET);

    CapsuleVoxelizer voxelizer(grid, threaded, interrupter);
    voxelizer(pt1, pt2, radius);

    return grid;
}

template <typename GridType>
typename GridType::Ptr
createLevelSetCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius, float voxelSize,
    float halfWidth, bool threaded)
{
    return createLevelSetCapsule<GridType, util::NullInterrupter>(
        pt1, pt2, radius, voxelSize, halfWidth, nullptr, threaded);
}


// ------------ createLevelSetTaperedCapsule ------------- //

template <typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetTaperedCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius1, float radius2,
    float voxelSize, float halfWidth, InterruptT* interrupter, bool threaded)
{
    using GridPtr = typename GridType::Ptr;
    using ValueT = typename GridType::ValueType;

    using CapsuleVoxelizer = typename lvlset::CapsuleVoxelizer<GridType, InterruptT>;
    using TaperedCapsuleVoxelizer = typename lvlset::TaperedCapsuleVoxelizer<GridType, InterruptT>;

    static_assert(std::is_floating_point<ValueT>::value,
        "createLevelSetTaperedCapsule must return a scalar grid");

    if (voxelSize <= 0) OPENVDB_THROW(ValueError, "voxel size must be positive");
    if (halfWidth <= 0) OPENVDB_THROW(ValueError, "half-width must be positive");

    const ValueT background = voxelSize * halfWidth;
    math::Transform transform(*(math::Transform::createLinearTransform(voxelSize)));

    GridPtr grid(new GridType(background));
    grid->setTransform(transform.copy());
    grid->setGridClass(GRID_LEVEL_SET);

    if ((pt1 - pt2).lengthSqr() <= math::Pow2(radius1 - radius2)) { // ball

        CapsuleVoxelizer voxelizer(grid, threaded, interrupter);
        if (radius1 >= radius2)
            voxelizer(pt1, pt1, radius1);
        else
            voxelizer(pt2, pt2, radius2);

    } else if (math::Abs(radius1 - radius2) < 0.001f*voxelSize) { // tube

        CapsuleVoxelizer voxelizer(grid, threaded, interrupter);
        voxelizer(pt1, pt2, radius1);

    } else { // tapered capsule

        TaperedCapsuleVoxelizer voxelizer(grid, threaded, interrupter);
        voxelizer(pt1, pt2, radius1, radius2);
    }

    return grid;
}

template <typename GridType>
typename GridType::Ptr
createLevelSetTaperedCapsule(const Vec3s& pt1, const Vec3s& pt2, float radius1, float radius2,
    float voxelSize, float halfWidth, bool threaded)
{
    return createLevelSetTaperedCapsule<GridType, util::NullInterrupter>(
        pt1, pt2, radius1, radius2, voxelSize, halfWidth, nullptr, threaded);
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_LEVELSETTUBES
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetTubeComplex<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec2I>&, float, float, float, util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetTubeComplex<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec2I>&, const std::vector<float>&, float, float, TubeRadiiPolicy, \
        util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetCapsule<Grid<TreeT>>(const Vec3s&, const Vec3s&, \
        float, float, float, util::NullInterrupter*, bool)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetTaperedCapsule<Grid<TreeT>>(const Vec3s&, const Vec3s&, \
        float, float, float, float, util::NullInterrupter*, bool)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETTUBES_HAS_BEEN_INCLUDED
