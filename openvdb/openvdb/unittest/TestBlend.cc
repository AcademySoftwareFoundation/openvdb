// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/Blend.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/MeshToVolume.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>


class TestBlend: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


namespace {

/// @brief  Generate the 8 vertices and 6 quad faces of an axis-aligned cube
///         centered at @a center with the given half @a extent, then rotate
///         every vertex by @a angleDeg degrees around the Z-axis.
///
/// @param center    World-space center of the cube.
/// @param extent    Half-size of the cube (distance from center to each face).
/// @param angleDeg  Rotation angle around the Z-axis, in degrees.
/// @param points    Output vertex list (index-space, divided by voxelSize).
/// @param quads     Output quad-index list.
/// @param voxelSize Voxel size used to convert world coords to index space.
void makeRotatedCube(
    const openvdb::Vec3s& center,
    float extent,
    float angleDeg,
    float voxelSize,
    std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec4I>& quads)
{
    using openvdb::Vec3s;

    const float lo = -extent;
    const float hi =  extent;

    // 8 corners of an axis-aligned cube centered at origin
    Vec3s corners[8] = {
        Vec3s(lo, lo, lo), // 0
        Vec3s(hi, lo, lo), // 1
        Vec3s(hi, hi, lo), // 2
        Vec3s(lo, hi, lo), // 3
        Vec3s(lo, lo, hi), // 4
        Vec3s(hi, lo, hi), // 5
        Vec3s(hi, hi, hi), // 6
        Vec3s(lo, hi, hi), // 7
    };

    // Rotation around Z-axis
    const float rad = angleDeg * float(M_PI) / 180.0f;
    const float c = std::cos(rad);
    const float s = std::sin(rad);

    points.clear();
    points.reserve(8);
    for (int i = 0; i < 8; ++i) {
        const float rx = c * corners[i][0] - s * corners[i][1];
        const float ry = s * corners[i][0] + c * corners[i][1];
        const float rz = corners[i][2];
        // Translate to center, then convert to index space
        points.emplace_back(
            (rx + center[0]) / voxelSize,
            (ry + center[1]) / voxelSize,
            (rz + center[2]) / voxelSize);
    }

    // 6 quad faces (winding consistent with outward normals)
    quads.clear();
    quads.reserve(6);
    quads.emplace_back(openvdb::Vec4I(0, 3, 2, 1)); // -Z face
    quads.emplace_back(openvdb::Vec4I(4, 5, 6, 7)); // +Z face
    quads.emplace_back(openvdb::Vec4I(0, 1, 5, 4)); // -Y face
    quads.emplace_back(openvdb::Vec4I(2, 3, 7, 6)); // +Y face
    quads.emplace_back(openvdb::Vec4I(0, 4, 7, 3)); // -X face
    quads.emplace_back(openvdb::Vec4I(1, 2, 6, 5)); // +X face
}

} // anonymous namespace


////////////////////////////////////////


TEST_F(TestBlend, testUnionFilletRotatedCubes)
{
    using namespace openvdb;

    // --- Parameters ----------------------------------------------------------
    const float voxelSize = 0.1f;
    const float halfWidth = float(LEVEL_SET_HALF_WIDTH); // in voxels
    const float cubeExtent = 1.0f;    // half-size of each cube (world units)

    // Place two cubes so they slightly overlap along the X-axis.
    // Each cube spans 2*cubeExtent = 2.0 in diameter before rotation.
    // After 45-degree rotation a cube's bounding width grows to 2*sqrt(2) ~ 2.83.
    // We offset them so the rotated edges overlap:
    const float separation = 2.0f;    // center-to-center distance along X
    const Vec3s centerA(-separation / 2.0f, 0.0f, 0.0f);
    const Vec3s centerB( separation / 2.0f, 0.0f, 0.0f);
    const float rotAngle = 45.0f;     // degrees around Z

    // Blend parameters (from the Allen et al. paper)
    const float alpha = 3.0f;  // bandwidth falloff
    const float beta  = 2.0f;  // exponent
    const float gamma = 1.0f;  // amplitude/multiplier

    // --- Build level-set cubes -----------------------------------------------
    math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);

    // Cube A
    std::vector<Vec3s> ptsA;
    std::vector<Vec4I> quadsA;
    makeRotatedCube(centerA, cubeExtent, rotAngle, voxelSize, ptsA, quadsA);

    tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> meshA(ptsA, quadsA);
    FloatGrid::Ptr gridA = tools::meshToVolume<FloatGrid>(meshA, *xform, halfWidth, halfWidth);
    ASSERT_TRUE(gridA);
    gridA->setGridClass(GRID_LEVEL_SET);

    // Cube B
    std::vector<Vec3s> ptsB;
    std::vector<Vec4I> quadsB;
    makeRotatedCube(centerB, cubeExtent, rotAngle, voxelSize, ptsB, quadsB);

    tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> meshB(ptsB, quadsB);
    FloatGrid::Ptr gridB = tools::meshToVolume<FloatGrid>(meshB, *xform, halfWidth, halfWidth);
    ASSERT_TRUE(gridB);
    gridB->setGridClass(GRID_LEVEL_SET);

    // Both grids must share the same transform (required by unionFillet)
    ASSERT_EQ(gridA->constTransform(), gridB->constTransform());

    // --- Compute plain CSG union ---------------------------------------------
    FloatGrid::Ptr csgResult = tools::csgUnionCopy(*gridA, *gridB);
    ASSERT_TRUE(csgResult);

    // --- Compute fillet union ------------------------------------------------
    FloatGrid::ConstPtr noMask; // nullptr mask
    FloatGrid::Ptr filletResult =
        tools::unionFillet<FloatGrid>(*gridA, *gridB, noMask, alpha, beta, gamma);
    ASSERT_TRUE(filletResult);
    EXPECT_EQ(int(GRID_LEVEL_SET), int(filletResult->getGridClass()));
    EXPECT_TRUE(filletResult->activeVoxelCount() > 0);

    // --- Verify smoothness near the seam -------------------------------------
    // Near the overlap region (around x = 0) the fillet should carve outward,
    // producing more-negative distance values than the plain union.
    FloatGrid::ConstAccessor csgAcc    = csgResult->getConstAccessor();
    FloatGrid::ConstAccessor filletAcc = filletResult->getConstAccessor();

    int countSmoother = 0;
    int countSampled  = 0;

    // Sample voxels in a slab around x = 0 (the seam)
    const int slabHalf = int(std::ceil(0.5f / voxelSize)); // +/- 0.5 world units
    const int extentIdx = int(std::ceil((cubeExtent + 0.5f) / voxelSize));

    for (int i = -slabHalf; i <= slabHalf; ++i) {
        for (int j = -extentIdx; j <= extentIdx; ++j) {
            for (int k = -extentIdx; k <= extentIdx; ++k) {
                const Coord ijk(i, j, k);
                const float csgVal    = csgAcc.getValue(ijk);
                const float filletVal = filletAcc.getValue(ijk);

                // Only look at voxels that are inside or near the surface for both results
                if (csgVal < 0.0f && csgVal > -2.0f * voxelSize * halfWidth) {
                    ++countSampled;
                    if (filletVal < csgVal) {
                        ++countSmoother;
                    }
                }
            }
        }
    }

    // We expect a meaningful number of voxels to have been sampled, and a
    // significant fraction of those should show the fillet effect.
    EXPECT_GT(countSampled, 0)
        << "No inside-surface voxels found in the seam region.";
    EXPECT_GT(countSmoother, 0)
        << "The fillet union did not produce any smoother (more negative) "
           "values than the plain union near the seam.";

    // At least 10 % of sampled interior voxels near the seam should be affected
    const float ratio = float(countSmoother) / float(countSampled);
    EXPECT_GT(ratio, 0.10f)
        << "Only " << (ratio * 100.0f) << "% of sampled voxels were smoothed; "
           "expected at least 10%.";
}


TEST_F(TestBlend, testUnionFilletMismatchedTransformsThrow)
{
    using namespace openvdb;

    // Two tiny grids with different transforms -- unionFillet must throw.
    math::Transform::Ptr xformA = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr xformB = math::Transform::createLinearTransform(0.2);

    FloatGrid::Ptr gridA = FloatGrid::create(/*background=*/3.0f);
    gridA->setTransform(xformA);
    gridA->setGridClass(GRID_LEVEL_SET);

    FloatGrid::Ptr gridB = FloatGrid::create(/*background=*/3.0f);
    gridB->setTransform(xformB);
    gridB->setGridClass(GRID_LEVEL_SET);

    FloatGrid::ConstPtr noMask;
    EXPECT_THROW(
        tools::unionFillet<FloatGrid>(*gridA, *gridB, noMask, 3.0f, 2.0f, 1.0f),
        std::runtime_error);
}


TEST_F(TestBlend, testUnionFilletWithMask)
{
    using namespace openvdb;

    // --- Parameters ----------------------------------------------------------
    const float voxelSize = 0.1f;
    const float halfWidth = float(LEVEL_SET_HALF_WIDTH);
    const float cubeExtent = 1.0f;
    const float separation = 2.0f;
    const Vec3s centerA(-separation / 2.0f, 0.0f, 0.0f);
    const Vec3s centerB( separation / 2.0f, 0.0f, 0.0f);
    const float rotAngle = 45.0f;

    const float alpha = 3.0f;
    const float beta  = 2.0f;
    const float gamma = 1.0f;

    math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);

    // Build cubes
    std::vector<Vec3s> ptsA, ptsB;
    std::vector<Vec4I> quadsA, quadsB;
    makeRotatedCube(centerA, cubeExtent, rotAngle, voxelSize, ptsA, quadsA);
    makeRotatedCube(centerB, cubeExtent, rotAngle, voxelSize, ptsB, quadsB);

    tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> meshA(ptsA, quadsA);
    FloatGrid::Ptr gridA = tools::meshToVolume<FloatGrid>(meshA, *xform, halfWidth, halfWidth);
    gridA->setGridClass(GRID_LEVEL_SET);

    tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> meshB(ptsB, quadsB);
    FloatGrid::Ptr gridB = tools::meshToVolume<FloatGrid>(meshB, *xform, halfWidth, halfWidth);
    gridB->setGridClass(GRID_LEVEL_SET);

    // Create a uniform mask grid with value 1.0 everywhere (same transform).
    // This should produce the same result as no mask.
    FloatGrid::Ptr maskGrid = FloatGrid::create(/*background=*/1.0f);
    maskGrid->setTransform(xform);

    FloatGrid::ConstPtr mask = maskGrid;
    FloatGrid::Ptr filletWithMask =
        tools::unionFillet<FloatGrid>(*gridA, *gridB, mask, alpha, beta, gamma);
    ASSERT_TRUE(filletWithMask);

    FloatGrid::ConstPtr noMask;
    FloatGrid::Ptr filletNoMask =
        tools::unionFillet<FloatGrid>(*gridA, *gridB, noMask, alpha, beta, gamma);
    ASSERT_TRUE(filletNoMask);

    // With a uniform mask of 1.0 the results should be identical.
    FloatGrid::ConstAccessor accMask   = filletWithMask->getConstAccessor();
    FloatGrid::ConstAccessor accNoMask = filletNoMask->getConstAccessor();

    for (auto iter = filletNoMask->cbeginValueOn(); iter; ++iter) {
        const Coord& ijk = iter.getCoord();
        EXPECT_FLOAT_EQ(accNoMask.getValue(ijk), accMask.getValue(ijk))
            << "Mismatch at coord " << ijk;
    }
}
