// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
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

/// @brief  Generate the 8 vertices and 6 quad faces of an axis-aligned box
///         centered at @a center with half-extents @a halfSize, then rotate
///         by Euler angles @a rotDeg (X, Y, Z order) in degrees.
///
/// @param center    World-space center of the box.
/// @param halfSize  Half-extents along each axis before rotation.
/// @param rotDeg    Euler rotation angles (X, Y, Z) in degrees.
/// @param points    Output vertex list (index-space, divided by voxelSize).
/// @param quads     Output quad-index list.
/// @param voxelSize Voxel size used to convert world coords to index space.
void makeRotatedBox(
    const openvdb::Vec3s& center,
    const openvdb::Vec3s& halfSize,
    const openvdb::Vec3s& rotDeg,
    float voxelSize,
    std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec4I>& quads)
{
    using openvdb::Vec3s;

    Vec3s corners[8] = {
        Vec3s(-halfSize[0], -halfSize[1], -halfSize[2]),
        Vec3s( halfSize[0], -halfSize[1], -halfSize[2]),
        Vec3s( halfSize[0],  halfSize[1], -halfSize[2]),
        Vec3s(-halfSize[0],  halfSize[1], -halfSize[2]),
        Vec3s(-halfSize[0], -halfSize[1],  halfSize[2]),
        Vec3s( halfSize[0], -halfSize[1],  halfSize[2]),
        Vec3s( halfSize[0],  halfSize[1],  halfSize[2]),
        Vec3s(-halfSize[0],  halfSize[1],  halfSize[2]),
    };

    const float rx = rotDeg[0] * float(M_PI) / 180.0f;
    const float ry = rotDeg[1] * float(M_PI) / 180.0f;
    const float rz = rotDeg[2] * float(M_PI) / 180.0f;

    const float cx = std::cos(rx), sx = std::sin(rx);
    const float cy = std::cos(ry), sy = std::sin(ry);
    const float cz = std::cos(rz), sz = std::sin(rz);

    // Combined rotation matrix R = Rz * Ry * Rx
    auto rotate = [&](const Vec3s& v) -> Vec3s {
        // Rx
        const float x1 = v[0];
        const float y1 = cx * v[1] - sx * v[2];
        const float z1 = sx * v[1] + cx * v[2];
        // Ry
        const float x2 =  cy * x1 + sy * z1;
        const float y2 =  y1;
        const float z2 = -sy * x1 + cy * z1;
        // Rz
        const float x3 = cz * x2 - sz * y2;
        const float y3 = sz * x2 + cz * y2;
        const float z3 = z2;
        return Vec3s(x3, y3, z3);
    };

    points.clear();
    points.reserve(8);
    for (int i = 0; i < 8; ++i) {
        Vec3s r = rotate(corners[i]);
        points.emplace_back(
            (r[0] + center[0]) / voxelSize,
            (r[1] + center[1]) / voxelSize,
            (r[2] + center[2]) / voxelSize);
    }

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


TEST_F(TestBlend, testBlendTwoBoxes)
{
    using namespace openvdb;

    const float voxelSize = 0.02f;
    const float exteriorBand = 6.0f;
    const float interiorBand = 3.0f;

    const float alpha = 2.0f;   // band radius
    const float beta  = 80.0f;  // exponent
    const float gamma = 1.0f;   // multiplier

    math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);

    // Box A: half-extents (0.5, 0.5, 0.5) rotated (45, 0, 45) degrees
    std::vector<Vec3s> ptsA;
    std::vector<Vec4I> quadsA;
    makeRotatedBox(
        Vec3s(0.0f), Vec3s(0.5f, 0.5f, 0.5f), Vec3s(45.0f, 0.0f, 45.0f),
        voxelSize, ptsA, quadsA);
    tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> meshA(ptsA, quadsA);
    FloatGrid::Ptr gridA = tools::meshToVolume<FloatGrid>(
        meshA, *xform, exteriorBand, interiorBand);
    ASSERT_TRUE(gridA);
    gridA->setGridClass(GRID_LEVEL_SET);

    // Box B: half-extents (1, 0.2, 1) axis-aligned
    std::vector<Vec3s> ptsB;
    std::vector<Vec4I> quadsB;
    makeRotatedBox(
        Vec3s(0.0f), Vec3s(1.0f, 0.2f, 1.0f), Vec3s(0.0f),
        voxelSize, ptsB, quadsB);
    tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> meshB(ptsB, quadsB);
    FloatGrid::Ptr gridB = tools::meshToVolume<FloatGrid>(
        meshB, *xform, exteriorBand, interiorBand);
    ASSERT_TRUE(gridB);
    gridB->setGridClass(GRID_LEVEL_SET);

    FloatGrid::ConstPtr noMask;
    FloatGrid::Ptr blendResult =
        tools::unionFillet<FloatGrid>(*gridA, *gridB, noMask, alpha, beta, gamma);
    ASSERT_TRUE(blendResult);
    EXPECT_TRUE(blendResult->activeVoxelCount() > 0);

    FloatGrid::Ptr csgResult = tools::csgUnionCopy(*gridA, *gridB);
    ASSERT_TRUE(csgResult);

    // The fillet blend should carve outward (more negative SDF) at
    // concave seam locations between the two boxes.
    {
        auto blendAcc = blendResult->getConstAccessor();
        auto csgAcc   = csgResult->getConstAccessor();

        const Vec3d probes[4] = {
            Vec3d(-0.80, 0.24, 0.0),
            Vec3d(-0.82, 0.22, 0.0),
            Vec3d(-0.84, 0.22, 0.0),
            Vec3d(-0.86, 0.22, 0.0),
        };

        for (int i = 0; i < 4; ++i) {
            const Coord ijk = xform->worldToIndexNodeCentered(probes[i]);
            const float blendVal = blendAcc.getValue(ijk);
            const float csgVal   = csgAcc.getValue(ijk);

            EXPECT_LT(blendVal, csgVal);
        }
    }
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