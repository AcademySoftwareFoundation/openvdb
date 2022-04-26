// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Maps.h> // for math::NonlinearFrustumMap
#include <openvdb/tools/Clip.h>

#include <gtest/gtest.h>

// See also TestGrid::testClipping()
class TestClip: public ::testing::Test
{
public:
    static const openvdb::CoordBBox kCubeBBox, kInnerBBox;

    TestClip(): mCube{
        []() {
            auto cube = openvdb::FloatGrid{0.0f};
            cube.fill(kCubeBBox, /*value=*/5.0f, /*active=*/true);
            return cube;
        }()}
    {}

    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::initialize(); }

protected:
    void validate(const openvdb::FloatGrid&);

    const openvdb::FloatGrid mCube;
};

const openvdb::CoordBBox
    // The volume to be clipped is a 21 x 21 x 21 solid cube.
    TestClip::kCubeBBox{openvdb::Coord{-10}, openvdb::Coord{10}},
    // The clipping mask is a 1 x 1 x 13 segment extending along the Z axis inside the cube.
    TestClip::kInnerBBox{openvdb::Coord{4, 4, -6}, openvdb::Coord{4, 4, 6}};


////////////////////////////////////////


void
TestClip::validate(const openvdb::FloatGrid& clipped)
{
    using namespace openvdb;

    const CoordBBox bbox = clipped.evalActiveVoxelBoundingBox();
    EXPECT_EQ(kInnerBBox.min().x(), bbox.min().x());
    EXPECT_EQ(kInnerBBox.min().y(), bbox.min().y());
    EXPECT_EQ(kInnerBBox.min().z(), bbox.min().z());
    EXPECT_EQ(kInnerBBox.max().x(), bbox.max().x());
    EXPECT_EQ(kInnerBBox.max().y(), bbox.max().y());
    EXPECT_EQ(kInnerBBox.max().z(), bbox.max().z());
    EXPECT_EQ(6 + 6 + 1, int(clipped.activeVoxelCount()));
    EXPECT_EQ(2, int(clipped.constTree().leafCount()));

    FloatGrid::ConstAccessor acc = clipped.getConstAccessor();
    const float bg = clipped.background();
    Coord xyz;
    int &x = xyz[0], &y = xyz[1], &z = xyz[2];
    for (x = kCubeBBox.min().x(); x <= kCubeBBox.max().x(); ++x) {
        for (y = kCubeBBox.min().y(); y <= kCubeBBox.max().y(); ++y) {
            for (z = kCubeBBox.min().z(); z <= kCubeBBox.max().z(); ++z) {
                if (x == 4 && y == 4 && z >= -6 && z <= 6) {
                    EXPECT_EQ(5.f, acc.getValue(Coord(4, 4, z)));
                } else {
                    EXPECT_EQ(bg, acc.getValue(Coord(x, y, z)));
                }
            }
        }
    }
}


////////////////////////////////////////


// Test clipping against a bounding box.
TEST_F(TestClip, testBBox)
{
    using namespace openvdb;
    BBoxd clipBox(Vec3d(4.0, 4.0, -6.0), Vec3d(4.9, 4.9, 6.0));
    FloatGrid::Ptr clipped = tools::clip(mCube, clipBox);
    validate(*clipped);
}


// Test clipping against a camera frustum.
TEST_F(TestClip, testFrustum)
{
    using namespace openvdb;

    const auto d = double(kCubeBBox.max().z());
    const math::NonlinearFrustumMap frustum{
        /*position=*/Vec3d{0.0, 0.0, 5.0 * d},
        /*direction=*/Vec3d{0.0, 0.0, -1.0},
        /*up=*/Vec3d{0.0, d / 2.0, 0.0},
        /*aspect=*/1.0,
        /*near=*/4.0 * d + 1.0,
        /*depth=*/kCubeBBox.dim().z() - 2.0,
        /*x_count=*/100,
        /*z_count=*/100};
    const auto frustumIndexBBox = frustum.getBBox();

    {
        auto clipped = tools::clip(mCube, frustum);

        const auto bbox = clipped->evalActiveVoxelBoundingBox();
        const auto cubeDim = kCubeBBox.dim();
        EXPECT_EQ(kCubeBBox.min().z() + 1, bbox.min().z());
        EXPECT_EQ(kCubeBBox.max().z() - 1, bbox.max().z());
        EXPECT_TRUE(int(bbox.volume()) < int(cubeDim.x() * cubeDim.y() * (cubeDim.z() - 2)));

        // Note: mCube index space corresponds to world space.
        for (auto it = clipped->beginValueOn(); it; ++it) {
            const auto xyz = frustum.applyInverseMap(it.getCoord().asVec3d());
            EXPECT_TRUE(frustumIndexBBox.isInside(xyz));
        }
    }
    {
        auto tile = openvdb::FloatGrid{0.0f};
        tile.tree().addTile(/*level=*/2, Coord{0}, /*value=*/5.0f, /*active=*/true);

        auto clipped = tools::clip(tile, frustum);
        EXPECT_TRUE(!clipped->empty());
        for (auto it = clipped->beginValueOn(); it; ++it) {
            const auto xyz = frustum.applyInverseMap(it.getCoord().asVec3d());
            EXPECT_TRUE(frustumIndexBBox.isInside(xyz));
        }

        clipped = tools::clip(tile, frustum, /*keepInterior=*/false);
        EXPECT_TRUE(!clipped->empty());
        for (auto it = clipped->beginValueOn(); it; ++it) {
            const auto xyz = frustum.applyInverseMap(it.getCoord().asVec3d());
            EXPECT_TRUE(!frustumIndexBBox.isInside(xyz));
        }
    }
}


// Test clipping against a MaskGrid.
TEST_F(TestClip, testMaskGrid)
{
    using namespace openvdb;
    MaskGrid mask(false);
    mask.fill(kInnerBBox, true, true);
    FloatGrid::Ptr clipped = tools::clip(mCube, mask);
    validate(*clipped);
}


// Test clipping against a boolean mask grid.
TEST_F(TestClip, testBoolMask)
{
    using namespace openvdb;
    BoolGrid mask(false);
    mask.fill(kInnerBBox, true, true);
    FloatGrid::Ptr clipped = tools::clip(mCube, mask);
    validate(*clipped);
}


// Test clipping against a boolean mask grid with mask inversion.
TEST_F(TestClip, testInvertedBoolMask)
{
    using namespace openvdb;
    // Construct a mask grid that is the "inverse" of the mask used in the other tests.
    // (This is not a true inverse, since the mask's active voxel bounds are finite.)
    BoolGrid mask(false);
    mask.fill(kCubeBBox, true, true);
    mask.fill(kInnerBBox, false, false);
    // Clipping against the "inverted" mask with mask inversion enabled
    // should give the same results as clipping normally against the normal mask.
    FloatGrid::Ptr clipped = tools::clip(mCube, mask, /*keepInterior=*/false);
    clipped->pruneGrid();
    validate(*clipped);
}


// Test clipping against a non-boolean mask grid.
TEST_F(TestClip, testNonBoolMask)
{
    using namespace openvdb;
    Int32Grid mask(0);
    mask.fill(kInnerBBox, -5, true);
    FloatGrid::Ptr clipped = tools::clip(mCube, mask);
    validate(*clipped);
}


// Test clipping against a non-boolean mask grid with mask inversion.
TEST_F(TestClip, testInvertedNonBoolMask)
{
    using namespace openvdb;
    // Construct a mask grid that is the "inverse" of the mask used in the other tests.
    // (This is not a true inverse, since the mask's active voxel bounds are finite.)
    Grid<UInt32Tree> mask(0);
    auto paddedCubeBBox = kCubeBBox;
    paddedCubeBBox.expand(2);
    mask.fill(paddedCubeBBox, 99, true);
    mask.fill(kInnerBBox, 0, false);
    // Clipping against the "inverted" mask with mask inversion enabled
    // should give the same results as clipping normally against the normal mask.
    FloatGrid::Ptr clipped = tools::clip(mCube, mask, /*keepInterior=*/false);
    clipped->pruneGrid();
    validate(*clipped);
}
