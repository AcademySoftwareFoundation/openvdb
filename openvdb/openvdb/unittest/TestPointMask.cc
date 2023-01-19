// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointMask.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointMask: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointMask


TEST_F(TestPointMask, testMask)
{
    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 5, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    const float voxelSize = 0.1f;
    openvdb::math::Transform::Ptr transform(
        openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                          pointList, *transform);

    { // simple topology copy
        auto mask = convertPointsToMask(*points);

        EXPECT_EQ(points->tree().activeVoxelCount(), Index64(4));
        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(4));

        // also test tree function signature
        auto maskTree = convertPointsToMask(points->tree());
        EXPECT_EQ(maskTree->activeVoxelCount(), Index64(4));
        EXPECT_TRUE(maskTree->hasSameTopology(mask->tree()));
    }

    { // mask grid instead of bool grid
        auto mask = convertPointsToMask<PointDataGrid, MaskGrid>(*points);

        EXPECT_EQ(points->tree().activeVoxelCount(), Index64(4));
        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(4));
    }

    { // identical transform
        auto mask = convertPointsToMask(*points, *transform);

        EXPECT_EQ(points->tree().activeVoxelCount(), Index64(4));
        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(4));
    }

    // assign point 3 to new group "test"

    appendGroup(points->tree(), "test");

    std::vector<short> groups{0,0,1,0};

    setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");

    std::vector<std::string> includeGroups{"test"};
    std::vector<std::string> excludeGroups;

    { // convert in turn "test" and not "test"
        MultiGroupFilter filter(includeGroups, excludeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        auto mask = convertPointsToMask(*points, filter);

        EXPECT_EQ(points->tree().activeVoxelCount(), Index64(4));
        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(1));

        MultiGroupFilter filter2(excludeGroups, includeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        mask = convertPointsToMask(*points, filter2);

        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(3));
    }

    { // use a much larger voxel size that splits the points into two regions
        const float newVoxelSize(4);
        openvdb::math::Transform::Ptr newTransform(
            openvdb::math::Transform::createLinearTransform(newVoxelSize));

        auto mask = convertPointsToMask(*points, *newTransform);

        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(2));

        MultiGroupFilter filter(includeGroups, excludeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        mask = convertPointsToMask(*points, *newTransform, filter);

        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(1));

        MultiGroupFilter filter2(excludeGroups, includeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        mask = convertPointsToMask(*points, *newTransform, filter2);

        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(2));
    }
}


struct StaticVoxelDeformer
{
    StaticVoxelDeformer(const Vec3d& position)
        : mPosition(position) { }

    template <typename LeafT>
    void reset(LeafT& /*leaf*/, size_t /*idx*/) { }

    template <typename IterT>
    void apply(Vec3d& position, IterT&) const { position = mPosition; }

private:
    Vec3d mPosition;
};

template <bool WorldSpace = true>
struct YOffsetDeformer
{
    YOffsetDeformer(const Vec3d& offset) : mOffset(offset) { }

    template <typename LeafT>
    void reset(LeafT& /*leaf*/, size_t /*idx*/) { }

    template <typename IterT>
    void apply(Vec3d& position, IterT&) const { position += mOffset; }

    Vec3d mOffset;
};

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

// configure both voxel deformers to be applied in index-space

template<>
struct DeformerTraits<StaticVoxelDeformer> {
    static const bool IndexSpace = true;
};

template<>
struct DeformerTraits<YOffsetDeformer<false>> {
    static const bool IndexSpace = true;
};

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


TEST_F(TestPointMask, testMaskDeformer)
{
    // This test validates internal functionality that is used in various applications, such as
    // building masks and producing count grids. Note that by convention, methods that live
    // in an "internal" namespace are typically not promoted as part of the public API
    // and thus do not receive the same level of rigour in avoiding breaking API changes.

    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 5, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    const float voxelSize = 0.1f;
    openvdb::math::Transform::Ptr transform(
        openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                          pointList, *transform);

    // assign point 3 to new group "test"

    appendGroup(points->tree(), "test");

    std::vector<short> groups{0,0,1,0};

    setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");

    NullFilter nullFilter;

    { // null deformer
        NullDeformer deformer;

        auto mask = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformer);

        auto mask2 = convertPointsToMask(*points);

        EXPECT_EQ(points->tree().activeVoxelCount(), Index64(4));
        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(4));
        EXPECT_TRUE(mask->tree().hasSameTopology(mask2->tree()));
        EXPECT_TRUE(mask->tree().hasSameTopology(points->tree()));
    }

    { // static voxel deformer
        // collapse all points into a random voxel at (9, 13, 106)
        StaticVoxelDeformer deformer(Vec3d(9, 13, 106));

        auto mask = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformer);

        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(1));
        EXPECT_TRUE(!mask->tree().cbeginLeaf()->isValueOn(Coord(9, 13, 105)));
        EXPECT_TRUE(mask->tree().cbeginLeaf()->isValueOn(Coord(9, 13, 106)));
    }

    { // +y offset deformer
        Vec3d offset(0, 41.7, 0);
        YOffsetDeformer</*world-space*/false> deformer(offset);

        auto mask = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformer);

        // (repeat with deformer configured as world-space)
        YOffsetDeformer</*world-space*/true> deformerWS(offset * voxelSize);

        auto maskWS = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformerWS);

        EXPECT_EQ(mask->tree().activeVoxelCount(), Index64(4));
        EXPECT_EQ(maskWS->tree().activeVoxelCount(), Index64(4));

        std::vector<Coord> maskVoxels;
        std::vector<Coord> maskVoxelsWS;
        std::vector<Coord> pointVoxels;

        for (auto leaf = mask->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                maskVoxels.emplace_back(iter.getCoord());
            }
        }

        for (auto leaf = maskWS->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                maskVoxelsWS.emplace_back(iter.getCoord());
            }
        }

        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                pointVoxels.emplace_back(iter.getCoord());
            }
        }

        std::sort(maskVoxels.begin(), maskVoxels.end());
        std::sort(maskVoxelsWS.begin(), maskVoxelsWS.end());
        std::sort(pointVoxels.begin(), pointVoxels.end());

        EXPECT_EQ(maskVoxels.size(), size_t(4));
        EXPECT_EQ(maskVoxelsWS.size(), size_t(4));
        EXPECT_EQ(pointVoxels.size(), size_t(4));

        for (int i = 0; i < int(pointVoxels.size()); i++) {
            Coord newCoord(pointVoxels[i]);
            newCoord.x() = static_cast<Int32>(newCoord.x() + offset.x());
            newCoord.y() = static_cast<Int32>(math::Round(newCoord.y() + offset.y()));
            newCoord.z() = static_cast<Int32>(newCoord.z() + offset.z());
            EXPECT_EQ(maskVoxels[i], newCoord);
            EXPECT_EQ(maskVoxelsWS[i], newCoord);
        }

        // use a different transform to verify deformers and transforms can be used together

        const float newVoxelSize = 0.02f;
        openvdb::math::Transform::Ptr newTransform(
            openvdb::math::Transform::createLinearTransform(newVoxelSize));

        auto mask2 = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *newTransform, nullFilter, deformer);

        EXPECT_EQ(mask2->tree().activeVoxelCount(), Index64(4));

        std::vector<Coord> maskVoxels2;

        for (auto leaf = mask2->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                maskVoxels2.emplace_back(iter.getCoord());
            }
        }

        std::sort(maskVoxels2.begin(), maskVoxels2.end());

        for (int i = 0; i < int(maskVoxels.size()); i++) {
            Coord newCoord(pointVoxels[i]);
            newCoord.x() = static_cast<Int32>((newCoord.x() + offset.x()) * 5);
            newCoord.y() = static_cast<Int32>(math::Round((newCoord.y() + offset.y()) * 5));
            newCoord.z() = static_cast<Int32>((newCoord.z() + offset.z()) * 5);
            EXPECT_EQ(maskVoxels2[i], newCoord);
        }

        // only use points in group "test"

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;
        MultiGroupFilter filter(includeGroups, excludeGroups,
            points->tree().cbeginLeaf()->attributeSet());

        auto mask3 = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, filter, deformer);

        EXPECT_EQ(mask3->tree().activeVoxelCount(), Index64(1));

        for (auto leaf = mask3->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                Coord newCoord(pointVoxels[2]);
                newCoord.x() = static_cast<Int32>(newCoord.x() + offset.x());
                newCoord.y() = static_cast<Int32>(math::Round(newCoord.y() + offset.y()));
                newCoord.z() = static_cast<Int32>(newCoord.z() + offset.z());
                EXPECT_EQ(iter.getCoord(), newCoord);
            }
        }
    }
}
