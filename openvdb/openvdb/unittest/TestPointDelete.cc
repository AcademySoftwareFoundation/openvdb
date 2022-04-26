// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointDelete.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace openvdb::points;

class TestPointDelete: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointDelete


////////////////////////////////////////

TEST_F(TestPointDelete, testDeleteFromGroups)
{
    using openvdb::math::Vec3s;
    using openvdb::tools::PointIndexGrid;
    using openvdb::Index64;

    const float voxelSize(1.0);
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(voxelSize));

    const std::vector<Vec3s> positions6Points =  {
                                                {1, 1, 1},
                                                {1, 2, 1},
                                                {2, 1, 1},
                                                {2, 2, 1},
                                                {100, 100, 100},
                                                {100, 101, 100}
                                           };
    const PointAttributeVector<Vec3s> pointList6Points(positions6Points);

    {
        // delete from a tree with 2 leaves, checking that group membership is updated as
        // expected

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList6Points, *transform);

        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList6Points,
                *transform);
        PointDataTree& tree = grid->tree();

        // first test will delete 3 groups, with the third one empty.

        appendGroup(tree, "test1");
        appendGroup(tree, "test2");
        appendGroup(tree, "test3");
        appendGroup(tree, "test4");

        EXPECT_EQ(pointCount(tree), Index64(6));

        std::vector<short> membership1{1, 0, 0, 0, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership1, "test1");

        std::vector<short> membership2{0, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership2, "test2");

        std::vector<std::string> groupsToDelete{"test1", "test2", "test3"};

        deleteFromGroups(tree, groupsToDelete);

        // 4 points should have been deleted, so only 2 remain
        EXPECT_EQ(pointCount(tree), Index64(2));

        // check that first three groups are deleted but the last is not

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();

        AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        EXPECT_TRUE(!descriptor.hasGroup("test1"));
        EXPECT_TRUE(!descriptor.hasGroup("test2"));
        EXPECT_TRUE(!descriptor.hasGroup("test3"));
        EXPECT_TRUE(descriptor.hasGroup("test4"));
    }

    {
        // check deletion from a single leaf tree and that attribute values are preserved
        // correctly after deletion

        std::vector<Vec3s> positions4Points = {
                                                {1, 1, 1},
                                                {1, 2, 1},
                                                {2, 1, 1},
                                                {2, 2, 1},
                                              };

        const PointAttributeVector<Vec3s> pointList4Points(positions4Points);

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList4Points, *transform);

        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                pointList4Points, *transform);
        PointDataTree& tree = grid->tree();

        appendGroup(tree, "test");
        appendAttribute(tree, "testAttribute", TypedAttributeArray<int32_t>::attributeType());

        EXPECT_TRUE(tree.beginLeaf());

        const PointDataTree::LeafIter leafIter = tree.beginLeaf();

        AttributeWriteHandle<int>
            testAttributeWriteHandle(leafIter->attributeArray("testAttribute"));

        for(int i = 0; i < 4; i++) {
            testAttributeWriteHandle.set(i,i+1);
        }

        std::vector<short> membership{0, 1, 1, 0};
        setGroup(tree, pointIndexGrid->tree(), membership, "test");

        deleteFromGroup(tree, "test");

        EXPECT_EQ(pointCount(tree), Index64(2));

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();
        const AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        const AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        EXPECT_TRUE(descriptor.find("testAttribute") != AttributeSet::INVALID_POS);

        AttributeHandle<int> testAttributeHandle(*attributeSetAfterDeletion.get("testAttribute"));

        EXPECT_EQ(1, testAttributeHandle.get(0));
        EXPECT_EQ(4, testAttributeHandle.get(1));
    }

    {
        // test the invert flag using data similar to that used in the first test

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList6Points, *transform);
        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList6Points,
                *transform);
        PointDataTree& tree = grid->tree();

        appendGroup(tree, "test1");
        appendGroup(tree, "test2");
        appendGroup(tree, "test3");
        appendGroup(tree, "test4");

        EXPECT_EQ(pointCount(tree), Index64(6));

        std::vector<short> membership1{1, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership1, "test1");

        std::vector<short> membership2{0, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership2, "test2");

        std::vector<std::string> groupsToDelete{"test1", "test3"};

        deleteFromGroups(tree, groupsToDelete, /*invert=*/ true);

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();
        const AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        const AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        // no groups should be dropped when invert = true
        EXPECT_EQ(static_cast<size_t>(descriptor.groupMap().size()),
                             static_cast<size_t>(4));

        // 4 points should remain since test1 and test3 have 4 members between then
        EXPECT_EQ(static_cast<size_t>(pointCount(tree)),
                             static_cast<size_t>(4));
    }

    {
        // similar to first test, but don't drop groups

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList6Points, *transform);

        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList6Points,
                *transform);
        PointDataTree& tree = grid->tree();

        // first test will delete 3 groups, with the third one empty.

        appendGroup(tree, "test1");
        appendGroup(tree, "test2");
        appendGroup(tree, "test3");
        appendGroup(tree, "test4");

        std::vector<short> membership1{1, 0, 0, 0, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership1, "test1");

        std::vector<short> membership2{0, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership2, "test2");

        std::vector<std::string> groupsToDelete{"test1", "test2", "test3"};

        deleteFromGroups(tree, groupsToDelete, /*invert=*/ false, /*drop=*/ false);

        // 4 points should have been deleted, so only 2 remain
        EXPECT_EQ(pointCount(tree), Index64(2));

        // check that first three groups are deleted but the last is not

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();

        AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        // all group should still be present

        EXPECT_TRUE(descriptor.hasGroup("test1"));
        EXPECT_TRUE(descriptor.hasGroup("test2"));
        EXPECT_TRUE(descriptor.hasGroup("test3"));
        EXPECT_TRUE(descriptor.hasGroup("test4"));
    }

}
