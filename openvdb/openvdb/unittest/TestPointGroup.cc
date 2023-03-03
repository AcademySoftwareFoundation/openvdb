// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <gtest/gtest.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>

#include <cstdio> // for std::remove()
#include <cstdlib> // for std::getenv()
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace openvdb;
using namespace openvdb::points;

class TestPointGroup: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointGroup


////////////////////////////////////////


class FirstFilter
{
public:
    static bool initialized() { return true; }

    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT> void reset(const LeafT&) { }

    template <typename IterT> bool valid(const IterT& iter) const
    {
        return *iter == 0;
    }
}; // class FirstFilter


////////////////////////////////////////


namespace {

    bool testStringVector(std::vector<Name>& input)
    {
        return input.size() == 0;
    }

    bool testStringVector(std::vector<Name>& input, const Name& name1)
    {
        if (input.size() != 1)  return false;
        if (input[0] != name1)  return false;
        return true;
    }

    bool testStringVector(std::vector<Name>& input, const Name& name1, const Name& name2)
    {
        if (input.size() != 2)  return false;
        if (input[0] != name1)  return false;
        if (input[1] != name2)  return false;
        return true;
    }

} // namespace


TEST_F(TestPointGroup, testDescriptor)
{
    // test missing groups deletion

    { // no groups, empty Descriptor
        std::vector<std::string> groups;
        AttributeSet::Descriptor descriptor;
        deleteMissingPointGroups(groups, descriptor);
        EXPECT_TRUE(testStringVector(groups));
    }

    { // one group, empty Descriptor
        std::vector<std::string> groups{"group1"};
        AttributeSet::Descriptor descriptor;
        deleteMissingPointGroups(groups, descriptor);
        EXPECT_TRUE(testStringVector(groups));
    }

    { // one group, Descriptor with same group
        std::vector<std::string> groups{"group1"};
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group1", 0);
        deleteMissingPointGroups(groups, descriptor);
        EXPECT_TRUE(testStringVector(groups, "group1"));
    }

    { // one group, Descriptor with different group
        std::vector<std::string> groups{"group1"};
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group2", 0);
        deleteMissingPointGroups(groups, descriptor);
        EXPECT_TRUE(testStringVector(groups));
    }

    { // three groups, Descriptor with three groups, one different
        std::vector<std::string> groups{"group1", "group3", "group4"};
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group1", 0);
        descriptor.setGroup("group2", 0);
        descriptor.setGroup("group4", 0);
        deleteMissingPointGroups(groups, descriptor);
        EXPECT_TRUE(testStringVector(groups, "group1", "group4"));
    }
}


////////////////////////////////////////


TEST_F(TestPointGroup, testAppendDrop)
{
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    EXPECT_EQ(tree.leafCount(), Index32(4));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    ++leafIter;
    ++leafIter;
    ++leafIter;

    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    { // throw on append or drop an empty group
        EXPECT_THROW(appendGroup(tree, ""), openvdb::KeyError);
        EXPECT_THROW(dropGroup(tree, ""), openvdb::KeyError);
    }

    { // append a group
        appendGroup(tree, "test");

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(1));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test"));
        EXPECT_TRUE(attributeSet4.descriptor().hasGroup("test"));
    }

    { // append a group with non-unique name (repeat the append)
        appendGroup(tree, "test");

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(1));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test"));
        EXPECT_TRUE(attributeSet4.descriptor().hasGroup("test"));
    }

    { // append multiple groups
        std::vector<Name> names{"test2", "test3"};

        appendGroups(tree, names);

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(3));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test"));
        EXPECT_TRUE(attributeSet4.descriptor().hasGroup("test"));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test2"));
        EXPECT_TRUE(attributeSet4.descriptor().hasGroup("test2"));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test3"));
        EXPECT_TRUE(attributeSet4.descriptor().hasGroup("test3"));
    }

    { // append to a copy
        PointDataTree tree2(tree);

        appendGroup(tree2, "copy1");

        EXPECT_TRUE(!attributeSet.descriptor().hasGroup("copy1"));
        EXPECT_TRUE(tree2.beginLeaf()->attributeSet().descriptor().hasGroup("copy1"));
    }

    { // drop a group
        dropGroup(tree, "test2");

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(2));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test"));
        EXPECT_TRUE(attributeSet4.descriptor().hasGroup("test"));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test3"));
        EXPECT_TRUE(attributeSet4.descriptor().hasGroup("test3"));
    }

    { // drop multiple groups
        std::vector<Name> names{"test", "test3"};

        dropGroups(tree, names);

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(0));
    }

    { // drop a copy
        appendGroup(tree, "copy2");

        PointDataTree tree2(tree);

        dropGroup(tree2, "copy2");

        EXPECT_TRUE(attributeSet.descriptor().hasGroup("copy2"));
        EXPECT_TRUE(!tree2.beginLeaf()->attributeSet().descriptor().hasGroup("copy2"));

        dropGroup(tree, "copy2");
    }

    { // set group membership
        appendGroup(tree, "test");

        setGroup(tree, "test", true);

        GroupFilter filter("test", tree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(tree, filter), Index64(4));

        setGroup(tree, "test", false);

        EXPECT_EQ(pointCount(tree, filter), Index64(0));

        dropGroup(tree, "test");
    }

    { // drop all groups
        appendGroup(tree, "test");
        appendGroup(tree, "test2");

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(2));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));

        dropGroups(tree);

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(0));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(0));
    }

    { // check that newly added groups have empty group membership

        // recreate the grid with 3 points in one leaf

        positions = {{1, 1, 1}, {1, 2, 1}, {2, 1, 1}};
        grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
        PointDataTree& newTree = grid->tree();

        appendGroup(newTree, "test");

        // test that a completely new group (with a new group attribute)
        // has empty membership

        EXPECT_TRUE(newTree.cbeginLeaf());
        GroupFilter filter("test", newTree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(newTree, filter), Index64(0));

        // check that membership in a group that was not created with a
        // new attribute array is still empty.
        // we will append a second group, set its membership, then
        // drop it and append a new group with the same name again

        appendGroup(newTree, "test2");

        PointDataTree::LeafIter leafIter2 = newTree.beginLeaf();
        EXPECT_TRUE(leafIter2);

        GroupWriteHandle test2Handle = leafIter2->groupWriteHandle("test2");

        test2Handle.set(0, true);
        test2Handle.set(2, true);

        GroupFilter filter2("test2", newTree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(newTree, filter2), Index64(2));

        // drop and re-add group

        dropGroup(newTree, "test2");
        appendGroup(newTree, "test2");

        // check that group is fully cleared and does not have previously existing data

        EXPECT_EQ(pointCount(newTree, filter2), Index64(0));
    }

}


TEST_F(TestPointGroup, testCompact)
{
    std::vector<Vec3s> positions{{1, 1, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf
    EXPECT_EQ(tree.leafCount(), Index32(1));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    std::stringstream ss;

    { // append nine groups
        for (int i = 0; i < 8; i++) {
            ss.str("");
            ss << "test" << i;
            appendGroup(tree, ss.str());
        }

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(8));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));

        appendGroup(tree, "test8");

        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test0"));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test7"));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test8"));

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(9));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(2));
    }

    { // drop first attribute then compact
        dropGroup(tree, "test5", /*compact=*/false);

        EXPECT_TRUE(!attributeSet.descriptor().hasGroup("test5"));
        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(8));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(2));

        compactGroups(tree);

        EXPECT_TRUE(!attributeSet.descriptor().hasGroup("test5"));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test7"));
        EXPECT_TRUE(attributeSet.descriptor().hasGroup("test8"));
        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(8));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));
    }

    { // append seventeen groups, drop most of them, then compact
        for (int i = 0; i < 17; i++) {
            ss.str("");
            ss << "test" << i;
            appendGroup(tree, ss.str());
        }

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(17));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(3));

        // delete all but 0, 5, 9, 15

        for (int i = 0; i < 17; i++) {
            if (i == 0 || i == 5 || i == 9 || i == 15)  continue;
            ss.str("");
            ss << "test" << i;
            dropGroup(tree, ss.str(), /*compact=*/false);
        }

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(4));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(3));

        // make a copy

        PointDataTree tree2(tree);

        // compact - should now occupy one attribute

        compactGroups(tree);

        EXPECT_EQ(attributeSet.descriptor().groupMap().size(), size_t(4));
        EXPECT_EQ(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));

        // check descriptor has been deep copied

        EXPECT_EQ(tree2.cbeginLeaf()->attributeSet().descriptor().groupMap().size(), size_t(4));
        EXPECT_EQ(tree2.cbeginLeaf()->attributeSet().descriptor().count(GroupAttributeArray::attributeType()), size_t(3));
    }
}


TEST_F(TestPointGroup, testSet)
{
    // four points in the same leaf

    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 2, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                        {100, 100, 100},
                                        {100, 101, 100}
                                    };

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    const PointAttributeVector<Vec3s> pointList(positions);

    openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
        openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    PointDataTree& tree = grid->tree();

    appendGroup(tree, "test");

    EXPECT_EQ(pointCount(tree), Index64(6));
    GroupFilter filter("test", tree.cbeginLeaf()->attributeSet());
    EXPECT_EQ(pointCount(tree, filter), Index64(0));

    // copy tree for descriptor sharing test

    PointDataTree tree2(tree);

    std::vector<short> membership{1, 0, 1, 1, 0, 1};

    // test add to group

    setGroup(tree, "test", true);
    EXPECT_EQ(pointCount(tree, filter), Index64(6));

    // test nothing is done if the index tree contains no valid indices

    tools::PointIndexGrid::Ptr tmpIndexGrid = tools::PointIndexGrid::create();
    setGroup(tree, tmpIndexGrid->tree(), {0,0,0,0,0,0}, "test", /*remove*/true);
    EXPECT_EQ(Index64(6), pointCount(tree, filter));

    // test throw on out of range index

    auto indexLeaf = tmpIndexGrid->tree().touchLeaf(tree.cbeginLeaf()->origin());
    indexLeaf->indices().emplace_back(membership.size());
    EXPECT_THROW(setGroup(tree, tmpIndexGrid->tree(), membership, "test"), IndexError);
    EXPECT_EQ(Index64(6), pointCount(tree, filter));

    // test remove flag

    setGroup(tree, pointIndexGrid->tree(), membership, "test", /*remove*/false);
    EXPECT_EQ(Index64(6), pointCount(tree, filter));

    setGroup(tree, pointIndexGrid->tree(), membership, "test", /*remove*/true);
    EXPECT_EQ(Index64(4), pointCount(tree, filter));

    setGroup(tree, pointIndexGrid->tree(), {0,1,0,0,1,0}, "test", /*remove*/false);
    EXPECT_EQ(Index64(6), pointCount(tree, filter));

    setGroup(tree, pointIndexGrid->tree(), membership, "test", /*remove*/true);

    // check that descriptor remains shared

    appendGroup(tree2, "copy1");

    EXPECT_TRUE(!tree.cbeginLeaf()->attributeSet().descriptor().hasGroup("copy1"));

    dropGroup(tree2, "copy1");

    EXPECT_EQ(pointCount(tree), Index64(6));
    GroupFilter filter2("test", tree.cbeginLeaf()->attributeSet());
    EXPECT_EQ(pointCount(tree, filter2), Index64(4));

    { // IO
        // setup temp directory

        std::string tempDir;
        if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _WIN32
        if (tempDir.empty()) {
            char tempDirBuffer[MAX_PATH+1];
            int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
            EXPECT_TRUE(tempDirLen > 0 && tempDirLen <= MAX_PATH);
            tempDir = tempDirBuffer;
        }
#else
        if (tempDir.empty()) tempDir = P_tmpdir;
#endif

        std::string filename;

        // write out grid to a temp file
        {
            filename = tempDir + "/openvdb_test_point_load";

            io::File fileOut(filename);

            GridCPtrVec grids{grid};

            fileOut.write(grids);
        }

        // read test groups
        {
            io::File fileIn(filename);
            fileIn.open();

            GridPtrVecPtr grids = fileIn.getGrids();

            fileIn.close();

            EXPECT_EQ(grids->size(), size_t(1));

            PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);
            PointDataTree& treex = inputGrid->tree();

            EXPECT_TRUE(treex.cbeginLeaf());

            const PointDataGrid::TreeType::LeafNodeType& leaf = *treex.cbeginLeaf();

            const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();

            EXPECT_TRUE(descriptor.hasGroup("test"));
            EXPECT_EQ(descriptor.groupMap().size(), size_t(1));

            EXPECT_EQ(pointCount(treex), Index64(6));
            GroupFilter filter3("test", leaf.attributeSet());
            EXPECT_EQ(pointCount(treex, filter3), Index64(4));
        }
        std::remove(filename.c_str());
    }
}


TEST_F(TestPointGroup, testFilter)
{
    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));
    PointDataGrid::Ptr grid;

    { // four points in the same leaf
        std::vector<Vec3s> positions =  {
                                            {1, 1, 1},
                                            {1, 2, 1},
                                            {2, 1, 1},
                                            {2, 2, 1},
                                            {100, 100, 100},
                                            {100, 101, 100}
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(pointList, *transform);

        grid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    }

    PointDataTree& tree = grid->tree();

    { // first point filter
        appendGroup(tree, "first");

        EXPECT_EQ(pointCount(tree), Index64(6));
        GroupFilter filter("first", tree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(tree, filter), Index64(0));

        FirstFilter filter2;

        setGroupByFilter<PointDataTree, FirstFilter>(tree, "first", filter2);

        auto iter = tree.cbeginLeaf();

        for ( ; iter; ++iter) {
            EXPECT_EQ(iter->groupPointCount("first"), Index64(1));
        }

        GroupFilter filter3("first", tree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(tree, filter3), Index64(2));
    }

    const openvdb::BBoxd bbox(openvdb::Vec3d(0, 1.5, 0), openvdb::Vec3d(101, 100.5, 101));

    { // bbox filter
        appendGroup(tree, "bbox");

        EXPECT_EQ(pointCount(tree), Index64(6));
        GroupFilter filter("bbox", tree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(tree, filter), Index64(0));

        BBoxFilter filter2(*transform, bbox);

        setGroupByFilter<PointDataTree, BBoxFilter>(tree, "bbox", filter2);

        GroupFilter filter3("bbox", tree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(tree, filter3), Index64(3));
    }

    { // first point filter and bbox filter (intersection of the above two filters)
        appendGroup(tree, "first_bbox");

        EXPECT_EQ(pointCount(tree), Index64(6));
        GroupFilter filter("first_bbox", tree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(tree, filter), Index64(0));

        using FirstBBoxFilter = BinaryFilter<FirstFilter, BBoxFilter>;

        FirstFilter firstFilter;
        BBoxFilter bboxFilter(*transform, bbox);
        FirstBBoxFilter filter2(firstFilter, bboxFilter);

        setGroupByFilter<PointDataTree, FirstBBoxFilter>(tree, "first_bbox", filter2);

        GroupFilter filter3("first_bbox", tree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(tree, filter3), Index64(1));

        std::vector<Vec3f> positions;

        for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
            GroupFilter filterx("first_bbox", iter->attributeSet());
            auto filterIndexIter = iter->beginIndexOn(filterx);

            auto handle = AttributeHandle<Vec3f>::create(iter->attributeArray("P"));

            for ( ; filterIndexIter; ++filterIndexIter) {
                const openvdb::Coord ijk = filterIndexIter.getCoord();
                positions.push_back(handle->get(*filterIndexIter) + ijk.asVec3d());
            }
        }

        EXPECT_EQ(positions.size(), size_t(1));
        EXPECT_EQ(positions[0], Vec3f(100, 100, 100));
    }

    { // add 1000 points in three leafs (positions aren't important)

        std::vector<Vec3s> positions(1000, {1, 1, 1});
        positions.insert(positions.end(), 1000, {1, 1, 9});
        positions.insert(positions.end(), 1000, {9, 9, 9});

        const PointAttributeVector<Vec3s> pointList(positions);

        openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(pointList, *transform);

        grid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

        PointDataTree& newTree = grid->tree();

        EXPECT_EQ(pointCount(newTree), Index64(3000));

        // random - maximum

        appendGroup(newTree, "random_maximum");

        const Index64 target = 1001;

        setGroupByRandomTarget(newTree, "random_maximum", target);

        GroupFilter filter("random_maximum", newTree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(newTree, filter), target);

        // random - percentage

        appendGroup(newTree, "random_percentage");

        setGroupByRandomPercentage(newTree, "random_percentage", 33.333333f);

        GroupFilter filter2("random_percentage", newTree.cbeginLeaf()->attributeSet());
        EXPECT_EQ(pointCount(newTree, filter2), Index64(1000));
    }
}
