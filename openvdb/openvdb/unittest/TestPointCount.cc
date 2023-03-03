// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/TempFile.h>
#include <openvdb/math/Math.h>

#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio> // for std::remove()
#include <cstdlib> // for std::getenv()
#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointCount: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointCount

using LeafType  = PointDataTree::LeafNodeType;
using ValueType = LeafType::ValueType;


struct NotZeroFilter
{
    NotZeroFilter() = default;
    static bool initialized() { return true; }
    template <typename LeafT>
    void reset(const LeafT&) { }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return *iter != 0;
    }
};

TEST_F(TestPointCount, testCount)
{
    // create a tree and check there are no points

    PointDataGrid::Ptr grid = createGrid<PointDataGrid>();
    PointDataTree& tree = grid->tree();

    EXPECT_EQ(pointCount(tree), Index64(0));

    // add a new leaf to a tree and re-test

    LeafType* leafPtr = tree.touchLeaf(openvdb::Coord(0, 0, 0));
    LeafType& leaf(*leafPtr);

    EXPECT_EQ(pointCount(tree), Index64(0));

    // now manually set some offsets

    leaf.setOffsetOn(0, 4);
    leaf.setOffsetOn(1, 7);

    ValueVoxelCIter voxelIter = leaf.beginValueVoxel(openvdb::Coord(0, 0, 0));

    IndexIter<ValueVoxelCIter, NullFilter> testIter(voxelIter, NullFilter());

    leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0));

    EXPECT_EQ(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0))), 0);
    EXPECT_EQ(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0)).end()), 4);

    EXPECT_EQ(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1))), 4);
    EXPECT_EQ(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1)).end()), 7);

    // test filtered, index voxel iterator

    EXPECT_EQ(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0), NotZeroFilter())), 1);
    EXPECT_EQ(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0), NotZeroFilter()).end()), 4);

    {
        LeafType::IndexVoxelIter iter = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0));

        EXPECT_EQ(int(*iter), 0);
        EXPECT_EQ(int(iter.end()), 4);

        LeafType::IndexVoxelIter iter2 = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1));

        EXPECT_EQ(int(*iter2), 4);
        EXPECT_EQ(int(iter2.end()), 7);

        EXPECT_EQ(iterCount(iter2), Index64(7 - 4));

        // check pointCount ignores active/inactive state

        leaf.setValueOff(1);

        LeafType::IndexVoxelIter iter3 = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1));

        EXPECT_EQ(iterCount(iter3), Index64(7 - 4));

        leaf.setValueOn(1);
    }

    // one point per voxel

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, i);
    }

    EXPECT_EQ(leaf.pointCount(), Index64(LeafType::SIZE - 1));
    EXPECT_EQ(leaf.onPointCount(), Index64(LeafType::SIZE - 1));
    EXPECT_EQ(leaf.offPointCount(), Index64(0));

    EXPECT_EQ(pointCount(tree), Index64(LeafType::SIZE - 1));
    EXPECT_EQ(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE - 1));
    EXPECT_EQ(pointCount(tree, InactiveFilter()), Index64(0));

    // manually de-activate two voxels

    leaf.setValueOff(100);
    leaf.setValueOff(101);

    EXPECT_EQ(leaf.pointCount(), Index64(LeafType::SIZE - 1));
    EXPECT_EQ(leaf.onPointCount(), Index64(LeafType::SIZE - 3));
    EXPECT_EQ(leaf.offPointCount(), Index64(2));

    EXPECT_EQ(pointCount(tree), Index64(LeafType::SIZE - 1));
    EXPECT_EQ(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE - 3));
    EXPECT_EQ(pointCount(tree, InactiveFilter()), Index64(2));

    // one point per every other voxel and de-activate empty voxels

    unsigned sum = 0;

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, sum);
        if (i % 2 == 0)     sum++;
    }

    leaf.updateValueMask();

    EXPECT_EQ(leaf.pointCount(), Index64(LeafType::SIZE / 2));
    EXPECT_EQ(leaf.onPointCount(), Index64(LeafType::SIZE / 2));
    EXPECT_EQ(leaf.offPointCount(), Index64(0));

    EXPECT_EQ(pointCount(tree), Index64(LeafType::SIZE / 2));
    EXPECT_EQ(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE / 2));
    EXPECT_EQ(pointCount(tree, InactiveFilter()), Index64(0));

    // add a new non-empty leaf and check totalPointCount is correct

    LeafType* leaf2Ptr = tree.touchLeaf(openvdb::Coord(0, 0, 8));
    LeafType& leaf2(*leaf2Ptr);

    // on adding, tree now obtains ownership and is reponsible for deletion

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf2.setOffsetOn(i, i);
    }

    EXPECT_EQ(pointCount(tree), Index64(LeafType::SIZE / 2 + LeafType::SIZE - 1));
    EXPECT_EQ(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE / 2 + LeafType::SIZE - 1));
    EXPECT_EQ(pointCount(tree, InactiveFilter()), Index64(0));
}


TEST_F(TestPointCount, testGroup)
{
    using namespace openvdb::math;

    using Descriptor = AttributeSet::Descriptor;

    // four points in the same leaf

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 2, 1}, {2, 1, 1}, {2, 2, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf
    EXPECT_EQ(tree.leafCount(), Index32(1));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafIter leafIter = tree.beginLeaf();
    const AttributeSet& firstAttributeSet = leafIter->attributeSet();

    // ensure zero groups
    EXPECT_EQ(firstAttributeSet.descriptor().groupMap().size(), size_t(0));

    {// add an empty group
        appendGroup(tree, "test");

        EXPECT_EQ(firstAttributeSet.descriptor().groupMap().size(), size_t(1));

        EXPECT_EQ(pointCount(tree), Index64(4));
        EXPECT_EQ(pointCount(tree, ActiveFilter()), Index64(4));
        EXPECT_EQ(pointCount(tree, InactiveFilter()), Index64(0));
        EXPECT_EQ(leafIter->pointCount(), Index64(4));
        EXPECT_EQ(leafIter->onPointCount(), Index64(4));
        EXPECT_EQ(leafIter->offPointCount(), Index64(0));

        // no points found when filtered by the empty group

        EXPECT_EQ(pointCount(tree, GroupFilter("test", firstAttributeSet)), Index64(0));
        EXPECT_EQ(leafIter->groupPointCount("test"), Index64(0));
    }

    { // assign two points to the group, test offsets and point counts
        const Descriptor::GroupIndex index = firstAttributeSet.groupIndex("test");

        EXPECT_TRUE(index.first != AttributeSet::INVALID_POS);
        EXPECT_TRUE(index.first < firstAttributeSet.size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        EXPECT_TRUE(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);
        groupArray.set(3, GroupType(1) << index.second);

        // only two out of four points should be found when group filtered

        GroupFilter firstGroupFilter("test", firstAttributeSet);

        EXPECT_EQ(pointCount(tree, GroupFilter("test", firstAttributeSet)), Index64(2));
        EXPECT_EQ(leafIter->groupPointCount("test"), Index64(2));

        {
            EXPECT_EQ(pointCount(tree, BinaryFilter<GroupFilter, ActiveFilter>(
                firstGroupFilter, ActiveFilter())), Index64(2));
            EXPECT_EQ(pointCount(tree, BinaryFilter<GroupFilter, InactiveFilter>(
                firstGroupFilter, InactiveFilter())), Index64(0));
        }

        EXPECT_NO_THROW(leafIter->validateOffsets());

        // manually modify offsets so one of the points is marked as inactive

        std::vector<ValueType> offsets, modifiedOffsets;
        offsets.resize(PointDataTree::LeafNodeType::SIZE);
        modifiedOffsets.resize(PointDataTree::LeafNodeType::SIZE);

        for (Index n = 0; n < PointDataTree::LeafNodeType::NUM_VALUES; n++) {
            const unsigned offset = leafIter->getValue(n);
            offsets[n] = offset;
            modifiedOffsets[n] = offset > 0 ? offset - 1 : offset;
        }

        leafIter->setOffsets(modifiedOffsets);

        // confirm that validation fails

        EXPECT_THROW(leafIter->validateOffsets(), openvdb::ValueError);

        // replace offsets with original offsets but leave value mask

        leafIter->setOffsets(offsets, /*updateValueMask=*/ false);

        // confirm that validation now succeeds

        EXPECT_NO_THROW(leafIter->validateOffsets());

        // ensure active / inactive point counts are correct

        EXPECT_EQ(pointCount(tree, GroupFilter("test", firstAttributeSet)), Index64(2));
        EXPECT_EQ(leafIter->groupPointCount("test"), Index64(2));
        EXPECT_EQ(pointCount(tree, BinaryFilter<GroupFilter, ActiveFilter>(
            firstGroupFilter, ActiveFilter())), Index64(1));
        EXPECT_EQ(pointCount(tree, BinaryFilter<GroupFilter, InactiveFilter>(
            firstGroupFilter, InactiveFilter())), Index64(1));

        EXPECT_EQ(pointCount(tree), Index64(4));
        EXPECT_EQ(pointCount(tree, ActiveFilter()), Index64(3));
        EXPECT_EQ(pointCount(tree, InactiveFilter()), Index64(1));

        std::string filename;

        // write out grid to a temp file
        {
            filename = "testPointCount1.vdb";
            io::File fileOut(filename);
            GridCPtrVec grids{grid};
            fileOut.write(grids);
        }

        // test point count of a delay-loaded grid
        {
            io::File fileIn(filename);
            fileIn.open();

            GridPtrVecPtr grids = fileIn.getGrids();

            fileIn.close();

            EXPECT_EQ(grids->size(), size_t(1));

            PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);

            EXPECT_TRUE(inputGrid);

            PointDataTree& inputTree = inputGrid->tree();
            const auto& attributeSet = inputTree.cbeginLeaf()->attributeSet();

            GroupFilter groupFilter("test", attributeSet);

            bool inCoreOnly;
#ifdef OPENVDB_USE_DELAYED_LOADING
            inCoreOnly = true;

            EXPECT_EQ(pointCount(inputTree, NullFilter(), inCoreOnly), Index64(0));
            EXPECT_EQ(pointCount(inputTree, ActiveFilter(), inCoreOnly), Index64(0));
            EXPECT_EQ(pointCount(inputTree, InactiveFilter(), inCoreOnly), Index64(0));
            EXPECT_EQ(pointCount(inputTree, groupFilter, inCoreOnly), Index64(0));
            EXPECT_EQ(pointCount(inputTree, BinaryFilter<GroupFilter, ActiveFilter>(
                groupFilter, ActiveFilter()), inCoreOnly), Index64(0));
            EXPECT_EQ(pointCount(inputTree, BinaryFilter<GroupFilter, InactiveFilter>(
                groupFilter, InactiveFilter()), inCoreOnly), Index64(0));
#endif

            inCoreOnly = false;

            EXPECT_EQ(pointCount(inputTree, NullFilter(), inCoreOnly), Index64(4));
            EXPECT_EQ(pointCount(inputTree, ActiveFilter(), inCoreOnly), Index64(3));
            EXPECT_EQ(pointCount(inputTree, InactiveFilter(), inCoreOnly), Index64(1));
            EXPECT_EQ(pointCount(inputTree, groupFilter, inCoreOnly), Index64(2));
            EXPECT_EQ(pointCount(inputTree, BinaryFilter<GroupFilter, ActiveFilter>(
                groupFilter, ActiveFilter()), inCoreOnly), Index64(1));
            EXPECT_EQ(pointCount(inputTree, BinaryFilter<GroupFilter, InactiveFilter>(
                groupFilter, InactiveFilter()), inCoreOnly), Index64(1));
        }

        std::remove(filename.c_str());

        // update the value mask and confirm point counts once again

        leafIter->updateValueMask();

        EXPECT_NO_THROW(leafIter->validateOffsets());

        auto& attributeSet = tree.cbeginLeaf()->attributeSet();

        EXPECT_EQ(pointCount(tree, GroupFilter("test", attributeSet)), Index64(2));
        EXPECT_EQ(leafIter->groupPointCount("test"), Index64(2));
        EXPECT_EQ(pointCount(tree, BinaryFilter<GroupFilter, ActiveFilter>(
            firstGroupFilter, ActiveFilter())), Index64(2));
        EXPECT_EQ(pointCount(tree, BinaryFilter<GroupFilter, InactiveFilter>(
            firstGroupFilter, InactiveFilter())), Index64(0));

        EXPECT_EQ(pointCount(tree), Index64(4));
        EXPECT_EQ(pointCount(tree, ActiveFilter()), Index64(4));
        EXPECT_EQ(pointCount(tree, InactiveFilter()), Index64(0));
    }

    // create a tree with multiple leaves

    positions.emplace_back(20, 1, 1);
    positions.emplace_back(1, 20, 1);
    positions.emplace_back(1, 1, 20);

    grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree2 = grid->tree();

    EXPECT_EQ(tree2.leafCount(), Index32(4));

    leafIter = tree2.beginLeaf();

    appendGroup(tree2, "test");

    { // assign two points to the group
        const auto& attributeSet = leafIter->attributeSet();
        const Descriptor::GroupIndex index = attributeSet.groupIndex("test");

        EXPECT_TRUE(index.first != AttributeSet::INVALID_POS);
        EXPECT_TRUE(index.first < attributeSet.size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        EXPECT_TRUE(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);
        groupArray.set(3, GroupType(1) << index.second);

        EXPECT_EQ(pointCount(tree2, GroupFilter("test", attributeSet)), Index64(2));
        EXPECT_EQ(leafIter->groupPointCount("test"), Index64(2));
        EXPECT_EQ(pointCount(tree2), Index64(7));
    }

    ++leafIter;

    EXPECT_TRUE(leafIter);

    { // assign another point to the group in a different leaf
        const auto& attributeSet = leafIter->attributeSet();
        const Descriptor::GroupIndex index = attributeSet.groupIndex("test");

        EXPECT_TRUE(index.first != AttributeSet::INVALID_POS);
        EXPECT_TRUE(index.first < leafIter->attributeSet().size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        EXPECT_TRUE(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);

        EXPECT_EQ(pointCount(tree2, GroupFilter("test", attributeSet)), Index64(3));
        EXPECT_EQ(leafIter->groupPointCount("test"), Index64(1));
        EXPECT_EQ(pointCount(tree2), Index64(7));
    }
}


TEST_F(TestPointCount, testOffsets)
{
    using namespace openvdb::math;

    // empty tree
    {
        PointDataTree tree;
        std::vector<Index64> offsets{ 10 };
        Index64 total = pointOffsets(offsets, tree);
        EXPECT_EQ(total, Index64(0));
        EXPECT_TRUE(offsets.empty());
    }

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    // five points across four leafs

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 101, 1}, {2, 101, 1}, {101, 1, 1}, {101, 101, 1}};

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    { // all point offsets
        std::vector<Index64> offsets;
        Index64 total = pointOffsets(offsets, tree);

        EXPECT_EQ(offsets.size(), size_t(4));
        EXPECT_EQ(offsets[0], Index64(1));
        EXPECT_EQ(offsets[1], Index64(3));
        EXPECT_EQ(offsets[2], Index64(4));
        EXPECT_EQ(offsets[3], Index64(5));
        EXPECT_EQ(total, Index64(5));
    }

    { // all point offsets when using a non-existant exclude group

        std::vector<Index64> offsets;

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups{"empty"};

        MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, tree, filter);

        EXPECT_EQ(offsets.size(), size_t(4));
        EXPECT_EQ(offsets[0], Index64(1));
        EXPECT_EQ(offsets[1], Index64(3));
        EXPECT_EQ(offsets[2], Index64(4));
        EXPECT_EQ(offsets[3], Index64(5));
        EXPECT_EQ(total, Index64(5));
    }

    appendGroup(tree, "test");

    // add one point to the group from the leaf that contains two points

    PointDataTree::LeafIter iter = ++tree.beginLeaf();
    GroupWriteHandle groupHandle = iter->groupWriteHandle("test");
    groupHandle.set(0, true);

    { // include this group
        std::vector<Index64> offsets;

        std::vector<Name> includeGroups{"test"};
        std::vector<Name> excludeGroups;

        MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, tree, filter);

        EXPECT_EQ(offsets.size(), size_t(4));
        EXPECT_EQ(offsets[0], Index64(0));
        EXPECT_EQ(offsets[1], Index64(1));
        EXPECT_EQ(offsets[2], Index64(1));
        EXPECT_EQ(offsets[3], Index64(1));
        EXPECT_EQ(total, Index64(1));
    }

    { // exclude this group
        std::vector<Index64> offsets;

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups{"test"};

        MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, tree, filter);

        EXPECT_EQ(offsets.size(), size_t(4));
        EXPECT_EQ(offsets[0], Index64(1));
        EXPECT_EQ(offsets[1], Index64(2));
        EXPECT_EQ(offsets[2], Index64(3));
        EXPECT_EQ(offsets[3], Index64(4));
        EXPECT_EQ(total, Index64(4));
    }

    std::string filename;

    // write out grid to a temp file
    {
        filename = "testPointCount1.vdb";
        io::File fileOut(filename);
        GridCPtrVec grids{grid};
        fileOut.write(grids);
    }

#ifdef OPENVDB_USE_DELAYED_LOADING
    // test point offsets for a delay-loaded grid
    {
        io::File fileIn(filename);
        fileIn.open();

        GridPtrVecPtr grids = fileIn.getGrids();

        fileIn.close();

        EXPECT_EQ(grids->size(), size_t(1));

        PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);

        EXPECT_TRUE(inputGrid);

        PointDataTree& inputTree = inputGrid->tree();

        std::vector<Index64> offsets;
        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups;

        MultiGroupFilter filter(includeGroups, excludeGroups, inputTree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, inputTree, filter, /*inCoreOnly=*/true);

        EXPECT_EQ(offsets.size(), size_t(4));
        EXPECT_EQ(offsets[0], Index64(0));
        EXPECT_EQ(offsets[1], Index64(0));
        EXPECT_EQ(offsets[2], Index64(0));
        EXPECT_EQ(offsets[3], Index64(0));
        EXPECT_EQ(total, Index64(0));

        offsets.clear();

        total = pointOffsets(offsets, inputTree, filter, /*inCoreOnly=*/false);

        EXPECT_EQ(offsets.size(), size_t(4));
        EXPECT_EQ(offsets[0], Index64(1));
        EXPECT_EQ(offsets[1], Index64(3));
        EXPECT_EQ(offsets[2], Index64(4));
        EXPECT_EQ(offsets[3], Index64(5));
        EXPECT_EQ(total, Index64(5));
    }
#endif

    std::remove(filename.c_str());
}


namespace {

// sum all voxel values
template<typename GridT>
inline Index64
voxelSum(const GridT& grid)
{
    Index64 total = 0;
    for (auto iter = grid.cbeginValueOn(); iter; ++iter) {
        total += static_cast<Index64>(*iter);
    }
    return total;
}

// Generate random points by uniformly distributing points on a unit-sphere.
inline void
genPoints(std::vector<Vec3R>& positions, const int numPoints, const double scale)
{
    // init
    math::Random01 randNumber(0);
    const int n = int(std::sqrt(double(numPoints)));
    const double xScale = (2.0 * openvdb::math::pi<double>()) / double(n);
    const double yScale = openvdb::math::pi<double>() / double(n);

    double x, y, theta, phi;
    Vec3R pos;

    positions.reserve(n*n);

    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {

            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();

            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi   = x * xScale; // [0 to 2PI]

            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            pos[0] = static_cast<float>(std::sin(theta)*std::cos(phi)*scale);
            pos[1] = static_cast<float>(std::sin(theta)*std::sin(phi)*scale);
            pos[2] = static_cast<float>(std::cos(theta)*scale);

            positions.push_back(pos);
        }
    }
}

} // namespace


TEST_F(TestPointCount, testCountGrid)
{
    using namespace openvdb::math;

    { // five points
        std::vector<Vec3s> positions{   {1, 1, 1},
                                        {1, 101, 1},
                                        {2, 101, 1},
                                        {101, 1, 1},
                                        {101, 101, 1}};

        { // in five voxels

            math::Transform::Ptr transform(math::Transform::createLinearTransform(1.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int32Grid::Ptr count = pointCountGrid(*points);

            EXPECT_EQ(count->activeVoxelCount(), points->activeVoxelCount());
            EXPECT_EQ(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int32Grid::Ptr count = pointCountGrid(*points);

            EXPECT_EQ(count->activeVoxelCount(), points->activeVoxelCount());
            EXPECT_EQ(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count), pointCount(points->tree()));
        }

        { // in one voxel

            math::Transform::Ptr transform(math::Transform::createLinearTransform(1000.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int32Grid::Ptr count = pointCountGrid(*points);

            EXPECT_EQ(count->activeVoxelCount(), points->activeVoxelCount());
            EXPECT_EQ(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels, Int64 grid

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int64Grid::Ptr count = pointCountGrid<PointDataGrid, Int64Grid>(*points);

            EXPECT_EQ(count->activeVoxelCount(), points->activeVoxelCount());
            EXPECT_EQ(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels, float grid

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            FloatGrid::Ptr count = pointCountGrid<PointDataGrid, FloatGrid>(*points);

            EXPECT_EQ(count->activeVoxelCount(), points->activeVoxelCount());
            EXPECT_EQ(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            const PointAttributeVector<Vec3s> pointList(positions);
            tools::PointIndexGrid::Ptr pointIndexGrid =
                tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

            PointDataGrid::Ptr points =
                    createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                                  pointList, *transform);

            auto& tree = points->tree();

            // assign point 3 to new group "test"

            appendGroup(tree, "test");

            std::vector<short> groups{0,0,1,0,0};

            setGroup(tree, pointIndexGrid->tree(), groups, "test");

            std::vector<std::string> includeGroups{"test"};
            std::vector<std::string> excludeGroups;

            // generate a count grid with the same transform

            MultiGroupFilter filter(includeGroups, excludeGroups,
                tree.cbeginLeaf()->attributeSet());
            Int32Grid::Ptr count = pointCountGrid(*points, filter);

            EXPECT_EQ(count->activeVoxelCount(), Index64(1));
            EXPECT_EQ(voxelSum(*count), Index64(1));

            MultiGroupFilter filter2(excludeGroups, includeGroups,
                tree.cbeginLeaf()->attributeSet());
            count = pointCountGrid(*points, filter2);

            EXPECT_EQ(count->activeVoxelCount(), Index64(4));
            EXPECT_EQ(voxelSum(*count), Index64(4));
        }
    }

    { // 40,000 points on a unit sphere
        std::vector<Vec3R> positions;
        const size_t total = 40000;
        genPoints(positions, total, /*scale=*/100.0);
        EXPECT_EQ(positions.size(), total);

        math::Transform::Ptr transform1(math::Transform::createLinearTransform(1.0f));
        math::Transform::Ptr transform5(math::Transform::createLinearTransform(5.0f));

        PointDataGrid::Ptr points1 =
            createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
        PointDataGrid::Ptr points5 =
            createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform5);

        EXPECT_TRUE(points1->activeVoxelCount() != points5->activeVoxelCount());
        EXPECT_TRUE(points1->evalActiveVoxelBoundingBox() != points5->evalActiveVoxelBoundingBox());
        EXPECT_EQ(pointCount(points1->tree()), pointCount(points5->tree()));

        { // generate count grids with the same transform

            Int32Grid::Ptr count1 = pointCountGrid(*points1);

            EXPECT_EQ(count1->activeVoxelCount(), points1->activeVoxelCount());
            EXPECT_EQ(count1->evalActiveVoxelBoundingBox(), points1->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count1), pointCount(points1->tree()));

            Int32Grid::Ptr count5 = pointCountGrid(*points5);

            EXPECT_EQ(count5->activeVoxelCount(), points5->activeVoxelCount());
            EXPECT_EQ(count5->evalActiveVoxelBoundingBox(), points5->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count5), pointCount(points5->tree()));
        }

        { // generate count grids with differing transforms

            Int32Grid::Ptr count1 = pointCountGrid(*points5, *transform1);

            EXPECT_EQ(count1->activeVoxelCount(), points1->activeVoxelCount());
            EXPECT_EQ(count1->evalActiveVoxelBoundingBox(), points1->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count1), pointCount(points5->tree()));

            Int32Grid::Ptr count5 = pointCountGrid(*points1, *transform5);

            EXPECT_EQ(count5->activeVoxelCount(), points5->activeVoxelCount());
            EXPECT_EQ(count5->evalActiveVoxelBoundingBox(), points5->evalActiveVoxelBoundingBox());
            EXPECT_EQ(voxelSum(*count5), pointCount(points1->tree()));
        }
    }
}
