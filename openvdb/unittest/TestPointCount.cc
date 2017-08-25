///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <cppunit/extensions/HelperMacros.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/openvdb.h>

#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>

#include <cstdio> // for std::remove()
#include <cstdlib> // for std::getenv()
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace openvdb;
using namespace openvdb::points;

class TestPointCount: public CppUnit::TestCase
{
public:

    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointCount);
    CPPUNIT_TEST(testCount);
    CPPUNIT_TEST(testGroup);
    CPPUNIT_TEST(testOffsets);
    CPPUNIT_TEST_SUITE_END();

    void testCount();
    void testGroup();
    void testOffsets();

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

void
TestPointCount::testCount()
{
    // create a tree and check there are no points

    PointDataGrid::Ptr grid = createGrid<PointDataGrid>();
    PointDataTree& tree = grid->tree();

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(0));

    // add a new leaf to a tree and re-test

    LeafType* leafPtr = tree.touchLeaf(openvdb::Coord(0, 0, 0));
    LeafType& leaf(*leafPtr);

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(0));

    // now manually set some offsets

    leaf.setOffsetOn(0, 4);
    leaf.setOffsetOn(1, 7);

    ValueVoxelCIter voxelIter = leaf.beginValueVoxel(openvdb::Coord(0, 0, 0));

    IndexIter<ValueVoxelCIter, NullFilter> testIter(voxelIter, NullFilter());

    leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0));

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0))), 0);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0)).end()), 4);

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1))), 4);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1)).end()), 7);

    // test filtered, index voxel iterator

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0), NotZeroFilter())), 1);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0), NotZeroFilter()).end()), 4);

    {
        LeafType::IndexVoxelIter iter = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(int(*iter), 0);
        CPPUNIT_ASSERT_EQUAL(int(iter.end()), 4);

        LeafType::IndexVoxelIter iter2 = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1));

        CPPUNIT_ASSERT_EQUAL(int(*iter2), 4);
        CPPUNIT_ASSERT_EQUAL(int(iter2.end()), 7);

        CPPUNIT_ASSERT_EQUAL(iterCount(iter2), Index64(7 - 4));

        // check pointCount ignores active/inactive state

        leaf.setValueOff(1);

        LeafType::IndexVoxelIter iter3 = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1));

        CPPUNIT_ASSERT_EQUAL(iterCount(iter3), Index64(7 - 4));

        leaf.setValueOn(1);
    }

    // one point per voxel

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, i);
    }

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.onPointCount(), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.offPointCount(), Index64(0));

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(activePointCount(tree), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(inactivePointCount(tree), Index64(0));

    // manually de-activate two voxels

    leaf.setValueOff(100);
    leaf.setValueOff(101);

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.onPointCount(), Index64(LeafType::SIZE - 3));
    CPPUNIT_ASSERT_EQUAL(leaf.offPointCount(), Index64(2));

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(activePointCount(tree), Index64(LeafType::SIZE - 3));
    CPPUNIT_ASSERT_EQUAL(inactivePointCount(tree), Index64(2));

    // one point per every other voxel and de-activate empty voxels

    unsigned sum = 0;

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, sum);
        if (i % 2 == 0)     sum++;
    }

    leaf.updateValueMask();

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(leaf.onPointCount(), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(leaf.offPointCount(), Index64(0));

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(activePointCount(tree), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(inactivePointCount(tree), Index64(0));

    // add a new non-empty leaf and check totalPointCount is correct

    LeafType* leaf2Ptr = tree.touchLeaf(openvdb::Coord(0, 0, 8));
    LeafType& leaf2(*leaf2Ptr);

    // on adding, tree now obtains ownership and is reponsible for deletion

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf2.setOffsetOn(i, i);
    }

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE / 2 + LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(activePointCount(tree), Index64(LeafType::SIZE / 2 + LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(inactivePointCount(tree), Index64(0));
}


void
TestPointCount::testGroup()
{
    using namespace openvdb::math;

    using Descriptor = AttributeSet::Descriptor;

    // four points in the same leaf

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 2, 1}, {2, 1, 1}, {2, 2, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // setup temp directory

    std::string tempDir;
    if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _MSC_VER
    if (tempDir.empty()) {
        char tempDirBuffer[MAX_PATH+1];
        int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
        CPPUNIT_ASSERT(tempDirLen > 0 && tempDirLen <= MAX_PATH);
        tempDir = tempDirBuffer;
    }
#else
    if (tempDir.empty()) tempDir = P_tmpdir;
#endif

    std::string filename;

    // check one leaf
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(1));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafIter leafIter = tree.beginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    // ensure zero groups
    CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(0));

    {// add an empty group
        appendGroup(tree, "test");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(1));

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(activePointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(inactivePointCount(tree), Index64(0));
        CPPUNIT_ASSERT_EQUAL(leafIter->pointCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(leafIter->onPointCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(leafIter->offPointCount(), Index64(0));

        // no points found when filtered by the empty group

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(0));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(0));
    }

    { // assign two points to the group, test offsets and point counts
        const Descriptor::GroupIndex index = attributeSet.groupIndex("test");

        CPPUNIT_ASSERT(index.first != AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(index.first < attributeSet.size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        CPPUNIT_ASSERT(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);
        groupArray.set(3, GroupType(1) << index.second);

        // only two out of four points should be found when group filtered

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));

        {
            CPPUNIT_ASSERT_EQUAL(activeGroupPointCount(tree, "test"), Index64(2));
            CPPUNIT_ASSERT_EQUAL(inactiveGroupPointCount(tree, "test"), Index64(0));
        }

        CPPUNIT_ASSERT_NO_THROW(leafIter->validateOffsets());

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

        CPPUNIT_ASSERT_THROW(leafIter->validateOffsets(), openvdb::ValueError);

        // replace offsets with original offsets but leave value mask

        leafIter->setOffsets(offsets, /*updateValueMask=*/ false);

        // confirm that validation now succeeds

        CPPUNIT_ASSERT_NO_THROW(leafIter->validateOffsets());

        // ensure active / inactive point counts are correct

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(activeGroupPointCount(tree, "test"), Index64(1));
        CPPUNIT_ASSERT_EQUAL(inactiveGroupPointCount(tree, "test"), Index64(1));

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(activePointCount(tree), Index64(3));
        CPPUNIT_ASSERT_EQUAL(inactivePointCount(tree), Index64(1));

        // write out grid to a temp file
        {
            filename = tempDir + "/openvdb_test_point_load";

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

            CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

            PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);

            CPPUNIT_ASSERT(inputGrid);

            PointDataTree& inputTree = inputGrid->tree();

#if OPENVDB_ABI_VERSION_NUMBER >= 3
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, /*inCoreOnly=*/true), Index64(0));
            CPPUNIT_ASSERT_EQUAL(activePointCount(inputTree, /*inCoreOnly=*/true), Index64(0));
            CPPUNIT_ASSERT_EQUAL(inactivePointCount(inputTree, /*inCoreOnly=*/true), Index64(0));
            CPPUNIT_ASSERT_EQUAL(groupPointCount(inputTree, "test", /*inCoreOnly=*/true), Index64(0));
            CPPUNIT_ASSERT_EQUAL(activeGroupPointCount(inputTree, "test", /*inCoreOnly=*/true), Index64(0));
            CPPUNIT_ASSERT_EQUAL(inactiveGroupPointCount(inputTree, "test", /*inCoreOnly=*/true), Index64(0));
#else
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, /*inCoreOnly=*/true), Index64(4));
            CPPUNIT_ASSERT_EQUAL(activePointCount(inputTree, /*inCoreOnly=*/true), Index64(3));
            CPPUNIT_ASSERT_EQUAL(inactivePointCount(inputTree, /*inCoreOnly=*/true), Index64(1));
            CPPUNIT_ASSERT_EQUAL(groupPointCount(inputTree, "test", /*inCoreOnly=*/true), Index64(2));
            CPPUNIT_ASSERT_EQUAL(activeGroupPointCount(inputTree, "test", /*inCoreOnly=*/true), Index64(1));
            CPPUNIT_ASSERT_EQUAL(inactiveGroupPointCount(inputTree, "test", /*inCoreOnly=*/true), Index64(1));
#endif

            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, /*inCoreOnly=*/false), Index64(4));
            CPPUNIT_ASSERT_EQUAL(activePointCount(inputTree, /*inCoreOnly=*/false), Index64(3));
            CPPUNIT_ASSERT_EQUAL(inactivePointCount(inputTree, /*inCoreOnly=*/false), Index64(1));
            CPPUNIT_ASSERT_EQUAL(groupPointCount(inputTree, "test", /*inCoreOnly=*/false), Index64(2));
            CPPUNIT_ASSERT_EQUAL(activeGroupPointCount(inputTree, "test", /*inCoreOnly=*/false), Index64(1));
            CPPUNIT_ASSERT_EQUAL(inactiveGroupPointCount(inputTree, "test", /*inCoreOnly=*/false), Index64(1));
        }

        // update the value mask and confirm point counts once again

        leafIter->updateValueMask();

        CPPUNIT_ASSERT_NO_THROW(leafIter->validateOffsets());

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(activeGroupPointCount(tree, "test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(inactiveGroupPointCount(tree, "test"), Index64(0));

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(activePointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(inactivePointCount(tree), Index64(0));
    }

    // create a tree with multiple leaves

    positions.emplace_back(20, 1, 1);
    positions.emplace_back(1, 20, 1);
    positions.emplace_back(1, 1, 20);

    grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree2 = grid->tree();

    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), Index32(4));

    leafIter = tree2.beginLeaf();

    appendGroup(tree2, "test");

    { // assign two points to the group
        const Descriptor::GroupIndex index = leafIter->attributeSet().groupIndex("test");

        CPPUNIT_ASSERT(index.first != AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(index.first < leafIter->attributeSet().size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        CPPUNIT_ASSERT(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);
        groupArray.set(3, GroupType(1) << index.second);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree2, "test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree2), Index64(7));
    }

    ++leafIter;

    CPPUNIT_ASSERT(leafIter);

    { // assign another point to the group in a different leaf
        const Descriptor::GroupIndex index = leafIter->attributeSet().groupIndex("test");

        CPPUNIT_ASSERT(index.first != AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(index.first < leafIter->attributeSet().size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        CPPUNIT_ASSERT(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree2, "test"), Index64(3));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree2), Index64(7));
    }
}


void
TestPointCount::testOffsets()
{
    using namespace openvdb::math;

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    // five points across four leafs

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 101, 1}, {2, 101, 1}, {101, 1, 1}, {101, 101, 1}};

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    { // all point offsets
        std::vector<Index64> pointOffsets;
        Index64 total = getPointOffsets(pointOffsets, tree);

        CPPUNIT_ASSERT_EQUAL(pointOffsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
    }

    { // all point offsets when using a non-existant exclude group

        std::vector<Index64> pointOffsets;

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups{"empty"};

        Index64 total = getPointOffsets(pointOffsets, tree, includeGroups, excludeGroups);

        CPPUNIT_ASSERT_EQUAL(pointOffsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
    }

    appendGroup(tree, "test");

    // add one point to the group from the leaf that contains two points

    PointDataTree::LeafIter iter = ++tree.beginLeaf();
    GroupWriteHandle groupHandle = iter->groupWriteHandle("test");
    groupHandle.set(0, true);

    { // include this group
        std::vector<Index64> pointOffsets;

        std::vector<Name> includeGroups{"test"};
        std::vector<Name> excludeGroups;

        Index64 total = getPointOffsets(pointOffsets, tree, includeGroups, excludeGroups);

        CPPUNIT_ASSERT_EQUAL(pointOffsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[0], Index64(0));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[1], Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[2], Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[3], Index64(1));
        CPPUNIT_ASSERT_EQUAL(total, Index64(1));
    }

    { // exclude this group
        std::vector<Index64> pointOffsets;

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups{"test"};

        Index64 total = getPointOffsets(pointOffsets, tree, includeGroups, excludeGroups);

        CPPUNIT_ASSERT_EQUAL(pointOffsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[1], Index64(2));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[2], Index64(3));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[3], Index64(4));
        CPPUNIT_ASSERT_EQUAL(total, Index64(4));
    }

    // setup temp directory

    std::string tempDir;
    if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _MSC_VER
    if (tempDir.empty()) {
        char tempDirBuffer[MAX_PATH+1];
        int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
        CPPUNIT_ASSERT(tempDirLen > 0 && tempDirLen <= MAX_PATH);
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

    // test point offsets for a delay-loaded grid
    {
        io::File fileIn(filename);
        fileIn.open();

        GridPtrVecPtr grids = fileIn.getGrids();

        fileIn.close();

        CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

        PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);

        CPPUNIT_ASSERT(inputGrid);

        PointDataTree& inputTree = inputGrid->tree();

        std::vector<Index64> pointOffsets;
        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups;

        Index64 total = getPointOffsets(pointOffsets, inputTree, includeGroups, excludeGroups, /*inCoreOnly=*/true);

#if OPENVDB_ABI_VERSION_NUMBER >= 3
        CPPUNIT_ASSERT_EQUAL(pointOffsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[0], Index64(0));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[1], Index64(0));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[2], Index64(0));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[3], Index64(0));
        CPPUNIT_ASSERT_EQUAL(total, Index64(0));
#else
        CPPUNIT_ASSERT_EQUAL(pointOffsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
#endif

        pointOffsets.clear();

        total = getPointOffsets(pointOffsets, inputTree, includeGroups, excludeGroups, /*inCoreOnly=*/false);

        CPPUNIT_ASSERT_EQUAL(pointOffsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointOffsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
    }
    std::remove(filename.c_str());
}


CPPUNIT_TEST_SUITE_REGISTRATION(TestPointCount);

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
