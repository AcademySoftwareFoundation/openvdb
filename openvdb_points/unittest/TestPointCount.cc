///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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

#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/openvdb.h>
#include <openvdb/openvdb.h>

#include <openvdb_points/tools/PointGroup.h>
#include <openvdb_points/tools/PointCount.h>
#include <openvdb_points/tools/PointConversion.h>

class TestPointCount: public CppUnit::TestCase
{
public:

    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointCount);
    CPPUNIT_TEST(testCount);
    CPPUNIT_TEST(testGroup);
    CPPUNIT_TEST_SUITE_END();

    void testCount();
    void testGroup();

}; // class TestPointCount

using openvdb::tools::PointDataTree;
using openvdb::tools::PointDataGrid;
typedef PointDataTree::LeafNodeType     LeafType;
typedef LeafType::ValueType             ValueType;
using openvdb::Index;
using openvdb::Index64;


void
TestPointCount::testCount()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    // create a tree and check there are no points

    PointDataGrid::Ptr grid = createGrid<PointDataGrid>();
    PointDataTree& tree = grid->tree();

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(0));

    // add a new leaf to a tree and re-test

    LeafType* leafPtr = new LeafType(openvdb::Coord(0, 0, 0));
    LeafType& leaf(*leafPtr);

    // on adding, tree now obtains ownership and is reponsible for deletion

    tree.addLeaf(leaf);

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(0));

    // now manually set some offsets

    leaf.setOffsetOn(0, 4);
    leaf.setOffsetOn(1, 7);

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndex(0)), 0);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndex(0).end()), 4);

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndex(1)), 4);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndex(1).end()), 7);

    {
        IndexIter iter = leaf.beginIndex(openvdb::Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(int(*iter), 0);
        CPPUNIT_ASSERT_EQUAL(int(iter.end()), 4);

        IndexIter iter2 = leaf.beginIndex(openvdb::Coord(0, 0, 1));

        CPPUNIT_ASSERT_EQUAL(int(*iter2), 4);
        CPPUNIT_ASSERT_EQUAL(int(iter2.end()), 7);

        CPPUNIT_ASSERT_EQUAL(iterCount(iter2), Index64(7 - 4));

        // check pointCount ignores active/inactive state

        leaf.setValueOff(1);

        IndexIter iter3 = leaf.beginIndex(openvdb::Coord(0, 0, 1));

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

    LeafType* leaf2Ptr = new LeafType(openvdb::Coord(0, 0, 8));
    LeafType& leaf2(*leaf2Ptr);

    // on adding, tree now obtains ownership and is reponsible for deletion

    tree.addLeaf(leaf2);

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
    using namespace openvdb;
    using namespace openvdb::tools;
    using namespace openvdb::math;

    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;
    typedef AttributeSet::Descriptor   Descriptor;

    // four points in the same leaf

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(1, 2, 1));
    positions.push_back(Vec3s(2, 1, 1));
    positions.push_back(Vec3s(2, 2, 1));

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // setup temp directory

    std::string tempDir(std::getenv("TMPDIR"));
    if (tempDir.empty())    tempDir = P_tmpdir;

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

        CPPUNIT_ASSERT(GroupAttributeArray::isGroup(array));

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

            GridCPtrVec grids;
            grids.push_back(grid);

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

#ifndef OPENVDB_2_ABI_COMPATIBLE
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

    positions.push_back(Vec3s(20, 1, 1));
    positions.push_back(Vec3s(1, 20, 1));
    positions.push_back(Vec3s(1, 1, 20));

    grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree2 = grid->tree();

    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), Index32(4));

    leafIter = tree2.beginLeaf();

    appendGroup(tree2, "test");

    { // assign two points to the group
        const Descriptor::GroupIndex index = leafIter->attributeSet().groupIndex("test");

        CPPUNIT_ASSERT(index.first != AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(index.first < leafIter->attributeSet().size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        CPPUNIT_ASSERT(GroupAttributeArray::isGroup(array));

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

        CPPUNIT_ASSERT(GroupAttributeArray::isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree2, "test"), Index64(3));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree2), Index64(7));
    }
}

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointCount);


////////////////////////////////////////


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
