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
#include <openvdb_points/tools/PointGroup.h>
#include <openvdb_points/tools/PointCount.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/openvdb.h>

#include <iostream>
#include <sstream>

using namespace openvdb;
using namespace openvdb::tools;

class TestPointGroup: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointGroup);
    CPPUNIT_TEST(testDescriptor);
    CPPUNIT_TEST(testAppendDrop);
    CPPUNIT_TEST(testCompact);
    CPPUNIT_TEST(testSet);
    CPPUNIT_TEST(testFilter);

    CPPUNIT_TEST_SUITE_END();

    void testDescriptor();
    void testAppendDrop();
    void testCompact();
    void testSet();
    void testFilter();
}; // class TestPointGroup

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointGroup);

////////////////////////////////////////


class FirstFilter
{
public:
    struct Data { };

    FirstFilter() { }

    template <typename LeafT>
    static FirstFilter create(const LeafT&, const Data&) {
        return FirstFilter();
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
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


void
TestPointGroup::testDescriptor()
{
    // test missing groups deletion

    { // no groups, empty Descriptor
        std::vector<std::string> groups;
        AttributeSet::Descriptor descriptor;
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups));
    }

    { // one group, empty Descriptor
        std::vector<std::string> groups;
        groups.push_back("group1");
        AttributeSet::Descriptor descriptor;
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups));
    }

    { // one group, Descriptor with same group
        std::vector<std::string> groups;
        groups.push_back("group1");
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group1", 0);
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups, "group1"));
    }

    { // one group, Descriptor with different group
        std::vector<std::string> groups;
        groups.push_back("group1");
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group2", 0);
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups));
    }

    { // three groups, Descriptor with three groups, one different
        std::vector<std::string> groups;
        groups.push_back("group1");
        groups.push_back("group3");
        groups.push_back("group4");
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group1", 0);
        descriptor.setGroup("group2", 0);
        descriptor.setGroup("group4", 0);
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups, "group1", "group4"));
    }
}


////////////////////////////////////////


void
TestPointGroup::testAppendDrop()
{
	typedef TypedAttributeArray<Vec3s>   AttributeVec3s;

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(1, 10, 1));
    positions.push_back(Vec3s(10, 1, 1));
    positions.push_back(Vec3s(10, 10, 1));

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    ++leafIter;
    ++leafIter;
    ++leafIter;

    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    { // throw on append or drop an empty group
        CPPUNIT_ASSERT_THROW(appendGroup(tree, ""), openvdb::KeyError);
        CPPUNIT_ASSERT_THROW(dropGroup(tree, ""), openvdb::KeyError);
    }

    { // append a group
        appendGroup(tree, "test");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(1));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
    }

    { // append a group with non-unique name (repeat the append)
        appendGroup(tree, "test");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(1));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
    }

    { // append multiple groups
        std::vector<Name> names;
        names.push_back("test2");
        names.push_back("test3");

        appendGroups(tree, names);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(3));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test2"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test2"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test3"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test3"));
    }

    { // append to a copy
        PointDataTree tree2(tree);

        appendGroup(tree2, "copy1");

        CPPUNIT_ASSERT(!attributeSet.descriptor().hasGroup("copy1"));
        CPPUNIT_ASSERT(tree2.beginLeaf()->attributeSet().descriptor().hasGroup("copy1"));
    }

    { // drop a group
        dropGroup(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(2));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test3"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test3"));
    }

    { // drop multiple groups
        std::vector<Name> names;
        names.push_back("test");
        names.push_back("test3");

        dropGroups(tree, names);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(0));
    }

    { // drop a copy
        appendGroup(tree, "copy2");

        PointDataTree tree2(tree);

        dropGroup(tree2, "copy2");

        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("copy2"));
        CPPUNIT_ASSERT(!tree2.beginLeaf()->attributeSet().descriptor().hasGroup("copy2"));

        dropGroup(tree, "copy2");
    }

    { // set group membership
        appendGroup(tree, "test");

        setGroup(tree, "test", true);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(4));

        setGroup(tree, "test", false);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(0));

        dropGroup(tree, "test");
    }

    { // drop all groups
        appendGroup(tree, "test");
        appendGroup(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(1));

        dropGroups(tree);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(0));
    }
}


void
TestPointGroup::testCompact()
{
    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(1));

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

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(8));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(1));

        appendGroup(tree, "test8");

        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test0"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test7"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test8"));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(9));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(2));
    }

    { // drop first attribute then compact
        dropGroup(tree, "test5", /*compact=*/false);

        CPPUNIT_ASSERT(!attributeSet.descriptor().hasGroup("test5"));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(8));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(2));

        compactGroups(tree);

        CPPUNIT_ASSERT(!attributeSet.descriptor().hasGroup("test5"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test7"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test8"));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(8));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(1));
    }

    { // append seventeen groups, drop most of them, then compact
        for (int i = 0; i < 17; i++) {
            ss.str("");
            ss << "test" << i;
            appendGroup(tree, ss.str());
        }

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(17));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(3));

        // delete all but 0, 5, 9, 15

        for (int i = 0; i < 17; i++) {
            if (i == 0 || i == 5 || i == 9 || i == 15)  continue;
            ss.str("");
            ss << "test" << i;
            dropGroup(tree, ss.str(), /*compact=*/false);
        }

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(3));

        // make a copy

        PointDataTree tree2(tree);

        // compact - should now occupy one attribute

        compactGroups(tree);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(1));

        // check descriptor has been deep copied

        CPPUNIT_ASSERT_EQUAL(tree2.cbeginLeaf()->attributeSet().descriptor().groupMap().size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(tree2.cbeginLeaf()->attributeSet().size(AttributeArray::GROUP), size_t(3));
    }
}


void
TestPointGroup::testSet()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;

    typedef PointIndexGrid PointIndexGrid;

    // four points in the same leaf

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(1, 2, 1));
    positions.push_back(Vec3s(2, 1, 1));
    positions.push_back(Vec3s(2, 2, 1));
    positions.push_back(Vec3s(100, 100, 100));
    positions.push_back(Vec3s(100, 101, 100));

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    const PointAttributeVector<Vec3s> pointList(positions);

    PointIndexGrid::Ptr pointIndexGrid =
        openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(*pointIndexGrid, pointList,
                                                                 AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    appendGroup(tree, "test");

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
    CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(0));

    std::vector<short> membership;
    membership.push_back(short(1));
    membership.push_back(short(0));
    membership.push_back(short(1));
    membership.push_back(short(1));
    membership.push_back(short(0));
    membership.push_back(short(1));

    // copy tree for descriptor sharing test

    PointDataTree tree2(tree);

    setGroup(tree, pointIndexGrid->tree(), membership, "test");

    // check that descriptor remains shared

    appendGroup(tree2, "copy1");

    CPPUNIT_ASSERT(!tree.cbeginLeaf()->attributeSet().descriptor().hasGroup("copy1"));

    dropGroup(tree2, "copy1");

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
    CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(4));

    { // IO
        // setup temp directory

        std::string tempDir(std::getenv("TMPDIR"));
        if (tempDir.empty())    tempDir = P_tmpdir;

        std::string filename;

        // write out grid to a temp file
        {
            filename = tempDir + "/openvdb_test_point_load";

            io::File fileOut(filename);

            GridCPtrVec grids;
            grids.push_back(grid);

            fileOut.write(grids);
        }

        // read test groups
        {
            io::File fileIn(filename);
            fileIn.open();

            GridPtrVecPtr grids = fileIn.getGrids();

            fileIn.close();

            CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

            PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);
            PointDataTree& tree = inputGrid->tree();

            CPPUNIT_ASSERT(tree.cbeginLeaf());

            const PointDataGrid::TreeType::LeafNodeType& leaf = *tree.cbeginLeaf();

            const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();

            CPPUNIT_ASSERT(descriptor.hasGroup("test"));
            CPPUNIT_ASSERT_EQUAL(descriptor.groupMap().size(), size_t(1));

            CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
            CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "test"), Index64(4));
        }
    }
}


void
TestPointGroup::testFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;

    typedef PointIndexGrid PointIndexGrid;

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));
    PointDataGrid::Ptr grid;

    { // four points in the same leaf
        std::vector<Vec3s> positions;
        positions.push_back(Vec3s(1, 1, 1));
        positions.push_back(Vec3s(1, 2, 1));
        positions.push_back(Vec3s(2, 1, 1));
        positions.push_back(Vec3s(2, 2, 1));
        positions.push_back(Vec3s(100, 100, 100));
        positions.push_back(Vec3s(100, 101, 100));

        const PointAttributeVector<Vec3s> pointList(positions);

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);

        grid = createPointDataGrid<PointDataGrid>(  *pointIndexGrid, pointList,
                                                    AttributeVec3s::attributeType(), *transform);
    }

    PointDataTree& tree = grid->tree();

    { // first point filter
        appendGroup(tree, "first");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "first"), Index64(0));

        FirstFilter::Data data;

        setGroupByFilter<PointDataTree, FirstFilter>(tree, "first", data);

        PointDataTree::LeafCIter iter = tree.cbeginLeaf();

        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT_EQUAL(iter->groupPointCount("first"), Index64(1));
        }

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "first"), Index64(2));
    }

    const openvdb::BBoxd bbox(openvdb::Vec3d(0, 1.5, 0), openvdb::Vec3d(101, 100.5, 101));

    { // bbox filter
        appendGroup(tree, "bbox");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "bbox"), Index64(0));

        BBoxFilter::Data data(*transform, bbox);

        setGroupByFilter<PointDataTree, BBoxFilter>(tree, "bbox", data);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "bbox"), Index64(3));
    }

    { // first point filter and bbox filter (intersection of the above two filters)
        appendGroup(tree, "first_bbox");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "first_bbox"), Index64(0));

        typedef BinaryFilter<FirstFilter, BBoxFilter>   FirstBBoxFilter;

        FirstFilter::Data firstData;
        BBoxFilter::Data bboxData(*transform, bbox);
        FirstBBoxFilter::Data data(firstData, bboxData);

        setGroupByFilter<PointDataTree, FirstBBoxFilter>(tree, "first_bbox", data);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(tree, "first_bbox"), Index64(1));

        std::vector<Vec3f> positions;

        for (PointDataTree::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
            typedef PointDataTree::LeafNodeType::IndexOnIter IndexOnIter;
            GroupFilter filter(GroupFilter::create(*iter, GroupFilter::Data("first_bbox")));
            FilterIndexIter<IndexOnIter, GroupFilter> filterIndexIter(iter->beginIndexOn(), filter);

            AttributeHandle<Vec3f>::Ptr handle = AttributeHandle<Vec3f>::create(iter->attributeArray("P"));

            for ( ; filterIndexIter; ++filterIndexIter) {
                const openvdb::Coord ijk = filterIndexIter.indexIter().getCoord();
                positions.push_back(handle->get(*filterIndexIter) + ijk.asVec3d());
            }
        }

        CPPUNIT_ASSERT_EQUAL(positions.size(), size_t(1));
        CPPUNIT_ASSERT_EQUAL(positions[0], Vec3f(100, 100, 100));
    }

    { // add 1000 points in three leafs (positions aren't important)

        std::vector<Vec3s> positions;
        for (int i = 0; i < 1000; i++) {
            positions.push_back(openvdb::Vec3f(1, 1, 1));
        }
        for (int i = 0; i < 1000; i++) {
            positions.push_back(openvdb::Vec3f(1, 1, 9));
        }
        for (int i = 0; i < 1000; i++) {
            positions.push_back(openvdb::Vec3f(9, 9, 9));
        }

        const PointAttributeVector<Vec3s> pointList(positions);

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);

        grid = createPointDataGrid<PointDataGrid>(  *pointIndexGrid, pointList,
                                                    AttributeVec3s::attributeType(), *transform);

        PointDataTree& newTree = grid->tree();

        CPPUNIT_ASSERT_EQUAL(pointCount(newTree), Index64(3000));

        // random - maximum

        appendGroup(newTree, "random_maximum");

        const Index64 target = 1001;

        setGroupByRandomTarget(newTree, "random_maximum", target);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(newTree, "random_maximum"), target);

        // random - percentage

        appendGroup(newTree, "random_percentage");

        setGroupByRandomPercentage(newTree, "random_percentage", 33.333333f);

        CPPUNIT_ASSERT_EQUAL(groupPointCount(newTree, "random_percentage"), Index64(1000));
    }
}


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
