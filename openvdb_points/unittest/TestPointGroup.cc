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
    CPPUNIT_TEST(testAppendDrop);
    CPPUNIT_TEST(testCompact);
    CPPUNIT_TEST(testSet);

    CPPUNIT_TEST_SUITE_END();

    void testAppendDrop();
    void testCompact();
    void testSet();
}; // class TestPointGroup

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointGroup);

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

        // compact - should now occupy one attribute

        compactGroups(tree);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(attributeSet.size(AttributeArray::GROUP), size_t(1));
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

    std::vector<bool> membership;
    membership.push_back(true);
    membership.push_back(false);
    membership.push_back(true);
    membership.push_back(true);
    membership.push_back(false);
    membership.push_back(true);

    setGroup(tree, pointIndexGrid->tree(), membership, "test");

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


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
