///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
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

#include "TestHarness.h"

#include <openvdb_ax/compiler/PointExecutable.h>

#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointGroup.h>

#include <cppunit/extensions/HelperMacros.h>

class TestVDBFunctions : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestVDBFunctions);
    CPPUNIT_TEST(addremovefromgroup);
    CPPUNIT_TEST(deletepoint);
    CPPUNIT_TEST(getcoord);
    CPPUNIT_TEST(getvoxelpws);
    CPPUNIT_TEST(ingroupOrder);
    CPPUNIT_TEST(ingroup);
    CPPUNIT_TEST_SUITE_END();

    void addremovefromgroup();
    void deletepoint();
    void getcoord();
    void getvoxelpws();
    void ingroupOrder();
    void ingroup();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVDBFunctions);

void
TestVDBFunctions::addremovefromgroup()
{
    const std::vector<openvdb::math::Vec3s> positions = {
        {1, 1, 1},
        {1, 2, 1},
        {2, 1, 1},
        {2, 2, 1},
    };

    const float voxelSize = 1.0f;
    const openvdb::math::Transform::ConstPtr transform =
        openvdb::math::Transform::createLinearTransform(voxelSize);
    const openvdb::points::PointAttributeVector<openvdb::math::Vec3s> pointList(positions);

    openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(
                pointList, *transform);

    openvdb::points::PointDataGrid::Ptr dataGrid =
        openvdb::points::createPointDataGrid<openvdb::points::NullCodec, openvdb::points::PointDataGrid>(
                *pointIndexGrid, pointList, *transform);

    openvdb::points::PointDataTree& dataTree = dataGrid->tree();

    // apppend a new attribute for stress testing

    openvdb::points::appendAttribute(dataTree, "existingTestAttribute", 2);
    openvdb::points::appendGroup(dataTree, "existingTestGroup");

    const std::vector<short> membershipTestGroup1{1, 0, 1, 0};
    openvdb::points::setGroup(dataTree, pointIndexGrid->tree(), membershipTestGroup1, "existingTestGroup");

    // second pre-existing group.
    openvdb::points::appendGroup(dataTree, "existingTestGroup2");
    openvdb::points::setGroup(dataTree, "existingTestGroup2", false);

    unittest_util::wrapExecution(*dataGrid, "test/snippets/vdb_functions/addremovefromgroup");

    auto leafIter = dataTree.cbeginLeaf();

    const openvdb::points::AttributeSet& attributeSet = leafIter->attributeSet();
    const openvdb::points::AttributeSet::Descriptor& desc = attributeSet.descriptor();

    for (size_t i = 1; i <= 9; i++) {
        const std::string groupName = "newTestGroup" + std::to_string(i);
        CPPUNIT_ASSERT_MESSAGE(groupName + " doesn't exist", desc.hasGroup(groupName));
    }

    openvdb::points::GroupHandle newTestGroupHandle = leafIter->groupHandle("newTestGroup9");
    CPPUNIT_ASSERT(!newTestGroupHandle.get(0));
    CPPUNIT_ASSERT(newTestGroupHandle.get(1));
    CPPUNIT_ASSERT(!newTestGroupHandle.get(2));
    CPPUNIT_ASSERT(newTestGroupHandle.get(3));

    // other new groups should be untouched
    for (size_t i = 1; i <= 8; i++) {
        openvdb::points::GroupHandle handle = leafIter->groupHandle("newTestGroup" + std::to_string(i));
        CPPUNIT_ASSERT(handle.get(0));
        CPPUNIT_ASSERT(handle.get(1));
        CPPUNIT_ASSERT(handle.get(2));
        CPPUNIT_ASSERT(handle.get(3));
    }

    openvdb::points::GroupHandle existingTestGroupHandle = leafIter->groupHandle("existingTestGroup");
    CPPUNIT_ASSERT(existingTestGroupHandle.get(0));
    CPPUNIT_ASSERT(!existingTestGroupHandle.get(1));
    CPPUNIT_ASSERT(existingTestGroupHandle.get(2));
    CPPUNIT_ASSERT(!existingTestGroupHandle.get(3));

    // membership of this group should now mirror exisingTestGroup
    openvdb::points::GroupHandle existingTestGroup2Handle = leafIter->groupHandle("existingTestGroup2");
    CPPUNIT_ASSERT(existingTestGroup2Handle.get(0));
    CPPUNIT_ASSERT(!existingTestGroup2Handle.get(1));
    CPPUNIT_ASSERT(existingTestGroup2Handle.get(2));
    CPPUNIT_ASSERT(!existingTestGroup2Handle.get(3));

    // check that "nonExistentGroup" was _not_ added to the tree, as it is removed from but not present
    CPPUNIT_ASSERT(!desc.hasGroup("nonExistentGroup"));

    // now check 2 new attributes added to tree
    openvdb::points::AttributeHandle<int> testResultAttributeHandle1(*attributeSet.get("newTestAttribute1"));
    openvdb::points::AttributeHandle<int> testResultAttributeHandle2(*attributeSet.get("newTestAttribute2"));
    for (openvdb::Index i = 0;i < 4; i++) {
        CPPUNIT_ASSERT(testResultAttributeHandle1.get(i));
    }

    // should match "existingTestGroup"
    CPPUNIT_ASSERT(testResultAttributeHandle2.get(0));
    CPPUNIT_ASSERT(!testResultAttributeHandle2.get(1));
    CPPUNIT_ASSERT(testResultAttributeHandle2.get(2));
    CPPUNIT_ASSERT(!testResultAttributeHandle2.get(3));

    // pre-existing attribute should still be present with the correct value

    for (; leafIter; ++leafIter) {
        openvdb::points::AttributeHandle<int>
            handle(leafIter->attributeArray("existingTestAttribute"));
        CPPUNIT_ASSERT(handle.isUniform());
        CPPUNIT_ASSERT_EQUAL(2, handle.get(0));
    }
}

void
TestVDBFunctions::deletepoint()
{
    // NOTE: the "deletepoint" function doesn't actually directly delete points - it adds them
    // to the "dead" group which marks them for deletion afterwards
    mHarness.testVolumes(false);

    mHarness.addInputGroups({"dead"}, {false});
    mHarness.addExpectedGroups({"dead"}, {true});

    mHarness.executeCode("test/snippets/vdb_functions/deletepoint");
    AXTESTS_STANDARD_ASSERT();

    // test without existing dead group

    mHarness.reset();
    mHarness.addExpectedGroups({"dead"}, {true});

    mHarness.executeCode("test/snippets/vdb_functions/deletepoint");
    AXTESTS_STANDARD_ASSERT();
}

void
TestVDBFunctions::getcoord()
{
    // create 3 test grids
    std::vector<openvdb::Int32Grid::Ptr> testGrids(3);
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(0.1);

    int i = 0;
    for (auto& grid : testGrids) {
        grid = openvdb::Int32Grid::create();
        grid->setTransform(transform);
        grid->setName("a" + std::to_string(i));
        openvdb::Int32Grid::Accessor accessor = grid->getAccessor();
        accessor.setValueOn(openvdb::Coord(1, 2, 3), 0);
        accessor.setValueOn(openvdb::Coord(1, 10, 3), 0);
        accessor.setValueOn(openvdb::Coord(-1, 1, 10), 0);
        ++i;
    }

    // convert to GridBase::Ptr to call wrapExecution
    openvdb::GridPtrVec testGridsBase(3);

    std::copy(testGrids.begin(), testGrids.end(), testGridsBase.begin());

    unittest_util::wrapExecution(testGridsBase, "test/snippets/vdb_functions/getcoord");

    // each grid has 3 active voxels.  These vectors hold the expected values of those voxels
    // for each grid
    std::vector<openvdb::Vec3I> expectedVoxelVals(3);
    expectedVoxelVals[0] = openvdb::Vec3I(1, 1, -1);
    expectedVoxelVals[1] = openvdb::Vec3I(2, 10, 1);
    expectedVoxelVals[2] = openvdb::Vec3I(3, 3, 10);

    std::vector<openvdb::Int32Grid::Ptr> expectedGrids(3);

    for (size_t i = 0; i < 3; i++) {
        openvdb::Int32Grid::Ptr grid = openvdb::Int32Grid::create();
        grid->setTransform(transform);
        grid->setName("a" + std::to_string(i) + "_expected");

        openvdb::Int32Grid::Accessor accessor = grid->getAccessor();
        const openvdb::Vec3I& expectedVals = expectedVoxelVals[i];

        accessor.setValueOn(openvdb::Coord(1, 2 ,3), expectedVals[0]);
        accessor.setValueOn(openvdb::Coord(1, 10, 3), expectedVals[1]);
        accessor.setValueOn(openvdb::Coord(-1, 1, 10), expectedVals[2]);

        expectedGrids[i] = grid;
    }

    // check grids
    bool check = true;
    std::stringstream outMessage;
    for (size_t i = 0; i < 3; i++){
        std::stringstream stream;
        unittest_util::ComparisonSettings settings;
        unittest_util::ComparisonResult result(stream);

        check &= unittest_util::compareGrids(result, *testGrids[i], *expectedGrids[i], settings, nullptr);

        if (!check) outMessage << stream.str() << std::endl;
    }

    CPPUNIT_ASSERT_MESSAGE(outMessage.str(), check);
}

void
TestVDBFunctions::getvoxelpws()
{
    mHarness.testPoints(false);

    mHarness.addAttribute<openvdb::Vec3f>("a", openvdb::Vec3f(10.0f), openvdb::Vec3f(0.0f));
    mHarness.executeCode("test/snippets/vdb_functions/getvoxelpws");
    AXTESTS_STANDARD_ASSERT();
}

void
TestVDBFunctions::ingroupOrder()
{
    // Test that groups inserted in a different alphabetical order are inferred
    // correctly (a regression test for a previous issue)
    mHarness.testVolumes(false);

    mHarness.addExpectedAttributes<int>({"test", "groupTest", "groupTest2"}, {1,1,1});
    mHarness.addInputGroups({"b", "a"}, {false, true});
    mHarness.addExpectedGroups({"b", "a"}, {false, true});

    mHarness.executeCode("test/snippets/vdb_functions/ingroup", nullptr, nullptr, true);
    AXTESTS_STANDARD_ASSERT();
}

void
TestVDBFunctions::ingroup()
{
    // test a tree with no groups
    CPPUNIT_ASSERT(mHarness.mInputPointGrids.size() > 0);
    openvdb::points::PointDataGrid::Ptr pointDataGrid1 = mHarness.mInputPointGrids.back();
    openvdb::points::PointDataTree& pointTree = pointDataGrid1->tree();

    // compile and execute

    openvdb::ax::Compiler compiler;
    openvdb::ax::CustomData::Ptr customData = openvdb::ax::CustomData::create();
    std::string code = unittest_util::loadText("test/snippets/vdb_functions/ingroup");
    openvdb::ax::PointExecutable::Ptr executable =
        compiler.compile<openvdb::ax::PointExecutable>(code, customData);

    CPPUNIT_ASSERT_NO_THROW(executable->execute(*pointDataGrid1));

    // the snippet of code adds "groupTest" and groupTest2 attributes which should both have the values
    // "1" everywhere

    for (auto leafIter = pointTree.cbeginLeaf(); leafIter; ++leafIter) {
        openvdb::points::AttributeHandle<int> handle1(leafIter->attributeArray("groupTest"));
        openvdb::points::AttributeHandle<int> handle2(leafIter->attributeArray("groupTest2"));
        for (auto iter = leafIter->beginIndexAll(); iter; ++iter) {
            CPPUNIT_ASSERT_EQUAL(1, handle1.get(*iter));
            CPPUNIT_ASSERT_EQUAL(1, handle2.get(*iter));
        }
    }

    // there should be no groups - ensure none have been added by accident by query code
    auto leafIter = pointTree.cbeginLeaf();
    const openvdb::points::AttributeSet& attributeSet = leafIter->attributeSet();
    const openvdb::points::AttributeSet::Descriptor& descriptor1 = attributeSet.descriptor();
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(0), descriptor1.groupMap().size());

    // now we add a single group and run the test again
    openvdb::points::appendGroup(pointTree, "testGroup");
    setGroup(pointTree, "testGroup", false);

    executable = compiler.compile<openvdb::ax::PointExecutable>(code, customData);
    CPPUNIT_ASSERT_NO_THROW(executable->execute(*pointDataGrid1));

    for (auto leafIter = pointTree.cbeginLeaf(); leafIter; ++leafIter) {
        openvdb::points::AttributeHandle<int> handle1(leafIter->attributeArray("groupTest"));
        openvdb::points::AttributeHandle<int> handle2(leafIter->attributeArray("groupTest2"));
        for (auto iter = leafIter->beginIndexAll(); iter; ++iter) {
            CPPUNIT_ASSERT_EQUAL(1, handle1.get(*iter));
            CPPUNIT_ASSERT_EQUAL(1, handle2.get(*iter));
        }
    }

    // for the next couple of tests we create a small tree with 4 points.  We wish to test queries of a single group
    // in a tree that has several groups
    const std::vector<openvdb::math::Vec3s> positions = {
        {1, 1, 1},
        {1, 2, 1},
        {2, 1, 1},
        {2, 2, 1},
    };

    const float voxelSize = 1.0f;
    const openvdb::math::Transform::ConstPtr transform =
        openvdb::math::Transform::createLinearTransform(voxelSize);
    const openvdb::points::PointAttributeVector<openvdb::math::Vec3s> pointList(positions);

    openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>
                (pointList, *transform);

    openvdb::points::PointDataGrid::Ptr pointDataGrid2 =
        openvdb::points::createPointDataGrid<openvdb::points::NullCodec, openvdb::points::PointDataGrid>
            (*pointIndexGrid, pointList, *transform);

    openvdb::points::PointDataTree::Ptr pointDataTree2 = pointDataGrid2->treePtr();

    // add 9 groups.  8 groups can be added by using a single group attribute, but this requires adding another attribute
    // and hence exercises the code better
    for (size_t i = 0; i < 9; i++) {
        openvdb::points::appendGroup(*pointDataTree2, "testGroup" + std::to_string(i));
    }
    std::vector<short> membershipTestGroup2{0, 0, 1, 0};
    openvdb::points::setGroup(*pointDataTree2, pointIndexGrid->tree(), membershipTestGroup2, "testGroup2");

    customData->reset();
    executable = compiler.compile<openvdb::ax::PointExecutable>(code, customData);
    CPPUNIT_ASSERT_NO_THROW(executable->execute(*pointDataGrid2));

    auto leafIter2 = pointDataTree2->cbeginLeaf();
    const openvdb::points::AttributeSet& attributeSet2 = leafIter2->attributeSet();
    openvdb::points::AttributeHandle<int> testResultAttributeHandle(*attributeSet2.get("groupTest2"));

    // these should line up with the defined membership
    CPPUNIT_ASSERT_EQUAL(testResultAttributeHandle.get(0), 1);
    CPPUNIT_ASSERT_EQUAL(testResultAttributeHandle.get(1), 1);
    CPPUNIT_ASSERT_EQUAL(testResultAttributeHandle.get(2), 2);
    CPPUNIT_ASSERT_EQUAL(testResultAttributeHandle.get(3), 1);

    // check that no new groups have been created or deleted
    const openvdb::points::AttributeSet::Descriptor& descriptor2 = attributeSet2.descriptor();
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(9), descriptor2.groupMap().size());

    for (size_t i = 0; i < 9; i++) {
        CPPUNIT_ASSERT(descriptor2.hasGroup("testGroup" + std::to_string(i)));
    }
}


// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
