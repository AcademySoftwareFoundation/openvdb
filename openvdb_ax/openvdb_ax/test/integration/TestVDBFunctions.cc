// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"
#include "../util.h"

#include <openvdb_ax/ax.h>
#include <openvdb_ax/codegen/Types.h>
#include <openvdb_ax/codegen/Functions.h>
#include <openvdb_ax/codegen/FunctionRegistry.h>
#include <openvdb_ax/codegen/FunctionTypes.h>
#include <openvdb_ax/compiler/PointExecutable.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>

#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointGroup.h>


class TestVDBFunctions : public unittest_util::AXTestCase
{
};


TEST_F(TestVDBFunctions, addremovefromgroup)
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

    const std::string code = unittest_util::loadText("test/snippets/vdb_functions/addremovefromgroup");
    openvdb::ax::run(code.c_str(), *dataGrid);

    auto leafIter = dataTree.cbeginLeaf();

    const openvdb::points::AttributeSet& attributeSet = leafIter->attributeSet();
    const openvdb::points::AttributeSet::Descriptor& desc = attributeSet.descriptor();

    for (size_t i = 1; i <= 9; i++) {
        const std::string groupName = "newTestGroup" + std::to_string(i);
        ASSERT_TRUE(desc.hasGroup(groupName)) << (groupName + " doesn't exist");
    }

    openvdb::points::GroupHandle newTestGroupHandle = leafIter->groupHandle("newTestGroup9");
    ASSERT_TRUE(!newTestGroupHandle.get(0));
    ASSERT_TRUE(newTestGroupHandle.get(1));
    ASSERT_TRUE(!newTestGroupHandle.get(2));
    ASSERT_TRUE(newTestGroupHandle.get(3));

    // other new groups should be untouched
    for (size_t i = 1; i <= 8; i++) {
        openvdb::points::GroupHandle handle = leafIter->groupHandle("newTestGroup" + std::to_string(i));
        ASSERT_TRUE(handle.get(0));
        ASSERT_TRUE(handle.get(1));
        ASSERT_TRUE(handle.get(2));
        ASSERT_TRUE(handle.get(3));
    }

    openvdb::points::GroupHandle existingTestGroupHandle = leafIter->groupHandle("existingTestGroup");
    ASSERT_TRUE(existingTestGroupHandle.get(0));
    ASSERT_TRUE(!existingTestGroupHandle.get(1));
    ASSERT_TRUE(existingTestGroupHandle.get(2));
    ASSERT_TRUE(!existingTestGroupHandle.get(3));

    // membership of this group should now mirror exisingTestGroup
    openvdb::points::GroupHandle existingTestGroup2Handle = leafIter->groupHandle("existingTestGroup2");
    ASSERT_TRUE(existingTestGroup2Handle.get(0));
    ASSERT_TRUE(!existingTestGroup2Handle.get(1));
    ASSERT_TRUE(existingTestGroup2Handle.get(2));
    ASSERT_TRUE(!existingTestGroup2Handle.get(3));

    // check that "nonExistentGroup" was _not_ added to the tree, as it is removed from but not present
    ASSERT_TRUE(!desc.hasGroup("nonExistentGroup"));

    // now check 2 new attributes added to tree
    openvdb::points::AttributeHandle<int> testResultAttributeHandle1(*attributeSet.get("newTestAttribute1"));
    openvdb::points::AttributeHandle<int> testResultAttributeHandle2(*attributeSet.get("newTestAttribute2"));
    for (openvdb::Index i = 0;i < 4; i++) {
        ASSERT_TRUE(testResultAttributeHandle1.get(i));
    }

    // should match "existingTestGroup"
    ASSERT_TRUE(testResultAttributeHandle2.get(0));
    ASSERT_TRUE(!testResultAttributeHandle2.get(1));
    ASSERT_TRUE(testResultAttributeHandle2.get(2));
    ASSERT_TRUE(!testResultAttributeHandle2.get(3));

    // pre-existing attribute should still be present with the correct value

    for (; leafIter; ++leafIter) {
        openvdb::points::AttributeHandle<int>
            handle(leafIter->attributeArray("existingTestAttribute"));
        ASSERT_TRUE(handle.isUniform());
        ASSERT_EQ(2, handle.get(0));
    }
}

TEST_F(TestVDBFunctions, deletepoint)
{
    // run first, should not modify grid as attribute doesn't exist
    // @todo - need to massively improve this test

    mHarness.testVolumes(false);
    mHarness.addAttribute<int>("delete", 0, 0);
    mHarness.executeCode("test/snippets/vdb_functions/deletepoint");
    AXTESTS_STANDARD_ASSERT();

    mHarness.reset();
    mHarness.addInputAttribute<int>("delete", 1);
    for (auto& grid : mHarness.mOutputPointGrids) {
        grid->clear();
    }
    mHarness.executeCode("test/snippets/vdb_functions/deletepoint");
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestVDBFunctions, getcoord)
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

    // convert to GridBase::Ptr
    openvdb::GridPtrVec testGridsBase(3);
    std::copy(testGrids.begin(), testGrids.end(), testGridsBase.begin());

    const std::string code = unittest_util::loadText("test/snippets/vdb_functions/getcoord");
    openvdb::ax::run(code.c_str(), testGridsBase);

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

    ASSERT_TRUE(check) << outMessage.str();
}

TEST_F(TestVDBFunctions, getvoxelpws)
{
    mHarness.testPoints(false);
    mHarness.testSparseVolumes(false); // disable as getvoxelpws will densify
    mHarness.testDenseVolumes(true);

    mHarness.addAttribute<openvdb::Vec3f>("a", openvdb::Vec3f(10.0f), openvdb::Vec3f(0.0f));
    mHarness.executeCode("test/snippets/vdb_functions/getvoxelpws");
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestVDBFunctions, ingroupOrder)
{
    // Test that groups inserted in a different alphabetical order are inferred
    // correctly (a regression test for a previous issue)
    mHarness.testVolumes(false);

    mHarness.addExpectedAttributes<int>({"test", "groupTest", "groupTest2"}, {1,1,1});
    mHarness.addInputGroups({"b", "a"}, {false, true});
    mHarness.addExpectedGroups({"b", "a"}, {false, true});

    mHarness.executeCode("test/snippets/vdb_functions/ingroup", nullptr, true);
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestVDBFunctions, ingroup)
{
    // test a tree with no groups
    ASSERT_TRUE(mHarness.mInputPointGrids.size() > 0);
    openvdb::points::PointDataGrid::Ptr pointDataGrid1 = mHarness.mInputPointGrids.back();
    openvdb::points::PointDataTree& pointTree = pointDataGrid1->tree();

    // compile and execute

    openvdb::ax::Compiler compiler;
    std::string code = unittest_util::loadText("test/snippets/vdb_functions/ingroup");
    openvdb::ax::PointExecutable::Ptr executable =
        compiler.compile<openvdb::ax::PointExecutable>(code);

    ASSERT_NO_THROW(executable->execute(*pointDataGrid1));

    // the snippet of code adds "groupTest" and groupTest2 attributes which should both have the values
    // "1" everywhere

    for (auto leafIter = pointTree.cbeginLeaf(); leafIter; ++leafIter) {
        openvdb::points::AttributeHandle<int> handle1(leafIter->attributeArray("groupTest"));
        openvdb::points::AttributeHandle<int> handle2(leafIter->attributeArray("groupTest2"));
        for (auto iter = leafIter->beginIndexAll(); iter; ++iter) {
            ASSERT_EQ(1, handle1.get(*iter));
            ASSERT_EQ(1, handle2.get(*iter));
        }
    }

    // there should be no groups - ensure none have been added by accident by query code
    auto leafIter = pointTree.cbeginLeaf();
    const openvdb::points::AttributeSet& attributeSet = leafIter->attributeSet();
    const openvdb::points::AttributeSet::Descriptor& descriptor1 = attributeSet.descriptor();
    ASSERT_EQ(static_cast<size_t>(0), descriptor1.groupMap().size());

    // now we add a single group and run the test again
    openvdb::points::appendGroup(pointTree, "testGroup");
    setGroup(pointTree, "testGroup", false);

    executable = compiler.compile<openvdb::ax::PointExecutable>(code);
    ASSERT_NO_THROW(executable->execute(*pointDataGrid1));

    for (auto leafIter = pointTree.cbeginLeaf(); leafIter; ++leafIter) {
        openvdb::points::AttributeHandle<int> handle1(leafIter->attributeArray("groupTest"));
        openvdb::points::AttributeHandle<int> handle2(leafIter->attributeArray("groupTest2"));
        for (auto iter = leafIter->beginIndexAll(); iter; ++iter) {
            ASSERT_EQ(1, handle1.get(*iter));
            ASSERT_EQ(1, handle2.get(*iter));
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

    executable = compiler.compile<openvdb::ax::PointExecutable>(code);
    ASSERT_NO_THROW(executable->execute(*pointDataGrid2));

    auto leafIter2 = pointDataTree2->cbeginLeaf();
    const openvdb::points::AttributeSet& attributeSet2 = leafIter2->attributeSet();
    openvdb::points::AttributeHandle<int> testResultAttributeHandle(*attributeSet2.get("groupTest2"));

    // these should line up with the defined membership
    ASSERT_EQ(testResultAttributeHandle.get(0), 1);
    ASSERT_EQ(testResultAttributeHandle.get(1), 1);
    ASSERT_EQ(testResultAttributeHandle.get(2), 2);
    ASSERT_EQ(testResultAttributeHandle.get(3), 1);

    // check that no new groups have been created or deleted
    const openvdb::points::AttributeSet::Descriptor& descriptor2 = attributeSet2.descriptor();
    ASSERT_EQ(static_cast<size_t>(9), descriptor2.groupMap().size());

    for (size_t i = 0; i < 9; i++) {
        ASSERT_TRUE(descriptor2.hasGroup("testGroup" + std::to_string(i)));
    }
}

TEST_F(TestVDBFunctions, testValidContext)
{
    std::shared_ptr<llvm::LLVMContext> C(new llvm::LLVMContext);
#if LLVM_VERSION_MAJOR >= 15
    // This will not work from LLVM 16. We'll need to fix this
    // https://llvm.org/docs/OpaquePointers.html
    C->setOpaquePointers(false);
#endif

    openvdb::ax::Compiler compiler;
    openvdb::ax::FunctionOptions ops;
    ops.mLazyFunctions = false;

    /// Generate code which calls the given function
    auto generate = [&C](const openvdb::ax::codegen::Function::Ptr F,
                         const std::string& name) -> std::string
    {
        std::vector<llvm::Type*> types;
        F->types(types, *C);

        std::string code;
        std::string args;
        size_t idx = 0;
        for (auto T : types) {
            const std::string axtype =
                openvdb::ax::ast::tokens::typeStringFromToken(
                    openvdb::ax::codegen::tokenFromLLVMType(T));
            code += axtype + " local" + std::to_string(idx) + ";\n";
            args += "local" + std::to_string(idx) + ",";
        }

        // remove last ","
        if (!args.empty()) args.pop_back();
        code += name + "(" + args + ");";
        return code;
    };


    /// Test Volumes fails when trying to call Point Functions
    {
        openvdb::ax::codegen::FunctionRegistry::UniquePtr
            registry(new openvdb::ax::codegen::FunctionRegistry);
        openvdb::ax::codegen::insertVDBPointFunctions(*registry, &ops);

        for (auto& func : registry->map()) {
            // Don't check internal functions
            if (func.second.isInternal()) continue;

            const openvdb::ax::codegen::FunctionGroup* const ptr = func.second.function();
            ASSERT_TRUE(ptr);
            const auto& signatures = ptr->list();
            ASSERT_TRUE(!signatures.empty());

            // Don't check C bindings
            const auto F = signatures.front();
            if (dynamic_cast<const openvdb::ax::codegen::CFunctionBase*>(F.get())) continue;

            const std::string code = generate(F, func.first);

            ASSERT_THROW(
                compiler.compile<openvdb::ax::VolumeExecutable>(code),
                openvdb::AXCompilerError
            ) << ERROR_MSG("Expected Compiler Error", code);
        }
    }

    /// Test Points fails when trying to call Volume Functions
    {
        openvdb::ax::codegen::FunctionRegistry::UniquePtr
            registry(new openvdb::ax::codegen::FunctionRegistry);
        openvdb::ax::codegen::insertVDBVolumeFunctions(*registry, &ops);

        for (auto& func : registry->map()) {
            // Don't check internal functions
            if (func.second.isInternal()) continue;

            const openvdb::ax::codegen::FunctionGroup* const ptr = func.second.function();
            ASSERT_TRUE(ptr);
            const auto& signatures = ptr->list();
            ASSERT_TRUE(!signatures.empty());

            // Don't check C bindings
            const auto F = signatures.front();
            if (dynamic_cast<const openvdb::ax::codegen::CFunctionBase*>(F.get())) continue;

            const std::string code = generate(F, func.first);

            ASSERT_THROW(
                compiler.compile<openvdb::ax::PointExecutable>(code),
                openvdb::AXCompilerError
            ) << ERROR_MSG("Expected Compiler Error", code);
        }
    }
}
