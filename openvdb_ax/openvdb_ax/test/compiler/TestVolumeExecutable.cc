// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "../compiler/Compiler.h"
#include "../compiler/VolumeExecutable.h"

#include <cppunit/extensions/HelperMacros.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>

class TestVolumeExecutable : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestVolumeExecutable);
    CPPUNIT_TEST(testConstructionDestruction);
    CPPUNIT_TEST(testCreateMissingGrids);
    CPPUNIT_TEST(testTreeExecutionLevel);
    CPPUNIT_TEST_SUITE_END();

    void testConstructionDestruction();
    void testCreateMissingGrids();
    void testTreeExecutionLevel();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVolumeExecutable);

void
TestVolumeExecutable::testConstructionDestruction()
{
    // Test the building and teardown of executable objects. This is primarily to test
    // the destruction of Context and ExecutionEngine LLVM objects. These must be destructed
    // in the correct order (ExecutionEngine, then Context) otherwise LLVM will crash

    // must be initialized, otherwise construction/destruction of llvm objects won't
    // exhibit correct behaviour

    CPPUNIT_ASSERT(openvdb::ax::isInitialized());

    std::shared_ptr<llvm::LLVMContext> C(new llvm::LLVMContext);
    std::unique_ptr<llvm::Module> M(new llvm::Module("test_module", *C));
    std::shared_ptr<const llvm::ExecutionEngine> E(llvm::EngineBuilder(std::move(M))
            .setEngineKind(llvm::EngineKind::JIT)
            .create());

    CPPUNIT_ASSERT(!M);
    CPPUNIT_ASSERT(E);

    std::weak_ptr<llvm::LLVMContext> wC = C;
    std::weak_ptr<const llvm::ExecutionEngine> wE = E;

    // Basic construction

    openvdb::ax::ast::Tree tree;
    openvdb::ax::AttributeRegistry::ConstPtr emptyReg =
        openvdb::ax::AttributeRegistry::create(tree);
    openvdb::ax::VolumeExecutable::Ptr volumeExecutable
        (new openvdb::ax::VolumeExecutable(C, E, emptyReg, nullptr, {}));

    CPPUNIT_ASSERT_EQUAL(2, int(wE.use_count()));
    CPPUNIT_ASSERT_EQUAL(2, int(wC.use_count()));

    C.reset();
    E.reset();

    CPPUNIT_ASSERT_EQUAL(1, int(wE.use_count()));
    CPPUNIT_ASSERT_EQUAL(1, int(wC.use_count()));

    // test destruction

    volumeExecutable.reset();

    CPPUNIT_ASSERT_EQUAL(0, int(wE.use_count()));
    CPPUNIT_ASSERT_EQUAL(0, int(wC.use_count()));
}

void
TestVolumeExecutable::testCreateMissingGrids()
{
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    openvdb::ax::VolumeExecutable::Ptr executable =
        compiler->compile<openvdb::ax::VolumeExecutable>("@a=v@b.x;");

    executable->setCreateMissing(false);
    executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::ON);

    openvdb::GridPtrVec grids;
    CPPUNIT_ASSERT_THROW(executable->execute(grids), openvdb::LookupError);
    CPPUNIT_ASSERT(grids.empty());

    executable->setCreateMissing(true);
    executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::ON);
    executable->execute(grids);

    openvdb::math::Transform::Ptr defaultTransform =
        openvdb::math::Transform::createLinearTransform();

    CPPUNIT_ASSERT_EQUAL(size_t(2), grids.size());
    CPPUNIT_ASSERT(grids[0]->getName() == "b");
    CPPUNIT_ASSERT(grids[0]->isType<openvdb::Vec3fGrid>());
    CPPUNIT_ASSERT(grids[0]->empty());
    CPPUNIT_ASSERT(grids[0]->transform() == *defaultTransform);

    CPPUNIT_ASSERT(grids[1]->getName() == "a");
    CPPUNIT_ASSERT(grids[1]->isType<openvdb::FloatGrid>());
    CPPUNIT_ASSERT(grids[1]->empty());
    CPPUNIT_ASSERT(grids[1]->transform() == *defaultTransform);
}

void
TestVolumeExecutable::testTreeExecutionLevel()
{
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    openvdb::ax::VolumeExecutable::Ptr executable =
        compiler->compile<openvdb::ax::VolumeExecutable>("f@test = 1.0f;");

    using NodeT0 = openvdb::FloatGrid::Accessor::NodeT0;
    using NodeT1 = openvdb::FloatGrid::Accessor::NodeT1;
    using NodeT2 = openvdb::FloatGrid::Accessor::NodeT2;

    openvdb::FloatGrid test;
    test.setName("test");

    // NodeT0 tile
    test.tree().addTile(1, openvdb::Coord(0), -2.0f, /*active*/true);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), test.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(1), test.tree().activeTileCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT0::NUM_VOXELS), test.tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(-2.0f, test.tree().getValue(openvdb::Coord(0)));

    // default is leaf nodes, expect no change
    executable->execute(test);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), test.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(1), test.tree().activeTileCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT0::NUM_VOXELS), test.tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(-2.0f, test.tree().getValue(openvdb::Coord(0)));

    executable->setTreeExecutionLevel(1);
    executable->execute(test);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), test.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(1), test.tree().activeTileCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT0::NUM_VOXELS), test.tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(1.0f, test.tree().getValue(openvdb::Coord(0)));

    // NodeT1 tile
    test.tree().addTile(2, openvdb::Coord(0), -2.0f, /*active*/true);
    // level is set to 1, expect no change
    executable->execute(test);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), test.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(1), test.tree().activeTileCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT1::NUM_VOXELS), test.tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(-2.0f, test.tree().getValue(openvdb::Coord(0)));

    executable->setTreeExecutionLevel(2);
    executable->execute(test);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), test.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(1), test.tree().activeTileCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT1::NUM_VOXELS), test.tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(1.0f, test.tree().getValue(openvdb::Coord(0)));

    // NodeT2 tile
    test.tree().addTile(3, openvdb::Coord(0), -2.0f, /*active*/true);
    // level is set to 2, expect no change
    executable->execute(test);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), test.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(1), test.tree().activeTileCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT2::NUM_VOXELS), test.tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(-2.0f, test.tree().getValue(openvdb::Coord(0)));

    executable->setTreeExecutionLevel(3);
    executable->execute(test);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), test.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(1), test.tree().activeTileCount());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT2::NUM_VOXELS), test.tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(1.0f, test.tree().getValue(openvdb::Coord(0)));

    // test higher values throw
    CPPUNIT_ASSERT_THROW(executable->setTreeExecutionLevel(4), openvdb::RuntimeError);
}

