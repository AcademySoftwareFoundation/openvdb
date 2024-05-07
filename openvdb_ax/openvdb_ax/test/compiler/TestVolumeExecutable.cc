// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb_ax/compiler/Compiler.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>
#include <openvdb/tools/ValueTransformer.h>

#include <cppunit/extensions/HelperMacros.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>

class TestVolumeExecutable : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestVolumeExecutable);
    CPPUNIT_TEST(testConstructionDestruction);
    CPPUNIT_TEST(testCreateMissingGrids);
    CPPUNIT_TEST(testTreeExecutionLevel);
    CPPUNIT_TEST(testActiveTileStreaming);
    CPPUNIT_TEST(testCompilerCases);
    CPPUNIT_TEST(testExecuteBindings);
    CPPUNIT_TEST(testCLI);
    CPPUNIT_TEST_SUITE_END();

    void testConstructionDestruction();
    void testCreateMissingGrids();
    void testTreeExecutionLevel();
    void testActiveTileStreaming();
    void testCompilerCases();
    void testExecuteBindings();
    void testCLI();
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
#if LLVM_VERSION_MAJOR >= 15
    // This will not work from LLVM 16. We'll need to fix this
    // https://llvm.org/docs/OpaquePointers.html
    C->setOpaquePointers(false);
#endif

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
        (new openvdb::ax::VolumeExecutable(C, E, emptyReg, nullptr, {}, tree));

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
    CPPUNIT_ASSERT(executable);

    executable->setCreateMissing(false);
    executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::ON);

    openvdb::GridPtrVec grids;
    CPPUNIT_ASSERT_THROW(executable->execute(grids), openvdb::AXExecutionError);
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
    openvdb::ax::CustomData::Ptr data = openvdb::ax::CustomData::create();
    openvdb::FloatMetadata* const meta =
        data->getOrInsertData<openvdb::FloatMetadata>("value");

    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    // generate an executable which does not stream active tiles
    openvdb::ax::VolumeExecutable::Ptr executable =
        compiler->compile<openvdb::ax::VolumeExecutable>("f@test = $value;", data);
    CPPUNIT_ASSERT(executable);
    CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
        executable->getActiveTileStreaming());

    using NodeT0 = openvdb::FloatGrid::Accessor::template NodeTypeAtLevel<0>;
    using NodeT1 = openvdb::FloatGrid::Accessor::template NodeTypeAtLevel<1>;
    using NodeT2 = openvdb::FloatGrid::Accessor::template NodeTypeAtLevel<2>;

    openvdb::FloatGrid grid;
    grid.setName("test");
    openvdb::FloatTree& tree = grid.tree();
    tree.addTile(3, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT2 tile
    tree.addTile(2, openvdb::Coord(NodeT2::DIM), -1.0f, /*active*/true); // NodeT1 tile
    tree.addTile(1, openvdb::Coord(NodeT2::DIM+NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile
    auto leaf = tree.touchLeaf(openvdb::Coord(NodeT2::DIM + NodeT1::DIM + NodeT0::DIM));
    CPPUNIT_ASSERT(leaf);
    leaf->fill(-1.0f, true);

    const openvdb::FloatTree copy = tree;
    // check config
    auto CHECK_CONFIG = [&]() {
        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(1), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(3), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-4), tree.getValueDepth(openvdb::Coord(0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        CPPUNIT_ASSERT_EQUAL(leaf, tree.probeLeaf(openvdb::Coord(NodeT2::DIM + NodeT1::DIM + NodeT0::DIM)));
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT2::NUM_VOXELS) +
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS), // leaf
                tree.activeVoxelCount());
        CPPUNIT_ASSERT(copy.hasSameTopology(tree));
    };

    float constant; bool active;

    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(-1.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(-1.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(-1.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(-1.0f, constant);
    CPPUNIT_ASSERT(active);

    openvdb::Index min,max;

    // process default config, all should change
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);
    meta->setValue(-2.0f);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(-2.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(-2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(-2.0f,  tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(-2.0f, constant);
    CPPUNIT_ASSERT(active);

    // process level 0, only leaf change
    meta->setValue(1.0f);
    executable->setTreeExecutionLevel(0);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(-2.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(-2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(-2.0f,  tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(1.0f, constant);
    CPPUNIT_ASSERT(active);

    // process level 1
    meta->setValue(3.0f);
    executable->setTreeExecutionLevel(1);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(1), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(1), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(-2.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(-2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(3.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(1.0f, constant);
    CPPUNIT_ASSERT(active);

    // process level 2
    meta->setValue(5.0f);
    executable->setTreeExecutionLevel(2);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(2), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(2), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(-2.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(5.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(3.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(1.0f, constant);
    CPPUNIT_ASSERT(active);

    // process level 3
    meta->setValue(10.0f);
    executable->setTreeExecutionLevel(3);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(3), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(3), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(10.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(5.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(3.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(1.0f, constant);
    CPPUNIT_ASSERT(active);

    // test higher values throw
    CPPUNIT_ASSERT_THROW(executable->setTreeExecutionLevel(4), openvdb::RuntimeError);

    // test level range 0-1
    meta->setValue(-4.0f);
    executable->setTreeExecutionLevel(0,1);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(1), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(10.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(5.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(-4.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(-4.0f, constant);
    CPPUNIT_ASSERT(active);

    // test level range 1-2
    meta->setValue(-6.0f);
    executable->setTreeExecutionLevel(1,2);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(1), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(2), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(10.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(-6.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(-6.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(-4.0f, constant);
    CPPUNIT_ASSERT(active);

    // test level range 2-3
    meta->setValue(-11.0f);
    executable->setTreeExecutionLevel(2,3);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(2), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(3), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(-11.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(-11.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(-6.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(-4.0f, constant);
    CPPUNIT_ASSERT(active);

    // test on complete range
    meta->setValue(20.0f);
    executable->setTreeExecutionLevel(0,3);
    executable->getTreeExecutionLevel(min,max);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(3), max);
    executable->execute(grid);
    CHECK_CONFIG();
    CPPUNIT_ASSERT_EQUAL(20.0f, tree.getValue(openvdb::Coord(0)));
    CPPUNIT_ASSERT_EQUAL(20.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    CPPUNIT_ASSERT_EQUAL(20.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    CPPUNIT_ASSERT(leaf->isConstant(constant, active));
    CPPUNIT_ASSERT_EQUAL(20.0f, constant);
    CPPUNIT_ASSERT(active);
}

void
TestVolumeExecutable::testActiveTileStreaming()
{
    using NodeT0 = openvdb::FloatGrid::Accessor::template NodeTypeAtLevel<0>;
    using NodeT1 = openvdb::FloatGrid::Accessor::template NodeTypeAtLevel<1>;
    using NodeT2 = openvdb::FloatGrid::Accessor::template NodeTypeAtLevel<2>;

    //

    openvdb::Index min,max;
    openvdb::ax::VolumeExecutable::Ptr executable;
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();

    // test no streaming
    {
        openvdb::FloatGrid grid;
        grid.setName("test");
        openvdb::FloatTree& tree = grid.tree();
        tree.addTile(3, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT2 tile
        tree.addTile(2, openvdb::Coord(NodeT2::DIM), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT2::DIM+NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile
        auto leaf = tree.touchLeaf(openvdb::Coord(NodeT2::DIM + NodeT1::DIM + NodeT0::DIM));
        CPPUNIT_ASSERT(leaf);
        leaf->fill(-1.0f, true);

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = 2.0f;");
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);
        executable->execute(grid);

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(1), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(3), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-4), tree.getValueDepth(openvdb::Coord(0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-1), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM+NodeT0::DIM)));
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT2::NUM_VOXELS) +
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(0)));
        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        CPPUNIT_ASSERT_EQUAL(leaf, tree.probeLeaf(openvdb::Coord(NodeT2::DIM + NodeT1::DIM + NodeT0::DIM)));
        float constant; bool active;
        CPPUNIT_ASSERT(leaf->isConstant(constant, active));
        CPPUNIT_ASSERT_EQUAL(2.0f, constant);
        CPPUNIT_ASSERT(active);
    }

    // test getvoxelpws which densifies everything
    {
        openvdb::FloatGrid grid;
        grid.setName("test");
        openvdb::FloatTree& tree = grid.tree();
        tree.addTile(2, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("vec3d p = getvoxelpws(); f@test = p.x;");
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        const openvdb::Index64 voxels =
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS);

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(voxels / openvdb::FloatTree::LeafNodeType::NUM_VOXELS), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(0), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-1), tree.getValueDepth(openvdb::Coord(0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-1), tree.getValueDepth(openvdb::Coord(NodeT1::DIM)));
        CPPUNIT_ASSERT_EQUAL(voxels, tree.activeVoxelCount());

        // test values - this isn't strictly necessary for this group of tests
        // as we really just want to check topology results

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            const double pos = grid.indexToWorld(coord).x();
            CPPUNIT_ASSERT_EQUAL(*it, float(pos));
        });
    }

    // test spatially varying voxelization
    // @note this tests execution over a NodeT2 which is slow
    {
        openvdb::FloatGrid grid;
        grid.setName("test");
        openvdb::FloatTree& tree = grid.tree();
        tree.addTile(3, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT2 tile
        tree.addTile(2, openvdb::Coord(NodeT2::DIM), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT2::DIM+NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile

        // sets all x == 0 coordinates to 2.0f. These all reside in the NodeT2 tile
        executable = compiler->compile<openvdb::ax::VolumeExecutable>("int x = getcoordx(); if (x == 0) f@test = 2.0f;");
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        const openvdb::Index64 face = NodeT2::DIM * NodeT2::DIM; // face voxel count of NodeT2 x==0
        const openvdb::Index64 leafs = // expected leaf nodes that need to be created
            (face * openvdb::FloatTree::LeafNodeType::DIM) /
            openvdb::FloatTree::LeafNodeType::NUM_VOXELS;

        // number of child nodes in NodeT2;
        const openvdb::Index64 n2ChildAxisCount = NodeT2::DIM / NodeT2::getChildDim();
        const openvdb::Index64 n2ChildCount = n2ChildAxisCount * n2ChildAxisCount * n2ChildAxisCount;

        // number of child nodes in NodeT1;
        const openvdb::Index64 n1ChildAxisCount = NodeT1::DIM / NodeT1::getChildDim();
        const openvdb::Index64 n1ChildCount = n1ChildAxisCount * n1ChildAxisCount * n1ChildAxisCount;

         const openvdb::Index64 tiles = // expected active tiles
            (n2ChildCount -  (n2ChildAxisCount * n2ChildAxisCount)) + // NodeT2 child - a single face
            ((n1ChildCount * (n2ChildAxisCount * n2ChildAxisCount)) - leafs) // NodeT1 face tiles (NodeT0) - leafs
            + 1 /*NodeT1*/ + 1 /*NodeT0*/;

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(leafs), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(tiles), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(NodeT2::NUM_VOXELS) +
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            if (coord.x() == 0) CPPUNIT_ASSERT_EQUAL(*it,  2.0f);
            else                CPPUNIT_ASSERT_EQUAL(*it, -1.0f);
        });
    }

    // test post pruning - force active streaming with a uniform kernel
    {
        openvdb::FloatGrid grid;
        grid.setName("test");
        openvdb::FloatTree& tree = grid.tree();
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*0, 0, 0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*1, 0, 0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*2, 0, 0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*3, 0, 0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT2::DIM), -1.0f, /*active*/true); // NodeT0 tile

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = 2.0f;");
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        // force stream
        executable->setActiveTileStreaming(openvdb::ax::VolumeExecutable::Streaming::ON);
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(0), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(5), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*0, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        CPPUNIT_ASSERT_EQUAL((openvdb::Index64(NodeT1::NUM_VOXELS)*4) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*0, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    }

    // test spatially varying voxelization for bool grids which use specialized implementations
    {
        openvdb::BoolGrid grid;
        grid.setName("test");
        openvdb::BoolTree& tree = grid.tree();
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*0, 0, 0), true, /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*1, 0, 0), true, /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*2, 0, 0), true, /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*3, 0, 0), true, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT2::DIM), true, /*active*/true); // NodeT0 tileile

        // sets all x == 0 coordinates to 2.0f. These all reside in the NodeT2 tile
        executable = compiler->compile<openvdb::ax::VolumeExecutable>("int x = getcoordx(); if (x == 0) bool@test = false;");
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::BOOL));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::BoolTree::DEPTH-1), max);

        executable->execute(grid);

        const openvdb::Index64 face = NodeT1::DIM * NodeT1::DIM; // face voxel count of NodeT2 x==0
        const openvdb::Index64 leafs = // expected leaf nodes that need to be created
            (face * openvdb::BoolTree::LeafNodeType::DIM) /
            openvdb::BoolTree::LeafNodeType::NUM_VOXELS;

        // number of child nodes in NodeT1;
        const openvdb::Index64 n1ChildAxisCount = NodeT1::DIM / NodeT1::getChildDim();
        const openvdb::Index64 n1ChildCount = n1ChildAxisCount * n1ChildAxisCount * n1ChildAxisCount;

         const openvdb::Index64 tiles = // expected active tiles
            (n1ChildCount - leafs) // NodeT1 face tiles (NodeT0) - leafs
            + 3 /*NodeT1*/ + 1 /*NodeT0*/;

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(leafs), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(tiles), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(int(openvdb::BoolTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::BoolTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::BoolTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::BoolTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        CPPUNIT_ASSERT_EQUAL((openvdb::Index64(NodeT1::NUM_VOXELS)*4) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            if (coord.x() == 0) CPPUNIT_ASSERT_EQUAL(*it, false);
            else                CPPUNIT_ASSERT_EQUAL(*it, true);
        });
    }

    // test spatially varying voxelization for string grids which use specialized implementations
    // Note: StringGrids are no longer registered by default
    {
        using StringTree = openvdb::tree::Tree4<std::string, 5, 4, 3>::Type;
        using StringGrid = openvdb::Grid<openvdb::tree::Tree4<std::string, 5, 4, 3>::Type>;
        StringGrid grid;
        grid.setName("test");
        StringTree& tree = grid.tree();
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*0, 0, 0), "foo", /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*1, 0, 0), "foo", /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*2, 0, 0), "foo", /*active*/true); // NodeT1 tile
        tree.addTile(2, openvdb::Coord(NodeT1::DIM*3, 0, 0), "foo", /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT2::DIM), "foo", /*active*/true); // NodeT0 tileile

        // sets all x == 0 coordinates to 2.0f. These all reside in the NodeT2 tile
        executable = compiler->compile<openvdb::ax::VolumeExecutable>("int x = getcoordx(); if (x == 0) s@test = \"bar\";");
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::STRING));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(StringTree::DEPTH-1), max);

        executable->execute(grid);

        const openvdb::Index64 face = NodeT1::DIM * NodeT1::DIM; // face voxel count of NodeT2 x==0
        const openvdb::Index64 leafs = // expected leaf nodes that need to be created
            (face * StringTree::LeafNodeType::DIM) /
            StringTree::LeafNodeType::NUM_VOXELS;

        // number of child nodes in NodeT1;
        const openvdb::Index64 n1ChildAxisCount = NodeT1::DIM / NodeT1::getChildDim();
        const openvdb::Index64 n1ChildCount = n1ChildAxisCount * n1ChildAxisCount * n1ChildAxisCount;

         const openvdb::Index64 tiles = // expected active tiles
            (n1ChildCount - leafs) // NodeT1 face tiles (NodeT0) - leafs
            + 3 /*NodeT1*/ + 1 /*NodeT0*/;

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(leafs), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(tiles), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(int(StringTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(StringTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(StringTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        CPPUNIT_ASSERT_EQUAL(int(StringTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        CPPUNIT_ASSERT_EQUAL((openvdb::Index64(NodeT1::NUM_VOXELS)*4) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            if (coord.x() == 0) CPPUNIT_ASSERT_EQUAL(*it, std::string("bar"));
            else                CPPUNIT_ASSERT_EQUAL(*it, std::string("foo"));
        });
    }

    // test streaming with an OFF iterator (no streaming behaviour) and an ALL iterator (streaming behaviour for ON values only)
    {
        openvdb::FloatGrid grid;
        grid.setName("test");
        openvdb::FloatTree& tree = grid.tree();
        tree.addTile(2, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile
        auto leaf = tree.touchLeaf(openvdb::Coord(NodeT1::DIM + NodeT0::DIM));
        CPPUNIT_ASSERT(leaf);
        leaf->fill(-1.0f, true);

        openvdb::FloatTree copy = tree;

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = float(getcoordx());");
        executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::OFF);

        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::STRING));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(1), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(2), tree.activeTileCount());
        CPPUNIT_ASSERT(tree.hasSameTopology(copy));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(0)));
        CPPUNIT_ASSERT_EQUAL(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT1::DIM)));
        CPPUNIT_ASSERT_EQUAL(leaf, tree.probeLeaf(openvdb::Coord(NodeT1::DIM + NodeT0::DIM)));
        float constant; bool active;
        CPPUNIT_ASSERT(leaf->isConstant(constant, active));
        CPPUNIT_ASSERT_EQUAL(-1.0f, constant);
        CPPUNIT_ASSERT(active);

        openvdb::tools::foreach(tree.cbeginValueOff(), [&](const auto& it) {
            CPPUNIT_ASSERT_EQUAL(*it, float(it.getCoord().x()));
        });

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            CPPUNIT_ASSERT_EQUAL(*it, -1.0f);
        });

        // test IterType::ALL

        tree.clear();
        tree.addTile(2, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile
        leaf = tree.touchLeaf(openvdb::Coord(NodeT1::DIM + NodeT0::DIM));
        CPPUNIT_ASSERT(leaf);
        leaf->fill(-1.0f, /*inactive*/false);

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = float(getcoordy());");
        executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::ALL);

        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::STRING));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), min);
        CPPUNIT_ASSERT_EQUAL(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        const openvdb::Index64 voxels =
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS);

        CPPUNIT_ASSERT_EQUAL(openvdb::Index32(voxels / openvdb::FloatTree::LeafNodeType::NUM_VOXELS) + 1, tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(openvdb::Index64(0), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(voxels, tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(leaf, tree.probeLeaf(openvdb::Coord(NodeT1::DIM + NodeT0::DIM)));
        CPPUNIT_ASSERT(leaf->getValueMask().isOff());

        openvdb::tools::foreach(tree.cbeginValueAll(), [&](const auto& it) {
            CPPUNIT_ASSERT_EQUAL(*it, float(it.getCoord().y()));
        });
    }

    // test auto streaming
    {
        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = f@other; v@test2 = 1; v@test3 = v@test2;");
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::AUTO ==
            executable->getActiveTileStreaming());
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("other", openvdb::ax::ast::tokens::CoreType::FLOAT));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("test2", openvdb::ax::ast::tokens::CoreType::VEC3F));
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test3", openvdb::ax::ast::tokens::CoreType::VEC3F));
        //
        CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::AUTO ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
    }

    // test that some particular functions cause streaming to turn on

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = rand();");
    CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("v@test = getcoord();");
    CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = getcoordx();");
    CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = getcoordy();");
    CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = getcoordz();");
    CPPUNIT_ASSERT(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());
}


void
TestVolumeExecutable::testCompilerCases()
{
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    CPPUNIT_ASSERT(compiler);
    {
        // with string only
        CPPUNIT_ASSERT(static_cast<bool>(compiler->compile<openvdb::ax::VolumeExecutable>("int i;")));
        CPPUNIT_ASSERT_THROW(compiler->compile<openvdb::ax::VolumeExecutable>("i;"), openvdb::AXCompilerError);
        CPPUNIT_ASSERT_THROW(compiler->compile<openvdb::ax::VolumeExecutable>("i"), openvdb::AXSyntaxError);
        // with AST only
        auto ast = openvdb::ax::ast::parse("i;");
        CPPUNIT_ASSERT_THROW(compiler->compile<openvdb::ax::VolumeExecutable>(*ast), openvdb::AXCompilerError);
    }

    openvdb::ax::Logger logger([](const std::string&) {});

    // using string and logger
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
        compiler->compile<openvdb::ax::VolumeExecutable>("", logger); // empty
        CPPUNIT_ASSERT(executable);
    }
    logger.clear();
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("i;", logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable);
        CPPUNIT_ASSERT(logger.hasError());
        logger.clear();
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>("i", logger); // expected ; error (parser)
        CPPUNIT_ASSERT(!executable2);
        CPPUNIT_ASSERT(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("int i = 18446744073709551615;", logger); // warning
        CPPUNIT_ASSERT(executable);
        CPPUNIT_ASSERT(logger.hasWarning());
    }

    // using syntax tree and logger
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("", logger);
        CPPUNIT_ASSERT(tree);
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // empty
        CPPUNIT_ASSERT(executable);
        logger.clear(); // no tree for line col numbers
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // empty
        CPPUNIT_ASSERT(executable2);
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("i;", logger);
        CPPUNIT_ASSERT(tree);
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable);
        CPPUNIT_ASSERT(logger.hasError());
        logger.clear(); // no tree for line col numbers
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable2);
        CPPUNIT_ASSERT(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("int i = 18446744073709551615;", logger);
        CPPUNIT_ASSERT(tree);
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // warning
        CPPUNIT_ASSERT(executable);
        CPPUNIT_ASSERT(logger.hasWarning());
        logger.clear(); // no tree for line col numbers
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // warning
        CPPUNIT_ASSERT(executable2);
        CPPUNIT_ASSERT(logger.hasWarning());
    }
    logger.clear();

    // with copied tree
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*copy, logger); // empty
        CPPUNIT_ASSERT(executable);
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("i;", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*copy, logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable);
        CPPUNIT_ASSERT(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("int i = 18446744073709551615;", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*copy, logger); // warning
        CPPUNIT_ASSERT(executable);
        CPPUNIT_ASSERT(logger.hasWarning());
    }
    logger.clear();
}

void
TestVolumeExecutable::testExecuteBindings()
{
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();

    openvdb::ax::AttributeBindings bindings;
    bindings.set("b", "a"); // bind b to a

    {
        // multi volumes
        openvdb::FloatGrid::Ptr f1(new openvdb::FloatGrid);
        f1->setName("a");
        f1->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f1 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f;");

        CPPUNIT_ASSERT(executable);
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        CPPUNIT_ASSERT_NO_THROW(executable->execute(v));
        CPPUNIT_ASSERT_EQUAL(1.0f, f1->tree().getValue({0,0,0}));
    }

    // binding to existing attribute AND not binding to attribute
    {
        openvdb::FloatGrid::Ptr f1(new openvdb::FloatGrid);
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f1->setName("a");
        f2->setName("c");
        f1->tree().setValueOn({0,0,0}, 0.0f);
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f1, f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");

        CPPUNIT_ASSERT(executable);
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        CPPUNIT_ASSERT_NO_THROW(executable->execute(v));
        CPPUNIT_ASSERT_EQUAL(1.0f, f1->tree().getValue({0,0,0}));
        CPPUNIT_ASSERT_EQUAL(2.0f, f2->tree().getValue({0,0,0}));
    }

    // binding to new created attribute AND not binding to new created attribute
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");

        CPPUNIT_ASSERT(executable);
        executable->setAttributeBindings(bindings);
        CPPUNIT_ASSERT_NO_THROW(executable->execute(v));
        CPPUNIT_ASSERT_EQUAL(2.0f, f2->tree().getValue({0,0,0}));
        CPPUNIT_ASSERT_EQUAL(size_t(2), v.size());
    }

    // binding to non existent attribute, not creating, error
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");

        CPPUNIT_ASSERT(executable);
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        CPPUNIT_ASSERT_THROW(executable->execute(v), openvdb::AXExecutionError);
    }

    // trying to bind to an attribute and use the original attribute name at same time
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","c"); // bind b to c
        CPPUNIT_ASSERT_THROW(executable->setAttributeBindings(bindings), openvdb::AXExecutionError);
   }

    // swap ax and data attributes with bindings
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","c"); // bind b to c
        bindings.set("c","b"); // bind c to b

        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        CPPUNIT_ASSERT_NO_THROW(executable->execute(v));
        CPPUNIT_ASSERT_EQUAL(1.0f, f2->tree().getValue({0,0,0}));
    }

    // test setting bindings and then resetting some of those bindings on the same executable
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @a = 2.0f; @c = 3.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","a"); // bind b to a
        bindings.set("c","b"); // bind c to b
        bindings.set("a","c"); // bind a to c
        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));

        bindings.set("a","b"); // bind a to b
        bindings.set("b","a"); // bind a to b
        CPPUNIT_ASSERT(!bindings.dataNameBoundTo("c")); // c should be unbound
        // check that the set call resets c to c
        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        const openvdb::ax::AttributeBindings& bindingsOnExecutable = executable->getAttributeBindings();
        CPPUNIT_ASSERT(bindingsOnExecutable.isBoundAXName("c"));
        CPPUNIT_ASSERT_EQUAL(*bindingsOnExecutable.dataNameBoundTo("c"), std::string("c"));
    }
}

void
TestVolumeExecutable::testCLI()
{
    using namespace openvdb;
    using CLI = openvdb::ax::VolumeExecutable::CLI;

    struct UnusedCLIParam : public openvdb::Exception {
        UnusedCLIParam() noexcept: Exception( "UnusedCLIParam" ) {} \
        explicit UnusedCLIParam(const std::string& msg) noexcept: Exception( "UnusedCLIParam" , &msg) {}
    };

    auto CreateCLI = [](const char* c, bool throwIfUnused = true)
    {
        std::vector<std::string> strs;
        const char* s = c;
        while (*c != '\0') {
            if (*c == ' ') {
                strs.emplace_back(std::string(s, c-s));
                ++c;
                s = c;
            }
            else {
                ++c;
            }
        }
        if (*s != '\0') strs.emplace_back(std::string(s, c-s));

        std::vector<const char*> args;
        for (auto& str : strs) args.emplace_back(str.c_str());

        std::unique_ptr<bool[]> flags(new bool[args.size()]);
        std::fill(flags.get(), flags.get()+args.size(), false);

        auto cli = CLI::create(args.size(), args.data(), flags.get());
        if (throwIfUnused) {
            for (size_t i = 0; i < args.size(); ++i) {
                if (!flags[i]) OPENVDB_THROW(UnusedCLIParam, "unused param");
            }
        }
        return cli;
    };

    ax::Compiler::UniquePtr compiler = ax::Compiler::create();

    auto defaultExe = compiler->compile<openvdb::ax::VolumeExecutable>("");
    Index defaultMinLevel, defaultMaxLevel;
    defaultExe->getTreeExecutionLevel(defaultMinLevel, defaultMaxLevel);
    const auto defaultCreateMissing = defaultExe->getCreateMissing();
    const auto defaultTileStream = defaultExe->getActiveTileStreaming();
    const auto defaultValueIter = defaultExe->getValueIterator();
    const auto defaultGrain = defaultExe->getGrainSize();
    const auto defaultTileGrain = defaultExe->getActiveTileStreamingGrainSize();
    const auto defaultBindings = defaultExe->getAttributeBindings();

    CPPUNIT_ASSERT_THROW(CreateCLI("--unknown"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("-unknown"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("-"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("--"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("-- "), UnusedCLIParam);

    {
        CLI cli = CreateCLI("");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --create-missing
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--create-missing"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--create-missing invalid"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--create-missing --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--create-missing ON");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(true, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --tile-stream
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--tile-stream"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--tile-stream invalid"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--tile-stream --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--tile-stream ON");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(openvdb::ax::VolumeExecutable::Streaming::ON, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --node-iter
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--node-iter"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--node-iter invalid"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--node-iter --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--node-iter ALL");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(openvdb::ax::VolumeExecutable::IterType::ALL, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --tree-level
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--tree-level"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--tree-level invalid"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--tree-level --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--tree-level 0");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(min, Index(0));
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());

        cli = CreateCLI("--tree-level 1:2");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(min, Index(1));
        CPPUNIT_ASSERT_EQUAL(max, Index(2));
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --tree-level
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--volume-grain"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--volume-grain invalid"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--volume-grain --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--volume-grain 0");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(size_t(0), exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());

        cli = CreateCLI("--volume-grain 1:2");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(size_t(1), exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(size_t(2), exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --bindings
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings :"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings ,"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings a:"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings a,b"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings :b"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings ,a:b"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings --create-missing ON"), openvdb::CLIError);

        CLI cli = CreateCLI("--bindings a:b,c:d,12:13");
        ax::AttributeBindings bindings;
        bindings.set("a", "b");
        bindings.set("c", "d");
        bindings.set("12", "13");

        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(bindings, exe->getAttributeBindings());
    }

    // multiple
    {
        CLI cli = CreateCLI("--volume-grain 5:10 --create-missing OFF");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(defaultMinLevel, min);
        CPPUNIT_ASSERT_EQUAL(defaultMaxLevel, max);
        CPPUNIT_ASSERT_EQUAL(false, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultTileStream, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(defaultValueIter, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(size_t(5), exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(size_t(10), exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());

        cli = CreateCLI("--tile-stream ON --node-iter OFF --tree-level 2:3 --volume-grain 10:20 --create-missing ON --bindings a:b");
        ax::AttributeBindings bindings;
        bindings.set("a", "b");

        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        exe->getTreeExecutionLevel(min, max);
        CPPUNIT_ASSERT_EQUAL(Index(2), min);
        CPPUNIT_ASSERT_EQUAL(Index(3), max);
        CPPUNIT_ASSERT_EQUAL(true, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(openvdb::ax::VolumeExecutable::Streaming::ON, exe->getActiveTileStreaming());
        CPPUNIT_ASSERT_EQUAL(openvdb::ax::VolumeExecutable::IterType::OFF, exe->getValueIterator());
        CPPUNIT_ASSERT_EQUAL(size_t(10), exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(size_t(20), exe->getActiveTileStreamingGrainSize());
        CPPUNIT_ASSERT_EQUAL(bindings, exe->getAttributeBindings());
    }
}
