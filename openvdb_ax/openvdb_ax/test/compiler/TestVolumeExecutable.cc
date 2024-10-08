// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb_ax/compiler/Compiler.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>
#include <openvdb/tools/ValueTransformer.h>

#include <gtest/gtest.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>

// namespace must be the same as where VolumeExecutable is defined in order
// to access private methods. See also
//https://google.github.io/googletest/advanced.html#testing-private-code
namespace openvdb {
namespace OPENVDB_VERSION_NAME {
namespace ax {

class TestVolumeExecutable : public ::testing::Test
{
};

TEST_F(TestVolumeExecutable, testConstructionDestruction)
{
    // Test the building and teardown of executable objects. This is primarily to test
    // the destruction of Context and ExecutionEngine LLVM objects. These must be destructed
    // in the correct order (ExecutionEngine, then Context) otherwise LLVM will crash

    // must be initialized, otherwise construction/destruction of llvm objects won't
    // exhibit correct behaviour

    ASSERT_TRUE(openvdb::ax::isInitialized());

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

    ASSERT_TRUE(!M);
    ASSERT_TRUE(E);

    std::weak_ptr<llvm::LLVMContext> wC = C;
    std::weak_ptr<const llvm::ExecutionEngine> wE = E;

    // Basic construction

    openvdb::ax::ast::Tree tree;
    openvdb::ax::AttributeRegistry::ConstPtr emptyReg =
        openvdb::ax::AttributeRegistry::create(tree);
    openvdb::ax::VolumeExecutable::Ptr volumeExecutable
        (new openvdb::ax::VolumeExecutable(C, E, emptyReg, nullptr, {}, tree));

    ASSERT_EQ(2, int(wE.use_count()));
    ASSERT_EQ(2, int(wC.use_count()));

    C.reset();
    E.reset();

    ASSERT_EQ(1, int(wE.use_count()));
    ASSERT_EQ(1, int(wC.use_count()));

    // test destruction

    volumeExecutable.reset();

    ASSERT_EQ(0, int(wE.use_count()));
    ASSERT_EQ(0, int(wC.use_count()));
}

TEST_F(TestVolumeExecutable, testCreateMissingGrids)
{
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    openvdb::ax::VolumeExecutable::Ptr executable =
        compiler->compile<openvdb::ax::VolumeExecutable>("@a=v@b.x;");
    ASSERT_TRUE(executable);

    executable->setCreateMissing(false);
    executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::ON);

    openvdb::GridPtrVec grids;
    ASSERT_THROW(executable->execute(grids), openvdb::AXExecutionError);
    ASSERT_TRUE(grids.empty());

    executable->setCreateMissing(true);
    executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::ON);
    executable->execute(grids);

    openvdb::math::Transform::Ptr defaultTransform =
        openvdb::math::Transform::createLinearTransform();

    ASSERT_EQ(size_t(2), grids.size());
    ASSERT_TRUE(grids[0]->getName() == "b");
    ASSERT_TRUE(grids[0]->isType<openvdb::Vec3fGrid>());
    ASSERT_TRUE(grids[0]->empty());
    ASSERT_TRUE(grids[0]->transform() == *defaultTransform);

    ASSERT_TRUE(grids[1]->getName() == "a");
    ASSERT_TRUE(grids[1]->isType<openvdb::FloatGrid>());
    ASSERT_TRUE(grids[1]->empty());
    ASSERT_TRUE(grids[1]->transform() == *defaultTransform);
}

TEST_F(TestVolumeExecutable, testTreeExecutionLevel)
{
    openvdb::ax::CustomData::Ptr data = openvdb::ax::CustomData::create();
    openvdb::FloatMetadata* const meta =
        data->getOrInsertData<openvdb::FloatMetadata>("value");

    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    // generate an executable which does not stream active tiles
    openvdb::ax::VolumeExecutable::Ptr executable =
        compiler->compile<openvdb::ax::VolumeExecutable>("f@test = $value;", data);
    ASSERT_TRUE(executable);
    ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
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
    ASSERT_TRUE(leaf);
    leaf->fill(-1.0f, true);

    const openvdb::FloatTree copy = tree;
    // check config
    auto CHECK_CONFIG = [&]() {
        ASSERT_EQ(openvdb::Index32(1), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(3), tree.activeTileCount());
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-4), tree.getValueDepth(openvdb::Coord(0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        ASSERT_EQ(leaf, tree.probeLeaf(openvdb::Coord(NodeT2::DIM + NodeT1::DIM + NodeT0::DIM)));
        ASSERT_EQ(openvdb::Index64(NodeT2::NUM_VOXELS) +
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS), // leaf
                tree.activeVoxelCount());
        ASSERT_TRUE(copy.hasSameTopology(tree));
    };

    float constant; bool active;

    CHECK_CONFIG();
    ASSERT_EQ(-1.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(-1.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(-1.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(-1.0f, constant);
    ASSERT_TRUE(active);

    openvdb::Index min,max;

    // process default config, all should change
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(0), min);
    ASSERT_EQ(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);
    meta->setValue(-2.0f);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(-2.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(-2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(-2.0f,  tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(-2.0f, constant);
    ASSERT_TRUE(active);

    // process level 0, only leaf change
    meta->setValue(1.0f);
    executable->setTreeExecutionLevel(0);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(0), min);
    ASSERT_EQ(openvdb::Index(0), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(-2.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(-2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(-2.0f,  tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(1.0f, constant);
    ASSERT_TRUE(active);

    // process level 1
    meta->setValue(3.0f);
    executable->setTreeExecutionLevel(1);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(1), min);
    ASSERT_EQ(openvdb::Index(1), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(-2.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(-2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(3.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(1.0f, constant);
    ASSERT_TRUE(active);

    // process level 2
    meta->setValue(5.0f);
    executable->setTreeExecutionLevel(2);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(2), min);
    ASSERT_EQ(openvdb::Index(2), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(-2.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(5.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(3.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(1.0f, constant);
    ASSERT_TRUE(active);

    // process level 3
    meta->setValue(10.0f);
    executable->setTreeExecutionLevel(3);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(3), min);
    ASSERT_EQ(openvdb::Index(3), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(10.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(5.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(3.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(1.0f, constant);
    ASSERT_TRUE(active);

    // test higher values throw
    ASSERT_THROW(executable->setTreeExecutionLevel(4), openvdb::RuntimeError);

    // test level range 0-1
    meta->setValue(-4.0f);
    executable->setTreeExecutionLevel(0,1);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(0), min);
    ASSERT_EQ(openvdb::Index(1), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(10.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(5.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(-4.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(-4.0f, constant);
    ASSERT_TRUE(active);

    // test level range 1-2
    meta->setValue(-6.0f);
    executable->setTreeExecutionLevel(1,2);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(1), min);
    ASSERT_EQ(openvdb::Index(2), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(10.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(-6.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(-6.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(-4.0f, constant);
    ASSERT_TRUE(active);

    // test level range 2-3
    meta->setValue(-11.0f);
    executable->setTreeExecutionLevel(2,3);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(2), min);
    ASSERT_EQ(openvdb::Index(3), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(-11.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(-11.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(-6.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(-4.0f, constant);
    ASSERT_TRUE(active);

    // test on complete range
    meta->setValue(20.0f);
    executable->setTreeExecutionLevel(0,3);
    executable->getTreeExecutionLevel(min,max);
    ASSERT_EQ(openvdb::Index(0), min);
    ASSERT_EQ(openvdb::Index(3), max);
    executable->execute(grid);
    CHECK_CONFIG();
    ASSERT_EQ(20.0f, tree.getValue(openvdb::Coord(0)));
    ASSERT_EQ(20.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
    ASSERT_EQ(20.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
    ASSERT_TRUE(leaf->isConstant(constant, active));
    ASSERT_EQ(20.0f, constant);
    ASSERT_TRUE(active);
}

TEST_F(TestVolumeExecutable, testActiveTileStreaming)
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
        ASSERT_TRUE(leaf);
        leaf->fill(-1.0f, true);

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = 2.0f;");
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);
        executable->execute(grid);

        ASSERT_EQ(openvdb::Index32(1), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(3), tree.activeTileCount());
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-4), tree.getValueDepth(openvdb::Coord(0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-1), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM+NodeT0::DIM)));
        ASSERT_EQ(openvdb::Index64(NodeT2::NUM_VOXELS) +
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(0)));
        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        ASSERT_EQ(leaf, tree.probeLeaf(openvdb::Coord(NodeT2::DIM + NodeT1::DIM + NodeT0::DIM)));
        float constant; bool active;
        ASSERT_TRUE(leaf->isConstant(constant, active));
        ASSERT_EQ(2.0f, constant);
        ASSERT_TRUE(active);
    }

    // test getvoxelpws which densifies everything
    {
        openvdb::FloatGrid grid;
        grid.setName("test");
        openvdb::FloatTree& tree = grid.tree();
        tree.addTile(2, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("vec3d p = getvoxelpws(); f@test = p.x;");
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        const openvdb::Index64 voxels =
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS);

        ASSERT_EQ(openvdb::Index32(voxels / openvdb::FloatTree::LeafNodeType::NUM_VOXELS), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(0), tree.activeTileCount());
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-1), tree.getValueDepth(openvdb::Coord(0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-1), tree.getValueDepth(openvdb::Coord(NodeT1::DIM)));
        ASSERT_EQ(voxels, tree.activeVoxelCount());

        // test values - this isn't strictly necessary for this group of tests
        // as we really just want to check topology results

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            const double pos = grid.indexToWorld(coord).x();
            ASSERT_EQ(*it, float(pos));
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
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

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

        ASSERT_EQ(openvdb::Index32(leafs), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(tiles), tree.activeTileCount());
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM+NodeT1::DIM)));
        ASSERT_EQ(openvdb::Index64(NodeT2::NUM_VOXELS) +
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            if (coord.x() == 0) ASSERT_EQ(*it,  2.0f);
            else                ASSERT_EQ(*it, -1.0f);
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
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        // force stream
        executable->setActiveTileStreaming(openvdb::ax::VolumeExecutable::Streaming::ON);
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));

        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        ASSERT_EQ(openvdb::Index32(0), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(5), tree.activeTileCount());
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*0, 0, 0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        ASSERT_EQ((openvdb::Index64(NodeT1::NUM_VOXELS)*4) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*0, 0, 0)));
        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        ASSERT_EQ(2.0f, tree.getValue(openvdb::Coord(NodeT2::DIM)));
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
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::BOOL));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(openvdb::BoolTree::DEPTH-1), max);

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

        ASSERT_EQ(openvdb::Index32(leafs), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(tiles), tree.activeTileCount());
        ASSERT_EQ(int(openvdb::BoolTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        ASSERT_EQ(int(openvdb::BoolTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        ASSERT_EQ(int(openvdb::BoolTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        ASSERT_EQ(int(openvdb::BoolTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        ASSERT_EQ((openvdb::Index64(NodeT1::NUM_VOXELS)*4) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            if (coord.x() == 0) ASSERT_EQ(*it, false);
            else                ASSERT_EQ(*it, true);
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
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::STRING));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(StringTree::DEPTH-1), max);

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

        ASSERT_EQ(openvdb::Index32(leafs), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(tiles), tree.activeTileCount());
        ASSERT_EQ(int(StringTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*1, 0, 0)));
        ASSERT_EQ(int(StringTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*2, 0, 0)));
        ASSERT_EQ(int(StringTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(NodeT1::DIM*3, 0, 0)));
        ASSERT_EQ(int(StringTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT2::DIM)));
        ASSERT_EQ((openvdb::Index64(NodeT1::NUM_VOXELS)*4) +
            openvdb::Index64(NodeT0::NUM_VOXELS), tree.activeVoxelCount());

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            const openvdb::Coord& coord = it.getCoord();
            if (coord.x() == 0) ASSERT_EQ(*it, std::string("bar"));
            else                ASSERT_EQ(*it, std::string("foo"));
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
        ASSERT_TRUE(leaf);
        leaf->fill(-1.0f, true);

        openvdb::FloatTree copy = tree;

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = float(getcoordx());");
        executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::OFF);

        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::STRING));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        ASSERT_EQ(openvdb::Index32(1), tree.leafCount());
        ASSERT_EQ(openvdb::Index64(2), tree.activeTileCount());
        ASSERT_TRUE(tree.hasSameTopology(copy));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-3), tree.getValueDepth(openvdb::Coord(0)));
        ASSERT_EQ(int(openvdb::FloatTree::DEPTH-2), tree.getValueDepth(openvdb::Coord(NodeT1::DIM)));
        ASSERT_EQ(leaf, tree.probeLeaf(openvdb::Coord(NodeT1::DIM + NodeT0::DIM)));
        float constant; bool active;
        ASSERT_TRUE(leaf->isConstant(constant, active));
        ASSERT_EQ(-1.0f, constant);
        ASSERT_TRUE(active);

        openvdb::tools::foreach(tree.cbeginValueOff(), [&](const auto& it) {
            ASSERT_EQ(*it, float(it.getCoord().x()));
        });

        openvdb::tools::foreach(tree.cbeginValueOn(), [&](const auto& it) {
            ASSERT_EQ(*it, -1.0f);
        });

        // test IterType::ALL

        tree.clear();
        tree.addTile(2, openvdb::Coord(0), -1.0f, /*active*/true); // NodeT1 tile
        tree.addTile(1, openvdb::Coord(NodeT1::DIM), -1.0f, /*active*/true); // NodeT0 tile
        leaf = tree.touchLeaf(openvdb::Coord(NodeT1::DIM + NodeT0::DIM));
        ASSERT_TRUE(leaf);
        leaf->fill(-1.0f, /*inactive*/false);

        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = float(getcoordy());");
        executable->setValueIterator(openvdb::ax::VolumeExecutable::IterType::ALL);

        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::STRING));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
        executable->getTreeExecutionLevel(min,max);
        ASSERT_EQ(openvdb::Index(0), min);
        ASSERT_EQ(openvdb::Index(openvdb::FloatTree::DEPTH-1), max);

        executable->execute(grid);

        const openvdb::Index64 voxels =
            openvdb::Index64(NodeT1::NUM_VOXELS) +
            openvdb::Index64(NodeT0::NUM_VOXELS);

        ASSERT_EQ(openvdb::Index32(voxels / openvdb::FloatTree::LeafNodeType::NUM_VOXELS) + 1, tree.leafCount());
        ASSERT_EQ(openvdb::Index64(0), tree.activeTileCount());
        ASSERT_EQ(voxels, tree.activeVoxelCount());
        ASSERT_EQ(leaf, tree.probeLeaf(openvdb::Coord(NodeT1::DIM + NodeT0::DIM)));
        ASSERT_TRUE(leaf->getValueMask().isOff());

        openvdb::tools::foreach(tree.cbeginValueAll(), [&](const auto& it) {
            ASSERT_EQ(*it, float(it.getCoord().y()));
        });
    }

    // test auto streaming
    {
        executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = f@other; v@test2 = 1; v@test3 = v@test2;");
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::AUTO ==
            executable->getActiveTileStreaming());
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test", openvdb::ax::ast::tokens::CoreType::FLOAT));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("other", openvdb::ax::ast::tokens::CoreType::FLOAT));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::OFF ==
            executable->getActiveTileStreaming("test2", openvdb::ax::ast::tokens::CoreType::VEC3F));
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
            executable->getActiveTileStreaming("test3", openvdb::ax::ast::tokens::CoreType::VEC3F));
        //
        ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::AUTO ==
            executable->getActiveTileStreaming("empty", openvdb::ax::ast::tokens::CoreType::FLOAT));
    }

    // test that some particular functions cause streaming to turn on

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = rand();");
    ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("v@test = getcoord();");
    ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = getcoordx();");
    ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = getcoordy();");
    ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());

    executable = compiler->compile<openvdb::ax::VolumeExecutable>("f@test = getcoordz();");
    ASSERT_TRUE(openvdb::ax::VolumeExecutable::Streaming::ON ==
        executable->getActiveTileStreaming());
}

TEST_F(TestVolumeExecutable, testCompilerCases)
{
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    ASSERT_TRUE(compiler);
    {
        // with string only
        ASSERT_TRUE(static_cast<bool>(compiler->compile<openvdb::ax::VolumeExecutable>("int i;")));
        ASSERT_THROW(compiler->compile<openvdb::ax::VolumeExecutable>("i;"), openvdb::AXCompilerError);
        ASSERT_THROW(compiler->compile<openvdb::ax::VolumeExecutable>("i"), openvdb::AXSyntaxError);
        // with AST only
        auto ast = openvdb::ax::ast::parse("i;");
        ASSERT_THROW(compiler->compile<openvdb::ax::VolumeExecutable>(*ast), openvdb::AXCompilerError);
    }

    openvdb::ax::Logger logger([](const std::string&) {});

    // using string and logger
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
        compiler->compile<openvdb::ax::VolumeExecutable>("", logger); // empty
        ASSERT_TRUE(executable);
    }
    logger.clear();
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("i;", logger); // undeclared variable error
        ASSERT_TRUE(!executable);
        ASSERT_TRUE(logger.hasError());
        logger.clear();
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>("i", logger); // expected ; error (parser)
        ASSERT_TRUE(!executable2);
        ASSERT_TRUE(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("int i = 18446744073709551615;", logger); // warning
        ASSERT_TRUE(executable);
        ASSERT_TRUE(logger.hasWarning());
    }

    // using syntax tree and logger
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("", logger);
        ASSERT_TRUE(tree);
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // empty
        ASSERT_TRUE(executable);
        logger.clear(); // no tree for line col numbers
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // empty
        ASSERT_TRUE(executable2);
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("i;", logger);
        ASSERT_TRUE(tree);
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // undeclared variable error
        ASSERT_TRUE(!executable);
        ASSERT_TRUE(logger.hasError());
        logger.clear(); // no tree for line col numbers
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // undeclared variable error
        ASSERT_TRUE(!executable2);
        ASSERT_TRUE(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("int i = 18446744073709551615;", logger);
        ASSERT_TRUE(tree);
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // warning
        ASSERT_TRUE(executable);
        ASSERT_TRUE(logger.hasWarning());
        logger.clear(); // no tree for line col numbers
        openvdb::ax::VolumeExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::VolumeExecutable>(*tree, logger); // warning
        ASSERT_TRUE(executable2);
        ASSERT_TRUE(logger.hasWarning());
    }
    logger.clear();

    // with copied tree
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*copy, logger); // empty
        ASSERT_TRUE(executable);
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("i;", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*copy, logger); // undeclared variable error
        ASSERT_TRUE(!executable);
        ASSERT_TRUE(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("int i = 18446744073709551615;", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>(*copy, logger); // warning
        ASSERT_TRUE(executable);
        ASSERT_TRUE(logger.hasWarning());
    }
    logger.clear();
}

TEST_F(TestVolumeExecutable, testExecuteBindings)
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

        ASSERT_TRUE(executable);
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        ASSERT_NO_THROW(executable->execute(v));
        ASSERT_EQ(1.0f, f1->tree().getValue({0,0,0}));
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

        ASSERT_TRUE(executable);
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        ASSERT_NO_THROW(executable->execute(v));
        ASSERT_EQ(1.0f, f1->tree().getValue({0,0,0}));
        ASSERT_EQ(2.0f, f2->tree().getValue({0,0,0}));
    }

    // binding to new created attribute AND not binding to new created attribute
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");

        ASSERT_TRUE(executable);
        executable->setAttributeBindings(bindings);
        ASSERT_NO_THROW(executable->execute(v));
        ASSERT_EQ(2.0f, f2->tree().getValue({0,0,0}));
        ASSERT_EQ(size_t(2), v.size());
    }

    // binding to non existent attribute, not creating, error
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");

        ASSERT_TRUE(executable);
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        ASSERT_THROW(executable->execute(v), openvdb::AXExecutionError);
    }

    // trying to bind to an attribute and use the original attribute name at same time
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");
        ASSERT_TRUE(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","c"); // bind b to c
        ASSERT_THROW(executable->setAttributeBindings(bindings), openvdb::AXExecutionError);
   }

    // swap ax and data attributes with bindings
    {
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f2->setName("c");
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f2 };
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @c = 2.0f;");
        ASSERT_TRUE(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","c"); // bind b to c
        bindings.set("c","b"); // bind c to b

        ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        ASSERT_NO_THROW(executable->execute(v));
        ASSERT_EQ(1.0f, f2->tree().getValue({0,0,0}));
    }

    // test setting bindings and then resetting some of those bindings on the same executable
    {
        openvdb::ax::VolumeExecutable::Ptr executable =
            compiler->compile<openvdb::ax::VolumeExecutable>("@b = 1.0f; @a = 2.0f; @c = 3.0f;");
        ASSERT_TRUE(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","a"); // bind b to a
        bindings.set("c","b"); // bind c to b
        bindings.set("a","c"); // bind a to c
        ASSERT_NO_THROW(executable->setAttributeBindings(bindings));

        bindings.set("a","b"); // bind a to b
        bindings.set("b","a"); // bind a to b
        ASSERT_TRUE(!bindings.dataNameBoundTo("c")); // c should be unbound
        // check that the set call resets c to c
        ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        const openvdb::ax::AttributeBindings& bindingsOnExecutable = executable->getAttributeBindings();
        ASSERT_TRUE(bindingsOnExecutable.isBoundAXName("c"));
        ASSERT_EQ(*bindingsOnExecutable.dataNameBoundTo("c"), std::string("c"));
    }
}

TEST_F(TestVolumeExecutable, testCLI)
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

    ASSERT_THROW(CreateCLI("--unknown"), UnusedCLIParam);
    ASSERT_THROW(CreateCLI("-unknown"), UnusedCLIParam);
    ASSERT_THROW(CreateCLI("-"), UnusedCLIParam);
    ASSERT_THROW(CreateCLI("--"), UnusedCLIParam);
    ASSERT_THROW(CreateCLI("-- "), UnusedCLIParam);

    {
        CLI cli = CreateCLI("");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(defaultGrain, exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());
    }

    // --create-missing
    {
        ASSERT_THROW(CreateCLI("--create-missing"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--create-missing invalid"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--create-missing --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--create-missing ON");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(true, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(defaultGrain, exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());
    }

    // --tile-stream
    {
        ASSERT_THROW(CreateCLI("--tile-stream"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--tile-stream invalid"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--tile-stream --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--tile-stream ON");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(openvdb::ax::VolumeExecutable::Streaming::ON, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(defaultGrain, exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());
    }

    // --node-iter
    {
        ASSERT_THROW(CreateCLI("--node-iter"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--node-iter invalid"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--node-iter --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--node-iter ALL");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(openvdb::ax::VolumeExecutable::IterType::ALL, exe->getValueIterator());
        ASSERT_EQ(defaultGrain, exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());
    }

    // --tree-level
    {
        ASSERT_THROW(CreateCLI("--tree-level"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--tree-level invalid"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--tree-level --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--tree-level 0");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(min, Index(0));
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(defaultGrain, exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());

        cli = CreateCLI("--tree-level 1:2");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(min, Index(1));
        ASSERT_EQ(max, Index(2));
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(defaultGrain, exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());
    }

    // --tree-level
    {
        ASSERT_THROW(CreateCLI("--volume-grain"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--volume-grain invalid"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--volume-grain --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--volume-grain 0");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(size_t(0), exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());

        cli = CreateCLI("--volume-grain 1:2");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(size_t(1), exe->getGrainSize());
        ASSERT_EQ(size_t(2), exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());
    }

    // --bindings
    {
        ASSERT_THROW(CreateCLI("--bindings"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--bindings :"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--bindings ,"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--bindings a:"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--bindings a,b"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--bindings :b"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--bindings ,a:b"), openvdb::CLIError);
        ASSERT_THROW(CreateCLI("--bindings --create-missing ON"), openvdb::CLIError);

        CLI cli = CreateCLI("--bindings a:b,c:d,12:13");
        ax::AttributeBindings bindings;
        bindings.set("a", "b");
        bindings.set("c", "d");
        bindings.set("12", "13");

        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(defaultCreateMissing, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(defaultGrain, exe->getGrainSize());
        ASSERT_EQ(defaultTileGrain, exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(bindings, exe->getAttributeBindings());
    }

    // multiple
    {
        CLI cli = CreateCLI("--volume-grain 5:10 --create-missing OFF");
        auto exe = compiler->compile<openvdb::ax::VolumeExecutable>("");
        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));

        Index min,max;
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(defaultMinLevel, min);
        ASSERT_EQ(defaultMaxLevel, max);
        ASSERT_EQ(false, exe->getCreateMissing());
        ASSERT_EQ(defaultTileStream, exe->getActiveTileStreaming());
        ASSERT_EQ(defaultValueIter, exe->getValueIterator());
        ASSERT_EQ(size_t(5), exe->getGrainSize());
        ASSERT_EQ(size_t(10), exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(defaultBindings, exe->getAttributeBindings());

        cli = CreateCLI("--tile-stream ON --node-iter OFF --tree-level 2:3 --volume-grain 10:20 --create-missing ON --bindings a:b");
        ax::AttributeBindings bindings;
        bindings.set("a", "b");

        ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        exe->getTreeExecutionLevel(min, max);
        ASSERT_EQ(Index(2), min);
        ASSERT_EQ(Index(3), max);
        ASSERT_EQ(true, exe->getCreateMissing());
        ASSERT_EQ(openvdb::ax::VolumeExecutable::Streaming::ON, exe->getActiveTileStreaming());
        ASSERT_EQ(openvdb::ax::VolumeExecutable::IterType::OFF, exe->getValueIterator());
        ASSERT_EQ(size_t(10), exe->getGrainSize());
        ASSERT_EQ(size_t(20), exe->getActiveTileStreamingGrainSize());
        ASSERT_EQ(bindings, exe->getAttributeBindings());
    }
}

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
