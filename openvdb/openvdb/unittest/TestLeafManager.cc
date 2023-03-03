// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Types.h>
#include <openvdb/TypeList.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/util/CpuTimer.h>
#include "util.h" // for unittest_util::makeSphere()
#include <gtest/gtest.h>


class TestLeafManager: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


TEST_F(TestLeafManager, testBasics)
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Vec3f;
    using openvdb::FloatGrid;
    using openvdb::FloatTree;

    const Vec3f center(0.35f, 0.35f, 0.35f);
    const float radius = 0.15f;
    const int dim = 128, half_width = 5;
    const float voxel_size = 1.0f/dim;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/half_width*voxel_size);
    FloatTree& tree = grid->tree();
    grid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/voxel_size));

    unittest_util::makeSphere<FloatGrid>(
        Coord(dim), center, radius, *grid, unittest_util::SPHERE_SPARSE_NARROW_BAND);
    const size_t leafCount = tree.leafCount();

    //grid->print(std::cout, 3);
    {// test with no aux buffers
        openvdb::tree::LeafManager<FloatTree> r(tree);
        EXPECT_EQ(leafCount, r.leafCount());
        EXPECT_EQ(size_t(0), r.auxBufferCount());
        EXPECT_EQ(size_t(0), r.auxBuffersPerLeaf());
        size_t n = 0;
        for (FloatTree::LeafCIter iter=tree.cbeginLeaf(); iter; ++iter, ++n) {
            EXPECT_TRUE(r.leaf(n) == *iter);
            EXPECT_TRUE(r.getBuffer(n,0) == iter->buffer());
        }
        EXPECT_EQ(r.leafCount(), n);
        EXPECT_TRUE(!r.swapBuffer(0,0));

        r.rebuildAuxBuffers(2);

        EXPECT_EQ(leafCount, r.leafCount());
        EXPECT_EQ(size_t(2), r.auxBuffersPerLeaf());
        EXPECT_EQ(size_t(2*leafCount),r.auxBufferCount());

         for (n=0; n<leafCount; ++n) {
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) == r.getBuffer(n,2));
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
    }
    {// test with 2 aux buffers
        openvdb::tree::LeafManager<FloatTree> r(tree, 2);
        EXPECT_EQ(leafCount, r.leafCount());
        EXPECT_EQ(size_t(2), r.auxBuffersPerLeaf());
        EXPECT_EQ(size_t(2*leafCount),r.auxBufferCount());
        size_t n = 0;
        for (FloatTree::LeafCIter iter=tree.cbeginLeaf(); iter; ++iter, ++n) {
            EXPECT_TRUE(r.leaf(n) == *iter);
            EXPECT_TRUE(r.getBuffer(n,0) == iter->buffer());

            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) == r.getBuffer(n,2));
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
        EXPECT_EQ(r.leafCount(), n);
        for (n=0; n<leafCount; ++n) r.leaf(n).buffer().setValue(4,2.4f);
        for (n=0; n<leafCount; ++n) {
            EXPECT_TRUE(r.getBuffer(n,0) != r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) == r.getBuffer(n,2));
            EXPECT_TRUE(r.getBuffer(n,0) != r.getBuffer(n,2));
        }
        r.syncAllBuffers();
        for (n=0; n<leafCount; ++n) {
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) == r.getBuffer(n,2));
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
        for (n=0; n<leafCount; ++n) r.getBuffer(n,1).setValue(4,5.4f);
        for (n=0; n<leafCount; ++n) {
            EXPECT_TRUE(r.getBuffer(n,0) != r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) != r.getBuffer(n,2));
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
        EXPECT_TRUE(r.swapLeafBuffer(1));
        for (n=0; n<leafCount; ++n) {
            EXPECT_TRUE(r.getBuffer(n,0) != r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) == r.getBuffer(n,2));
            EXPECT_TRUE(r.getBuffer(n,0) != r.getBuffer(n,2));
        }
        r.syncAuxBuffer(1);
        for (n=0; n<leafCount; ++n) {
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) != r.getBuffer(n,2));
            EXPECT_TRUE(r.getBuffer(n,0) != r.getBuffer(n,2));
        }
        r.syncAuxBuffer(2);
        for (n=0; n<leafCount; ++n) {
            EXPECT_TRUE(r.getBuffer(n,0) == r.getBuffer(n,1));
            EXPECT_TRUE(r.getBuffer(n,1) == r.getBuffer(n,2));
        }
    }
    {// test with const tree (buffers are not swappable)
        openvdb::tree::LeafManager<const FloatTree> r(tree);

        for (size_t numAuxBuffers = 0; numAuxBuffers <= 2; ++numAuxBuffers += 2) {
            r.rebuildAuxBuffers(numAuxBuffers);

            EXPECT_EQ(leafCount, r.leafCount());
            EXPECT_EQ(int(numAuxBuffers * leafCount), int(r.auxBufferCount()));
            EXPECT_EQ(numAuxBuffers, r.auxBuffersPerLeaf());

            size_t n = 0;
            for (FloatTree::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter, ++n) {
                EXPECT_TRUE(r.leaf(n) == *iter);
                // Verify that each aux buffer was initialized with a copy of the leaf buffer.
                for (size_t bufIdx = 0; bufIdx < numAuxBuffers; ++bufIdx) {
                    EXPECT_TRUE(r.getBuffer(n, bufIdx) == iter->buffer());
                }
            }
            EXPECT_EQ(r.leafCount(), n);

            for (size_t i = 0; i < numAuxBuffers; ++i) {
                for (size_t j = 0; j < numAuxBuffers; ++j) {
                    // Verify that swapping buffers with themselves and swapping
                    // leaf buffers with aux buffers have no effect.
                    const bool canSwap = (i != j && i != 0 && j != 0);
                    EXPECT_EQ(canSwap, r.swapBuffer(i, j));
                }
            }
        }
    }
}

TEST_F(TestLeafManager, testActiveLeafVoxelCount)
{
    using namespace openvdb;

    for (const Int32 dim: { 87, 1023, 1024, 2023 }) {
        const CoordBBox denseBBox{Coord{0}, Coord{dim - 1}};
        const auto size = denseBBox.volume();

        // Create a large dense tree for testing but use a MaskTree to
        // minimize the memory overhead
        MaskTree tree{false};
        tree.denseFill(denseBBox, true, true);
        // Add some tiles, which should not contribute to the leaf voxel count.
        tree.addTile(/*level=*/2, Coord{10000}, true, true);
        tree.addTile(/*level=*/1, Coord{-10000}, true, true);
        tree.addTile(/*level=*/1, Coord{20000}, false, false);

        tree::LeafManager<MaskTree> mgr(tree);

        // On a dual CPU Intel(R) Xeon(R) E5-2697 v3 @ 2.60GHz
        // the speedup of LeafManager::activeLeafVoxelCount over
        // Tree::activeLeafVoxelCount is ~15x (assuming a LeafManager already exists)
        //openvdb::util::CpuTimer t("\nTree::activeVoxelCount");
        const auto treeActiveVoxels = tree.activeVoxelCount();
        //t.restart("\nTree::activeLeafVoxelCount");
        const auto treeActiveLeafVoxels = tree.activeLeafVoxelCount();
        //t.restart("\nLeafManager::activeLeafVoxelCount");
        const auto mgrActiveLeafVoxels = mgr.activeLeafVoxelCount();//multi-threaded
        //t.stop();
        //std::cerr << "Old1 = " << treeActiveVoxels << " old2 = " << treeActiveLeafVoxels
        //    << " New = " << mgrActiveLeafVoxels << std::endl;
        EXPECT_TRUE(size < treeActiveVoxels);
        EXPECT_EQ(size, treeActiveLeafVoxels);
        EXPECT_EQ(size, mgrActiveLeafVoxels);
    }
}

namespace {

struct ForeachOp
{
    ForeachOp(float v) : mV(v) {}
    template <typename T>
    void operator()(T &leaf, size_t) const
    {
        for (typename T::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
            if ( *iter > mV) iter.setValue( 2.0f );
        }
    }
    const float mV;
};// ForeachOp

struct ReduceOp
{
    ReduceOp(float v) : mV(v), mN(0) {}
    ReduceOp(const ReduceOp &other) : mV(other.mV), mN(other.mN) {}
    ReduceOp(const ReduceOp &other, tbb::split) : mV(other.mV), mN(0) {}
    template <typename T>
    void operator()(T &leaf, size_t)
    {
        for (typename T::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
            if ( *iter > mV) ++mN;
        }
    }
    void join(const ReduceOp &other) {mN += other.mN;}
    const float mV;
    openvdb::Index mN;
};// ReduceOp

}//unnamed namespace

TEST_F(TestLeafManager, testForeach)
{
    using namespace openvdb;

    FloatTree tree( 0.0f );
    const int dim = int(FloatTree::LeafNodeType::dim());
    const CoordBBox bbox1(Coord(0),Coord(dim-1));
    const CoordBBox bbox2(Coord(dim),Coord(2*dim-1));

    tree.fill( bbox1, -1.0f);
    tree.fill( bbox2,  1.0f);
    tree.voxelizeActiveTiles();

    for (CoordBBox::Iterator<true> iter(bbox1); iter; ++iter) {
        EXPECT_EQ( -1.0f, tree.getValue(*iter));
    }
    for (CoordBBox::Iterator<true> iter(bbox2); iter; ++iter) {
        EXPECT_EQ(  1.0f, tree.getValue(*iter));
    }

    tree::LeafManager<FloatTree> r(tree);
    EXPECT_EQ(size_t(2), r.leafCount());
    EXPECT_EQ(size_t(0), r.auxBufferCount());
    EXPECT_EQ(size_t(0), r.auxBuffersPerLeaf());

    ForeachOp op(0.0f);
    r.foreach(op);

    EXPECT_EQ(size_t(2), r.leafCount());
    EXPECT_EQ(size_t(0), r.auxBufferCount());
    EXPECT_EQ(size_t(0), r.auxBuffersPerLeaf());

    for (CoordBBox::Iterator<true> iter(bbox1); iter; ++iter) {
        EXPECT_EQ( -1.0f, tree.getValue(*iter));
    }
    for (CoordBBox::Iterator<true> iter(bbox2); iter; ++iter) {
        EXPECT_EQ(  2.0f, tree.getValue(*iter));
    }
}

TEST_F(TestLeafManager, testReduce)
{
    using namespace openvdb;

    FloatTree tree( 0.0f );
    const int dim = int(FloatTree::LeafNodeType::dim());
    const CoordBBox bbox1(Coord(0),Coord(dim-1));
    const CoordBBox bbox2(Coord(dim),Coord(2*dim-1));

    tree.fill( bbox1, -1.0f);
    tree.fill( bbox2,  1.0f);
    tree.voxelizeActiveTiles();

    for (CoordBBox::Iterator<true> iter(bbox1); iter; ++iter) {
        EXPECT_EQ( -1.0f, tree.getValue(*iter));
    }
    for (CoordBBox::Iterator<true> iter(bbox2); iter; ++iter) {
        EXPECT_EQ(  1.0f, tree.getValue(*iter));
    }

    tree::LeafManager<FloatTree> r(tree);
    EXPECT_EQ(size_t(2), r.leafCount());
    EXPECT_EQ(size_t(0), r.auxBufferCount());
    EXPECT_EQ(size_t(0), r.auxBuffersPerLeaf());

    ReduceOp op(0.0f);
    r.reduce(op);
    EXPECT_EQ(FloatTree::LeafNodeType::numValues(), op.mN);

    EXPECT_EQ(size_t(2), r.leafCount());
    EXPECT_EQ(size_t(0), r.auxBufferCount());
    EXPECT_EQ(size_t(0), r.auxBuffersPerLeaf());

    Index n = 0;
    for (CoordBBox::Iterator<true> iter(bbox1); iter; ++iter) {
        ++n;
        EXPECT_EQ( -1.0f, tree.getValue(*iter));
    }
    EXPECT_EQ(FloatTree::LeafNodeType::numValues(), n);

    n = 0;
    for (CoordBBox::Iterator<true> iter(bbox2); iter; ++iter) {
        ++n;
        EXPECT_EQ(  1.0f, tree.getValue(*iter));
    }
    EXPECT_EQ(FloatTree::LeafNodeType::numValues(), n);
}

TEST_F(TestLeafManager, testTreeConfigurations)
{
    using Tree2Type = openvdb::tree::Tree<
        openvdb::tree::RootNode<
        openvdb::tree::LeafNode<float, 3> > >;
    using Tree3Type = openvdb::tree::Tree3<float, 4, 3>::Type;
    using Tree4Type = openvdb::tree::Tree4<float, 5, 4, 3>::Type;
    using Tree5Type = openvdb::tree::Tree5<float, 5, 5, 4, 3>::Type;

    using TestConfigurations = openvdb::TypeList<
        Tree2Type, Tree3Type, Tree4Type, Tree5Type
    >;

    TestConfigurations::foreach([](auto tree) {
        using TreeType = typename std::decay<decltype(tree)>::type;
        using LeafNodeType = typename TreeType::LeafNodeType;
        using LeafManagerT = openvdb::tree::LeafManager<TreeType>;
        using ConstLeafManagerT = openvdb::tree::LeafManager<const TreeType>;

        // Add 20 leaf nodes and make sure they are constructed correctly
        constexpr openvdb::Int32 Count = 20;

        const openvdb::Int32 start = -(Count/2)*openvdb::Int32(LeafNodeType::DIM);
        const openvdb::Int32 end   =  (Count/2)*openvdb::Int32(LeafNodeType::DIM);
        for (openvdb::Int32 idx = start; idx < end; idx+=openvdb::Int32(LeafNodeType::DIM)) {
            tree.touchLeaf(openvdb::math::Coord(idx));
        }

        EXPECT_EQ(tree.leafCount(), Count);
        LeafManagerT manager(tree);
        EXPECT_EQ(manager.leafCount(), Count);
        ConstLeafManagerT cmanager(tree);
        EXPECT_EQ(cmanager.leafCount(), Count);
    });
}
