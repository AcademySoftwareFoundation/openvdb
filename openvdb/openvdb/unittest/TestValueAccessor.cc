// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/Prune.h>

#include <gtest/gtest.h>
#include <tbb/task_group.h>

#include <type_traits>


using Tree2Type = openvdb::tree::Tree<
    openvdb::tree::RootNode<
    openvdb::tree::LeafNode<float, 3> > >;
using Tree3Type = openvdb::tree::Tree<
    openvdb::tree::RootNode<
    openvdb::tree::InternalNode<
    openvdb::tree::LeafNode<float, 3>, 4> > >;
using Tree4Type = openvdb::tree::Tree4<float, 5, 4, 3>::Type;
using Tree5Type = openvdb::tree::Tree<
    openvdb::tree::RootNode<
    openvdb::tree::InternalNode<
    openvdb::tree::InternalNode<
    openvdb::tree::InternalNode<
    openvdb::tree::LeafNode<float, 3>, 4>, 5>, 5> > >;
using TreeType = Tree4Type;


// Recursive tree with equal Dim at every level
// Depth includes the root node, so Depth=2 will create a
// RootNode<LeafNode> tree type.
template<size_t Depth, typename NodeT, int Log2Dim>
struct RecursiveTreeBuilder;

template<typename NodeT, int Log2Dim>
struct RecursiveTreeBuilder<1, NodeT, Log2Dim>
{
    using Type = openvdb::tree::Tree<openvdb::tree::RootNode<NodeT>>;
};

template<size_t Depth, typename NodeT, int Log2Dim>
struct RecursiveTreeBuilder
{
    using Type = typename RecursiveTreeBuilder<
        Depth-1,
        openvdb::tree::InternalNode<NodeT, Log2Dim>,
        Log2Dim
    >::Type;
};

template<size_t Depth, typename ValueT, int Log2Dim = 1>
struct RecursiveGrid
{
    using GridType = typename openvdb::Grid<typename RecursiveTreeBuilder<
        Depth-1,
        openvdb::tree::LeafNode<ValueT, Log2Dim>,
        Log2Dim
    >::Type>;

    using TreeType     = typename GridType::TreeType;
    using AccessorType = typename GridType::Accessor;

    static_assert(TreeType::DEPTH == Depth);
};


using namespace openvdb::tree;


class TestValueAccessor: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }

protected:
    template<typename AccessorT> void accessorTest();
    template<typename AccessorT> void constAccessorTest();
    template<typename AccessorT> void multithreadedAccessorTest();
};


////////////////////////////////////////


namespace {

template <typename T>
struct Plus
{
    T addend;
    Plus(T f) : addend(f) {}
    inline void operator()(T& f) const { f += addend; }
    inline void operator()(T& f, bool& b) const { f += addend; b = false; }
};

}


template<typename AccessorT>
void
TestValueAccessor::accessorTest()
{
    using TreeType = typename AccessorT::TreeType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename AccessorT::ValueType;
    using RootNodeType = typename TreeType::RootNodeType;

    // Largest dim of a supported node by these tree
    static const size_t LDIM = RootNodeType::ChildNodeType::DIM;

    const int leafDepth = int(TreeType::DEPTH) - 1;
    // subtract one because getValueDepth() returns 0 for values at the root

    const ValueType background = static_cast<ValueType>(5.0f);
    const ValueType value = static_cast<ValueType>(-9.345f);
    const openvdb::Coord c0(0), c1(LDIM*5, LDIM*2, LDIM*3);

    {
        TreeType tree(background);
        EXPECT_TRUE(!tree.isValueOn(c0));
        EXPECT_TRUE(!tree.isValueOn(c1));
        EXPECT_EQ(background, tree.getValue(c0));
        EXPECT_EQ(background, tree.getValue(c1));
        tree.setValue(c0, value);
        EXPECT_TRUE(tree.isValueOn(c0));
        EXPECT_TRUE(!tree.isValueOn(c1));
        EXPECT_EQ(value, tree.getValue(c0));
        EXPECT_EQ(background, tree.getValue(c1));
    }
    {
        TreeType tree(background);
        AccessorT acc(tree);
        ValueType v;

        // test addLeaf
        acc.addLeaf(new LeafNodeType(c0));
        EXPECT_EQ(1ul, tree.leafCount());
        EXPECT_TRUE(acc.probeLeaf(c0));
        tree.clear();
        // unsafe accessors won't be automatically cleaned up
        if (acc.isSafe()) {
            EXPECT_TRUE(!acc.isCached(c0));
        }
        acc.clear();
        EXPECT_TRUE(!acc.isCached(c0));

        // test probeNode methods
        // @todo improve based on what levels we're caching
        EXPECT_TRUE(!acc.template probeNode<LeafNodeType>(c0));
        EXPECT_TRUE(!acc.template probeConstNode<LeafNodeType>(c0));
        EXPECT_TRUE(!acc.isCached(c0));

        EXPECT_TRUE(acc.touchLeaf(c0));
        EXPECT_TRUE(acc.template probeConstNode<LeafNodeType>(c0));
        EXPECT_TRUE(acc.template probeNode<LeafNodeType>(c0));
        // test we can access other nodes which have to have been created
        // (may simply do the above for 3 level trees where ChildNodeType == LeafNodeType)
        EXPECT_TRUE(acc.template probeNode<typename TreeType::RootNodeType::ChildNodeType>(c0));
        EXPECT_TRUE(acc.template probeConstNode<typename TreeType::RootNodeType::ChildNodeType>(c0));
        tree.clear();
        acc.clear();

        // test addTile
        // @todo improve based on what levels we're caching
        acc.addTile(/*level=*/1, c0, value, /*state=*/true);
        EXPECT_EQ(1ul, tree.activeTileCount());
        EXPECT_TRUE(acc.isValueOn(c0));
        tree.clear();
        acc.clear();

        //
        EXPECT_TRUE(!tree.isValueOn(c0));
        EXPECT_TRUE(!tree.isValueOn(c1));
        EXPECT_EQ(background, tree.getValue(c0));
        EXPECT_EQ(background, tree.getValue(c1));
        EXPECT_TRUE(!acc.isCached(c0));
        EXPECT_TRUE(!acc.isCached(c1));
        EXPECT_TRUE(!acc.probeValue(c0,v));
        EXPECT_EQ(background, v);
        EXPECT_TRUE(!acc.probeValue(c1,v));
        EXPECT_EQ(background, v);
        EXPECT_EQ(-1, acc.getValueDepth(c0));
        EXPECT_EQ(-1, acc.getValueDepth(c1));
        EXPECT_TRUE(!acc.isVoxel(c0));
        EXPECT_TRUE(!acc.isVoxel(c1));

        acc.setValue(c0, value);

        EXPECT_TRUE(tree.isValueOn(c0));
        EXPECT_TRUE(!tree.isValueOn(c1));
        EXPECT_EQ(value, tree.getValue(c0));
        EXPECT_EQ(background, tree.getValue(c1));
        EXPECT_TRUE(acc.probeValue(c0,v));
        EXPECT_EQ(value, v);
        EXPECT_TRUE(!acc.probeValue(c1,v));
        EXPECT_EQ(background, v);
        EXPECT_EQ(leafDepth, acc.getValueDepth(c0)); // leaf-level voxel value
        EXPECT_EQ(-1, acc.getValueDepth(c1)); // background value

        auto leaf = acc.probeLeaf(c0);
        ASSERT_TRUE(leaf);
        EXPECT_EQ(leafDepth, acc.getValueDepth(leaf->origin() + openvdb::Coord(leaf->dim()-1))); // resides in same node
        const int depth = leafDepth == 1 ? -1 : leafDepth - 1;
        EXPECT_EQ(depth, acc.getValueDepth(leaf->origin() + openvdb::Coord(leaf->dim()))); // outside of any child node
        EXPECT_TRUE( acc.isVoxel(c0)); // leaf-level voxel value
        EXPECT_TRUE(!acc.isVoxel(c1));
        EXPECT_TRUE( acc.isVoxel(leaf->origin() + openvdb::Coord(leaf->dim()-1)));
        EXPECT_TRUE(!acc.isVoxel(leaf->origin() + openvdb::Coord(leaf->dim())));
        leaf = nullptr;

        EXPECT_EQ(background, acc.getValue(c1));
        EXPECT_TRUE(!acc.isCached(c1)); // uncached background value
        EXPECT_TRUE(!acc.isValueOn(c1)); // inactive background value
        EXPECT_EQ(value, acc.getValue(c0));
        EXPECT_TRUE(
            (acc.NumCacheLevels>0) == acc.isCached(c0)); // active, leaf-level voxel value
        EXPECT_TRUE(acc.isValueOn(c0));

        acc.setValue(c1, value);

        EXPECT_TRUE(acc.isValueOn(c1));
        EXPECT_EQ(value, tree.getValue(c0));
        EXPECT_EQ(value, tree.getValue(c1));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c1));
        EXPECT_EQ(value, acc.getValue(c1));
        EXPECT_TRUE(!acc.isCached(c0));
        EXPECT_EQ(value, acc.getValue(c0));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c0));
        EXPECT_EQ(leafDepth, acc.getValueDepth(c0));
        EXPECT_EQ(leafDepth, acc.getValueDepth(c1));
        EXPECT_TRUE(acc.isVoxel(c0));
        EXPECT_TRUE(acc.isVoxel(c1));

        tree.setValueOff(c1);

        EXPECT_EQ(value, tree.getValue(c0));
        EXPECT_EQ(value, tree.getValue(c1));
        EXPECT_TRUE(!acc.isCached(c0));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c1));
        EXPECT_TRUE( acc.isValueOn(c0));
        EXPECT_TRUE(!acc.isValueOn(c1));

        acc.setValueOn(c1);

        EXPECT_TRUE(!acc.isCached(c0));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c1));
        EXPECT_TRUE( acc.isValueOn(c0));
        EXPECT_TRUE( acc.isValueOn(c1));

        acc.modifyValueAndActiveState(c1, Plus<ValueType>(-value)); // subtract value & mark inactive
        EXPECT_TRUE(!acc.isValueOn(c1));

        acc.modifyValue(c1, Plus<ValueType>(-value)); // subtract value again & mark active

        EXPECT_TRUE(acc.isValueOn(c1));
        EXPECT_EQ(value, tree.getValue(c0));
        EXPECT_EQ(-value, tree.getValue(c1));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c1));
        EXPECT_EQ(-value, acc.getValue(c1));
        EXPECT_TRUE(!acc.isCached(c0));
        EXPECT_EQ(value, acc.getValue(c0));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c0));
        EXPECT_EQ(leafDepth, acc.getValueDepth(c0));
        EXPECT_EQ(leafDepth, acc.getValueDepth(c1));
        EXPECT_TRUE(acc.isVoxel(c0));
        EXPECT_TRUE(acc.isVoxel(c1));

        acc.setValueOnly(c1, 3*value);

        EXPECT_TRUE(acc.isValueOn(c1));
        EXPECT_EQ(value, tree.getValue(c0));
        EXPECT_EQ(3*value, tree.getValue(c1));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c1));
        EXPECT_EQ(3*value, acc.getValue(c1));
        EXPECT_TRUE(!acc.isCached(c0));
        EXPECT_EQ(value, acc.getValue(c0));
        EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c0));
        EXPECT_EQ(leafDepth, acc.getValueDepth(c0));
        EXPECT_EQ(leafDepth, acc.getValueDepth(c1));
        EXPECT_TRUE(acc.isVoxel(c0));
        EXPECT_TRUE(acc.isVoxel(c1));

        acc.clear();
        EXPECT_TRUE(!acc.isCached(c0));
        EXPECT_TRUE(!acc.isCached(c1));
    }
}


template<typename AccessorT>
void
TestValueAccessor::constAccessorTest()
{
    using TreeType = typename std::remove_const<typename AccessorT::TreeType>::type;
    using ValueType = typename TreeType::ValueType;

    const int leafDepth = int(TreeType::DEPTH) - 1;
        // subtract one because getValueDepth() returns 0 for values at the root

    const ValueType background = 5.0f, value = -9.345f;
    const openvdb::Coord c0(5, 10, 20), c1(500000, 200000, 300000);
    ValueType v;

    TreeType tree(background);
    AccessorT acc(tree);

    EXPECT_TRUE(!tree.isValueOn(c0));
    EXPECT_TRUE(!tree.isValueOn(c1));
    EXPECT_EQ(background, tree.getValue(c0));
    EXPECT_EQ(background, tree.getValue(c1));
    EXPECT_TRUE(!acc.isCached(c0));
    EXPECT_TRUE(!acc.isCached(c1));
    EXPECT_TRUE(!acc.probeValue(c0,v));
    EXPECT_EQ(background, v);
    EXPECT_TRUE(!acc.probeValue(c1,v));
    EXPECT_EQ(background, v);
    EXPECT_EQ(-1, acc.getValueDepth(c0));
    EXPECT_EQ(-1, acc.getValueDepth(c1));
    EXPECT_TRUE(!acc.isVoxel(c0));
    EXPECT_TRUE(!acc.isVoxel(c1));

    tree.setValue(c0, value);

    EXPECT_TRUE(tree.isValueOn(c0));
    EXPECT_TRUE(!tree.isValueOn(c1));
    EXPECT_EQ(background, acc.getValue(c1));
    EXPECT_TRUE(!acc.isCached(c1));
    EXPECT_TRUE(!acc.isCached(c0));
    EXPECT_TRUE(acc.isValueOn(c0));
    EXPECT_TRUE(!acc.isValueOn(c1));
    EXPECT_TRUE(acc.probeValue(c0,v));
    EXPECT_EQ(value, v);
    EXPECT_TRUE(!acc.probeValue(c1,v));
    EXPECT_EQ(background, v);
    EXPECT_EQ(leafDepth, acc.getValueDepth(c0));
    EXPECT_EQ(-1, acc.getValueDepth(c1));
    EXPECT_TRUE( acc.isVoxel(c0));
    EXPECT_TRUE(!acc.isVoxel(c1));

    EXPECT_EQ(value, acc.getValue(c0));
    EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c0));
    EXPECT_EQ(background, acc.getValue(c1));
    EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c0));
    EXPECT_TRUE(!acc.isCached(c1));
    EXPECT_TRUE(acc.isValueOn(c0));
    EXPECT_TRUE(!acc.isValueOn(c1));

    tree.setValue(c1, value);

    EXPECT_EQ(value, acc.getValue(c1));
    EXPECT_TRUE(!acc.isCached(c0));
    EXPECT_TRUE((acc.NumCacheLevels>0) == acc.isCached(c1));
    EXPECT_TRUE(acc.isValueOn(c0));
    EXPECT_TRUE(acc.isValueOn(c1));
    EXPECT_EQ(leafDepth, acc.getValueDepth(c0));
    EXPECT_EQ(leafDepth, acc.getValueDepth(c1));
    EXPECT_TRUE(acc.isVoxel(c0));
    EXPECT_TRUE(acc.isVoxel(c1));

    // The next two lines should not compile, because the acc references a const tree:
    //acc.setValue(c1, value);
    //acc.setValueOff(c1);

    acc.clear();
    EXPECT_TRUE(!acc.isCached(c0));
    EXPECT_TRUE(!acc.isCached(c1));
}


template<typename AccessorT>
void
TestValueAccessor::multithreadedAccessorTest()
{
#define MAX_COORD 5000

    using TreeType = typename AccessorT::TreeType;

    // Task to perform multiple reads through a shared accessor
    struct ReadTask {
        AccessorT& acc;
        ReadTask(AccessorT& c): acc(c) {}
        void execute()
        {
            for (int i = -MAX_COORD; i < MAX_COORD; ++i) {
                EXPECT_EQ(double(i), acc.getValue(openvdb::Coord(i)));
            }
        }
    };
    // Task to perform multiple writes through a shared accessor
    struct WriteTask {
        AccessorT& acc;
        WriteTask(AccessorT& c): acc(c) {}
        void execute()
        {
            for (int i = -MAX_COORD; i < MAX_COORD; ++i) {
                float f = acc.getValue(openvdb::Coord(i));
                EXPECT_EQ(float(i), f);
                acc.setValue(openvdb::Coord(i), float(i));
                EXPECT_EQ(float(i), acc.getValue(openvdb::Coord(i)));
            }
        }
    };
    // Parent task to spawn multiple parallel read and write tasks
    struct RootTask {
        AccessorT& acc;
        RootTask(AccessorT& c): acc(c) {}
        void execute()
        {
            tbb::task_group tasks;
            for (int i = 0; i < 3; ++i) {
                tasks.run([&] { ReadTask r(acc); r.execute(); });
                tasks.run([&] { WriteTask w(acc); w.execute(); });
            }
            tasks.wait();
        }
    };

    TreeType tree(/*background=*/0.5);
    AccessorT acc(tree);
    // Populate the tree.
    for (int i = -MAX_COORD; i < MAX_COORD; ++i) {
        acc.setValue(openvdb::Coord(i), float(i));
    }

    // Run multiple read and write tasks in parallel.
    RootTask root(acc);
    root.execute();

#undef MAX_COORD
}

/// Static assert class sizes
template <typename Type, size_t Expect, size_t Actual = sizeof(Type)>
struct CheckClassSize { static_assert(Expect == Actual); };

// Build a type list of all accessor types and make sure they are all the
// expected size.
template <typename GridT> using GridToAccCheckA =
    CheckClassSize<typename GridT::Accessor, 96ul>;
template <typename GridT> using GridToAccCheckB =
    CheckClassSize<typename GridT::Accessor, 88ul>;

void StaticAsssertVASizes()
{
    // Accessors with a leaf buffer cache have an extra pointer
    using AccessorsWithLeafCache =
        openvdb::GridTypes
            ::Remove<openvdb::BoolGrid>
            ::Remove<openvdb::MaskGrid>
            ::Transform<GridToAccCheckA>;

    // Accessors without a leaf buffer cache
    using AccessorsWithoutLeafCache =
        openvdb::TypeList<openvdb::BoolGrid, openvdb::MaskGrid>
            ::Transform<GridToAccCheckB>;

    // instantiate these types to force the static check
    [[maybe_unused]] AccessorsWithLeafCache::AsTupleList test;
    [[maybe_unused]] AccessorsWithoutLeafCache::AsTupleList test2;
}

// cache all node levels
TEST_F(TestValueAccessor, testTree2Accessor)        { accessorTest<ValueAccessor<Tree2Type> >(); }
TEST_F(TestValueAccessor, testTree2AccessorRW)      { accessorTest<ValueAccessorRW<Tree2Type> >(); }
TEST_F(TestValueAccessor, testTree2ConstAccessor)   { constAccessorTest<ValueAccessor<const Tree2Type> >(); }
TEST_F(TestValueAccessor, testTree2ConstAccessorRW) { constAccessorTest<ValueAccessorRW<const Tree2Type> >(); }

// cache all node levels
TEST_F(TestValueAccessor, testTree3Accessor)        { accessorTest<ValueAccessor<Tree3Type> >(); }
TEST_F(TestValueAccessor, testTree3AccessorRW)      { accessorTest<ValueAccessorRW<Tree3Type> >(); }
TEST_F(TestValueAccessor, testTree3ConstAccessor)   { constAccessorTest<ValueAccessor<const Tree3Type> >(); }
TEST_F(TestValueAccessor, testTree3ConstAccessorRW) { constAccessorTest<ValueAccessorRW<const Tree3Type> >(); }

// cache all node levels
TEST_F(TestValueAccessor, testTree4Accessor)        { accessorTest<ValueAccessor<Tree4Type> >(); }
TEST_F(TestValueAccessor, testTree4AccessorRW)      { accessorTest<ValueAccessorRW<Tree4Type> >(); }
TEST_F(TestValueAccessor, testTree4ConstAccessor)   { constAccessorTest<ValueAccessor<const Tree4Type> >(); }
TEST_F(TestValueAccessor, testTree4ConstAccessorRW) { constAccessorTest<ValueAccessorRW<const Tree4Type> >(); }

// cache all node levels
TEST_F(TestValueAccessor, testTree5Accessor)        { accessorTest<ValueAccessor<Tree5Type> >(); }
TEST_F(TestValueAccessor, testTree5AccessorRW)      { accessorTest<ValueAccessorRW<Tree5Type> >(); }
TEST_F(TestValueAccessor, testTree5ConstAccessor)   { constAccessorTest<ValueAccessor<const Tree5Type> >(); }
TEST_F(TestValueAccessor, testTree5ConstAccessorRW) { constAccessorTest<ValueAccessorRW<const Tree5Type> >(); }

// Test different tree configurations with their default accessors
TEST_F(TestValueAccessor, testTreeRecursive2)       { accessorTest<RecursiveGrid<2, float>::AccessorType>(); }
TEST_F(TestValueAccessor, testTreeRecursive4)       { accessorTest<RecursiveGrid<4, int32_t>::AccessorType>(); }
TEST_F(TestValueAccessor, testTreeRecursive5)       { accessorTest<RecursiveGrid<5, double>::AccessorType>(); }
TEST_F(TestValueAccessor, testTreeRecursive6)       { accessorTest<RecursiveGrid<6, int64_t>::AccessorType>(); }
TEST_F(TestValueAccessor, testTreeRecursive7)       { accessorTest<RecursiveGrid<7, float>::AccessorType>(); }
TEST_F(TestValueAccessor, testTreeRecursive8)       { accessorTest<RecursiveGrid<8, int32_t>::AccessorType>(); }

// Test odd combinations of trees and ValueAccessors
// cache node level 0 and 1
TEST_F(TestValueAccessor, testTree3Accessor2)
{
    accessorTest<ValueAccessor<Tree3Type, true,  2> >();
    accessorTest<ValueAccessor<Tree3Type, false, 2> >();
}

TEST_F(TestValueAccessor, testTree3ConstAccessor2)
{
    constAccessorTest<ValueAccessor<const Tree3Type, true,  2> >();
    constAccessorTest<ValueAccessor<const Tree3Type, false, 2> >();
}

TEST_F(TestValueAccessor, testTree4Accessor2)
{
    accessorTest<ValueAccessor<Tree4Type, true,  2> >();
    accessorTest<ValueAccessor<Tree4Type, false, 2> >();
}

TEST_F(TestValueAccessor, testTree4ConstAccessor2)
{
    constAccessorTest<ValueAccessor<const Tree4Type, true,  2> >();
    constAccessorTest<ValueAccessor<const Tree4Type, false, 2> >();
}

TEST_F(TestValueAccessor, testTree5Accessor2)
{
    accessorTest<ValueAccessor<Tree5Type, true,  2> >();
    accessorTest<ValueAccessor<Tree5Type, false, 2> >();
}

TEST_F(TestValueAccessor, testTree5ConstAccessor2)
{
    constAccessorTest<ValueAccessor<const Tree5Type, true,  2> >();
    constAccessorTest<ValueAccessor<const Tree5Type, false, 2> >();
}

// only cache leaf level
TEST_F(TestValueAccessor, testTree4Accessor1)
{
    accessorTest<ValueAccessor<Tree5Type, true,  1> >();
    accessorTest<ValueAccessor<Tree5Type, false, 1> >();
}

TEST_F(TestValueAccessor, testTree4ConstAccessor1)
{
    constAccessorTest<ValueAccessor<const Tree5Type, true,  1> >();
    constAccessorTest<ValueAccessor<const Tree5Type, false, 1> >();
}

// disable node caching
TEST_F(TestValueAccessor, testTree4Accessor0)
{
    accessorTest<ValueAccessor<Tree5Type, true,  0> >();
    accessorTest<ValueAccessor<Tree5Type, false, 0> >();
}

TEST_F(TestValueAccessor, testTree4ConstAccessor0)
{
    constAccessorTest<ValueAccessor<const Tree5Type, true,  0> >();
    constAccessorTest<ValueAccessor<const Tree5Type, false, 0> >();
}

//cache node level 2
TEST_F(TestValueAccessor, testTree4Accessor12)
{
    accessorTest<ValueAccessor1<Tree4Type, true,  2> >();
    accessorTest<ValueAccessor1<Tree4Type, false, 2> >();
}

//cache node level 1 and 3
TEST_F(TestValueAccessor, testTree5Accessor213)
{
    accessorTest<ValueAccessor2<Tree5Type, true,  1, 3> >();
    accessorTest<ValueAccessor2<Tree5Type, false, 1, 3> >();
}

TEST_F(TestValueAccessor, testMultiThreadedRWAccessors)
{
    multithreadedAccessorTest<ValueAccessorRW<Tree2Type>>();
    multithreadedAccessorTest<ValueAccessorRW<Tree3Type>>();
    multithreadedAccessorTest<ValueAccessorRW<Tree4Type>>();
    multithreadedAccessorTest<ValueAccessorRW<Tree5Type>>();

    // @todo also test a std compatible mutex
    // using ValueAccessorStdMutex = ValueAccessor<Tree4Type, true, Tree4Type::DEPTH-1, std::mutex>;
    //multithreadedAccessorTest<ValueAccessorStdMutex>();
}


TEST_F(TestValueAccessor, testAccessorRegistration)
{
    using openvdb::Index;

    const float background = 5.0f, value = -9.345f;
    const openvdb::Coord c0(5, 10, 20);

    openvdb::FloatTree::Ptr tree(new openvdb::FloatTree(background));
    openvdb::tree::ValueAccessor<openvdb::FloatTree> acc(*tree);

    // Set a single leaf voxel via the accessor and verify that
    // the cache is populated.
    acc.setValue(c0, value);
    EXPECT_EQ(Index(1), tree->leafCount());
    EXPECT_EQ(tree->root().getLevel(), tree->nonLeafCount());
    EXPECT_TRUE(acc.getNode<openvdb::FloatTree::LeafNodeType>() != nullptr);

    // Reset the voxel to the background value and verify that no nodes
    // have been deleted and that the cache is still populated.
    tree->setValueOff(c0, background);
    EXPECT_EQ(Index(1), tree->leafCount());
    EXPECT_EQ(tree->root().getLevel(), tree->nonLeafCount());
    EXPECT_TRUE(acc.getNode<openvdb::FloatTree::LeafNodeType>() != nullptr);

    // Prune the tree and verify that only the root node remains and that
    // the cache has been cleared.
    openvdb::tools::prune(*tree);
    //tree->prune();
    EXPECT_EQ(Index(0), tree->leafCount());
    EXPECT_EQ(Index(1), tree->nonLeafCount()); // root node only
    EXPECT_TRUE(acc.getNode<openvdb::FloatTree::LeafNodeType>() == nullptr);

    // Set the leaf voxel again and verify that the cache is repopulated.
    acc.setValue(c0, value);
    EXPECT_EQ(Index(1), tree->leafCount());
    EXPECT_EQ(tree->root().getLevel(), tree->nonLeafCount());
    EXPECT_TRUE(acc.getNode<openvdb::FloatTree::LeafNodeType>() != nullptr);

    // Delete the tree and verify that the cache has been cleared.
    tree.reset();
    EXPECT_TRUE(acc.getTree() == nullptr);
    EXPECT_TRUE(acc.getNode<openvdb::FloatTree::RootNodeType>() == nullptr);
    EXPECT_TRUE(acc.getNode<openvdb::FloatTree::LeafNodeType>() == nullptr);
}


TEST_F(TestValueAccessor, testGetNode)
{
    using LeafT = Tree4Type::LeafNodeType;

    const LeafT::ValueType background = 5.0f, value = -9.345f;
    const openvdb::Coord c0(5, 10, 20);

    Tree4Type tree(background);
    tree.setValue(c0, value);
    {
        openvdb::tree::ValueAccessor<Tree4Type> acc(tree);
        // Prime the cache.
        acc.getValue(c0);
        // Verify that the cache contains a leaf node.
        LeafT* node = acc.getNode<LeafT>();
        EXPECT_TRUE(node != nullptr);

        // Erase the leaf node from the cache and verify that it is gone.
        acc.eraseNode<LeafT>();
        node = acc.getNode<LeafT>();
        EXPECT_TRUE(node == nullptr);
    }
    {
        // As above, but with a const tree.
        openvdb::tree::ValueAccessor<const Tree4Type> acc(tree);
        acc.getValue(c0);
        const LeafT* node = acc.getNode<const LeafT>();
        EXPECT_TRUE(node != nullptr);

        acc.eraseNode<LeafT>();
        node = acc.getNode<const LeafT>();
        EXPECT_TRUE(node == nullptr);
    }
}

#if OPENVDB_ABI_VERSION_NUMBER >= 10

template <typename TreeT> struct AssertBypass
{
    inline void operator()() {
        static_assert(TreeT::Accessor::BypassLeafAPI);
        static_assert(TreeT::ConstAccessor::BypassLeafAPI);
    }
};

TEST_F(TestValueAccessor, testBypassLeafAPI)
{
    using namespace openvdb;

    // Assert default types bypass
    GridTypes::Remove<BoolGrid, MaskGrid>::foreach<AssertBypass>();
    // Bool/Mask grids don't use delay loading and their buffers can't
    // be ptr accessed as they use bit/word storage
    static_assert(!BoolGrid::Accessor::BypassLeafAPI);
    static_assert(!BoolGrid::ConstAccessor::BypassLeafAPI);
    static_assert(!MaskGrid::Accessor::BypassLeafAPI);
    static_assert(!MaskGrid::ConstAccessor::BypassLeafAPI);

    // Check some custom instantiations
    static_assert(ValueAccessor1<FloatTree, true, 0>::BypassLeafAPI);
    static_assert(ValueAccessor2<FloatTree, true, 0, 1>::BypassLeafAPI);
    static_assert(ValueAccessor3<FloatTree, true, 0, 1, 2>::BypassLeafAPI);
    static_assert(ValueAccessor3<FloatTree, false, 0, 1, 2>::BypassLeafAPI);
    static_assert(ValueAccessor<FloatTree, true, 1>::BypassLeafAPI);
    static_assert(ValueAccessor<FloatTree, false, 2>::BypassLeafAPI);
    //static_assert(ValueAccessor<FloatTree, true, 3, std::mutex>::BypassLeafAPI);

    // These don't cache leaf nodes
    static_assert(!ValueAccessor0<FloatTree, true>::BypassLeafAPI);
    static_assert(!ValueAccessor1<FloatTree, true, 1>::BypassLeafAPI);
    static_assert(!ValueAccessor2<FloatTree, true, 1, 2>::BypassLeafAPI);
    static_assert(!ValueAccessor3<MaskTree, true, 0, 1, 2>::BypassLeafAPI);
}

#endif
