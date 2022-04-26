// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file TestTreeVisitor.h
///
/// @author Peter Cucka

#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <sstream>
#include <type_traits>


class TestTreeVisitor: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }

    void testVisitTreeBool() { visitTree<openvdb::BoolTree>(); }
    void testVisitTreeInt32() { visitTree<openvdb::Int32Tree>(); }
    void testVisitTreeFloat() { visitTree<openvdb::FloatTree>(); }
    void testVisitTreeVec2I() { visitTree<openvdb::Vec2ITree>(); }
    void testVisitTreeVec3S() { visitTree<openvdb::VectorTree>(); }
    void testVisit2Trees();

protected:
    template<typename TreeT> TreeT createTestTree() const;
    template<typename TreeT> void visitTree();
};


////////////////////////////////////////


template<typename TreeT>
TreeT
TestTreeVisitor::createTestTree() const
{
    using ValueT = typename TreeT::ValueType;
    const ValueT zero = openvdb::zeroVal<ValueT>(), one = zero + 1;

    // Create a sparse test tree comprising the eight corners of
    // a 200 x 200 x 200 cube.
    TreeT tree(/*background=*/one);
    tree.setValue(openvdb::Coord(  0,   0,   0),  /*value=*/zero);
    tree.setValue(openvdb::Coord(200,   0,   0),  zero);
    tree.setValue(openvdb::Coord(  0, 200,   0),  zero);
    tree.setValue(openvdb::Coord(  0,   0, 200),  zero);
    tree.setValue(openvdb::Coord(200,   0, 200),  zero);
    tree.setValue(openvdb::Coord(  0, 200, 200),  zero);
    tree.setValue(openvdb::Coord(200, 200,   0),  zero);
    tree.setValue(openvdb::Coord(200, 200, 200),  zero);

    // Verify that the bounding box of all On values is 200 x 200 x 200.
    openvdb::CoordBBox bbox;
    EXPECT_TRUE(tree.evalActiveVoxelBoundingBox(bbox));
    EXPECT_TRUE(bbox.min() == openvdb::Coord(0, 0, 0));
    EXPECT_TRUE(bbox.max() == openvdb::Coord(200, 200, 200));

    return tree;
}


////////////////////////////////////////


namespace {

/// Single-tree visitor that accumulates node counts
class Visitor
{
public:
    using NodeMap = std::map<openvdb::Index, std::set<const void*> >;

    Visitor(): mSkipLeafNodes(false) { reset(); }

    void reset()
    {
        mSkipLeafNodes = false;
        mNodes.clear();
        mNonConstIterUseCount = mConstIterUseCount = 0;
    }

    void setSkipLeafNodes(bool b) { mSkipLeafNodes = b; }

    template<typename IterT>
    bool operator()(IterT& iter)
    {
        incrementIterUseCount(std::is_const<typename IterT::NodeType>::value);
        EXPECT_TRUE(iter.getParentNode() != nullptr);

        if (mSkipLeafNodes && iter.parent().getLevel() == 1) return true;

        using ValueT = typename IterT::NonConstValueType;
        using ChildT = typename IterT::ChildNodeType;
        ValueT value;
        if (const ChildT* child = iter.probeChild(value)) {
            insertChild<ChildT>(child);
        }
        return false;
    }

    openvdb::Index leafCount() const
    {
        NodeMap::const_iterator it = mNodes.find(0);
        return openvdb::Index((it != mNodes.end()) ? it->second.size() : 0);
    }
    openvdb::Index nonLeafCount() const
    {
        openvdb::Index count = 1; // root node
        for (NodeMap::const_iterator i = mNodes.begin(), e = mNodes.end(); i != e; ++i) {
            if (i->first != 0) count = openvdb::Index(count + i->second.size());
        }
        return count;
    }

    bool usedOnlyConstIterators() const
    {
        return (mConstIterUseCount > 0 && mNonConstIterUseCount == 0);
    }
    bool usedOnlyNonConstIterators() const
    {
        return (mConstIterUseCount == 0 && mNonConstIterUseCount > 0);
    }

private:
    template<typename ChildT>
    void insertChild(const ChildT* child)
    {
        if (child != nullptr) {
            const openvdb::Index level = child->getLevel();
            if (!mSkipLeafNodes || level > 0) {
                mNodes[level].insert(child);
            }
        }
    }

    void incrementIterUseCount(bool isConst)
    {
        if (isConst) ++mConstIterUseCount; else ++mNonConstIterUseCount;
    }

    bool mSkipLeafNodes;
    NodeMap mNodes;
    int mNonConstIterUseCount, mConstIterUseCount;
};

/// Specialization for LeafNode iterators, whose ChildNodeType is void
/// (therefore can't call child->getLevel())
template<> inline void Visitor::insertChild<void>(const void*) {}

} // unnamed namespace


template<typename TreeT>
void
TestTreeVisitor::visitTree()
{
    OPENVDB_NO_DEPRECATION_WARNING_BEGIN

    TreeT tree = createTestTree<TreeT>();
    {
        // Traverse the tree, accumulating node counts.
        Visitor visitor;
        const_cast<const TreeT&>(tree).visit(visitor);

        EXPECT_TRUE(visitor.usedOnlyConstIterators());
        EXPECT_EQ(tree.leafCount(), visitor.leafCount());
        EXPECT_EQ(tree.nonLeafCount(), visitor.nonLeafCount());
    }
    {
        // Traverse the tree, accumulating node counts as above,
        // but using non-const iterators.
        Visitor visitor;
        tree.visit(visitor);

        EXPECT_TRUE(visitor.usedOnlyNonConstIterators());
        EXPECT_EQ(tree.leafCount(), visitor.leafCount());
        EXPECT_EQ(tree.nonLeafCount(), visitor.nonLeafCount());
    }
    {
        // Traverse the tree, accumulating counts of non-leaf nodes only.
        Visitor visitor;
        visitor.setSkipLeafNodes(true);
        const_cast<const TreeT&>(tree).visit(visitor);

        EXPECT_TRUE(visitor.usedOnlyConstIterators());
        EXPECT_EQ(0U, visitor.leafCount()); // leaf nodes were skipped
        EXPECT_EQ(tree.nonLeafCount(), visitor.nonLeafCount());
    }

    OPENVDB_NO_DEPRECATION_WARNING_END
}


////////////////////////////////////////


namespace {

/// Two-tree visitor that accumulates node counts
class Visitor2
{
public:
    using NodeMap = std::map<openvdb::Index, std::set<const void*> >;

    Visitor2() { reset(); }

    void reset()
    {
        mSkipALeafNodes = mSkipBLeafNodes = false;
        mANodeCount.clear();
        mBNodeCount.clear();
    }

    void setSkipALeafNodes(bool b) { mSkipALeafNodes = b; }
    void setSkipBLeafNodes(bool b) { mSkipBLeafNodes = b; }

    openvdb::Index aLeafCount() const { return leafCount(/*useA=*/true); }
    openvdb::Index bLeafCount() const { return leafCount(/*useA=*/false); }
    openvdb::Index aNonLeafCount() const { return nonLeafCount(/*useA=*/true); }
    openvdb::Index bNonLeafCount() const { return nonLeafCount(/*useA=*/false); }

    template<typename AIterT, typename BIterT>
    int operator()(AIterT& aIter, BIterT& bIter)
    {
        EXPECT_TRUE(aIter.getParentNode() != nullptr);
        EXPECT_TRUE(bIter.getParentNode() != nullptr);

        typename AIterT::NodeType& aNode = aIter.parent();
        typename BIterT::NodeType& bNode = bIter.parent();

        const openvdb::Index aLevel = aNode.getLevel(), bLevel = bNode.getLevel();
        mANodeCount[aLevel].insert(&aNode);
        mBNodeCount[bLevel].insert(&bNode);

        int skipBranch = 0;
        if (aLevel == 1 && mSkipALeafNodes) skipBranch = (skipBranch | 1);
        if (bLevel == 1 && mSkipBLeafNodes) skipBranch = (skipBranch | 2);
        return skipBranch;
    }

private:
    openvdb::Index leafCount(bool useA) const
    {
        const NodeMap& theMap = (useA ? mANodeCount : mBNodeCount);
        NodeMap::const_iterator it = theMap.find(0);
        if (it != theMap.end()) return openvdb::Index(it->second.size());
        return 0;
    }
    openvdb::Index nonLeafCount(bool useA) const
    {
        openvdb::Index count = 0;
        const NodeMap& theMap = (useA ? mANodeCount : mBNodeCount);
        for (NodeMap::const_iterator i = theMap.begin(), e = theMap.end(); i != e; ++i) {
            if (i->first != 0) count = openvdb::Index(count + i->second.size());
        }
        return count;
    }

    bool mSkipALeafNodes, mSkipBLeafNodes;
    NodeMap mANodeCount, mBNodeCount;
};

} // unnamed namespace


TEST_F(TestTreeVisitor, testVisitTreeBool) { visitTree<openvdb::BoolTree>(); }
TEST_F(TestTreeVisitor, testVisitTreeInt32) { visitTree<openvdb::Int32Tree>(); }
TEST_F(TestTreeVisitor, testVisitTreeFloat) { visitTree<openvdb::FloatTree>(); }
TEST_F(TestTreeVisitor, testVisitTreeVec2I) { visitTree<openvdb::Vec2ITree>(); }
TEST_F(TestTreeVisitor, testVisitTreeVec3S) { visitTree<openvdb::VectorTree>(); }

TEST_F(TestTreeVisitor, testVisit2Trees)
{
    OPENVDB_NO_DEPRECATION_WARNING_BEGIN

    using TreeT = openvdb::FloatTree;
    using Tree2T = openvdb::VectorTree;
    using ValueT = TreeT::ValueType;

    // Create a test tree.
    TreeT tree = createTestTree<TreeT>();
    // Create another test tree of a different type but with the same topology.
    Tree2T tree2 = createTestTree<Tree2T>();

    // Traverse both trees.
    Visitor2 visitor;
    tree.visit2(tree2, visitor);

    //EXPECT_TRUE(visitor.usedOnlyConstIterators());
    EXPECT_EQ(tree.leafCount(), visitor.aLeafCount());
    EXPECT_EQ(tree2.leafCount(), visitor.bLeafCount());
    EXPECT_EQ(tree.nonLeafCount(), visitor.aNonLeafCount());
    EXPECT_EQ(tree2.nonLeafCount(), visitor.bNonLeafCount());

    visitor.reset();

    // Change the topology of the first tree.
    tree.setValue(openvdb::Coord(-200, -200, -200), openvdb::zeroVal<ValueT>());

    // Traverse both trees.
    tree.visit2(tree2, visitor);

    EXPECT_EQ(tree.leafCount(), visitor.aLeafCount());
    EXPECT_EQ(tree2.leafCount(), visitor.bLeafCount());
    EXPECT_EQ(tree.nonLeafCount(), visitor.aNonLeafCount());
    EXPECT_EQ(tree2.nonLeafCount(), visitor.bNonLeafCount());

    visitor.reset();

    // Traverse the two trees in the opposite order.
    tree2.visit2(tree, visitor);

    EXPECT_EQ(tree2.leafCount(), visitor.aLeafCount());
    EXPECT_EQ(tree.leafCount(), visitor.bLeafCount());
    EXPECT_EQ(tree2.nonLeafCount(), visitor.aNonLeafCount());
    EXPECT_EQ(tree.nonLeafCount(), visitor.bNonLeafCount());

    // Repeat, skipping leaf nodes of tree2.
    visitor.reset();
    visitor.setSkipALeafNodes(true);
    tree2.visit2(tree, visitor);

    EXPECT_EQ(0U, visitor.aLeafCount());
    EXPECT_EQ(tree.leafCount(), visitor.bLeafCount());
    EXPECT_EQ(tree2.nonLeafCount(), visitor.aNonLeafCount());
    EXPECT_EQ(tree.nonLeafCount(), visitor.bNonLeafCount());

    // Repeat, skipping leaf nodes of tree.
    visitor.reset();
    visitor.setSkipBLeafNodes(true);
    tree2.visit2(tree, visitor);

    EXPECT_EQ(tree2.leafCount(), visitor.aLeafCount());
    EXPECT_EQ(0U, visitor.bLeafCount());
    EXPECT_EQ(tree2.nonLeafCount(), visitor.aNonLeafCount());
    EXPECT_EQ(tree.nonLeafCount(), visitor.bNonLeafCount());

    OPENVDB_NO_DEPRECATION_WARNING_END
}
