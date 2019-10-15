//////////////////////////////////////////////////////////////////////////
//
// TM & (c) Lucasfilm Entertainment Company Ltd. and Lucasfilm Ltd.
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
///////////////////////////////////////////////////////////////////////////
//
/// @file Count.h
///
/// @author Dan Bailey
///
/// @brief Counting tools.

#ifndef OPENVDB_TOOLS_COUNT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_COUNT_HAS_BEEN_INCLUDED

#include <openvdb/tree/NodeManager.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Return a vector with node counts. The number of node of type
/// NodeType is given as element NodeType::LEVEL in the return vector.
/// Thus, the size if this vector corresponds to the height (or depth)
/// of this tree.
///
/// @param tree       the tree to be counted
/// @param threaded   enable or disable threading (threading is enabled by default)
template <typename TreeT>
std::vector<Index>
nodeCount(const TreeT& tree, bool threaded = true);


/// @brief Return the total number of active voxels.
///
/// @param tree       the tree to be counted
/// @param threaded   enable or disable threading (threading is enabled by default)
template <typename TreeT>
Index64
activeVoxelCount(TreeT& tree, bool threaded = true);


////////////////////////////////////////


namespace count_internal {


template <Index DEPTH>
struct NodeCountOp
{
    template <typename ArrayT, typename NodeT>
    static void count(ArrayT& array, const NodeT& node)
    {
        for (auto iter = node.cbeginChildOn(); iter; ++iter) {
            array[DEPTH-1] += iter->getChildMask().countOn();
            NodeCountOp<DEPTH-1>::count(array, *iter);
        }
    }
};

template <>
struct NodeCountOp<0>
{
    template <typename ArrayT, typename NodeT>
    static void count(ArrayT&, const NodeT&) { }
};


template<typename TreeT>
class ActiveVoxelCountOp
{
public:
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    ActiveVoxelCountOp() { }
    ActiveVoxelCountOp(const ActiveVoxelCountOp&, tbb::split) { }
    void join(const ActiveVoxelCountOp& other) { count += other.count; }

    void operator()(RootT& root)
    {
        count += root.onTileCount() * RootT::ChildNodeType::NUM_VOXELS;
    }
    void operator()(LeafT& node)
    {
        count += node.onVoxelCount();
    }
    template<typename NodeT>
    void operator()(NodeT& node)
    {
        count += node.getValueMask().countOn() * NodeT::ChildNodeType::NUM_VOXELS;
    }

    Index64 count = Index64(0);
};// ActiveVoxelCountOp


} // namespace count_internal


////////////////////////////////////////


template <typename TreeT>
std::vector<Index> nodeCount(const TreeT& tree, bool threaded)
{
    using RootT = typename TreeT::RootNodeType;

    using namespace count_internal;

    std::vector<Index> countArray(TreeT::DEPTH, Index(0));

    // root node

    auto countIter = countArray.rbegin();
    *(countIter++) = 1;

    if (TreeT::DEPTH == 1)  return countArray;

    // root node children

    Index rootNodeChildren(0);

    const typename TreeT::RootNodeType& root = tree.root();
    for (auto iter = root.cbeginChildOn(); iter; ++iter) {
        rootNodeChildren++;
    }

    *(countIter++) = rootNodeChildren;

    if (TreeT::DEPTH == 2)  return countArray;

    // other children

    static constexpr Index LEVELS = RootT::LEVEL-1;

    if (threaded) {
        using ArrayT = std::array<Index, LEVELS>;

        std::vector<ArrayT> arrays(rootNodeChildren, ArrayT());
        tbb::parallel_for(
            tbb::blocked_range<Index>(0, rootNodeChildren),
            [&](tbb::blocked_range<Index>& r) {
                for (Index n = r.begin(); n < r.end(); n++) {
                    auto rootIter = root.cbeginChildOn();
                    rootIter.increment(n);
                    arrays[n].fill(Index(0));
                    arrays[n].back() += rootIter->getChildMask().countOn();
                    NodeCountOp<LEVELS-1>::count(arrays[n], *rootIter);
                }
            }
        );
        for (size_t i = 0; i < LEVELS; i++) {
            for (size_t n = 0; n < rootNodeChildren; n++) {
                countArray[i] += arrays[n][i];
            }
        }
    } else {
        NodeCountOp<LEVELS>::count(countArray, root);
    }

    return countArray;
}


template <>
std::vector<Index> nodeCount(const GridBase& grid, bool threaded)
{
    using AllowedGridTypes = openvdb::TypeList<
        openvdb::Int32Grid, openvdb::Int64Grid,
        openvdb::FloatGrid, openvdb::DoubleGrid>;

    std::vector<Index> counts;
    grid.apply<AllowedGridTypes>(
        [&](auto& grid) {
            counts = nodeCount(grid.constTree(), threaded);
        }
    );
    return counts;
}


template <typename TreeT>
Index64 activeVoxelCount(TreeT& tree, bool threaded)
{
    tree::NodeManager<TreeT> nodeManager(tree);
    count_internal::ActiveVoxelCountOp<TreeT> op;
    nodeManager.reduceTopDown(op, threaded);
    return op.count;
}


template <>
Index64 activeVoxelCount(GridBase& grid, bool threaded)
{
    using AllowedGridTypes = openvdb::TypeList<
        openvdb::Int32Grid, openvdb::Int64Grid,
        openvdb::FloatGrid, openvdb::DoubleGrid>;

    Index64 count(0);
    grid.apply<AllowedGridTypes>(
        [&](auto& grid) {
            count = activeVoxelCount(grid.tree(), threaded);
        }
    );
    return count;
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_COUNT_HAS_BEEN_INCLUDED
