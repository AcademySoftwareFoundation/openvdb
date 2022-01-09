// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <gtest/gtest.h>

class TestDepth : public ::testing::Test
{
};


// Recursive tree with equal Dim at every level
template<int Depth, typename NodeT, int Log2Dim>
struct TestTree;

template<typename NodeT, int Log2Dim>
struct TestTree<0,NodeT,Log2Dim>
{
  using Type = openvdb::tree::Tree< openvdb::tree::RootNode< NodeT > >;
};

template<int Depth, typename NodeT, int Log2Dim>
struct TestTree
{
  using Type = typename TestTree<Depth-1, openvdb::tree::InternalNode<NodeT, Log2Dim>, Log2Dim>::Type;
};

template<int Depth, typename ValueT, int Log2Dim>
struct TestGrid
{
  using Type = typename openvdb::Grid< typename TestTree<Depth, openvdb::tree::LeafNode<ValueT, Log2Dim>, Log2Dim>::Type >;
};

// function creates empty grid,
// adds one default leaf
// and checks if the tree empty or not after that
template<int Depth, typename ValueT, int Log2Dim>
bool check_grid()
{
    using GridT = typename TestGrid<Depth, ValueT, Log2Dim>::Type;
    GridT grid(0);

    typename GridT::Accessor acc = grid.getAccessor();
    acc.addLeaf( new openvdb::tree::LeafNode<float,Log2Dim>(openvdb::Coord(0), 0.0f, true) );

    return grid.empty();
}

TEST_F(TestDepth, test)
{
    using namespace openvdb;

    bool is_grid_empty = check_grid<2,float,1>();
    EXPECT_TRUE(!is_grid_empty);

    is_grid_empty = check_grid<3,float,1>();
    EXPECT_TRUE(!is_grid_empty);

    is_grid_empty = check_grid<4,float,1>();
    EXPECT_TRUE(!is_grid_empty);

//    is_grid_empty = check_grid<5,float,1>();
//    EXPECT_TRUE(!is_grid_empty);

//    is_grid_empty = check_grid<6,float,1>();
//    EXPECT_TRUE(!is_grid_empty);

//    is_grid_empty = check_grid<7,float,1>();
//    EXPECT_TRUE(!is_grid_empty);
    
//    is_grid_empty = check_grid<8,float,1>();
//    EXPECT_TRUE(!is_grid_empty);

}
