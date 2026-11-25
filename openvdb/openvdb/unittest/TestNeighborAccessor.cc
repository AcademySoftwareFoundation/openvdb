// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tree/NeighborAccessor.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using namespace openvdb;

namespace {

template<typename T>
T makeValue(int i)
{
    if constexpr (std::is_same_v<T, bool>) {
        return (i & 1) != 0;
    } else {
        return T(i);
    }
}

// Leaves, active and inactive tiles at every level, and background
template<typename TreeT>
void populate(TreeT& tree, unsigned seed)
{
    using ValueT = typename TreeT::ValueType;
    std::mt19937 generator(seed);
    auto randomCoord = [&](int range) {
        std::uniform_int_distribution<int> distribution(-range, range);
        return Coord(distribution(generator), distribution(generator), distribution(generator));
    };

    for (int i = 0; i < 2000; ++i) tree.setValueOn(randomCoord(300), makeValue<ValueT>(i));

    // A block of neighboring leaves
    int count = 0;
    for (int x = -10; x <= 10; ++x) {
        for (int y = -10; y <= 10; ++y) {
            for (int z = -10; z <= 10; ++z) {
                if (((x * 31 + y * 17 + z) & 3) == 0) {
                    tree.setValueOn(Coord(x, y, z), makeValue<ValueT>(count++));
                }
            }
        }
    }

    for (Index level = 1; level <= TreeT::RootNodeType::LEVEL; ++level) {
        for (int k = 0; k < 8; ++k) {
            const Coord xyz = randomCoord(k < 4 ? 300 : 3000);
            tree.addTile(level, xyz, makeValue<ValueT>(1000 + int(level) * 10 + k), (k & 1) != 0);
        }
    }
}

// Leaves next to each other and next to tiles
template<typename TreeT>
void populateDense(TreeT& tree, unsigned seed)
{
    using ValueT = typename TreeT::ValueType;
    populate(tree, seed);
    tree.fill(CoordBBox(Coord(-9), Coord(9)), makeValue<ValueT>(1), true);
    for (int i = 0; i < 700; ++i) {
        tree.setValueOn(Coord(i % 23 - 11, i % 13 - 6, i % 7 - 3), makeValue<ValueT>(i));
    }
}

std::vector<Coord> makeCoords(unsigned seed, int count, int range)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(-range, range);
    std::vector<Coord> coords;
    for (int i = 0; i < count; ++i) {
        coords.emplace_back(distribution(generator), distribution(generator), distribution(generator));
    }
    return coords;
}

// z fastest, through the block of neighboring leaves
std::vector<Coord> scanCoords()
{
    std::vector<Coord> coords;
    for (int x = -20; x <= 20; ++x)
        for (int y = -20; y <= 20; ++y)
            for (int z = -20; z <= 20; ++z)
                coords.emplace_back(x, y, z);
    return coords;
}

// Alternate between two distant places
std::vector<Coord> alternatingCoords()
{
    std::vector<Coord> coords;
    for (int i = 0; i < 4000; ++i) {
        coords.emplace_back(i % 17, (i / 17) % 13, i % 5);
        coords.emplace_back(2000 + i % 11, -3000 + (i / 11) % 7, 5000 - i % 3);
    }
    return coords;
}

std::vector<std::vector<Coord>> allOrders()
{
    std::vector<std::vector<Coord>> orders;
    orders.push_back(scanCoords());
    orders.push_back(makeCoords(1, 20000, 400));
    orders.push_back(makeCoords(2, 20000, 4000));
    orders.push_back(alternatingCoords());
    std::vector<Coord> sorted = makeCoords(3, 20000, 300);
    std::sort(sorted.begin(), sorted.end());
    orders.push_back(sorted);
    return orders;
}

template<int DX, int DY, int DZ>
struct Offset {};

// Offsets from every voxel of every leaf, by offset and by iterator
template<typename TreeT, int DX, int DY, int DZ>
void checkOffset(const TreeT& tree, tree::NeighborAccessor<const TreeT>& acc, Offset<DX, DY, DZ>)
{
    using LeafT = typename TreeT::LeafNodeType;
    for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        const LeafT& leaf = *leafIter;
        acc.moveTo(leaf.origin());
        for (auto iter = leaf.cbeginValueAll(); iter; ++iter) {
            const Coord xyz = iter.getCoord() + Coord(DX, DY, DZ);
            ASSERT_EQ(tree.getValue(xyz), (acc.template getValue<DX, DY, DZ>(iter)))
                << DX << " " << DY << " " << DZ << " at " << xyz;
            ASSERT_EQ(tree.getValue(xyz), (acc.template getValue<DX, DY, DZ>(iter.pos())))
                << DX << " " << DY << " " << DZ << " at " << xyz;
        }
    }
}

template<typename TreeT, typename... Offsets>
void checkOffsets(const TreeT& tree, Offsets... offsets)
{
    tree::NeighborAccessor<const TreeT> acc(tree);
    (checkOffset(tree, acc, offsets), ...);
}

// Offsets from a voxel in the bound region, which may be a tile or the background
template<typename TreeT, int DX, int DY, int DZ>
void checkHere(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc, const Coord& xyz,
    Offset<DX, DY, DZ>)
{
    const Index offset = TreeT::LeafNodeType::coordToOffset(xyz);
    ASSERT_EQ(tree.getValue(xyz.offsetBy(DX, DY, DZ)), (acc.template getValue<DX, DY, DZ>(offset)))
        << DX << " " << DY << " " << DZ << " at " << xyz;
}

template<typename TreeT>
void checkCoordinates(const TreeT& tree, const std::vector<Coord>& coords)
{
    using LeafT = typename TreeT::LeafNodeType;
    tree::NeighborAccessor<const TreeT> acc(tree);
    for (const Coord& xyz : coords) {
        acc.moveTo(xyz);
        const Index offset = LeafT::coordToOffset(xyz);
        ASSERT_EQ(xyz & ~Int32(LeafT::DIM - 1), acc.origin());
        ASSERT_EQ(tree.getValue(xyz), acc.getValue(offset)) << xyz;
        checkHere(tree, acc, xyz, Offset<0, 0, 0>());
        checkHere(tree, acc, xyz, Offset<1, 0, 0>());
        checkHere(tree, acc, xyz, Offset<-1, 0, 0>());
        checkHere(tree, acc, xyz, Offset<0, 1, 0>());
        checkHere(tree, acc, xyz, Offset<0, -1, 0>());
        checkHere(tree, acc, xyz, Offset<0, 0, 1>());
        checkHere(tree, acc, xyz, Offset<0, 0, -1>());
        checkHere(tree, acc, xyz, Offset<-1, 1, 1>());
        checkHere(tree, acc, xyz, Offset<1, -1, 1>());
        checkHere(tree, acc, xyz, Offset<2, 0, 0>());
        checkHere(tree, acc, xyz, Offset<0, -3, 0>());
        checkHere(tree, acc, xyz, Offset<3, 2, -5>());
        checkHere(tree, acc, xyz, Offset<-8, 8, 0>());
        checkHere(tree, acc, xyz, Offset<9, -9, 9>());
        ASSERT_EQ(tree.getValue(xyz.offsetBy(0, 4, -2)), acc.getValue(offset, 0, 4, -2));
        ASSERT_EQ(tree.getValue(xyz.offsetBy(-7, 1, 6)), acc.getValue(offset, -7, 1, 6));
    }
}

template<typename TreeT>
void checkTree(unsigned seed)
{
    TreeT tree(makeValue<typename TreeT::ValueType>(1000));
    populateDense(tree, seed);
    for (const std::vector<Coord>& coords : allOrders()) {
        checkCoordinates(tree, coords);
    }
}

// Reads of @a tree at the voxels of the leaves of @a source
template<typename TreeT, typename SourceTreeT>
void checkSource(const TreeT& tree, const SourceTreeT& source)
{
    tree::NeighborAccessor<const TreeT> moved(tree);
    for (auto leafIter = source.cbeginLeaf(); leafIter; ++leafIter) {
        // One built per leaf, and one moved from leaf to leaf
        tree::NeighborAccessor<const TreeT> built(tree, leafIter->origin());
        moved.moveTo(leafIter->origin());
        for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
            const Coord xyz = iter.getCoord();
            ASSERT_EQ(tree.getValue(xyz), built.getValue(iter)) << xyz;
            ASSERT_EQ(tree.getValue(xyz), moved.getValue(iter)) << xyz;
            ASSERT_EQ(tree.getValue(xyz.offsetBy(1, 0, 0)), (moved.template getValue<1, 0, 0>(iter))) << xyz;
            ASSERT_EQ(tree.getValue(xyz.offsetBy(0, -1, 1)), (moved.template getValue<0, -1, 1>(iter))) << xyz;
            ASSERT_EQ(tree.getValue(xyz.offsetBy(-3, 0, 2)), moved.getValue(iter, -3, 0, 2)) << xyz;
        }
    }
}

// Every row of the bound region, at one shift
template<typename TreeT, int DX, int DY, int DZ>
void checkRow(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc, Offset<DX, DY, DZ>)
{
    using LeafT = typename TreeT::LeafNodeType;
    constexpr int dim = int(LeafT::DIM);
    for (int x = 0; x < dim; ++x) {
        for (int y = 0; y < dim; ++y) {
            const Index rowOffset = LeafT::coordToOffset(Coord(x, y, 0));
            const auto row = acc.template getRow<DX, DY, DZ>(rowOffset);
            ASSERT_TRUE(row.data != nullptr);
            const Coord start = acc.origin().offsetBy(x + DX, y + DY, DZ);
            ASSERT_EQ(tree.probeConstLeaf(start) != nullptr, row.contiguous) << start;
            for (int z = 0; z < dim; ++z) {
                const typename TreeT::ValueType expected = tree.getValue(start.offsetBy(0, 0, z));
                ASSERT_EQ(expected, row.data[row.contiguous ? z : 0])
                    << DX << " " << DY << " " << DZ << " at " << start << " + " << z;
            }
            // The same elements that a voxel read gives
            if (row.contiguous) {
                for (int z = 0; z < dim; ++z) {
                    ASSERT_EQ(row.data + z, &acc.getValue(rowOffset, DX, DY, DZ + z));
                }
            } else {
                ASSERT_EQ(row.data, &acc.getValue(rowOffset, DX, DY, DZ));
            }
        }
    }
}

template<typename TreeT, typename... Offsets>
void checkRowsHere(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc, Offsets... offsets)
{
    (checkRow(tree, acc, offsets), ...);
}

// Rows of leaves, and of regions that are tiles or the background
template<typename TreeT>
void checkRows(const TreeT& tree)
{
    constexpr int dim = int(TreeT::LeafNodeType::DIM);
    tree::NeighborAccessor<const TreeT> acc(tree);
    auto checkAll = [&]() {
        checkRowsHere(tree, acc,
            Offset<0, 0, 0>(), Offset<1, 0, 0>(), Offset<-1, 0, 0>(), Offset<0, 1, 0>(),
            Offset<0, -1, 0>(), Offset<0, 0, dim>(), Offset<0, 0, -dim>(), Offset<dim, 0, 0>(),
            Offset<0, -dim, 0>(), Offset<1, -1, dim>(), Offset<-dim, dim, -dim>());
        if constexpr (dim >= 4) {
            checkRowsHere(tree, acc, Offset<3, 0, 0>(), Offset<-3, 0, 0>(), Offset<0, 3, 0>(),
                Offset<0, -3, 0>(), Offset<2, -3, -dim>());
        }
    };

    for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        acc.moveTo(leafIter->origin());
        checkAll();
    }
    for (const Coord& xyz : makeCoords(51, 500, 4000)) {
        acc.moveTo(xyz);
        checkAll();
    }
}

// The value mask of the region at region index N, from -1 to 1 regions away in each axis
template<typename TreeT, int N>
void checkRegion(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc)
{
    using LeafT = typename TreeT::LeafNodeType;
    constexpr int I = N / 9 - 1, J = (N / 3) % 3 - 1, K = N % 3 - 1;
    constexpr int dim = int(LeafT::DIM);
    const Coord origin = acc.origin().offsetBy(I * dim, J * dim, K * dim);
    const auto& mask = acc.template getValueMask<I, J, K>();
    if (const LeafT* leaf = tree.probeConstLeaf(origin)) {
        ASSERT_TRUE((acc.template isLeaf<I, J, K>())) << origin;
        ASSERT_EQ(&leaf->getValueMask(), &mask) << origin;
    } else {
        ASSERT_FALSE((acc.template isLeaf<I, J, K>())) << origin;
        const typename LeafT::NodeMaskType expected(tree.isValueOn(origin));
        ASSERT_TRUE(expected == mask) << origin;
    }
}

template<typename TreeT, int... N>
void checkRegions(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc,
    std::integer_sequence<int, N...>)
{
    (checkRegion<TreeT, N>(tree, acc), ...);
}

// The value masks of all 27 regions, after moves to leaves and to random places
template<typename TreeT>
void checkMasks(const TreeT& tree)
{
    tree::NeighborAccessor<const TreeT> acc(tree);
    const auto regions = std::make_integer_sequence<int, 27>();
    for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        acc.moveTo(leafIter->origin());
        checkRegions(tree, acc, regions);
    }
    for (const Coord& xyz : makeCoords(61, 2000, 400)) {
        acc.moveTo(xyz);
        checkRegions(tree, acc, regions);
    }
}

template<typename TreeT>
void checkMasksOf(unsigned seed)
{
    TreeT tree(makeValue<typename TreeT::ValueType>(1000));
    populateDense(tree, seed);
    checkMasks(tree);
}

// The size of a parent node of the leaves, or a large size if the parent is the root
template<typename TreeT>
constexpr int parentDimOf()
{
    using RootT = typename TreeT::RootNodeType;
    using ParentT = typename RootT::NodeChainType::template Get<1>;
    if constexpr (std::is_same_v<ParentT, RootT>) {
        return 1024;
    } else {
        return int(ParentT::DIM);
    }
}

// Every offset up to two leaves and one voxel away, from voxels in leaves, tiles and the background
template<typename TreeT>
void checkFarReads(const TreeT& tree, const std::vector<Coord>& centers)
{
    using LeafT = typename TreeT::LeafNodeType;
    constexpr int reach = 2 * int(LeafT::DIM) + 1;
    tree::NeighborAccessor<const TreeT> acc(tree);
    for (const Coord& center : centers) {
        acc.moveTo(center);
        const Index offset = LeafT::coordToOffset(center);
        for (int dx = -reach; dx <= reach; ++dx) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dz = -reach; dz <= reach; ++dz) {
                    ASSERT_EQ(tree.getValue(center.offsetBy(dx, dy, dz)), acc.getValue(offset, dx, dy, dz))
                        << center << " " << dx << " " << dy << " " << dz;
                }
            }
        }
    }
}

// Reads that alternate between far regions, in the parent node of the bound
// region, in the next parent node and farther away, so that the far region is refilled
template<typename TreeT>
void checkFarRefills(const TreeT& tree, const std::vector<Coord>& centers)
{
    using LeafT = typename TreeT::LeafNodeType;
    constexpr int dim = int(LeafT::DIM);
    constexpr int parentDim = parentDimOf<TreeT>();
    const std::vector<Coord> far = {
        Coord(3 * dim, 0, 0), Coord(-3 * dim, 1, 0), Coord(parentDim, 0, 2),
        Coord(-parentDim - 1, 0, 0), Coord(0, parentDim + 1, -3 * dim), Coord(0, 0, -2 * parentDim - 1),
        Coord(2 * parentDim + 1, -parentDim, parentDim), Coord(-5000, 7, 3)};
    tree::NeighborAccessor<const TreeT> acc(tree);
    for (const Coord& center : centers) {
        acc.moveTo(center);
        const Index offset = LeafT::coordToOffset(center);
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (size_t i = 0; i < far.size(); ++i) {
                for (size_t j = i + 1; j < far.size(); ++j) {
                    for (const Coord& delta : {far[i], far[j], far[i]}) {
                        ASSERT_EQ(tree.getValue(center + delta),
                            acc.getValue(offset, delta[0], delta[1], delta[2])) << center << " " << delta;
                    }
                }
            }
        }
    }
}

template<typename TreeT>
void checkFar(unsigned seed, size_t centerCount)
{
    TreeT tree(makeValue<typename TreeT::ValueType>(1000));
    populateDense(tree, seed);
    using LeafT = typename TreeT::LeafNodeType;
    std::vector<Coord> centers;
    size_t leafCount = 0;
    for (auto leafIter = tree.cbeginLeaf(); leafIter && centers.size() < centerCount; ++leafIter) {
        if (leafCount++ % 7 == 0) {
            centers.push_back(leafIter->offsetToGlobalCoord(Index((leafCount * 13) % LeafT::SIZE)));
        }
    }
    for (const Coord& xyz : makeCoords(seed, int(centerCount), 400)) centers.push_back(xyz);
    for (const Coord& xyz : makeCoords(seed + 1, int(centerCount / 4), 4000)) centers.push_back(xyz);
    checkFarReads(tree, centers);
    checkFarRefills(tree, centers);
}

// The active state and value at an offset from a voxel in the bound region,
// which may be a tile or the background
template<typename TreeT, int DX, int DY, int DZ>
void checkActiveHere(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc, const Coord& xyz,
    Offset<DX, DY, DZ>)
{
    using ValueT = typename TreeT::ValueType;
    const Index offset = TreeT::LeafNodeType::coordToOffset(xyz);
    ValueT expected{}, value{};
    const bool expectedOn = tree.probeValue(xyz.offsetBy(DX, DY, DZ), expected);
    ASSERT_EQ(expectedOn, (acc.template isValueOn<DX, DY, DZ>(offset)))
        << DX << " " << DY << " " << DZ << " at " << xyz;
    ASSERT_EQ(expectedOn, acc.isValueOn(offset, DX, DY, DZ))
        << DX << " " << DY << " " << DZ << " at " << xyz;
    ASSERT_EQ(expectedOn, (acc.template probeValue<DX, DY, DZ>(offset, value)))
        << DX << " " << DY << " " << DZ << " at " << xyz;
    ASSERT_EQ(expected, value) << DX << " " << DY << " " << DZ << " at " << xyz;
    value = ValueT{};
    ASSERT_EQ(expectedOn, acc.probeValue(offset, DX, DY, DZ, value))
        << DX << " " << DY << " " << DZ << " at " << xyz;
    ASSERT_EQ(expected, value) << DX << " " << DY << " " << DZ << " at " << xyz;
}

// The voxel at region index N, from -1 to 1 voxels away in each axis
template<typename TreeT, int N>
void checkActiveNeighbor(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc, const Coord& xyz)
{
    constexpr int I = N / 9 - 1, J = (N / 3) % 3 - 1, K = N % 3 - 1;
    checkActiveHere(tree, acc, xyz, Offset<I, J, K>());
}

template<typename TreeT, int... N>
void checkActiveNeighbors(const TreeT& tree, const tree::NeighborAccessor<const TreeT>& acc,
    const Coord& xyz, std::integer_sequence<int, N...>)
{
    (checkActiveNeighbor<TreeT, N>(tree, acc, xyz), ...);
}

template<typename TreeT>
void checkActive(unsigned seed)
{
    using ValueT = typename TreeT::ValueType;
    using LeafT = typename TreeT::LeafNodeType;
    constexpr int dim = int(LeafT::DIM);
    TreeT tree(makeValue<ValueT>(1000));
    populateDense(tree, seed);
    tree::NeighborAccessor<const TreeT> acc(tree);

    // At the voxels of leaves, by iterator
    for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        acc.moveTo(leafIter->origin());
        for (auto iter = leafIter->cbeginValueAll(); iter; ++iter) {
            const Coord xyz = iter.getCoord();
            ValueT value{};
            ASSERT_EQ(iter.isValueOn(), acc.isValueOn(iter)) << xyz;
            ASSERT_EQ(iter.isValueOn(), acc.probeValue(iter, value)) << xyz;
            ASSERT_EQ(*iter, value) << xyz;
            ASSERT_EQ(tree.isValueOn(xyz.offsetBy(1, 0, 0)), (acc.template isValueOn<1, 0, 0>(iter))) << xyz;
            ASSERT_EQ(tree.isValueOn(xyz.offsetBy(0, -1, 2)), acc.isValueOn(iter, 0, -1, 2)) << xyz;
            ASSERT_EQ(tree.isValueOn(xyz.offsetBy(0, 0, -dim)), (acc.template probeValue<0, 0, -dim>(iter, value)))
                << xyz;
            ASSERT_EQ(tree.getValue(xyz.offsetBy(0, 0, -dim)), value) << xyz;
        }
    }

    // In leaves, tiles and the background, at offsets in the 27 regions and beyond
    std::vector<Coord> coords = makeCoords(seed, 4000, 400);
    for (const Coord& xyz : makeCoords(seed + 1, 1000, 4000)) coords.push_back(xyz);
    for (const Coord& xyz : coords) {
        acc.moveTo(xyz);
        const Index offset = LeafT::coordToOffset(xyz);
        ValueT expected{}, value{};
        const bool expectedOn = tree.probeValue(xyz, expected);
        ASSERT_EQ(expectedOn, acc.isValueOn(offset)) << xyz;
        ASSERT_EQ(expectedOn, acc.probeValue(offset, value)) << xyz;
        ASSERT_EQ(expected, value) << xyz;
        checkActiveNeighbors(tree, acc, xyz, std::make_integer_sequence<int, 27>());
        checkActiveHere(tree, acc, xyz, Offset<dim, 0, -dim>());
        checkActiveHere(tree, acc, xyz, Offset<2, -3, 1>());
        checkActiveHere(tree, acc, xyz, Offset<dim + 1, 0, 0>());
        checkActiveHere(tree, acc, xyz, Offset<-2 * dim - 1, dim, 3>());
        checkActiveHere(tree, acc, xyz, Offset<0, 0, 5000>());
    }
}

using Tree4Type = tree::Tree4<float, 5, 4, 3>::Type;
using Tree2Type = tree::Tree<tree::RootNode<tree::LeafNode<float, 3>>>;
using Tree3Type = tree::Tree<tree::RootNode<tree::InternalNode<tree::LeafNode<float, 3>, 4>>>;
using Tree5Type = tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode<
    tree::InternalNode<tree::LeafNode<float, 3>, 3>, 3>, 3>>>;
using SmallLeafTreeType = tree::Tree4<float, 3, 3, 2>::Type;     // 4^3 leaves
using TinyLeafTreeType = tree::Tree4<float, 3, 3, 1>::Type;      // 2^3 leaves

} // anonymous namespace


class TestNeighborAccessor: public ::testing::Test {};


TEST_F(TestNeighborAccessor, testOffsets)
{
    // 4^3 leaves, so that offsets of 3 cross most of a leaf
    SmallLeafTreeType tree(0.0f);
    populateDense(tree, 41);

    checkOffsets(tree,
        Offset<-1, -1, -1>(), Offset<-1, -1, 0>(), Offset<-1, -1, 1>(),
        Offset<-1, 0, -1>(), Offset<-1, 0, 0>(), Offset<-1, 0, 1>(),
        Offset<-1, 1, -1>(), Offset<-1, 1, 0>(), Offset<-1, 1, 1>(),
        Offset<0, -1, -1>(), Offset<0, -1, 0>(), Offset<0, -1, 1>(),
        Offset<0, 0, -1>(), Offset<0, 0, 0>(), Offset<0, 0, 1>(),
        Offset<0, 1, -1>(), Offset<0, 1, 0>(), Offset<0, 1, 1>(),
        Offset<1, -1, -1>(), Offset<1, -1, 0>(), Offset<1, -1, 1>(),
        Offset<1, 0, -1>(), Offset<1, 0, 0>(), Offset<1, 0, 1>(),
        Offset<1, 1, -1>(), Offset<1, 1, 0>(), Offset<1, 1, 1>());

    checkOffsets(tree,
        Offset<-2, 0, 0>(), Offset<2, 0, 0>(), Offset<0, -2, 0>(), Offset<0, 2, 0>(),
        Offset<0, 0, -2>(), Offset<0, 0, 2>(), Offset<-3, 0, 0>(), Offset<3, 0, 0>(),
        Offset<0, -3, 0>(), Offset<0, 3, 0>(), Offset<0, 0, -3>(), Offset<0, 0, 3>(),
        Offset<-2, -2, 0>(), Offset<2, 2, 0>(), Offset<-3, 0, 3>(), Offset<0, 3, -3>(),
        Offset<-2, 1, 2>(), Offset<3, -1, 2>());

    // Up to one full leaf away
    checkOffsets(tree,
        Offset<4, 0, 0>(), Offset<-4, 0, 0>(), Offset<0, 4, 0>(), Offset<0, -4, 0>(),
        Offset<0, 0, 4>(), Offset<0, 0, -4>(),
        Offset<4, 4, 4>(), Offset<-4, -4, -4>(), Offset<4, -4, 4>(), Offset<-4, 4, -4>(),
        Offset<4, 4, -4>(), Offset<-4, -4, 4>(), Offset<4, -4, -4>(), Offset<-4, 4, 4>());

    // Past one leaf, which reads the far region
    checkOffsets(tree,
        Offset<5, 0, 0>(), Offset<0, -5, 0>(), Offset<-7, 6, 5>(), Offset<9, -9, 9>());

    // Past one parent node of 32^3 voxels
    checkOffsets(tree,
        Offset<33, 0, 0>(), Offset<0, -40, 0>(), Offset<-70, 40, 5>(), Offset<0, 3, 65>());
}

TEST_F(TestNeighborAccessor, testOffsetsDefaultLeaf)
{
    Tree4Type tree(0.0f);
    populateDense(tree, 42);
    checkOffsets(tree, Offset<1, 0, 0>(), Offset<0, -1, 0>(), Offset<-1, 1, 1>(),
        Offset<3, 0, 0>(), Offset<0, 0, -3>(), Offset<8, 0, 0>(), Offset<-8, -8, 8>(),
        Offset<3, 2, -5>(), Offset<9, 0, 0>());
}

TEST_F(TestNeighborAccessor, testOffsetsTinyLeaf)
{
    // 2^3 leaves, so a 19 point stencil reaches two leaves away
    TinyLeafTreeType tree(0.0f);
    populateDense(tree, 43);
    checkOffsets(tree, Offset<1, 0, 0>(), Offset<-1, 1, 0>(), Offset<2, 0, 0>(),
        Offset<0, -2, 0>(), Offset<0, 0, 3>(), Offset<-3, 2, 1>(), Offset<-2, -2, -2>());
}

TEST_F(TestNeighborAccessor, testRuntimeOffsets)
{
    SmallLeafTreeType tree(0.0f);
    populateDense(tree, 44);
    using LeafT = SmallLeafTreeType::LeafNodeType;
    constexpr int reach = 2 * int(LeafT::DIM) + 1;

    tree::NeighborAccessor<const SmallLeafTreeType> acc(tree);
    int leafCount = 0;
    for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter, ++leafCount) {
        if (leafCount % 11) continue;
        const LeafT& leaf = *leafIter;
        acc.moveTo(leaf.origin());
        for (Index offset = 0; offset < Index(LeafT::SIZE); offset += 5) {
            const Coord center = leaf.offsetToGlobalCoord(offset);
            for (int dx = -reach; dx <= reach; ++dx) {
                for (int dy = -reach; dy <= reach; dy += 2) {
                    for (int dz = -reach; dz <= reach; dz += 3) {
                        ASSERT_EQ(tree.getValue(center.offsetBy(dx, dy, dz)),
                            acc.getValue(offset, dx, dy, dz)) << center << " " << dx << " " << dy << " " << dz;
                    }
                }
            }
        }
    }
    EXPECT_GT(leafCount, 100);
}

TEST_F(TestNeighborAccessor, testFarRegions)
{
    checkFar<Tree4Type>(81, 20);
    checkFar<SmallLeafTreeType>(82, 80);
    checkFar<TinyLeafTreeType>(83, 200);
    checkFar<Tree2Type>(84, 20);
    checkFar<Tree3Type>(85, 20);
    checkFar<Tree5Type>(86, 20);
    checkFar<tree::Tree4<bool, 5, 4, 3>::Type>(87, 20);
    checkFar<tree::Tree4<ValueMask, 5, 4, 3>::Type>(88, 20);

    // A far read gives the element of the leaf buffer, or the tile value in place
    using LeafT = Tree4Type::LeafNodeType;
    constexpr int dim = int(LeafT::DIM);
    Tree4Type tree(0.0f);
    tree.setValueOn(Coord(1, 2, 3), 5.0f);
    tree.addTile(1, Coord(-3 * dim, 0, 0), 6.0f, true);
    const LeafT* leaf = tree.probeConstLeaf(Coord(1, 2, 3));
    ASSERT_TRUE(leaf != nullptr);
    tree::NeighborAccessor<const Tree4Type> acc(tree, Coord(3 * dim + 1, 2, 3));
    const Index offset = LeafT::coordToOffset(Coord(1, 2, 3));
    EXPECT_EQ(&leaf->getValue(Coord(1, 2, 3)), &acc.getValue(offset, -3 * dim, 0, 0));
    EXPECT_EQ(&leaf->getValue(Coord(1, 2, 3)), &(acc.getValue<-3 * dim, 0, 0>(offset)));
    const float* tileValue = &acc.getValue(offset, -6 * dim, 0, 0);
    EXPECT_EQ(6.0f, *tileValue);
    EXPECT_EQ(tileValue, &acc.getValue(Index(0), -6 * dim, 0, 0));
    EXPECT_EQ(&tree.root().background(), &acc.getValue(offset, 0, 0, 5000));
}

TEST_F(TestNeighborAccessor, testActiveStates)
{
    checkActive<Tree4Type>(91);
    checkActive<SmallLeafTreeType>(92);
    checkActive<TinyLeafTreeType>(93);
    checkActive<Tree2Type>(94);
    checkActive<Tree3Type>(95);
    checkActive<Tree5Type>(96);
    checkActive<tree::Tree4<double, 5, 4, 3>::Type>(97);
    checkActive<tree::Tree4<bool, 5, 4, 3>::Type>(98);
    checkActive<tree::Tree4<ValueMask, 5, 4, 3>::Type>(99);

    // A change to the active state of a voxel is seen, since a leaf gives its own mask
    using LeafT = Tree4Type::LeafNodeType;
    Tree4Type tree(0.0f);
    const Coord xyz(1, 2, 3);
    tree.setValueOn(xyz, 2.0f);
    tree::NeighborAccessor<const Tree4Type> acc(tree, xyz.offsetBy(-1, 0, 0));
    const Index offset = LeafT::coordToOffset(xyz.offsetBy(-1, 0, 0));
    EXPECT_TRUE((acc.isValueOn<1, 0, 0>(offset)));
    tree.setValueOff(xyz);
    EXPECT_FALSE((acc.isValueOn<1, 0, 0>(offset)));
    float value = 0.0f;
    EXPECT_FALSE(acc.probeValue(offset, 1, 0, 0, value));
    EXPECT_EQ(2.0f, value);
}

TEST_F(TestNeighborAccessor, testRows)
{
    Tree4Type tree(0.0f);
    populateDense(tree, 45);
    checkRows(tree);

    SmallLeafTreeType smallLeafTree(0.0f);
    populateDense(smallLeafTree, 46);
    checkRows(smallLeafTree);

    TinyLeafTreeType tinyLeafTree(0.0f);
    populateDense(tinyLeafTree, 47);
    checkRows(tinyLeafTree);

    tree::Tree4<double, 5, 4, 3>::Type doubleTree(0.0);
    populateDense(doubleTree, 48);
    checkRows(doubleTree);
}

TEST_F(TestNeighborAccessor, testValueMasks)
{
    checkMasksOf<Tree4Type>(61);
    checkMasksOf<SmallLeafTreeType>(62);
    checkMasksOf<TinyLeafTreeType>(63);
    checkMasksOf<Tree2Type>(64);
    checkMasksOf<Tree3Type>(65);
    checkMasksOf<Tree5Type>(66);
    checkMasksOf<tree::Tree4<double, 5, 4, 3>::Type>(67);
    checkMasksOf<tree::Tree4<bool, 5, 4, 3>::Type>(68);
    checkMasksOf<tree::Tree4<ValueMask, 5, 4, 3>::Type>(69);
}

TEST_F(TestNeighborAccessor, testTileMasks)
{
    using LeafT = Tree4Type::LeafNodeType;
    using MaskT = LeafT::NodeMaskType;
    constexpr int dim = int(LeafT::DIM);
    const MaskT full(true), empty(false);

    Tree4Type tree(0.0f);
    tree.setValueOn(Coord(1, 2, 3), 1.0f);
    tree.addTile(1, Coord(dim, 0, 0), 2.0f, true);
    tree.addTile(1, Coord(-dim, 0, 0), 3.0f, false);
    // A leaf next to an active root tile
    tree.setValueOn(Coord(0, 0, 4095), 4.0f);
    tree.root().addTile(Coord(0, 0, 4096), 5.0f, true);

    tree::NeighborAccessor<const Tree4Type> acc(tree, Coord(0, 0, 0));
    EXPECT_TRUE(acc.isLeaf());
    EXPECT_EQ(&tree.probeConstLeaf(Coord(0, 0, 0))->getValueMask(), &acc.getValueMask());
    EXPECT_FALSE((acc.isLeaf<1, 0, 0>()));
    EXPECT_TRUE(full == (acc.getValueMask<1, 0, 0>()));
    EXPECT_FALSE((acc.isLeaf<-1, 0, 0>()));
    EXPECT_TRUE(empty == (acc.getValueMask<-1, 0, 0>()));
    EXPECT_TRUE(empty == (acc.getValueMask<0, 1, 0>()));
    EXPECT_TRUE(empty == (acc.getValueMask<1, 1, 1>()));

    // Bound to a tile
    acc.moveTo(Coord(dim + 1, 1, 1));
    EXPECT_FALSE(acc.isLeaf());
    EXPECT_TRUE(full == acc.getValueMask());
    EXPECT_TRUE((acc.isLeaf<-1, 0, 0>()));
    EXPECT_EQ(&tree.probeConstLeaf(Coord(0, 0, 0))->getValueMask(), &(acc.getValueMask<-1, 0, 0>()));

    acc.moveTo(Coord(0, 0, 4095));
    EXPECT_TRUE(acc.isLeaf());
    EXPECT_TRUE(full == (acc.getValueMask<0, 0, 1>()));
    EXPECT_TRUE(empty == (acc.getValueMask<0, 0, -1>()));

    // The active state of a mask tile, not its value
    MaskTree mask(false);
    mask.setValueOn(Coord(0, 0, 0));
    mask.addTile(1, Coord(dim, 0, 0), true, false);
    mask.addTile(1, Coord(-dim, 0, 0), false, true);
    tree::NeighborAccessor maskAcc(mask, Coord(0, 0, 0));
    EXPECT_TRUE(maskAcc.isLeaf());
    EXPECT_TRUE(maskAcc.getValueMask().isOn(0));
    EXPECT_TRUE(MaskTree::LeafNodeType::NodeMaskType(false) == (maskAcc.getValueMask<1, 0, 0>()));
    EXPECT_TRUE(MaskTree::LeafNodeType::NodeMaskType(true) == (maskAcc.getValueMask<-1, 0, 0>()));
}

TEST_F(TestNeighborAccessor, testInteriorChildren)
{
    // A leaf at almost every child of 27 parent nodes, so that most regions
    // are away from the faces of their parent node and have leaves around them.
    // 4^3 leaves and 8^3 children per parent node keep it small.
    using LeafT = SmallLeafTreeType::LeafNodeType;
    using ParentT = SmallLeafTreeType::RootNodeType::NodeChainType::Get<1>;
    constexpr int dim = int(LeafT::DIM);
    constexpr int parentDim = int(ParentT::DIM);

    SmallLeafTreeType tree(-1.0f);
    int count = 0;
    for (int x = -parentDim; x < 2 * parentDim; x += dim) {
        for (int y = -parentDim; y < 2 * parentDim; y += dim) {
            for (int z = -parentDim; z < 2 * parentDim; z += dim) {
                ++count;
                tree.setValueOn(Coord(x + count % dim, y + (count / 3) % dim, z + (count / 7) % dim),
                    float(count));
                if (count % 5 == 0) tree.setValueOn(Coord(x, y, z), float(-count));
            }
        }
    }
    // Leaf-sized tiles among the leaves
    for (int i = 0; i < 200; ++i) {
        const Coord xyz((i * 37) % (3 * parentDim) - parentDim, (i * 53) % (3 * parentDim) - parentDim,
            (i * 71) % (3 * parentDim) - parentDim);
        tree.addTile(1, xyz, float(i), (i & 1) != 0);
    }

    checkMasks(tree);
    checkOffsets(tree,
        Offset<1, 0, 0>(), Offset<-1, 0, 0>(), Offset<0, 1, -1>(), Offset<-1, -1, 1>(),
        Offset<-dim, dim, 0>());
}

TEST_F(TestNeighborAccessor, testMoveTo)
{
    // Tiles at every level, background, leaves, in scan, random and alternating order
    checkTree<Tree4Type>(1);
    checkTree<Tree4Type>(2);
    checkTree<SmallLeafTreeType>(3);
    checkTree<TinyLeafTreeType>(4);
}

TEST_F(TestNeighborAccessor, testTreeTypes)
{
    checkTree<Tree2Type>(5);
    checkTree<Tree3Type>(6);
    checkTree<Tree5Type>(7);
}

TEST_F(TestNeighborAccessor, testLeafTypes)
{
    checkTree<tree::Tree4<double, 5, 4, 3>::Type>(8);
    checkTree<tree::Tree4<Vec3f, 5, 4, 3>::Type>(9);
    checkTree<tree::Tree4<bool, 5, 4, 3>::Type>(10);
    checkTree<tree::Tree4<ValueMask, 5, 4, 3>::Type>(11);
    checkTree<tree::Tree4<Int32, 5, 4, 3>::Type>(12);
    checkTree<tree::Tree4<PointIndex32, 5, 4, 3>::Type>(13);
}

TEST_F(TestNeighborAccessor, testSources)
{
    Tree4Type tree(-1.0f);
    populateDense(tree, 21);

    // The tree itself, which is the stencil case
    checkSource(tree, tree);

    // Another tree with the same topology
    Tree4Type copy(tree);
    checkSource(tree, copy);

    // Another leaf type
    MaskTree mask(copy, false, TopologyCopy());
    checkSource(tree, mask);

    // Voxels whose region in the tree is a tile or the background.
    // The tiles are too large to dilate.
    Tree4Type dilated(copy);
    tools::dilateActiveValues(dilated, 3, tools::NN_FACE, tools::IGNORE_TILES);
    checkSource(tree, dilated);

    // A tree read at the voxels of a bool tree
    BoolTree boolTree(false);
    populateDense(boolTree, 22);
    checkSource(boolTree, copy);
    checkSource(boolTree, boolTree);
}

TEST_F(TestNeighborAccessor, testMoveToLeaf)
{
    Tree4Type tree(0.0f);
    for (int i = 0; i < 40; ++i) {
        tree.setValueOn(Coord(i * 3, i * 5 % 37, i * 7 % 29), float(i + 1));
    }
    tree.addTile(1, Coord(400, 0, 0), 3.0f, true);

    tree::NeighborAccessor<const Tree4Type> acc(tree);
    for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        EXPECT_NO_THROW(acc.moveToLeaf(leafIter->origin()));
        EXPECT_EQ(leafIter->origin(), acc.origin());
        for (auto iter = leafIter->cbeginValueAll(); iter; ++iter) {
            ASSERT_EQ(*iter, acc.getLeafValueUnsafe(iter));
            ASSERT_EQ(*iter, acc.getLeafValueUnsafe(iter.pos()));
            ASSERT_EQ(*iter, acc.getValue(iter));
            ASSERT_EQ(tree.getValue(iter.getCoord().offsetBy(1, 0, 0)), (acc.getValue<1, 0, 0>(iter)));

            // A relative read that stays in the leaf
            const Coord local = Tree4Type::LeafNodeType::offsetToLocalCoord(iter.pos());
            const int last = int(Tree4Type::LeafNodeType::DIM) - 1;
            if (local.x() < last) {
                ASSERT_EQ(tree.getValue(iter.getCoord().offsetBy(1, 0, 0)),
                    (acc.getLeafValueUnsafe<1, 0, 0>(iter)));
            }
            if (local.y() > 0 && local.z() < last) {
                ASSERT_EQ(tree.getValue(iter.getCoord().offsetBy(0, -1, 1)),
                    (acc.getLeafValueUnsafe<0, -1, 1>(iter.pos())));
            }
        }
    }

    // A tile and the background are bound, then refused
    EXPECT_THROW(acc.moveToLeaf(Coord(401, 2, 3)), LookupError);
    EXPECT_EQ(Coord(400, 0, 0), acc.origin());
    EXPECT_EQ(3.0f, acc.getValue(Index(0)));
    EXPECT_THROW(acc.moveToLeaf(Coord(-5000, 0, 0)), LookupError);
    EXPECT_EQ(0.0f, acc.getValue(Index(7)));

    // A refusal in the bound region is the same as one after a move
    EXPECT_THROW(acc.moveToLeaf(Coord(-4999, 1, 1)), LookupError);
    EXPECT_EQ(Coord(-5000, 0, 0), acc.origin());

    EXPECT_NO_THROW(acc.moveToLeaf(Coord(0, 0, 0)));
    EXPECT_EQ(1.0f, acc.getLeafValueUnsafe(Index(0)));
}

TEST_F(TestNeighborAccessor, testReferences)
{
    using LeafT = Tree4Type::LeafNodeType;
    Tree4Type tree(0.0f);
    tree.setValueOn(Coord(1, 2, 3), 5.0f);
    tree.root().addTile(Coord(40000, 0, 0), 0.0f, false);
    tree.root().addTile(Coord(80000, 0, 0), 7.0f, true);

    tree::NeighborAccessor<const Tree4Type> acc(tree);

    // A value in a leaf is the element of the leaf buffer
    const Coord inLeaf(1, 2, 3);
    const LeafT* leaf = tree.probeConstLeaf(inLeaf);
    ASSERT_TRUE(leaf != nullptr);
    acc.moveTo(inLeaf);
    const Index offset = LeafT::coordToOffset(inLeaf);
    EXPECT_EQ(&leaf->getValue(inLeaf), &acc.getValue(offset));
    EXPECT_EQ(&leaf->getValue(Coord(2, 2, 3)), &(acc.getValue<1, 0, 0>(offset)));
    EXPECT_EQ(&leaf->getValue(Coord(1, 1, 3)), &(acc.getValue<0, -1, 0>(offset)));

    // Background is the background of the root
    acc.moveTo(Coord(-30000, 500, 9000));
    EXPECT_EQ(&tree.root().background(), &acc.getValue(Index(0)));
    EXPECT_EQ(&tree.root().background(), &(acc.getValue<1, 1, 1>(Index(0))));

    // A root tile with the same value as the background is not the background
    acc.moveTo(Coord(40010, 5, 5));
    EXPECT_NE(&tree.root().background(), &acc.getValue(Index(0)));
    EXPECT_EQ(tree.root().background(), acc.getValue(Index(0)));

    // Every voxel of a tile gives the same address
    acc.moveTo(Coord(80010, 5, 5));
    EXPECT_EQ(7.0f, acc.getValue(Index(0)));
    const float* tileValue = &acc.getValue(Index(0));
    EXPECT_EQ(tileValue, &acc.getValue(Index(LeafT::SIZE - 1)));
    EXPECT_EQ(tileValue, &(acc.getValue<1, 0, 0>(Index(LeafT::SIZE - 1))));
    acc.moveTo(Coord(80011, 6, 7));
    EXPECT_EQ(tileValue, &acc.getValue(Index(3)));
}

TEST_F(TestNeighborAccessor, testParentNodes)
{
    // Regions in and between parent nodes, so the parent node of the last
    // lookup is both reused and replaced
    using LeafT = Tree4Type::LeafNodeType;
    using ParentT = Tree4Type::RootNodeType::NodeChainType::Get<1>;
    constexpr int parentDim = int(ParentT::DIM);

    Tree4Type tree(-2.0f);
    for (int i = -3; i <= 3; ++i) {
        // A leaf at each side of a parent node face
        tree.setValueOn(Coord(i * parentDim - 1, 0, 0), float(i));
        tree.setValueOn(Coord(i * parentDim, 0, 0), float(i + 10));
    }
    // A tile at the level of the parent, and one above it
    tree.addTile(1, Coord(0, parentDim, 0), 4.0f, true);
    tree.addTile(2, Coord(0, -1, 0), 6.0f, false);

    tree::NeighborAccessor<const Tree4Type> acc(tree);
    for (int x = -4 * parentDim; x < 4 * parentDim; x += int(LeafT::DIM)) {
        for (int y : {-parentDim - 8, -8, 0, parentDim - 8, parentDim}) {
            const Coord xyz(x, y, 0);
            acc.moveTo(xyz);
            for (Index offset : {Index(0), Index(LeafT::SIZE - 1)}) {
                const Coord voxel = xyz + LeafT::offsetToLocalCoord(offset);
                ASSERT_EQ(tree.getValue(voxel), acc.getValue(offset)) << voxel;
                ASSERT_EQ(tree.getValue(voxel.offsetBy(1, 0, 0)), (acc.getValue<1, 0, 0>(offset))) << voxel;
                ASSERT_EQ(tree.getValue(voxel.offsetBy(-1, 0, 0)), (acc.getValue<-1, 0, 0>(offset))) << voxel;
                ASSERT_EQ(tree.getValue(voxel.offsetBy(0, 1, 0)), (acc.getValue<0, 1, 0>(offset))) << voxel;
                ASSERT_EQ(tree.getValue(voxel.offsetBy(0, -1, 0)), (acc.getValue<0, -1, 0>(offset))) << voxel;
            }
        }
    }
}

TEST_F(TestNeighborAccessor, testCopyAndClear)
{
    using LeafT = Tree4Type::LeafNodeType;
    Tree4Type tree(0.0f);
    populate(tree, 31);
    tree::NeighborAccessor<const Tree4Type> first(tree);
    const std::vector<Coord> coords = makeCoords(32, 5000, 400);
    for (size_t i = 0; i < 100; ++i) first.moveTo(coords[i]);

    // A copy gives the same values and moves on its own
    tree::NeighborAccessor<const Tree4Type> second(first);
    const Index lastOffset = LeafT::coordToOffset(coords[99]);
    EXPECT_EQ(first.origin(), second.origin());
    EXPECT_EQ(first.getValue(lastOffset), second.getValue(lastOffset));
    second.moveTo(Coord(300, 300, 300));
    EXPECT_EQ(coords[99] & ~Int32(LeafT::DIM - 1), first.origin());
    EXPECT_EQ(Coord(296, 296, 296), second.origin());
    for (size_t i = 0; i < coords.size(); ++i) {
        const Index offset = LeafT::coordToOffset(coords[i]);
        second.moveTo(coords[i]);
        ASSERT_EQ(tree.getValue(coords[i]), second.getValue(offset));
        ASSERT_EQ(tree.getValue(coords[i].offsetBy(1, 0, 0)), (second.getValue<1, 0, 0>(offset)));
        if (i % 2) {
            first.moveTo(coords[i]);
            ASSERT_EQ(tree.getValue(coords[i].offsetBy(0, 0, -1)), (first.getValue<0, 0, -1>(offset)));
        }
    }

    // After a topology change, clear() and moveTo() read the new tree
    Tree4Type changing(0.0f);
    tree::NeighborAccessor<const Tree4Type> acc(changing);
    const Coord xyz(100, 100, 100);
    const Index offset = LeafT::coordToOffset(xyz);
    acc.moveTo(xyz);
    EXPECT_THROW(acc.moveToLeaf(xyz), LookupError);
    EXPECT_EQ(0.0f, (acc.getValue<1, 0, 0>(offset)));

    changing.setValueOn(xyz, 2.0f);
    changing.setValueOn(xyz.offsetBy(int(LeafT::DIM), 0, 0), 3.0f);
    acc.clear();
    EXPECT_EQ(Coord::max(), acc.origin());
    EXPECT_NO_THROW(acc.moveToLeaf(xyz));
    EXPECT_EQ(2.0f, acc.getValue(offset));
    EXPECT_EQ(3.0f, (acc.getValue<int(LeafT::DIM), 0, 0>(offset)));

    changing.clear();
    acc.clear();
    acc.moveTo(xyz);
    EXPECT_THROW(acc.moveToLeaf(xyz), LookupError);
    EXPECT_EQ(0.0f, acc.getValue(offset));
}

TEST_F(TestNeighborAccessor, testConstTreeType)
{
    static_assert(std::is_same_v<tree::NeighborAccessor<const Tree4Type>::TreeType, Tree4Type>);
    static_assert(std::is_same_v<tree::NeighborAccessor<Tree4Type>::TreeType, Tree4Type>);

    Tree4Type tree(0.0f);
    tree.setValueOn(Coord(1, 1, 1), 4.0f);
    const Index offset = Tree4Type::LeafNodeType::coordToOffset(Coord(1, 1, 1));

    // The tree type is deduced and made const
    tree::NeighborAccessor acc(tree);
    static_assert(std::is_same_v<decltype(acc), tree::NeighborAccessor<const Tree4Type>>);
    EXPECT_EQ(&tree, &acc.tree());
    tree::NeighborAccessor bound(tree, Coord(1, 1, 1));
    static_assert(std::is_same_v<decltype(bound), tree::NeighborAccessor<const Tree4Type>>);
    EXPECT_EQ(4.0f, bound.getValue(offset));
    EXPECT_EQ(&tree, &bound.tree());

    tree::NeighborAccessor<Tree4Type> mutableType(tree);
    mutableType.moveTo(Coord(1, 1, 1));
    EXPECT_EQ(4.0f, mutableType.getValue(offset));
}
