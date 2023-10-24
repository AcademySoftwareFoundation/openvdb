// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <gtest/gtest.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Count.h>
#include <openvdb/tools/LevelSetSphere.h> // tools::createLevelSetSphere
#include <openvdb/tools/LevelSetUtil.h> // tools::sdfToFogVolume
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/io/TempFile.h>


class TestCount: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestCount, testCount)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(25.0f, Vec3f(0), 0.1f);
    tools::sdfToFogVolume(*grid); // convert to fog volume to generate active tiles

    // count the number of active voxels by hand in active tiles and leaf nodes

    using Internal1NodeT = FloatTree::RootNodeType::ChildNodeType;
    using Internal2NodeT = Internal1NodeT::ChildNodeType;
    using LeafNodeT = Internal2NodeT::ChildNodeType;

    Index64 activeVoxelCount1(0);
    Index64 activeLeafVoxelCount1(0);
    Index64 inactiveVoxelCount1(0);
    Index64 inactiveLeafVoxelCount1(0);
    Index64 activeTileCount1(0);

    const auto& tree = grid->tree();

    // ensure there are active tiles in this example grid

    EXPECT_TRUE(tree.activeTileCount() > 0);

    const auto& root = tree.root();

    for (auto valueIter = root.cbeginValueOn(); valueIter; ++valueIter) {
        activeVoxelCount1 += Internal1NodeT::NUM_VOXELS;
        activeTileCount1++;
    }

    for (auto valueIter = root.cbeginValueOff(); valueIter; ++valueIter) {
        if (!math::isApproxEqual(*valueIter, root.background())) {
            inactiveVoxelCount1 += Internal1NodeT::NUM_VOXELS;
        }
    }

    for (auto internal1Iter = root.cbeginChildOn(); internal1Iter; ++internal1Iter) {
        for (auto valueIter = internal1Iter->cbeginValueOn(); valueIter; ++valueIter) {
            activeVoxelCount1 += Internal2NodeT::NUM_VOXELS;
            activeTileCount1++;
        }
        for (auto valueIter = internal1Iter->cbeginChildOff(); valueIter; ++valueIter) {
            if (!valueIter.isValueOn()) {
                inactiveVoxelCount1 += Internal2NodeT::NUM_VOXELS;
            }
        }

        for (auto internal2Iter = internal1Iter->cbeginChildOn(); internal2Iter; ++internal2Iter) {
            for (auto valueIter = internal2Iter->cbeginValueOn(); valueIter; ++valueIter) {
                activeVoxelCount1 += LeafNodeT::NUM_VOXELS;
                activeTileCount1++;
            }
            for (auto valueIter = internal2Iter->cbeginChildOff(); valueIter; ++valueIter) {
                if (!valueIter.isValueOn()) {
                    inactiveVoxelCount1 += LeafNodeT::NUM_VOXELS;
                }
            }

            for (auto leafIter = internal2Iter->cbeginChildOn(); leafIter; ++leafIter) {
                activeVoxelCount1 += leafIter->onVoxelCount();
                activeLeafVoxelCount1 += leafIter->onVoxelCount();
                inactiveVoxelCount1 += leafIter->offVoxelCount();
                inactiveLeafVoxelCount1 += leafIter->offVoxelCount();
            }
        }
    }

    Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree());
    Index64 activeLeafVoxelCount2 = tools::countActiveLeafVoxels(grid->tree());
    Index64 inactiveVoxelCount2 = tools::countInactiveVoxels(grid->tree());
    Index64 inactiveLeafVoxelCount2 = tools::countInactiveLeafVoxels(grid->tree());
    Index64 activeTileCount2 = tools::countActiveTiles(grid->tree());

    EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
    EXPECT_EQ(activeLeafVoxelCount1, activeLeafVoxelCount2);
    EXPECT_EQ(inactiveVoxelCount1, inactiveVoxelCount2);
    EXPECT_EQ(inactiveLeafVoxelCount1, inactiveLeafVoxelCount2);
    EXPECT_EQ(activeTileCount1, activeTileCount2);
}


TEST_F(TestCount, testCountBBox)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(0), 0.1f);
    tools::sdfToFogVolume(*grid); // convert to fog volume to generate active tiles

    // ensure there are active tiles in this example grid

    EXPECT_TRUE(grid->tree().activeTileCount() > 0);

    { // entire bbox
        const CoordBBox bbox(Coord(-110), Coord(110));

        // count manually - iterate over all Coords in bbox and test each one

        Index64 activeVoxelCount1(0);
        Index64 activeLeafVoxelCount1(0);
        tree::ValueAccessor<const FloatTree> acc(grid->constTree());
        for (auto iter = bbox.begin(); iter; ++iter) {
            if (acc.isValueOn(*iter)) {
                activeVoxelCount1++;
                if (acc.isVoxel(*iter)) {
                    activeLeafVoxelCount1++;
                }
            }
        }

        Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree(), bbox, false);
        Index64 activeLeafVoxelCount2 = tools::countActiveLeafVoxels(grid->tree(), bbox, false);

        EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
        EXPECT_EQ(activeLeafVoxelCount1, activeLeafVoxelCount2);
    }

    { // tiny bbox
        const CoordBBox bbox(Coord(-2), Coord(2));

        // count manually - iterate over all Coords in bbox and test each one

        Index64 activeVoxelCount1(0);
        Index64 activeLeafVoxelCount1(0);
        tree::ValueAccessor<const FloatTree> acc(grid->constTree());
        for (auto iter = bbox.begin(); iter; ++iter) {
            if (acc.isValueOn(*iter)) {
                activeVoxelCount1++;
                if (acc.isVoxel(*iter)) {
                    activeLeafVoxelCount1++;
                }
            }
        }

        Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree(), bbox);
        Index64 activeLeafVoxelCount2 = tools::countActiveLeafVoxels(grid->tree(), bbox);

        EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
        EXPECT_EQ(activeLeafVoxelCount1, activeLeafVoxelCount2);
    }

    { // subset bbox
        const CoordBBox bbox(Coord(-80, -110, -80), Coord(80, 110, 80));

        // count manually - iterate over all Coords in bbox and test each one

        Index64 activeVoxelCount1(0);
        Index64 activeLeafVoxelCount1(0);
        tree::ValueAccessor<const FloatTree> acc(grid->constTree());
        for (auto iter = bbox.begin(); iter; ++iter) {
            if (acc.isValueOn(*iter)) {
                activeVoxelCount1++;
                if (acc.isVoxel(*iter)) {
                    activeLeafVoxelCount1++;
                }
            }
        }

        Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree(), bbox);
        Index64 activeLeafVoxelCount2 = tools::countActiveLeafVoxels(grid->tree(), bbox);

        EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
        EXPECT_EQ(activeLeafVoxelCount1, activeLeafVoxelCount2);
    }
}


TEST_F(TestCount, testMemUsage)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(0), 0.1f);
    tools::sdfToFogVolume(*grid); // convert to fog volume to generate active tiles

    // count the memory usage manually across all nodes

    using Internal1NodeT = FloatTree::RootNodeType::ChildNodeType;
    using Internal2NodeT = Internal1NodeT::ChildNodeType;

    const auto& tree = grid->tree();

    // ensure there are active tiles in this example grid

    EXPECT_TRUE(tree.activeTileCount() > 0);

    const auto& root = tree.root();

    Index64 internalNodeMemUsage(0);
    Index64 expectedMaxMem(sizeof(tree) + sizeof(root));
    [[maybe_unused]] Index64 leafCount(0);

    for (auto internal1Iter = root.cbeginChildOn(); internal1Iter; ++internal1Iter) {
        internalNodeMemUsage += Internal1NodeT::NUM_VALUES * sizeof(Internal1NodeT::UnionType);
        internalNodeMemUsage += internal1Iter->getChildMask().memUsage();
        internalNodeMemUsage += internal1Iter->getValueMask().memUsage();
        internalNodeMemUsage += sizeof(Coord);

        for (auto internal2Iter = internal1Iter->cbeginChildOn(); internal2Iter; ++internal2Iter) {
            internalNodeMemUsage += Internal2NodeT::NUM_VALUES * sizeof(Internal2NodeT::UnionType);
            internalNodeMemUsage += internal2Iter->getChildMask().memUsage();
            internalNodeMemUsage += internal2Iter->getValueMask().memUsage();
            internalNodeMemUsage += sizeof(Coord);

            for (auto leafIter = internal2Iter->cbeginChildOn(); leafIter; ++leafIter) {
                EXPECT_EQ(leafIter->memUsage(), leafIter->memUsageIfLoaded());
                expectedMaxMem += leafIter->memUsageIfLoaded();
                ++leafCount;
            }
        }
    }

    expectedMaxMem += internalNodeMemUsage;

    Index64 inCoreMemUsage = tools::memUsage(grid->tree());
    Index64 memUsageIfLoaded = tools::memUsageIfLoaded(grid->tree());

    EXPECT_EQ(expectedMaxMem, inCoreMemUsage);
    EXPECT_EQ(expectedMaxMem, memUsageIfLoaded);

#ifdef OPENVDB_USE_DELAYED_LOADING
    // Write out the grid and read it in with delay-loading. Check the
    // expected memory usage values.]

    openvdb::initialize();

    std::string filename;

    // write out grid to a temp file
    {
        io::TempFile file;
        filename = file.filename();
        io::File fileOut(filename);
        fileOut.write({grid});
    }

    io::File fileIn(filename);
    fileIn.open(true); // delay-load
    auto grids = fileIn.getGrids();
    fileIn.close();

    grid = GridBase::grid<FloatGrid>((*grids)[0]);
    EXPECT_TRUE(grid);

    inCoreMemUsage = tools::memUsage(grid->tree());
    memUsageIfLoaded = tools::memUsageIfLoaded(grid->tree());

    EXPECT_EQ(expectedMaxMem, memUsageIfLoaded);
    EXPECT_TRUE(inCoreMemUsage < expectedMaxMem);

    // in core memory should be the max memory without the leaf buffers but
    // with the FileInfo

    const Index64 leafBuffers = sizeof(FloatGrid::ValueType) * FloatTree::LeafNodeType::SIZE;
    const Index64 fileInfo = sizeof(FloatTree::LeafNodeType::Buffer::FileInfo);
    const Index64 expectedInCoreMemUsage = expectedMaxMem + (leafCount * (-leafBuffers + fileInfo));
    EXPECT_EQ(expectedInCoreMemUsage, inCoreMemUsage);

    std::remove(filename.c_str());

    openvdb::uninitialize();
#endif
}


namespace {

/// Helper function to test tools::minMax() for various tree types
template<typename TreeT>
void
minMaxTest()
{
    using ValueT = typename TreeT::ValueType;

    const ValueT
        zero = openvdb::zeroVal<ValueT>(),
        minusTwo = zero + (-2),
        plusTwo = zero + 2,
        five = zero + 5,
        ten = zero + 10,
        twenty = zero + 20;

    static constexpr int64_t DIM = TreeT::LeafNodeType::DIM;

    TreeT tree(/*background=*/five);

    // No set voxels (defaults to min = max = zero)
    openvdb::math::MinMax<ValueT> extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(zero, extrema.min());
    EXPECT_EQ(zero, extrema.max());

    // Only one set voxel
    tree.setValue(openvdb::Coord(0), minusTwo);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(minusTwo, extrema.min());
    EXPECT_EQ(minusTwo, extrema.max());

    // Multiple set voxels, single value
    tree.setValue(openvdb::Coord(DIM), minusTwo);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(minusTwo, extrema.min());
    EXPECT_EQ(minusTwo, extrema.max());

    // Multiple set voxels, multiple values
    tree.setValue(openvdb::Coord(DIM), plusTwo);
    tree.setValue(openvdb::Coord(DIM*2), zero);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(minusTwo, extrema.min());
    EXPECT_EQ(plusTwo, extrema.max());

    // add some empty leaf nodes to test the join op
    tree.setValueOnly(openvdb::Coord(DIM*3), ten);
    tree.setValueOnly(openvdb::Coord(DIM*4),-ten);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(minusTwo, extrema.min());
    EXPECT_EQ(plusTwo, extrema.max());

    tree.clear();

    // test tiles
    using NodeChainT = typename TreeT::RootNodeType::NodeChainType;
    using ChildT1 = typename NodeChainT::template Get<1>; // Leaf parent
    using ChildT2 = typename NodeChainT::template Get<2>; // ChildT1 parent
    tree.addTile(ChildT2::LEVEL, openvdb::Coord(0), -ten, true);
    tree.addTile(ChildT2::LEVEL, openvdb::Coord(ChildT2::DIM), ten, true);
    tree.addTile(ChildT1::LEVEL, openvdb::Coord(ChildT2::DIM + ChildT2::DIM), -twenty, false);
    tree.setValueOnly(openvdb::Coord(-1), twenty);
    tree.setValue(openvdb::Coord(-2), five);

    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(-ten, extrema.min());
    EXPECT_EQ( ten, extrema.max());
}

/// Specialization for boolean trees
template<>
void
minMaxTest<openvdb::BoolTree>()
{
    openvdb::BoolTree tree(/*background=*/false);

    // No set voxels (defaults to min = max = zero)
    openvdb::math::MinMax<bool> extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(false, extrema.min());
    EXPECT_EQ(false, extrema.max());

    // Only one set voxel
    tree.setValue(openvdb::Coord(0, 0, 0), true);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(true, extrema.min());
    EXPECT_EQ(true, extrema.max());

    // Multiple set voxels, single value
    tree.setValue(openvdb::Coord(-10, -10, -10), true);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(true, extrema.min());
    EXPECT_EQ(true, extrema.max());

    // Multiple set voxels, multiple values
    tree.setValue(openvdb::Coord(10, 10, 10), false);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(false, extrema.min());
    EXPECT_EQ(true, extrema.max());
}

/// Specialization for Coord trees
template<>
void
minMaxTest<openvdb::Coord>()
{
    using CoordTree = openvdb::tree::Tree4<openvdb::Coord,5,4,3>::Type;
    const openvdb::Coord backg(5,4,-6), a(5,4,-7), b(5,5,-6);

    CoordTree tree(backg);

    // No set voxels (defaults to min = max = zero)
    openvdb::math::MinMax<openvdb::Coord> extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(openvdb::Coord(0), extrema.min());
    EXPECT_EQ(openvdb::Coord(0), extrema.max());

    // Only one set voxel
    tree.setValue(openvdb::Coord(0, 0, 0), a);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(a, extrema.min());
    EXPECT_EQ(a, extrema.max());

    // Multiple set voxels
    tree.setValue(openvdb::Coord(-10, -10, -10), b);
    extrema = openvdb::tools::minMax(tree);
    EXPECT_EQ(a, extrema.min());
    EXPECT_EQ(b, extrema.max());
}

} // unnamed namespace

TEST_F(TestCount, testMinMax)
{
    minMaxTest<openvdb::BoolTree>();
    minMaxTest<openvdb::FloatTree>();
    minMaxTest<openvdb::Int32Tree>();
    minMaxTest<openvdb::Vec3STree>();
    minMaxTest<openvdb::Vec2ITree>();
    minMaxTest<openvdb::Coord>();
}
