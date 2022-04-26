// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/util/logging.h>
#include "util.h" // for unittest_util::makeSphere()
#include <gtest/gtest.h>
#include <set>

class TestLeafMask: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


typedef openvdb::tree::LeafNode<openvdb::ValueMask, 3> LeafType;


////////////////////////////////////////


TEST_F(TestLeafMask, testGetValue)
{
    {
        LeafType leaf1(openvdb::Coord(0, 0, 0));
        openvdb::tree::LeafNode<bool, 3> leaf2(openvdb::Coord(0, 0, 0));
        EXPECT_TRUE( leaf1.memUsage() < leaf2.memUsage() );
        //std::cerr << "\nLeafNode<ActiveState, 3> uses " << leaf1.memUsage() << " bytes" << std::endl;
        //std::cerr << "LeafNode<bool, 3> uses " << leaf2.memUsage() << " bytes" << std::endl;
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), false);
        for (openvdb::Index n = 0; n < leaf.numValues(); ++n) {
            EXPECT_EQ(false, leaf.getValue(leaf.offsetToLocalCoord(n)));
        }
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), true);
        for (openvdb::Index n = 0; n < leaf.numValues(); ++n) {
            EXPECT_EQ(true, leaf.getValue(leaf.offsetToLocalCoord(n)));
        }
    }
    {// test Buffer::data()
        LeafType leaf(openvdb::Coord(0, 0, 0), false);
        leaf.fill(true);
        LeafType::Buffer::WordType* w = leaf.buffer().data();
        for (openvdb::Index n = 0; n < LeafType::Buffer::WORD_COUNT; ++n) {
            EXPECT_EQ(~LeafType::Buffer::WordType(0), w[n]);
        }
    }
    {// test const Buffer::data()
        LeafType leaf(openvdb::Coord(0, 0, 0), false);
        leaf.fill(true);
        const LeafType& cleaf = leaf;
        const LeafType::Buffer::WordType* w = cleaf.buffer().data();
        for (openvdb::Index n = 0; n < LeafType::Buffer::WORD_COUNT; ++n) {
            EXPECT_EQ(~LeafType::Buffer::WordType(0), w[n]);
        }
    }
}


TEST_F(TestLeafMask, testSetValue)
{
    LeafType leaf(openvdb::Coord(0, 0, 0), false);

    openvdb::Coord xyz(0, 0, 0);
    EXPECT_TRUE(!leaf.isValueOn(xyz));
    leaf.setValueOn(xyz);
    EXPECT_TRUE(leaf.isValueOn(xyz));

    xyz.reset(7, 7, 7);
    EXPECT_TRUE(!leaf.isValueOn(xyz));
    leaf.setValueOn(xyz);
    EXPECT_TRUE(leaf.isValueOn(xyz));
    leaf.setValueOn(xyz, true);
    EXPECT_TRUE(leaf.isValueOn(xyz));
    leaf.setValueOn(xyz, false); // value and state are the same!
    EXPECT_TRUE(!leaf.isValueOn(xyz));

    leaf.setValueOff(xyz);
    EXPECT_TRUE(!leaf.isValueOn(xyz));

    xyz.reset(2, 3, 6);
    leaf.setValueOn(xyz);
    EXPECT_TRUE(leaf.isValueOn(xyz));

    leaf.setValueOff(xyz);
    EXPECT_TRUE(!leaf.isValueOn(xyz));
}

TEST_F(TestLeafMask, testProbeValue)
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 6, 5));

    bool val;
    EXPECT_TRUE(leaf.probeValue(openvdb::Coord(1, 6, 5), val));
    EXPECT_TRUE(!leaf.probeValue(openvdb::Coord(1, 6, 4), val));
}


TEST_F(TestLeafMask, testIterators)
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 2, 3));
    leaf.setValueOn(openvdb::Coord(5, 2, 3));
    openvdb::Coord sum;
    for (LeafType::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
        sum += iter.getCoord();
    }
    EXPECT_EQ(openvdb::Coord(1 + 5, 2 + 2, 3 + 3), sum);

    openvdb::Index count = 0;
    for (LeafType::ValueOffIter iter = leaf.beginValueOff(); iter; ++iter, ++count);
    EXPECT_EQ(leaf.numValues() - 2, count);

    count = 0;
    for (LeafType::ValueAllIter iter = leaf.beginValueAll(); iter; ++iter, ++count);
    EXPECT_EQ(leaf.numValues(), count);

    count = 0;
    for (LeafType::ChildOnIter iter = leaf.beginChildOn(); iter; ++iter, ++count);
    EXPECT_EQ(openvdb::Index(0), count);

    count = 0;
    for (LeafType::ChildOffIter iter = leaf.beginChildOff(); iter; ++iter, ++count);
    EXPECT_EQ(openvdb::Index(0), count);

    count = 0;
    for (LeafType::ChildAllIter iter = leaf.beginChildAll(); iter; ++iter, ++count);
    EXPECT_EQ(leaf.numValues(), count);
}


TEST_F(TestLeafMask, testIteratorGetCoord)
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(8, 8, 0));

    EXPECT_EQ(Coord(8, 8, 0), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    EXPECT_EQ(Coord(9, 10, 3), xyz);

    ++iter;
    xyz = iter.getCoord();
    EXPECT_EQ(Coord(13, 10, 3), xyz);
}


TEST_F(TestLeafMask, testEquivalence)
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    {
        LeafType leaf(Coord(0, 0, 0), false); // false and inactive
        LeafType leaf2(Coord(0, 0, 0), true); // true and inactive

        EXPECT_TRUE(leaf != leaf2);

        leaf.fill(CoordBBox(Coord(0), Coord(LeafType::DIM - 1)), true, false);
        EXPECT_TRUE(leaf == leaf2); // true and inactive

        leaf.setValuesOn(); // true and active

        leaf2.fill(CoordBBox(Coord(0), Coord(LeafType::DIM - 1)), false); // false and active
        EXPECT_TRUE(leaf != leaf2);

        leaf.negate(); // false and active
        EXPECT_TRUE(leaf == leaf2);

        // Set some values.
        leaf.setValueOn(Coord(0, 0, 0), true);
        leaf.setValueOn(Coord(0, 1, 0), true);
        leaf.setValueOn(Coord(1, 1, 0), true);
        leaf.setValueOn(Coord(1, 1, 2), true);

        leaf2.setValueOn(Coord(0, 0, 0), true);
        leaf2.setValueOn(Coord(0, 1, 0), true);
        leaf2.setValueOn(Coord(1, 1, 0), true);
        leaf2.setValueOn(Coord(1, 1, 2), true);

        EXPECT_TRUE(leaf == leaf2);

        leaf2.setValueOn(Coord(0, 0, 1), true);

        EXPECT_TRUE(leaf != leaf2);

        leaf2.setValueOff(Coord(0, 0, 1), false);

        EXPECT_TRUE(leaf == leaf2);//values and states coinside

        leaf2.setValueOn(Coord(0, 0, 1));

        EXPECT_TRUE(leaf != leaf2);//values and states coinside
    }
    {// test LeafNode<bool>::operator==()
        LeafType leaf1(Coord(0            , 0, 0), true); // true and inactive
        LeafType leaf2(Coord(1            , 0, 0), true); // true and inactive
        LeafType leaf3(Coord(LeafType::DIM, 0, 0), true); // true and inactive
        LeafType leaf4(Coord(0            , 0, 0), true, true);//true and active
        EXPECT_TRUE(leaf1 == leaf2);
        EXPECT_TRUE(leaf1 != leaf3);
        EXPECT_TRUE(leaf2 != leaf3);
        EXPECT_TRUE(leaf1 == leaf4);
        EXPECT_TRUE(leaf2 == leaf4);
        EXPECT_TRUE(leaf3 != leaf4);
    }

}


TEST_F(TestLeafMask, testGetOrigin)
{
    {
        LeafType leaf(openvdb::Coord(1, 0, 0), 1);
        EXPECT_EQ(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), 1);
        EXPECT_EQ(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 0, 0), 1);
        EXPECT_EQ(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 1, 0), 1);
        EXPECT_EQ(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1024, 1, 3), 1);
        EXPECT_EQ(openvdb::Coord(128*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1023, 1, 3), 1);
        EXPECT_EQ(openvdb::Coord(127*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(512, 512, 512), 1);
        EXPECT_EQ(openvdb::Coord(512, 512, 512), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(2, 52, 515), 1);
        EXPECT_EQ(openvdb::Coord(0, 48, 512), leaf.origin());
    }
}


TEST_F(TestLeafMask, testNegativeIndexing)
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(-9, -2, -8));

    EXPECT_EQ(Coord(-16, -8, -8), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3));
    leaf.setValueOn(Coord(5, 2, 3));

    EXPECT_TRUE(leaf.isValueOn(Coord(1, 2, 3)));
    EXPECT_TRUE(leaf.isValueOn(Coord(5, 2, 3)));

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    EXPECT_EQ(Coord(-15, -6, -5), xyz);

    ++iter;
    xyz = iter.getCoord();
    EXPECT_EQ(Coord(-11, -6, -5), xyz);
}


TEST_F(TestLeafMask, testIO)
{
    LeafType leaf(openvdb::Coord(1, 3, 5));
    const openvdb::Coord origin = leaf.origin();

    leaf.setValueOn(openvdb::Coord(0, 1, 0));
    leaf.setValueOn(openvdb::Coord(1, 0, 0));

    std::ostringstream ostr(std::ios_base::binary);

    leaf.writeBuffers(ostr);

    leaf.setValueOff(openvdb::Coord(0, 1, 0));
    leaf.setValueOn(openvdb::Coord(0, 1, 1));

    std::istringstream istr(ostr.str(), std::ios_base::binary);
    // Since the input stream doesn't include a VDB header with file format version info,
    // tag the input stream explicitly with the current version number.
    openvdb::io::setCurrentVersion(istr);

    leaf.readBuffers(istr);

    EXPECT_EQ(origin, leaf.origin());

    EXPECT_TRUE(leaf.isValueOn(openvdb::Coord(0, 1, 0)));
    EXPECT_TRUE(leaf.isValueOn(openvdb::Coord(1, 0, 0)));

    EXPECT_TRUE(leaf.onVoxelCount() == 2);
}


TEST_F(TestLeafMask, testTopologyCopy)
{
    using openvdb::Coord;

    // LeafNode<float, Log2Dim> having the same Log2Dim as LeafType
    typedef LeafType::ValueConverter<float>::Type FloatLeafType;

    FloatLeafType fleaf(Coord(10, 20, 30), -1.0);
    std::set<Coord> coords;
    for (openvdb::Index n = 0; n < fleaf.numValues(); n += 10) {
        Coord xyz = fleaf.offsetToGlobalCoord(n);
        fleaf.setValueOn(xyz, float(n));
        coords.insert(xyz);
    }

    LeafType leaf(fleaf, openvdb::TopologyCopy());
    EXPECT_EQ(fleaf.onVoxelCount(), leaf.onVoxelCount());

    EXPECT_TRUE(leaf.hasSameTopology(&fleaf));

    for (LeafType::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
        coords.erase(iter.getCoord());
    }
    EXPECT_TRUE(coords.empty());
}


TEST_F(TestLeafMask, testMerge)
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    for (openvdb::Index n = 0; n < leaf.numValues(); n += 10) {
        leaf.setValueOn(n);
    }
    EXPECT_TRUE(!leaf.isValueMaskOn());
    EXPECT_TRUE(!leaf.isValueMaskOff());
    bool val = false, active = false;
    EXPECT_TRUE(!leaf.isConstant(val, active));

    LeafType leaf2(leaf);
    leaf2.getValueMask().toggle();
    EXPECT_TRUE(!leaf2.isValueMaskOn());
    EXPECT_TRUE(!leaf2.isValueMaskOff());
    val = active = false;
    EXPECT_TRUE(!leaf2.isConstant(val, active));

    leaf.merge<openvdb::MERGE_ACTIVE_STATES>(leaf2);
    EXPECT_TRUE(leaf.isValueMaskOn());
    EXPECT_TRUE(!leaf.isValueMaskOff());
    val = active = false;
    EXPECT_TRUE(leaf.isConstant(val, active));
    EXPECT_TRUE(active);
}


TEST_F(TestLeafMask, testCombine)
{
    struct Local {
        static void op(openvdb::CombineArgs<bool>& args) {
            args.setResult(args.aIsActive() ^ args.bIsActive());// state = value
        }
    };

    LeafType leaf(openvdb::Coord(0, 0, 0));
    for (openvdb::Index n = 0; n < leaf.numValues(); n += 10) leaf.setValueOn(n);
    EXPECT_TRUE(!leaf.isValueMaskOn());
    EXPECT_TRUE(!leaf.isValueMaskOff());
    const LeafType::NodeMaskType savedMask = leaf.getValueMask();
    OPENVDB_LOG_DEBUG_RUNTIME(leaf.str());

    LeafType leaf2(leaf);
    for (openvdb::Index n = 0; n < leaf.numValues(); n += 4) leaf2.setValueOn(n);

    EXPECT_TRUE(!leaf2.isValueMaskOn());
    EXPECT_TRUE(!leaf2.isValueMaskOff());
    OPENVDB_LOG_DEBUG_RUNTIME(leaf2.str());

    leaf.combine(leaf2, Local::op);
    OPENVDB_LOG_DEBUG_RUNTIME(leaf.str());

    EXPECT_TRUE(leaf.getValueMask() == (savedMask ^ leaf2.getValueMask()));
}


TEST_F(TestLeafMask, testTopologyTree)
{
    using namespace openvdb;

#if 0
    FloatGrid::Ptr inGrid;
    FloatTree::Ptr inTree;
    {
        //io::File vdbFile("/work/rd/fx_tools/vdb_unittest/TestGridCombine::testCsg/large1.vdb2");
        io::File vdbFile("/hosts/whitestar/usr/pic1/VDB/bunny_0256.vdb2");
        vdbFile.open();
        inGrid = gridPtrCast<FloatGrid>(vdbFile.readGrid("LevelSet"));
        EXPECT_TRUE(inGrid.get() != NULL);
        inTree = inGrid->treePtr();
        EXPECT_TRUE(inTree.get() != NULL);
    }
#else
    FloatGrid::Ptr inGrid = FloatGrid::create();
    EXPECT_TRUE(inGrid.get() != NULL);
    FloatTree& inTree = inGrid->tree();
    inGrid->setName("LevelSet");

    unittest_util::makeSphere<FloatGrid>(Coord(128),//dim
                                         Vec3f(0, 0, 0),//center
                                         5,//radius
                                         *inGrid, unittest_util::SPHERE_DENSE);
#endif

    const Index64
        floatTreeMem = inTree.memUsage(),
        floatTreeLeafCount = inTree.leafCount(),
        floatTreeVoxelCount = inTree.activeVoxelCount();

    TreeBase::Ptr outTree(new TopologyTree(inTree, false, true, TopologyCopy()));
    EXPECT_TRUE(outTree.get() != NULL);

    TopologyGrid::Ptr outGrid = TopologyGrid::create(*inGrid); // copy transform and metadata
    outGrid->setTree(outTree);
    outGrid->setName("Boolean");

    const Index64
        boolTreeMem = outTree->memUsage(),
        boolTreeLeafCount = outTree->leafCount(),
        boolTreeVoxelCount = outTree->activeVoxelCount();

#if 0
    GridPtrVec grids;
    grids.push_back(inGrid);
    grids.push_back(outGrid);
    io::File vdbFile("bool_tree.vdb2");
    vdbFile.write(grids);
    vdbFile.close();
#endif

    EXPECT_EQ(floatTreeLeafCount, boolTreeLeafCount);
    EXPECT_EQ(floatTreeVoxelCount, boolTreeVoxelCount);

    //std::cerr << "\nboolTree mem=" << boolTreeMem << " bytes" << std::endl;
    //std::cerr << "floatTree mem=" << floatTreeMem << " bytes" << std::endl;

    // Considering only voxel buffer memory usage, the BoolTree would be expected
    // to use (2 mask bits/voxel / ((32 value bits + 1 mask bit)/voxel)) = ~1/16
    // as much memory as the FloatTree.  Considering total memory usage, verify that
    // the BoolTree is no more than 1/10 the size of the FloatTree.
    EXPECT_TRUE(boolTreeMem * 10 <= floatTreeMem);
}

TEST_F(TestLeafMask, testMedian)
{
    using namespace openvdb;
    LeafType leaf(openvdb::Coord(0, 0, 0), /*background=*/false);
    bool state = false;

    EXPECT_EQ(Index(0), leaf.medianOn(state));
    EXPECT_TRUE(state == true);
    EXPECT_EQ(leaf.numValues(), leaf.medianOff(state));
    EXPECT_TRUE(state == false);
    EXPECT_TRUE(!leaf.medianAll());

    leaf.setValue(Coord(0,0,0), true);
    EXPECT_EQ(Index(1), leaf.medianOn(state));
    EXPECT_TRUE(state == true);
    EXPECT_EQ(leaf.numValues()-1, leaf.medianOff(state));
    EXPECT_TRUE(state == false);
    EXPECT_TRUE(!leaf.medianAll());


    leaf.setValue(Coord(0,0,1), true);
    EXPECT_EQ(Index(2), leaf.medianOn(state));
    EXPECT_TRUE(state == true);
    EXPECT_EQ(leaf.numValues()-2, leaf.medianOff(state));
    EXPECT_TRUE(state == false);
    EXPECT_TRUE(!leaf.medianAll());


    leaf.setValue(Coord(5,0,1), true);
    EXPECT_EQ(Index(3), leaf.medianOn(state));
    EXPECT_TRUE(state == true);
    EXPECT_EQ(leaf.numValues()-3, leaf.medianOff(state));
    EXPECT_TRUE(state == false);
    EXPECT_TRUE(!leaf.medianAll());


    leaf.fill(false, false);
    EXPECT_EQ(Index(0), leaf.medianOn(state));
    EXPECT_TRUE(state == true);
    EXPECT_EQ(leaf.numValues(), leaf.medianOff(state));
    EXPECT_TRUE(state == false);
    EXPECT_TRUE(!leaf.medianAll());


    for (Index i=0; i<leaf.numValues()/2; ++i) {
        leaf.setValueOn(i, true);
        EXPECT_TRUE(!leaf.medianAll());
        EXPECT_EQ(Index(i+1), leaf.medianOn(state));
        EXPECT_TRUE(state == true);
        EXPECT_EQ(leaf.numValues()-i-1, leaf.medianOff(state));
        EXPECT_TRUE(state == false);
    }
    for (Index i=leaf.numValues()/2; i<leaf.numValues(); ++i) {
        leaf.setValueOn(i, true);
        EXPECT_TRUE(leaf.medianAll());
        EXPECT_EQ(Index(i+1), leaf.medianOn(state));
        EXPECT_TRUE(state == true);
        EXPECT_EQ(leaf.numValues()-i-1, leaf.medianOff(state));
        EXPECT_TRUE(state == false);
    }
}

// void
// TestLeafMask::testFilter()
// {
//     using namespace openvdb;

//     BoolGrid::Ptr grid = BoolGrid::create();
//     EXPECT_TRUE(grid.get() != NULL);
//     BoolTree::Ptr tree = grid->treePtr();
//     EXPECT_TRUE(tree.get() != NULL);
//     grid->setName("filtered");

//     unittest_util::makeSphere<BoolGrid>(Coord(32),// dim
//                                         Vec3f(0, 0, 0),// center
//                                         10,// radius
//                                         *grid, unittest_util::SPHERE_DENSE);

//     BoolTree::Ptr copyOfTree(new BoolTree(*tree));
//     BoolGrid::Ptr copyOfGrid = BoolGrid::create(copyOfTree);
//     copyOfGrid->setName("original");

//     tools::Filter<BoolGrid> filter(*grid);
//     filter.offset(1);

// #if 0
//     GridPtrVec grids;
//     grids.push_back(copyOfGrid);
//     grids.push_back(grid);
//     io::File vdbFile("TestLeafMask::testFilter.vdb2");
//     vdbFile.write(grids);
//     vdbFile.close();
// #endif

//     // Verify that offsetting all active voxels by 1 (true) has no effect,
//     // since the active voxels were all true to begin with.
//     EXPECT_TRUE(tree->hasSameTopology(*copyOfTree));
// }

#if OPENVDB_ABI_VERSION_NUMBER >= 9
TEST_F(TestLeafMask, testTransientData)
{
    LeafType leaf(openvdb::Coord(0, 0, 0), /*background=*/false);

    EXPECT_EQ(openvdb::Index32(0), leaf.transientData());
    leaf.setTransientData(openvdb::Index32(5));
    EXPECT_EQ(openvdb::Index32(5), leaf.transientData());
    LeafType leaf2(leaf);
    EXPECT_EQ(openvdb::Index32(5), leaf2.transientData());
    LeafType leaf3 = leaf;
    EXPECT_EQ(openvdb::Index32(5), leaf3.transientData());
}
#endif
