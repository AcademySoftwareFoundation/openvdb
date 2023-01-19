// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/ValueTransformer.h>
#include "util.h" // for unittest_util::makeSphere()
#include <gtest/gtest.h>

// @todo gaussian!
class TestFilter: public ::testing::Test {};


////////////////////////////////////////


inline openvdb::FloatGrid::ConstPtr
createReferenceGrid(const openvdb::Coord& dim)
{
    openvdb::FloatGrid::Ptr referenceGrid =
        openvdb::FloatGrid::create(/*background=*/5.0);

    const openvdb::Vec3f center(25.0f, 20.0f, 20.0f);
    const float radius = 10.0f;
    unittest_util::makeSphere<openvdb::FloatGrid>(
        dim, center, radius, *referenceGrid, unittest_util::SPHERE_DENSE);

    EXPECT_EQ(dim[0]*dim[1]*dim[2],
        int(referenceGrid->tree().activeVoxelCount()));
    return referenceGrid;
}


////////////////////////////////////////


TEST_F(TestFilter, testOffset)
{
    const openvdb::Coord dim(40);
    const openvdb::FloatGrid::ConstPtr referenceGrid = createReferenceGrid(dim);
    const openvdb::FloatTree& sphere = referenceGrid->tree();

    openvdb::Coord xyz;
    openvdb::FloatGrid::Ptr grid = referenceGrid->deepCopy();
    openvdb::FloatTree& tree = grid->tree();
    openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
    const float offset = 2.34f;
    filter.setGrainSize(0);//i.e. disable threading
    filter.offset(offset);
    for (int x=0; x<dim[0]; ++x) {
        xyz[0]=x;
        for (int y=0; y<dim[1]; ++y) {
            xyz[1]=y;
            for (int z=0; z<dim[2]; ++z) {
                xyz[2]=z;
                float delta = sphere.getValue(xyz) + offset - tree.getValue(xyz);
                //if (fabs(delta)>0.0001f) std::cerr << " failed at " << xyz << std::endl;
                EXPECT_NEAR(0.0f, delta, /*tolerance=*/0.0001);
            }
        }
    }
    filter.setGrainSize(1);//i.e. enable threading
    filter.offset(-offset);//default is multi-threaded
    for (int x=0; x<dim[0]; ++x) {
        xyz[0]=x;
        for (int y=0; y<dim[1]; ++y) {
            xyz[1]=y;
            for (int z=0; z<dim[2]; ++z) {
                xyz[2]=z;
                float delta = sphere.getValue(xyz) - tree.getValue(xyz);
                //if (fabs(delta)>0.0001f) std::cerr << " failed at " << xyz << std::endl;
                EXPECT_NEAR(0.0f, delta, /*tolerance=*/0.0001);
            }
        }
    }
}


TEST_F(TestFilter, testMedian)
{
    const openvdb::Coord dim(40);
    const openvdb::FloatGrid::ConstPtr referenceGrid = createReferenceGrid(dim);
    const openvdb::FloatTree& sphere = referenceGrid->tree();

    openvdb::Coord xyz;
    openvdb::FloatGrid::Ptr filteredGrid = referenceGrid->deepCopy();
    openvdb::FloatTree& filteredTree = filteredGrid->tree();
    const int width = 2;
    openvdb::math::DenseStencil<openvdb::FloatGrid> stencil(*referenceGrid, width);
    openvdb::tools::Filter<openvdb::FloatGrid> filter(*filteredGrid);
    filter.median(width, /*interations=*/1);
    std::vector<float> tmp;
    for (int x=0; x<dim[0]; ++x) {
        xyz[0]=x;
        for (int y=0; y<dim[1]; ++y) {
            xyz[1]=y;
            for (int z=0; z<dim[2]; ++z) {
                xyz[2]=z;
                for (int i = xyz[0] - width, ie= xyz[0] + width; i <= ie; ++i) {
                    openvdb::Coord ijk(i,0,0);
                    for (int j = xyz[1] - width, je = xyz[1] + width; j <= je; ++j) {
                        ijk.setY(j);
                        for (int k = xyz[2] - width, ke = xyz[2] + width; k <= ke; ++k) {
                            ijk.setZ(k);
                            tmp.push_back(sphere.getValue(ijk));
                        }
                    }
                }
                std::sort(tmp.begin(), tmp.end());
                stencil.moveTo(xyz);
                EXPECT_NEAR(
                    tmp[(tmp.size()-1)/2], stencil.median(), /*tolerance=*/0.0001);
                EXPECT_NEAR(
                    stencil.median(), filteredTree.getValue(xyz), /*tolerance=*/0.0001);
                tmp.clear();
            }
        }
    }
}


TEST_F(TestFilter, testMean)
{
    const openvdb::Coord dim(40);
    const openvdb::FloatGrid::ConstPtr referenceGrid = createReferenceGrid(dim);
    const openvdb::FloatTree& sphere = referenceGrid->tree();

    openvdb::Coord xyz;
    openvdb::FloatGrid::Ptr filteredGrid = referenceGrid->deepCopy();
    openvdb::FloatTree& filteredTree = filteredGrid->tree();
    const int width = 2;
    openvdb::math::DenseStencil<openvdb::FloatGrid> stencil(*referenceGrid, width);
    openvdb::tools::Filter<openvdb::FloatGrid> filter(*filteredGrid);
    filter.mean(width,  /*interations=*/1);
    for (int x=0; x<dim[0]; ++x) {
        xyz[0]=x;
        for (int y=0; y<dim[1]; ++y) {
            xyz[1]=y;
            for (int z=0; z<dim[2]; ++z) {
                xyz[2]=z;
                double sum =0.0, count=0.0;
                for (int i = xyz[0] - width, ie= xyz[0] + width; i <= ie; ++i) {
                    openvdb::Coord ijk(i,0,0);
                    for (int j = xyz[1] - width, je = xyz[1] + width; j <= je; ++j) {
                        ijk.setY(j);
                        for (int k = xyz[2] - width, ke = xyz[2] + width; k <= ke; ++k) {
                            ijk.setZ(k);
                            sum += sphere.getValue(ijk);
                            count += 1.0;
                        }
                    }
                }
                stencil.moveTo(xyz);
                EXPECT_NEAR(
                    sum/count, stencil.mean(), /*tolerance=*/0.0001);
                EXPECT_NEAR(
                    stencil.mean(), filteredTree.getValue(xyz), 0.0001);
            }
        }
    }
}


TEST_F(TestFilter, testFilterTiles)
{
    using openvdb::Coord;
    using openvdb::Index32;
    using openvdb::Index64;

    using InternalNode1 = openvdb::FloatTree::RootNodeType::ChildNodeType; // usually 4096^3
    using InternalNode2 = InternalNode1::ChildNodeType; // usually 128^3
    using LeafT = openvdb::FloatTree::LeafNodeType;

    struct Settings {
        Settings(Index32 a, Index64 b, Index32 c, Index64 d)
            : mLevel(a), mVoxels(b), mLeafs(c), mTiles(d) {}
        Index32 mLevel; // level to create the tile
        Index64 mVoxels; // expected active voxel count
        Index32 mLeafs; // num leaf nodes after tile processing
        Index64 mTiles; // num tiles after tile processing
    };

    struct CheckMeanValues {
        mutable openvdb::math::DenseStencil<openvdb::FloatGrid> mStencil;
        CheckMeanValues(openvdb::math::DenseStencil<openvdb::FloatGrid>& s) : mStencil(s) {}
        inline void operator()(const openvdb::FloatTree::ValueOnCIter& iter) const {
            mStencil.moveTo(iter.getCoord());
            EXPECT_NEAR(mStencil.mean(), *iter, /*tolerance=*/0.0001);
        }
    };

    struct CheckMedianValues {
        mutable openvdb::math::DenseStencil<openvdb::FloatGrid> mStencil;
        CheckMedianValues(openvdb::math::DenseStencil<openvdb::FloatGrid>& s) : mStencil(s) {}
        inline void operator()(const openvdb::FloatTree::ValueOnCIter& iter) const {
            mStencil.moveTo(iter.getCoord());
            EXPECT_NEAR(mStencil.median(), *iter, /*tolerance=*/0.0001);
        }
    };

    std::vector<Settings> tests;
    // leaf level tile, 8x8x8, 512 voxels
    tests.emplace_back(Settings(1, 512, 1, 0));

    // given a dimension in voxels, compute how many boundary nodes exist
    auto computeBoundaryNodeCount = [](const Index32 voxels, const Index32 nodedim) {
        Index32 leafPerDim = voxels/nodedim;
        Index32 faceLeafNodes = openvdb::math::Pow2(leafPerDim);
        Index32 boundary = faceLeafNodes * 2; // x faces
        boundary += (faceLeafNodes * 2) - ((leafPerDim)*2)*2; // y faces
        boundary += openvdb::math::Pow2(leafPerDim-2)*2; // z faces
        return boundary;
    };

    // first internal node, usually 128x128x128, 2097152 voxels
    // with a width =1 and iter = 1 all edge leaf nodes should be generated
    Index32 expectedLeafNodes = computeBoundaryNodeCount(InternalNode2::DIM, LeafT::DIM);
    Index32 expectedTiles = InternalNode2::NUM_VALUES - expectedLeafNodes;
    tests.emplace_back(Settings(2, InternalNode2::NUM_VOXELS, expectedLeafNodes, expectedTiles));

    // @note  Not testing larger tiles values as it requires more memory/time.
    //  Uncomment the below test to test with level 3 tiles.
    /*
    expectedLeafNodes = computeBoundaryNodeCount(InternalNode1::DIM, LeafT::DIM);
    Index32 numBoundary = computeBoundaryNodeCount(InternalNode1::DIM, InternalNode2::DIM);
    expectedTiles = InternalNode1::NUM_VALUES - numBoundary;
    expectedTiles += (numBoundary * InternalNode2::NUM_VALUES) - expectedLeafNodes;
    tests.emplace_back(Settings(3, InternalNode1::NUM_VOXELS, expectedLeafNodes, expectedTiles));
    */


    int width = 1, iter = 1;

    // Test the behaviour with tiled grids - the mean/median tests check that the
    // filtering  operations correctly create leaf nodes on all face/edge/vertex
    // boundaries  of a given tile as the tiles value differs from the trees
    // background value

    for(const auto& test : tests)
    {
        { // single tile
            openvdb::FloatGrid::ConstPtr refTile;
            {
                openvdb::FloatGrid::Ptr ref = openvdb::FloatGrid::create(0.0f);
                auto& tree = ref->tree();
                tree.addTile(test.mLevel, Coord(0), 1.0f, true);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));
                refTile = ref;
            }

            openvdb::math::DenseStencil<openvdb::FloatGrid> stencil(*refTile, width);

            { // offset
                openvdb::FloatGrid::Ptr grid = refTile->deepCopy();
                auto& tree = grid->tree();
                openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
                // disable tile processing, do nothing
                filter.setProcessTiles(false);
                filter.offset(1.0f);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));

                // enable
                filter.setProcessTiles(true);
                filter.offset(1.0f);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(2.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));
            }

            { // mean
                openvdb::FloatGrid::Ptr grid = refTile->deepCopy();
                auto& tree = grid->tree();
                openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
                // disable tile processing, do nothing
                filter.setProcessTiles(false);
                filter.mean(width, iter);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));

                // enable
                filter.setProcessTiles(true);
                filter.mean(width, iter);
                EXPECT_EQ(test.mLeafs, tree.leafCount());
                EXPECT_EQ(test.mTiles, tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                CheckMeanValues op(stencil);
                openvdb::tools::foreach(tree.cbeginValueOn(), op, true, false);
            }

            { // median
                openvdb::FloatGrid::Ptr grid = refTile->deepCopy();
                auto& tree = grid->tree();
                openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
                // disable tile processing, do nothing
                filter.setProcessTiles(false);
                filter.median(width, iter);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));

                // enable
                filter.setProcessTiles(true);
                filter.median(width, iter);
                EXPECT_EQ(test.mLeafs, tree.leafCount());
                EXPECT_EQ(test.mTiles, tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                CheckMedianValues op(stencil);
                openvdb::tools::foreach(tree.cbeginValueOn(), op, true, false);

            }
        }
    }

    // test with matching background - tree should not
    // be voxelized as there is no work to do

    for(const auto& test : tests)
    {
        // Test the behaviour with tiled grids
        { // single tile
            openvdb::FloatGrid::ConstPtr refTile;
            {
                openvdb::FloatGrid::Ptr ref = openvdb::FloatGrid::create(1.0f);
                auto& tree = ref->tree();
                tree.addTile(test.mLevel, Coord(0), 1.0f, true);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));
                refTile = ref;
            }

            openvdb::math::DenseStencil<openvdb::FloatGrid> stencil(*refTile, width);

            { // mean
                openvdb::FloatGrid::Ptr grid = refTile->deepCopy();
                auto& tree = grid->tree();
                openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
                filter.setProcessTiles(true);
                filter.mean(width, iter);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));
            }

            { // median
                openvdb::FloatGrid::Ptr grid = refTile->deepCopy();
                auto& tree = grid->tree();
                openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
                filter.setProcessTiles(true);
                filter.median(width, iter);
                EXPECT_EQ(Index32(0), tree.leafCount());
                EXPECT_EQ(Index64(1), tree.activeTileCount());
                EXPECT_EQ(test.mVoxels, tree.activeVoxelCount());
                EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
                EXPECT_TRUE(tree.isValueOn(Coord(0)));
            }
        }
    }

    // test that node neighbours with a different value forces
    // voxelization with different width/iter combos. Ensure
    // matching background with main tile

    { // single tile at level 1
        openvdb::FloatGrid::ConstPtr refTile;
        {
            openvdb::FloatGrid::Ptr ref = openvdb::FloatGrid::create(1.0f);
            auto& tree = ref->tree();
            tree.addTile(1, Coord(0), 1.0f, true);
            EXPECT_EQ(Index32(0), tree.leafCount());
            EXPECT_EQ(Index64(1), tree.activeTileCount());
            EXPECT_EQ(Index64(LeafT::NUM_VALUES), tree.activeVoxelCount());
            EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
            EXPECT_TRUE(tree.isValueOn(Coord(0)));
            refTile = ref;
        }

        { // mean
            openvdb::FloatGrid::Ptr grid = refTile->deepCopy();
            auto& tree = grid->tree();
            openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
            //
            filter.setProcessTiles(true);
            filter.mean(1, 1);
            EXPECT_EQ(Index32(0), tree.leafCount());
            EXPECT_EQ(Index64(1), tree.activeTileCount());
            EXPECT_EQ(Index64(LeafT::NUM_VALUES), tree.activeVoxelCount());
            EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
            EXPECT_TRUE(tree.isValueOn(Coord(0)));

            // create leaf neighbour
            tree.touchLeaf(Coord(-1,0,0));
            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index64(1), tree.activeTileCount());

            filter.mean(1, 1);
            EXPECT_EQ(Index32(2), tree.leafCount());
            EXPECT_EQ(Index64(0), tree.activeTileCount());
            EXPECT_EQ(Index64(LeafT::NUM_VALUES), tree.activeVoxelCount());
        }
    }

    {
        // single tile at a given level with a leaf and level 1 neighbour
        auto reset = [](const int level) {
            openvdb::FloatGrid::Ptr ref = openvdb::FloatGrid::create(1.0f);
            auto& tree = ref->tree();
            tree.addTile(level, Coord(0), 1.0f, true);
            EXPECT_EQ(Index32(0), tree.leafCount());
            EXPECT_EQ(Index64(1), tree.activeTileCount());
            EXPECT_EQ(1.0f, tree.getValue(Coord(0)));
            EXPECT_TRUE(tree.isValueOn(Coord(0)));

            // create a leaf and tile neighbour
            tree.touchLeaf(Coord(-int(LeafT::DIM),0,0));
            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index64(1), tree.activeTileCount());
            // create tile level 1 neighbour with a different value
            tree.addTile(1, Coord(-int(LeafT::DIM),0,LeafT::DIM*3), 2.0f, true);
            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index64(2), tree.activeTileCount());
            return ref;
        };

        {
            openvdb::FloatGrid::Ptr grid = reset(3);
            auto& tree = grid->tree();
            openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
            filter.setProcessTiles(true);
            // with a width = 9 and iter = 1, only face neighbours need to be
            // created. for a leaf at (-8,0,0) which is on the corner, this
            // creates 4 nodes. for a tile at (-8,0,24) this creates 5 nodes
            // (+ itself becomes a leaf)
            filter.mean(/*width*/LeafT::DIM+1, /*iter*/1);
            // 2 leaf nodes from the tile/leaf neighbours + their neighbours
            EXPECT_EQ(Index32(2+4+5), tree.leafCount());
            EXPECT_EQ((Index64(InternalNode1::NUM_VALUES) - 1) +
                (Index64(InternalNode2::NUM_VALUES) - (4+5)), tree.activeTileCount());
            EXPECT_EQ(Index64(InternalNode1::NUM_VOXELS) +
                Index64(LeafT::NUM_VOXELS), tree.activeVoxelCount());
        }

        {
            openvdb::FloatGrid::Ptr grid = reset(3);
            auto& tree = grid->tree();
            openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
            filter.setProcessTiles(true);
            // with width = 2 and iter = 2, edge/vertex neighbours should also be voxelized
            filter.mean(/*width*/2, /*iter*/2);
            EXPECT_EQ(Index32(2+4+6), tree.leafCount());
            EXPECT_EQ((Index64(InternalNode1::NUM_VALUES) - 1) +
                (Index64(InternalNode2::NUM_VALUES) - (4+6)), tree.activeTileCount());
            EXPECT_EQ(Index64(InternalNode1::NUM_VOXELS) +
                Index64(LeafT::NUM_VOXELS), tree.activeVoxelCount());
        }

        {
            openvdb::FloatGrid::Ptr grid = reset(2); // test at level 2 for speed
            auto& tree = grid->tree();
            openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
            filter.setProcessTiles(true);
            // with width = 1 and iter = 9 - checks an iter count > LeafT::DIM
            filter.mean(/*width*/1, /*iter*/LeafT::DIM+1);
            EXPECT_EQ(Index32(38), tree.leafCount());
            EXPECT_EQ((Index64(InternalNode2::NUM_VALUES) - 36), tree.activeTileCount());
            EXPECT_EQ(Index64(InternalNode2::NUM_VOXELS) +
                Index64(LeafT::NUM_VOXELS), tree.activeVoxelCount());
        }
    }
}

