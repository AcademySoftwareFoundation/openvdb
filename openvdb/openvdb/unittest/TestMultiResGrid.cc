// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/MultiResGrid.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Diagnostics.h>

#include <gtest/gtest.h>

#include <cstdio> // for remove()


class TestMultiResGrid : public ::testing::Test
{
public:
    // Use to test logic in openvdb::tools::MultiResGrid
    struct CoordMask {
        static int Mask(int i, int j, int k) { return (i & 1) | ((j & 1) << 1) | ((k & 1) << 2); }
        CoordMask() : mask(0) {}
        CoordMask(const openvdb::Coord &c ) : mask( Mask(c[0],c[1],c[2]) ) {}
        inline void setCoord(int i, int j, int k) { mask = Mask(i,j,k); }
        inline void setCoord(const openvdb::Coord &c) { mask = Mask(c[0],c[1],c[2]); }
        inline bool allEven() const { return mask == 0; }
        inline bool xOdd()    const { return mask == 1; }
        inline bool yOdd()    const { return mask == 2; }
        inline bool zOdd()    const { return mask == 4; }
        inline bool xyOdd()   const { return mask == 3; }
        inline bool xzOdd()   const { return mask == 5; }
        inline bool yzOdd()   const { return mask == 6; }
        inline bool allOdd()  const { return mask == 7; }
        int mask;
    };// CoordMask
};


// Uncomment to test on models from our web-site
//#define TestMultiResGrid_DATA_PATH "/home/kmu/src/openvdb/data/"
//#define TestMultiResGrid_DATA_PATH "/usr/pic1/Data/OpenVDB/LevelSetModels/"

TEST_F(TestMultiResGrid, testTwosComplement)
{
    // test bit-operations that assume 2's complement representation of negative integers
    EXPECT_EQ( 1, 13 & 1 );// odd
    EXPECT_EQ( 1,-13 & 1 );// odd
    EXPECT_EQ( 0, 12 & 1 );// even
    EXPECT_EQ( 0,-12 & 1 );// even
    EXPECT_EQ( 0,  0 & 1 );// even
    for (int i=-50; i<=50; ++i) {
        if ( (i % 2) == 0 ) {//i.e. even number
            EXPECT_EQ( 0, i & 1);
            EXPECT_EQ( i, (i >> 1) << 1 );
        } else {//i.e. odd number
            EXPECT_EQ( 1, i & 1);
            EXPECT_TRUE( i != (i >> 1) << 1 );
        }
    }
}


TEST_F(TestMultiResGrid, testCoordMask)
{
    using namespace openvdb;
    CoordMask  mask;

    mask.setCoord(-4, 2, 18);
    EXPECT_TRUE(mask.allEven());

    mask.setCoord(1, 2, -6);
    EXPECT_TRUE(mask.xOdd());

    mask.setCoord(4, -3, -6);
    EXPECT_TRUE(mask.yOdd());

    mask.setCoord(-8, 2, -7);
    EXPECT_TRUE(mask.zOdd());

    mask.setCoord(1, -3, 2);
    EXPECT_TRUE(mask.xyOdd());

    mask.setCoord(1, 2, -7);
    EXPECT_TRUE(mask.xzOdd());

    mask.setCoord(-10, 3, -3);
    EXPECT_TRUE(mask.yzOdd());

    mask.setCoord(1, 3,-3);
    EXPECT_TRUE(mask.allOdd());
}

TEST_F(TestMultiResGrid, testManualTopology)
{
    // Perform tests when the sparsity (or topology) of the multiple grids is defined manually
    using namespace openvdb;

    typedef tools::MultiResGrid<DoubleTree> MultiResGridT;
    const double background = -1.0;
    const size_t levels = 4;

    MultiResGridT::Ptr mrg(new MultiResGridT( levels, background));

    EXPECT_TRUE(mrg != nullptr);
    EXPECT_EQ(levels  , mrg->numLevels());
    EXPECT_EQ(size_t(0), mrg->finestLevel());
    EXPECT_EQ(levels-1, mrg->coarsestLevel());

    // Define grid domain so they exactly overlap!
    const int w = 16;//half-width of dense patch on the finest grid level
    const CoordBBox bbox( Coord(-w), Coord(w) );// both inclusive

    // First check all trees against the background value
    for (size_t level = 0; level < mrg->numLevels(); ++level) {
        for (CoordBBox::Iterator<true> iter(bbox>>level); iter; ++iter) {
            EXPECT_NEAR(background,
                mrg->tree(level).getValue(*iter), /*tolerance=*/0.0);
        }
    }

    // Fill all trees according to a power of two refinement pattern
    for (size_t level = 0; level < mrg->numLevels(); ++level) {
        mrg->tree(level).fill( bbox>>level, double(level));
        mrg->tree(level).voxelizeActiveTiles();// avoid active tiles
        // Check values
        for (CoordBBox::Iterator<true> iter(bbox>>level); iter; ++iter) {
            EXPECT_NEAR(double(level),
                mrg->tree(level).getValue(*iter), /*tolerance=*/0.0);
        }
        //mrg->tree( level ).print(std::cerr, 2);
        // Check bounding box of active voxels
        CoordBBox bbox_actual;// Expected Tree dimensions: 33^3 -> 17^3 -> 9^3 ->5^3
        mrg->tree( level ).evalActiveVoxelBoundingBox( bbox_actual );
        EXPECT_EQ( bbox >> level, bbox_actual );
    }

    //pick a grid point that is shared between all the grids
    const Coord ijk(0);

    // Value at ijk equals the level
    EXPECT_NEAR(2.0, mrg->tree(2).getValue(ijk>>2), /*tolerance=*/ 0.001);
    EXPECT_NEAR(2.0, mrg->sampleValue<0>(ijk, size_t(2)), /*tolerance=*/ 0.001);
    EXPECT_NEAR(2.0, mrg->sampleValue<1>(ijk, size_t(2)), /*tolerance=*/ 0.001);
    EXPECT_NEAR(2.0, mrg->sampleValue<2>(ijk, size_t(2)), /*tolerance=*/ 0.001);
    EXPECT_NEAR(2.0, mrg->sampleValue<1>(ijk, 2.0f), /*tolerance=*/ 0.001);
    EXPECT_NEAR(2.0, mrg->sampleValue<1>(ijk, float(2)), /*tolerance=*/ 0.001);

    // Value at ijk at floating point level
    EXPECT_NEAR(2.25, mrg->sampleValue<1>(ijk, 2.25f), /*tolerance=*/ 0.001);

    // Value at a floating-point position close to ijk and a floating point level
    EXPECT_NEAR(2.25,
        mrg->sampleValue<1>(Vec3R(0.124), 2.25f), /*tolerance=*/ 0.001);

    // prolongate at a given point at top level
    EXPECT_NEAR(1.0, mrg->prolongateVoxel(ijk, 0), /*tolerance=*/ 0.0);

    // First check the coarsest level (3)
    for (CoordBBox::Iterator<true> iter(bbox>>size_t(3)); iter; ++iter) {
        EXPECT_NEAR(3.0, mrg->tree(3).getValue(*iter), /*tolerance=*/0.0);
    }

    // Prolongate from level 3 -> level 2 and check values
    mrg->prolongateActiveVoxels(2);
    for (CoordBBox::Iterator<true> iter(bbox>>size_t(2)); iter; ++iter) {
        EXPECT_NEAR(3.0, mrg->tree(2).getValue(*iter), /*tolerance=*/0.0);
    }

    // Prolongate from level 2 -> level 1 and check values
    mrg->prolongateActiveVoxels(1);
    for (CoordBBox::Iterator<true> iter(bbox>>size_t(1)); iter; ++iter) {
        EXPECT_NEAR(3.0, mrg->tree(1).getValue(*iter), /*tolerance=*/0.0);
    }

    // Prolongate from level 1 -> level 0 and check values
    mrg->prolongateActiveVoxels(0);
    for (CoordBBox::Iterator<true> iter(bbox); iter; ++iter) {
        EXPECT_NEAR(3.0, mrg->tree(0).getValue(*iter), /*tolerance=*/0.0);
    }

    // Redefine values at the finest level and check values
    mrg->finestTree().fill( bbox, 5.0 );
    mrg->finestTree().voxelizeActiveTiles();// avoid active tiles
    for (CoordBBox::Iterator<true> iter(bbox); iter; ++iter) {
        EXPECT_NEAR(5.0, mrg->tree(0).getValue(*iter), /*tolerance=*/0.0);
    }

    // USE RESTRICTION BY INJECTION since it doesn't have boundary issues
    // // Restrict from level 0 -> level 1 and check values
    // mrg->restrictActiveVoxels(1);
    // for (CoordBBox::Iterator<true> iter((bbox>>1UL).expandBy(-1)); iter; ++iter) {
    //     EXPECT_NEAR(5.0, mrg->tree(1).getValue(*iter), /*tolerance=*/0.0);
    // }

    // // Restrict from level 1 -> level 2 and check values
    // mrg->restrictActiveVoxels(2);
    // for (CoordBBox::Iterator<true> iter(bbox>>2UL); iter; ++iter) {
    //     EXPECT_NEAR(5.0, mrg->tree(2).getValue(*iter), /*tolerance=*/0.0);
    // }

    // // Restrict from level 2 -> level 3 and check values
    // mrg->restrictActiveVoxels(3);
    // for (CoordBBox::Iterator<true> iter(bbox>>3UL); iter; ++iter) {
    //     EXPECT_NEAR(5.0, mrg->tree(3).getValue(*iter), /*tolerance=*/0.0);
    // }
}

TEST_F(TestMultiResGrid, testIO)
{
    using namespace openvdb;

    const float radius = 1.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 0.01f;

    openvdb::FloatGrid::Ptr ls =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize);
    ls->setName("LevelSetSphere");

    typedef tools::MultiResGrid<FloatTree> MultiResGridT;
    const size_t levels = 4;

    // Generate LOD sequence
    MultiResGridT mrg( levels, *ls, /* reduction by injection */ false );
    //mrg.print( std::cout, 3 );

    EXPECT_EQ(levels  , mrg.numLevels());
    EXPECT_EQ(size_t(0), mrg.finestLevel());
    EXPECT_EQ(levels-1, mrg.coarsestLevel());

    // Check inside and outside values
    for ( size_t level = 1; level < mrg.numLevels(); ++level) {
        const float inside = mrg.sampleValue<1>( Coord(0,0,0), 0UL, level );
        EXPECT_NEAR( -ls->background(), inside,/*tolerance=*/ 0.001 );
        const float outside = mrg.sampleValue<1>( Coord( int(1.1*radius/voxelSize) ), 0UL, level );
        EXPECT_NEAR(  ls->background(), outside,/*tolerance=*/ 0.001 );
    }

    const std::string filename( "sphere.vdb" );

    // Write grids
    io::File outputFile( filename );
    outputFile.write( *mrg.grids() );
    outputFile.close();

    // Read grids
    openvdb::initialize();
    openvdb::io::File file( filename );
    file.open();
    GridPtrVecPtr grids = file.getGrids();
    EXPECT_EQ( levels, grids->size() );
    //std::cerr << "\nsize = " << grids->size() << std::endl;
    for ( size_t i=0; i<grids->size(); ++i ) {
        FloatGrid::Ptr grid = gridPtrCast<FloatGrid>(grids->at(i));
        EXPECT_EQ( grid->activeVoxelCount(), mrg.tree(i).activeVoxelCount() );
        //grid->print(std::cerr, 3);
    }
    file.close();

    ::remove(filename.c_str());
}

TEST_F(TestMultiResGrid, testModels)
{
    using namespace openvdb;

#ifdef TestMultiResGrid_DATA_PATH
    initialize();//required whenever I/O of OpenVDB files is performed!
    const std::string path(TestMultiResGrid_DATA_PATH);
    std::vector<std::string> filenames;
    filenames.push_back("armadillo.vdb");
    filenames.push_back("buddha.vdb");
    filenames.push_back("bunny.vdb");
    filenames.push_back("crawler.vdb");
    filenames.push_back("dragon.vdb");
    filenames.push_back("iss.vdb");
    filenames.push_back("utahteapot.vdb");
    util::CpuTimer timer;
    for ( size_t i=0; i<filenames.size(); ++i) {
        std::cerr << "\n=====================>\"" << filenames[i]
            << "\" =====================" << std::endl;
        std::cerr << "Reading \"" << filenames[i] << "\" ...";
        io::File file( path + filenames[i] );
        file.open(false);//disable delayed loading
        FloatGrid::Ptr model = gridPtrCast<FloatGrid>(file.getGrids()->at(0));
        std::cerr << " done\nProcessing \"" << filenames[i] << "\" ...";
        timer.start("\nMultiResGrid processing");
        tools::MultiResGrid<FloatTree> mrg( 6, model );
        timer.stop();
        std::cerr << "\n High-res level set " <<  tools::checkLevelSet(*mrg.grid(0)) << "\n";
        std::cerr << " done\nWriting \"" << filenames[i] << "\" ...";

        io::File file( "/tmp/" + filenames[i] );
        file.write( *mrg.grids() );
        file.close();

        std::cerr << " done\n" << std::endl;
        // {// in-betweening
        //     timer.start("\nIn-betweening");
        //     FloatGrid::Ptr model3 = mrg.createGrid( 1.9999f );
        //     timer.stop();
        //     //
        //     std::cerr << "\n" <<  tools::checkLevelSet(*model3) << "\n";
        //     //
        //     GridPtrVecPtr grids2( new GridPtrVec );
        //     grids2->push_back( model3 );
        //     io::File file2( "/tmp/inbetween_" + filenames[i] );
        //     file2.write( *grids2 );
        //     file2.close();
        // }
        // {// prolongate
        //     timer.start("\nProlongate");
        //     mrg.prolongateActiveVoxels(1);
        //     FloatGrid::Ptr model31= mrg.grid(1);
        //     timer.stop();
        //     GridPtrVecPtr grids2( new GridPtrVec );
        //     grids2->push_back( model31 );
        //     io::File file2( "/tmp/prolongate_" + filenames[i] );
        //     file2.write( *grids2 );
        //     file2.close();
        // }
        //::remove(filenames[i].c_str());
    }
#endif
}
