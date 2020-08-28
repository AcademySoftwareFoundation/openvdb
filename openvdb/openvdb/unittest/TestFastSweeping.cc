// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file    TestFastSweeping.cc
///
/// @author  Ken Museth

//#define BENCHMARK_FAST_SWEEPING
//#define TIMING_FAST_SWEEPING

#include <sstream>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Diagnostics.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetUtil.h>
#ifdef TIMING_FAST_SWEEPING
#include <openvdb/util/CpuTimer.h>
#endif

// Uncomment to test on models from our web-site
//#define TestFastSweeping_DATA_PATH "/Users/ken/dev/data/vdb/"
//#define TestFastSweeping_DATA_PATH "/home/kmu/dev/data/vdb/"
//#define TestFastSweeping_DATA_PATH "/usr/pic1/Data/OpenVDB/LevelSetModels/"

class TestFastSweeping: public CppUnit::TestFixture
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestFastSweeping);
    CPPUNIT_TEST(dilateSignedDistance);
    CPPUNIT_TEST(testMaskSdf);
    CPPUNIT_TEST(testSdfToFogVolume);
    CPPUNIT_TEST(testIntersection);
    CPPUNIT_TEST(fogToSdfAndExt);
    CPPUNIT_TEST(sdfToSdfAndExt);
    CPPUNIT_TEST(sdfToSdfAndExt_velocity);
#ifdef TestFastSweeping_DATA_PATH
    CPPUNIT_TEST(velocityExtensionOfFogBunny);
    CPPUNIT_TEST(velocityExtensionOfSdfBunny);
#endif
#ifdef BENCHMARK_FAST_SWEEPING
    CPPUNIT_TEST(testBenchmarks);
#endif

    CPPUNIT_TEST_SUITE_END();

    void dilateSignedDistance();
    void testMaskSdf();
    void testSdfToFogVolume();
    void testIntersection();
    void fogToSdfAndExt();
    void sdfToSdfAndExt();
    void sdfToSdfAndExt_velocity();
#ifdef TestFastSweeping_DATA_PATH
    void velocityExtensionOfFogBunny();
    void velocityExtensionOfSdfBunny();
#endif
#ifdef BENCHMARK_FAST_SWEEPING
    void testBenchmarks();
#endif
    void writeFile(const std::string &name, openvdb::FloatGrid::Ptr grid)
    {
        openvdb::io::File file(name);
        file.setCompression(openvdb::io::COMPRESS_NONE);
        openvdb::GridPtrVec grids;
        grids.push_back(grid);
        file.write(grids);
    }
};// TestFastSweeping

CPPUNIT_TEST_SUITE_REGISTRATION(TestFastSweeping);

void
TestFastSweeping::dilateSignedDistance()
{
    using namespace openvdb;
    // Define parameters for the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f;//half width
    const int width = 3, new_width = 50;//half width

    FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, float(width));
    const size_t oldVoxelCount = grid->activeVoxelCount();

    tools::FastSweeping<FloatGrid> fs;
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.sweepingVoxelCount());
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.boundaryVoxelCount());
    fs.initDilate(*grid, new_width - width);
    CPPUNIT_ASSERT(fs.sweepingVoxelCount() > 0);
    CPPUNIT_ASSERT(fs.boundaryVoxelCount() > 0);
    fs.sweep();
    CPPUNIT_ASSERT(fs.sweepingVoxelCount() > 0);
    CPPUNIT_ASSERT(fs.boundaryVoxelCount() > 0);
    auto grid2 = fs.sdfGrid();
    fs.clear();
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.sweepingVoxelCount());
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.boundaryVoxelCount());
    const Index64 sweepingVoxelCount = grid2->activeVoxelCount();
    CPPUNIT_ASSERT(sweepingVoxelCount > oldVoxelCount);

    {// Check that the norm of the gradient for all active voxels is close to unity
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, 0.99f, 1.01f);
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        CPPUNIT_ASSERT(message.empty());
        CPPUNIT_ASSERT_EQUAL(Index64(0), diagnose.failureCount());
        //std::cout << "\nOutput 1: " << message << std::endl;
    }
    {// Make sure all active voxels fail the following test
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, std::numeric_limits<float>::min(), 0.99f);
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        CPPUNIT_ASSERT(!message.empty());
        CPPUNIT_ASSERT_EQUAL(sweepingVoxelCount, diagnose.failureCount());
        //std::cout << "\nOutput 2: " << message << std::endl;
    }
    {// Make sure all active voxels fail the following test
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, 1.01f, std::numeric_limits<float>::max());
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        CPPUNIT_ASSERT(!message.empty());
        CPPUNIT_ASSERT_EQUAL(sweepingVoxelCount, diagnose.failureCount());
        //std::cout << "\nOutput 3: " << message << std::endl;
    }
}// dilateSignedDistance


void
TestFastSweeping::testMaskSdf()
{
    using namespace openvdb;
    // Define parameterS FOR the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f, width = 3.0f;//half width
    const float new_width = 50;

    {// Use box as a mask
        //std::cerr << "\nUse box as a mask" << std::endl;
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        CoordBBox bbox(Coord(150,-50,-50), Coord(250,50,50));
        MaskGrid mask;
        mask.sparseFill(bbox, true);

        //this->writeFile("/tmp/box_mask_input.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping with a box mask");
#endif
        grid = tools::maskSdf(*grid, mask);
        //tools::FastSweeping<FloatGrid> fs;
        //fs.initMask(*grid, mask);
        //fs.sweep();
        //std::cerr << "voxel count = " << fs.sweepingVoxelCount() << std::endl;
        //std::cerr << "boundary count = " << fs.boundaryVoxelCount() << std::endl;
        //CPPUNIT_ASSERT(fs.sweepingVoxelCount() > 0);
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        //writeFile("/tmp/box_mask_output.vdb", grid);
        {// Check that the norm of the gradient for all active voxels is close to unity
            tools::Diagnose<FloatGrid> diagnose(*grid);
            tools::CheckNormGrad<FloatGrid> test(*grid, 0.99f, 1.01f);
            const std::string message = diagnose.check(test,
                                                       false,// don't generate a mask grid
                                                       true,// check active voxels
                                                       false,// ignore active tiles since a level set has none
                                                       false);// no need to check the background value
            //std::cerr << message << std::endl;
            const double percent = 100.0*double(diagnose.failureCount())/double(grid->activeVoxelCount());
            //std::cerr << "Failures = " << percent << "%" << std::endl;
            //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
            //std::cerr << "Total : " << grid->activeVoxelCount() << std::endl;
            CPPUNIT_ASSERT(percent < 0.01);
            //CPPUNIT_ASSERT(message.empty());
            //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
        }
    }

    {// Use sphere as a mask
        //std::cerr << "\nUse sphere as a mask" << std::endl;
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        FloatGrid::Ptr mask = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, new_width);

        //this->writeFile("/tmp/sphere_mask_input.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping with a sphere mask");
#endif
        grid = tools::maskSdf(*grid, *mask);
        //tools::FastSweeping<FloatGrid> fs;
        //fs.initMask(*grid, *mask);
        //fs.sweep();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        //std::cerr << "voxel count = " << fs.sweepingVoxelCount() << std::endl;
        //std::cerr << "boundary count = " << fs.boundaryVoxelCount() << std::endl;
        //CPPUNIT_ASSERT(fs.sweepingVoxelCount() > 0);
        //this->writeFile("/tmp/sphere_mask_output.vdb", grid);
        {// Check that the norm of the gradient for all active voxels is close to unity
            tools::Diagnose<FloatGrid> diagnose(*grid);
            tools::CheckNormGrad<FloatGrid> test(*grid, 0.99f, 1.01f);
            const std::string message = diagnose.check(test,
                                                       false,// don't generate a mask grid
                                                       true,// check active voxels
                                                       false,// ignore active tiles since a level set has none
                                                       false);// no need to check the background value
            //std::cerr << message << std::endl;
            const double percent = 100.0*double(diagnose.failureCount())/double(grid->activeVoxelCount());
            //std::cerr << "Failures = " << percent << "%" << std::endl;
            //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
            //std::cerr << "Total : " << grid->activeVoxelCount() << std::endl;
            //CPPUNIT_ASSERT(message.empty());
            //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
            CPPUNIT_ASSERT(percent < 0.01);
            //std::cout << "\nOutput 1: " << message << std::endl;
        }
    }

    {// Use dodecahedron as a mask
        //std::cerr << "\nUse dodecahedron as a mask" << std::endl;
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        FloatGrid::Ptr mask = tools::createLevelSetDodecahedron<FloatGrid>(50, Vec3f(radius, 0.0f, 0.0f),
                                                                           voxelSize, 10);

        //this->writeFile("/tmp/dodecahedron_mask_input.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping with a dodecahedron mask");
#endif
        grid = tools::maskSdf(*grid, *mask);
        //tools::FastSweeping<FloatGrid> fs;
        //fs.initMask(*grid, *mask);
        //std::cerr << "voxel count = " << fs.sweepingVoxelCount() << std::endl;
        //std::cerr << "boundary count = " << fs.boundaryVoxelCount() << std::endl;
        //CPPUNIT_ASSERT(fs.sweepingVoxelCount() > 0);
        //fs.sweep();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        //this->writeFile("/tmp/dodecahedron_mask_output.vdb", grid);
        {// Check that the norm of the gradient for all active voxels is close to unity
            tools::Diagnose<FloatGrid> diagnose(*grid);
            tools::CheckNormGrad<FloatGrid> test(*grid, 0.99f, 1.01f);
            const std::string message = diagnose.check(test,
                                                       false,// don't generate a mask grid
                                                       true,// check active voxels
                                                       false,// ignore active tiles since a level set has none
                                                       false);// no need to check the background value
            //std::cerr << message << std::endl;
            const double percent = 100.0*double(diagnose.failureCount())/double(grid->activeVoxelCount());
            //std::cerr << "Failures = " << percent << "%" << std::endl;
            //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
            //std::cerr << "Total : " << grid->activeVoxelCount() << std::endl;
            //CPPUNIT_ASSERT(message.empty());
            //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
            CPPUNIT_ASSERT(percent < 0.01);
            //std::cout << "\nOutput 1: " << message << std::endl;
        }
    }
#ifdef TestFastSweeping_DATA_PATH
     {// Use bunny as a mask
         //std::cerr << "\nUse bunny as a mask" << std::endl;
         FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(-10,0,0), 0.05f, width);
         openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
         const std::string path(TestFastSweeping_DATA_PATH);
         io::File file( path + "bunny.vdb" );
         file.open(false);//disable delayed loading
         FloatGrid::Ptr mask = openvdb::gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));

         //this->writeFile("/tmp/bunny_mask_input.vdb", grid);
         tools::FastSweeping<FloatGrid> fs;
#ifdef TIMING_FAST_SWEEPING
         util::CpuTimer timer("\nParallel sparse fast sweeping with a bunny mask");
#endif
         fs.initMask(*grid, *mask);
         //std::cerr << "voxel count = " << fs.sweepingVoxelCount() << std::endl;
         //std::cerr << "boundary count = " << fs.boundaryVoxelCount() << std::endl;
         fs.sweep();
         auto grid2 = fs.sdfGrid();
#ifdef TIMING_FAST_SWEEPING
         timer.stop();
#endif
         //this->writeFile("/tmp/bunny_mask_output.vdb", grid2);
         {// Check that the norm of the gradient for all active voxels is close to unity
             tools::Diagnose<FloatGrid> diagnose(*grid2);
             tools::CheckNormGrad<FloatGrid> test(*grid2, 0.99f, 1.01f);
             const std::string message = diagnose.check(test,
                                                        false,// don't generate a mask grid
                                                        true,// check active voxels
                                                        false,// ignore active tiles since a level set has none
                                                        false);// no need to check the background value
             //std::cerr << message << std::endl;
             const double percent = 100.0*double(diagnose.failureCount())/double(grid2->activeVoxelCount());
             //std::cerr << "Failures = " << percent << "%" << std::endl;
             //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
             //std::cerr << "Total : " << grid2->activeVoxelCount() << std::endl;
             //CPPUNIT_ASSERT(message.empty());
             //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
             CPPUNIT_ASSERT(percent < 4.5);// crossing characteristics!
             //std::cout << "\nOutput 1: " << message << std::endl;
         }
     }
#endif
}// testMaskSdf

void
TestFastSweeping::testSdfToFogVolume()
{
    using namespace openvdb;
    // Define parameterS FOR the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f, width = 3.0f;//half width

    FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, float(width));
    tools::sdfToFogVolume(*grid);
    const Index64 sweepingVoxelCount = grid->activeVoxelCount();

    //this->writeFile("/tmp/fog_input.vdb", grid);
    tools::FastSweeping<FloatGrid> fs;
#ifdef TIMING_FAST_SWEEPING
    util::CpuTimer timer("\nParallel sparse fast sweeping with a fog volume");
#endif
    fs.initSdf(*grid, /*isoValue*/0.5f,/*isInputSdf*/false);
    CPPUNIT_ASSERT(fs.sweepingVoxelCount() > 0);
    //std::cerr << "voxel count = " << fs.sweepingVoxelCount() << std::endl;
    //std::cerr << "boundary count = " << fs.boundaryVoxelCount() << std::endl;
    fs.sweep();
    auto grid2 = fs.sdfGrid();
#ifdef TIMING_FAST_SWEEPING
    timer.stop();
#endif
    CPPUNIT_ASSERT_EQUAL(sweepingVoxelCount, grid->activeVoxelCount());
    //this->writeFile("/tmp/ls_output.vdb", grid2);

    {// Check that the norm of the gradient for all active voxels is close to unity
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, 0.99f, 1.01f);
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        //std::cerr << message << std::endl;
        const double percent = 100.0*double(diagnose.failureCount())/double(grid2->activeVoxelCount());
        //std::cerr << "Failures = " << percent << "%" << std::endl;
        //std::cerr << "Failure count = " << diagnose.failureCount() << std::endl;
        //std::cerr << "Total active voxel count = " << grid2->activeVoxelCount() << std::endl;
        CPPUNIT_ASSERT(percent < 3.0);
    }
}// testSdfToFogVolume


#ifdef BENCHMARK_FAST_SWEEPING
void
TestFastSweeping::testBenchmarks()
{
    using namespace openvdb;
    // Define parameterS FOR the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f, width = 3.0f;//half width
    const float new_width = 50;

    {// Use rebuildLevelSet (limited to closed and symmetric narrow-band level sets)
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nRebuild level set");
#endif
        FloatGrid::Ptr ls = tools::levelSetRebuild(*grid, 0.0f, new_width);
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        std::cout << "Diagnostics:\n" << tools::checkLevelSet(*ls, 9) << std::endl;
        //this->writeFile("/tmp/rebuild_sdf.vdb", ls);
    }
    {// Use LevelSetTracker::normalize()
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        tools::dilateActiveValues(grid->tree(), int(new_width-width), tools::NN_FACE, tools::IGNORE_TILES);
        tools::changeLevelSetBackground(grid->tree(), new_width);
        std::cout << "Diagnostics:\n" << tools::checkLevelSet(*grid, 9) << std::endl;
        //std::cerr << "Number of active tiles = " << grid->tree().activeTileCount() << std::endl;
        //grid->print(std::cout, 3);
        tools::LevelSetTracker<FloatGrid> track(*grid);
        track.setNormCount(int(new_width/0.3f));//CFL is 1/3 for RK1
        track.setSpatialScheme(math::FIRST_BIAS);
        track.setTemporalScheme(math::TVD_RK1);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nConventional re-normalization");
#endif
        track.normalize();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        std::cout << "Diagnostics:\n" << tools::checkLevelSet(*grid, 9) << std::endl;
        //this->writeFile("/tmp/old_sdf.vdb", grid);
    }
    {// Use new sparse and parallel fast sweeping
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);

        //this->writeFile("/tmp/original_sdf.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping");
#endif
        auto grid2 = tools::dilateSdf(*grid, int(new_width - width), tools::NN_FACE_EDGE);
        //tools::FastSweeping<FloatGrid> fs(*grid);
        //CPPUNIT_ASSERT(fs.sweepingVoxelCount() > 0);
        //tbb::task_scheduler_init init(4);//thread count
        //fs.sweep();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        //std::cout << "Diagnostics:\n" << tools::checkLevelSet(*grid, 9) << std::endl;
        //this->writeFile("/tmp/new_sdf.vdb", grid2);
    }
}
#endif

void
TestFastSweeping::testIntersection()
{
  using namespace openvdb;
  const Coord ijk(1,4,-9);
  FloatGrid grid(0.0f);
  auto acc = grid.getAccessor();
  math::GradStencil<FloatGrid> stencil(grid);
  acc.setValue(ijk,-1.0f);
  int cases = 0;
  for (int mx=0; mx<2; ++mx) {
    acc.setValue(ijk.offsetBy(-1,0,0), mx ? 1.0f : -1.0f);
    for (int px=0; px<2; ++px) {
      acc.setValue(ijk.offsetBy(1,0,0), px ? 1.0f : -1.0f);
      for (int my=0; my<2; ++my) {
        acc.setValue(ijk.offsetBy(0,-1,0), my ? 1.0f : -1.0f);
        for (int py=0; py<2; ++py) {
          acc.setValue(ijk.offsetBy(0,1,0), py ? 1.0f : -1.0f);
          for (int mz=0; mz<2; ++mz) {
            acc.setValue(ijk.offsetBy(0,0,-1), mz ? 1.0f : -1.0f);
            for (int pz=0; pz<2; ++pz) {
              acc.setValue(ijk.offsetBy(0,0,1), pz ? 1.0f : -1.0f);
              ++cases;
              CPPUNIT_ASSERT_EQUAL(Index64(7), grid.activeVoxelCount());
              stencil.moveTo(ijk);
              const size_t count = mx + px + my + py + mz + pz;// number of intersections
              CPPUNIT_ASSERT(stencil.intersects() == (count > 0));
              auto mask = stencil.intersectionMask();
              CPPUNIT_ASSERT(mask.none() == (count == 0));
              CPPUNIT_ASSERT(mask.any() == (count > 0));
              CPPUNIT_ASSERT_EQUAL(count, mask.count());
              CPPUNIT_ASSERT(mask.test(0) == mx);
              CPPUNIT_ASSERT(mask.test(1) == px);
              CPPUNIT_ASSERT(mask.test(2) == my);
              CPPUNIT_ASSERT(mask.test(3) == py);
              CPPUNIT_ASSERT(mask.test(4) == mz);
              CPPUNIT_ASSERT(mask.test(5) == pz);
            }//pz
          }//mz
        }//py
      }//my
    }//px
  }//mx
  CPPUNIT_ASSERT_EQUAL(64, cases);// = 2^6
}//testIntersection

void
TestFastSweeping::fogToSdfAndExt()
{
  using namespace openvdb;
  const float isoValue = 0.5f;
  const float radius = 100.0f;
  const float background = 0.0f;
  const float tolerance = 0.00001f;
  const Vec3f center(0.0f, 0.0f, 0.0f);
  const float voxelSize = 1.0f, width = 3.0f;//half width
  FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, float(width));
  tools::sdfToFogVolume(*grid);
  CPPUNIT_ASSERT(grid);
  const float fog[] = {grid->tree().getValue( Coord(102, 0, 0) ),
                       grid->tree().getValue( Coord(101, 0, 0) ),
                       grid->tree().getValue( Coord(100, 0, 0) ),
                       grid->tree().getValue( Coord( 99, 0, 0) ),
                       grid->tree().getValue( Coord( 98, 0, 0) )};
  //for (auto v : fog) std::cerr << v << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(fog[0], 0.0f, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual(fog[1], 0.0f, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual(fog[2], 0.0f, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual(fog[3], 1.0f/3.0f, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual(fog[4], 2.0f/3.0f, tolerance) );
  //this->writeFile("/tmp/sphere1_fog_in.vdb", grid);

  auto op = [radius](const Vec3R &xyz) {return math::Sin(2*3.14*(xyz[0]+xyz[1]+xyz[2])/radius);};
  auto grids = tools::fogToSdfAndExt(*grid, op, background, isoValue);

  const auto sdf1 = grids.first->tree().getValue( Coord(100, 0, 0) );
  const auto sdf2 = grids.first->tree().getValue( Coord( 99, 0, 0) );
  const auto sdf3 = grids.first->tree().getValue( Coord( 98, 0, 0) );
  //std::cerr << "\nsdf1 = " << sdf1 << ", sdf2 = " << sdf2 << ", sdf3 = " << sdf3 << std::endl;
  CPPUNIT_ASSERT( sdf1 > sdf2 );
  CPPUNIT_ASSERT( math::isApproxEqual( sdf2, 0.5f, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual( sdf3,-0.5f, tolerance) );

  const auto ext1 = grids.second->tree().getValue( Coord(100, 0, 0) );
  const auto ext2 = grids.second->tree().getValue( Coord( 99, 0, 0) );
  const auto ext3 = grids.second->tree().getValue( Coord( 98, 0, 0) );
  //std::cerr << "\next1 = " << ext1 << ", ext2 = " << ext2 << ", ext3 = " << ext3 << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(ext1, background, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual(ext2, ext3, tolerance) );
  //this->writeFile("/tmp/sphere1_sdf_out.vdb", grids.first);
  //this->writeFile("/tmp/sphere1_ext_out.vdb", grids.second);
}// fogToSdfAndExt

void
TestFastSweeping::sdfToSdfAndExt()
{
  using namespace openvdb;
  const float isoValue = 0.0f;
  const float radius = 100.0f;
  const float background = 1.234f;
  const float tolerance = 0.00001f;
  const Vec3f center(0.0f, 0.0f, 0.0f);
  const float voxelSize = 1.0f, width = 3.0f;//half width
  FloatGrid::Ptr lsGrid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
  //std::cerr << "\nls(100,0,0) = " << lsGrid->tree().getValue( Coord(100, 0, 0) ) << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(lsGrid->tree().getValue( Coord(100, 0, 0) ), 0.0f, tolerance) );

  auto op = [radius](const Vec3R &xyz) {return math::Sin(2*3.14*xyz[0]/radius);};
  auto grids = tools::sdfToSdfAndExt(*lsGrid, op, background, isoValue);
  CPPUNIT_ASSERT(grids.first);
  CPPUNIT_ASSERT(grids.second);

  //std::cerr << "\nsdf = " << grids.first->tree().getValue( Coord(100, 0, 0) ) << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(grids.first->tree().getValue( Coord(100, 0, 0) ), 0.0f, tolerance) );

  //std::cerr << "\nBackground = " << grids.second->background() << std::endl;
  //std::cerr << "\nBackground = " << grids.second->tree().getValue( Coord(10000) ) << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(grids.second->background(), background, tolerance) );

  const auto sdf1 = grids.first->tree().getValue( Coord(100, 0, 0) );
  const auto sdf2 = grids.first->tree().getValue( Coord(102, 0, 0) );
  const auto sdf3 = grids.first->tree().getValue( Coord(102, 1, 1) );
  //std::cerr << "\nsdf1 = " << sdf1 << ", sdf2 = " << sdf2 << ", sdf3 = " << sdf3 << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual( sdf1, 0.0f, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual( sdf2, 2.0f, tolerance) );
  CPPUNIT_ASSERT( sdf3 > 2.0f );

  const auto ext1 = grids.second->tree().getValue( Coord(100, 0, 0) );
  const auto ext2 = grids.second->tree().getValue( Coord(102, 0, 0) );
  const auto ext3 = grids.second->tree().getValue( Coord(102, 1, 0) );
  //std::cerr << "\next1 = " << ext1 << ", ext2 = " << ext2 << ", ext3 = " << ext3 << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(float(op(Vec3R(100, 0, 0))), ext1, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual(ext1, ext2, tolerance) );
  CPPUNIT_ASSERT(!math::isApproxEqual(ext1, ext3, tolerance) );
  //writeFile("/tmp/sphere2_sdf_out.vdb", grids.first);
  //writeFile("/tmp/sphere2_ext_out.vdb", grids.second);
}// sdfToSdfAndExt

void
TestFastSweeping::sdfToSdfAndExt_velocity()
{
  using namespace openvdb;
  const float isoValue = 0.0f;
  const float radius = 100.0f;
  const Vec3f background(-1.0f, 2.0f, 1.234f);
  const float tolerance = 0.00001f;
  const Vec3f center(0.0f, 0.0f, 0.0f);
  const float voxelSize = 1.0f, width = 3.0f;//half width
  FloatGrid::Ptr lsGrid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
  //std::cerr << "\nls(100,0,0) = " << lsGrid->tree().getValue( Coord(100, 0, 0) ) << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(lsGrid->tree().getValue( Coord(100, 0, 0) ), 0.0f, tolerance) );
  //tools::sdfToFogVolume(*grid);
  //writeFile("/tmp/sphere1_fog_in.vdb", grid);
  //tools::fogToSdf(*grid, isoValue);

  // Vector valued extension field, e.g. a velocity field
  auto op = [radius](const Vec3R &xyz) {
    return Vec3f(float(xyz[0]), float(-xyz[1]), float(math::Sin(2*3.14*xyz[2]/radius)));
  };
  auto grids = tools::sdfToSdfAndExt(*lsGrid, op, background, isoValue);
  CPPUNIT_ASSERT(grids.first);
  CPPUNIT_ASSERT(grids.second);

  //std::cerr << "\nBackground = " << grids.second->background() << std::endl;
  //std::cerr << "\nBackground = " << grids.second->tree().getValue( Coord(10000) ) << std::endl;
  CPPUNIT_ASSERT( math::isApproxZero((grids.second->background()-background).length(), tolerance) );
  //std::cerr << "\nsdf = " << grids.first->tree().getValue( Coord(100, 0, 0) ) << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual(grids.first->tree().getValue( Coord(100, 0, 0) ), 0.0f, tolerance) );

  const auto sdf1 = grids.first->tree().getValue( Coord(100, 0, 0) );
  const auto sdf2 = grids.first->tree().getValue( Coord(102, 0, 0) );
  const auto sdf3 = grids.first->tree().getValue( Coord(102, 1, 1) );
  //std::cerr << "\nsdf1 = " << sdf1 << ", sdf2 = " << sdf2 << ", sdf3 = " << sdf3 << std::endl;
  CPPUNIT_ASSERT( math::isApproxEqual( sdf1, 0.0f, tolerance) );
  CPPUNIT_ASSERT( math::isApproxEqual( sdf2, 2.0f, tolerance) );
  CPPUNIT_ASSERT( sdf3 > 2.0f );

  const auto ext1 = grids.second->tree().getValue( Coord(100, 0, 0) );
  const auto ext2 = grids.second->tree().getValue( Coord(102, 0, 0) );
  const auto ext3 = grids.second->tree().getValue( Coord(102, 1, 0) );
  //std::cerr << "\next1 = " << ext1 << ", ext2 = " << ext2 << ", ext3 = " << ext3 << std::endl;
  CPPUNIT_ASSERT( math::isApproxZero((op(Vec3R(100, 0, 0)) - ext1).length(), tolerance) );
  CPPUNIT_ASSERT( math::isApproxZero((ext1 - ext2).length(), tolerance) );
  CPPUNIT_ASSERT(!math::isApproxZero((ext1 - ext3).length(), tolerance) );

  //writeFile("/tmp/sphere2_sdf_out.vdb", grids.first);
  //writeFile("/tmp/sphere2_ext_out.vdb", grids.second);
}// sdfToSdfAndExt_velocity

#ifdef TestFastSweeping_DATA_PATH
void
TestFastSweeping::velocityExtensionOfFogBunny()
{
  using namespace openvdb;

  openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
  const std::string path(TestFastSweeping_DATA_PATH);
  io::File file( path + "bunny.vdb" );
  file.open(false);//disable delayed loading
  auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));
  tools::sdfToFogVolume(*grid);
  writeFile("/tmp/bunny1_fog_in.vdb", grid);
  auto bbox = grid->evalActiveVoxelBoundingBox();
  const double xSize = bbox.dim()[0]*grid->voxelSize()[0];
  std::cerr << "\ndim=" << bbox.dim() << ", voxelSize="<< grid->voxelSize()[0]
            << ", xSize=" << xSize << std::endl;

  auto op = [xSize](const Vec3R &xyz) {
    return math::Sin(2*3.14*xyz[0]/xSize);
  };
  auto grids = tools::fogToSdfAndExt(*grid, op, 0.0f, 0.5f);
  std::cerr << "before writing" << std::endl;
  writeFile("/tmp/bunny1_sdf_out.vdb", grids.first);
  writeFile("/tmp/bunny1_ext_out.vdb", grids.second);
  std::cerr << "after writing" << std::endl;
}//velocityExtensionOfFogBunnyevalActiveVoxelBoundingBox

void
TestFastSweeping::velocityExtensionOfSdfBunny()
{
  using namespace openvdb;
  const std::string path(TestFastSweeping_DATA_PATH);
  io::File file( path + "bunny.vdb" );
  file.open(false);//disable delayed loading
  auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));
  writeFile("/tmp/bunny2_sdf_in.vdb", grid);
  auto bbox = grid->evalActiveVoxelBoundingBox();
  const double xSize = bbox.dim()[0]*grid->voxelSize()[0];
  std::cerr << "\ndim=" << bbox.dim() << ", voxelSize="<< grid->voxelSize()[0]
            << ", xSize=" << xSize << std::endl;

  auto op = [xSize](const Vec3R &xyz) {
    return math::Sin(2*3.14*xyz[0]/xSize);
  };
  auto grids = tools::sdfToSdfAndExt(*grid, op, 0.0f);
  std::cerr << "before writing" << std::endl;
  writeFile("/tmp/bunny2_sdf_out.vdb", grids.first);
  writeFile("/tmp/bunny2_ext_out.vdb", grids.second);
  std::cerr << "after writing" << std::endl;

}//velocityExtensionOfFogBunnyevalActiveVoxelBoundingBox
#endif
