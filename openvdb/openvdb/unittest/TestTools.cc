// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Composite.h>        // for csunion()
#include <openvdb/tools/Diagnostics.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetMorph.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/Mask.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/PointAdvect.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/VectorTransformer.h>
#include <openvdb/tools/VolumeAdvect.h>
#include <openvdb/util/Util.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Stats.h>
#include "util.h" // for unittest_util::makeSphere()
#include <gtest/gtest.h>
#include <algorithm> // for std::sort
#include <atomic>
#include <random>
#include <sstream>


// Uncomment to test on models from our web-site
//#define TestTools_DATA_PATH "/home/kmu/src/openvdb/data/"
//#define TestTools_DATA_PATH "/usr/pic1/Data/OpenVDB/LevelSetModels/"

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/0.0);

class TestTools: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


#if 0
namespace {

// Simple helper class to write out numbered vdbs
template<typename GridT>
class FrameWriter
{
public:
    FrameWriter(int version, typename GridT::Ptr grid):
        mFrame(0), mVersion(version), mGrid(grid)
    {}

    void operator()(const std::string& name, float time, size_t n)
    {
        std::ostringstream ostr;
        ostr << name << "_" << mVersion << "_" << mFrame << ".vdb";
        openvdb::io::File file(ostr.str());
        openvdb::GridPtrVec grids;
        grids.push_back(mGrid);
        file.write(grids);
        std::cerr << "\nWrote \"" << ostr.str() << "\" with time = "
                  << time << " after CFL-iterations = " << n << std::endl;
        ++mFrame;
    }

private:
    int mFrame, mVersion;
    typename GridT::Ptr mGrid;
};

} // unnamed namespace
#endif


TEST_F(TestTools, testInteriorMask)
{
    using namespace openvdb;

    const CoordBBox
        extBand{Coord{-1}, Coord{100}},
        isoBand{Coord{0}, Coord{99}},
        intBand{Coord{1}, Coord{98}},
        inside{Coord{2}, Coord{97}};

    // Construct a "level set" with a three-voxel narrow band
    // (the distances aren't correct, but they have the right sign).
    FloatGrid lsgrid{/*background=*/2.f};
    lsgrid.fill(extBand, 1.f);
    lsgrid.fill(isoBand, 0.f);
    lsgrid.fill(intBand, -1.f);
    lsgrid.fill(inside, -2.f, /*active=*/false);

    // For a non-level-set grid, tools::interiorMask() should return
    // a mask of the active voxels.
    auto mask = tools::interiorMask(lsgrid);
    EXPECT_EQ(extBand.volume() - inside.volume(), mask->activeVoxelCount());

    // For a level set, tools::interiorMask() should return a mask
    // of the interior of the isosurface.
    lsgrid.setGridClass(GRID_LEVEL_SET);
    mask = tools::interiorMask(lsgrid);
    EXPECT_EQ(intBand.volume(), mask->activeVoxelCount());
}


TEST_F(TestTools, testLevelSetSphere)
{
    const float radius = 4.3f;
    const openvdb::Vec3f center(15.8f, 13.2f, 16.7f);
    const float voxelSize = 1.5f, width = 3.25f;
    const int dim = 32;

    openvdb::FloatGrid::Ptr grid1 =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);

    /// Also test ultra slow makeSphere in unittest/util.h
    openvdb::FloatGrid::Ptr grid2 = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, width);
    unittest_util::makeSphere<openvdb::FloatGrid>(
        openvdb::Coord(dim), center, radius, *grid2, unittest_util::SPHERE_SPARSE_NARROW_BAND);

    const float outside = grid1->background(), inside = -outside;
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            for (int k=0; k<dim; ++k) {
                const openvdb::Vec3f p(voxelSize*float(i), voxelSize*float(j), voxelSize*float(k));
                const float dist = (p-center).length() - radius;
                const float val1 = grid1->tree().getValue(openvdb::Coord(i,j,k));
                const float val2 = grid2->tree().getValue(openvdb::Coord(i,j,k));
                if (dist > outside) {
                    EXPECT_NEAR( outside, val1, 0.0001);
                    EXPECT_NEAR( outside, val2, 0.0001);
                } else if (dist < inside) {
                    EXPECT_NEAR( inside, val1, 0.0001);
                    EXPECT_NEAR( inside, val2, 0.0001);
                } else {
                    EXPECT_NEAR(  dist, val1, 0.0001);
                    EXPECT_NEAR(  dist, val2, 0.0001);
                }
            }
        }
    }

    EXPECT_EQ(grid1->activeVoxelCount(), grid2->activeVoxelCount());
}// testLevelSetSphere

TEST_F(TestTools, testLevelSetPlatonic)
{
    using namespace openvdb;

    const float scale = 0.5f;
    const Vec3f center(1.0f, 2.0f, 3.0f);
    const float voxelSize = 0.025f, width = 2.0f, background = width*voxelSize;
    const Coord ijk(int(center[0]/voxelSize),
                    int(center[1]/voxelSize),
                    int(center[2]/voxelSize));// inside

    // The tests below are not particularly good (a visual inspection
    // in Houdini is much better) but at least it exercises the code
    // and performs an elementary suite of tests.

    {// test tetrahedron
        FloatGrid::Ptr ls = tools::createLevelSetTetrahedron<FloatGrid>(scale, center,
                                                                        voxelSize, width);
        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(background, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
    }
    {// test cube
        FloatGrid::Ptr ls = tools::createLevelSetCube<FloatGrid>(scale, center,
                                                                 voxelSize, width);
        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(),ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(background, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
    }
    {// test octahedron
        FloatGrid::Ptr ls = tools::createLevelSetOctahedron<FloatGrid>(scale, center,
                                                                       voxelSize, width);
        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(),ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(background, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
    }
    {// test icosahedron
        FloatGrid::Ptr ls = tools::createLevelSetIcosahedron<FloatGrid>(scale, center,
                                                                        voxelSize, width);
        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(),ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(background, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
    }
    {// test dodecahedron
        FloatGrid::Ptr ls = tools::createLevelSetDodecahedron<FloatGrid>(scale, center,
                                                                         voxelSize, width);
        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(),ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(background, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
    }

}// testLevelSetPlatonic

TEST_F(TestTools, testLevelSetAdvect)
{
    // Uncomment sections below to run this (time-consuming) test
    using namespace openvdb;

    const int dim = 128;
    const Vec3f center(0.35f, 0.35f, 0.35f);
    const float radius = 0.15f, voxelSize = 1.0f/(dim-1);
    const float halfWidth = 3.0f, gamma = halfWidth*voxelSize;

    using GridT = FloatGrid;
    //using VectT = Vec3fGrid;

    {//test tracker::resize
        GridT::Ptr grid = tools::createLevelSetSphere<GridT>(radius, center, voxelSize, halfWidth);
        using TrackerT = tools::LevelSetTracker<GridT>;
        TrackerT tracker(*grid);
        tracker.setSpatialScheme(math::FIRST_BIAS);
        tracker.setTemporalScheme(math::TVD_RK1);

        ASSERT_DOUBLES_EXACTLY_EQUAL( gamma, grid->background());
        ASSERT_DOUBLES_EXACTLY_EQUAL( halfWidth, tracker.getHalfWidth());

        EXPECT_TRUE(!tracker.resize());

        {// check range of on values in a sphere w/o mask
            tools::CheckRange<GridT, true, true, GridT::ValueOnCIter> c(-gamma, gamma);
            tools::Diagnose<GridT> d(*grid);
            std::string str = d.check(c);
            //std::cerr << "Values out of range:\n" << str;
            EXPECT_TRUE(str.empty());
            EXPECT_EQ(0, int(d.valueCount()));
            EXPECT_EQ(0, int(d.failureCount()));
        }
        {// check norm of gradient of sphere w/o mask
            tools::CheckNormGrad<GridT> c(*grid, 0.9f, 1.1f);
            tools::Diagnose<GridT> d(*grid);
            std::string str = d.check(c, false, true, false, false);
            //std::cerr << "NormGrad:\n" << str;
            EXPECT_TRUE(str.empty());
            EXPECT_EQ(0, int(d.valueCount()));
            EXPECT_EQ(0, int(d.failureCount()));
        }

        EXPECT_TRUE(tracker.resize(4));

        ASSERT_DOUBLES_EXACTLY_EQUAL( 4*voxelSize, grid->background());
        ASSERT_DOUBLES_EXACTLY_EQUAL( 4.0f, tracker.getHalfWidth());

        {// check range of on values in a sphere w/o mask
            const float g = gamma + voxelSize;
            tools::CheckRange<GridT, true, true, GridT::ValueOnCIter> c(-g, g);
            tools::Diagnose<GridT> d(*grid);
            std::string str = d.check(c);
            //std::cerr << "Values out of range:\n" << str;
            EXPECT_TRUE(str.empty());
            EXPECT_EQ(0, int(d.valueCount()));
            EXPECT_EQ(0, int(d.failureCount()));
        }
        {// check norm of gradient of sphere w/o mask
            tools::CheckNormGrad<GridT> c(*grid, 0.4f, 1.1f);
            tools::Diagnose<GridT> d(*grid);
            std::string str = d.check(c, false, true, false, false);
            //std::cerr << "NormGrad:\n" << str;
            EXPECT_TRUE(str.empty());
            EXPECT_EQ(0, int(d.valueCount()));
            EXPECT_EQ(0, int(d.failureCount()));
        }
    }
    /*
    {//test tracker
        GridT::Ptr grid = openvdb::tools::createLevelSetSphere<GridT>(radius, center, voxelSize);
        using TrackerT = openvdb::tools::LevelSetTracker<GridT>;
        TrackerT tracker(*grid);
        tracker.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        tracker.setTemporalScheme(openvdb::math::TVD_RK1);

        FrameWriter<GridT> fw(dim, grid); fw("Tracker",0, 0);
        //for (float t = 0, dt = 0.005f; !grid->empty() && t < 3.0f; t += dt) {
        //    fw("Enright", t + dt, advect.advect(t, t + dt));
        //}
        for (float t = 0, dt = 0.5f; !grid->empty() && t < 1.0f; t += dt) {
            tracker.track();
            fw("Tracker", 0, 0);
        }
    */

    /*
    {//test EnrightField
        GridT::Ptr grid = openvdb::tools::createLevelSetSphere<GridT>(radius, center, voxelSize);
        using FieldT = openvdb::tools::EnrightField<float>;
        FieldT field;

        using AdvectT = openvdb::tools::LevelSetAdvection<GridT, FieldT>;
        AdvectT advect(*grid, field);
        advect.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTemporalScheme(openvdb::math::TVD_RK2);
        advect.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTrackerTemporalScheme(openvdb::math::TVD_RK1);

        FrameWriter<GridT> fw(dim, grid); fw("Enright",0, 0);
        //for (float t = 0, dt = 0.005f; !grid->empty() && t < 3.0f; t += dt) {
        //    fw("Enright", t + dt, advect.advect(t, t + dt));
        //}
        for (float t = 0, dt = 0.5f; !grid->empty() && t < 1.0f; t += dt) {
            fw("Enright", t + dt, advect.advect(t, t + dt));
        }
        }
    */
    /*
    {// test DiscreteGrid - Aligned
        GridT::Ptr grid = openvdb::tools::createLevelSetSphere<GridT>(radius, center, voxelSize);
        VectT vect(openvdb::Vec3f(1,0,0));
        using FieldT = openvdb::tools::DiscreteField<VectT>;
        FieldT field(vect);
        using AdvectT = openvdb::tools::LevelSetAdvection<GridT, FieldT>;
        AdvectT advect(*grid, field);
        advect.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTemporalScheme(openvdb::math::TVD_RK2);

        FrameWriter<GridT> fw(dim, grid); fw("Aligned",0, 0);
        //for (float t = 0, dt = 0.005f; !grid->empty() && t < 3.0f; t += dt) {
        //    fw("Aligned", t + dt, advect.advect(t, t + dt));
        //}
        for (float t = 0, dt = 0.5f; !grid->empty() && t < 1.0f; t += dt) {
            fw("Aligned", t + dt, advect.advect(t, t + dt));
        }
        }
    */
    /*
    {// test DiscreteGrid - Transformed
        GridT::Ptr grid = openvdb::tools::createLevelSetSphere<GridT>(radius, center, voxelSize);
        VectT vect(openvdb::Vec3f(0,0,0));
        VectT::Accessor acc = vect.getAccessor();
        for (openvdb::Coord ijk(0); ijk[0]<dim; ++ijk[0])
            for (ijk[1]=0; ijk[1]<dim; ++ijk[1])
                for (ijk[2]=0; ijk[2]<dim; ++ijk[2])
                    acc.setValue(ijk, openvdb::Vec3f(1,0,0));
        vect.transform().scale(2.0f);
        using FieldT = openvdb::tools::DiscreteField<VectT>;
        FieldT field(vect);
        using AdvectT = openvdb::tools::LevelSetAdvection<GridT, FieldT>;
        AdvectT advect(*grid, field);
        advect.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTemporalScheme(openvdb::math::TVD_RK2);

        FrameWriter<GridT> fw(dim, grid); fw("Xformed",0, 0);
        //for (float t = 0, dt = 0.005f; !grid->empty() && t < 3.0f; t += dt) {
        //    fw("Xformed", t + dt, advect.advect(t, t + dt));
        //}
        for (float t = 0, dt = 0.5f; !grid->empty() && t < 1.0f; t += dt) {
            fw("Xformed", t + dt, advect.advect(t, t + dt));
        }
        }
    */
}//testLevelSetAdvect


////////////////////////////////////////

TEST_F(TestTools, testLevelSetMorph)
{
    using GridT = openvdb::FloatGrid;
    {//test morphing overlapping but aligned spheres
        const int dim = 64;
        const openvdb::Vec3f C1(0.35f, 0.35f, 0.35f), C2(0.4f, 0.4f, 0.4f);
        const float radius = 0.15f, voxelSize = 1.0f/(dim-1);

        GridT::Ptr source = openvdb::tools::createLevelSetSphere<GridT>(radius, C1, voxelSize);
        GridT::Ptr target = openvdb::tools::createLevelSetSphere<GridT>(radius, C2, voxelSize);

        using MorphT = openvdb::tools::LevelSetMorphing<GridT>;
        MorphT morph(*source, *target);
        morph.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        morph.setTemporalScheme(openvdb::math::TVD_RK3);
        morph.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        morph.setTrackerTemporalScheme(openvdb::math::TVD_RK2);

        const std::string name("SphereToSphere");
        //FrameWriter<GridT> fw(dim, source);
        //fw(name, 0.0f, 0);
        //util::CpuTimer timer;
        const float tMax =  0.05f/voxelSize;
        //std::cerr << "\nt-max = " << tMax << std::endl;
        //timer.start("\nMorphing");
        for (float t = 0, dt = 0.1f; !source->empty() && t < tMax; t += dt) {
            morph.advect(t, t + dt);
            //fw(name, t + dt, morph.advect(t, t + dt));
        }
        // timer.stop();

        const float invDx = 1.0f/voxelSize;
        openvdb::math::Stats s;
        for (GridT::ValueOnCIter it = source->tree().cbeginValueOn(); it; ++it) {
            s.add( invDx*(*it - target->tree().getValue(it.getCoord())) );
        }
        for (GridT::ValueOnCIter it = target->tree().cbeginValueOn(); it; ++it) {
            s.add( invDx*(*it - target->tree().getValue(it.getCoord())) );
        }
        //s.print("Morph");
        EXPECT_NEAR(0.0, s.min(), 0.50);
        EXPECT_NEAR(0.0, s.max(), 0.50);
        EXPECT_NEAR(0.0, s.avg(), 0.02);
        /*
        openvdb::math::Histogram h(s, 30);
        for (GridT::ValueOnCIter it = source->tree().cbeginValueOn(); it; ++it) {
            h.add( invDx*(*it - target->tree().getValue(it.getCoord())) );
        }
        for (GridT::ValueOnCIter it = target->tree().cbeginValueOn(); it; ++it) {
            h.add( invDx*(*it - target->tree().getValue(it.getCoord())) );
        }
        h.print("Morph");
        */
    }
    /*
    // Uncomment sections below to run this (very time-consuming) test
    {//test morphing between the bunny and the buddha models loaded from files
        util::CpuTimer timer;
        openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
        openvdb::io::File sourceFile("/usr/pic1/Data/OpenVDB/LevelSetModels/bunny.vdb");
        sourceFile.open();
        GridT::Ptr source = openvdb::gridPtrCast<GridT>(sourceFile.getGrids()->at(0));

        openvdb::io::File targetFile("/usr/pic1/Data/OpenVDB/LevelSetModels/buddha.vdb");
        targetFile.open();
        GridT::Ptr target = openvdb::gridPtrCast<GridT>(targetFile.getGrids()->at(0));

        using MorphT = openvdb::tools::LevelSetMorphing<GridT>;
        MorphT morph(*source, *target);
        morph.setSpatialScheme(openvdb::math::FIRST_BIAS);
        //morph.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        morph.setTemporalScheme(openvdb::math::TVD_RK2);
        morph.setTrackerSpatialScheme(openvdb::math::FIRST_BIAS);
        //morph.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        morph.setTrackerTemporalScheme(openvdb::math::TVD_RK2);

        const std::string name("Bunny2Buddha");
        FrameWriter<GridT> fw(1, source);
        fw(name, 0.0f, 0);
        for (float t = 0, dt = 1.0f; !source->empty() && t < 300.0f; t += dt) {
        timer.start("Morphing");
        const int cflCount = morph.advect(t, t + dt);
        timer.stop();
        fw(name, t + dt, cflCount);
        }
    }
    */
    /*
    // Uncomment sections below to run this (very time-consuming) test
    {//test morphing between the dragon and the teapot models loaded from files
        util::CpuTimer timer;
        openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
        openvdb::io::File sourceFile("/usr/pic1/Data/OpenVDB/LevelSetModels/dragon.vdb");
        sourceFile.open();
        GridT::Ptr source = openvdb::gridPtrCast<GridT>(sourceFile.getGrids()->at(0));

        openvdb::io::File targetFile("/usr/pic1/Data/OpenVDB/LevelSetModels/utahteapot.vdb");
        targetFile.open();
        GridT::Ptr target = openvdb::gridPtrCast<GridT>(targetFile.getGrids()->at(0));

        using MorphT = openvdb::tools::LevelSetMorphing<GridT>;
        MorphT morph(*source, *target);
        morph.setSpatialScheme(openvdb::math::FIRST_BIAS);
        //morph.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        morph.setTemporalScheme(openvdb::math::TVD_RK2);
        //morph.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        morph.setTrackerSpatialScheme(openvdb::math::FIRST_BIAS);
        morph.setTrackerTemporalScheme(openvdb::math::TVD_RK2);

        const std::string name("Dragon2Teapot");
        FrameWriter<GridT> fw(5, source);
        fw(name, 0.0f, 0);
        for (float t = 0, dt = 0.4f; !source->empty() && t < 110.0f; t += dt) {
            timer.start("Morphing");
            const int cflCount = morph.advect(t, t + dt);
            timer.stop();
            fw(name, t + dt, cflCount);
        }
        }

    */
}//testLevelSetMorph

////////////////////////////////////////

TEST_F(TestTools, testLevelSetMeasure)
{
    const double percentage = 0.1/100.0;//i.e. 0.1%
    using GridT = openvdb::FloatGrid;
    const int dim = 256;
    openvdb::Real area, volume, mean, gauss;

    // First sphere
    openvdb::Vec3f C(0.35f, 0.35f, 0.35f);
    openvdb::Real r = 0.15, voxelSize = 1.0/(dim-1);
    const openvdb::Real Pi = openvdb::math::pi<openvdb::Real>();
    GridT::Ptr sphere = openvdb::tools::createLevelSetSphere<GridT>(float(r), C, float(voxelSize));

    using MeasureT = openvdb::tools::LevelSetMeasure<GridT>;
    MeasureT m(*sphere);

    /// Test area and volume of sphere in world units
    area = 4*Pi*r*r;
    volume = 4.0/3.0*Pi*r*r*r;
    //std::cerr << "\nArea of sphere = " << area << "  " << a << std::endl;
    //std::cerr << "\nVolume of sphere = " << volume << "  " << v << std::endl;
    // Test accuracy of computed measures to within 0.1% of the exact measure.
    EXPECT_NEAR(area,   m.area(), percentage*area);
    EXPECT_NEAR(volume, m.volume(), percentage*volume);

    // Test area, volume and average mean curvature of sphere in world units
    mean = 1.0/r;
    //std::cerr << "\nArea of sphere = " << area << "  " << a << std::endl;
    //std::cerr << "Volume of sphere = " << volume << "  " << v << std::endl;
    //std::cerr << "radius in world units = " << r << std::endl;
    //std::cerr << "Avg mean curvature of sphere = " << mean << "  " << cm << std::endl;
    // Test accuracy of computed measures to within 0.1% of the exact measure.
    EXPECT_NEAR(area,   m.area(), percentage*area);
    EXPECT_NEAR(volume, m.volume(), percentage*volume);
    EXPECT_NEAR(mean,   m.avgMeanCurvature(), percentage*mean);

    // Test area, volume, average mean curvature and average gaussian curvature of sphere in world units
    gauss = 1.0/(r*r);
    //std::cerr << "\nArea of sphere = " << area << "  " << a << std::endl;
    //std::cerr << "Volume of sphere = " << volume << "  " << v << std::endl;
    //std::cerr << "radius in world units = " << r << std::endl;
    //std::cerr << "Avg mean curvature of sphere = " << mean << "  " << cm << std::endl;
    //std::cerr << "Avg gaussian curvature of sphere = " << gauss << "  " << cg << std::endl;
    // Test accuracy of computed measures to within 0.1% of the exact measure.
    EXPECT_NEAR(area,  m.area(), percentage*area);
    EXPECT_NEAR(volume, m.volume(), percentage*volume);
    EXPECT_NEAR(mean,   m.avgMeanCurvature(), percentage*mean);
    EXPECT_NEAR(gauss, m.avgGaussianCurvature(), percentage*gauss);
    EXPECT_EQ(0, m.genus());

    // Test measures of sphere in voxel units
    r /= voxelSize;
    area = 4*Pi*r*r;
    volume = 4.0/3.0*Pi*r*r*r;
    mean = 1.0/r;
    //std::cerr << "\nArea of sphere = " << area << "  " << a << std::endl;
    //std::cerr << "Volume of sphere = " << volume << "  " << v << std::endl;
    //std::cerr << "Avg mean curvature of sphere = " << curv << "  " << cm << std::endl;
    // Test accuracy of computed measures to within 0.1% of the exact measure.
    EXPECT_NEAR(area,   m.area(false), percentage*area);
    EXPECT_NEAR(volume, m.volume(false), percentage*volume);
    EXPECT_NEAR(mean,   m.avgMeanCurvature(false), percentage*mean);

    gauss = 1.0/(r*r);
    //std::cerr << "\nArea of sphere = " << area << "  " << a << std::endl;
    //std::cerr << "Volume of sphere = " << volume << "  " << v << std::endl;
    //std::cerr << "radius in voxel units = " << r << std::endl;
    //std::cerr << "Avg mean curvature of sphere = " << mean << "  " << cm << std::endl;
    //std::cerr << "Avg gaussian curvature of sphere = " << gauss << "  " << cg << std::endl;
    // Test accuracy of computed measures to within 0.1% of the exact measure.
    EXPECT_NEAR(area,   m.area(false), percentage*area);
    EXPECT_NEAR(volume, m.volume(false), percentage*volume);
    EXPECT_NEAR(mean,   m.avgMeanCurvature(false), percentage*mean);
    EXPECT_NEAR(gauss,  m.avgGaussianCurvature(false), percentage*gauss);
    EXPECT_EQ(0, m.genus());

    // Second sphere
    C = openvdb::Vec3f(5.4f, 6.4f, 8.4f);
    r = 0.57;
    sphere = openvdb::tools::createLevelSetSphere<GridT>(float(r), C, float(voxelSize));
    m.init(*sphere);

    // Test all measures of sphere in world units
    area = 4*Pi*r*r;
    volume = 4.0/3.0*Pi*r*r*r;
    mean = 1.0/r;
    gauss = 1.0/(r*r);
    //std::cerr << "\nArea of sphere = " << area << "  " << a << std::endl;
    //std::cerr << "Volume of sphere = " << volume << "  " << v << std::endl;
    //std::cerr << "radius in world units = " << r << std::endl;
    //std::cerr << "Avg mean curvature of sphere = " << mean << "  " << cm << std::endl;
    //std::cerr << "Avg gaussian curvature of sphere = " << gauss << "  " << cg << std::endl;
    // Test accuracy of computed measures to within 0.1% of the exact measure.
    EXPECT_NEAR(area,   m.area(), percentage*area);
    EXPECT_NEAR(volume, m.volume(), percentage*volume);
    EXPECT_NEAR(mean,   m.avgMeanCurvature(), percentage*mean);
    EXPECT_NEAR(gauss,  m.avgGaussianCurvature(), percentage*gauss);
    EXPECT_EQ(0, m.genus());
    //EXPECT_NEAR(area,  openvdb::tools::levelSetArea(*sphere),  percentage*area);
    //EXPECT_NEAR(volume,openvdb::tools::levelSetVolume(*sphere),percentage*volume);
    //EXPECT_EQ(0, openvdb::tools::levelSetGenus(*sphere));

     // Test all measures of sphere in voxel units
    r /= voxelSize;
    area = 4*Pi*r*r;
    volume = 4.0/3.0*Pi*r*r*r;
    mean = 1.0/r;
    gauss = 1.0/(r*r);
    //std::cerr << "\nArea of sphere = " << area << "  " << a << std::endl;
    //std::cerr << "Volume of sphere = " << volume << "  " << v << std::endl;
    //std::cerr << "radius in voxel units = " << r << std::endl;
    //std::cerr << "Avg mean curvature of sphere = " << mean << "  " << cm << std::endl;
    //std::cerr << "Avg gaussian curvature of sphere = " << gauss << "  " << cg << std::endl;
    // Test accuracy of computed measures to within 0.1% of the exact measure.
    EXPECT_NEAR(area,   m.area(false), percentage*area);
    EXPECT_NEAR(volume, m.volume(false), percentage*volume);
    EXPECT_NEAR(mean,   m.avgMeanCurvature(false), percentage*mean);
    EXPECT_NEAR(gauss,  m.avgGaussianCurvature(false), percentage*gauss);
    EXPECT_NEAR(area,  openvdb::tools::levelSetArea(*sphere,false),
                                 percentage*area);
    EXPECT_NEAR(volume,openvdb::tools::levelSetVolume(*sphere,false),
                                 percentage*volume);
    EXPECT_EQ(0, openvdb::tools::levelSetGenus(*sphere));

    // Read level set from file
    /*
    util::CpuTimer timer;
    openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
    openvdb::io::File sourceFile("/usr/pic1/Data/OpenVDB/LevelSetModels/venusstatue.vdb");
    sourceFile.open();
    GridT::Ptr model = openvdb::gridPtrCast<GridT>(sourceFile.getGrids()->at(0));
    m.reinit(*model);

    //m.setGrainSize(1);
    timer.start("\nParallel measure of area and volume");
    m.measure(a, v, false);
    timer.stop();
    std::cerr << "Model: area = " << a << ", volume = " << v << std::endl;

    timer.start("\nParallel measure of area, volume and curvature");
    m.measure(a, v, c, false);
    timer.stop();
    std::cerr << "Model: area = " << a << ", volume = " << v
              << ", average curvature = " << c << std::endl;

    m.setGrainSize(0);
    timer.start("\nSerial measure of area and volume");
    m.measure(a, v, false);
    timer.stop();
    std::cerr << "Model: area = " << a << ", volume = " << v << std::endl;

    timer.start("\nSerial measure of area, volume and curvature");
    m.measure(a, v, c, false);
    timer.stop();
    std::cerr << "Model: area = " << a << ", volume = " << v
              << ", average curvature = " << c << std::endl;
    */

   {// testing total genus of multiple disjoint level set spheres with different radius
     const float dx = 0.5f, r = 50.0f;
     auto grid = openvdb::createLevelSet<openvdb::FloatGrid>(dx);
     EXPECT_THROW(openvdb::tools::levelSetGenus(*grid), openvdb::RuntimeError);
     for (int i=1; i<=3; ++i) {
       auto sphere = openvdb::tools::createLevelSetSphere<GridT>(r+float(i)*5.0f , openvdb::Vec3f(100.0f*float(i)), dx);
       openvdb::tools::csgUnion(*grid, *sphere);
       const int x = openvdb::tools::levelSetEulerCharacteristic(*grid);// since they are not overlapping re-normalization is not required
       //std::cerr << "Euler characteristics of " << i << " sphere(s) = " << x << std::endl;
       EXPECT_EQ(2*i, x);
     }
   }
   {// testing total genus of multiple disjoint level set cubes of different size
     const float dx = 0.5f, size = 50.0f;
     auto grid = openvdb::createLevelSet<openvdb::FloatGrid>(dx);
     EXPECT_THROW(openvdb::tools::levelSetGenus(*grid), openvdb::RuntimeError);
     for (int i=1; i<=2; ++i) {
       auto shape = openvdb::tools::createLevelSetCube<openvdb::FloatGrid>(size, openvdb::Vec3f(100.0f*float(i)), dx);
       openvdb::tools::csgUnion(*grid, *shape);
       const int x = openvdb::tools::levelSetEulerCharacteristic(*grid);
       //std::cerr << "Euler characteristics of " << i << " cubes(s) = " << x << std::endl;
       EXPECT_EQ(2*i, x);
     }
   }
   {// testing Euler characteristic and total genus of multiple intersecting (connected) level set spheres
     const float dx = 0.5f, r = 50.0f;
     auto grid = openvdb::createLevelSet<openvdb::FloatGrid>(dx);
     EXPECT_THROW(openvdb::tools::levelSetGenus(*grid), openvdb::RuntimeError);
     for (int i=1; i<=4; ++i) {
       auto sphere = openvdb::tools::createLevelSetSphere<GridT>( r , openvdb::Vec3f(30.0f*float(i), 0.0f, 0.0f), dx);
       openvdb::tools::csgUnion(*grid, *sphere);
       const int genus = openvdb::tools::levelSetGenus(*grid);
       const int x = openvdb::tools::levelSetEulerCharacteristic(*grid);
       //std::cerr << "Genus of " << i << " sphere(s) = " << genus << std::endl;
       EXPECT_EQ(0, genus);
       //std::cerr << "Euler characteristics of " << i << " sphere(s) = " << genus << std::endl;
       EXPECT_EQ(2, x);
     }
   }

}//testLevelSetMeasure

TEST_F(TestTools, testMagnitude)
{
    using namespace openvdb;
    {
        FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
        FloatTree& tree = grid->tree();
        EXPECT_TRUE(tree.empty());

        const Coord dim(64,64,64);
        const Vec3f center(35.0f, 30.0f, 40.0f);
        const float radius=0.0f;
        unittest_util::makeSphere(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        EXPECT_TRUE(!tree.empty());
        EXPECT_EQ(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

        VectorGrid::Ptr gradGrid = tools::gradient(*grid);
        EXPECT_EQ(int(tree.activeVoxelCount()), int(gradGrid->activeVoxelCount()));

        FloatGrid::Ptr mag = tools::magnitude(*gradGrid);
        EXPECT_EQ(int(tree.activeVoxelCount()), int(mag->activeVoxelCount()));

        FloatGrid::ConstAccessor accessor = mag->getConstAccessor();

        Coord xyz(35,30,30);
        float v = accessor.getValue(xyz);
        EXPECT_NEAR(1.0, v, 0.01);

        xyz.reset(35,10,40);
        v = accessor.getValue(xyz);
        EXPECT_NEAR(1.0, v, 0.01);
    }
    {
        // Test on a grid with (only) tile values.

        Vec3fGrid grid;
        Vec3fTree& tree = grid.tree();
        EXPECT_TRUE(tree.empty());

        const Vec3f v(1.f, 2.f, 2.f);
        const float expectedLength = v.length();

        tree.addTile(/*level=*/1, Coord(-100), v, /*active=*/true);
        tree.addTile(/*level=*/1, Coord(100), v, /*active=*/true);

        EXPECT_TRUE(!tree.empty());

        FloatGrid::Ptr length = tools::magnitude(grid);

        EXPECT_EQ(int(tree.activeVoxelCount()), int(length->activeVoxelCount()));

        for (auto it = length->cbeginValueOn(); it; ++it) {
            EXPECT_NEAR(expectedLength, *it, 1.0e-6);
        }
    }
}


TEST_F(TestTools, testMaskedMagnitude)
{
    using namespace openvdb;
    {
        FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
        FloatTree& tree = grid->tree();
        EXPECT_TRUE(tree.empty());

        const Coord dim(64,64,64);
        const Vec3f center(35.0f, 30.0f, 40.0f);
        const float radius=0.0f;
        unittest_util::makeSphere(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        EXPECT_TRUE(!tree.empty());
        EXPECT_EQ(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

        VectorGrid::Ptr gradGrid = tools::gradient(*grid);
        EXPECT_EQ(int(tree.activeVoxelCount()), int(gradGrid->activeVoxelCount()));

        // create a masking grid
        const CoordBBox maskbbox(Coord(35, 30, 30), Coord(41, 41, 41));
        BoolGrid::Ptr maskGrid = BoolGrid::create(false);
        maskGrid->fill(maskbbox, true/*value*/, true/*activate*/);

        // compute the magnitude in masked region
        FloatGrid::Ptr mag = tools::magnitude(*gradGrid, *maskGrid);

        FloatGrid::ConstAccessor accessor = mag->getConstAccessor();

        // test in the masked region
        Coord xyz(35,30,30);
        EXPECT_TRUE(maskbbox.isInside(xyz));
        float v = accessor.getValue(xyz);
        EXPECT_NEAR(1.0, v, 0.01);

        // test outside the masked region
        xyz.reset(35,10,40);
        EXPECT_TRUE(!maskbbox.isInside(xyz));
        v = accessor.getValue(xyz);
        EXPECT_NEAR(0.0, v, 0.01);
    }
    {
        // Test on a grid with (only) tile values.

        Vec3fGrid grid;
        Vec3fTree& tree = grid.tree();
        EXPECT_TRUE(tree.empty());

        const Vec3f v(1.f, 2.f, 2.f);
        const float expectedLength = v.length();

        tree.addTile(/*level=*/1, Coord(100), v, /*active=*/true);
        const int expectedActiveVoxelCount = int(tree.activeVoxelCount());
        tree.addTile(/*level=*/1, Coord(-100), v, /*active=*/true);

        EXPECT_TRUE(!tree.empty());

        BoolGrid mask;
        mask.fill(CoordBBox(Coord(90), Coord(200)), true, true);

        FloatGrid::Ptr length = tools::magnitude(grid, mask);

        EXPECT_EQ(expectedActiveVoxelCount, int(length->activeVoxelCount()));

        for (auto it = length->cbeginValueOn(); it; ++it) {
            EXPECT_NEAR(expectedLength, *it, 1.0e-6);
        }
    }
}


TEST_F(TestTools, testNormalize)
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(5.0);
    openvdb::FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=10.0f;
    unittest_util::makeSphere<openvdb::FloatGrid>(
        dim,center,radius,*grid, unittest_util::SPHERE_DENSE);

    EXPECT_EQ(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));
    openvdb::Coord xyz(10, 20, 30);

    openvdb::VectorGrid::Ptr grad = openvdb::tools::gradient(*grid);

    using Vec3Type = openvdb::VectorGrid::ValueType;

    using ValueIter = openvdb::VectorGrid::ValueOnIter;

    struct Local {
        static inline Vec3Type op(const Vec3Type &x) { return x * 2.0f; }
        static inline void visit(const ValueIter& it) { it.setValue(op(*it)); }
    };

    openvdb::tools::foreach(grad->beginValueOn(), Local::visit, true);

    openvdb::VectorGrid::ConstAccessor accessor = grad->getConstAccessor();

    xyz = openvdb::Coord(35,10,40);
    Vec3Type v = accessor.getValue(xyz);
    //std::cerr << "\nPassed testNormalize(" << xyz << ")=" << v.length() << std::endl;
    EXPECT_NEAR(2.0,v.length(),0.001);
    openvdb::VectorGrid::Ptr norm = openvdb::tools::normalize(*grad);

    accessor = norm->getConstAccessor();
    v = accessor.getValue(xyz);
    //std::cerr << "\nPassed testNormalize(" << xyz << ")=" << v.length() << std::endl;
    EXPECT_NEAR(1.0, v.length(), 0.0001);
}


TEST_F(TestTools, testMaskedNormalize)
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(5.0);
    openvdb::FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=10.0f;
    unittest_util::makeSphere<openvdb::FloatGrid>(
        dim,center,radius,*grid, unittest_util::SPHERE_DENSE);

    EXPECT_EQ(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));
    openvdb::Coord xyz(10, 20, 30);

    openvdb::VectorGrid::Ptr grad = openvdb::tools::gradient(*grid);

    using Vec3Type = openvdb::VectorGrid::ValueType;

    using ValueIter = openvdb::VectorGrid::ValueOnIter;

    struct Local {
        static inline Vec3Type op(const Vec3Type &x) { return x * 2.0f; }
        static inline void visit(const ValueIter& it) { it.setValue(op(*it)); }
    };

    openvdb::tools::foreach(grad->beginValueOn(), Local::visit, true);

    openvdb::VectorGrid::ConstAccessor accessor = grad->getConstAccessor();

    xyz = openvdb::Coord(35,10,40);
    Vec3Type v = accessor.getValue(xyz);

    // create a masking grid

    const openvdb::CoordBBox maskbbox(openvdb::Coord(35, 30, 30), openvdb::Coord(41, 41, 41));
    openvdb::BoolGrid::Ptr maskGrid = openvdb::BoolGrid::create(false);
    maskGrid->fill(maskbbox, true/*value*/, true/*activate*/);

    EXPECT_NEAR(2.0,v.length(),0.001);

    // compute the normalized valued in the masked region
    openvdb::VectorGrid::Ptr norm = openvdb::tools::normalize(*grad, *maskGrid);

    accessor = norm->getConstAccessor();
    { // outside the masked region
        EXPECT_TRUE(!maskbbox.isInside(xyz));
        v = accessor.getValue(xyz);
        EXPECT_NEAR(0.0, v.length(), 0.0001);
    }
    { // inside the masked region
        xyz.reset(35, 30, 30);
        v = accessor.getValue(xyz);
        EXPECT_NEAR(1.0, v.length(), 0.0001);
    }
}

////////////////////////////////////////


TEST_F(TestTools, testPointAdvect)
{
    {
        // Setup:    Advect a number of points in a uniform velocity field (1,1,1).
        //           over a time dt=1 with each of the 4 different advection schemes.
        //           Points initialized at latice points.
        //
        // Uses:     FloatTree (velocity), collocated sampling, advection
        //
        // Expected: All advection schemes will have the same result.  Each point will
        //           be advanced to a new latice point.  The i-th point will be at (i+1,i+1,i+1)
        //

        const size_t numPoints = 2000000;

        // create a uniform velocity field in SINGLE PRECISION
        const openvdb::Vec3f velocityBackground(1, 1, 1);
        openvdb::Vec3fGrid::Ptr velocityGrid = openvdb::Vec3fGrid::create(velocityBackground);

        // using all the default template arguments
        openvdb::tools::PointAdvect<> advectionTool(*velocityGrid);

        // create points
        std::vector<openvdb::Vec3f> pointList(numPoints);  /// larger than the tbb chunk size
        for (size_t i = 0; i < numPoints; i++) {
            pointList[i] = openvdb::Vec3f(float(i), float(i), float(i));
        }

        for (unsigned int order = 1; order < 5; ++order) {
            // check all four time integrations schemes
            // construct an advection tool.  By default the number of cpt iterations is zero
            advectionTool.setIntegrationOrder(order);
            advectionTool.advect(pointList, /*dt=*/1.0, /*iterations=*/1);

            // check locations
            for (size_t i = 0; i < numPoints; i++) {
                openvdb::Vec3f expected(float(i + 1), float(i + 1), float(i + 1));
                EXPECT_EQ(expected, pointList[i]);
            }
            // reset values
            for (size_t i = 0; i < numPoints; i++) {
                pointList[i] = openvdb::Vec3f(float(i), float(i), float(i));
            }
        }

    }

    {
        // Setup:    Advect a number of points in a uniform velocity field (1,1,1).
        //           over a time dt=1 with each of the 4 different advection schemes.
        //           And then project the point location onto the x-y plane
        //           Points initialized at latice points.
        //
        // Uses:     DoubleTree (velocity), staggered sampling, constraint projection, advection
        //
        // Expected: All advection schemes will have the same result.  Modes 1-4: Each point will
        //           be advanced to a new latice point and projected to x-y plane.
        //           The i-th point will be at (i+1,i+1,0).  For mode 0 (no advection), i-th point
        //           will be found at (i, i, 0)

        const size_t numPoints = 4;

        // create a uniform velocity field in DOUBLE PRECISION
        const openvdb::Vec3d velocityBackground(1, 1, 1);
        openvdb::Vec3dGrid::Ptr velocityGrid = openvdb::Vec3dGrid::create(velocityBackground);

        // create a simple (horizontal) constraint field valid for a
        // (-10,10)x(-10,10)x(-10,10)
        const openvdb::Vec3d cptBackground(0, 0, 0);
        openvdb::Vec3dGrid::Ptr cptGrid = openvdb::Vec3dGrid::create(cptBackground);
        openvdb::Vec3dTree& cptTree = cptGrid->tree();

        // create points
        std::vector<openvdb::Vec3d> pointList(numPoints);
        for (unsigned int i = 0; i < numPoints; i++) pointList[i] = openvdb::Vec3d(i, i, i);

        // Initialize the constraint field in a [-10,10]x[-10,10]x[-10,10] box
        // this test will only work if the points remain in the box
        openvdb::Coord ijk(0, 0, 0);
        for (int i = -10; i < 11; i++) {
            ijk.setX(i);
            for (int j = -10; j < 11; j++) {
                ijk.setY(j);
                for (int k = -10; k < 11; k++) {
                    ijk.setZ(k);
                    // set the value as projection onto the x-y plane
                    cptTree.setValue(ijk, openvdb::Vec3d(i, j, 0));
                }
            }
        }

        // construct an advection tool.  By default the number of cpt iterations is zero
        openvdb::tools::ConstrainedPointAdvect<openvdb::Vec3dGrid,
            std::vector<openvdb::Vec3d>, true> constrainedAdvectionTool(*velocityGrid, *cptGrid, 0);
        constrainedAdvectionTool.setThreaded(false);

        // change the number of constraint interation from default 0 to 5
        constrainedAdvectionTool.setConstraintIterations(5);

        // test the pure-projection mode (order = 0)
        constrainedAdvectionTool.setIntegrationOrder(0);

        // change the number of constraint interation (from 0 to 5)
        constrainedAdvectionTool.setConstraintIterations(5);

        constrainedAdvectionTool.advect(pointList, /*dt=*/1.0, /*iterations=*/1);

        // check locations
        for (unsigned int i = 0; i < numPoints; i++) {
            openvdb::Vec3d expected(i, i, 0);  // location (i, i, i) projected on to x-y plane
            for (int n=0; n<3; ++n) {
                EXPECT_NEAR(expected[n], pointList[i][n], /*tolerance=*/1e-6);
            }
        }

        // reset values
        for (unsigned int i = 0; i < numPoints; i++) pointList[i] = openvdb::Vec3d(i, i, i);

        // test all four time integrations schemes
        for (unsigned int order = 1; order < 5; ++order) {

            constrainedAdvectionTool.setIntegrationOrder(order);

            constrainedAdvectionTool.advect(pointList, /*dt=*/1.0, /*iterations=*/1);

            // check locations
            for (unsigned int i = 0; i < numPoints; i++) {
                openvdb::Vec3d expected(i+1, i+1, 0); // location (i,i,i) projected onto x-y plane
                for (int n=0; n<3; ++n) {
                    EXPECT_NEAR(expected[n], pointList[i][n], /*tolerance=*/1e-6);
                }
            }
            // reset values
            for (unsigned int i = 0; i < numPoints; i++) pointList[i] = openvdb::Vec3d(i, i, i);
        }
    }
}


////////////////////////////////////////


namespace {

    struct PointList
    {
        struct Point { double x,y,z; };
        std::vector<Point> list;
        openvdb::Index64 size() const { return openvdb::Index64(list.size()); }
        void add(const openvdb::Vec3d &p) { Point q={p[0],p[1],p[2]}; list.push_back(q); }
    };
}


TEST_F(TestTools, testPointScatter)
{
    using GridType = openvdb::FloatGrid;
    const openvdb::Coord dim(64, 64, 64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius = 20.0;
    using RandGen = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19,
        0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>; // mt11213b
    RandGen mtRand;

    GridType::Ptr grid = GridType::create(/*background=*/2.0);
    unittest_util::makeSphere<GridType>(
        dim, center, radius, *grid, unittest_util::SPHERE_DENSE_NARROW_BAND);

    {// test fixed point count scattering
        const openvdb::Index64 pointCount = 1000;
        PointList points;
        openvdb::tools::UniformPointScatter<PointList, RandGen> scatter(points, pointCount, mtRand);
        scatter.operator()<GridType>(*grid);
        EXPECT_EQ( pointCount, scatter.getPointCount() );
        EXPECT_EQ( pointCount, points.size() );
    }
    {// test uniform density scattering
        const float density = 1.0f;//per volume = per voxel since voxel size = 1
        PointList points;
        openvdb::tools::UniformPointScatter<PointList, RandGen> scatter(points, density, mtRand);
        scatter.operator()<GridType>(*grid);
        EXPECT_EQ( scatter.getVoxelCount(), scatter.getPointCount() );
        EXPECT_EQ( scatter.getVoxelCount(), points.size() );
    }
    {// test non-uniform density scattering
        const float density = 1.0f;//per volume = per voxel since voxel size = 1
        PointList points;
        openvdb::tools::NonUniformPointScatter<PointList, RandGen> scatter(points, density, mtRand);
        scatter.operator()<GridType>(*grid);
        EXPECT_TRUE( scatter.getVoxelCount() < scatter.getPointCount() );
        EXPECT_EQ( scatter.getPointCount(), points.size() );
    }
     {// test dense uniform scattering
        const size_t pointsPerVoxel = 8;
        PointList points;
        openvdb::tools::DenseUniformPointScatter<PointList, RandGen>
            scatter(points, pointsPerVoxel, mtRand);
        scatter.operator()<GridType>(*grid);
        EXPECT_EQ( scatter.getVoxelCount()*pointsPerVoxel, scatter.getPointCount() );
        EXPECT_EQ( scatter.getPointCount(), points.size() );
    }
}

////////////////////////////////////////

TEST_F(TestTools, testVolumeAdvect)
{
    using namespace openvdb;

    Vec3fGrid velocity(Vec3f(1.0f, 0.0f, 0.0f));
    using GridT = FloatGrid;
    using AdvT = tools::VolumeAdvection<Vec3fGrid>;
    using SamplerT = tools::Sampler<1>;

    {//test non-uniform grids (throws)
        GridT::Ptr density0 = GridT::create(0.0f);
        density0->transform().preScale(Vec3d(1.0, 2.0, 3.0));//i.e. non-uniform voxels
        AdvT a(velocity);
        EXPECT_THROW((a.advect<GridT, SamplerT>(*density0, 0.1f)), RuntimeError);
    }

    {// test spatialOrder and temporalOrder
        AdvT a(velocity);

        // Default should be SEMI
        EXPECT_EQ(1, a.spatialOrder());
        EXPECT_EQ(1, a.temporalOrder());
        EXPECT_TRUE(!a.isLimiterOn());

        a.setIntegrator(tools::Scheme::SEMI);
        EXPECT_EQ(1, a.spatialOrder());
        EXPECT_EQ(1, a.temporalOrder());
        EXPECT_TRUE(!a.isLimiterOn());

        a.setIntegrator(tools::Scheme::MID);
        EXPECT_EQ(1, a.spatialOrder());
        EXPECT_EQ(2, a.temporalOrder());
        EXPECT_TRUE(!a.isLimiterOn());

        a.setIntegrator(tools::Scheme::RK3);
        EXPECT_EQ(1, a.spatialOrder());
        EXPECT_EQ(3, a.temporalOrder());
        EXPECT_TRUE(!a.isLimiterOn());

        a.setIntegrator(tools::Scheme::RK4);
        EXPECT_EQ(1, a.spatialOrder());
        EXPECT_EQ(4, a.temporalOrder());
        EXPECT_TRUE(!a.isLimiterOn());

        a.setIntegrator(tools::Scheme::MAC);
        EXPECT_EQ(2, a.spatialOrder());
        EXPECT_EQ(2, a.temporalOrder());
        EXPECT_TRUE( a.isLimiterOn());

        a.setIntegrator(tools::Scheme::BFECC);
        EXPECT_EQ(2, a.spatialOrder());
        EXPECT_EQ(2, a.temporalOrder());
        EXPECT_TRUE( a.isLimiterOn());

        a.setLimiter(tools::Scheme::NO_LIMITER);
        EXPECT_EQ(2, a.spatialOrder());
        EXPECT_EQ(2, a.temporalOrder());
        EXPECT_TRUE(!a.isLimiterOn());
    }

    {//test RK4 advect without a mask
        GridT::Ptr density0 = GridT::create(0.0f), density1;
        density0->fill(CoordBBox(Coord(0),Coord(6)), 1.0f);
        EXPECT_EQ(density0->tree().getValue(Coord( 3,3,3)), 1.0f);
        EXPECT_EQ(density0->tree().getValue(Coord(24,3,3)), 0.0f);
        EXPECT_TRUE( density0->tree().isValueOn(Coord( 3,3,3)));
        EXPECT_TRUE(!density0->tree().isValueOn(Coord(24,3,3)));

        AdvT a(velocity);
        a.setIntegrator(tools::Scheme::RK4);
        for (int i=1; i<=240; ++i) {
            density1 = a.advect<GridT, SamplerT>(*density0, 0.1f);
            //std::ostringstream ostr;
            //ostr << "densityAdvect" << "_" << i << ".vdb";
            //std::cerr << "Writing " << ostr.str() << std::endl;
            //openvdb::io::File file(ostr.str());
            //openvdb::GridPtrVec grids;
            //grids.push_back(density1);
            //file.write(grids);
            density0 = density1;
        }
        EXPECT_EQ(density0->tree().getValue(Coord(3,3,3)), 0.0f);
        EXPECT_TRUE(density0->tree().getValue(Coord(24,3,3)) > 0.0f);
        EXPECT_TRUE(!density0->tree().isValueOn(Coord( 3,3,3)));
        EXPECT_TRUE( density0->tree().isValueOn(Coord(24,3,3)));
    }
    {//test MAC advect without a mask
        GridT::Ptr density0 = GridT::create(0.0f), density1;
        density0->fill(CoordBBox(Coord(0),Coord(6)), 1.0f);
        EXPECT_EQ(density0->tree().getValue(Coord( 3,3,3)), 1.0f);
        EXPECT_EQ(density0->tree().getValue(Coord(24,3,3)), 0.0f);
        EXPECT_TRUE( density0->tree().isValueOn(Coord( 3,3,3)));
        EXPECT_TRUE(!density0->tree().isValueOn(Coord(24,3,3)));

        AdvT a(velocity);
        a.setIntegrator(tools::Scheme::BFECC);
        for (int i=1; i<=240; ++i) {
            density1 = a.advect<GridT, SamplerT>(*density0, 0.1f);
            //std::ostringstream ostr;
            //ostr << "densityAdvect" << "_" << i << ".vdb";
            //std::cerr << "Writing " << ostr.str() << std::endl;
            //openvdb::io::File file(ostr.str());
            //openvdb::GridPtrVec grids;
            //grids.push_back(density1);
            //file.write(grids);
            density0 = density1;
        }
        EXPECT_EQ(density0->tree().getValue(Coord(3,3,3)), 0.0f);
        EXPECT_TRUE(density0->tree().getValue(Coord(24,3,3)) > 0.0f);
        EXPECT_TRUE(!density0->tree().isValueOn(Coord( 3,3,3)));
        EXPECT_TRUE( density0->tree().isValueOn(Coord(24,3,3)));
    }
    {//test advect with a mask
        GridT::Ptr density0 = GridT::create(0.0f), density1;
        density0->fill(CoordBBox(Coord(0),Coord(6)), 1.0f);
        EXPECT_EQ(density0->tree().getValue(Coord( 3,3,3)), 1.0f);
        EXPECT_EQ(density0->tree().getValue(Coord(24,3,3)), 0.0f);
        EXPECT_TRUE( density0->tree().isValueOn(Coord( 3,3,3)));
        EXPECT_TRUE(!density0->tree().isValueOn(Coord(24,3,3)));

        BoolGrid::Ptr mask = BoolGrid::create(false);
        mask->fill(CoordBBox(Coord(4,0,0),Coord(30,8,8)), true);

        AdvT a(velocity);
        a.setGrainSize(0);
        a.setIntegrator(tools::Scheme::MAC);
        //a.setIntegrator(tools::Scheme::BFECC);
        //a.setIntegrator(tools::Scheme::RK4);
        for (int i=1; i<=240; ++i) {
            density1 = a.advect<GridT, BoolGrid, SamplerT>(*density0, *mask, 0.1f);
            //std::ostringstream ostr;
            //ostr << "densityAdvectMask" << "_" << i << ".vdb";
            //std::cerr << "Writing " << ostr.str() << std::endl;
            //openvdb::io::File file(ostr.str());
            //openvdb::GridPtrVec grids;
            //grids.push_back(density1);
            //file.write(grids);
            density0 = density1;
        }
        EXPECT_EQ(density0->tree().getValue(Coord(3,3,3)), 1.0f);
        EXPECT_TRUE(density0->tree().getValue(Coord(24,3,3)) > 0.0f);
        EXPECT_TRUE(density0->tree().isValueOn(Coord( 3,3,3)));
        EXPECT_TRUE(density0->tree().isValueOn(Coord(24,3,3)));
    }

    /*
    {//benchmark on a sphere
        util::CpuTimer timer;
        GridT::Ptr density0 = GridT::create(0.0f), density1;
        density0->fill(CoordBBox(Coord(0), Coord(600)), 1.0f);
        timer.start("densify");
        density0->tree().voxelizeActiveTiles();
        timer.stop();
        AdvT a(velocity);
        a.setGrainSize(1);
        //a.setLimiter(tools::Scheme::NO_LIMITER);
        //a.setIntegrator(tools::Scheme::MAC);
        //a.setIntegrator(tools::Scheme::BFECC);
        a.setIntegrator(tools::Scheme::RK4);

        for (int i=1; i<=10; ++i) {
            timer.start("Volume Advection");
            density1 = a.advect<GridT, SamplerT>(*density0, 0.1f);
            timer.stop();
            std::ostringstream ostr;
            ostr << "densityAdvectMask" << "_" << i << ".vdb";
            std::cerr << "Writing " << ostr.str() << std::endl;
            io::File file(ostr.str());
            GridPtrVec grids;
            grids.push_back(density1);
            file.write(grids);
            density0.swap(density1);
            }
            }
    */
}// testVolumeAdvect

////////////////////////////////////////


TEST_F(TestTools, testFloatApply)
{
    using ValueIter = openvdb::FloatTree::ValueOnIter;

    struct Local {
        static inline float op(float x) { return x * 2.f; }
        static inline void visit(const ValueIter& it) { it.setValue(op(*it)); }
    };

    const float background = 1.0;
    openvdb::FloatTree tree(background);

    const int MIN = -1000, MAX = 1000, STEP = 50;
    openvdb::Coord xyz;
    for (int z = MIN; z < MAX; z += STEP) {
        xyz.setZ(z);
        for (int y = MIN; y < MAX; y += STEP) {
            xyz.setY(y);
            for (int x = MIN; x < MAX; x += STEP) {
                xyz.setX(x);
                tree.setValue(xyz, float(x + y + z));
            }
        }
    }
    /// @todo set some tile values

    openvdb::tools::foreach(tree.begin<ValueIter>(), Local::visit, /*threaded=*/true);

    float expected = Local::op(background);
    //EXPECT_NEAR(expected, tree.background(), /*tolerance=*/0.0);
    //expected = Local::op(-background);
    //EXPECT_NEAR(expected, -tree.background(), /*tolerance=*/0.0);

    for (openvdb::FloatTree::ValueOnCIter it = tree.cbeginValueOn(); it; ++it) {
        xyz = it.getCoord();
        expected = Local::op(float(xyz[0] + xyz[1] + xyz[2]));
        EXPECT_NEAR(expected, it.getValue(), /*tolerance=*/0.0);
    }
}


////////////////////////////////////////


namespace {

template<typename IterT>
struct MatMul {
    openvdb::math::Mat3s mat;
    MatMul(const openvdb::math::Mat3s& _mat): mat(_mat) {}
    openvdb::Vec3s xform(const openvdb::Vec3s& v) const { return mat.transform(v); }
    void operator()(const IterT& it) const { it.setValue(xform(*it)); }
};

}


TEST_F(TestTools, testVectorApply)
{
    using ValueIter = openvdb::VectorTree::ValueOnIter;

    const openvdb::Vec3s background(1, 1, 1);
    openvdb::VectorTree tree(background);

    const int MIN = -1000, MAX = 1000, STEP = 80;
    openvdb::Coord xyz;
    for (int z = MIN; z < MAX; z += STEP) {
        xyz.setZ(z);
        for (int y = MIN; y < MAX; y += STEP) {
            xyz.setY(y);
            for (int x = MIN; x < MAX; x += STEP) {
                xyz.setX(x);
                tree.setValue(xyz, openvdb::Vec3s(float(x), float(y), float(z)));
            }
        }
    }
    /// @todo set some tile values

    MatMul<ValueIter> op(openvdb::math::Mat3s(1, 2, 3, -1, -2, -3, 3, 2, 1));
    openvdb::tools::foreach(tree.beginValueOn(), op, /*threaded=*/true);

    openvdb::Vec3s expected;
    for (openvdb::VectorTree::ValueOnCIter it = tree.cbeginValueOn(); it; ++it) {
        xyz = it.getCoord();
        expected = op.xform(openvdb::Vec3s(float(xyz[0]), float(xyz[1]), float(xyz[2])));
        EXPECT_EQ(expected, it.getValue());
    }
}


////////////////////////////////////////


namespace {

struct AccumSum {
    int64_t sum; int joins;
    AccumSum(): sum(0), joins(0) {}
    void operator()(const openvdb::Int32Tree::ValueOnCIter& it)
    {
        if (it.isVoxelValue()) sum += *it;
        else sum += (*it) * it.getVoxelCount();
    }
    void join(AccumSum& other) { sum += other.sum; joins += 1 + other.joins; }
};


struct AccumLeafVoxelCount {
    using LeafRange = openvdb::tree::LeafManager<openvdb::Int32Tree>::LeafRange;
    openvdb::Index64 count;
    AccumLeafVoxelCount(): count(0) {}
    void operator()(const LeafRange::Iterator& it) { count += it->onVoxelCount(); }
    void join(AccumLeafVoxelCount& other) { count += other.count; }
};

}


TEST_F(TestTools, testAccumulate)
{
    using namespace openvdb;

    const int value = 2;
    Int32Tree tree(/*background=*/0);
    tree.fill(CoordBBox::createCube(Coord(0), 198), value, /*active=*/true);

    const int64_t expected = tree.activeVoxelCount() * value;
    {
        AccumSum op;
        tools::accumulate(tree.cbeginValueOn(), op, /*threaded=*/false);
        EXPECT_EQ(expected, op.sum);
        EXPECT_EQ(0, op.joins);
    }
    {
        AccumSum op;
        tools::accumulate(tree.cbeginValueOn(), op, /*threaded=*/true);
        EXPECT_EQ(expected, op.sum);
    }
    {
        AccumLeafVoxelCount op;
        tree::LeafManager<Int32Tree> mgr(tree);
        tools::accumulate(mgr.leafRange().begin(), op, /*threaded=*/true);
        EXPECT_EQ(tree.activeLeafVoxelCount(), op.count);
    }
}


////////////////////////////////////////


namespace {

template<typename InIterT, typename OutTreeT>
struct FloatToVec
{
    using ValueT = typename InIterT::ValueT;
    using Accessor = typename openvdb::tree::ValueAccessor<OutTreeT>;

    // Transform a scalar value into a vector value.
    static openvdb::Vec3s toVec(const ValueT& v) { return openvdb::Vec3s(v, v*2, v*3); }

    FloatToVec() : numTiles{0} {}
    FloatToVec(const FloatToVec& other) : numTiles{other.numTiles.load(std::memory_order_acquire)} {}

    void operator()(const InIterT& it, Accessor& acc)
    {
        if (it.isVoxelValue()) { // set a single voxel
            acc.setValue(it.getCoord(), toVec(*it));
        } else { // fill an entire tile
            numTiles.fetch_add(1);
            openvdb::CoordBBox bbox;
            it.getBoundingBox(bbox);
            acc.tree().fill(bbox, toVec(*it));
        }
    }

    std::atomic<int> numTiles;
};

}


TEST_F(TestTools, testTransformValues)
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Vec3s;

    using Tree323f = openvdb::tree::Tree4<float, 3, 2, 3>::Type;
    using Tree323v = openvdb::tree::Tree4<Vec3s, 3, 2, 3>::Type;

    const float background = 1.0;
    Tree323f ftree(background);

    const int MIN = -1000, MAX = 1000, STEP = 80;
    Coord xyz;
    for (int z = MIN; z < MAX; z += STEP) {
        xyz.setZ(z);
        for (int y = MIN; y < MAX; y += STEP) {
            xyz.setY(y);
            for (int x = MIN; x < MAX; x += STEP) {
                xyz.setX(x);
                ftree.setValue(xyz, float(x + y + z));
            }
        }
    }
    // Set some tile values.
    ftree.fill(CoordBBox(Coord(1024), Coord(1024 + 8 - 1)), 3 * 1024); // level-1 tile
    ftree.fill(CoordBBox(Coord(2048), Coord(2048 + 32 - 1)), 3 * 2048); // level-2 tile
    ftree.fill(CoordBBox(Coord(3072), Coord(3072 + 256 - 1)), 3 * 3072); // level-3 tile

    for (int shareOp = 0; shareOp <= 1; ++shareOp) {
        FloatToVec<Tree323f::ValueOnCIter, Tree323v> op;
        Tree323v vtree;
        openvdb::tools::transformValues(ftree.cbeginValueOn(), vtree, op,
            /*threaded=*/true, shareOp);

        // The tile count is accurate only if the functor is shared.  Otherwise,
        // it is initialized to zero in the main thread and never changed.
        EXPECT_EQ(shareOp ? 3 : 0, int(op.numTiles));

        Vec3s expected;
        for (Tree323v::ValueOnCIter it = vtree.cbeginValueOn(); it; ++it) {
            xyz = it.getCoord();
            expected = op.toVec(float(xyz[0] + xyz[1] + xyz[2]));
            EXPECT_EQ(expected, it.getValue());
        }
        // Check values inside the tiles.
        EXPECT_EQ(op.toVec(3 * 1024), vtree.getValue(Coord(1024 + 4)));
        EXPECT_EQ(op.toVec(3 * 2048), vtree.getValue(Coord(2048 + 16)));
        EXPECT_EQ(op.toVec(3 * 3072), vtree.getValue(Coord(3072 + 128)));
    }
}


////////////////////////////////////////


TEST_F(TestTools, testUtil)
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Vec3s;

    using CharTree = openvdb::tree::Tree4<bool, 3, 2, 3>::Type;

    // Test boolean operators
    CharTree treeA(false), treeB(false);

    treeA.fill(CoordBBox(Coord(-10), Coord(10)), true);
    treeA.voxelizeActiveTiles();

    treeB.fill(CoordBBox(Coord(-10), Coord(10)), true);
    treeB.voxelizeActiveTiles();

    const size_t voxelCountA = treeA.activeVoxelCount();
    const size_t voxelCountB = treeB.activeVoxelCount();

    EXPECT_EQ(voxelCountA, voxelCountB);

    CharTree::Ptr tree = openvdb::util::leafTopologyDifference(treeA, treeB);
    EXPECT_TRUE(tree->activeVoxelCount() == 0);

    tree = openvdb::util::leafTopologyIntersection(treeA, treeB);
    EXPECT_TRUE(tree->activeVoxelCount() == voxelCountA);

    treeA.fill(CoordBBox(Coord(-10), Coord(22)), true);
    treeA.voxelizeActiveTiles();

    const size_t voxelCount = treeA.activeVoxelCount();

    tree = openvdb::util::leafTopologyDifference(treeA, treeB);
    EXPECT_TRUE(tree->activeVoxelCount() == (voxelCount - voxelCountA));

    tree = openvdb::util::leafTopologyIntersection(treeA, treeB);
    EXPECT_TRUE(tree->activeVoxelCount() == voxelCountA);
}


////////////////////////////////////////


TEST_F(TestTools, testVectorTransformer)
{
    using namespace openvdb;

    Mat4d xform = Mat4d::identity();
    xform.preTranslate(Vec3d(0.1, -2.5, 3));
    xform.preScale(Vec3d(0.5, 1.1, 2));
    xform.preRotate(math::X_AXIS, 30.0 * openvdb::math::pi<double>() / 180.0);
    xform.preRotate(math::Y_AXIS, 300.0 * openvdb::math::pi<double>() / 180.0);

    Mat4d invXform = xform.inverse();
    invXform = invXform.transpose();

    {
        // Set some vector values in a grid, then verify that tools::transformVectors()
        // transforms them as expected for each VecType.

        const Vec3s refVec0(0, 0, 0), refVec1(1, 0, 0), refVec2(0, 1, 0), refVec3(0, 0, 1);

        Vec3SGrid grid;
        Vec3SGrid::Accessor acc = grid.getAccessor();

#define resetGrid() \
    { \
        grid.clear(); \
        acc.setValue(Coord(0), refVec0); \
        acc.setValue(Coord(1), refVec1); \
        acc.setValue(Coord(2), refVec2); \
        acc.setValue(Coord(3), refVec3); \
    }

        // Verify that grid values are in world space by default.
        EXPECT_TRUE(grid.isInWorldSpace());

        resetGrid();
        grid.setVectorType(VEC_INVARIANT);
        tools::transformVectors(grid, xform);
        EXPECT_TRUE(acc.getValue(Coord(0)).eq(refVec0));
        EXPECT_TRUE(acc.getValue(Coord(1)).eq(refVec1));
        EXPECT_TRUE(acc.getValue(Coord(2)).eq(refVec2));
        EXPECT_TRUE(acc.getValue(Coord(3)).eq(refVec3));

        resetGrid();
        grid.setVectorType(VEC_COVARIANT);
        tools::transformVectors(grid, xform);
        EXPECT_TRUE(acc.getValue(Coord(0)).eq(invXform.transform3x3(refVec0)));
        EXPECT_TRUE(acc.getValue(Coord(1)).eq(invXform.transform3x3(refVec1)));
        EXPECT_TRUE(acc.getValue(Coord(2)).eq(invXform.transform3x3(refVec2)));
        EXPECT_TRUE(acc.getValue(Coord(3)).eq(invXform.transform3x3(refVec3)));

        resetGrid();
        grid.setVectorType(VEC_COVARIANT_NORMALIZE);
        tools::transformVectors(grid, xform);
        EXPECT_EQ(refVec0, acc.getValue(Coord(0)));
        EXPECT_TRUE(acc.getValue(Coord(1)).eq(invXform.transform3x3(refVec1).unit()));
        EXPECT_TRUE(acc.getValue(Coord(2)).eq(invXform.transform3x3(refVec2).unit()));
        EXPECT_TRUE(acc.getValue(Coord(3)).eq(invXform.transform3x3(refVec3).unit()));

        resetGrid();
        grid.setVectorType(VEC_CONTRAVARIANT_RELATIVE);
        tools::transformVectors(grid, xform);
        EXPECT_TRUE(acc.getValue(Coord(0)).eq(xform.transform3x3(refVec0)));
        EXPECT_TRUE(acc.getValue(Coord(1)).eq(xform.transform3x3(refVec1)));
        EXPECT_TRUE(acc.getValue(Coord(2)).eq(xform.transform3x3(refVec2)));
        EXPECT_TRUE(acc.getValue(Coord(3)).eq(xform.transform3x3(refVec3)));

        resetGrid();
        grid.setVectorType(VEC_CONTRAVARIANT_ABSOLUTE);
        /// @todo This doesn't really test the behavior w.r.t. homogeneous coords.
        tools::transformVectors(grid, xform);
        EXPECT_TRUE(acc.getValue(Coord(0)).eq(xform.transformH(refVec0)));
        EXPECT_TRUE(acc.getValue(Coord(1)).eq(xform.transformH(refVec1)));
        EXPECT_TRUE(acc.getValue(Coord(2)).eq(xform.transformH(refVec2)));
        EXPECT_TRUE(acc.getValue(Coord(3)).eq(xform.transformH(refVec3)));

        // Verify that transformVectors() has no effect on local-space grids.
        resetGrid();
        grid.setVectorType(VEC_CONTRAVARIANT_RELATIVE);
        grid.setIsInWorldSpace(false);
        tools::transformVectors(grid, xform);
        EXPECT_TRUE(acc.getValue(Coord(0)).eq(refVec0));
        EXPECT_TRUE(acc.getValue(Coord(1)).eq(refVec1));
        EXPECT_TRUE(acc.getValue(Coord(2)).eq(refVec2));
        EXPECT_TRUE(acc.getValue(Coord(3)).eq(refVec3));

#undef resetGrid
    }
    {
        // Verify that transformVectors() operates only on vector-valued grids.
        FloatGrid scalarGrid;
        EXPECT_THROW(tools::transformVectors(scalarGrid, xform), TypeError);
    }
}


////////////////////////////////////////


TEST_F(TestTools, testPrune)
{
    /// @todo Add more unit-tests!

    using namespace openvdb;

    {// try prunning a tree with const values
        const float value = 5.345f;

        FloatTree tree(value);
        EXPECT_EQ(Index32(0), tree.leafCount());
        EXPECT_EQ(Index32(1), tree.nonLeafCount()); // root node
        EXPECT_TRUE(tree.empty());

        tree.fill(CoordBBox(Coord(-10), Coord(10)), value, /*active=*/false);
        EXPECT_TRUE(!tree.empty());

        tools::prune(tree);

        EXPECT_EQ(Index32(0), tree.leafCount());
        EXPECT_EQ(Index32(1), tree.nonLeafCount()); // root node
        EXPECT_TRUE(tree.empty());
    }

    {// Prune a tree with a single leaf node with random values in the range [0,1]
        using LeafNodeT = tree::LeafNode<float,3>;
        const float val = 1.0, tol = 1.1f;

        // Fill a leaf node with random values in the range [0,1]
        LeafNodeT *leaf = new LeafNodeT(Coord(0), val, true);
        math::Random01 r(145);
        std::vector<float> data(LeafNodeT::NUM_VALUES);
        for (Index i=0; i<LeafNodeT::NUM_VALUES; ++i) {
            const float v = float(r());
            data[i] = v;
            leaf->setValueOnly(i, v);
        }

        // Insert leaf node into an empty tree
        FloatTree tree(val);
        tree.addLeaf(leaf);

        EXPECT_EQ(Index32(1), tree.leafCount());
        EXPECT_EQ(Index32(3), tree.nonLeafCount()); // root+2*internal

        tools::prune(tree);// tolerance is zero

        EXPECT_EQ(Index32(1), tree.leafCount());
        EXPECT_EQ(Index32(3), tree.nonLeafCount()); // root+2*internal

        tools::prune(tree, tol);

        EXPECT_EQ(Index32(0), tree.leafCount());
        EXPECT_EQ(Index32(3), tree.nonLeafCount()); // root+2*internal

        std::sort(data.begin(), data.end());
        const float median = data[(LeafNodeT::NUM_VALUES-1)>>1];

        ASSERT_DOUBLES_EXACTLY_EQUAL(median, tree.getValue(Coord(0)));
    }

    /*
    {// Benchmark serial prune
        util::CpuTimer timer;
        initialize();//required whenever I/O of OpenVDB files is performed!
        io::File sourceFile("/usr/pic1/Data/OpenVDB/LevelSetModels/crawler.vdb");
        sourceFile.open(false);//disable delayed loading
        FloatGrid::Ptr grid = gridPtrCast<FloatGrid>(sourceFile.getGrids()->at(0));
        const Index32 leafCount = grid->tree().leafCount();

        timer.start("\nSerial tolerance prune");
        grid->tree().prune();
        timer.stop();
        EXPECT_EQ(leafCount, grid->tree().leafCount());
    }
    {// Benchmark parallel prune
        util::CpuTimer timer;
        initialize();//required whenever I/O of OpenVDB files is performed!
        io::File sourceFile("/usr/pic1/Data/OpenVDB/LevelSetModels/crawler.vdb");
        sourceFile.open(false);//disable delayed loading
        FloatGrid::Ptr grid = gridPtrCast<FloatGrid>(sourceFile.getGrids()->at(0));
        const Index32 leafCount = grid->tree().leafCount();

        timer.start("\nParallel tolerance prune");
        tools::prune(grid->tree());
        timer.stop();
        EXPECT_EQ(leafCount, grid->tree().leafCount());
    }
    */
}
