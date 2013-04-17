///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <sstream>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/random/mersenne_twister.hpp>
#include <tbb/atomic.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/PointAdvect.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/util/Util.h>
#include "util.h" // for unittest_util::makeSphere()

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);


class TestTools: public CppUnit::TestFixture
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestTools);
    CPPUNIT_TEST(testDilateVoxels);
    CPPUNIT_TEST(testErodeVoxels);
    CPPUNIT_TEST(testFilter);
    CPPUNIT_TEST(testFloatApply);
    CPPUNIT_TEST(testLevelSetSphere);
    CPPUNIT_TEST(testLevelSetAdvect);
    CPPUNIT_TEST(testMagnitude);
    CPPUNIT_TEST(testNormalize);
    CPPUNIT_TEST(testPointAdvect);
    CPPUNIT_TEST(testPointScatter);
    CPPUNIT_TEST(testTransformValues);
    CPPUNIT_TEST(testVectorApply);
    CPPUNIT_TEST(testUtil);

    CPPUNIT_TEST_SUITE_END();

    void testDilateVoxels();
    void testErodeVoxels();
    void testFilter();
    void testFloatApply();
    void testLevelSetSphere();
    void testLevelSetAdvect();
    void testMagnitude();
    void testNormalize();
    void testPointAdvect();
    void testPointScatter();
    void testTransformValues();
    void testVectorApply();
    void testUtil();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTools);

void
TestTools::testDilateVoxels()
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Index32;
    using openvdb::Index64;

    typedef openvdb::tree::Tree4<float, 5, 4, 3>::Type Tree543f;

    Tree543f::Ptr tree(new Tree543f);
    tree->setBackground(/*background=*/5.0);
    CPPUNIT_ASSERT(tree->empty());

    const openvdb::Index leafDim = Tree543f::LeafNodeType::DIM;
    CPPUNIT_ASSERT_EQUAL(1 << 3, int(leafDim));

    {
        // Set and dilate a single voxel at the center of a leaf node.
        tree->clear();
        tree->setValue(Coord(leafDim >> 1), 1.0);
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree->activeVoxelCount());
        openvdb::tools::dilateVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(Index64(7), tree->activeVoxelCount());
    }
    {
        // Create an active, leaf node-sized tile.
        tree->clear();
        tree->fill(CoordBBox(Coord(0), Coord(leafDim - 1)), 1.0);
        CPPUNIT_ASSERT_EQUAL(Index32(0), tree->leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(leafDim * leafDim * leafDim), tree->activeVoxelCount());

        tree->setValue(Coord(leafDim, leafDim - 1, leafDim - 1), 1.0);
        CPPUNIT_ASSERT_EQUAL(Index64(leafDim * leafDim * leafDim + 1),
                             tree->activeVoxelCount());

        openvdb::tools::dilateVoxels(*tree);

        CPPUNIT_ASSERT_EQUAL(Index64(leafDim * leafDim * leafDim + 1 + 5),
                             tree->activeVoxelCount());
    }
    {
        // Set and dilate a single voxel at each of the eight corners of a leaf node.
        for (int i = 0; i < 8; ++i) {
            tree->clear();

            openvdb::Coord xyz(
                i & 1 ? leafDim - 1 : 0,
                i & 2 ? leafDim - 1 : 0,
                i & 4 ? leafDim - 1 : 0);
            tree->setValue(xyz, 1.0);
            CPPUNIT_ASSERT_EQUAL(Index64(1), tree->activeVoxelCount());

            openvdb::tools::dilateVoxels(*tree);
            CPPUNIT_ASSERT_EQUAL(Index64(7), tree->activeVoxelCount());
        }
    }
    {
        tree->clear();
        tree->setValue(Coord(0), 1.0);
        tree->setValue(Coord( 1, 0, 0), 1.0);
        tree->setValue(Coord(-1, 0, 0), 1.0);
        CPPUNIT_ASSERT_EQUAL(Index64(3), tree->activeVoxelCount());
        openvdb::tools::dilateVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(Index64(17), tree->activeVoxelCount());
    }
    {
        struct Info { int activeVoxelCount, leafCount, nonLeafCount; };
        Info iterInfo[11] = {
            { 1,     1,  3 },
            { 7,     1,  3 },
            { 25,    1,  3 },
            { 63,    1,  3 },
            { 129,   4,  3 },
            { 231,   7,  9 },
            { 377,   7,  9 },
            { 575,   7,  9 },
            { 833,  10,  9 },
            { 1159, 16,  9 },
            { 1561, 19, 15 },
        };

        // Perform repeated dilations, starting with a single voxel.
        tree->clear();
        tree->setValue(Coord(leafDim >> 1), 1.0);
        for (int i = 0; i < 11; ++i) {
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].activeVoxelCount, int(tree->activeVoxelCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].leafCount,        int(tree->leafCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].nonLeafCount,     int(tree->nonLeafCount()));

            openvdb::tools::dilateVoxels(*tree);
        }
    }

    {// dialte a narrow band of a sphere
        typedef openvdb::Grid<Tree543f> GridType;
        GridType grid(tree->background());
        unittest_util::makeSphere<GridType>(/*dim=*/openvdb::Coord(64, 64, 64),
                                            /*center=*/openvdb::Vec3f(0, 0, 0),
                                            /*radius=*/20, grid, /*dx=*/1.0f,
                                            unittest_util::SPHERE_DENSE_NARROW_BAND);
        const openvdb::Index64 count = grid.tree().activeVoxelCount();
        openvdb::tools::dilateVoxels(grid.tree());
        CPPUNIT_ASSERT(grid.tree().activeVoxelCount() > count);
    }

    {// dilate a fog volume of a sphere
        typedef openvdb::Grid<Tree543f> GridType;
        GridType grid(tree->background());
        unittest_util::makeSphere<GridType>(/*dim=*/openvdb::Coord(64, 64, 64),
                                            /*center=*/openvdb::Vec3f(0, 0, 0),
                                            /*radius=*/20, grid, /*dx=*/1.0f,
                                            unittest_util::SPHERE_DENSE_NARROW_BAND);
        openvdb::tools::sdfToFogVolume(grid);
        const openvdb::Index64 count = grid.tree().activeVoxelCount();
        //std::cerr << "\nBefore: active voxel count = " << count << std::endl;
        //grid.print(std::cerr,5);
        openvdb::tools::dilateVoxels(grid.tree());
        CPPUNIT_ASSERT(grid.tree().activeVoxelCount() > count);
        //std::cerr << "\nAfter: active voxel count = " << grid.tree().activeVoxelCount() << std::endl;
    }
//     {// Test a grid from a file that has proven to be challenging
//         openvdb::initialize();
//         openvdb::io::File file("/usr/home/kmuseth/Data/vdb/dilation.vdb");
//         file.open();
//         openvdb::GridBase::Ptr baseGrid = file.readGrid(file.beginName().gridName());
//         file.close();
//         openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
//         const openvdb::Index64 count = grid->tree().activeVoxelCount();
//         //std::cerr << "\nBefore: active voxel count = " << count << std::endl;
//         //grid->print(std::cerr,5);
//         openvdb::tools::dilateVoxels(grid->tree());
//         CPPUNIT_ASSERT(grid->tree().activeVoxelCount() > count);
//         //std::cerr << "\nAfter: active voxel count = " << grid->tree().activeVoxelCount() << std::endl;
//     }
}

void
TestTools::testErodeVoxels()
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Index32;
    using openvdb::Index64;

    typedef openvdb::tree::Tree4<float, 5, 4, 3>::Type TreeType;

    TreeType::Ptr tree(new TreeType);
    tree->setBackground(/*background=*/5.0);
    CPPUNIT_ASSERT(tree->empty());

    const int leafDim = TreeType::LeafNodeType::DIM;
    CPPUNIT_ASSERT_EQUAL(1 << 3, leafDim);

    {
        // Set, dilate and erode a single voxel at the center of a leaf node.
        tree->clear();
        CPPUNIT_ASSERT_EQUAL(0, int(tree->activeVoxelCount()));

        tree->setValue(Coord(leafDim >> 1), 1.0);
        CPPUNIT_ASSERT_EQUAL(1, int(tree->activeVoxelCount()));

        openvdb::tools::dilateVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(7, int(tree->activeVoxelCount()));

        openvdb::tools::erodeVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(1, int(tree->activeVoxelCount()));

        openvdb::tools::erodeVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(0, int(tree->activeVoxelCount()));
    }
    {
        // Create an active, leaf node-sized tile.
        tree->clear();
        tree->fill(CoordBBox(Coord(0), Coord(leafDim - 1)), 1.0);
        CPPUNIT_ASSERT_EQUAL(0, int(tree->leafCount()));
        CPPUNIT_ASSERT_EQUAL(leafDim * leafDim * leafDim, int(tree->activeVoxelCount()));

        tree->setValue(Coord(leafDim, leafDim - 1, leafDim - 1), 1.0);
        CPPUNIT_ASSERT_EQUAL(1, int(tree->leafCount()));
        CPPUNIT_ASSERT_EQUAL(leafDim * leafDim * leafDim + 1,int(tree->activeVoxelCount()));

        openvdb::tools::dilateVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(3, int(tree->leafCount()));
        CPPUNIT_ASSERT_EQUAL(leafDim * leafDim * leafDim + 1 + 5,int(tree->activeVoxelCount()));

        openvdb::tools::erodeVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(1, int(tree->leafCount()));
        CPPUNIT_ASSERT_EQUAL(leafDim * leafDim * leafDim + 1, int(tree->activeVoxelCount()));
    }
    {
        // Set and dilate a single voxel at each of the eight corners of a leaf node.
        for (int i = 0; i < 8; ++i) {
            tree->clear();

            openvdb::Coord xyz(
                i & 1 ? leafDim - 1 : 0,
                i & 2 ? leafDim - 1 : 0,
                i & 4 ? leafDim - 1 : 0);
            tree->setValue(xyz, 1.0);
            CPPUNIT_ASSERT_EQUAL(1, int(tree->activeVoxelCount()));

            openvdb::tools::dilateVoxels(*tree);
            CPPUNIT_ASSERT_EQUAL(7, int(tree->activeVoxelCount()));

            openvdb::tools::erodeVoxels(*tree);
            CPPUNIT_ASSERT_EQUAL(1, int(tree->activeVoxelCount()));
        }
    }
    {
        // Set three active voxels and dilate and erode
        tree->clear();
        tree->setValue(Coord(0), 1.0);
        tree->setValue(Coord( 1, 0, 0), 1.0);
        tree->setValue(Coord(-1, 0, 0), 1.0);
        CPPUNIT_ASSERT_EQUAL(3, int(tree->activeVoxelCount()));

        openvdb::tools::dilateVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(17, int(tree->activeVoxelCount()));

        openvdb::tools::erodeVoxels(*tree);
        CPPUNIT_ASSERT_EQUAL(3, int(tree->activeVoxelCount()));
    }
    {
        struct Info {
            void test(TreeType::Ptr tree) {
                CPPUNIT_ASSERT_EQUAL(activeVoxelCount, int(tree->activeVoxelCount()));
                CPPUNIT_ASSERT_EQUAL(leafCount,        int(tree->leafCount()));
                CPPUNIT_ASSERT_EQUAL(nonLeafCount,     int(tree->nonLeafCount()));
            }
            int activeVoxelCount, leafCount, nonLeafCount;
        };
        Info iterInfo[12] = {
            { 0,     0,  1 },//an empty tree only contains a root node
            { 1,     1,  3 },
            { 7,     1,  3 },
            { 25,    1,  3 },
            { 63,    1,  3 },
            { 129,   4,  3 },
            { 231,   7,  9 },
            { 377,   7,  9 },
            { 575,   7,  9 },
            { 833,  10,  9 },
            { 1159, 16,  9 },
            { 1561, 19, 15 },
        };

        // Perform repeated dilations, starting with a single voxel.
        tree->clear();
        iterInfo[0].test(tree);

        tree->setValue(Coord(leafDim >> 1), 1.0);
        iterInfo[1].test(tree);

        for (int i = 2; i < 12; ++i) {
            openvdb::tools::dilateVoxels(*tree);
            iterInfo[i].test(tree);
        }
        for (int i = 10; i >= 0; --i) {
            openvdb::tools::erodeVoxels(*tree);
            iterInfo[i].test(tree);
        }

        // Now try it using the resursive calls
        for (int i = 2; i < 12; ++i) {
            tree->clear();
            tree->setValue(Coord(leafDim >> 1), 1.0);
            openvdb::tools::dilateVoxels(*tree, i-1);
            iterInfo[i].test(tree);
        }
        for (int i = 10; i >= 0; --i) {
            tree->clear();
            tree->setValue(Coord(leafDim >> 1), 1.0);
            openvdb::tools::dilateVoxels(*tree, 10);
            openvdb::tools::erodeVoxels(*tree, 11-i);
            iterInfo[i].test(tree);
        }
    }

    {// erode a narrow band of a sphere
        typedef openvdb::Grid<TreeType> GridType;
        GridType grid(tree->background());
        unittest_util::makeSphere<GridType>(/*dim=*/openvdb::Coord(64, 64, 64),
                                            /*center=*/openvdb::Vec3f(0, 0, 0),
                                            /*radius=*/20, grid, /*dx=*/1.0f,
                                            unittest_util::SPHERE_DENSE_NARROW_BAND);
        const openvdb::Index64 count = grid.tree().activeVoxelCount();
        openvdb::tools::erodeVoxels(grid.tree());
        CPPUNIT_ASSERT(grid.tree().activeVoxelCount() < count);
    }

    {// erode a fog volume of a sphere
        typedef openvdb::Grid<TreeType> GridType;
        GridType grid(tree->background());
        unittest_util::makeSphere<GridType>(/*dim=*/openvdb::Coord(64, 64, 64),
                                            /*center=*/openvdb::Vec3f(0, 0, 0),
                                            /*radius=*/20, grid, /*dx=*/1.0f,
                                            unittest_util::SPHERE_DENSE_NARROW_BAND);
        openvdb::tools::sdfToFogVolume(grid);
        const openvdb::Index64 count = grid.tree().activeVoxelCount();
        openvdb::tools::erodeVoxels(grid.tree());
        CPPUNIT_ASSERT(grid.tree().activeVoxelCount() < count);
    }
}

void
TestTools::testFilter()
{
    openvdb::FloatGrid::Ptr referenceGrid = openvdb::FloatGrid::create(/*background=*/5.0);

    const openvdb::Coord dim(40);
    const openvdb::Vec3f center(25.0f, 20.0f, 20.0f);
    const float radius = 10.0f;
    unittest_util::makeSphere<openvdb::FloatGrid>(
        dim, center, radius, *referenceGrid, unittest_util::SPHERE_DENSE);
    const openvdb::FloatTree& sphere = referenceGrid->tree();

    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(sphere.activeVoxelCount()));
    openvdb::Coord xyz;

    {// test Filter::offsetFilter
        openvdb::FloatGrid::Ptr grid = referenceGrid->deepCopy();
        openvdb::FloatTree& tree = grid->tree();
        openvdb::tools::Filter<openvdb::FloatGrid> filter(*grid);
        const float offset = 2.34f;
        filter.offset(offset, true/*=serial*/);
        for (int x=0; x<dim[0]; ++x) {
            xyz[0]=x;
            for (int y=0; y<dim[1]; ++y) {
                xyz[1]=y;
                for (int z=0; z<dim[2]; ++z) {
                    xyz[2]=z;
                    float delta = sphere.getValue(xyz) + offset - tree.getValue(xyz);
                    //if (fabs(delta)>0.0001f) std::cerr << " failed at " << xyz << std::endl;
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, delta, /*tolerance=*/0.0001);
                }
            }
        }
        filter.offset(-offset);//default is multi-threaded
        for (int x=0; x<dim[0]; ++x) {
            xyz[0]=x;
            for (int y=0; y<dim[1]; ++y) {
                xyz[1]=y;
                for (int z=0; z<dim[2]; ++z) {
                    xyz[2]=z;
                    float delta = sphere.getValue(xyz) - tree.getValue(xyz);
                    //if (fabs(delta)>0.0001f) std::cerr << " failed at " << xyz << std::endl;
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, delta, /*tolerance=*/0.0001);
                }
            }
        }
        //std::cerr << "Successfully completed TestTools::testFilter offset test" << std::endl;
    }
    {// test Filter::median
        openvdb::FloatGrid::Ptr filteredGrid = referenceGrid->deepCopy();
        openvdb::FloatTree& filteredTree = filteredGrid->tree();
        const int width = 2;
        openvdb::math::DenseStencil<openvdb::FloatGrid> stencil(*referenceGrid, width);
        openvdb::tools::Filter<openvdb::FloatGrid> filter(*filteredGrid);
        filter.median(width, /*interations=*/1, false/*=serial*/);
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
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        tmp[(tmp.size()-1)/2], stencil.median(), /*tolerance=*/0.0001);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        stencil.median(), filteredTree.getValue(xyz), /*tolerance=*/0.0001);
                    tmp.clear();
                }
            }
        }
        //std::cerr << "Successfully completed TestTools::testFilter median test" << std::endl;
    }
    {// test Filter::mean
        openvdb::FloatGrid::Ptr filteredGrid = referenceGrid->deepCopy();
        openvdb::FloatTree& filteredTree = filteredGrid->tree();
        const int width = 2;
        openvdb::math::DenseStencil<openvdb::FloatGrid> stencil(*referenceGrid, width);
        openvdb::tools::Filter<openvdb::FloatGrid> filter(*filteredGrid);
        filter.mean(width,  /*interations=*/1, false/*=serial*/);
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
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        sum/count, stencil.mean(), /*tolerance=*/0.0001);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        stencil.mean(), filteredTree.getValue(xyz), 0.0001);
                }
            }
        }
        //std::cerr << "Successfully completed TestTools::testFilter mean test" << std::endl;
    }
}

void
TestTools::testLevelSetSphere()
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
                const openvdb::Vec3f p(voxelSize*i,voxelSize*j,voxelSize*k);
                const float dist = (p-center).length() - radius;
                const float val1 = grid1->tree().getValue(openvdb::Coord(i,j,k));
                const float val2 = grid2->tree().getValue(openvdb::Coord(i,j,k));
                if (dist > outside) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( outside, val1, 0.0001);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( outside, val2, 0.0001);
                } else if (dist < inside) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( inside, val1, 0.0001);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( inside, val2, 0.0001);
                } else {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(  dist, val1, 0.0001);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(  dist, val2, 0.0001);
                }
            }
        }
    }

    CPPUNIT_ASSERT_EQUAL(grid1->activeVoxelCount(), grid2->activeVoxelCount());
}

void
TestTools::testLevelSetAdvect()
{
    // Uncomment sections below to run this (time-consuming) test
    /*
    const int dim = 64;//256
    const openvdb::Vec3f center(0.35f, 0.35f, 0.35f);
    const float radius = 0.15f, voxelSize = 1.0f/(dim-1);

    typedef openvdb::FloatGrid GridT;
    typedef openvdb::Vec3fGrid VectT;

    class FrameWriter {
    public:
        FrameWriter(int dim, GridT::Ptr grid) : mFrame(0), mDim(dim), mGrid(grid) {}
        void operator()(const char* name, float time, size_t n) {
            std::ostringstream ostr;
            ostr << "/usr/pic1/tmp/" << name << "_" << mDim << "_" << mFrame << ".vdb";
            openvdb::io::File file(ostr.str());
            openvdb::GridPtrVec grids;
            grids.push_back(mGrid);
            file.write(grids);
            std::cerr << "\nWrote \"" << ostr.str() << "\" with time = "
                      << time << " after CFL-iterations = " << n << std::endl;
            ++mFrame;
        }
    private:
        int mFrame, mDim;
        GridT::Ptr mGrid;
        };
    */
    /*
    {//test tracker
        GridT::Ptr grid = openvdb::tools::createLevelSetSphere<GridT>(radius, center, voxelSize);
        typedef openvdb::tools::LevelSetTracker<GridT>  TrackerT;
        TrackerT tracker(*grid);
        tracker.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        tracker.setTemporalScheme(openvdb::math::TVD_RK1);

        FrameWriter fw(dim, grid); fw("Tracker",0, 0);
        //for (float t = 0, dt = 0.005f; !grid->empty() && t < 3.0f; t += dt) {
        //    fw("Enright", t + dt, advect.advect(t, t + dt));
        //}
        for (float t = 0, dt = 0.5f; !grid->empty() && t < 1.0f; t += dt) {
            tracker.track();
            fw("Tracker", 0, 0);
        }
        }
    */
    /*
    {//test EnrightField
        GridT::Ptr grid = openvdb::tools::createLevelSetSphere<GridT>(radius, center, voxelSize);
        typedef openvdb::tools::EnrightField<float> FieldT;
        FieldT field;

        typedef openvdb::tools::LevelSetAdvection<GridT, FieldT>  AdvectT;
        AdvectT advect(*grid, field);
        advect.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTemporalScheme(openvdb::math::TVD_RK2);
        advect.setTrackerSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTrackerTemporalScheme(openvdb::math::TVD_RK1);

        FrameWriter fw(dim, grid); fw("Enright",0, 0);
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
        typedef openvdb::tools::DiscreteField<VectT> FieldT;
        FieldT field(vect);
        typedef openvdb::tools::LevelSetAdvection<GridT, FieldT>  AdvectT;
        AdvectT advect(*grid, field);
        advect.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTemporalScheme(openvdb::math::TVD_RK2);

        FrameWriter fw(dim, grid); fw("Aligned",0, 0);
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
        typedef openvdb::tools::DiscreteField<VectT> FieldT;
        FieldT field(vect);
        typedef openvdb::tools::LevelSetAdvection<GridT, FieldT>  AdvectT;
        AdvectT advect(*grid, field);
        advect.setSpatialScheme(openvdb::math::HJWENO5_BIAS);
        advect.setTemporalScheme(openvdb::math::TVD_RK2);

        FrameWriter fw(dim, grid); fw("Xformed",0, 0);
        //for (float t = 0, dt = 0.005f; !grid->empty() && t < 3.0f; t += dt) {
        //    fw("Xformed", t + dt, advect.advect(t, t + dt));
        //}
        for (float t = 0, dt = 0.5f; !grid->empty() && t < 1.0f; t += dt) {
            fw("Xformed", t + dt, advect.advect(t, t + dt));
        }
        }
    */
}

void
TestTools::testFloatApply()
{
    typedef openvdb::FloatTree::ValueOnIter ValueIter;

    struct Local {
        static inline float op(float x) { return x * 2.0; }
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
                tree.setValue(xyz, x + y + z);
            }
        }
    }
    /// @todo set some tile values

    openvdb::tools::foreach(tree.begin<ValueIter>(), Local::visit, /*threaded=*/true);

    float expected = Local::op(background);
    //CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, tree.background(), /*tolerance=*/0.0);
    //expected = Local::op(-background);
    //CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, -tree.background(), /*tolerance=*/0.0);

    for (openvdb::FloatTree::ValueOnCIter it = tree.cbeginValueOn(); it; ++it) {
        xyz = it.getCoord();
        expected = Local::op(xyz[0] + xyz[1] + xyz[2]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, it.getValue(), /*tolerance=*/0.0);
    }
}



void
TestTools::testMagnitude()
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(/*background=*/5.0);
    openvdb::FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=0.0f;
    unittest_util::makeSphere<openvdb::FloatGrid>(dim,center,radius,*grid,
                                                  unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    openvdb::VectorGrid::Ptr gradGrid = openvdb::tools::gradient(*grid);
    CPPUNIT_ASSERT_EQUAL(int(tree.activeVoxelCount()), int(gradGrid->activeVoxelCount()));

    openvdb::FloatGrid::Ptr mag = openvdb::tools::magnitude(*gradGrid);
    CPPUNIT_ASSERT_EQUAL(int(tree.activeVoxelCount()), int(mag->activeVoxelCount()));

    openvdb::FloatGrid::ConstAccessor accessor = mag->getConstAccessor();

    openvdb::Coord xyz(35,30,30);
    float v = accessor.getValue(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, v, 0.01);

    xyz.reset(35,10,40);
    v = accessor.getValue(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, v, 0.01);
}


void
TestTools::testNormalize()
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(5.0);
    openvdb::FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=10.0f;
    unittest_util::makeSphere<openvdb::FloatGrid>(
        dim,center,radius,*grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));
    openvdb::Coord xyz(10, 20, 30);

    openvdb::VectorGrid::Ptr grad = openvdb::tools::gradient(*grid);

    typedef openvdb::VectorGrid::ValueType Vec3Type;

    typedef openvdb::VectorGrid::ValueOnIter ValueIter;

    struct Local {
        static inline Vec3Type op(const Vec3Type &x) { return x * 2.0f; }
        static inline void visit(const ValueIter& it) { it.setValue(op(*it)); }
    };

    openvdb::tools::foreach(grad->beginValueOn(), Local::visit, true);

    openvdb::VectorGrid::ConstAccessor accessor = grad->getConstAccessor();

    xyz = openvdb::Coord(35,10,40);
    Vec3Type v = accessor.getValue(xyz);
    //std::cerr << "\nPassed testNormalize(" << xyz << ")=" << v.length() << std::endl;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0,v.length(),0.001);
    openvdb::VectorGrid::Ptr norm = openvdb::tools::normalize(*grad);

    accessor = norm->getConstAccessor();
    v = accessor.getValue(xyz);
    //std::cerr << "\nPassed testNormalize(" << xyz << ")=" << v.length() << std::endl;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, v.length(), 0.0001);
}

void
TestTools::testPointAdvect()
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
        for (size_t i = 0; i < numPoints; i++) pointList[i] = openvdb::Vec3f(i, i, i);

        for (unsigned int order = 1; order < 5; ++order) {
            // check all four time integrations schemes
            // construct an advection tool.  By default the number of cpt iterations is zero
            advectionTool.setIntegrationOrder(order);
            advectionTool.advect(pointList, /*dt=*/1.0, /*iterations=*/1);

            // check locations
            for (size_t i = 0; i < numPoints; i++) {
                openvdb::Vec3f expected(i + 1, i + 1 , i + 1);
                CPPUNIT_ASSERT_EQUAL(expected, pointList[i]);
            }
            // reset values
            for (size_t i = 0; i < numPoints; i++) pointList[i] = openvdb::Vec3f(i, i, i);
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
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[n], pointList[i][n], /*tolerance=*/1e-6);
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
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[n], pointList[i][n], /*tolerance=*/1e-6);
                }
            }
            // reset values
            for (unsigned int i = 0; i < numPoints; i++) pointList[i] = openvdb::Vec3d(i, i, i);
        }
    }
}

////////////////////////////////////////

struct PointList
{
    struct Point { float x,y,z; };
    std::vector<Point> list;
    void add(const openvdb::Vec3R &p) { Point q={p[0],p[1],p[2]}; list.push_back(q); }
};

void
TestTools::testPointScatter()
{
    typedef openvdb::FloatGrid GridType;
    const openvdb::Coord dim(64, 64, 64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius = 20.0;
    typedef boost::mt11213b RandGen;
    RandGen mtRand;

    GridType::Ptr grid = GridType::create(/*background=*/2.0);
    unittest_util::makeSphere<GridType>(dim, center, radius, *grid,
                                        unittest_util::SPHERE_DENSE_NARROW_BAND);

    {
        const int pointCount = 1000;
        PointList points;
        openvdb::tools::UniformPointScatter<PointList, RandGen> scatter(points, pointCount, mtRand);
        scatter.operator()<GridType>(*grid);
        CPPUNIT_ASSERT_EQUAL( pointCount, scatter.getPointCount() );
        CPPUNIT_ASSERT_EQUAL( pointCount, int(points.list.size()) );
    }
    {
        const float density = 1.0f;//per volume = per voxel since voxel size = 1
        PointList points;
        openvdb::tools::UniformPointScatter<PointList, RandGen> scatter(points, density, mtRand);
        scatter.operator()<GridType>(*grid);
        CPPUNIT_ASSERT_EQUAL( int(scatter.getVoxelCount()), scatter.getPointCount() );
        CPPUNIT_ASSERT_EQUAL( int(scatter.getVoxelCount()), int(points.list.size()) );
    }
    {
        const float density = 1.0f;//per volume = per voxel since voxel size = 1
        PointList points;
        openvdb::tools::NonUniformPointScatter<PointList, RandGen> scatter(points, density, mtRand);
        scatter.operator()<GridType>(*grid);
        CPPUNIT_ASSERT( int(scatter.getVoxelCount()) < scatter.getPointCount() );
        CPPUNIT_ASSERT_EQUAL( scatter.getPointCount(), int(points.list.size()) );
    }
}

////////////////////////////////////////


namespace {

template<typename IterT>
struct MatMul {
    openvdb::math::Mat3s M;
    MatMul(const openvdb::math::Mat3s& M): M(M) {}
    openvdb::Vec3s xform(const openvdb::Vec3s& v) const { return M.transform(v); }
    void operator()(const IterT& it) const { it.setValue(xform(*it)); }
};

}


void
TestTools::testVectorApply()
{
    typedef openvdb::VectorTree::ValueOnIter ValueIter;

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
                tree.setValue(xyz, openvdb::Vec3s(x, y, z));
            }
        }
    }
    /// @todo set some tile values

    MatMul<ValueIter> op(openvdb::math::Mat3s(1, 2, 3, -1, -2, -3, 3, 2, 1));
    openvdb::tools::foreach(tree.beginValueOn(), op, /*threaded=*/true);

    openvdb::Vec3s expected;
    for (openvdb::VectorTree::ValueOnCIter it = tree.cbeginValueOn(); it; ++it) {
        xyz = it.getCoord();
        expected = op.xform(openvdb::Vec3s(xyz[0], xyz[1], xyz[2]));
        CPPUNIT_ASSERT_EQUAL(expected, it.getValue());
    }
}


////////////////////////////////////////


namespace {

template<typename InIterT, typename OutTreeT>
struct FloatToVec {
    typedef typename InIterT::ValueT ValueT;
    typedef typename openvdb::tree::ValueAccessor<OutTreeT> Accessor;

    // Transform a scalar value into a vector value.
    static openvdb::Vec3s toVec(const ValueT& v) { return openvdb::Vec3s(v, v*2, v*3); }

    FloatToVec() { numTiles = 0; }

    void operator()(const InIterT& it, Accessor& acc)
    {
        if (it.isVoxelValue()) { // set a single voxel
            acc.setValue(it.getCoord(), toVec(*it));
        } else { // fill an entire tile
            numTiles.fetch_and_increment();
            openvdb::CoordBBox bbox;
            it.getBoundingBox(bbox);
            if (acc.getTree()) acc.getTree()->fill(bbox, toVec(*it));
        }
    }

    tbb::atomic<int> numTiles;
};

}


void
TestTools::testTransformValues()
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Vec3s;

    typedef openvdb::tree::Tree4<float, 3, 2, 3>::Type Tree323f;
    typedef openvdb::tree::Tree4<Vec3s, 3, 2, 3>::Type Tree323v;

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
                ftree.setValue(xyz, x + y + z);
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
        CPPUNIT_ASSERT_EQUAL(shareOp ? 3 : 0, int(op.numTiles));

        Vec3s expected;
        for (Tree323v::ValueOnCIter it = vtree.cbeginValueOn(); it; ++it) {
            xyz = it.getCoord();
            expected = op.toVec(xyz[0] + xyz[1] + xyz[2]);
            CPPUNIT_ASSERT_EQUAL(expected, it.getValue());
        }
        // Check values inside the tiles.
        CPPUNIT_ASSERT_EQUAL(op.toVec(3 * 1024), vtree.getValue(Coord(1024 + 4)));
        CPPUNIT_ASSERT_EQUAL(op.toVec(3 * 2048), vtree.getValue(Coord(2048 + 16)));
        CPPUNIT_ASSERT_EQUAL(op.toVec(3 * 3072), vtree.getValue(Coord(3072 + 128)));
    }
}

void
TestTools::testUtil()
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Vec3s;

    typedef openvdb::tree::Tree4<bool, 3, 2, 3>::Type CharTree;


    // Test boolean operators
    CharTree treeA(false), treeB(false);

    treeA.fill(CoordBBox(Coord(-10), Coord(10)), true);
    treeA.voxelizeActiveTiles();

    treeB.fill(CoordBBox(Coord(-10), Coord(10)), true);
    treeB.voxelizeActiveTiles();
    
    const size_t voxelCountA = treeA.activeVoxelCount();
    const size_t voxelCountB = treeB.activeVoxelCount();

    CPPUNIT_ASSERT_EQUAL(voxelCountA, voxelCountB);
    
    CharTree::Ptr tree = openvdb::util::leafTopologyDifference(treeA, treeB);
    CPPUNIT_ASSERT(tree->activeVoxelCount() == 0);
    
    tree = openvdb::util::leafTopologyIntersection(treeA, treeB);
    CPPUNIT_ASSERT(tree->activeVoxelCount() == voxelCountA);
    
    treeA.fill(CoordBBox(Coord(-10), Coord(22)), true);
    treeA.voxelizeActiveTiles();
    
    const size_t voxelCount = treeA.activeVoxelCount();
    
    tree = openvdb::util::leafTopologyDifference(treeA, treeB);
    CPPUNIT_ASSERT(tree->activeVoxelCount() == (voxelCount - voxelCountA));
    
    tree = openvdb::util::leafTopologyIntersection(treeA, treeB);
    CPPUNIT_ASSERT(tree->activeVoxelCount() == voxelCountA);
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
