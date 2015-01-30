///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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

#include <vector>
#include <cppunit/extensions/HelperMacros.h>

#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/util/Util.h>


class TestMeshToVolume: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMeshToVolume);
    CPPUNIT_TEST(testUtils);
    CPPUNIT_TEST(testVoxelizer);
    CPPUNIT_TEST(testPrimitiveVoxelRatio);
    CPPUNIT_TEST(testIntersectingVoxelCleaner);
    CPPUNIT_TEST(testShellVoxelCleaner);
    CPPUNIT_TEST(testConversion);
    CPPUNIT_TEST_SUITE_END();

    void testUtils();
    void testVoxelizer();
    void testPrimitiveVoxelRatio();
    void testIntersectingVoxelCleaner();
    void testShellVoxelCleaner();
    void testConversion();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMeshToVolume);


////////////////////////////////////////


void
TestMeshToVolume::testUtils()
{
    /// Test nearestCoord
    openvdb::Vec3d xyz(0.7, 2.2, -2.7);
    openvdb::Coord ijk = openvdb::util::nearestCoord(xyz);
    CPPUNIT_ASSERT(ijk[0] == 0 && ijk[1] == 2 && ijk[2] == -3);

    xyz = openvdb::Vec3d(-22.1, 4.6, 202.34);
    ijk = openvdb::util::nearestCoord(xyz);
    CPPUNIT_ASSERT(ijk[0] == -23 && ijk[1] == 4 && ijk[2] == 202);

    /// Test the coordinate offset table for neghbouring voxels
    openvdb::Coord sum(0, 0, 0);

    unsigned int pX = 0, pY = 0, pZ = 0, mX = 0, mY = 0, mZ = 0;

    for (unsigned int i = 0; i < 26; ++i) {
        ijk = openvdb::util::COORD_OFFSETS[i];
        sum += ijk;

        if (ijk[0] == 1)       ++pX;
        else if (ijk[0] == -1) ++mX;

        if (ijk[1] == 1)       ++pY;
        else if (ijk[1] == -1) ++mY;

        if (ijk[2] == 1)       ++pZ;
        else if (ijk[2] == -1) ++mZ;
    }

    CPPUNIT_ASSERT(sum == openvdb::Coord(0, 0, 0));

    CPPUNIT_ASSERT( pX == 9);
    CPPUNIT_ASSERT( pY == 9);
    CPPUNIT_ASSERT( pZ == 9);
    CPPUNIT_ASSERT( mX == 9);
    CPPUNIT_ASSERT( mY == 9);
    CPPUNIT_ASSERT( mZ == 9);
}


void
TestMeshToVolume::testVoxelizer()
{
    std::vector<openvdb::Vec3s> pointList;
    std::vector<openvdb::Vec4I> polygonList;

    typedef openvdb::tools::internal::MeshVoxelizer<openvdb::FloatTree> MeshVoxelizer;

    // CASE 1: One triangle

    pointList.push_back(openvdb::Vec3s(0.0, 0.0, 0.0));
    pointList.push_back(openvdb::Vec3s(0.0, 0.0, 3.0));
    pointList.push_back(openvdb::Vec3s(0.0, 3.0, 0.0));

    polygonList.push_back(openvdb::Vec4I(0, 1, 2, openvdb::util::INVALID_IDX));

    {
        MeshVoxelizer voxelizer(pointList, polygonList);
        voxelizer.run();

        // Check for mesh intersecting voxels
        CPPUNIT_ASSERT(13== voxelizer.intersectionTree().activeVoxelCount());

        // topologically unique voxels.
        CPPUNIT_ASSERT(99 == voxelizer.sqrDistTree().activeVoxelCount());
        CPPUNIT_ASSERT(99 == voxelizer.primIndexTree().activeVoxelCount());
    }

    // CASE 2: Two triangles

    pointList.push_back(openvdb::Vec3s(0.0, 3.0, 3.0));
    polygonList.push_back(openvdb::Vec4I(1, 3, 2, openvdb::util::INVALID_IDX));

    {
        MeshVoxelizer voxelizer(pointList, polygonList);
        voxelizer.run();

        // Check for mesh intersecting voxels
        CPPUNIT_ASSERT(16 == voxelizer.intersectionTree().activeVoxelCount());

        // topologically unique voxels.
        CPPUNIT_ASSERT(108 == voxelizer.sqrDistTree().activeVoxelCount());
        CPPUNIT_ASSERT(108 == voxelizer.primIndexTree().activeVoxelCount());
    }

    // CASE 3: One quad

    polygonList.clear();
    polygonList.push_back(openvdb::Vec4I(0, 1, 3, 2));

    {
        MeshVoxelizer voxelizer(pointList, polygonList);
        voxelizer.run();

        // Check for mesh intersecting voxels
        CPPUNIT_ASSERT(16 == voxelizer.intersectionTree().activeVoxelCount());

        // topologically unique voxels.
        CPPUNIT_ASSERT(108 == voxelizer.sqrDistTree().activeVoxelCount());
        CPPUNIT_ASSERT(108 == voxelizer.primIndexTree().activeVoxelCount());
    }

    // CASE 4: Two triangles and one quad

    pointList.push_back(openvdb::Vec3s(0.0, 0.0, 6.0));
    pointList.push_back(openvdb::Vec3s(0.0, 3.0, 6.0));

    polygonList.clear();
    polygonList.push_back(openvdb::Vec4I(0, 1, 2, openvdb::util::INVALID_IDX));
    polygonList.push_back(openvdb::Vec4I(1, 3, 2, openvdb::util::INVALID_IDX));
    polygonList.push_back(openvdb::Vec4I(1, 4, 5, 3));

    {
        MeshVoxelizer voxelizer(pointList, polygonList);
        voxelizer.run();

        // Check for 28 mesh intersecting voxels
        CPPUNIT_ASSERT(28 == voxelizer.intersectionTree().activeVoxelCount());

        // 154 topologically unique voxels.
        CPPUNIT_ASSERT(162 == voxelizer.sqrDistTree().activeVoxelCount());
        CPPUNIT_ASSERT(162 == voxelizer.primIndexTree().activeVoxelCount());
    }
}


void
TestMeshToVolume::testPrimitiveVoxelRatio()
{
    std::vector<openvdb::Vec3s> pointList;
    std::vector<openvdb::Vec4I> polygonList;

    // Create one big triangle
    pointList.push_back(openvdb::Vec3s(0.0, 0.0, 0.0));
    pointList.push_back(openvdb::Vec3s(0.0, 0.0, 250.0));
    pointList.push_back(openvdb::Vec3s(0.0, 100.0, 0.0));

    polygonList.push_back(openvdb::Vec4I(0, 1, 2, openvdb::util::INVALID_IDX));

    openvdb::tools::internal::MeshVoxelizer<openvdb::FloatTree>
        voxelizer(pointList, polygonList);

    voxelizer.run();

    CPPUNIT_ASSERT(0 != voxelizer.intersectionTree().activeVoxelCount());
}


void
TestMeshToVolume::testIntersectingVoxelCleaner()
{
    // Empty tree's

    openvdb::FloatTree distTree(std::numeric_limits<float>::max());
    openvdb::BoolTree intersectionTree(false);
    openvdb::Int32Tree indexTree(openvdb::util::INVALID_IDX);

    openvdb::tree::ValueAccessor<openvdb::FloatTree> distAcc(distTree);
    openvdb::tree::ValueAccessor<openvdb::BoolTree> intersectionAcc(intersectionTree);
    openvdb::tree::ValueAccessor<openvdb::Int32Tree> indexAcc(indexTree);

    // Add a row of intersecting voxels surrounded by both positive and negative distance values.
    for (int i = 0; i < 10; ++i) {
        for (int j = -1; j < 2; ++j) {
            distAcc.setValue(openvdb::Coord(i,j,0), (float)j);
            indexAcc.setValue(openvdb::Coord(i,j,0), 10);
        }
        intersectionAcc.setValue(openvdb::Coord(i,0,0), 1);
    }

    openvdb::Index64
        numSDFVoxels = distTree.activeVoxelCount(),
        numIVoxels = intersectionTree.activeVoxelCount(),
        numCPVoxels = indexTree.activeVoxelCount();

    {
        openvdb::tree::LeafManager<openvdb::BoolTree> leafs(intersectionTree);

        openvdb::tools::internal::IntersectingVoxelCleaner<openvdb::FloatTree>
            cleaner(distTree, indexTree, intersectionTree, leafs);

        cleaner.run();
    }

    CPPUNIT_ASSERT_EQUAL(numSDFVoxels, distTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(numIVoxels, intersectionTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(numCPVoxels, indexTree.activeVoxelCount());


    // Add a row of intersecting voxels that are not surrounded by any positive distance values.
    for (int i = 0; i < 10; ++i) {
        for (int j = -1; j < 2; ++j) {
            distAcc.setValue(openvdb::Coord(i,j,0), -1.0);
            indexAcc.setValue(openvdb::Coord(i,j,0), 10);
        }
        intersectionAcc.setValue(openvdb::Coord(i,0,0), 1);
    }

    numIVoxels = 0;

    {
        openvdb::tree::LeafManager<openvdb::BoolTree> leafs(intersectionTree);

        openvdb::tools::internal::IntersectingVoxelCleaner<openvdb::FloatTree>
            cleaner(distTree, indexTree, intersectionTree, leafs);

        cleaner.run();
    }

    CPPUNIT_ASSERT_EQUAL(numSDFVoxels, distTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(numIVoxels, intersectionTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(numCPVoxels, indexTree.activeVoxelCount());
}


void
TestMeshToVolume::testShellVoxelCleaner()
{
    // Empty tree's

    openvdb::FloatTree distTree(std::numeric_limits<float>::max());
    openvdb::BoolTree intersectionTree(false);
    openvdb::Int32Tree indexTree(openvdb::util::INVALID_IDX);

    openvdb::tree::ValueAccessor<openvdb::FloatTree> distAcc(distTree);
    openvdb::tree::ValueAccessor<openvdb::BoolTree> intersectionAcc(intersectionTree);
    openvdb::tree::ValueAccessor<openvdb::Int32Tree> indexAcc(indexTree);

    /// Add a row of intersecting voxels surrounded by negative distance values.
    for (int i = 0; i < 10; ++i) {
        for (int j = -1; j < 2; ++j) {
            distAcc.setValue(openvdb::Coord(i,j,0), -1.0);
            indexAcc.setValue(openvdb::Coord(i,j,0), 10);
        }
        intersectionAcc.setValue(openvdb::Coord(i,0,0), 1);
    }

    openvdb::Index64
        numSDFVoxels = distTree.activeVoxelCount(),
        numIVoxels = intersectionTree.activeVoxelCount(),
        numCPVoxels = indexTree.activeVoxelCount();

    {
        openvdb::tree::LeafManager<openvdb::FloatTree> leafs(distTree);

        openvdb::tools::internal::ShellVoxelCleaner<openvdb::FloatTree>
            cleaner(distTree, leafs, indexTree, intersectionTree);

        cleaner.run();
    }

    CPPUNIT_ASSERT_EQUAL(numSDFVoxels, distTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(numIVoxels, intersectionTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(numCPVoxels, indexTree.activeVoxelCount());

    intersectionTree.clear();

    {
        openvdb::tree::LeafManager<openvdb::FloatTree> leafs(distTree);

        openvdb::tools::internal::ShellVoxelCleaner<openvdb::FloatTree>
            cleaner(distTree, leafs, indexTree, intersectionTree);

        cleaner.run();
    }

    const openvdb::Index64 zero(0);
    CPPUNIT_ASSERT_EQUAL(zero, distTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(zero, intersectionTree.activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(zero, indexTree.activeVoxelCount());;
}


void
TestMeshToVolume::testConversion()
{
    using namespace openvdb;

    std::vector<Vec3s> points;
    std::vector<Vec4I> quads;

    // cube vertices
    points.push_back(Vec3s(2, 2, 2)); // 0       6--------7
    points.push_back(Vec3s(5, 2, 2)); // 1      /|       /|
    points.push_back(Vec3s(2, 5, 2)); // 2     2--------3 |
    points.push_back(Vec3s(5, 5, 2)); // 3     | |      | |
    points.push_back(Vec3s(2, 2, 5)); // 4     | 4------|-5
    points.push_back(Vec3s(5, 2, 5)); // 5     |/       |/
    points.push_back(Vec3s(2, 5, 5)); // 6     0--------1
    points.push_back(Vec3s(5, 5, 5)); // 7

    // cube faces
    quads.push_back(Vec4I(0, 1, 3, 2)); // front
    quads.push_back(Vec4I(5, 4, 6, 7)); // back
    quads.push_back(Vec4I(0, 2, 6, 4)); // left
    quads.push_back(Vec4I(1, 5, 7, 3)); // right
    quads.push_back(Vec4I(2, 3, 7, 6)); // top
    quads.push_back(Vec4I(0, 4, 5, 1)); // bottom

    FloatGrid::Ptr grid = tools::meshToLevelSet<FloatGrid>(
        *math::Transform::createLinearTransform(), points, quads);

    //io::File("testConversion.vdb").write(GridPtrVec(1, grid));

    CPPUNIT_ASSERT(grid.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(int(GRID_LEVEL_SET), int(grid->getGridClass()));
    CPPUNIT_ASSERT_EQUAL(1, int(grid->baseTree().leafCount()));
    /// @todo validate output
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
