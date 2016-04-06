///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#include <cppunit/extensions/HelperMacros.h>

#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/Exceptions.h>
#include <openvdb/tree/LeafManager.h>

#include <vector>

class TestVolumeToMesh: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestVolumeToMesh);
    CPPUNIT_TEST(testAuxData);
    CPPUNIT_TEST(testConversion);
    CPPUNIT_TEST_SUITE_END();

    void testAuxData();
    void testConversion();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVolumeToMesh);


////////////////////////////////////////


void
TestVolumeToMesh::testAuxData()
{
    typedef openvdb::tree::Tree4<float, 5, 4, 3>::Type Tree543f;
    Tree543f::Ptr tree(new Tree543f(0));

    // create one voxel with 3 upwind edges (that have a sign change)
    tree->setValue(openvdb::Coord(0,0,0), -1);
    tree->setValue(openvdb::Coord(1,0,0),  1);
    tree->setValue(openvdb::Coord(0,1,0),  1);
    tree->setValue(openvdb::Coord(0,0,1),  1);

    typedef openvdb::tree::LeafManager<const Tree543f> LeafManager;

    LeafManager leafs(*tree);

    CPPUNIT_ASSERT(openvdb::tools::internal::needsActiveVoxePadding(leafs, 0.0, 1.0));

    openvdb::tools::internal::SignData<Tree543f, LeafManager> op(*tree, leafs, 0.0);
    op.run();

    CPPUNIT_ASSERT(op.signTree()->activeVoxelCount() == 1);
    CPPUNIT_ASSERT(op.signTree()->activeVoxelCount() == op.idxTree()->activeVoxelCount());


    int flags = int(op.signTree()->getValue(openvdb::Coord(0,0,0)));

    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::INSIDE));
    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::EDGES));
    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::XEDGE));
    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::YEDGE));
    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::ZEDGE));


    tree->setValueOff(openvdb::Coord(0,0,1), -1);
    op.run();

    CPPUNIT_ASSERT(op.signTree()->activeVoxelCount() == 1);
    CPPUNIT_ASSERT(op.signTree()->activeVoxelCount() == op.idxTree()->activeVoxelCount());

    flags = int(op.signTree()->getValue(openvdb::Coord(0,0,0)));

    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::INSIDE));
    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::EDGES));
    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::XEDGE));
    CPPUNIT_ASSERT(bool(flags & openvdb::tools::internal::YEDGE));
    CPPUNIT_ASSERT(!bool(flags & openvdb::tools::internal::ZEDGE));
}


void
TestVolumeToMesh::testConversion()
{
    using namespace openvdb;

    typedef tree::Tree4<float, 5, 4, 3>::Type Tree543f;
    typedef Grid<Tree543f> GridType;

    GridType::Ptr grid = createGrid<GridType>(/*background=*/1);

    grid->fill(CoordBBox(Coord(0), Coord(7)), 0.0);
    grid->fill(CoordBBox(Coord(1), Coord(6)), -1.0);

    std::vector<Vec3s> points;
    std::vector<Vec4I> quads;
    std::vector<Vec3I> triangles;

    openvdb::tools::volumeToMesh(*grid, points, quads);
    CPPUNIT_ASSERT(points.size() >= 4);
    CPPUNIT_ASSERT(!quads.empty());
    /// @todo validate output

    points.clear();
    quads.clear();
    triangles.clear();

    tools::volumeToMesh(*grid, points, triangles, quads, /*isovalue=*/0.01, /*adaptivity=*/0);
    CPPUNIT_ASSERT(points.size() >= 3);
    CPPUNIT_ASSERT(!triangles.empty() || !quads.empty());
    /// @todo validate output
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
