///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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

#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/tools/Load.h>
#include <openvdb_points/tools/AttributeArray.h>
#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/io/File.h>

#include <cstdlib> // for std::getenv(), mkstemp()

class TestLoad: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestLoad);
    CPPUNIT_TEST(testLoad);

    CPPUNIT_TEST_SUITE_END();

    void testLoad();
}; // class TestLoad

CPPUNIT_TEST_SUITE_REGISTRATION(TestLoad);


////////////////////////////////////////


void
TestLoad::testLoad()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    std::string tempDir(std::getenv("TMPDIR"));
    if (tempDir.empty())    tempDir = P_tmpdir;

    std::string filename;

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    // create a tree with multiple points, four leaves

    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 2, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                        {20, 1, 1},
                                        {1, 20, 1},
                                        {1, 1, 20}
                                    };

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree2 = grid->tree();

    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), Index32(4));

    // write out grid to a temp file
    {
        filename = tempDir + "/openvdb_test_point_load";

        io::File fileOut(filename);

        GridCPtrVec grids{grid};

        fileOut.write(grids);
    }

    // read and load all leaf nodes
    {
        io::File fileIn(filename);
        fileIn.open();

        GridPtrVecPtr grids = fileIn.getGrids();

        fileIn.close();

        CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

        PointDataGrid::Ptr grid = GridBase::grid<PointDataGrid>((*grids)[0]);

        CPPUNIT_ASSERT(grid);

        PointDataGrid::TreeType::LeafCIter leafIter = grid->tree().cbeginLeaf();

#ifndef OPENVDB_2_ABI_COMPATIBLE
        // all leaves out of core

        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore());

        loadGrid(*grid);

        leafIter = grid->tree().cbeginLeaf();
#endif

        // all leaves loaded into memory

        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore());
    }

#ifndef OPENVDB_2_ABI_COMPATIBLE
    // read and load leaf nodes by bbox
    {
        io::File fileIn(filename);
        fileIn.open();

        GridPtrVecPtr grids = fileIn.getGrids();

        fileIn.close();

        CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

        PointDataGrid::Ptr grid = GridBase::grid<PointDataGrid>((*grids)[0]);

        CPPUNIT_ASSERT(grid);

        PointDataGrid::TreeType::LeafCIter leafIter = grid->tree().cbeginLeaf();

        // all leaves out of core

        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore());

        BBoxd bbox(Vec3i(0, 0, 0), Vec3i(4, 30, 4));

        loadGrid(*grid, bbox);

        leafIter = grid->tree().cbeginLeaf();

        // only first and third leaf loaded into memory

        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore());
    }
#endif

#ifndef OPENVDB_2_ABI_COMPATIBLE
    // read and load leaf nodes by mask
    {
        io::File fileIn(filename);
        fileIn.open();

        GridPtrVecPtr grids = fileIn.getGrids();

        fileIn.close();

        CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

        PointDataGrid::Ptr grid = GridBase::grid<PointDataGrid>((*grids)[0]);

        CPPUNIT_ASSERT(grid);

        PointDataGrid::TreeType::LeafCIter leafIter = grid->tree().cbeginLeaf();

        // all leaves out of core

        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore());

        BoolGrid::Ptr mask = BoolGrid::create(false);

        mask->tree().touchLeaf(Coord(0, 0, 0));
        mask->tree().touchLeaf(Coord(1, 1, 20));

        loadGrid(*grid, *mask);

        leafIter = grid->tree().cbeginLeaf();

        // only first and second leaves loaded into memory

        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(!leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore()); ++leafIter;
        CPPUNIT_ASSERT(leafIter->buffer().isOutOfCore());
    }
#endif

    // cleanup temp files

    std::remove(filename.c_str());
}



// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
