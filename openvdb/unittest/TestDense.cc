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

//#define BENCHMARK_TEST

#include <openvdb/openvdb.h>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Dense.h>
#include <sstream>
#ifdef BENCHMARK_TEST
#include <boost/date_time/posix_time/posix_time.hpp>
#endif


class TestDense: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestDense);
    CPPUNIT_TEST(testDense);
    CPPUNIT_TEST(testCopy);
    CPPUNIT_TEST(testCopyBool);
    CPPUNIT_TEST_SUITE_END();

    void testDense();
    void testCopy();
    void testCopyBool();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDense);


void
TestDense::testDense()
{
    const openvdb::CoordBBox bbox(openvdb::Coord(-40,-5, 6),
                                  openvdb::Coord(-11, 7,22));
    openvdb::tools::Dense<float> dense(bbox);

    // Check Dense::valueCount
    const int size = dense.valueCount();
    CPPUNIT_ASSERT_EQUAL(30*13*17, size);

    // Cehck Dense::fill(float) and Dense::getValue(size_t)
    const float v = 0.234f;
    dense.fill(v);
    for (int i=0; i<size; ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(v, dense.getValue(i),/*tolerance=*/0.0001);
    }

    // Check Dense::data() and Dense::getValue(Coord, float)
    float* a = dense.data();
    int s = size;
    while(s--) CPPUNIT_ASSERT_DOUBLES_EQUAL(v, *a++, /*tolerance=*/0.0001);

    for (openvdb::Coord P(bbox.min()); P[0] <= bbox.max()[0]; ++P[0]) {
        for (P[1] = bbox.min()[1]; P[1] <= bbox.max()[1]; ++P[1]) {
            for (P[2] = bbox.min()[2]; P[2] <= bbox.max()[2]; ++P[2]) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(v, dense.getValue(P), /*tolerance=*/0.0001);
            }
        }
    }

    // Check Dense::setValue(Coord, float)
    const openvdb::Coord C(-30, 3,12);
    const float v1 = 3.45f;
    dense.setValue(C, v1);
    for (openvdb::Coord P(bbox.min()); P[0] <= bbox.max()[0]; ++P[0]) {
        for (P[1] = bbox.min()[1]; P[1] <= bbox.max()[1]; ++P[1]) {
            for (P[2] = bbox.min()[2]; P[2] <= bbox.max()[2]; ++P[2]) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(P==C ? v1 : v, dense.getValue(P),
                    /*tolerance=*/0.0001);
            }
        }
    }

    // Check Dense::setValue(size_t, size_t, size_t, float)
    dense.setValue(C, v);
    const openvdb::Coord L(1,2,3), C1 = bbox.min() + L;
    dense.setValue(L[0], L[1], L[2], v1);
    for (openvdb::Coord P(bbox.min()); P[0] <= bbox.max()[0]; ++P[0]) {
        for (P[1] = bbox.min()[1]; P[1] <= bbox.max()[1]; ++P[1]) {
            for (P[2] = bbox.min()[2]; P[2] <= bbox.max()[2]; ++P[2]) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(P==C1 ? v1 : v, dense.getValue(P),
                    /*tolerance=*/0.0001);
            }
        }
    }

}


// The check is so slow that we're going to multi-thread it :)
template <typename TreeT>
class CheckDense
{
public:
    typedef typename TreeT::ValueType     ValueT;
    typedef openvdb::tools::Dense<ValueT> DenseT;

    CheckDense() : mTree(NULL), mDense(NULL) {}

    void check(const TreeT& tree, const DenseT& dense)
    {
        mTree  = &tree;
        mDense = &dense;
        tbb::parallel_for(dense.bbox(), *this);
    }
    void operator()(const openvdb::CoordBBox& bbox) const
    {
        openvdb::tree::ValueAccessor<const TreeT> acc(*mTree);
        for (openvdb::Coord P(bbox.min()); P[0] <= bbox.max()[0]; ++P[0]) {
            for (P[1] = bbox.min()[1]; P[1] <= bbox.max()[1]; ++P[1]) {
                for (P[2] = bbox.min()[2]; P[2] <= bbox.max()[2]; ++P[2]) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(acc.getValue(P), mDense->getValue(P),
                        /*tolerance=*/0.0001);
                }
            }
        }
    }
private:
    const TreeT*  mTree;
    const DenseT* mDense;
};// CheckDense


void
TestDense::testCopy()
{
    CheckDense<openvdb::FloatTree> checkDense;
    const float radius = 10.0f, tolerance = 0.00001f;
    const openvdb::Vec3f center(0.0f);
    // decrease the voxelSize to test larger grids
#ifdef BENCHMARK_TEST
    const float voxelSize = 0.05f, width = 5.0f;
#else
    const float voxelSize = 0.2f, width = 5.0f;
#endif

    // Create a VDB containing a level set of a sphere
    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);
    openvdb::FloatTree& tree0 = grid->tree();

    // Create an empty dense grid
    openvdb::tools::Dense<float> dense(grid->evalActiveVoxelBoundingBox());
#ifdef BENCHMARK_TEST
    std::cerr << "\nBBox = " << grid->evalActiveVoxelBoundingBox() << std::endl;
#endif
    
    {//check Dense::fill
        dense.fill(voxelSize);
#ifndef BENCHMARK_TEST
        checkDense.check(openvdb::FloatTree(voxelSize), dense);
#endif
    }

    {// parallel convert to dense
#ifdef BENCHMARK_TEST
        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();
#endif
        openvdb::tools::copyToDense(*grid, dense);
#ifdef BENCHMARK_TEST
        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nCopyToDense took " << msdiff.total_milliseconds() << " ms\n";
#else
        checkDense.check(tree0, dense);
#endif
    }

    {// Parallel create from dense
#ifdef BENCHMARK_TEST
        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();
#endif
        openvdb::FloatTree tree1(tree0.background());
        openvdb::tools::copyFromDense(dense, tree1, tolerance);
#ifdef BENCHMARK_TEST
        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nCopyFromDense took " << msdiff.total_milliseconds() << " ms\n";
#else
        checkDense.check(tree1, dense);
#endif
    }
}


void
TestDense::testCopyBool()
{
    using namespace openvdb;

    const Coord bmin(-1), bmax(8);
    const CoordBBox bbox(bmin, bmax);

    BoolGrid::Ptr grid = createGrid<BoolGrid>(false);
    BoolGrid::ConstAccessor acc = grid->getConstAccessor();

    tools::Dense<bool> dense(bbox);
    dense.fill(false);

    // Start with sparse and dense grids both filled with false.
    Coord xyz;
    int &x = xyz[0], &y = xyz[1], &z = xyz[2];
    for (x = bmin.x(); x <= bmax.x(); ++x) {
        for (y = bmin.y(); y <= bmax.y(); ++y) {
            for (z = bmin.z(); z <= bmax.z(); ++z) {
                CPPUNIT_ASSERT_EQUAL(false, dense.getValue(xyz));
                CPPUNIT_ASSERT_EQUAL(false, acc.getValue(xyz));
            }
        }
    }

    // Fill the dense grid with true.
    dense.fill(true);
    // Copy the contents of the dense grid to the sparse grid.
    openvdb::tools::copyFromDense(dense, *grid, /*tolerance=*/false);

    // Verify that both sparse and dense grids are now filled with true.
    for (x = bmin.x(); x <= bmax.x(); ++x) {
        for (y = bmin.y(); y <= bmax.y(); ++y) {
            for (z = bmin.z(); z <= bmax.z(); ++z) {
                CPPUNIT_ASSERT_EQUAL(true, dense.getValue(xyz));
                CPPUNIT_ASSERT_EQUAL(true, acc.getValue(xyz));
            }
        }
    }

    // Fill the dense grid with false.
    dense.fill(false);
    // Copy the contents (= true) of the sparse grid to the dense grid.
    openvdb::tools::copyToDense(*grid, dense);

    // Verify that the dense grid is now filled with true.
    for (x = bmin.x(); x <= bmax.x(); ++x) {
        for (y = bmin.y(); y <= bmax.y(); ++y) {
            for (z = bmin.z(); z <= bmax.z(); ++z) {
                CPPUNIT_ASSERT_EQUAL(true, dense.getValue(xyz));
            }
        }
    }
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
