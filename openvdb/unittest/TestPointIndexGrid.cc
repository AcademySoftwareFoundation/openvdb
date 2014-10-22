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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/math/Math.h> // for math::Random01
#include <openvdb/tools/PointIndexGrid.h>
#include <vector>
#include <algorithm>
#include <cmath>


class TestPointIndexGrid: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestPointIndexGrid);
    CPPUNIT_TEST(testPointIndexGrid);
    CPPUNIT_TEST_SUITE_END();

    void testPointIndexGrid();

private:
    // Generate random points by uniformly distributing points
    // on a unit-sphere.
    void genPoints(const int numPoints, std::vector<openvdb::Vec3R>& points) const
    {
        // init
        openvdb::math::Random01 randNumber(0);
        const int n = int(std::sqrt(double(numPoints)));
        const double xScale = (2.0 * M_PI) / double(n);
        const double yScale = M_PI / double(n);

        double x, y, theta, phi;
        openvdb::Vec3R pos;

        points.reserve(n*n);

        // loop over a [0 to n) x [0 to n) grid.
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {

                // jitter, move to random pos. inside the current cell
                x = double(a) + randNumber();
                y = double(b) + randNumber();

                // remap to a lat/long map
                theta = y * yScale; // [0 to PI]
                phi   = x * xScale; // [0 to 2PI]

                // convert to cartesian coordinates on a unit sphere.
                // spherical coordinate triplet (r=1, theta, phi)
                pos[0] = std::sin(theta)*std::cos(phi);
                pos[1] = std::sin(theta)*std::sin(phi);
                pos[2] = std::cos(theta);

                points.push_back(pos);
            }
        }
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointIndexGrid);

////////////////////////////////////////

namespace {

class PointList
{
public:
    typedef openvdb::Vec3R value_type;

    PointList(const std::vector<openvdb::Vec3R>& points)
        : mPoints(&points)
    {
    }

    size_t size() const {
        return mPoints->size();
    }

    void getPos(size_t n, openvdb::Vec3R& xyz) const {
        xyz = (*mPoints)[n];
    }

protected:
    std::vector<openvdb::Vec3R> const * const mPoints;
}; // PointList


} // namespace



////////////////////////////////////////


void
TestPointIndexGrid::testPointIndexGrid()
{
    const float voxelSize = 0.01f;
    const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

    // generate points

    std::vector<openvdb::Vec3R> points;
    genPoints(40000, points);

    PointList pointList(points);


    // construct data structure
    typedef openvdb::tools::PointIndexGrid PointIndexGrid;

    PointIndexGrid::Ptr pointGridPtr =
        openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);

    openvdb::CoordBBox bbox;
    pointGridPtr->tree().evalActiveVoxelBoundingBox(bbox);


    // bbox search

    typedef PointIndexGrid::ConstAccessor ConstAccessor;
    ConstAccessor acc = pointGridPtr->getConstAccessor();
    openvdb::tools::PointIndexIterator<> it(bbox, acc);

    CPPUNIT_ASSERT(it.test());
    CPPUNIT_ASSERT_EQUAL(points.size(), it.size());

    // Check partitioning

    CPPUNIT_ASSERT(openvdb::tools::isValidPartition(pointList, *pointGridPtr));

    points[10000].x() += 1.5; // manually modify a few points.
    points[20000].x() += 1.5;
    points[30000].x() += 1.5;

    CPPUNIT_ASSERT(!openvdb::tools::isValidPartition(pointList, *pointGridPtr));

    PointIndexGrid::Ptr pointGrid2Ptr =
        openvdb::tools::getValidPointIndexGrid<PointIndexGrid>(pointList, pointGridPtr);

    CPPUNIT_ASSERT(openvdb::tools::isValidPartition(pointList, *pointGrid2Ptr));
}


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
