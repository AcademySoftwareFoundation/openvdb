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
#include <openvdb_points/tools/IndexIterator.h>
#include <openvdb_points/tools/IndexFilter.h>
#include <openvdb_points/tools/PointConversion.h>

#include <boost/random/mersenne_twister.hpp>

#include <sstream>
#include <iostream>

using namespace openvdb;
using namespace openvdb::tools;

class TestIndexFilter: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestIndexFilter);
    CPPUNIT_TEST(testRandomLeafFilter);
    CPPUNIT_TEST(testBBoxFilter);

    CPPUNIT_TEST_SUITE_END();

    void testRandomLeafFilter();
    void testBBoxFilter();
}; // class TestIndexFilter

CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexFilter);


////////////////////////////////////////


struct OriginLeaf
{
    OriginLeaf(const openvdb::Coord& _leafOrigin): leafOrigin(_leafOrigin) { }
    openvdb::Coord origin() const { return leafOrigin; }
    openvdb::Coord leafOrigin;
};


struct SimpleIter
{
    SimpleIter() : i(0) { }
    int operator*() const { return const_cast<int&>(i)++; }
    openvdb::Coord getCoord() const { return coord; }
    int i;
    openvdb::Coord coord;
};


void
TestIndexFilter::testRandomLeafFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef RandomLeafFilter<boost::mt11213b> RandFilter;
    typedef RandFilter::LeafSeedMap LeafSeedMap;

    LeafSeedMap leafSeedMap;

    // empty leaf offset

    CPPUNIT_ASSERT_THROW(RandFilter::create(OriginLeaf(openvdb::Coord(0, 0, 0)), RandFilter::Data(0.5f, leafSeedMap)), openvdb::KeyError);

    // add some origin values

    std::vector<Coord> origins;
    origins.push_back(Coord(0, 0, 0));
    origins.push_back(Coord(0, 8, 0));
    origins.push_back(Coord(0, 0, 8));
    origins.push_back(Coord(8, 8, 8));

    leafSeedMap[origins[0]] = 0;
    leafSeedMap[origins[1]] = 1;
    leafSeedMap[origins[2]] = 2;
    leafSeedMap[origins[3]] = 100;

    // 10,000,000 values, multiple origins

    const int total = 1000 * 1000 * 10;
    const float threshold = 0.25f;

    std::vector<double> errors;

    for (std::vector<Coord>::const_iterator it = origins.begin(), itEnd = origins.end(); it != itEnd; ++it)
    {
        RandFilter filter = RandFilter::create(OriginLeaf(*it), RandFilter::Data(threshold, leafSeedMap));

        SimpleIter iter;

        int success = 0;

        for (int i = 0; i < total; i++) {
            if (filter.valid(iter))     success++;
        }

        // ensure error is within a reasonable tolerance

        const double error = fabs(success - total * threshold) / total;
        errors.push_back(error);

        CPPUNIT_ASSERT(error < 1e-3);
    }

    CPPUNIT_ASSERT(errors[0] != errors[1]);
}


void
TestIndexFilter::testBBoxFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;
    typedef PointDataTree::LeafNodeType::ValueOnCIter ValueOnCIter;

    AttributeVec3s::registerType();

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(1, 2, 1));
    positions.push_back(Vec3s(10.1, 10, 1));

    const float voxelSize(0.5);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(2));

    // build some bounding box filters to test

    BBoxFilter::Data data1 = BBoxFilter::Data(*transform, BBoxd(Vec3d(0.5, 0.5, 0.5), Vec3d(1.5, 1.5, 1.5)));
    BBoxFilter::Data data2 = BBoxFilter::Data(*transform, BBoxd(Vec3d(0.5, 0.5, 0.5), Vec3d(1.5, 2.01, 1.5)));
    BBoxFilter::Data data3 = BBoxFilter::Data(*transform, BBoxd(Vec3d(0.5, 0.5, 0.5), Vec3d(11, 11, 1.5)));
    BBoxFilter::Data data4 = BBoxFilter::Data(*transform, BBoxd(Vec3d(-10, 0, 0), Vec3d(11, 1.2, 1.2)));

    // leaf 1

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();

    {
        ValueOnCIter valueIter(leafIter->beginValueOn());
        ValueIndexIter<ValueOnCIter> iter(valueIter);

        // point 1

        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data1).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data2).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data3).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data4).valid(iter));

        ++iter;

        // point 2

        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data1).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data2).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data3).valid(iter));
        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data4).valid(iter));

        ++iter;
        CPPUNIT_ASSERT(!iter);
    }

    ++leafIter;

    // leaf 2

    {
        ValueOnCIter valueIter(leafIter->beginValueOn());
        ValueIndexIter<ValueOnCIter> iter(valueIter);

        // point 3

        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data1).valid(iter));
        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data2).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data3).valid(iter));
        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data4).valid(iter));

        ++iter;
        CPPUNIT_ASSERT(!iter);
    }
}


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
