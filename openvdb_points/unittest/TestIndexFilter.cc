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

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>

#include <sstream>
#include <iostream>

using namespace openvdb;
using namespace openvdb::tools;

class TestIndexFilter: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestIndexFilter);
    CPPUNIT_TEST(testRandomLeafFilter);

    CPPUNIT_TEST_SUITE_END();

    void testRandomLeafFilter();
}; // class TestIndexFilter

CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexFilter);


////////////////////////////////////////


struct OriginLeaf
{
    OriginLeaf(const openvdb::Coord& _leafOrigin): leafOrigin(_leafOrigin) { }
    openvdb::Coord origin() const { return leafOrigin; }
    openvdb::Coord leafOrigin;
};


struct SimpleIterator
{
    SimpleIterator() : i(0) { }
    int operator*() const { return const_cast<int&>(i)++; }
    int i;
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

        SimpleIterator iter;

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

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
