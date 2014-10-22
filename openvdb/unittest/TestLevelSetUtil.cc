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
#include <openvdb/tools/LevelSetUtil.h>

class TestLevelSetUtil: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestLevelSetUtil);
    CPPUNIT_TEST(testMinMaxVoxel);
    CPPUNIT_TEST(testLevelSetToFogVolume);
    CPPUNIT_TEST_SUITE_END();

    void testMinMaxVoxel();
    void testRelativeIsoOffset();
    void testLevelSetToFogVolume();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLevelSetUtil);


////////////////////////////////////////


void
TestLevelSetUtil::testMinMaxVoxel()
{

    openvdb::FloatTree myTree(std::numeric_limits<float>::max());

    openvdb::tree::ValueAccessor<openvdb::FloatTree> acc(myTree);

    for (int i = -9; i < 10; ++i) {
        for (int j = -9; j < 10; ++j) {
            acc.setValue(openvdb::Coord(i,j,0), static_cast<float>(j));
        }
    }

    openvdb::tree::LeafManager<openvdb::FloatTree> leafs(myTree);

    openvdb::tools::MinMaxVoxel<openvdb::FloatTree> minmax(leafs);
    minmax.runParallel();

    CPPUNIT_ASSERT(!(minmax.minVoxel() < -9.0));
    CPPUNIT_ASSERT(!(minmax.maxVoxel() >  9.0));
}

void
TestLevelSetUtil::testLevelSetToFogVolume()
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(10.0);

    grid->fill(openvdb::CoordBBox(openvdb::Coord(-100), openvdb::Coord(100)), 9.0);
    grid->fill(openvdb::CoordBBox(openvdb::Coord(-50), openvdb::Coord(50)), -9.0);


    openvdb::tools::sdfToFogVolume(*grid);

    CPPUNIT_ASSERT(grid->background() < 1e-7);

    openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn();
    for (; iter; ++iter) {
        CPPUNIT_ASSERT(iter.getValue() > 0.0);
        CPPUNIT_ASSERT(std::abs(iter.getValue() - 1.0) < 1e-7);
    }
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
