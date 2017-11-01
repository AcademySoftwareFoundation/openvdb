///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/openvdb.h>

#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointMask.h>

#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointMask: public CppUnit::TestCase
{
public:

    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointMask);
    CPPUNIT_TEST(testMask);
    CPPUNIT_TEST_SUITE_END();

    void testMask();

}; // class TestPointMask


void
TestPointMask::testMask()
{
    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 5, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    const float voxelSize = 0.1f;
    openvdb::math::Transform::Ptr transform(
        openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                          pointList, *transform);

    { // simple topology copy
        auto mask = convertPointsToMask(*points);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
    }

    { // mask grid instead of bool grid
        auto mask = convertPointsToMask<PointDataGrid, MaskGrid>(*points);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
    }

    { // identical transform
        auto mask = convertPointsToMask(*points, *transform);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
    }

    // assign point 3 to new group "test"

    appendGroup(points->tree(), "test");

    std::vector<short> groups{0,0,1,0};

    setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");

    std::vector<std::string> includeGroups{"test"};
    std::vector<std::string> excludeGroups;

    { // convert in turn "test" and not "test"
        auto mask = convertPointsToMask(*points, includeGroups, excludeGroups);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(1));

        mask = convertPointsToMask(*points, excludeGroups, includeGroups);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(3));
    }

    { // use a much larger voxel size that splits the points into two regions
        const float newVoxelSize(2);
        openvdb::math::Transform::Ptr newTransform(
            openvdb::math::Transform::createLinearTransform(newVoxelSize));

        auto mask = convertPointsToMask(*points, *newTransform);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(2));

        mask = convertPointsToMask(*points, *newTransform, includeGroups, excludeGroups);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(1));

        mask = convertPointsToMask(*points, *newTransform, excludeGroups, includeGroups);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(2));
    }
}


CPPUNIT_TEST_SUITE_REGISTRATION(TestPointMask);

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
