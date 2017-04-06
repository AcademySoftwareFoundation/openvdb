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

#include <openvdb/openvdb.h> // for FloatGrid
#include <openvdb/tools/LevelSetSphere.h> // for createLevelSetSphere
#include <openvdb/tools/LevelSetUtil.h> // for sdfToFogVolume
#include <openvdb/tools/VolumeToSpheres.h> // for fillWithSpheres
#include <openvdb/Exceptions.h>

#include <vector>


class TestVolumeToSpheres: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestVolumeToSpheres);
    CPPUNIT_TEST(testFromLevelSet);
    CPPUNIT_TEST(testFromFog);
    CPPUNIT_TEST_SUITE_END();

    void testFromLevelSet();
    void testFromFog();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVolumeToSpheres);


////////////////////////////////////////


void
TestVolumeToSpheres::testFromLevelSet()
{
    const float
        radius = 20.0f,
        voxelSize = 1.0f,
        halfWidth = 3.0f;
    const openvdb::Vec3f center(15.0f, 13.0f, 16.0f);

    openvdb::FloatGrid::ConstPtr grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, center, voxelSize, halfWidth);

    const int
        maxSphereCount = 100,
        instanceCount = 10000;
    const bool overlapping = false;
    const float
        isovalue = 0.0,
        minRadius = 5.0,
        maxRadius = std::numeric_limits<float>::max();

    {
        std::vector<openvdb::Vec4s> spheres;

        openvdb::tools::fillWithSpheres(*grid, spheres, maxSphereCount, overlapping,
            minRadius, maxRadius, isovalue, instanceCount);

        CPPUNIT_ASSERT_EQUAL(1, int(spheres.size()));

        //for (size_t i=0; i< spheres.size(); ++i) {
        //    std::cout << "\nSphere #" << i << ": " << spheres[i] << std::endl;
        //}

        const auto tolerance = 2.0 * voxelSize;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[0], spheres[0][0], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[1], spheres[0][1], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[2], spheres[0][2], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(radius,    spheres[0][3], tolerance);
    }
    {
        // Verify that an isovalue outside the narrow band still produces a valid sphere.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, maxSphereCount,
            overlapping, minRadius, maxRadius, 1.5 * halfWidth, instanceCount);
        CPPUNIT_ASSERT_EQUAL(1, int(spheres.size()));
    }
    {
        // Verify that an isovalue inside the narrow band produces no spheres.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, maxSphereCount,
            overlapping, minRadius, maxRadius, -1.5 * halfWidth, instanceCount);
        CPPUNIT_ASSERT_EQUAL(0, int(spheres.size()));
    }
}


void
TestVolumeToSpheres::testFromFog()
{
    const float
        radius = 20.0f,
        voxelSize = 1.0f,
        halfWidth = 3.0f;
    const openvdb::Vec3f center(15.0f, 13.0f, 16.0f);

    auto grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, center, voxelSize, halfWidth);
    openvdb::tools::sdfToFogVolume(*grid);

    const int
        maxSphereCount = 100,
        instanceCount = 10000;
    const bool overlapping = false;
    const float
        isovalue = 0.01f,
        minRadius = 5.0,
        maxRadius = std::numeric_limits<float>::max();

    {
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, maxSphereCount, overlapping,
            minRadius, maxRadius, isovalue, instanceCount);

        //for (size_t i=0; i< spheres.size(); ++i) {
        //    std::cout << "\nSphere #" << i << ": " << spheres[i] << std::endl;
        //}

        CPPUNIT_ASSERT_EQUAL(1, int(spheres.size()));

        const auto tolerance = 2.0 * voxelSize;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[0], spheres[0][0], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[1], spheres[0][1], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[2], spheres[0][2], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(radius,    spheres[0][3], tolerance);
    }
    {
        // Verify that an isovalue outside the narrow band still produces valid spheres.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, maxSphereCount, overlapping,
            minRadius, maxRadius, 10.0f, instanceCount);
        CPPUNIT_ASSERT(!spheres.empty());
    }
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
