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

/// @author Ken Museth

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/RayIntersector.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);

#define ASSERT_DOUBLES_APPROX_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/1.e-6);


class TestVolumeRayIntersector : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestVolumeRayIntersector);
    CPPUNIT_TEST(testAll);
    CPPUNIT_TEST_SUITE_END();

    void testAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVolumeRayIntersector);

void
TestVolumeRayIntersector::testAll()
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;
    
    {//one single leaf node
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0,0,0), 1.0f);
        grid.tree().setValue(Coord(7,7,7), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        CPPUNIT_ASSERT(inter.setIndexRay(ray));
        double t0=0, t1=0;
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t1);
        CPPUNIT_ASSERT_EQUAL(0, inter.march(t0, t1));
    }
    {//two adjacent leaf nodes
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0,0,0), 1.0f);
        grid.tree().setValue(Coord(8,0,0), 1.0f);
        grid.tree().setValue(Coord(15,7,7), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        CPPUNIT_ASSERT(inter.setIndexRay(ray));
        double t0=0, t1=0;
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t1);
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        CPPUNIT_ASSERT_EQUAL(0, inter.march(t0, t1));
    }
    {//two adjacent leafs followed by a gab and leaf
        FloatGrid grid(0.0f);
        
        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8+7,7,7), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        CPPUNIT_ASSERT(inter.setIndexRay(ray));
        double t0=0, t1=0;
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t1);
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(33.0, t1);
        CPPUNIT_ASSERT_EQUAL(0, inter.march(t0, t1));
    }
    {//two adjacent leafs followed by a gab, a leaf and an active tile
        FloatGrid grid(0.0f);
        
        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        grid.fill(CoordBBox(Coord(4*8,0,0), Coord(4*8+7,7,7)), 2.0f, true);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        CPPUNIT_ASSERT(inter.setIndexRay(ray));
        double t0=0, t1=0;
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t1);
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(33.0, t1);
        CPPUNIT_ASSERT_EQUAL(1, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(33.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(41.0, t1);
        CPPUNIT_ASSERT_EQUAL(0, inter.march(t0, t1));
    }
    
    {// Test submitted by "Jan" @ GitHub
        FloatGrid grid(0.0f);
        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        
        const Vec3T dir(-1.0, 0.0, 0.0);
        const Vec3T eye(50.0, 0.0, 0.0);
        const RayT ray(eye, dir);
        CPPUNIT_ASSERT(inter.setIndexRay(ray));
        double t0=0, t1=0;
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(18.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(26.0, t1);
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(34.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(42.0, t1);
        CPPUNIT_ASSERT_EQUAL(2, inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(42.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(50.0, t1);
        CPPUNIT_ASSERT_EQUAL(0, inter.march(t0, t1));
    }
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
