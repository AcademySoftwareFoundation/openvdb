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

#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/unittest/util.h> // for PointAttributeWrapper
#include <openvdb_points/openvdb.h>

class TestPointConversion: public CppUnit::TestCase
{
public:

    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointConversion);
    CPPUNIT_TEST(testPointConversion);

    CPPUNIT_TEST_SUITE_END();

    void testPointConversion();

}; // class TestPointConversion

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointConversion);


////////////////////////////////////////


void
TestPointConversion::testPointConversion()
{
    using namespace unittest_util;

    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>          AttributeS;
    typedef openvdb::tools::TypedAttributeArray<int>            AttributeI;
    typedef openvdb::tools::TypedAttributeArray<openvdb::Vec3s> AttributeVec3s;

    AttributeS::registerType();
    AttributeI::registerType();
    AttributeVec3s::registerType();

    typedef openvdb::tools::PointAttributeList<openvdb::Vec3R, PointAttributeWrapper> PointAttributeList;
    typedef boost::scoped_ptr<PointAttributeList> PointAttributeListScopedPtr;

    PointAttributeListScopedPtr points(new PointAttributeList(
        PointAttributeWrapper::Ptr(new PointAttributeWrapper("P", AttributeVec3s::attributeType(), 1))));

    points->addAttribute(PointAttributeWrapper::Ptr(
        new PointAttributeWrapper("test",  AttributeS::attributeType(), 1)));

    const float voxelSize = 0.1f;
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(voxelSize));
    openvdb::tools::createPointDataGrid<openvdb::tools::PointDataGrid>(*points, *transform);
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
