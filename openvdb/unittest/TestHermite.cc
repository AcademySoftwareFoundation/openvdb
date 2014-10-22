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
#include <openvdb/Exceptions.h>
#include <openvdb/math/Hermite.h>
#include <openvdb/math/QuantizedUnitVec.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Vec3.h>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <ctime>
class TestHermite: public CppUnit::TestFixture
{
public:

    CPPUNIT_TEST_SUITE(TestHermite);
    CPPUNIT_TEST(testAccessors);
    CPPUNIT_TEST(testComparisons);
    CPPUNIT_TEST(testIO);
    CPPUNIT_TEST_SUITE_END();

    void testAccessors();
    void testComparisons();
    void testIO();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestHermite);


////////////////////////////////////////


void
TestHermite::testAccessors()
{
    using namespace openvdb;
    using namespace openvdb::math;
    const double offsetTol = 0.001;
    const float normalTol = 0.015f;

    //////////

    // Check initial values.

    Hermite hermite;

    CPPUNIT_ASSERT(!hermite);
    CPPUNIT_ASSERT(!hermite.isInside());

    CPPUNIT_ASSERT(!hermite.hasOffsetX());
    CPPUNIT_ASSERT(!hermite.hasOffsetY());
    CPPUNIT_ASSERT(!hermite.hasOffsetZ());


    //////////

    // Check set & get

    // x
    Vec3s n0(1.0, 0.0, 0.0);
    hermite.setX(0.5f, n0);
    CPPUNIT_ASSERT(hermite.hasOffsetX());
    CPPUNIT_ASSERT(!hermite.hasOffsetY());
    CPPUNIT_ASSERT(!hermite.hasOffsetZ());

    float offset = hermite.getOffsetX();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, offset, offsetTol);

    Vec3s n1 = hermite.getNormalX();
    CPPUNIT_ASSERT(n0.eq(n1, normalTol));

    // y
    n0 = Vec3s(0.0, 1.0, 0.0);
    hermite.setY(0.3f, n0);
    CPPUNIT_ASSERT(hermite.hasOffsetX());
    CPPUNIT_ASSERT(hermite.hasOffsetY());
    CPPUNIT_ASSERT(!hermite.hasOffsetZ());

    offset = hermite.getOffsetY();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.3, offset, offsetTol);

    n1 = hermite.getNormalY();
    CPPUNIT_ASSERT(n0.eq(n1, normalTol));

    // z
    n0 = Vec3s(0.0, 0.0, 1.0);
    hermite.setZ(0.75f, n0);
    CPPUNIT_ASSERT(hermite.hasOffsetX());
    CPPUNIT_ASSERT(hermite.hasOffsetY());
    CPPUNIT_ASSERT(hermite.hasOffsetZ());

    offset = hermite.getOffsetZ();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.75, offset, offsetTol);

    n1 = hermite.getNormalZ();
    CPPUNIT_ASSERT(n0.eq(n1, normalTol));


    //////////

    // Check inside/outside state

    hermite.setIsInside(true);
    CPPUNIT_ASSERT(hermite.isInside());

    hermite.clear();

    CPPUNIT_ASSERT(!hermite);
    CPPUNIT_ASSERT(!hermite.isInside());

    CPPUNIT_ASSERT(!hermite.hasOffsetX());
    CPPUNIT_ASSERT(!hermite.hasOffsetY());
    CPPUNIT_ASSERT(!hermite.hasOffsetZ());

    n0 = Vec3s(0.0, 0.0, -1.0);
    hermite.setZ(0.15f, n0);
    CPPUNIT_ASSERT(!hermite.hasOffsetX());
    CPPUNIT_ASSERT(!hermite.hasOffsetY());
    CPPUNIT_ASSERT(hermite.hasOffsetZ());

    offset = hermite.getOffsetZ();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.15, offset, offsetTol);

    n1 = hermite.getNormalZ();
    CPPUNIT_ASSERT(n0.eq(n1, normalTol));

    hermite.setIsInside(true);
    CPPUNIT_ASSERT(hermite.isInside());

    CPPUNIT_ASSERT(hermite);
}


////////////////////////////////////////


void
TestHermite::testComparisons()
{
    using namespace openvdb;
    using namespace openvdb::math;
    const double offsetTol = 0.001;
    const float normalTol = 0.015f;


    //////////


    Vec3s offsets(0.50f, 0.82f, 0.14f);
    Vec3s nX(1.0, 0.0, 0.0);
    Vec3s nY(0.0, 1.0, 0.0);
    Vec3s nZ(0.0, 0.0, 1.0);

    Hermite A, B;

    A.setX(offsets[0], nX);
    A.setY(offsets[1], nY);
    A.setZ(offsets[2], nZ);
    A.setIsInside(true);

    B = A;

    CPPUNIT_ASSERT(B);
    CPPUNIT_ASSERT(B == A);
    CPPUNIT_ASSERT(!(B != A));
    CPPUNIT_ASSERT(B.isInside());

    CPPUNIT_ASSERT_DOUBLES_EQUAL(offsets[0], B.getOffsetX(), offsetTol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(offsets[1], B.getOffsetY(), offsetTol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(offsets[2], B.getOffsetZ(), offsetTol);

    CPPUNIT_ASSERT(B.getNormalX().eq(nX, normalTol));
    CPPUNIT_ASSERT(B.getNormalY().eq(nY, normalTol));
    CPPUNIT_ASSERT(B.getNormalZ().eq(nZ, normalTol));

    CPPUNIT_ASSERT(!A.isLessX(B));
    CPPUNIT_ASSERT(!A.isLessY(B));
    CPPUNIT_ASSERT(!A.isLessZ(B));

    CPPUNIT_ASSERT(!A.isGreaterX(B));
    CPPUNIT_ASSERT(!A.isGreaterY(B));
    CPPUNIT_ASSERT(!A.isGreaterZ(B));

    B = -B;

    CPPUNIT_ASSERT(B);
    CPPUNIT_ASSERT(B != A);
    CPPUNIT_ASSERT(!B.isInside());

    CPPUNIT_ASSERT_DOUBLES_EQUAL(offsets[0], B.getOffsetX(), offsetTol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(offsets[1], B.getOffsetY(), offsetTol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(offsets[2], B.getOffsetZ(), offsetTol);

    CPPUNIT_ASSERT(B.getNormalX().eq(-nX, normalTol));
    CPPUNIT_ASSERT(B.getNormalY().eq(-nY, normalTol));
    CPPUNIT_ASSERT(B.getNormalZ().eq(-nZ, normalTol));

    CPPUNIT_ASSERT(A.isLessX(B));
    CPPUNIT_ASSERT(A.isLessY(B));
    CPPUNIT_ASSERT(A.isLessZ(B));

    CPPUNIT_ASSERT(!A.isGreaterX(B));
    CPPUNIT_ASSERT(!A.isGreaterY(B));
    CPPUNIT_ASSERT(!A.isGreaterZ(B));


    //////////

    // min / max

    Hermite C = min(A, B);

    CPPUNIT_ASSERT(C);
    CPPUNIT_ASSERT(C == A);
    CPPUNIT_ASSERT(C != B);

    C = max(A, B);

    CPPUNIT_ASSERT(C);
    CPPUNIT_ASSERT(C != A);
    CPPUNIT_ASSERT(C == B);


    A.clear();
    B.clear();
    C.clear();

    A.setX(offsets[0], nX);
    A.setY(offsets[1], nY);
    A.setZ(offsets[2], nZ);
    A.setIsInside(true);

    B.setX(offsets[2], nX);
    B.setY(offsets[0], nY);
    B.setZ(offsets[1], nZ);
    B.setIsInside(true);

    C = max(A, B);

    CPPUNIT_ASSERT(C);
    CPPUNIT_ASSERT(C != A);
    CPPUNIT_ASSERT(C != B);
    CPPUNIT_ASSERT(C.isGreaterX(A));
    CPPUNIT_ASSERT(C.isGreaterY(A));
    CPPUNIT_ASSERT(C.isGreaterZ(B));

    C = min(A, B);
    CPPUNIT_ASSERT(C);
    CPPUNIT_ASSERT(C != A);
    CPPUNIT_ASSERT(C != B);
    CPPUNIT_ASSERT(C.isLessX(B));
    CPPUNIT_ASSERT(C.isLessY(B));
    CPPUNIT_ASSERT(C.isLessZ(A));


    A.clear();
    B.clear();
    C.clear();


    A.setY(offsets[1], nY);
    A.setZ(offsets[2], nZ);

    B.setX(offsets[2], nX);
    B.setY(offsets[0], nY);

    C = min(A, B);

    CPPUNIT_ASSERT(C);
    CPPUNIT_ASSERT(C != A);
    CPPUNIT_ASSERT(C != B);

    CPPUNIT_ASSERT(!C.hasOffsetX());
    CPPUNIT_ASSERT(C.hasOffsetY());
    CPPUNIT_ASSERT(!C.hasOffsetZ());

    CPPUNIT_ASSERT(C.isLessY(A));

    C = max(A, B);

    CPPUNIT_ASSERT(C);
    CPPUNIT_ASSERT(C != A);
    CPPUNIT_ASSERT(C != B);

    CPPUNIT_ASSERT(C.hasOffsetX());
    CPPUNIT_ASSERT(C.hasOffsetY());
    CPPUNIT_ASSERT(C.hasOffsetZ());

    CPPUNIT_ASSERT(C.isGreaterX(A));
    CPPUNIT_ASSERT(C.isGreaterY(B));
    CPPUNIT_ASSERT(C.isGreaterZ(B));
}


////////////////////////////////////////


void
TestHermite::testIO()
{
    using namespace openvdb;
    using namespace openvdb::math;

    std::stringstream
        ss(std::stringstream::in | std::stringstream::out | std::stringstream::binary);

    Hermite A, B;

    A.setX(0.50f, Vec3s(1.0, 0.0, 0.0));
    A.setY(0.82f, Vec3s(0.0, 1.0, 0.0));
    A.setZ(0.14f, Vec3s(0.0, 0.0, 1.0));
    A.setIsInside(true);

    CPPUNIT_ASSERT(A);
    CPPUNIT_ASSERT(!B);

    A.write(ss);

    B.read(ss);

    CPPUNIT_ASSERT(A);
    CPPUNIT_ASSERT(B);

    CPPUNIT_ASSERT(A == B);
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
