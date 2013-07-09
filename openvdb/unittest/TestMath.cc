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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/math/Math.h>


class TestMath: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMath);
    CPPUNIT_TEST(testAll);
    CPPUNIT_TEST_SUITE_END();

    void testAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMath);

// This suit of tests obviously needs to be expanded!
void
TestMath::testAll()
{
    using namespace openvdb;

    {// Sign
        CPPUNIT_ASSERT_EQUAL(math::Sign( 3   ), 1);
        CPPUNIT_ASSERT_EQUAL(math::Sign(-1.0 ),-1);
        CPPUNIT_ASSERT_EQUAL(math::Sign( 0.0f), 0);
    }
    {// SignChange
        CPPUNIT_ASSERT( math::SignChange( -1, 1));
        CPPUNIT_ASSERT(!math::SignChange( 0.0f, 0.5f));
        CPPUNIT_ASSERT( math::SignChange( 0.0f,-0.5f));
        CPPUNIT_ASSERT( math::SignChange(-0.1, 0.0001));

    }
    {// isApproxZero
        CPPUNIT_ASSERT( math::isApproxZero( 0.0f) );
        CPPUNIT_ASSERT( math::isApproxZero( 0.0000009f) );
        CPPUNIT_ASSERT( math::isApproxZero(-0.0000009f) );
        CPPUNIT_ASSERT( math::isApproxZero( 0.01, 0.1) );
    }
    {// Cbrt
        const double a = math::Cbrt(3.0);
        CPPUNIT_ASSERT(math::isApproxEqual(a*a*a, 3.0, 1e-6));
    }
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
