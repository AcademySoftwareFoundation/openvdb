///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
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

#include "TestHarness.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestConditional : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestConditional);
    CPPUNIT_TEST(testConditionalIfWithinElse);
    CPPUNIT_TEST(testConditionalScopingStatement);
    CPPUNIT_TEST(testConditionalSimpleStatement);
    CPPUNIT_TEST(testConditionalSimpleElseIf);
    CPPUNIT_TEST_SUITE_END();

    void testConditionalIfWithinElse();
    void testConditionalSimpleStatement();
    void testConditionalScopingStatement();
    void testConditionalSimpleElseIf();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestConditional);

void
TestConditional::testConditionalIfWithinElse()
{
    mHarness.addAttribute<bool>("bool_test", true);
    mHarness.executeCode("test/snippets/conditional/conditionalIfWithinElse");

    AXTESTS_STANDARD_ASSERT();
}

void
TestConditional::testConditionalSimpleStatement()
{
    mHarness.addAttribute<bool>("bool_test", true);
    mHarness.addAttribute<float>("float_test", 1.0f);

    mHarness.executeCode("test/snippets/conditional/conditionalSimpleStatement");

    AXTESTS_STANDARD_ASSERT();
}

void
TestConditional::testConditionalScopingStatement()
{
    mHarness.addAttribute<int>("int_test", 1);
    mHarness.executeCode("test/snippets/conditional/conditionalScopingStatement");

    AXTESTS_STANDARD_ASSERT();
}

void
TestConditional::testConditionalSimpleElseIf()
{
    mHarness.addAttribute("bool_test", true);
    mHarness.addAttribute("int_test", 2);

    mHarness.executeCode("test/snippets/conditional/conditionalSimpleElseIf");

    AXTESTS_STANDARD_ASSERT();
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
