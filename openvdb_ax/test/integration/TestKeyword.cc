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

class TestKeyword : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestKeyword);
    CPPUNIT_TEST(testKeywordSimpleReturn);
    CPPUNIT_TEST(testKeywordReturnBranchIf);
    CPPUNIT_TEST(testKeywordReturnBranchLoop);
    CPPUNIT_TEST(testKeywordConditionalReturn);
    CPPUNIT_TEST(testKeywordForLoopKeywords);
    CPPUNIT_TEST(testKeywordWhileLoopKeywords);
    CPPUNIT_TEST(testKeywordDoWhileLoopKeywords);
    CPPUNIT_TEST_SUITE_END();

    void testKeywordSimpleReturn();
    void testKeywordReturnBranchIf();
    void testKeywordReturnBranchLoop();
    void testKeywordConditionalReturn();
    void testKeywordForLoopKeywords();
    void testKeywordWhileLoopKeywords();
    void testKeywordDoWhileLoopKeywords();

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestKeyword);

void
TestKeyword::testKeywordSimpleReturn()
{
    mHarness.addAttribute<int>("return_test0", 0);
    mHarness.executeCode("test/snippets/keyword/simpleReturn");

    AXTESTS_STANDARD_ASSERT();
}

void
TestKeyword::testKeywordReturnBranchIf()
{
    mHarness.addAttribute<int>("return_test1", 1);
    mHarness.executeCode("test/snippets/keyword/returnBranchIf");

    AXTESTS_STANDARD_ASSERT();
}

void
TestKeyword::testKeywordReturnBranchLoop()
{
    mHarness.addAttribute<int>("return_test2", 1);
    mHarness.executeCode("test/snippets/keyword/returnBranchLoop");

    AXTESTS_STANDARD_ASSERT();
}

void
TestKeyword::testKeywordConditionalReturn()
{
    mHarness.addAttribute<int>("return_test3", 3);
    mHarness.executeCode("test/snippets/keyword/conditionalReturn");

    AXTESTS_STANDARD_ASSERT();
}

void
TestKeyword::testKeywordForLoopKeywords()
{
    mHarness.addAttribute<openvdb::Vec3f>("loop_test4", openvdb::Vec3f(1.0,0.0,0.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test5", openvdb::Vec3f(1.0,0.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test6", openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test7", openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test8", openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::math::Mat3s>("loop_test19",
        openvdb::math::Mat3s(1.0,2.0,3.0, 0.0,0.0,0.0, 7.0,8.0,9.0));
    mHarness.addAttribute<openvdb::math::Mat3s>("loop_test20",
        openvdb::math::Mat3s(1.0,0.0,0.0, 0.0,0.0,0.0, 7.0,0.0,0.0));
    mHarness.addAttribute<openvdb::math::Mat3s>("loop_test21",
        openvdb::math::Mat3s(1.0,0.0,3.0, 0.0,0.0,0.0, 7.0,0.0,9.0));
    mHarness.addAttribute<openvdb::Vec3f>("return_test4", openvdb::Vec3f(10,10,10));
    mHarness.executeCode("test/snippets/keyword/forLoopKeywords");

    AXTESTS_STANDARD_ASSERT();
}

void
TestKeyword::testKeywordWhileLoopKeywords()
{
    mHarness.addAttribute<openvdb::Vec3f>("loop_test10", openvdb::Vec3f(1.0,0.0,0.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test11", openvdb::Vec3f(0.0,0.0,2.0));
    mHarness.addAttribute<openvdb::Vec3f>("return_test6", openvdb::Vec3f(100,100,100));
    mHarness.executeCode("test/snippets/keyword/whileLoopKeywords");

    AXTESTS_STANDARD_ASSERT();
}


void
TestKeyword::testKeywordDoWhileLoopKeywords()
{
    mHarness.addAttribute<openvdb::Vec3f>("loop_test13", openvdb::Vec3f(1.0,0.0,0.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test14", openvdb::Vec3f(0.0,0.0,2.0));
    mHarness.addAttribute<openvdb::Vec3f>("return_test7", openvdb::Vec3f(100,100,100));
    mHarness.executeCode("test/snippets/keyword/doWhileLoopKeywords");

    AXTESTS_STANDARD_ASSERT();
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
