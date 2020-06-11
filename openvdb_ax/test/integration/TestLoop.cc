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

class TestLoop : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestLoop);
    CPPUNIT_TEST(testLoopForLoop);
    CPPUNIT_TEST(testLoopWhileLoop);
    CPPUNIT_TEST(testLoopDoWhileLoop);
    CPPUNIT_TEST(testLoopOverflow);
    CPPUNIT_TEST_SUITE_END();

    void testLoopForLoop();
    void testLoopWhileLoop();
    void testLoopDoWhileLoop();
    void testLoopOverflow();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLoop);

void
TestLoop::testLoopForLoop()
{
    mHarness.addAttribute<openvdb::Vec3f>("loop_test1",  openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test2",  openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test3",  openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test15", openvdb::Vec3f(0.0,0.0,0.0));
    mHarness.addAttribute<openvdb::math::Mat3s>("loop_test18",
        openvdb::math::Mat3s(1.0,2.0,3.0, 4.0,5.0,6.0, 7.0,8.0,9.0));
    mHarness.addAttribute<int>("loop_test22", 3);
    mHarness.addAttribute<int>("loop_test23", 4);
    mHarness.addAttribute<int>("loop_test25", 1);
    mHarness.addAttribute<int>("loop_test27", 14);
    mHarness.addAttribute<int>("loop_test30", 19);
    mHarness.executeCode("test/snippets/loop/forLoop");

    AXTESTS_STANDARD_ASSERT();
}

void
TestLoop::testLoopWhileLoop()
{
    mHarness.addAttribute<openvdb::Vec3f>("loop_test9",  openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test16", openvdb::Vec3f(0.0,0.0,0.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test28", openvdb::Vec3f(0.0,0.0,0.0));
    mHarness.addAttribute<int>("loop_test31", 2);
    mHarness.executeCode("test/snippets/loop/whileLoop");

    AXTESTS_STANDARD_ASSERT();
}

void
TestLoop::testLoopDoWhileLoop()
{
    mHarness.addAttribute<openvdb::Vec3f>("loop_test12", openvdb::Vec3f(1.0,2.0,3.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test17", openvdb::Vec3f(1.0,0.0,0.0));
    mHarness.addAttribute<openvdb::Vec3f>("loop_test29", openvdb::Vec3f(1.0,0.0,0.0));
    mHarness.addAttribute<int>("loop_test32", 2);
    mHarness.executeCode("test/snippets/loop/doWhileLoop");

    AXTESTS_STANDARD_ASSERT();
}

void
TestLoop::testLoopOverflow()
{
    // Disable all optimizations to force the loop to not remove the interior
    // allocation. The loop should generate its allocas in the function prologue
    // to avoid stack overflow
    openvdb::ax::CompilerOptions opts;
    opts.mOptLevel = openvdb::ax::CompilerOptions::OptLevel::NONE;
    mHarness.mOpts = opts;
    mHarness.executeCode("test/snippets/loop/loopOverflow");
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
