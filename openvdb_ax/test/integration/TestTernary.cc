///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2019 DNEG
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

class TestTernary : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestTernary);
    CPPUNIT_TEST(testTernary);
    CPPUNIT_TEST(testTernaryVoid);
    CPPUNIT_TEST_SUITE_END();

    void testTernary();
    void testTernaryVoid();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTernary);

void
TestTernary::testTernary()
{
    mHarness.addAttribute<bool>("ternary_test1", true);
    mHarness.addAttribute<bool>("ternary_test2", true);
    mHarness.addAttribute<int>("ternary_test3", 3);
    mHarness.addAttribute<int>("ternary_test4", 1);
    mHarness.addAttribute<int>("ternary_test5", 2);
    mHarness.addAttribute<float>("ternary_test6", 10.f);
    mHarness.addAttribute<double>("ternary_test7", 0.75);
    mHarness.addAttribute<openvdb::Vec3i>("ternary_test8", openvdb::Vec3i(1,2,3));
    mHarness.addAttribute<openvdb::Vec3d>("ternary_test9", openvdb::Vec3f(4.5,5.5,6.5));
    mHarness.addAttribute<int>("ternary_test10", 1);
    mHarness.addAttribute<int>("ternary_test11", 123);
    mHarness.addAttribute<int>("ternary_test12", 2);
    mHarness.addAttribute<int>("ternary_test13", 2);
    mHarness.addAttribute<int>("ternary_test14", 123);
    mHarness.addAttribute<float>("ternary_test15", 2.f);
    mHarness.addAttribute<float>("ternary_test16", 1.5f);
    mHarness.addAttribute<openvdb::Vec3i>("ternary_test17", openvdb::Vec3i(1,2,3));
    mHarness.addAttribute<openvdb::Vec3i>("ternary_test18", openvdb::Vec3i(4,5,6));
    mHarness.addAttribute<std::string>("ternary_test19", "foo");
    mHarness.addAttribute<std::string>("ternary_test20", "foo");
    mHarness.addAttribute<std::string>("ternary_test21", "bar");
    mHarness.addAttribute<openvdb::Vec3f>("ternary_test22", openvdb::Vec3f(1.5f,1.5f,1.5f));
    mHarness.addAttribute<openvdb::Vec3f>("ternary_test23", openvdb::Vec3f(1.6f,1.6f,1.6f));
    mHarness.addAttribute<openvdb::math::Mat3<double>>("ternary_test24",
                                                      openvdb::math::Mat3<double>(1.8,0.0,0.0,
                                                                                  0.0,1.8,0.0,
                                                                                  0.0,0.0,1.8));
    mHarness.addAttribute<openvdb::math::Mat3<double>>("ternary_test25",
                                                      openvdb::math::Mat3<double>(1.9,0.0,0.0,
                                                                                  0.0,1.9,0.0,
                                                                                  0.0,0.0,1.9));
    mHarness.addAttribute<openvdb::math::Mat4<double>>("ternary_test26",
                                                      openvdb::math::Mat4<double>(1.8,0.0,0.0,0.0,
                                                                                  0.0,1.8,0.0,0.0,
                                                                                  0.0,0.0,1.8,0.0,
                                                                                  0.0,0.0,0.0,1.8));
    mHarness.addAttribute<openvdb::math::Mat4<double>>("ternary_test27",
                                                      openvdb::math::Mat4<double>(1.9,0.0,0.0,0.0,
                                                                                  0.0,1.9,0.0,0.0,
                                                                                  0.0,0.0,1.9,0.0,
                                                                                  0.0,0.0,0.0,1.9));
    mHarness.addAttribute<openvdb::Vec3f>("ternary_test28", openvdb::Vec3f(1.76f,1.76f,1.76f));
    mHarness.addAttribute<openvdb::Vec3f>("ternary_test29", openvdb::Vec3f(1.76f,1.76f,1.76f));
    mHarness.addAttribute<float>("ternary_test30", openvdb::Vec3f(1.3f,1.3f,1.3f).length());
    mHarness.addAttribute<float>("ternary_test31", openvdb::Vec3f(1.3f,1.3f,1.3f).length());
    mHarness.addAttribute<float>("ternary_test32", openvdb::Vec3f(1.5f,2.5f,3.5f).length());
    mHarness.addAttribute<float>("ternary_test33", openvdb::Vec3f(1.5f,2.5f,3.5f).length());

    mHarness.executeCode("test/snippets/ternary/ternary");

    AXTESTS_STANDARD_ASSERT();
}

void
TestTernary::testTernaryVoid()
{
    mHarness.testVolumes(false);
    mHarness.addExpectedGroups({"notdead"}, {true});
    mHarness.executeCode("test/snippets/ternary/ternaryVoid");

    AXTESTS_STANDARD_ASSERT();
}

// Copyright (c) 2015-2019 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
