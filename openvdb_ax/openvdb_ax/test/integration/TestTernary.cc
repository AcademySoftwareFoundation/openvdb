// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestTernary : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestTernary);
    CPPUNIT_TEST(testTernary);
    CPPUNIT_TEST(testTernaryVoid);
    CPPUNIT_TEST(testTernaryErrors);
    CPPUNIT_TEST_SUITE_END();

    void testTernary();
    void testTernaryVoid();
    void testTernaryErrors();
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

void
TestTernary::testTernaryErrors()
{
    const bool success = mHarness.executeCode("test/snippets/ternary/ternaryErrors");
    CPPUNIT_ASSERT(!success);
}

