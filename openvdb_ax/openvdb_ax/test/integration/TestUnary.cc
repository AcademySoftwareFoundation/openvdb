// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"
#include "../util.h"


class TestUnary : public unittest_util::AXTestCase
{
};


TEST_F(TestUnary, testBitwiseNot)
{
    mHarness.addAttributes<int>({"int_test", "int_test2"}, {-9, -8});

    mHarness.executeCode("test/snippets/unary/unaryBitwiseNot");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestUnary, testNegate)
{
    mHarness.addAttribute<int>("int_test", -3);
    mHarness.addAttribute<float>("float_test", -5.5f);

    mHarness.executeCode("test/snippets/unary/unaryNegate");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestUnary, testNot)
{
    mHarness.addAttributes<bool>({"bool_test", "bool_test2"}, {false, true});

    mHarness.executeCode("test/snippets/unary/unaryNot");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestUnary, testUnaryVector)
{
    // vec3

    mHarness.addAttributes<openvdb::math::Vec3<int32_t>>
        (unittest_util::nameSequence("v3i", 4), {
            openvdb::math::Vec3<int32_t>(0, 1,-1),
            openvdb::math::Vec3<int32_t>(0,-1, 1),
            openvdb::math::Vec3<int32_t>(-1,-2,0),
            openvdb::math::Vec3<int32_t>(1, 0, 0)
        });

    mHarness.addAttributes<openvdb::math::Vec3<float>>
        (unittest_util::nameSequence("v3f", 2), {
            openvdb::math::Vec3<float>(0.0f, 1.1f,-1.1f),
            openvdb::math::Vec3<float>(0.0f,-1.1f, 1.1f),
        });

    mHarness.addAttributes<openvdb::math::Vec3<double>>
        (unittest_util::nameSequence("v3d", 2), {
            openvdb::math::Vec3<double>(0.0, 1.1,-1.1),
            openvdb::math::Vec3<double>(0.0,-1.1, 1.1),
        });

    // vec4

    mHarness.addAttributes<openvdb::math::Vec4<int32_t>>
        (unittest_util::nameSequence("v4i", 4), {
            openvdb::math::Vec4<int32_t>(0, 1,-1, 2),
            openvdb::math::Vec4<int32_t>(0,-1, 1, -2),
            openvdb::math::Vec4<int32_t>(-1,-2,0,-3),
            openvdb::math::Vec4<int32_t>(1, 0, 0, 0)
        });

    mHarness.addAttributes<openvdb::math::Vec4<float>>
        (unittest_util::nameSequence("v4f", 2), {
            openvdb::math::Vec4<float>(0.0f, 1.1f,-1.1f, 2.1f),
            openvdb::math::Vec4<float>(0.0f,-1.1f, 1.1f, -2.1f)
        });

    mHarness.addAttributes<openvdb::math::Vec4<double>>
        (unittest_util::nameSequence("v4d", 2), {
            openvdb::math::Vec4<double>(0.0, 1.1,-1.1, 2.1),
            openvdb::math::Vec4<double>(0.0,-1.1, 1.1, -2.1)
        });

    mHarness.executeCode("test/snippets/unary/unaryVector");

    AXTESTS_STANDARD_ASSERT();
}

