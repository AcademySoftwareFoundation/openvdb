// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"
#include "util.h"

#include <cppunit/extensions/HelperMacros.h>


class TestUnary : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestUnary);
    CPPUNIT_TEST(minus);
    CPPUNIT_TEST(bitnot);
    CPPUNIT_TEST(logicalnot);
    CPPUNIT_TEST_SUITE_END();

    void minus();
    void bitnot();
    void logicalnot();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestUnary);

void
TestUnary::minus()
{
    const std::string code = R"(
_T1_@_A1_ = -_T1_@_A2_ ;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first + "1");
            unittest_util::replace(tmp, "_A2_", "test" + config.first + "2");
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    generate(floating);
    generate(vec2);
    generate(vec3);
    generate(vec4);
    generate(mat3);
    generate(mat4);
    this->registerTest(repl, "unary_minus.ax");

    auto add = [](auto in, auto out) {
        using T = decltype(in);
        static_assert(std::is_same<T, decltype(out)>::value);
        const std::string type = typeStringFromToken(tokenFromType<T>());
        mHarness.addAttribute<T>("test" + type + "1", out);
        mHarness.addAttribute<T>("test" + type + "2", in, in);
    };

    const std::map<std::string, std::vector<std::function<void()>>> expected = {
        { "bool",   {
                [&](){ add(true, true); },
                [&](){ add(false, false); },
            }
        },
        { "short",  [&](){ mHarness.addAttribute<int16_t>("testshort", 5); } },
        { "int",    [&](){ mHarness.addAttribute<int32_t>("testint", 5); } },
        { "long",   [&](){ mHarness.addAttribute<int64_t>("testlong", 5); } },
        { "float",  [&](){ mHarness.addAttribute<float>("testfloat", 1.1f + 2.3f); } },
        { "double", [&](){ mHarness.addAttribute<double>("testdouble", 1.1 + 2.3); } },
        { "vec2i",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(4,6)); } },
        { "vec2f",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(1.1f+4.1f, 2.3f+5.3f)); } },
        { "vec2d",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(1.1+4.1, 2.3+5.3)); } },
        { "vec3i",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(5,7,9)); } },
        { "vec3f",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(1.1f+4.1f, 2.3f+5.3f, 4.3f+6.3f)); } },
        { "vec3d",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(1.1+4.1, 2.3+5.3, 4.3+6.3)); } },
        { "vec4i",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(6,8,10,12)); } },
        { "vec4f",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(1.1f+5.1f, 2.3f+6.3f, 4.3f+7.3f, 5.4f+8.4f)); } },
        { "vec4d",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(1.1+5.1, 2.3+6.3, 4.3+7.3, 5.4+8.4)); } },
        { "mat3f",  [&](){ mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f",
                            openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,  5.4f, 6.7f, 7.8f,  9.1f, 4.5f, 8.2f) +
                            openvdb::math::Mat3<float>(9.1f, 7.3f,-1.3f,  4.4f,-6.7f, 0.8f,  9.1f,-0.5f, 8.2f)); }
        },
        { "mat3d",  [&](){ mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d",
                            openvdb::math::Mat3<double>(1.1, 2.3, 4.3,  5.4, 6.7, 7.8,  9.1, 4.5, 8.2) +
                            openvdb::math::Mat3<double>(9.1, 7.3,-1.3,  4.4,-6.7, 0.8,  9.1,-0.5, 8.2)); }
        },
        { "mat4f",  [&](){ mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f",
                            openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f) +
                            openvdb::math::Mat4<float>(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f)); }
        },
        { "mat4d",  [&](){ mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d",
                            openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9) +
                            openvdb::math::Mat4<double>(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7)); }
        },
        { "string", [&](){ mHarness.addAttribute<std::string>("teststring", "foobar"); } }
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_plus.ax");
}
