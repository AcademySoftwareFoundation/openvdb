// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include "../util.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

// Configuration values for binary code

static const unittest_util::ConfigMap integral = {
    { "bool",   { { "_L1_", "true" }, { "_L2_", "false" } }  },
    { "int16",  { { "_L1_", "2" },    { "_L2_", "3" } }      },
    { "int32",  { { "_L1_", "2" },    { "_L2_", "3" } }      },
    { "int64",  { { "_L1_", "2l" },   { "_L2_", "3l" } }     }
};

static const unittest_util::ConfigMap floating = {
    { "float",  { { "_L1_", "1.1f" }, { "_L2_", "2.3f" } }   },
    { "double", { { "_L1_", "1.1" },  { "_L2_", "2.3" } }    }
};

static const unittest_util::ConfigMap vec2 = {
    { "vec2i",  { { "_L1_",  "{1, 2}" },                    { "_L2_", "{3, 4}" } } },
    { "vec2f",  { { "_L1_",  "{1.1f, 2.3f}" },              { "_L2_", "{4.1f, 5.3f}" } } },
    { "vec2d",  { { "_L1_",  "{1.1, 2.3}" },                { "_L2_", "{4.1, 5.3}" } } }
};

static const unittest_util::ConfigMap vec3 = {
    { "vec3i",  { { "_L1_",  "{1, 2, 3}" },                 { "_L2_", "{4, 5, 6}" } } },
    { "vec3f",  { { "_L1_",  "{1.1f, 2.3f, 4.3f}" },        { "_L2_", "{4.1f, 5.3f, 6.3f}" } } },
    { "vec3d",  { { "_L1_",  "{1.1,  2.3 , 4.3}" },         { "_L2_", "{4.1,  5.3, 6.3}" } } }
};

static const unittest_util::ConfigMap vec4 = {
    { "vec4i",  { { "_L1_",  "{1, 2, 3, 4}" },              { "_L2_", "{5, 6, 7, 8}" } } },
    { "vec4f",  { { "_L1_",  "{1.1f, 2.3f, 4.3f, 5.4f}" },  { "_L2_", "{5.1f, 6.3f, 7.3f, 8.4f}" } } },
    { "vec4d",  { { "_L1_",  "{1.1,  2.3, 4.3, 5.4}" },     { "_L2_", "{5.1, 6.3, 7.3, 8.4}" } } }
};

static const unittest_util::ConfigMap mat3 = {
    { "mat3f",  { { "_L1_",  "{1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f }" },
                  { "_L2_",  "{9.1f, 7.3f, -1.3f,  4.4f, -6.7f, 0.8f,  9.1f,-0.5f, 8.2f }" } }
    },
    { "mat3d",  { { "_L1_",  "{1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2 }" },
                  { "_L2_",  "{9.1, 7.3, -1.3,  4.4, -6.7, 0.8,  9.1,-0.5, 8.2 }" } }
    }
};

static const unittest_util::ConfigMap mat4 = {
    { "mat4f",  { { "_L1_",  "{1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f}" },
                  { "_L2_",  "{0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f}" } }
    },
    { "mat4d",  { { "_L1_",  "{1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9}" },
                  { "_L2_",  "{0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7}" } }
    }
};

static const unittest_util::ConfigMap string = {
    { "string",  { { "_L1_", "\"foo\"" }, { "_L2_", "\"bar\"" } }   }
};


class TestBinary : public unittest_util::AXTestCase
{
public:

    std::string dir() const override { return GET_TEST_DIRECTORY(); }

    CPPUNIT_TEST_SUITE(TestBinary);
    CPPUNIT_TEST(plus);
    CPPUNIT_TEST(minus);
    CPPUNIT_TEST(mult);
    CPPUNIT_TEST(div);
    CPPUNIT_TEST(mod);
    CPPUNIT_TEST(btand);
    CPPUNIT_TEST(btor);
    CPPUNIT_TEST(btxor);
    CPPUNIT_TEST(logicaland);
    CPPUNIT_TEST(logicalor);
    CPPUNIT_TEST(equalsequals);
    CPPUNIT_TEST(notequals);
    CPPUNIT_TEST(greaterthan);
    CPPUNIT_TEST(lessthan);
    CPPUNIT_TEST(greaterthanequals);
    CPPUNIT_TEST(lessthanequals);
    CPPUNIT_TEST(shiftleft);
    CPPUNIT_TEST(shiftright);
    CPPUNIT_TEST_SUITE_END();

    void plus();
    void minus();
    void mult();
    void div();
    void mod();
    void btand();
    void btor();
    void btxor();
    void logicaland();
    void logicalor();
    void equalsequals();
    void notequals();
    void greaterthan();
    void lessthan();
    void greaterthanequals();
    void lessthanequals();
    void shiftleft();
    void shiftright();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBinary);


void
TestBinary::plus()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ + _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
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
    generate(string);
    this->registerTest(repl, "binary_plus.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", true); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", 5); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", 5); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", 5); } },
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


void
TestBinary::minus()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ - _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
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
    this->registerTest(repl, "binary_minus.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", true); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", -1); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", -1); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", -1); } },
        { "float",  [&](){ mHarness.addAttribute<float>("testfloat", 1.1f - 2.3f); } },
        { "double", [&](){ mHarness.addAttribute<double>("testdouble", 1.1 - 2.3); } },
        { "vec2i",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(-2,-2)); } },
        { "vec2f",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(1.1f-4.1f, 2.3f-5.3f)); } },
        { "vec2d",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(1.1-4.1, 2.3-5.3)); } },
        { "vec3i",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(-3,-3,-3)); } },
        { "vec3f",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(1.1f-4.1f, 2.3f-5.3f, 4.3f-6.3f)); } },
        { "vec3d",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(1.1-4.1, 2.3-5.3, 4.3-6.3)); } },
        { "vec4i",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(-4,-4,-4,-4)); } },
        { "vec4f",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(1.1f-5.1f, 2.3f-6.3f, 4.3f-7.3f, 5.4f-8.4f)); } },
        { "vec4d",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(1.1-5.1, 2.3-6.3, 4.3-7.3, 5.4-8.4)); } },
        { "mat3f",  [&](){ mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f",
                            openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,  5.4f, 6.7f, 7.8f,  9.1f, 4.5f, 8.2f) -
                            openvdb::math::Mat3<float>(9.1f, 7.3f,-1.3f,  4.4f,-6.7f, 0.8f,  9.1f,-0.5f, 8.2f)); }
        },
        { "mat3d",  [&](){ mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d",
                            openvdb::math::Mat3<double>(1.1, 2.3, 4.3,  5.4, 6.7, 7.8,  9.1, 4.5, 8.2) -
                            openvdb::math::Mat3<double>(9.1, 7.3,-1.3,  4.4,-6.7, 0.8,  9.1,-0.5, 8.2)); }
        },
        { "mat4f",  [&](){ mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f",
                            openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f) -
                            openvdb::math::Mat4<float>(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f)); }
        },
        { "mat4d",  [&](){ mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d",
                            openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9) -
                            openvdb::math::Mat4<double>(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7)); }
        },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_minus.ax");
}

void
TestBinary::mult()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ * _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
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
    this->registerTest(repl, "binary_mult.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", false); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", 6); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", 6); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", 6); } },
        { "float",  [&](){ mHarness.addAttribute<float>("testfloat", 1.1f * 2.3f); } },
        { "double", [&](){ mHarness.addAttribute<double>("testdouble", 1.1 * 2.3); } },
        { "vec2i",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(3,8)); } },
        { "vec2f",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(1.1f*4.1f, 2.3f*5.3f)); } },
        { "vec2d",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(1.1*4.1, 2.3*5.3)); } },
        { "vec3i",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(4,10,18)); } },
        { "vec3f",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(1.1f*4.1f, 2.3f*5.3f, 4.3f*6.3f)); } },
        { "vec3d",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(1.1*4.1, 2.3*5.3, 4.3*6.3)); } },
        { "vec4i",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(5,12,21,32)); } },
        { "vec4f",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(1.1f*5.1f, 2.3f*6.3f, 4.3f*7.3f, 5.4f*8.4f)); } },
        { "vec4d",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(1.1*5.1, 2.3*6.3, 4.3*7.3, 5.4*8.4)); } },
        { "mat3f",  [&](){ mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f",
                            openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,  5.4f, 6.7f, 7.8f,  9.1f, 4.5f, 8.2f) *
                            openvdb::math::Mat3<float>(9.1f, 7.3f,-1.3f,  4.4f,-6.7f, 0.8f,  9.1f,-0.5f, 8.2f)); }
        },
        { "mat3d",  [&](){ mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d",
                            openvdb::math::Mat3<double>(1.1, 2.3, 4.3,  5.4, 6.7, 7.8,  9.1, 4.5, 8.2) *
                            openvdb::math::Mat3<double>(9.1, 7.3,-1.3,  4.4,-6.7, 0.8,  9.1,-0.5, 8.2)); }
        },
        { "mat4f",  [&](){ mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f",
                            openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f) *
                            openvdb::math::Mat4<float>(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f)); }
        },
        { "mat4d",  [&](){ mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d",
                            openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9) *
                            openvdb::math::Mat4<double>(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7)); }
        }
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_mult.ax");
}


void
TestBinary::div()
{
    // @note  reverses L1 and L2 as L2 is usually larger
    const std::string code = R"(
_T1_@_A1_ = _L2_ / _L1_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
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
    this->registerTest(repl, "binary_div.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", false); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", 1); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", 1); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", 1); } },
        { "float",  [&](){ mHarness.addAttribute<float>("testfloat", 2.3f/1.1f); } },
        { "double", [&](){ mHarness.addAttribute<double>("testdouble", 2.3/1.1); } },
        { "vec2i",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(3,2)); } },
        { "vec2f",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(4.1f/1.1f, 5.3f/2.3f)); } },
        { "vec2d",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(4.1/1.1, 5.3/2.3)); } },
        { "vec3i",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(4,2,2)); } },
        { "vec3f",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(4.1f/1.1f, 5.3f/2.3f, 6.3f/4.3f)); } },
        { "vec3d",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(4.1/1.1, 5.3/2.3, 6.3/4.3)); } },
        { "vec4i",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(5,3,2,2)); } },
        { "vec4f",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(5.1f/1.1f, 6.3f/2.3f, 7.3f/4.3f, 8.4f/5.4f)); } },
        { "vec4d",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(5.1/1.1, 6.3/2.3, 7.3/4.3, 8.4/5.4)); } },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_div.ax");
}


void
TestBinary::mod()
{
    // @note  reverses L1 and L2 as L2 is usually larger
    const std::string code = R"(
_T1_@_A1_ = _L2_ % _L1_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
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
    this->registerTest(repl, "binary_mod.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", false); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", 1); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", 1); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", 1); } },
        { "float",  [&](){ mHarness.addAttribute<float>("testfloat", std::fmod(2.3f,1.1f)); } },
        { "double", [&](){ mHarness.addAttribute<double>("testdouble", std::fmod(2.3,1.1)); } },
        { "vec2i",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(0,0)); } },
        { "vec2f",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(std::fmod(4.1f,1.1f), std::fmod(5.3f,2.3f))); } },
        { "vec2d",  [&](){ mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(std::fmod(4.1,1.1), std::fmod(5.3,2.3))); } },
        { "vec3i",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(0,1,0)); } },
        { "vec3f",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(std::fmod(4.1f,1.1f), std::fmod(5.3f,2.3f), std::fmod(6.3f,4.3f))); } },
        { "vec3d",  [&](){ mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(std::fmod(4.1,1.1), std::fmod(5.3,2.3), std::fmod(6.3,4.3))); } },
        { "vec4i",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(0,0,1,0)); } },
        { "vec4f",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(std::fmod(5.1f,1.1f), std::fmod(6.3f,2.3f), std::fmod(7.3f,4.3f), std::fmod(8.4f,5.4f))); } },
        { "vec4d",  [&](){ mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(std::fmod(5.1,1.1), std::fmod(6.3,2.3), std::fmod(7.3,4.3), std::fmod(8.4,5.4))); } },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_mod.ax");
}


void
TestBinary::btand()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ & _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    this->registerTest(repl, "binary_bitand.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", false); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", 2); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", 2); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", 2); } },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_bitand.ax");
}


void
TestBinary::btor()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ | _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    this->registerTest(repl, "binary_bitor.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",  [&](){ mHarness.addAttribute<bool>("testbool", true); } },
        { "int16", [&](){ mHarness.addAttribute<int16_t>("testint16", 3); } },
        { "int32", [&](){ mHarness.addAttribute<int32_t>("testint32", 3); } },
        { "int64", [&](){ mHarness.addAttribute<int64_t>("testint64", 3); } },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_bitor.ax");
}


void
TestBinary::btxor()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ ^ _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    this->registerTest(repl, "binary_bitxor.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",  [&](){ mHarness.addAttribute<bool>("testbool", true); } },
        { "int16", [&](){ mHarness.addAttribute<int16_t>("testint16", 1); } },
        { "int32", [&](){ mHarness.addAttribute<int32_t>("testint32", 1); } },
        { "int64", [&](){ mHarness.addAttribute<int64_t>("testint64", 1); } },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_bitxor.ax");
}


void
TestBinary::logicaland()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ && _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    generate(floating);
    this->registerTest(repl, "binary_logicaland.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", false); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", 1); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", 1); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", 1); } },
        { "float",  [&](){ mHarness.addAttribute<float>("testfloat", 1.0f); } },
        { "double", [&](){ mHarness.addAttribute<double>("testdouble", 1.0); } },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_logicaland.ax");

    // Also test short circuiting logical op for &&

    mHarness.reset();
    this->registerTest(R"(
int@scircuit1 = 0;
int@scircuit2 = 1;
int@scircuit3 = 2;
int@scircuit1++ && ++int@scircuit2;
++int@scircuit1 && ++int@scircuit3;
int@scircuit4 = 1;
int@scircuit5 = 1;
true && int@scircuit4 = 2;
false && int@scircuit5 = 2;)",
    "binary_logicaland_scircuit.ax");

    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("scircuit", 5), { 2, 1, 3, 2, 1 });
    this->execute("binary_logicaland_scircuit.ax");
}


void
TestBinary::logicalor()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ || _L2_;)";

    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", config.first);
            unittest_util::replace(tmp, "_A1_", "test" + config.first);
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    generate(floating);
    this->registerTest(repl, "binary_logicalor.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){ mHarness.addAttribute<bool>("testbool", true); } },
        { "int16",  [&](){ mHarness.addAttribute<int16_t>("testint16", 1); } },
        { "int32",  [&](){ mHarness.addAttribute<int32_t>("testint32", 1); } },
        { "int64",  [&](){ mHarness.addAttribute<int64_t>("testint64", 1); } },
        { "float",  [&](){ mHarness.addAttribute<float>("testfloat", 1.0f); } },
        { "double", [&](){ mHarness.addAttribute<double>("testdouble", 1.0); } },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_logicalor.ax");

    // Also test short circuiting logical op for ||

    mHarness.reset();
    this->registerTest(R"(
int@scircuit1 = 0;
int@scircuit2 = 1;
int@scircuit3 = 2;
int@scircuit1++ || ++int@scircuit2;
++int@scircuit1 || ++int@scircuit3;
int@scircuit4 = 1;
int@scircuit5 = 1;
true || int@scircuit4 = 2;
false || int@scircuit5 = 2;)",
    "binary_logicalor_scircuit.ax");

    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("scircuit", 5), { 2, 2, 2, 1, 2 });
    this->execute("binary_logicalor_scircuit.ax");
}


void
TestBinary::equalsequals()
{
    const std::string code = R"(
bool@_A1_ = _L1_ == _L2_;
bool@_A2_ = _L2_ == _L2_;)";

    size_t idx = 1;
    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_A1_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A2_", "test" + std::to_string(idx++));
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
    this->registerTest(repl, "binary_relational_equalsequals.ax");

    CPPUNIT_ASSERT(idx != 0);

    const auto names = unittest_util::nameSequence("test", idx-1);
    std::vector<bool> results;
    for (size_t i = 0; i < idx-1; ++i) {
        results.emplace_back((i % 2 == 0) ? false : true);
    }

    mHarness.addAttributes<bool>(names, results);
    this->execute("binary_relational_equalsequals.ax");
}


void
TestBinary::notequals()
{
    const std::string code = R"(
bool@_A1_ = _L1_ != _L2_;
bool@_A2_ = _L2_ != _L2_;)";

    size_t idx = 1;
    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_A1_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A2_", "test" + std::to_string(idx++));
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
    this->registerTest(repl, "binary_relational_notequals.ax");

    CPPUNIT_ASSERT(idx != 0);

    const auto names = unittest_util::nameSequence("test", idx-1);
    std::vector<bool> results;
    for (size_t i = 0; i < idx-1; ++i) {
        results.emplace_back((i % 2 == 1) ? false : true);
    }

    mHarness.addAttributes<bool>(names, results);
    this->execute("binary_relational_notequals.ax");
}


void
TestBinary::greaterthan()
{
    const std::string code = R"(
bool@_A1_ = _L1_ > _L2_;
bool@_A2_ = _L2_ > _L1_;
bool@_A3_ = _L2_ > _L2_;)";

    size_t idx = 1;
    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_A1_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A2_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A3_", "test" + std::to_string(idx++));
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    generate(floating);
    this->registerTest(repl, "binary_relational_greaterthan.ax");

    CPPUNIT_ASSERT(idx != 0);

    const auto names = unittest_util::nameSequence("test", idx-1);

    std::vector<bool> results;
    for (const auto& config : integral) {
        if (config.first == "bool") {
            // L1 and L2 for bools are swapped
            results.emplace_back(true);
            results.emplace_back(false);
            results.emplace_back(false);
        }
        else {
            results.emplace_back(false);
            results.emplace_back(true);
            results.emplace_back(false);
        }
    }

    const size_t typecount = floating.size();
    for (size_t i = 0; i < typecount; ++i) {
        results.emplace_back(false);
        results.emplace_back(true);
        results.emplace_back(false);
    }

    mHarness.addAttributes<bool>(names, results);
    this->execute("binary_relational_greaterthan.ax");
}


void
TestBinary::lessthan()
{
    const std::string code = R"(
bool@_A1_ = _L1_ < _L2_;
bool@_A2_ = _L2_ < _L1_;
bool@_A3_ = _L2_ < _L2_;)";

    size_t idx = 1;
    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_A1_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A2_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A3_", "test" + std::to_string(idx++));
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    generate(floating);
    this->registerTest(repl, "binary_relational_lessthan.ax");

    CPPUNIT_ASSERT(idx != 0);

    const auto names = unittest_util::nameSequence("test", idx-1);

    std::vector<bool> results;
    for (const auto& config : integral) {
        if (config.first == "bool") {
            // L1 and L2 for bools are swapped
            results.emplace_back(false);
            results.emplace_back(true);
            results.emplace_back(false);
        }
        else {
            results.emplace_back(true);
            results.emplace_back(false);
            results.emplace_back(false);
        }
    }

    const size_t typecount = floating.size();
    for (size_t i = 0; i < typecount; ++i) {
        results.emplace_back(true);
        results.emplace_back(false);
        results.emplace_back(false);
    }

    mHarness.addAttributes<bool>(names, results);
    this->execute("binary_relational_lessthan.ax");
}



void
TestBinary::greaterthanequals()
{
    const std::string code = R"(
bool@_A1_ = _L1_ >= _L2_;
bool@_A2_ = _L2_ >= _L1_;
bool@_A3_ = _L2_ >= _L2_;)";

    size_t idx = 1;
    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_A1_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A2_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A3_", "test" + std::to_string(idx++));
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    generate(floating);
    this->registerTest(repl, "binary_relational_greaterthanequals.ax");

    CPPUNIT_ASSERT(idx != 0);

    const auto names = unittest_util::nameSequence("test", idx-1);

    std::vector<bool> results;
    for (const auto& config : integral) {
        if (config.first == "bool") {
            // L1 and L2 for bools are swapped
            results.emplace_back(true);
            results.emplace_back(false);
            results.emplace_back(true);
        }
        else {
            results.emplace_back(false);
            results.emplace_back(true);
            results.emplace_back(true);
        }
    }

    const size_t typecount = floating.size();
    for (size_t i = 0; i < typecount; ++i) {
        results.emplace_back(false);
        results.emplace_back(true);
        results.emplace_back(true);
    }

    mHarness.addAttributes<bool>(names, results);
    this->execute("binary_relational_greaterthanequals.ax");
}


void
TestBinary::lessthanequals()
{
    const std::string code = R"(
bool@_A1_ = _L1_ <= _L2_;
bool@_A2_ = _L2_ <= _L1_;
bool@_A3_ = _L2_ <= _L2_;)";

    size_t idx = 1;
    std::string repl;
    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_A1_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A2_", "test" + std::to_string(idx++));
            unittest_util::replace(tmp, "_A3_", "test" + std::to_string(idx++));
            for (const auto& settings : config.second) {
                unittest_util::replace(tmp, settings.first, settings.second);
            }
            repl += tmp;
        }
    };

    generate(integral);
    generate(floating);
    this->registerTest(repl, "binary_relational_lessthanequals.ax");

    CPPUNIT_ASSERT(idx != 0);

    const auto names = unittest_util::nameSequence("test", idx-1);

    std::vector<bool> results;
    for (const auto& config : integral) {
        if (config.first == "bool") {
            // L1 and L2 for bools are swapped
            results.emplace_back(false);
            results.emplace_back(true);
            results.emplace_back(true);
        }
        else {
            results.emplace_back(true);
            results.emplace_back(false);
            results.emplace_back(true);
        }
    }

    const size_t typecount = floating.size();
    for (size_t i = 0; i < typecount; ++i) {
        results.emplace_back(true);
        results.emplace_back(false);
        results.emplace_back(true);
    }

    mHarness.addAttributes<bool>(names, results);
    this->execute("binary_relational_lessthanequals.ax");
}


void
TestBinary::shiftleft()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ << _L2_;
_T1_@_A2_ = _L2_ << _L1_;)";

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
    this->registerTest(repl, "binary_shiftleft.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){
                mHarness.addAttribute<bool>("testbool1", true);
                mHarness.addAttribute<bool>("testbool2", false);
            }
        },
        { "int16",  [&](){
                mHarness.addAttribute<int16_t>("testint161", 16);
                mHarness.addAttribute<int16_t>("testint162", 12);
            }
        },
        { "int32",    [&](){
                mHarness.addAttribute<int32_t>("testint321", 16);
                mHarness.addAttribute<int32_t>("testint322", 12);
            }
        },
        { "int64",   [&](){
                mHarness.addAttribute<int64_t>("testint641", 16);
                mHarness.addAttribute<int64_t>("testint642", 12);
            }
        },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_shiftleft.ax");
}


void
TestBinary::shiftright()
{
    const std::string code = R"(
_T1_@_A1_ = _L1_ >> _L2_;
_T1_@_A2_ = _L2_ >> _L1_;)";

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
    this->registerTest(repl, "binary_shiftright.ax");

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",   [&](){
                mHarness.addAttribute<bool>("testbool1", true);
                mHarness.addAttribute<bool>("testbool2", false);
            }
        },
        { "int16",  [&](){
                mHarness.addAttribute<int16_t>("testint161", 0);
                mHarness.addAttribute<int16_t>("testint162", 0);
            }
        },
        { "int",    [&](){
                mHarness.addAttribute<int32_t>("testint321", 0);
                mHarness.addAttribute<int32_t>("testint322", 0);
            }
        },
        { "int64",   [&](){
                mHarness.addAttribute<int64_t>("testint641", 0);
                mHarness.addAttribute<int64_t>("testint642", 0);
            }
        },
    };

    for (const auto& expc : expected) {
        expc.second.operator()();
    }
    this->execute("binary_shiftright.ax");
}

