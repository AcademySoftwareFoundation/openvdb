// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "CompareGrids.h"
#include "TestHarness.h"

#include "../util.h"

#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/Exceptions.h>

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestArrayUnpack : public unittest_util::AXTestCase
{
public:

    std::string dir() const override { return GET_TEST_DIRECTORY(); }

    CPPUNIT_TEST_SUITE(TestArrayUnpack);
    CPPUNIT_TEST(componentVectorAssignment);
    CPPUNIT_TEST(componentMatrixAssignment);
    CPPUNIT_TEST_SUITE_END();

    void componentVectorAssignment();
    void componentMatrixAssignment();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestArrayUnpack);

void
TestArrayUnpack::componentVectorAssignment()
{
    const std::string code = R"(
vec2@test1[0] = vec2@test2[1];
vec2@test2[1] = vec2@test1[0];

vec3@test3[1] = vec3@test3[2];
vec3@test4[2] = vec3@test4[0];
vec3@test3[0] = vec3@test4[1];

vec4@test5[0] = vec4@test6[2];
vec4@test5[3] = vec4@test5[1];
vec4@test5[2] = vec4@test6[3];
vec4@test6[1] = vec4@test6[0];
)";

    auto generate = [&](const auto& suffixes) {
        for (const auto& s : suffixes) {
            std::string repl = code;
            const std::string type = (s == 'i' ? "int" : (s == 'f' ? "float" : (s == 'd' ? "double" : "")));
            CPPUNIT_ASSERT(!type.empty());

            unittest_util::replace(repl, "vec2", std::string("vec2").append(1, s));
            unittest_util::replace(repl, "vec3", std::string("vec3").append(1, s));
            unittest_util::replace(repl, "vec4", std::string("vec4").append(1, s));
            this->registerTest(repl, "array_unpack.vec." + type + ".ax");

            unittest_util::replace(repl, "[0]", ".x");
            unittest_util::replace(repl, "[1]", ".y");
            unittest_util::replace(repl, "[2]", ".z");
            this->registerTest(repl, "array_unpack.vec." + type + ".xyz" + ".ax");

            unittest_util::replace(repl, ".x", ".r");
            unittest_util::replace(repl, ".y", ".g");
            unittest_util::replace(repl, ".z", ".b");
            this->registerTest(repl, "array_unpack.vec." + type + ".rgb" + ".ax");
        }
    };

    generate(std::vector<char>{'i', 'f', 'd'});

    const std::map<std::string, std::function<void()>> expected = {
        { "int", [&]() {
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("test1", openvdb::math::Vec2<int32_t>( 1, 2), openvdb::math::Vec2<int32_t>( 4,2));
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("test2", openvdb::math::Vec2<int32_t>( 3, 4), openvdb::math::Vec2<int32_t>( 3, 4));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("test3", openvdb::math::Vec3<int32_t>( 5 ,6, 7), openvdb::math::Vec3<int32_t>( 8 ,7, 7));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("test4", openvdb::math::Vec3<int32_t>( 9, 8,-1), openvdb::math::Vec3<int32_t>( 9, 8, 9));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("test5", openvdb::math::Vec4<int32_t>(-1,-2,-3,-4), openvdb::math::Vec4<int32_t>(-7,-2,-8,-2));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("test6", openvdb::math::Vec4<int32_t>(-5,-6,-7,-8), openvdb::math::Vec4<int32_t>(-5,-5,-7,-8));
            }
        },
        { "float", [&]() {
                mHarness.addAttribute<openvdb::math::Vec2<float>>("test1", openvdb::math::Vec2<float>( 1.2f, 2.7f), openvdb::math::Vec2<float>(4.7f, 2.7f));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("test2", openvdb::math::Vec2<float>( 3.2f, 4.7f), openvdb::math::Vec2<float>(3.2f ,4.7f));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("test3", openvdb::math::Vec3<float>( 5.2f ,6.7f, 7.4f), openvdb::math::Vec3<float>( 8.7f ,7.4f, 7.4f));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("test4", openvdb::math::Vec3<float>( 9.2f, 8.7f,-1.4f), openvdb::math::Vec3<float>( 9.2f, 8.7f, 9.2f));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("test5", openvdb::math::Vec4<float>(-1.2f,-2.7f,-3.4f,-4.1f), openvdb::math::Vec4<float>(-7.4f,-2.7f,-8.1f,-2.7f));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("test6", openvdb::math::Vec4<float>(-5.2f,-6.7f,-7.4f,-8.1f), openvdb::math::Vec4<float>(-5.2f,-5.2f,-7.4f,-8.1f));
            }
        },
        { "double", [&]() {
                mHarness.addAttribute<openvdb::math::Vec2<double>>("test1", openvdb::math::Vec2<double>( 1.2, 2.7), openvdb::math::Vec2<double>(4.7, 2.7));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("test2", openvdb::math::Vec2<double>( 3.2, 4.7), openvdb::math::Vec2<double>(3.2, 4.7));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("test3", openvdb::math::Vec3<double>( 5.2 ,6.7, 7.4), openvdb::math::Vec3<double>( 8.7 ,7.4, 7.4));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("test4", openvdb::math::Vec3<double>( 9.2, 8.7,-1.4), openvdb::math::Vec3<double>( 9.2, 8.7, 9.2));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("test5", openvdb::math::Vec4<double>(-1.2,-2.7,-3.4,-4.1), openvdb::math::Vec4<double>(-7.4,-2.7,-8.1,-2.7));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("test6", openvdb::math::Vec4<double>(-5.2,-6.7,-7.4,-8.1), openvdb::math::Vec4<double>(-5.2,-5.2,-7.4,-8.1));
            }
        },
    };

    const std::array<std::string, 3> suffixes {{ "", ".xyz", ".rgb" }};

    for (const auto& expc : expected) {
        for (const auto& suffix : suffixes) {
            mHarness.reset();
            expc.second.operator()();
            this->execute("array_unpack.vec." + expc.first + suffix + ".ax");
        }
    }
}

void
TestArrayUnpack::componentMatrixAssignment()
{
    const std::string code = R"(
mat3@test1[0] = mat3@test2[4];
mat3@test2[1] = mat3@test1[0];
mat3@test1[2] = mat3@test2[5];
mat3@test2[3] = mat3@test1[6];
mat3@test1[4] = mat3@test2[3];
mat3@test2[5] = mat3@test1[1];
mat3@test1[6] = mat3@test2[7];
mat3@test2[7] = mat3@test1[8];
mat3@test1[8] = mat3@test2[2];

mat3@test3[0,0] = mat3@test4[1,1];
mat3@test4[0,1] = mat3@test3[0,0];
mat3@test3[0,2] = mat3@test4[1,2];
mat3@test4[1,0] = mat3@test3[2,0];
mat3@test3[1,1] = mat3@test4[1,0];
mat3@test4[1,2] = mat3@test3[0,1];
mat3@test3[2,0] = mat3@test4[2,1];
mat3@test4[2,1] = mat3@test3[2,2];
mat3@test3[2,2] = mat3@test4[0,2];

mat4@test5[0] =  mat4@test6[15];
mat4@test6[1] =  mat4@test5[0];
mat4@test5[2] =  mat4@test6[11];
mat4@test6[3] =  mat4@test5[6];
mat4@test5[4] =  mat4@test6[13];
mat4@test6[5] =  mat4@test5[1];
mat4@test5[6] =  mat4@test6[10];
mat4@test6[7] =  mat4@test5[8];
mat4@test5[8] =  mat4@test6[2];
mat4@test6[9] =  mat4@test5[7];
mat4@test5[10] = mat4@test6[14];
mat4@test6[11] = mat4@test5[3];
mat4@test5[12] = mat4@test6[4];
mat4@test6[13] = mat4@test5[12];
mat4@test5[14] = mat4@test6[5];
mat4@test6[15] = mat4@test5[9];

mat4@test7[0,0] = mat4@test8[3,3];
mat4@test8[0,1] = mat4@test7[0,0];
mat4@test7[0,2] = mat4@test8[2,3];
mat4@test8[0,3] = mat4@test7[1,2];
mat4@test7[1,0] = mat4@test8[3,1];
mat4@test8[1,1] = mat4@test7[0,1];
mat4@test7[1,2] = mat4@test8[2,2];
mat4@test8[1,3] = mat4@test7[2,0];
mat4@test7[2,0] = mat4@test8[0,2];
mat4@test8[2,1] = mat4@test7[1,3];
mat4@test7[2,2] = mat4@test8[3,2];
mat4@test8[2,3] = mat4@test7[0,3];
mat4@test7[3,0] = mat4@test8[1,0];
mat4@test8[3,1] = mat4@test7[3,0];
mat4@test7[3,2] = mat4@test8[1,1];
mat4@test8[3,3] = mat4@test7[2,1];
)";

    auto generate = [&](const auto& suffixes) {
        for (const auto& s : suffixes) {
            std::string repl = code;
            unittest_util::replace(repl, "mat3", std::string("mat3").append(1,s));
            unittest_util::replace(repl, "mat4", std::string("mat4").append(1,s));
            const std::string type = s == 'f' ? "float" : s == 'd' ? "double" : "";
            CPPUNIT_ASSERT(!type.empty());
            this->registerTest(repl, "array_unpack.mat." + type + ".ax");
        }
    };

    generate(std::vector<char>{'f', 'd'});

    const std::map<std::string, std::function<void()>> expected = {
        { "float", [&]() {
                mHarness.addAttribute<openvdb::math::Mat3<float>>("test1",
                    openvdb::math::Mat3<float>( 1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f),  // in
                    openvdb::math::Mat3<float>(-6.7f, 2.3f, 0.8f,   5.4f, 9.1f, 7.8f,  -0.5f, 4.5f,-1.3f)); // expected
                mHarness.addAttribute<openvdb::math::Mat3<float>>("test2",
                    openvdb::math::Mat3<float>(9.1f, 7.3f, -1.3f,  4.4f, -6.7f, 0.8f,  9.1f,-0.5f, 8.2f),
                    openvdb::math::Mat3<float>(9.1f,-6.7f, -1.3f,  9.1f, -6.7f, 2.3f,  9.1f, 8.2f, 8.2f));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("test3",
                    openvdb::math::Mat3<float>( 1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f),  // in
                    openvdb::math::Mat3<float>(-6.7f, 2.3f, 0.8f,   5.4f, 9.1f, 7.8f,  -0.5f, 4.5f,-1.3f)); // expected
                mHarness.addAttribute<openvdb::math::Mat3<float>>("test4",
                    openvdb::math::Mat3<float>(9.1f, 7.3f, -1.3f,  4.4f, -6.7f, 0.8f,  9.1f,-0.5f, 8.2f),
                    openvdb::math::Mat3<float>(9.1f,-6.7f, -1.3f,  9.1f, -6.7f, 2.3f,  9.1f, 8.2f, 8.2f));

                mHarness.addAttribute<openvdb::math::Mat4<float>>("test5",
                    openvdb::math::Mat4<float>( 1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f),  // in
                    openvdb::math::Mat4<float>(-1.7f, 2.3f, 2.5f, 5.4f,   0.5f, 7.8f,-0.3f, 4.5f, -9.3f, 3.3f, 8.1f, 5.9f, -1.7f, 0.3f, 2.3f, 1.9f)); // expected
                mHarness.addAttribute<openvdb::math::Mat4<float>>("test6",
                    openvdb::math::Mat4<float>(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f),
                    openvdb::math::Mat4<float>(0.1f,-1.7f,-9.3f, 9.1f,  -1.7f, 2.3f, 2.1f, 8.2f,  3.3f, 4.5f,-0.3f, 5.4f,  5.1f,-1.7f, 8.1f, 3.3f));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("test7",
                    openvdb::math::Mat4<float>( 1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f),  // in
                    openvdb::math::Mat4<float>(-1.7f, 2.3f, 2.5f, 5.4f,   0.5f, 7.8f,-0.3f, 4.5f, -9.3f, 3.3f, 8.1f, 5.9f, -1.7f, 0.3f, 2.3f, 1.9f)); // expected
                mHarness.addAttribute<openvdb::math::Mat4<float>>("test8",
                    openvdb::math::Mat4<float>(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f),
                    openvdb::math::Mat4<float>(0.1f,-1.7f,-9.3f, 9.1f,  -1.7f, 2.3f, 2.1f, 8.2f,  3.3f, 4.5f,-0.3f, 5.4f,  5.1f,-1.7f, 8.1f, 3.3f));
            }
        },
        { "double", [&]() {
                mHarness.addAttribute<openvdb::math::Mat3<double>>("test1",
                    openvdb::math::Mat3<double>( 1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2),  // in
                    openvdb::math::Mat3<double>(-6.7, 2.3, 0.8,   5.4, 9.1, 7.8,  -0.5, 4.5,-1.3)); // expected
                mHarness.addAttribute<openvdb::math::Mat3<double>>("test2",
                    openvdb::math::Mat3<double>(9.1, 7.3, -1.3,  4.4, -6.7, 0.8,  9.1,-0.5, 8.2),
                    openvdb::math::Mat3<double>(9.1,-6.7, -1.3,  9.1, -6.7, 2.3,  9.1, 8.2, 8.2));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("test3",
                    openvdb::math::Mat3<double>( 1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2),  // in
                    openvdb::math::Mat3<double>(-6.7, 2.3, 0.8,   5.4, 9.1, 7.8,  -0.5, 4.5,-1.3)); // expected
                mHarness.addAttribute<openvdb::math::Mat3<double>>("test4",
                    openvdb::math::Mat3<double>(9.1, 7.3, -1.3,  4.4, -6.7, 0.8,  9.1,-0.5, 8.2),
                    openvdb::math::Mat3<double>(9.1,-6.7, -1.3,  9.1, -6.7, 2.3,  9.1, 8.2, 8.2));

                mHarness.addAttribute<openvdb::math::Mat4<double>>("test5",
                    openvdb::math::Mat4<double>( 1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9),  // in
                    openvdb::math::Mat4<double>(-1.7, 2.3, 2.5, 5.4,   0.5, 7.8,-0.3, 4.5, -9.3, 3.3, 8.1, 5.9, -1.7, 0.3, 2.3, 1.9)); // expected
                mHarness.addAttribute<openvdb::math::Mat4<double>>("test6",
                    openvdb::math::Mat4<double>(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7),
                    openvdb::math::Mat4<double>(0.1,-1.7,-9.3, 9.1,  -1.7, 2.3, 2.1, 8.2,  3.3, 4.5,-0.3, 5.4,  5.1,-1.7, 8.1, 3.3));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("test7",
                    openvdb::math::Mat4<double>( 1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9),  // in
                    openvdb::math::Mat4<double>(-1.7, 2.3, 2.5, 5.4,   0.5, 7.8,-0.3, 4.5, -9.3, 3.3, 8.1, 5.9, -1.7, 0.3, 2.3, 1.9)); // expected
                mHarness.addAttribute<openvdb::math::Mat4<double>>("test8",
                    openvdb::math::Mat4<double>(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7),
                    openvdb::math::Mat4<double>(0.1,-1.7,-9.3, 9.1,  -1.7, 2.3, 2.1, 8.2,  3.3, 4.5,-0.3, 5.4,  5.1,-1.7, 8.1, 3.3));
            }
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("array_unpack.mat." + expc.first + ".ax");
    }

}


