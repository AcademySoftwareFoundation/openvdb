// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "CompareGrids.h"
#include "TestHarness.h"

#include "../util.h"

#include <openvdb_ax/codegen/String.h>
#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/Exceptions.h>

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestExternals : public unittest_util::AXTestCase
{
public:

    std::string dir() const override { return GET_TEST_DIRECTORY(); }

    CPPUNIT_TEST_SUITE(TestExternals);
    CPPUNIT_TEST(assignFrom);
    CPPUNIT_TEST_SUITE_END();

    void assignFrom();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestExternals);

void
TestExternals::assignFrom()
{
    const std::string code = R"(
_T1_@test1 = _T1_$ext1;)";

    auto generate = [&](const auto& types) {
        for (const auto& type : types) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", type);
            this->registerTest(repl, "external_assign_from." + type + ".ax");
        }
    };

    generate(std::vector<std::string>{
        "bool", "int32", "int64", "float", "double",
        "vec2i", "vec2f", "vec2d",
        "vec3i", "vec3f", "vec3d",
        "vec4i", "vec4f", "vec4d",
        "mat3f", "mat3d",
        "mat4f", "mat4d",
        "string"
    });

    const std::map<std::string, std::function<void()>> expected = {
        { "bool",
            [&](){
                mHarness.addAttribute<bool>("test1", true);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<bool>(true).copy());
             },
        },
        { "int32",
            [&](){
                mHarness.addAttribute<int32_t>("test1", -2);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<int32_t>(-2).copy());
            },
        },
        { "int64",
            [&](){
                mHarness.addAttribute<int64_t>("test1", 3);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<int64_t>(3).copy());
             },
        },
        { "float",
            [&](){
                mHarness.addAttribute<float>("test1", 4.5f);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<float>(4.5f).copy());
             },
        },
        { "double",
            [&](){
                mHarness.addAttribute<double>("test1", -3);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<double>(-3).copy());
             },
        },
        { "vec2i",
            [&](){
                const openvdb::math::Vec2<int32_t> value(5,-6);
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec2<int32_t>>(value).copy());
             },
        },
        { "vec2f",
            [&](){
                const openvdb::math::Vec2<float> value(2.3f,-7.8f);
                mHarness.addAttribute<openvdb::math::Vec2<float>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec2<float>>(value).copy());
             },
        },
        { "vec2d",
            [&](){
                const openvdb::math::Vec2<double> value(-1.3,9.8);
                mHarness.addAttribute<openvdb::math::Vec2<double>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec2<double>>(value).copy());
             },
        },
        { "vec3i",
            [&](){
                const openvdb::math::Vec3<int32_t> value(-1,3,8);
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec3<int32_t>>(value).copy());
             },
        },
        { "vec3f",
            [&](){
                const openvdb::math::Vec3<float> value(4.3f,-9.0f, 1.1f);
                mHarness.addAttribute<openvdb::math::Vec3<float>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec3<float>>(value).copy());
             },
        },
        { "vec3d",
            [&](){
                const openvdb::math::Vec3<double> value(8.2, 5.9, 1.6);
                mHarness.addAttribute<openvdb::math::Vec3<double>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec3<double>>(value).copy());
             },
        },
        { "vec4i",
            [&](){
                const openvdb::math::Vec4<int32_t> value(10,1,3,-8);
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec4<int32_t>>(value).copy());
             },
        },
        { "vec4f",
            [&](){
                const openvdb::math::Vec4<float> value(4.4f, 3.3f, -0.1f, 0.3f);
                mHarness.addAttribute<openvdb::math::Vec4<float>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec4<float>>(value).copy());
             },
        },
        { "vec4d",
            [&](){
                const openvdb::math::Vec4<double> value(4.5, 5.3, 1.1, 3.3);
                mHarness.addAttribute<openvdb::math::Vec4<double>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Vec4<double>>(value).copy());
             },
        },
        { "mat3f",
            [&](){
                const openvdb::math::Mat3<float> value(1.1f, 2.3f, 4.3f,  5.4f, 6.7f, 7.8f,  9.1f, 4.5f, 8.2f);
                mHarness.addAttribute<openvdb::math::Mat3<float>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Mat3<float>>(value).copy());
             },
        },
        { "mat3d",
            [&](){
                const openvdb::math::Mat3<double> value(6.7f, 2.9f,-1.1f,  3.2f, 2.2f, 0.8f,  -5.1f, 9.3f, 2.5f);
                mHarness.addAttribute<openvdb::math::Mat3<double>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Mat3<double>>(value).copy());
             },
        },
        { "mat4f",
            [&](){
                const openvdb::math::Mat4<float> value(1.1f,-2.3f,-0.3f, 7.8f,  -9.1f,-4.5f, 1.1f, 8.2f,  -4.3f, 5.4f, 6.7f,-0.2f,  8.8f, 5.5f, -6.6f, 7.7f);
                mHarness.addAttribute<openvdb::math::Mat4<float>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Mat4<float>>(value).copy());
             },
        },
        { "mat4d",
            [&](){
                const openvdb::math::Mat4<double> value(-2.3,0.0,-0.3,9.8, 0.0, 6.5, 3.7, 1.2,  -7.8,-0.3,-5.5,3.3,  -0.2, 9.1, 0.1,-9.1);
                mHarness.addAttribute<openvdb::math::Mat4<double>>("test1", value);
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::math::Mat4<double>>(value).copy());
             },
        },
        { "string",
            [&](){
                mHarness.addAttribute<std::string>("test1", "foo");
                mHarness.mCustomData.reset(new openvdb::ax::CustomData());
                mHarness.mCustomData->insertData("ext1", openvdb::TypedMetadata<openvdb::ax::codegen::String>("foo").copy());
             },
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("external_assign_from." + expc.first + ".ax");
    }
}


