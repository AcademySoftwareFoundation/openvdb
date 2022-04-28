// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "CompareGrids.h"
#include "TestHarness.h"

#include "../util.h"

#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/Exceptions.h>

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

// Configuration values for assignment code

static const unittest_util::ConfigMap integral = {
    { "bool",   { { "_l1_", "true" }, { "_l2_", "false" } }  },
    { "int32",  { { "_l1_", "2" },    { "_l2_", "3" } }      },
    { "int64",  { { "_l1_", "2l" },   { "_l2_", "3l" } }     }
};

static const unittest_util::ConfigMap floating = {
    { "float",  { { "_l1_", "1.1f" }, { "_l2_", "2.3f" } }   },
    { "double", { { "_l1_", "1.1" },  { "_l2_", "2.3" } }    }
};

static const unittest_util::ConfigMap vec2 = {
    { "vec2i",  { { "_l1_",  "{1, 2}" },                    { "_l2_", "{3, 4}" } } },
    { "vec2f",  { { "_l1_",  "{1.1f, 2.3f}" },              { "_l2_", "{4.1f, 5.3f}" } } },
    { "vec2d",  { { "_l1_",  "{1.1, 2.3}" },                { "_l2_", "{4.1, 5.3}" } } }
};

static const unittest_util::ConfigMap vec3 = {
    { "vec3i",  { { "_l1_",  "{1, 2, 3}" },                 { "_l2_", "{4, 5, 6}" } } },
    { "vec3f",  { { "_l1_",  "{1.1f, 2.3f, 4.3f}" },        { "_l2_", "{4.1f, 5.3f, 6.3f}" } } },
    { "vec3d",  { { "_l1_",  "{1.1,  2.3 , 4.3}" },         { "_l2_", "{4.1,  5.3, 6.3}" } } }
};

static const unittest_util::ConfigMap vec4 = {
    { "vec4i",  { { "_l1_",  "{1, 2, 3, 4}" },              { "_l2_", "{5, 6, 7, 8}" } } },
    { "vec4f",  { { "_l1_",  "{1.1f, 2.3f, 4.3f, 5.4f}" },  { "_l2_", "{5.1f, 6.3f, 7.3f, 8.4f}" } } },
    { "vec4d",  { { "_l1_",  "{1.1,  2.3, 4.3, 5.4}" },     { "_l2_", "{5.1, 6.3, 7.3, 8.4}" } } }
};

static const unittest_util::ConfigMap mat3 = {
    { "mat3f",  { { "_l1_",  "{1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f }" },
                  { "_l2_",  "{9.1f, 7.3f, -1.3f,  4.4f, -6.7f, 0.8f,  9.1f,-0.5f, 8.2f }" } }
    },
    { "mat3d",  { { "_l1_",  "{1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2 }" },
                  { "_l2_",  "{9.1, 7.3, -1.3,  4.4, -6.7, 0.8,  9.1,-0.5, 8.2 }" } }
    }
};

static const unittest_util::ConfigMap mat4 = {
    { "mat4f",  { { "_l1_",  "{1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f}" },
                  { "_l2_",  "{0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f}" } }
    },
    { "mat4d",  { { "_l1_",  "{1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9}" },
                  { "_l2_",  "{0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7}" } }
    }
};

static const unittest_util::ConfigMap string = {
    { "string",  { { "_l1_", "\"foo\"" }, { "_l2_", "\"bar\"" } }   }
};

//

class TestAssign : public unittest_util::AXTestCase
{
public:

    std::string dir() const override { return GET_TEST_DIRECTORY(); }

    CPPUNIT_TEST_SUITE(TestAssign);
    CPPUNIT_TEST(directAssignment);
    CPPUNIT_TEST(compoundIntegralAssignment);
    CPPUNIT_TEST(compoundFloatingAssignment);
    CPPUNIT_TEST(compoundVectorAssignment);
    CPPUNIT_TEST(compoundMatrixAssignment);
    CPPUNIT_TEST(compoundStringAssignment);
    CPPUNIT_TEST(implicitScalarAssignment);
    CPPUNIT_TEST(implicitContainerAssignment);
    CPPUNIT_TEST(implicitContainerScalarAssignment);
    CPPUNIT_TEST(scopedAssign);
    CPPUNIT_TEST_SUITE_END();

    void directAssignment();
    void compoundIntegralAssignment();
    void compoundFloatingAssignment();
    void compoundVectorAssignment();
    void compoundMatrixAssignment();
    void compoundStringAssignment();
    void implicitScalarAssignment();
    void implicitContainerAssignment();
    void implicitContainerScalarAssignment();
    void scopedAssign();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAssign);

void
TestAssign::directAssignment()
{
    const std::string code = R"(
_T1_@test1 = _l1_;
_T1_ local1 = _l1_;
_T1_@test2 = local1;
_T1_@test3 =
    _T1_@test4 =
        _T1_@test2;
_T1_ local3,
      local2 = _l2_;
_T1_@test5 =
    local3 =
        local2;
_T1_@test6 = _l2_,
     _T1_@test7 = _l1_;
_T1_@test8 = _l2_;
_T1_@test8 = _l1_;
)";

    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", config.first); // replace type
            // replace literal values
            for (const auto& settings : config.second) {
                unittest_util::replace(repl, settings.first, settings.second);
            }

            this->registerTest(repl, "assign." + config.first + ".ax");
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

    const auto names = unittest_util::nameSequence("test", 8);
    const std::map<std::string, std::function<void()>> expected = {
        { "bool",
            [&](){ mHarness.addAttributes<bool>(names,
                    { true, true, true, true, false, false, true, true });
            },
        },
        { "int32",
            [&](){ mHarness.addAttributes<int32_t>(names,
                    { 2, 2, 2, 2, 3, 3, 2, 2 });
            },
        },
        { "int64",
            [&](){ mHarness.addAttributes<int64_t>(names,
                    { 2, 2, 2, 2, 3, 3, 2, 2 });
            },
        },
        { "float",
            [&](){ mHarness.addAttributes<float>(names,
                    { 1.1f, 1.1f, 1.1f, 1.1f, 2.3f, 2.3f, 1.1f, 1.1f });
            },
        },
        { "double",
            [&](){ mHarness.addAttributes<double>(names,
                    { 1.1, 1.1, 1.1, 1.1, 2.3, 2.3, 1.1, 1.1 });
            },
        },
        { "vec2i",
            [&](){ mHarness.addAttributes<openvdb::math::Vec2<int32_t>>(names,
                    { openvdb::math::Vec2<int32_t>(1,2),
                      openvdb::math::Vec2<int32_t>(1,2),
                      openvdb::math::Vec2<int32_t>(1,2),
                      openvdb::math::Vec2<int32_t>(1,2),
                      openvdb::math::Vec2<int32_t>(3,4),
                      openvdb::math::Vec2<int32_t>(3,4),
                      openvdb::math::Vec2<int32_t>(1,2),
                      openvdb::math::Vec2<int32_t>(1,2)
                    });
            },
        },
        { "vec2f",
            [&](){ mHarness.addAttributes<openvdb::math::Vec2<float>>(names,
                    { openvdb::math::Vec2<float>(1.1f, 2.3f),
                      openvdb::math::Vec2<float>(1.1f, 2.3f),
                      openvdb::math::Vec2<float>(1.1f, 2.3f),
                      openvdb::math::Vec2<float>(1.1f, 2.3f),
                      openvdb::math::Vec2<float>(4.1f, 5.3f),
                      openvdb::math::Vec2<float>(4.1f, 5.3f),
                      openvdb::math::Vec2<float>(1.1f, 2.3f),
                      openvdb::math::Vec2<float>(1.1f, 2.3f)
                    });
            },
        },
        { "vec2d",
            [&](){ mHarness.addAttributes<openvdb::math::Vec2<double>>(names,
                    { openvdb::math::Vec2<double>(1.1, 2.3),
                      openvdb::math::Vec2<double>(1.1, 2.3),
                      openvdb::math::Vec2<double>(1.1, 2.3),
                      openvdb::math::Vec2<double>(1.1, 2.3),
                      openvdb::math::Vec2<double>(4.1, 5.3),
                      openvdb::math::Vec2<double>(4.1, 5.3),
                      openvdb::math::Vec2<double>(1.1, 2.3),
                      openvdb::math::Vec2<double>(1.1, 2.3)
                    });
            },
        },
        { "vec3i",
            [&](){ mHarness.addAttributes<openvdb::math::Vec3<int32_t>>(names,
                    { openvdb::math::Vec3<int32_t>(1,2,3),
                      openvdb::math::Vec3<int32_t>(1,2,3),
                      openvdb::math::Vec3<int32_t>(1,2,3),
                      openvdb::math::Vec3<int32_t>(1,2,3),
                      openvdb::math::Vec3<int32_t>(4,5,6),
                      openvdb::math::Vec3<int32_t>(4,5,6),
                      openvdb::math::Vec3<int32_t>(1,2,3),
                      openvdb::math::Vec3<int32_t>(1,2,3)
                    });
            },
        },
        { "vec3f",
            [&](){ mHarness.addAttributes<openvdb::math::Vec3<float>>(names,
                    { openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f),
                      openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f),
                      openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f),
                      openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f),
                      openvdb::math::Vec3<float>(4.1f, 5.3f, 6.3f),
                      openvdb::math::Vec3<float>(4.1f, 5.3f, 6.3f),
                      openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f),
                      openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f)
                    });
            },
        },
        { "vec3d",
            [&](){ mHarness.addAttributes<openvdb::math::Vec3<double>>(names,
                    { openvdb::math::Vec3<double>(1.1, 2.3, 4.3),
                      openvdb::math::Vec3<double>(1.1, 2.3, 4.3),
                      openvdb::math::Vec3<double>(1.1, 2.3, 4.3),
                      openvdb::math::Vec3<double>(1.1, 2.3, 4.3),
                      openvdb::math::Vec3<double>(4.1, 5.3, 6.3),
                      openvdb::math::Vec3<double>(4.1, 5.3, 6.3),
                      openvdb::math::Vec3<double>(1.1, 2.3, 4.3),
                      openvdb::math::Vec3<double>(1.1, 2.3, 4.3)
                    });
            },
        },
        { "vec4i",
            [&](){ mHarness.addAttributes<openvdb::math::Vec4<int32_t>>(names,
                    { openvdb::math::Vec4<int32_t>(1, 2, 3, 4),
                      openvdb::math::Vec4<int32_t>(1, 2, 3, 4),
                      openvdb::math::Vec4<int32_t>(1, 2, 3, 4),
                      openvdb::math::Vec4<int32_t>(1, 2, 3, 4),
                      openvdb::math::Vec4<int32_t>(5, 6, 7, 8),
                      openvdb::math::Vec4<int32_t>(5, 6, 7, 8),
                      openvdb::math::Vec4<int32_t>(1, 2, 3, 4),
                      openvdb::math::Vec4<int32_t>(1, 2, 3, 4)
                    });
            },
        },
        { "vec4f",
            [&](){ mHarness.addAttributes<openvdb::math::Vec4<float>>(names,
                    { openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f),
                      openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f),
                      openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f),
                      openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f),
                      openvdb::math::Vec4<float>(5.1f, 6.3f, 7.3f, 8.4f),
                      openvdb::math::Vec4<float>(5.1f, 6.3f, 7.3f, 8.4f),
                      openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f),
                      openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f)
                    });
            },
        },
        { "vec4d",
            [&](){ mHarness.addAttributes<openvdb::math::Vec4<double>>(names,
                    { openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4),
                      openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4),
                      openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4),
                      openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4),
                      openvdb::math::Vec4<double>(5.1, 6.3, 7.3, 8.4),
                      openvdb::math::Vec4<double>(5.1, 6.3, 7.3, 8.4),
                      openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4),
                      openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4)
                    });
            },
        },
        { "mat3f",
            [&](){ mHarness.addAttributes<openvdb::math::Mat3<float>>(names,
                    { openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f),
                      openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f),
                      openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f),
                      openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f),
                      openvdb::math::Mat3<float>(9.1f, 7.3f, -1.3f,  4.4f, -6.7f, 0.8f,  9.1f, -0.5f, 8.2f),
                      openvdb::math::Mat3<float>(9.1f, 7.3f, -1.3f,  4.4f, -6.7f, 0.8f,  9.1f, -0.5f, 8.2f),
                      openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f),
                      openvdb::math::Mat3<float>(1.1f, 2.3f, 4.3f,   5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f)
                    });
            },
        },
        { "mat3d",
            [&](){ mHarness.addAttributes<openvdb::math::Mat3<double>>(names,
                    { openvdb::math::Mat3<double>(1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2),
                      openvdb::math::Mat3<double>(1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2),
                      openvdb::math::Mat3<double>(1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2),
                      openvdb::math::Mat3<double>(1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2),
                      openvdb::math::Mat3<double>(9.1, 7.3, -1.3,  4.4, -6.7, 0.8,  9.1, -0.5, 8.2),
                      openvdb::math::Mat3<double>(9.1, 7.3, -1.3,  4.4, -6.7, 0.8,  9.1, -0.5, 8.2),
                      openvdb::math::Mat3<double>(1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2),
                      openvdb::math::Mat3<double>(1.1, 2.3, 4.3,   5.4, 6.7, 7.8,   9.1, 4.5, 8.2)
                    });
            },
        },
        { "mat4f",
            [&](){ mHarness.addAttributes<openvdb::math::Mat4<float>>(names,
                    { openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f),
                      openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f),
                      openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f),
                      openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f),
                      openvdb::math::Mat4<float>(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f),
                      openvdb::math::Mat4<float>(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f),
                      openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f),
                      openvdb::math::Mat4<float>(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f)
                    });
            },
        },
        { "mat4d",
            [&](){ mHarness.addAttributes<openvdb::math::Mat4<double>>(names,
                    { openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9),
                      openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9),
                      openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9),
                      openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9),
                      openvdb::math::Mat4<double>(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7),
                      openvdb::math::Mat4<double>(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7),
                      openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9),
                      openvdb::math::Mat4<double>(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9)
                    });
            },
        },
        { "string",
            [&](){ mHarness.addAttributes<std::string>(names,
                    { "foo", "foo", "foo", "foo", "bar", "bar", "foo", "foo" });
            },
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("assign." + expc.first + ".ax");
    }
}

void
TestAssign::compoundIntegralAssignment()
{
    const std::string code = R"(
_T1_@test1 += _l1_;
_T1_@test2 -= _l1_;
_T1_@test3 *= _l1_;
_T1_@test4 /= _l1_;
_T1_@test5 %= _l1_;
_T1_@test6 <<= _l1_;
_T1_@test7 >>= _l1_;
_T1_@test8 &= _l1_;
_T1_@test9 ^= _l1_;
_T1_@test10 |= _l1_;

_T1_ local1 = _l1_,
      local2 = _l2_;

local1 += local2;
_T1_@test11 = local1;
_T1_@test12 += _T1_@test13;
_T1_@test14 += local2;
)";

    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", config.first); // replace type
            // replace literal values
            for (const auto& settings : config.second) {
                unittest_util::replace(repl, settings.first, settings.second);
            }

            this->registerTest(repl, "assign_compound." + config.first + ".ax");
        }
    };

    generate(integral);

    const auto names = unittest_util::nameSequence("test", 14);
    const std::map<std::string, std::vector<std::function<void()>>> expected = {
        { "bool", {
                [&](){ mHarness.addAttributes<bool>(names,
                        { true, true, false, false, false, false, false, false, true, true,  true, false, false, false });
                },
                [&](){ mHarness.addAttributes<bool>(names,
                        { true,  true,  true, true,  true, true,  true, true,  true, true,  false, true, true, true },  // in
                        { true, false, true, true, false, true, false, true, false, true,  true, true, true, true }); // expected
                },
            }
        },
        { "int32", {
                [&](){ mHarness.addAttributes<int32_t>(names,
                        { 2, -2, 0, 0, 0, 0, 0, 0, 2, 2,  5, 0, 0, 3 });
                },
                [&](){ mHarness.addAttributes<int32_t>(names,
                        { 1, 2, 3, 4, 5,  6, 7, 8,  9, 10,  0, 12, 13, 14 },  // in
                        { 3, 0, 6, 2, 1, 24, 1, 0, 11, 10,  5, 25, 13, 17 }); // expected
                },
            }
        },
        { "int64", {
                [&](){ mHarness.addAttributes<int64_t>(names,
                        { 2, -2, 0, 0, 0, 0, 0, 0, 2, 2,  5, 0, 0, 3 });
                },
                [&](){ mHarness.addAttributes<int64_t>(names,
                        { 1, 2, 3, 4, 5,  6, 7, 8,  9, 10,  0, 12, 13, 14 },  // in
                        { 3, 0, 6, 2, 1, 24, 1, 0, 11, 10,  5, 25, 13, 17 }); // expected
                },
            }
        }
    };

    for (const auto& expc : expected) {
        for (const auto& test : expc.second) {
            mHarness.reset();
            test.operator()();
            this->execute("assign_compound." + expc.first + ".ax");
        }
    }
}

void
TestAssign::compoundFloatingAssignment()
{
    const std::string code = R"(
_T1_@test1 += _l1_;
_T1_@test2 -= _l1_;
_T1_@test3 *= _l1_;
_T1_@test4 /= _l1_;
_T1_@test5 %= _l1_;

_T1_ local1 = _l1_,
      local2 = _l2_;

local1 += local2;
_T1_@test6 = local1;
_T1_@test7 += _T1_@test8;
_T1_@test9 += local2;
)";

    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", config.first); // replace type
            // replace literal values
            for (const auto& settings : config.second) {
                unittest_util::replace(repl, settings.first, settings.second);
            }

            this->registerTest(repl, "assign_compound." + config.first + ".ax");
        }
    };

    generate(floating);

    const auto names = unittest_util::nameSequence("test", 9);
    const std::map<std::string, std::vector<std::function<void()>>> expected = {
        { "float", {
                [&](){ mHarness.addAttributes<float>(names,
                        { 1.1f, -1.1f, 0.0f, 0.0f, 0.0f,  (1.1f+2.3f), 0.0f, 0.0f, 2.3f });
                },
                [&](){ mHarness.addAttributes<float>(names,
                        { 1.1f,  2.3f, 4.5f, 6.7f, 8.9f, -1.1f, -2.3f, -4.5f, 6.1f },  // in
                        { (1.1f+1.1f), (2.3f-1.1f), (4.5f*1.1f), (6.7f/1.1f),
                          std::fmod(8.9f,1.1f),
                          (1.1f+2.3f), (-2.3f+-4.5f), (-4.5f), (6.1f+2.3f) }); // expected
                }
            }
        },
        { "double", {
                [&](){ mHarness.addAttributes<double>(names,
                        { 1.1, -1.1, 0.0, 0.0, 0.0,  (1.1+2.3), 0.0, 0.0, 2.3 });
                },
                [&](){ mHarness.addAttributes<double>(names,
                        { 1.1,  2.3, 4.5, 6.7, 8.9, -1.1, -2.3, -4.5, 6.1 },  // in
                        { (1.1+1.1), (2.3-1.1), (4.5*1.1), (6.7/1.1),
                          std::fmod(8.9,1.1),
                          (1.1+2.3), (-2.3+-4.5), (-4.5), (6.1+2.3) }); // expected
                }
            }
        },
    };

    for (const auto& expc : expected) {
        for (const auto& test : expc.second) {
            mHarness.reset();
            test.operator()();
            this->execute("assign_compound." + expc.first + ".ax");
        }
    }
}


void
TestAssign::compoundVectorAssignment()
{
    const std::string code = R"(
_T1_@test1 += _l1_;
_T1_@test2 -= _l1_;
_T1_@test3 *= _l1_;
_T1_@test4 /= _l1_;
_T1_@test5 %= _l1_;

_T1_ local1 = _l1_,
    local2 = _l2_;

local1 += local2;
_T1_@test6 = local1;
_T1_@test7 += _T1_@test8;
_T1_@test9 += local2;
)";

    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", config.first); // replace type
            // replace literal values
            for (const auto& settings : config.second) {
                unittest_util::replace(repl, settings.first, settings.second);
            }

            this->registerTest(repl, "assign_compound." + config.first + ".ax");
        }
    };

    generate(vec2);
    generate(vec3);
    generate(vec4);

    const auto names = unittest_util::nameSequence("test", 9);
    const std::map<std::string, std::vector<std::function<void()>>> expected = {
        { "vec2i", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec2<int32_t>>(names,
                        { openvdb::math::Vec2<int32_t>(1,2),
                          openvdb::math::Vec2<int32_t>(-1,-2),
                          openvdb::math::Vec2<int32_t>(0,0),
                          openvdb::math::Vec2<int32_t>(0,0),
                          openvdb::math::Vec2<int32_t>(0,0),
                          openvdb::math::Vec2<int32_t>(4,6),
                          openvdb::math::Vec2<int32_t>(0,0),
                          openvdb::math::Vec2<int32_t>(0,0),
                          openvdb::math::Vec2<int32_t>(3,4) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec2<int32_t>>(names,
                        { openvdb::math::Vec2<int32_t>(1,2),
                          openvdb::math::Vec2<int32_t>(3,4),
                          openvdb::math::Vec2<int32_t>(5,6),
                          openvdb::math::Vec2<int32_t>(7,8),
                          openvdb::math::Vec2<int32_t>(3,9),
                          openvdb::math::Vec2<int32_t>(9,-1),
                          openvdb::math::Vec2<int32_t>(-2,-3),
                          openvdb::math::Vec2<int32_t>(-4,-5),
                          openvdb::math::Vec2<int32_t>(-6,-7) }, // in
                        { openvdb::math::Vec2<int32_t>(2,4),
                          openvdb::math::Vec2<int32_t>(2,2),
                          openvdb::math::Vec2<int32_t>(5,12),
                          openvdb::math::Vec2<int32_t>(7,4),
                          openvdb::math::Vec2<int32_t>(0,1),
                          openvdb::math::Vec2<int32_t>(4,6),
                          openvdb::math::Vec2<int32_t>(-6,-8),
                          openvdb::math::Vec2<int32_t>(-4,-5),
                          openvdb::math::Vec2<int32_t>(-3,-3) }); // expected
                }
            }
        },
        { "vec2f", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec2<float>>(names,
                        { openvdb::math::Vec2<float>(1.1f,2.3f),
                          openvdb::math::Vec2<float>(-1.1f,-2.3f),
                          openvdb::math::Vec2<float>(0.0f,0.0f),
                          openvdb::math::Vec2<float>(0.0f,0.0f),
                          openvdb::math::Vec2<float>(0.0f,0.0f),
                          openvdb::math::Vec2<float>(1.1f, 2.3f) + openvdb::math::Vec2<float>(4.1f, 5.3f),
                          openvdb::math::Vec2<float>(0.0f,0.0f),
                          openvdb::math::Vec2<float>(0.0f,0.0f),
                          openvdb::math::Vec2<float>(4.1f,5.3f) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec2<float>>(names,
                        { openvdb::math::Vec2<float>(1.1f,2.2f),
                          openvdb::math::Vec2<float>(3.3f,4.4f),
                          openvdb::math::Vec2<float>(5.5f,6.6f),
                          openvdb::math::Vec2<float>(7.7f,8.8f),
                          openvdb::math::Vec2<float>(2.3f,5.5f),
                          openvdb::math::Vec2<float>(9.9f,-1.1f),
                          openvdb::math::Vec2<float>(-2.2f,-3.3f),
                          openvdb::math::Vec2<float>(-4.3f,-5.5f),
                          openvdb::math::Vec2<float>(-6.1f,-8.2f) }, // in
                        { openvdb::math::Vec2<float>(1.1f,2.2f) + openvdb::math::Vec2<float>(1.1f,2.3f),
                          openvdb::math::Vec2<float>(3.3f,4.4f) - openvdb::math::Vec2<float>(1.1f,2.3f),
                          openvdb::math::Vec2<float>(5.5f,6.6f) * openvdb::math::Vec2<float>(1.1f,2.3f),
                          openvdb::math::Vec2<float>(7.7f,8.8f) / openvdb::math::Vec2<float>(1.1f,2.3f),
                          openvdb::math::Vec2<float>(std::fmod(2.3f, 1.1f), std::fmod(5.5f, 2.3f)),
                          openvdb::math::Vec2<float>(1.1f, 2.3f) + openvdb::math::Vec2<float>(4.1f, 5.3f),
                          openvdb::math::Vec2<float>(-2.2f,-3.3f) + openvdb::math::Vec2<float>(-4.3f,-5.5f),
                          openvdb::math::Vec2<float>(-4.3f,-5.5f),
                          openvdb::math::Vec2<float>(-6.1f,-8.2f) + openvdb::math::Vec2<float>(4.1f,5.3f)
                        }); // expected
                }
            }
        },
        { "vec2d", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec2<double>>(names,
                        { openvdb::math::Vec2<double>(1.1,2.3),
                          openvdb::math::Vec2<double>(-1.1,-2.3),
                          openvdb::math::Vec2<double>(0.0,0.0),
                          openvdb::math::Vec2<double>(0.0,0.0),
                          openvdb::math::Vec2<double>(0.0,0.0),
                          openvdb::math::Vec2<double>(1.1, 2.3) + openvdb::math::Vec2<double>(4.1, 5.3),
                          openvdb::math::Vec2<double>(0.0,0.0),
                          openvdb::math::Vec2<double>(0.0,0.0),
                          openvdb::math::Vec2<double>(4.1,5.3) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec2<double>>(names,
                        { openvdb::math::Vec2<double>(1.1,2.2),
                          openvdb::math::Vec2<double>(3.3,4.4),
                          openvdb::math::Vec2<double>(5.5,6.6),
                          openvdb::math::Vec2<double>(7.7,8.8),
                          openvdb::math::Vec2<double>(2.3,5.5),
                          openvdb::math::Vec2<double>(9.9,-1.1),
                          openvdb::math::Vec2<double>(-2.2,-3.3),
                          openvdb::math::Vec2<double>(-4.3,-5.5),
                          openvdb::math::Vec2<double>(-6.1,-8.2) }, // in
                        { openvdb::math::Vec2<double>(1.1,2.2) + openvdb::math::Vec2<double>(1.1,2.3),
                          openvdb::math::Vec2<double>(3.3,4.4) - openvdb::math::Vec2<double>(1.1,2.3),
                          openvdb::math::Vec2<double>(5.5,6.6) * openvdb::math::Vec2<double>(1.1,2.3),
                          openvdb::math::Vec2<double>(7.7,8.8) / openvdb::math::Vec2<double>(1.1,2.3),
                          openvdb::math::Vec2<double>(std::fmod(2.3, 1.1), std::fmod(5.5, 2.3)),
                          openvdb::math::Vec2<double>(1.1, 2.3) + openvdb::math::Vec2<double>(4.1, 5.3),
                          openvdb::math::Vec2<double>(-2.2,-3.3) + openvdb::math::Vec2<double>(-4.3,-5.5),
                          openvdb::math::Vec2<double>(-4.3,-5.5),
                          openvdb::math::Vec2<double>(-6.1,-8.2) + openvdb::math::Vec2<double>(4.1,5.3)
                        }); // expected
                }
            }
        },
        { "vec3i", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec3<int32_t>>(names,
                        { openvdb::math::Vec3<int32_t>(1,2,3),
                          openvdb::math::Vec3<int32_t>(-1,-2,-3),
                          openvdb::math::Vec3<int32_t>(0,0,0),
                          openvdb::math::Vec3<int32_t>(0,0,0),
                          openvdb::math::Vec3<int32_t>(0,0,0),
                          openvdb::math::Vec3<int32_t>(5,7,9),
                          openvdb::math::Vec3<int32_t>(0,0,0),
                          openvdb::math::Vec3<int32_t>(0,0,0),
                          openvdb::math::Vec3<int32_t>(4,5,6) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec3<int32_t>>(names,
                        { openvdb::math::Vec3<int32_t>(1,2,3),
                          openvdb::math::Vec3<int32_t>(4,5,6),
                          openvdb::math::Vec3<int32_t>(7,8,9),
                          openvdb::math::Vec3<int32_t>(-1,-2,-3),
                          openvdb::math::Vec3<int32_t>(4,-5,6),
                          openvdb::math::Vec3<int32_t>(5,7,9),
                          openvdb::math::Vec3<int32_t>(-7,-8,-9),
                          openvdb::math::Vec3<int32_t>(-1,2,-3),
                          openvdb::math::Vec3<int32_t>(-4,5,-6) }, // in
                        { openvdb::math::Vec3<int32_t>(2,4,6),
                          openvdb::math::Vec3<int32_t>(3,3,3),
                          openvdb::math::Vec3<int32_t>(7,16,27),
                          openvdb::math::Vec3<int32_t>(-1,-1,-1),
                          openvdb::math::Vec3<int32_t>(0,1,0),
                          openvdb::math::Vec3<int32_t>(5,7,9),
                          openvdb::math::Vec3<int32_t>(-8,-6,-12),
                          openvdb::math::Vec3<int32_t>(-1,2,-3),
                          openvdb::math::Vec3<int32_t>(0,10,0) }); // expected
                }
            }
        },
        { "vec3f", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec3<float>>(names,
                        { openvdb::math::Vec3<float>(1.1f,2.3f,4.3f),
                          openvdb::math::Vec3<float>(-1.1f,-2.3f,-4.3f),
                          openvdb::math::Vec3<float>(0.0f,0.0f,0.0f),
                          openvdb::math::Vec3<float>(0.0f,0.0f,0.0f),
                          openvdb::math::Vec3<float>(0.0f,0.0f,0.0f),
                          openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f) + openvdb::math::Vec3<float>(4.1f, 5.3f, 6.3f),
                          openvdb::math::Vec3<float>(0.0f,0.0f,0.0f),
                          openvdb::math::Vec3<float>(0.0f,0.0f,0.0f),
                          openvdb::math::Vec3<float>(4.1f, 5.3f, 6.3f) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec3<float>>(names,
                        { openvdb::math::Vec3<float>(1.1f,2.2f,3.3f),
                          openvdb::math::Vec3<float>(3.3f,4.4f,5.5f),
                          openvdb::math::Vec3<float>(5.5f,6.6f,7.7f),
                          openvdb::math::Vec3<float>(7.7f,8.8f,9.9f),
                          openvdb::math::Vec3<float>(7.7f,8.8f,9.9f),
                          openvdb::math::Vec3<float>(9.9f,-1.1f,-2.2f),
                          openvdb::math::Vec3<float>(-2.2f,-3.3f,-4.4f),
                          openvdb::math::Vec3<float>(-4.3f,-5.5f,-6.6f),
                          openvdb::math::Vec3<float>(-7.1f,8.5f,-9.9f), }, // in
                        { openvdb::math::Vec3<float>(1.1f,2.2f,3.3f) + openvdb::math::Vec3<float>(1.1f,2.3f,4.3f),
                          openvdb::math::Vec3<float>(3.3f,4.4f,5.5f) - openvdb::math::Vec3<float>(1.1f,2.3f,4.3f),
                          openvdb::math::Vec3<float>(5.5f,6.6f,7.7f) * openvdb::math::Vec3<float>(1.1f,2.3f,4.3f),
                          openvdb::math::Vec3<float>(7.7f,8.8f,9.9f) / openvdb::math::Vec3<float>(1.1f,2.3f,4.3f),
                          openvdb::math::Vec3<float>(std::fmod(7.7f,1.1f), std::fmod(8.8f,2.3f), std::fmod(9.9f,4.3f)),
                          openvdb::math::Vec3<float>(1.1f, 2.3f, 4.3f) + openvdb::math::Vec3<float>(4.1f, 5.3f, 6.3f),
                          openvdb::math::Vec3<float>(-2.2f,-3.3f,-4.4f) + openvdb::math::Vec3<float>(-4.3f,-5.5f,-6.6f),
                          openvdb::math::Vec3<float>(-4.3f,-5.5f,-6.6f),
                          openvdb::math::Vec3<float>(-7.1f,8.5f,-9.9f) + openvdb::math::Vec3<float>(4.1f, 5.3f, 6.3f)
                        }); // expected
                }
            }
        },
        { "vec3d", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec3<double>>(names,
                        { openvdb::math::Vec3<double>(1.1,2.3,4.3),
                          openvdb::math::Vec3<double>(-1.1,-2.3,-4.3),
                          openvdb::math::Vec3<double>(0.0,0.0,0.0),
                          openvdb::math::Vec3<double>(0.0,0.0,0.0),
                          openvdb::math::Vec3<double>(0.0,0.0,0.0),
                          openvdb::math::Vec3<double>(1.1, 2.3, 4.3) + openvdb::math::Vec3<double>(4.1, 5.3, 6.3),
                          openvdb::math::Vec3<double>(0.0,0.0,0.0),
                          openvdb::math::Vec3<double>(0.0,0.0,0.0),
                          openvdb::math::Vec3<double>(4.1, 5.3, 6.3) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec3<double>>(names,
                        { openvdb::math::Vec3<double>(1.1,2.2,3.3),
                          openvdb::math::Vec3<double>(3.3,4.4,5.5),
                          openvdb::math::Vec3<double>(5.5,6.6,7.7),
                          openvdb::math::Vec3<double>(7.7,8.8,9.9),
                          openvdb::math::Vec3<double>(7.7,8.8,9.9),
                          openvdb::math::Vec3<double>(9.9,-1.1,-2.2),
                          openvdb::math::Vec3<double>(-2.2,-3.3,-4.4),
                          openvdb::math::Vec3<double>(-4.3,-5.5,-6.6),
                          openvdb::math::Vec3<double>(-7.1,8.5,-9.9), }, // in
                        { openvdb::math::Vec3<double>(1.1,2.2,3.3) + openvdb::math::Vec3<double>(1.1,2.3,4.3),
                          openvdb::math::Vec3<double>(3.3,4.4,5.5) - openvdb::math::Vec3<double>(1.1,2.3,4.3),
                          openvdb::math::Vec3<double>(5.5,6.6,7.7) * openvdb::math::Vec3<double>(1.1,2.3,4.3),
                          openvdb::math::Vec3<double>(7.7,8.8,9.9) / openvdb::math::Vec3<double>(1.1,2.3,4.3),
                          openvdb::math::Vec3<double>(std::fmod(7.7,1.1), std::fmod(8.8,2.3), std::fmod(9.9,4.3)),
                          openvdb::math::Vec3<double>(1.1, 2.3, 4.3) + openvdb::math::Vec3<double>(4.1, 5.3, 6.3),
                          openvdb::math::Vec3<double>(-2.2,-3.3,-4.4) + openvdb::math::Vec3<double>(-4.3,-5.5,-6.6),
                          openvdb::math::Vec3<double>(-4.3,-5.5,-6.6),
                          openvdb::math::Vec3<double>(-7.1,8.5,-9.9) + openvdb::math::Vec3<double>(4.1, 5.3, 6.3)
                        }); // expected
                }
            }
        },
        { "vec4i", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec4<int32_t>>(names,
                        { openvdb::math::Vec4<int32_t>(1,2,3,4),
                          openvdb::math::Vec4<int32_t>(-1,-2,-3,-4),
                          openvdb::math::Vec4<int32_t>(0,0,0,0),
                          openvdb::math::Vec4<int32_t>(0,0,0,0),
                          openvdb::math::Vec4<int32_t>(0,0,0,0),
                          openvdb::math::Vec4<int32_t>(6,8,10,12),
                          openvdb::math::Vec4<int32_t>(0,0,0,0),
                          openvdb::math::Vec4<int32_t>(0,0,0,0),
                          openvdb::math::Vec4<int32_t>(5,6,7,8) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec4<int32_t>>(names,
                        { openvdb::math::Vec4<int32_t>(1,2,3,4),
                          openvdb::math::Vec4<int32_t>(4,5,6,7),
                          openvdb::math::Vec4<int32_t>(7,8,9,-1),
                          openvdb::math::Vec4<int32_t>(-1,-2,-3,1),
                          openvdb::math::Vec4<int32_t>(-4,-5,-6,2),
                          openvdb::math::Vec4<int32_t>(4,5,-6,2),
                          openvdb::math::Vec4<int32_t>(-7,-8,-9,3),
                          openvdb::math::Vec4<int32_t>(-1,2,-3,4),
                          openvdb::math::Vec4<int32_t>(-5,6,-7,8) }, // in
                        { openvdb::math::Vec4<int32_t>(2,4,6,8),
                          openvdb::math::Vec4<int32_t>(3,3,3,3),
                          openvdb::math::Vec4<int32_t>(7,16,27,-4),
                          openvdb::math::Vec4<int32_t>(-1,-1,-1,0),
                          openvdb::math::Vec4<int32_t>(0,1,0,2),
                          openvdb::math::Vec4<int32_t>(6,8,10,12),
                          openvdb::math::Vec4<int32_t>(-8,-6,-12,7),
                          openvdb::math::Vec4<int32_t>(-1,2,-3,4),
                          openvdb::math::Vec4<int32_t>(0,12,0,16) }); // expected
                }
            }
        },
        { "vec4f", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec4<float>>(names,
                        { openvdb::math::Vec4<float>(1.1f,2.3f,4.3f,5.4f),
                          openvdb::math::Vec4<float>(-1.1f,-2.3f,-4.3f,-5.4f),
                          openvdb::math::Vec4<float>(0.0f,0.0f,0.0f,0.0f),
                          openvdb::math::Vec4<float>(0.0f,0.0f,0.0f,0.0f),
                          openvdb::math::Vec4<float>(0.0f,0.0f,0.0f,0.0f),
                          openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f) + openvdb::math::Vec4<float>(5.1f, 6.3f, 7.3f, 8.4f),
                          openvdb::math::Vec4<float>(0.0f,0.0f,0.0f,0.0f),
                          openvdb::math::Vec4<float>(0.0f,0.0f,0.0f,0.0f),
                          openvdb::math::Vec4<float>(5.1f, 6.3f, 7.3f, 8.4f) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec4<float>>(names,
                        { openvdb::math::Vec4<float>(1.1f,2.2f,3.3f,4.4f),
                          openvdb::math::Vec4<float>(3.3f,4.4f,5.5f,6.6f),
                          openvdb::math::Vec4<float>(5.5f,6.6f,7.7f,8.8f),
                          openvdb::math::Vec4<float>(7.7f,8.8f,9.9f,-1.1f),
                          openvdb::math::Vec4<float>(7.7f,8.8f,9.9f,-1.1f),
                          openvdb::math::Vec4<float>(9.9f,-1.1f,-2.2f,-3.3f),
                          openvdb::math::Vec4<float>(-2.2f,-3.3f,-4.4f,-5.5f),
                          openvdb::math::Vec4<float>(-4.3f,-5.5f,-6.6f,-7.7f),
                          openvdb::math::Vec4<float>(-8.2f,-9.3f,0.6f,-1.7f) }, // in
                        { openvdb::math::Vec4<float>(1.1f,2.2f,3.3f,4.4f) + openvdb::math::Vec4<float>(1.1f,2.3f,4.3f,5.4f),
                          openvdb::math::Vec4<float>(3.3f,4.4f,5.5f,6.6f) - openvdb::math::Vec4<float>(1.1f,2.3f,4.3f,5.4f),
                          openvdb::math::Vec4<float>(5.5f,6.6f,7.7f,8.8f) * openvdb::math::Vec4<float>(1.1f,2.3f,4.3f,5.4f),
                          openvdb::math::Vec4<float>(7.7f,8.8f,9.9f,-1.1f) / openvdb::math::Vec4<float>(1.1f,2.3f,4.3f,5.4f),
                          openvdb::math::Vec4<float>(std::fmod(7.7f,1.1f),std::fmod(8.8f,2.3f),std::fmod(9.9f,4.3f),std::fmod(-1.1f,5.4f)+5.4f), // floored mod
                          openvdb::math::Vec4<float>(1.1f, 2.3f, 4.3f, 5.4f) + openvdb::math::Vec4<float>(5.1f, 6.3f, 7.3f, 8.4f),
                          openvdb::math::Vec4<float>(-2.2f,-3.3f,-4.4f,-5.5f) + openvdb::math::Vec4<float>(-4.3f,-5.5f,-6.6f,-7.7f),
                          openvdb::math::Vec4<float>(-4.3f,-5.5f,-6.6f,-7.7f),
                          openvdb::math::Vec4<float>(-8.2f,-9.3f,0.6f,-1.7f) + openvdb::math::Vec4<float>(5.1f, 6.3f, 7.3f, 8.4f)
                        }); // expected
                }
            }
        },
        { "vec4d", {
                [&](){ mHarness.addAttributes<openvdb::math::Vec4<double>>(names,
                        { openvdb::math::Vec4<double>(1.1,2.3,4.3,5.4),
                          openvdb::math::Vec4<double>(-1.1,-2.3,-4.3,-5.4),
                          openvdb::math::Vec4<double>(0.0,0.0,0.0,0.0),
                          openvdb::math::Vec4<double>(0.0,0.0,0.0,0.0),
                          openvdb::math::Vec4<double>(0.0,0.0,0.0,0.0),
                          openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4) + openvdb::math::Vec4<double>(5.1, 6.3, 7.3, 8.4),
                          openvdb::math::Vec4<double>(0.0,0.0,0.0,0.0),
                          openvdb::math::Vec4<double>(0.0,0.0,0.0,0.0),
                          openvdb::math::Vec4<double>(5.1, 6.3, 7.3, 8.4) });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Vec4<double>>(names,
                        { openvdb::math::Vec4<double>(1.1,2.2,3.3,4.4),
                          openvdb::math::Vec4<double>(3.3,4.4,5.5,6.6),
                          openvdb::math::Vec4<double>(5.5,6.6,7.7,8.8),
                          openvdb::math::Vec4<double>(7.7,8.8,9.9,-1.1),
                          openvdb::math::Vec4<double>(7.7,8.8,9.9,-1.1),
                          openvdb::math::Vec4<double>(9.9,-1.1,-2.2,-3.3),
                          openvdb::math::Vec4<double>(-2.2,-3.3,-4.4,-5.5),
                          openvdb::math::Vec4<double>(-4.3,-5.5,-6.6,-7.7),
                          openvdb::math::Vec4<double>(-8.2,-9.3,0.6,-1.7) }, // in
                        { openvdb::math::Vec4<double>(1.1,2.2,3.3,4.4) + openvdb::math::Vec4<double>(1.1,2.3,4.3,5.4),
                          openvdb::math::Vec4<double>(3.3,4.4,5.5,6.6) - openvdb::math::Vec4<double>(1.1,2.3,4.3,5.4),
                          openvdb::math::Vec4<double>(5.5,6.6,7.7,8.8) * openvdb::math::Vec4<double>(1.1,2.3,4.3,5.4),
                          openvdb::math::Vec4<double>(7.7,8.8,9.9,-1.1) / openvdb::math::Vec4<double>(1.1,2.3,4.3,5.4),
                          openvdb::math::Vec4<double>(std::fmod(7.7,1.1),std::fmod(8.8,2.3),std::fmod(9.9,4.3),std::fmod(-1.1,5.4)+5.4), // floored mod
                          openvdb::math::Vec4<double>(1.1, 2.3, 4.3, 5.4) + openvdb::math::Vec4<double>(5.1, 6.3, 7.3, 8.4),
                          openvdb::math::Vec4<double>(-2.2,-3.3,-4.4,-5.5) + openvdb::math::Vec4<double>(-4.3,-5.5,-6.6,-7.7),
                          openvdb::math::Vec4<double>(-4.3,-5.5,-6.6,-7.7),
                          openvdb::math::Vec4<double>(-8.2,-9.3,0.6,-1.7) + openvdb::math::Vec4<double>(5.1, 6.3, 7.3, 8.4)
                        }); // expected
                }
            }
        }
    };

    for (const auto& expc : expected) {
        for (const auto& test : expc.second) {
            mHarness.reset();
            test.operator()();
            this->execute("assign_compound." + expc.first + ".ax");
        }
    }
}


void
TestAssign::compoundMatrixAssignment()
{
    const std::string code = R"(
_T1_@test1 += _l1_;
_T1_@test2 -= _l1_;
_T1_@test3 *= _l1_;

_T1_ local1 = _l1_,
      local2 = _l2_;

local1 += local2;
_T1_@test4 = local1;
_T1_@test5 += _T1_@test6;
_T1_@test7 += local2;
)";

    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", config.first); // replace type
            // replace literal values
            for (const auto& settings : config.second) {
                unittest_util::replace(repl, settings.first, settings.second);
            }

            this->registerTest(repl, "assign_compound." + config.first + ".ax");
        }
    };

    generate(mat3);
    generate(mat4);

    const openvdb::math::Mat3<float> m3fl1(1.1f, 2.3f, 4.3f,  5.4f, 6.7f, 7.8f,  9.1f, 4.5f, 8.2f);
    const openvdb::math::Mat3<float> m3fl2(9.1f, 7.3f,-1.3f,  4.4f,-6.7f, 0.8f,  9.1f,-0.5f, 8.2f);
    const openvdb::math::Mat3<double> m3dl1(1.1, 2.3, 4.3,  5.4, 6.7, 7.8,  9.1, 4.5, 8.2);
    const openvdb::math::Mat3<double> m3dl2(9.1, 7.3,-1.3,  4.4,-6.7, 0.8,  9.1,-0.5, 8.2);

    const openvdb::math::Mat4<float> m4fl1(1.1f, 2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f, 4.5f,  8.2f, 3.3f, 2.9f, 5.9f,  0.1f, 0.3f, 5.1f, 1.9f);
    const openvdb::math::Mat4<float> m4fl2(0.1f, 2.3f,-9.3f, 4.5f,  -1.7f, 7.8f, 2.1f, 3.3f,  3.3f,-3.3f,-0.3f, 2.5f,  5.1f, 0.5f, 8.1f,-1.7f);
    const openvdb::math::Mat4<double> m4dl1(1.1, 2.3, 4.3, 5.4,   6.7, 7.8, 9.1, 4.5,  8.2, 3.3, 2.9, 5.9,  0.1, 0.3, 5.1, 1.9);
    const openvdb::math::Mat4<double> m4dl2(0.1, 2.3,-9.3, 4.5,  -1.7, 7.8, 2.1, 3.3,  3.3,-3.3,-0.3, 2.5,  5.1, 0.5, 8.1,-1.7);

    const auto names = unittest_util::nameSequence("test", 7);
    const std::map<std::string, std::vector<std::function<void()>>> expected = {
        { "mat3f", {
                [&](){ mHarness.addAttributes<openvdb::math::Mat3<float>>(names,
                        { m3fl1,
                          -m3fl1,
                          openvdb::math::Mat3<float>::zero(),
                          m3fl1 + m3fl2,
                          openvdb::math::Mat3<float>::zero(),
                          openvdb::math::Mat3<float>::zero(),
                          m3fl2 });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Mat3<float>>(names,
                        { openvdb::math::Mat3<float>(2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f,   4.5f, 8.2f, 1.1f),
                          openvdb::math::Mat3<float>(4.3f, 5.4f, 6.7f,   7.8f, 9.1f, 4.5f,   8.2f, 1.1f, 2.3f),
                          openvdb::math::Mat3<float>(5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f,   1.1f, 2.3f, 4.3f),
                          openvdb::math::Mat3<float>(8.3f, 2.3f, 6.1f,   4.5f, 0.1f, 0.1f,   5.3f, 4.5f, 8.9f),
                          openvdb::math::Mat3<float>(6.7f, 7.8f, 9.1f,   4.5f, 8.2f, 1.1f,   2.3f, 4.3f, 5.4f),
                          openvdb::math::Mat3<float>(7.8f, 9.1f, 4.5f,   8.2f, 1.1f, 2.3f,   4.3f, 5.4f, 6.7f),
                          openvdb::math::Mat3<float>(-6.8f,-8.1f,-4.5f,  5.2f,-1.1f, 2.3f,  -0.3f, 5.4f,-3.7f)
                        }, // in
                        { openvdb::math::Mat3<float>(2.3f, 4.3f, 5.4f,   6.7f, 7.8f, 9.1f,   4.5f, 8.2f, 1.1f) + m3fl1,
                          openvdb::math::Mat3<float>(4.3f, 5.4f, 6.7f,   7.8f, 9.1f, 4.5f,   8.2f, 1.1f, 2.3f) - m3fl1,
                          openvdb::math::Mat3<float>(5.4f, 6.7f, 7.8f,   9.1f, 4.5f, 8.2f,   1.1f, 2.3f, 4.3f) * m3fl1,
                          m3fl1 + m3fl2,
                          openvdb::math::Mat3<float>(6.7f, 7.8f, 9.1f,   4.5f, 8.2f, 1.1f,   2.3f, 4.3f, 5.4f) +
                              openvdb::math::Mat3<float>(7.8f, 9.1f, 4.5f,   8.2f, 1.1f, 2.3f,   4.3f, 5.4f, 6.7f),
                          openvdb::math::Mat3<float>(7.8f, 9.1f, 4.5f,   8.2f, 1.1f, 2.3f,   4.3f, 5.4f, 6.7f),
                          openvdb::math::Mat3<float>(-6.8f,-8.1f,-4.5f,  5.2f,-1.1f, 2.3f,  -0.3f, 5.4f,-3.7f)  + m3fl2
                        }); // expected
                }
            }
        },
        { "mat3d", {
                [&](){ mHarness.addAttributes<openvdb::math::Mat3<double>>(names,
                        { m3dl1,
                          -m3dl1,
                          openvdb::math::Mat3<double>::zero(),
                          m3dl1 + m3dl2,
                          openvdb::math::Mat3<double>::zero(),
                          openvdb::math::Mat3<double>::zero(),
                          m3dl2 });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Mat3<double>>(names,
                        { openvdb::math::Mat3<double>(2.3, 4.3, 5.4,   6.7, 7.8, 9.1,   4.5, 8.2, 1.1),
                          openvdb::math::Mat3<double>(4.3, 5.4, 6.7,   7.8, 9.1, 4.5,   8.2, 1.1, 2.3),
                          openvdb::math::Mat3<double>(5.4, 6.7, 7.8,   9.1, 4.5, 8.2,   1.1, 2.3, 4.3),
                          openvdb::math::Mat3<double>(8.3, 2.3, 6.1,   4.5, 0.1, 0.1,   5.3, 4.5, 8.9),
                          openvdb::math::Mat3<double>(6.7, 7.8, 9.1,   4.5, 8.2, 1.1,   2.3, 4.3, 5.4),
                          openvdb::math::Mat3<double>(7.8, 9.1, 4.5,   8.2, 1.1, 2.3,   4.3, 5.4, 6.7),
                          openvdb::math::Mat3<double>(-6.8,-8.1,-4.5,  5.2,-1.1, 2.3,  -0.3, 5.4,-3.7)
                        }, // in
                        { openvdb::math::Mat3<double>(2.3, 4.3, 5.4,   6.7, 7.8, 9.1,   4.5, 8.2, 1.1) + m3dl1,
                          openvdb::math::Mat3<double>(4.3, 5.4, 6.7,   7.8, 9.1, 4.5,   8.2, 1.1, 2.3) - m3dl1,
                          openvdb::math::Mat3<double>(5.4, 6.7, 7.8,   9.1, 4.5, 8.2,   1.1, 2.3, 4.3) * m3dl1,
                          m3dl1 + m3dl2,
                          openvdb::math::Mat3<double>(6.7, 7.8, 9.1,   4.5, 8.2, 1.1,   2.3, 4.3, 5.4) +
                              openvdb::math::Mat3<double>(7.8, 9.1, 4.5,   8.2, 1.1, 2.3,   4.3, 5.4, 6.7),
                          openvdb::math::Mat3<double>(7.8, 9.1, 4.5,   8.2, 1.1, 2.3,   4.3, 5.4, 6.7),
                          openvdb::math::Mat3<double>(-6.8,-8.1,-4.5,  5.2,-1.1, 2.3,  -0.3, 5.4,-3.7)  + m3dl2
                        }); // expected
                }
            }
        },
        { "mat4f", {
                [&](){ mHarness.addAttributes<openvdb::math::Mat4<float>>(names,
                        { m4fl1,
                          -m4fl1,
                          openvdb::math::Mat4<float>::zero(),
                          m4fl1 + m4fl2,
                          openvdb::math::Mat4<float>::zero(),
                          openvdb::math::Mat4<float>::zero(),
                          m4fl2 });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Mat4<float>>(names,
                        { openvdb::math::Mat4<float>(2.3f,-4.3f, 5.4f, 6.7f,  7.8f,-9.1f, 4.5f, 8.2f,  1.1f,-5.4f,-6.7f, 7.8f,  6.7f, 7.8f, 9.1f,-2.4f),
                          openvdb::math::Mat4<float>(4.3f,-5.4f, 6.7f, 7.8f,  9.1f, 4.5f, 8.2f, 1.1f,  2.3f, 6.7f, 7.8f, 9.1f, -1.3f,-0.1f, 1.1f, 0.9f),
                          openvdb::math::Mat4<float>(5.4f, 6.7f, 7.8f, 9.1f, -4.5f, 8.2f, 1.1f,-2.3f, -4.3f,-7.8f, 9.1f, 4.5f, -6.7f, 2.2f,-7.1f, 1.1f),
                          openvdb::math::Mat4<float>(1.2f, 5.1f, 8.2f, 3.1f, -3.3f, -7.3f, 0.2f,-0.1f,  1.4f, 0.8f, 8.8f,-1.1f, -7.8f, 4.1f, 4.4f, -4.7f),
                          openvdb::math::Mat4<float>(5.4f, 6.7f, 8.2f, 1.1f, -2.3f, -4.3f, 2.2f,-7.1f,  1.1f, 7.8f, 9.1f,-4.5f, -7.8f, 9.1f, 4.5f, -6.7f),
                          openvdb::math::Mat4<float>(8.2f, 1.1f, 6.3f,-4.3f,  9.1f, -4.5f,-7.8f, 9.1f,  4.5f, 6.7f,-5.4f, 6.7f,  2.2f,-7.1f, 1.1f, 7.8f),
                          openvdb::math::Mat4<float>(4.3f,-5.1f,-5.3f, 2.2f,  2.1f, -4.2f, 2.3f,-1.1f,  0.5f, 0.7f, 1.3f, 0.7f, -1.2f, 3.4f, 9.9f, 9.8f),
                        }, // in
                        { openvdb::math::Mat4<float>(2.3f,-4.3f, 5.4f, 6.7f,  7.8f,-9.1f, 4.5f, 8.2f,  1.1f,-5.4f,-6.7f, 7.8f,  6.7f, 7.8f, 9.1f,-2.4f) + m4fl1,
                          openvdb::math::Mat4<float>(4.3f,-5.4f, 6.7f, 7.8f,  9.1f, 4.5f, 8.2f, 1.1f,  2.3f, 6.7f, 7.8f, 9.1f, -1.3f,-0.1f, 1.1f, 0.9f) - m4fl1,
                          openvdb::math::Mat4<float>(5.4f, 6.7f, 7.8f, 9.1f, -4.5f, 8.2f, 1.1f,-2.3f, -4.3f,-7.8f, 9.1f, 4.5f, -6.7f, 2.2f,-7.1f, 1.1f) * m4fl1,
                          m4fl1 + m4fl2,
                          openvdb::math::Mat4<float>(5.4f, 6.7f, 8.2f, 1.1f, -2.3f, -4.3f, 2.2f,-7.1f,  1.1f, 7.8f, 9.1f,-4.5f, -7.8f, 9.1f, 4.5f, -6.7f) +
                              openvdb::math::Mat4<float>(8.2f, 1.1f, 6.3f,-4.3f,  9.1f, -4.5f,-7.8f, 9.1f,  4.5f, 6.7f,-5.4f, 6.7f,  2.2f,-7.1f,  1.1f, 7.8f),
                          openvdb::math::Mat4<float>(8.2f, 1.1f, 6.3f,-4.3f,  9.1f, -4.5f,-7.8f, 9.1f,  4.5f, 6.7f,-5.4f, 6.7f,  2.2f,-7.1f,  1.1f, 7.8f),
                          openvdb::math::Mat4<float>(4.3f,-5.1f,-5.3f, 2.2f,  2.1f, -4.2f, 2.3f,-1.1f,  0.5f, 0.7f, 1.3f, 0.7f, -1.2f, 3.4f, 9.9f, 9.8f) + m4fl2
                        }); // expected
                }
            }
        },
        { "mat4d", {
                [&](){ mHarness.addAttributes<openvdb::math::Mat4<double>>(names,
                        { m4dl1,
                          -m4dl1,
                          openvdb::math::Mat4<double>::zero(),
                          m4dl1 + m4dl2,
                          openvdb::math::Mat4<double>::zero(),
                          openvdb::math::Mat4<double>::zero(),
                          m4dl2 });
                },
                [&](){ mHarness.addAttributes<openvdb::math::Mat4<double>>(names,
                        { openvdb::math::Mat4<double>(2.3,-4.3, 5.4, 6.7,  7.8,-9.1, 4.5, 8.2,  1.1,-5.4,-6.7, 7.8,  6.7, 7.8, 9.1,-2.4),
                          openvdb::math::Mat4<double>(4.3,-5.4, 6.7, 7.8,  9.1, 4.5, 8.2, 1.1,  2.3, 6.7, 7.8, 9.1, -1.3,-0.1, 1.1, 0.9),
                          openvdb::math::Mat4<double>(5.4, 6.7, 7.8, 9.1, -4.5, 8.2, 1.1,-2.3, -4.3,-7.8, 9.1, 4.5, -6.7, 2.2,-7.1, 1.1),
                          openvdb::math::Mat4<double>(1.2, 5.1, 8.2, 3.1, -3.3, -7.3, 0.2,-0.1,  1.4, 0.8, 8.8,-1.1, -7.8, 4.1, 4.4, -4.7),
                          openvdb::math::Mat4<double>(5.4, 6.7, 8.2, 1.1, -2.3, -4.3, 2.2,-7.1,  1.1, 7.8, 9.1,-4.5, -7.8, 9.1, 4.5, -6.7),
                          openvdb::math::Mat4<double>(8.2, 1.1, 6.3,-4.3,  9.1, -4.5,-7.8, 9.1,  4.5, 6.7,-5.4, 6.7,  2.2,-7.1, 1.1, 7.8),
                          openvdb::math::Mat4<double>(4.3,-5.1,-5.3, 2.2,  2.1, -4.2, 2.3,-1.1,  0.5, 0.7, 1.3, 0.7, -1.2, 3.4, 9.9, 9.8),
                        }, // in
                        { openvdb::math::Mat4<double>(2.3,-4.3, 5.4, 6.7,  7.8,-9.1, 4.5, 8.2,  1.1,-5.4,-6.7, 7.8,  6.7, 7.8, 9.1,-2.4) + m4dl1,
                          openvdb::math::Mat4<double>(4.3,-5.4, 6.7, 7.8,  9.1, 4.5, 8.2, 1.1,  2.3, 6.7, 7.8, 9.1, -1.3,-0.1, 1.1, 0.9) - m4dl1,
                          openvdb::math::Mat4<double>(5.4, 6.7, 7.8, 9.1, -4.5, 8.2, 1.1,-2.3, -4.3,-7.8, 9.1, 4.5, -6.7, 2.2,-7.1, 1.1) * m4dl1,
                          m4dl1 + m4dl2,
                          openvdb::math::Mat4<double>(5.4, 6.7, 8.2, 1.1, -2.3, -4.3, 2.2,-7.1,  1.1, 7.8, 9.1,-4.5, -7.8, 9.1, 4.5, -6.7) +
                              openvdb::math::Mat4<double>(8.2, 1.1, 6.3,-4.3,  9.1, -4.5,-7.8, 9.1,  4.5, 6.7,-5.4, 6.7,  2.2,-7.1,  1.1, 7.8),
                          openvdb::math::Mat4<double>(8.2, 1.1, 6.3,-4.3,  9.1, -4.5,-7.8, 9.1,  4.5, 6.7,-5.4, 6.7,  2.2,-7.1,  1.1, 7.8),
                          openvdb::math::Mat4<double>(4.3,-5.1,-5.3, 2.2,  2.1, -4.2, 2.3,-1.1,  0.5, 0.7, 1.3, 0.7, -1.2, 3.4, 9.9, 9.8) + m4dl2
                        }); // expected
                }
            }
        }
    };

    for (const auto& expc : expected) {
        for (const auto& test : expc.second) {
            mHarness.reset();
            test.operator()();
            this->execute("assign_compound." + expc.first + ".ax");
        }
    }
}


void
TestAssign::compoundStringAssignment()
{
    const std::string code = R"(
_T1_@test1 += _l1_;

_T1_ local1 = _l1_,
      local2 = _l2_;

// test default init and empty string
string empty = "";
string defaultstr;
local1 += local2;
defaultstr += local1;
defaultstr += empty;

_T1_@test2 = defaultstr;
_T1_@test3 += _T1_@test4;
_T1_@test5 += local2;
)";

    auto generate = [&](const auto& map) {
        for (const auto& config : map) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", config.first); // replace type
            // replace literal values
            for (const auto& settings : config.second) {
                unittest_util::replace(repl, settings.first, settings.second);
            }

            this->registerTest(repl, "assign_compound." + config.first + ".ax");
        }
    };

    generate(string);

    const auto names = unittest_util::nameSequence("test", 5);
    const std::map<std::string, std::vector<std::function<void()>>> expected = {
        { "string", {
                [&](){ mHarness.addAttributes<std::string>(names,
                        { "foo", "foobar", "", "", "bar" });
                },
                [&](){ mHarness.addAttributes<std::string>(names,
                        { "abc ", "xyz", " 123", "4560", " " },            // in
                        { "abc foo", "foobar", " 1234560", "4560", " bar" }); // expected
                },
            }
        }
    };

    for (const auto& expc : expected) {
        for (const auto& test : expc.second) {
            mHarness.reset();
            test.operator()();
            this->execute("assign_compound." + expc.first + ".ax");
        }
    }
}


void
TestAssign::implicitScalarAssignment()
{
    auto generate = [this](const auto& source, const auto& targets) {
        for (const auto& t1 : source) {
            std::string code = "_T1_ local = _l1_;\n";
            unittest_util::replace(code, "_T1_", t1.first);
            unittest_util::replace(code, "_l1_", t1.second.at("_l1_"));

            for (const auto& target : targets) {
                for (const auto& t2 : *target) {
                    if (t1.first == t2.first) continue;
                    std::string tmp = "_T2_@_A1_ = local;";
                    unittest_util::replace(tmp, "_A1_", "test" +  t2.first);
                    unittest_util::replace(tmp, "_T2_", t2.first);
                    code += tmp + "\n";
                }
            }

            this->registerTest(code, "assign_implicit_scalar." + t1.first + ".ax");
        }
    };

    // source -> dest
    generate(integral, std::vector<decltype(integral)*>{ &integral, &floating });
    generate(floating, std::vector<decltype(integral)*>{ &integral, &floating });

    // source -> dest
    const std::map<std::string, std::function<void()>> expected = {
        { "bool", [&](){
             mHarness.addAttribute<int32_t>("testint32", 1);
             mHarness.addAttribute<int64_t>("testint64", 1);
             mHarness.addAttribute<float>("testfloat", 1.0f);
             mHarness.addAttribute<double>("testdouble", 1.0);
            }
        },
        { "int32", [&](){
             mHarness.addAttribute<bool>("testbool", true);
             mHarness.addAttribute<int64_t>("testint64", 2);
             mHarness.addAttribute<float>("testfloat", 2.0f);
             mHarness.addAttribute<double>("testdouble", 2.0);
            }
        },
        { "int64", [&](){
             mHarness.addAttribute<bool>("testbool", true);
             mHarness.addAttribute<int32_t>("testint32", 2);
             mHarness.addAttribute<float>("testfloat", 2.0f);
             mHarness.addAttribute<double>("testdouble", 2.0);
            }
        },
        { "float", [&](){
             mHarness.addAttribute<bool>("testbool", true);
             mHarness.addAttribute<int32_t>("testint32", 1);
             mHarness.addAttribute<int64_t>("testint64", 1);
             mHarness.addAttribute<double>("testdouble", double(1.1f));
            }
        },
        { "double", [&](){
             mHarness.addAttribute<bool>("testbool", true);
             mHarness.addAttribute<int32_t>("testint32", 1);
             mHarness.addAttribute<int64_t>("testint64", 1);
             mHarness.addAttribute<float>("testfloat", float(1.1));
            }
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("assign_implicit_scalar." + expc.first + ".ax");
    }
}


void
TestAssign::implicitContainerAssignment()
{
    auto generate = [this](const auto& source, const auto& target) {
        for (const auto& t1 : source) {
            std::string code = "_T1_ local = _l1_;\n";
            unittest_util::replace(code, "_T1_", t1.first);
            unittest_util::replace(code, "_l1_", t1.second.at("_l1_"));

            for (const auto& t2 : target) {
                if (t1.first == t2.first) continue;
                std::string tmp = "_T2_@_A1_ = local;";
                unittest_util::replace(tmp, "_A1_", "test" +  t2.first);
                unittest_util::replace(tmp, "_T2_", t2.first);
                code += tmp + "\n";
            }

            this->registerTest(code, "assign_implicit_container." + t1.first + ".ax");
        }
    };

    // source -> dest
    generate(vec2, vec2);
    generate(vec3, vec3);
    generate(vec4, vec4);
    generate(mat3, mat3);
    generate(mat4, mat4);

    // test name is the source type in use. source -> dest
    const std::map<std::string, std::function<void()>> expected = {
        { "vec2i", [&]() {
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(1.0f,2.0f));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(1.0,2.0));
            }
        },
        { "vec2f", [&]() {
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(1,2));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(double(1.1f),double(2.3f)));
            }
        },
        { "vec2d", [&]() {
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(1,2));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(float(1.1),float(2.3)));
            }
        },
        { "vec3i", [&]() {
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(1.0f,2.0f,3.0f));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(1.0,2.0,3.0));
            }
        },
        { "vec3f", [&]() {
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(1,2,4));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(double(1.1f),double(2.3f),double(4.3f)));
            }
        },
        { "vec3d", [&]() {
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(1,2,4));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(float(1.1),float(2.3),float(4.3)));
            }
        },
        { "vec4i", [&]() {
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(1.0f,2.0f,3.0f,4.0f));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(1.0,2.0,3.0,4.0));
            }
        },
        { "vec4f", [&]() {
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(1,2,4,5));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(double(1.1f),double(2.3f),double(4.3f),double(5.4f)));
            }
        },
        { "vec4d", [&]() {
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(1,2,4,5));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(float(1.1),float(2.3),float(4.3),float(5.4)));
            }
        },
        { "mat3f",
            [&](){ mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d",
                     openvdb::math::Mat3<double>(
                        double(1.1f),double(2.3f),double(4.3f),
                        double(5.4f),double(6.7f),double(7.8f),
                        double(9.1f),double(4.5f),double(8.2f)));
            }
        },
        { "mat3d",
            [&](){ mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f",
                     openvdb::math::Mat3<float>(
                         float(1.1),float(2.3),float(4.3),
                         float(5.4),float(6.7),float(7.8),
                         float(9.1),float(4.5),float(8.2)));
            }
        },
        { "mat4f",
            [&](){ mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d",
                openvdb::math::Mat4<double>(
                    double(1.1f),double(2.3f),double(4.3f),double(5.4f),
                    double(6.7f),double(7.8f),double(9.1f),double(4.5f),
                    double(8.2f),double(3.3f),double(2.9f),double(5.9f),
                    double(0.1f),double(0.3f),double(5.1f),double(1.9f)));
            }
        },
        { "mat4d",
            [&](){ mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f",
                    openvdb::math::Mat4<float>(
                        float(1.1),float(2.3),float(4.3),float(5.4),
                        float(6.7),float(7.8),float(9.1),float(4.5),
                        float(8.2),float(3.3),float(2.9),float(5.9),
                        float(0.1),float(0.3),float(5.1),float(1.9)));
            }
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("assign_implicit_container." + expc.first + ".ax");
    }
}


void
TestAssign::implicitContainerScalarAssignment()
{
    auto generate = [this](const auto& source, const auto& targets) {
        for (const auto& t1 : source) {
            std::string code = "_T1_ local = _l1_;\n";
            unittest_util::replace(code, "_T1_", t1.first);
            unittest_util::replace(code, "_l1_", t1.second.at("_l1_"));

            for (const auto& target : targets) {
                for (const auto& t2 : *target) {
                    if (t1.first == t2.first) continue;
                    std::string tmp = "_T2_@_A1_ = local;";
                    unittest_util::replace(tmp, "_A1_", "test" +  t2.first);
                    unittest_util::replace(tmp, "_T2_", t2.first);
                    code += tmp + "\n";
                }
            }

            this->registerTest(code, "assign_implicit_container_scalar." + t1.first + ".ax");
        }
    };

    generate(integral, std::vector<decltype(integral)*>{ &vec2, &vec3, &vec4, &mat3, &mat4 });
    generate(floating, std::vector<decltype(integral)*>{ &vec2, &vec3, &vec4, &mat3, &mat4 });

    auto symmetric3 = [](auto val) -> openvdb::math::Mat3<decltype(val)> {
        openvdb::math::Mat3<decltype(val)> mat;
        mat.setZero();
        mat(0,0) = val;
        mat(1,1) = val;
        mat(2,2) = val;
        return mat;
    };

    auto symmetric4 = [](auto val) -> openvdb::math::Mat4<decltype(val)> {
        openvdb::math::Mat4<decltype(val)> mat;
        mat.setZero();
        mat(0,0) = val;
        mat(1,1) = val;
        mat(2,2) = val;
        mat(3,3) = val;
        return mat;
    };

    const std::map<std::string, std::function<void()>> expected = {
        { "bool", [&]() {
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(1,1));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(1.0f,1.0f));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(1.0,1.0));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(1,1,1));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(1.0f,1.0f,1.0f));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(1.0,1.0,1.0));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(1,1,1,1));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(1.0f,1.0f,1.0f,1.0f));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(1.0,1.0,1.0,1.0));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f", symmetric3(1.0f));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d", symmetric3(1.0));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f", symmetric4(1.0f));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d", symmetric4(1.0));
            }
        },
        { "int32", [&](){
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(2,2));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(2.0,2.0));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(2.0f,2.0f));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(2,2,2));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(2.0,2.0,2.0));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(2.0f,2.0f,2.0f));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(2,2,2,2));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(2.0,2.0,2.0,2.0));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(2.0f,2.0f,2.0f,2.0f));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f", symmetric3(2.0f));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d", symmetric3(2.0));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f", symmetric4(2.0f));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d", symmetric4(2.0));
            }
        },
        { "int64", [&](){
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(2,2));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(2.0,2.0));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(2.0f,2.0f));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(2,2,2));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(2.0,2.0,2.0));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(2.0f,2.0f,2.0f));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(2,2,2,2));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(2.0,2.0,2.0,2.0));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(2.0f,2.0f,2.0f,2.0f));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f", symmetric3(2.0f));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d", symmetric3(2.0));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f", symmetric4(2.0f));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d", symmetric4(2.0));
            }
        },
        { "float", [&](){
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(1,1));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(double(1.1f),double(1.1f)));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(1.1f,1.1f));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(1,1,1));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(double(1.1f),double(1.1f),double(1.1f)));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(1.1f,1.1f,1.1f));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(1,1,1,1));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(double(1.1f),double(1.1f),double(1.1f),double(1.1f)));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(1.1f,1.1f,1.1f,1.1f));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f", symmetric3(1.1f));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d", symmetric3(double(1.1f)));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f", symmetric4(1.1f));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d", symmetric4(double(1.1f)));
            }
        },
        { "double",  [&](){
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i", openvdb::math::Vec2<int32_t>(1,1));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d", openvdb::math::Vec2<double>(1.1,1.1));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f", openvdb::math::Vec2<float>(float(1.1),float(1.1)));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i", openvdb::math::Vec3<int32_t>(1,1,1));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d", openvdb::math::Vec3<double>(1.1,1.1,1.1));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f", openvdb::math::Vec3<float>(float(1.1),float(1.1),float(1.1)));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i", openvdb::math::Vec4<int32_t>(1,1,1,1));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d", openvdb::math::Vec4<double>(1.1,1.1,1.1,1.1));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f", openvdb::math::Vec4<float>(float(1.1),float(1.1),float(1.1),float(1.1)));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f", symmetric3(float(1.1)));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d", symmetric3(1.1));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f", symmetric4(float(1.1)));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d", symmetric4(1.1));
            }
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("assign_implicit_container_scalar." + expc.first + ".ax");
    }
}


void
TestAssign::scopedAssign()
{
    const std::string code = R"(
float var = 30.0f;

{
    float var = 3.0f;
}
{
    float var = 1.0f;
    float@test2 = var;
    {
        float var = -10.0f;
        float@test3 = var;
    }
    {
        float@test7 = var;
    }
}
{
    float var = -100.0f;
}
{
    float var = 50.0f;
    {
        float var = -15.0f;
        float@test4 = var;
    }
    {
        float var = -10.0f;
    }
    {
        float@test5 = var;
    }

    float@test6 = var;
}

float@test1 = var;
)";

    this->registerTest(code, "assign_scoped.float.ax");

    const auto names = unittest_util::nameSequence("test", 7);
    mHarness.addAttributes<float>(names, {30.0f, 1.0f, -10.0f, -15.0f, 50.0f, 50.0f, 1.0f});
    this->execute("assign_scoped.float.ax");

    CPPUNIT_ASSERT(mHarness.mLogger.hasWarning());
}

