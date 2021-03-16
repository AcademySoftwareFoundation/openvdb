// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include "../util.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestCast : public unittest_util::AXTestCase
{
public:
    std::string dir() const override { return GET_TEST_DIRECTORY(); }

    CPPUNIT_TEST_SUITE(TestCast);
    CPPUNIT_TEST(explicitScalar);
    CPPUNIT_TEST_SUITE_END();

    void explicitScalar();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCast);


void
TestCast::explicitScalar()
{
    auto generate = [this](const auto& types) {
        for (const auto& t1 : types) {
            std::string code;
            size_t idx = 1;
            for (const auto& t2 : types) {
                if (t1 == t2) continue;
                std::string tmp = "_T1_@_A1_ = _T1_(_T2_@_A2_);";
                unittest_util::replace(tmp, "_A1_", "test" +  std::to_string(idx));
                unittest_util::replace(tmp, "_A2_", "test" +  t2);
                unittest_util::replace(tmp, "_T1_", t1);
                unittest_util::replace(tmp, "_T2_", t2);
                code += tmp + "\n";
                ++idx;
            }

            this->registerTest(code, "cast_explicit." + t1 + ".ax");
        }
    };

    generate(std::vector<std::string>{ "bool", "int32", "int64", "float", "double" });

    const auto names = unittest_util::nameSequence("test", 4);
    const std::map<std::string, std::function<void()>> expected = {
        { "bool", [&](){
             mHarness.addAttribute<int32_t>("testint32", 1, 1);
             mHarness.addAttribute<int64_t>("testint64", 0, 0);
             mHarness.addAttribute<float>("testfloat", 2.3f, 2.3f);
             mHarness.addAttribute<double>("testdouble", 0.1, 0.1);
             mHarness.addAttributes<bool>(names, {true, false, true, true});
            }
        },
        { "int32", [&](){
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int64_t>("testint64", 2, 2);
             mHarness.addAttribute<float>("testfloat", 2.3f, 2.3f);
             mHarness.addAttribute<double>("testdouble", 2.1, 2.1);
             mHarness.addAttributes<int32_t>(names, {1, 2, 2, 2});
            }
        },
        { "int64", [&]() {
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int32_t>("testint32", 2, 2);
             mHarness.addAttribute<float>("testfloat", 2.3f, 2.3f);
             mHarness.addAttribute<double>("testdouble", 2.1, 2.1);
             mHarness.addAttributes<int64_t>(names, {1, 2, 2, 2});
            }
        },
        { "float", [&]() {
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int32_t>("testint32", 1, 1);
             mHarness.addAttribute<int64_t>("testint64", 1, 1);
             mHarness.addAttribute<double>("testdouble", 1.1, 1.1);
             mHarness.addAttributes<float>(names, {1.0f, 1.0f, 1.0f, float(1.1)});
            }
        },
        { "double", [&]() {
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int32_t>("testint32", 1, 1);
             mHarness.addAttribute<int64_t>("testint64", 1, 1);
             mHarness.addAttribute<float>("testfloat", 1.1f, 1.1f);
             mHarness.addAttributes<double>(names, {1.0, 1.0, 1.0, double(1.1f)});
            }
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("cast_explicit." + expc.first + ".ax");
    }
}


