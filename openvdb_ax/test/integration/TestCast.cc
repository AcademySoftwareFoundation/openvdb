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

#include <openvdb_ax/test/util.h>

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

    generate(std::vector<std::string>{ "bool", "short", "int", "long", "float", "double" });

    const auto names = unittest_util::nameSequence("test", 5);
    const std::map<std::string, std::function<void()>> expected = {
        { "bool", [&](){
             mHarness.addAttribute<int16_t>("testshort", 1, 1);
             mHarness.addAttribute<int32_t>("testint", 1, 1);
             mHarness.addAttribute<int64_t>("testlong", 0, 0);
             mHarness.addAttribute<float>("testfloat", 2.3f, 2.3f);
             mHarness.addAttribute<double>("testdouble", 0.1, 0.1);
             mHarness.addAttributes<bool>(names, {true, true, false, true, true});
            }
        },
        { "short", [&](){
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int32_t>("testint", 2, 2);
             mHarness.addAttribute<int64_t>("testlong", 2, 2);
             mHarness.addAttribute<float>("testfloat", 2.3f, 2.3f);
             mHarness.addAttribute<double>("testdouble", 2.1, 2.1);
             mHarness.addAttributes<int16_t>(names, {1, 2, 2, 2, 2});
            }
        },
        { "int", [&](){
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int16_t>("testshort", 2, 2);
             mHarness.addAttribute<int64_t>("testlong", 2, 2);
             mHarness.addAttribute<float>("testfloat", 2.3f, 2.3f);
             mHarness.addAttribute<double>("testdouble", 2.1, 2.1);
             mHarness.addAttributes<int32_t>(names, {1, 2, 2, 2, 2});
            }
        },
        { "long", [&]() {
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int16_t>("testshort", 2, 2);
             mHarness.addAttribute<int32_t>("testint", 2, 2);
             mHarness.addAttribute<float>("testfloat", 2.3f, 2.3f);
             mHarness.addAttribute<double>("testdouble", 2.1, 2.1);
             mHarness.addAttributes<int64_t>(names, {1, 2, 2, 2, 2});
            }
        },
        { "float", [&]() {
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int16_t>("testshort", 1, 1);
             mHarness.addAttribute<int32_t>("testint", 1, 1);
             mHarness.addAttribute<int64_t>("testlong", 1, 1);
             mHarness.addAttribute<double>("testdouble", 1.1, 1.1);
             mHarness.addAttributes<float>(names, {1.0f, 1.0f, 1.0f, 1.0f, float(1.1)});
            }
        },
        { "double", [&]() {
             mHarness.addAttribute<bool>("testbool", true, true);
             mHarness.addAttribute<int16_t>("testshort", 1, 1);
             mHarness.addAttribute<int32_t>("testint", 1, 1);
             mHarness.addAttribute<int64_t>("testlong", 1, 1);
             mHarness.addAttribute<float>("testfloat", 1.1f, 1.1f);
             mHarness.addAttributes<double>(names, {1.0, 1.0, 1.0, 1.0, double(1.1f)});
            }
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("cast_explicit." + expc.first + ".ax");
    }
}


// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
