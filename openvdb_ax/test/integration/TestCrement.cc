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

#include "CompareGrids.h"
#include "TestHarness.h"

#include <openvdb_ax/test/util.h>

#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/Exceptions.h>

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestCrement : public unittest_util::AXTestCase
{
public:

    std::string dir() const override { return GET_TEST_DIRECTORY(); }

    CPPUNIT_TEST_SUITE(TestCrement);
    CPPUNIT_TEST(crementScalar);
    CPPUNIT_TEST(crementComponent);
    CPPUNIT_TEST_SUITE_END();

    void crementScalar();
    void crementComponent();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCrement);

void
TestCrement::crementScalar()
{
    const std::string code = R"(
_T1_@test1 = ++_T1_@test2;
_T1_@test3 = _T1_@test4++;
_T1_@test5 = (_T1_@test6++, _T1_@test7++, ++_T1_@test6);
_T1_@test8 = (++_T1_@test6, ++_T1_@test7, _T1_@test6++);
++_T1_@test9 = _T1_@test9;
)";

    auto generate = [&](const auto& types) {
        for (const auto& type : types) {
            std::string repl = code;
            unittest_util::replace(repl, "_T1_", type);
            this->registerTest(repl, "crement_inc." + type + ".ax");
            unittest_util::replace(repl, "++", "--");
            this->registerTest(repl, "crement_dec." + type + ".ax");
        }
    };

    generate(std::vector<std::string>{
        "short", "int", "long", "float", "double",
    });

    const auto names = unittest_util::nameSequence("test", 9);
    const std::map<std::string, std::function<void(bool)>> expected = {
        { "short",
            [&](bool inc){
                if (inc)
                    mHarness.addAttributes<int16_t>(names,
                        { 0, 1, -1, 2, -3, 4, -3, 4, -4 },
                        { 2, 2,  2, 3,  6, 8, -1, 7, -3 });
                else
                    mHarness.addAttributes<int16_t>(names,
                        { 0, 1, -1, 2, -3, 4, -3, 4, -4 },
                        { 0, 0,  2, 1,  2, 0, -5, 1, -5 });
            },
        },
        { "int",
            [&](bool inc){
                if (inc)
                    mHarness.addAttributes<int32_t>(names,
                        { 0, 1, -1, 2, -3, 4, -3, 4, -4 },
                        { 2, 2,  2, 3,  6, 8, -1, 7, -3 });
                else
                    mHarness.addAttributes<int32_t>(names,
                        { 0, 1, -1, 2, -3, 4, -3, 4, -4 },
                        { 0, 0,  2, 1,  2, 0, -5, 1, -5 });
            },
        },
        { "long",
            [&](bool inc){
                if (inc)
                    mHarness.addAttributes<int64_t>(names,
                        { 0, 1, -1, 2, -3, 4, -3, 4, -4 },
                        { 2, 2,  2, 3,  6, 8, -1, 7, -3 });
                else
                    mHarness.addAttributes<int64_t>(names,
                        { 0, 1, -1, 2, -3, 4, -3, 4, -4 },
                        { 0, 0,  2, 1,  2, 0, -5, 1, -5 });
             },
        },
        { "float",
            [&](bool inc){
                if (inc)
                    mHarness.addAttributes<float>(names,
                        { 0.1f, 1.4f, -1.9f, 2.5f, -3.3f, 4.5f, -3.3f, 4.7f, -4.8f },
                        { (1.4f+1.0f),
                          (1.4f+1.0f), 2.5f,
                          (2.5f+1.0f),
                          (4.5f+1.0f+1.0f),
                          (4.5f+1.0f+1.0f+1.0f+1.0f),
                          (-3.3f+1.0f+1.0f),
                          (4.5f+1.0f+1.0f+1.0f),
                          (-4.8f+1.0f) });
                else
                    mHarness.addAttributes<float>(names,
                        { 0.1f, 1.4f, -1.9f, 2.5f, -3.3f, 4.5f, -3.3f, 4.7f, -4.8f },
                        { (1.4f-1.0f),
                          (1.4f-1.0f), 2.5f,
                          (2.5f-1.0f),
                          (4.5f-1.0f-1.0f),
                          (4.5f-1.0f-1.0f-1.0f-1.0f),
                          (-3.3f-1.0f-1.0f),
                          (4.5f-1.0f-1.0f-1.0f),
                          (-4.8f-1.0f) });
            },
        },
        { "double",
            [&](bool inc){
                if (inc)
                    mHarness.addAttributes<double>(names,
                        { 0.1, 1.4, -1.9, 2.5, -3.3, 4.5, -3.3, 4.7, -4.8 },
                        { (1.4+1.0),
                          (1.4+1.0), 2.5,
                          (2.5+1.0),
                          (4.5+1.0+1.0),
                          (4.5+1.0+1.0+1.0+1.0),
                          (-3.3+1.0+1.0),
                          (4.5+1.0+1.0+1.0),
                          (-4.8+1.0) });
                else
                    mHarness.addAttributes<double>(names,
                        { 0.1, 1.4, -1.9, 2.5, -3.3, 4.5, -3.3, 4.7, -4.8 },
                        { (1.4-1.0),
                          (1.4-1.0), 2.5,
                          (2.5-1.0),
                          (4.5-1.0-1.0),
                          (4.5-1.0-1.0-1.0-1.0),
                          (-3.3-1.0-1.0),
                          (4.5-1.0-1.0-1.0),
                          (-4.8-1.0) });
            },
        },
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()(true); // increment
        this->execute("crement_inc." + expc.first + ".ax");

        mHarness.reset();
        expc.second.operator()(false); // decrement
        this->execute("crement_dec." + expc.first + ".ax");
    }
}


void
TestCrement::crementComponent()
{
    // Just tests the first two components of every container
    const std::string code = R"(
_T1_@_A1_[0] = ++_T1_@_A2_[0];
_T1_@_A1_[1] = _T1_@_A2_[1]++;
)";

    auto generate = [&](const auto& types) {
        std::string repl;
        for (const auto& type : types) {
            std::string tmp = code;
            unittest_util::replace(tmp, "_T1_", type);
            unittest_util::replace(tmp, "_A1_", "test" + type + "1");
            unittest_util::replace(tmp, "_A2_", "test" + type + "2");
            repl += tmp;
        }

        this->registerTest(repl, "crement_inc.component.ax");
        unittest_util::replace(repl, "++", "--");
        this->registerTest(repl, "crement_dec.component.ax");
    };

    generate(std::vector<std::string>{
        "vec2i", "vec2f", "vec2d",
        "vec3i", "vec3f", "vec3d",
        "vec4i", "vec4f", "vec4d",
        "mat3f", "mat3d",
        "mat4f", "mat4d"
    });

    const std::map<std::string, std::function<void()>> expected = {
        { "inc",
            [&](){
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i1", openvdb::math::Vec2<int32_t>(0,1));
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i2",
                    openvdb::math::Vec2<int32_t>(-1,1), openvdb::math::Vec2<int32_t>(0,2));

                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f1", openvdb::math::Vec2<float>(-1.1f+1.0f, 1.1f));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f2",
                    openvdb::math::Vec2<float>(-1.1f,1.1f), openvdb::math::Vec2<float>(-1.1f+1.0f, 1.1f+1.0f));

                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d1", openvdb::math::Vec2<double>(-1.1+1.0, 1.1));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d2",
                    openvdb::math::Vec2<double>(-1.1,1.1), openvdb::math::Vec2<double>(-1.1+1.0, 1.1+1.0));


                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i1", openvdb::math::Vec3<int32_t>(0,1,0));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i2",
                    openvdb::math::Vec3<int32_t>(-1,1,0), openvdb::math::Vec3<int32_t>(0,2,0));

                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f1", openvdb::math::Vec3<float>(-1.1f+1.0f, 1.1f, 0.0f));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f2",
                    openvdb::math::Vec3<float>(-1.1f,1.1f,0.0f), openvdb::math::Vec3<float>(-1.1f+1.0f, 1.1f+1.0f, 0.0f));

                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d1", openvdb::math::Vec3<double>(-1.1+1.0, 1.1, 0.0));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d2",
                    openvdb::math::Vec3<double>(-1.1,1.1,0.0), openvdb::math::Vec3<double>(-1.1+1.0, 1.1+1.0 ,0.0));


                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i1", openvdb::math::Vec4<int32_t>(0,1,0,0));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i2",
                    openvdb::math::Vec4<int32_t>(-1,1,0,0), openvdb::math::Vec4<int32_t>(0,2,0,0));

                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f1", openvdb::math::Vec4<float>(-1.1f+1.0f, 1.1f, 0.0f, 0.0f));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f2",
                    openvdb::math::Vec4<float>(-1.1f,1.1f,0.0f,0.0f), openvdb::math::Vec4<float>(-1.1f+1.0f, 1.1f+1.0f, 0.0f, 0.0f));

                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d1", openvdb::math::Vec4<double>(-1.1+1.0, 1.1, 0.0, 0.0));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d2",
                    openvdb::math::Vec4<double>(-1.1,1.1,0.0,0.0), openvdb::math::Vec4<double>(-1.1+1.0, 1.1+1.0, 0.0, 0.0));

                auto getmat = [](auto x, auto a, auto b) -> decltype(x) {
                    x = decltype(x)::zero();
                    x(0,0) = a;
                    x(0,1) = b;
                    return x;
                };

                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f1", getmat(openvdb::math::Mat3<float>(), -1.1f+1.0f, 1.1f));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f2",
                    getmat(openvdb::math::Mat3<float>(),-1.1f,1.1f),
                    getmat(openvdb::math::Mat3<float>(),-1.1f+1.0f,1.1f+1.0f));

                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d1", getmat(openvdb::math::Mat3<double>(), -1.1+1.0, 1.1));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d2",
                    getmat(openvdb::math::Mat3<double>(),-1.1,1.1),
                    getmat(openvdb::math::Mat3<double>(),-1.1+1.0, 1.1+1.0));


                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f1", getmat(openvdb::math::Mat4<float>(), -1.1f+1.0f, 1.1f));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f2",
                    getmat(openvdb::math::Mat4<float>(),-1.1f,1.1f),
                    getmat(openvdb::math::Mat4<float>(),-1.1f+1.0f,1.1f+1.0f));

                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d1", getmat(openvdb::math::Mat4<double>(), -1.1+1.0, 1.1));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d2",
                    getmat(openvdb::math::Mat4<double>(),-1.1,1.1),
                    getmat(openvdb::math::Mat4<double>(),-1.1+1.0, 1.1+1.0));
            }
        },
        { "dec",
            [&](){
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i1", openvdb::math::Vec2<int32_t>(-2,1));
                mHarness.addAttribute<openvdb::math::Vec2<int32_t>>("testvec2i2",
                    openvdb::math::Vec2<int32_t>(-1,1), openvdb::math::Vec2<int32_t>(-2,0));

                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f1", openvdb::math::Vec2<float>(-1.1f-1.0f, 1.1f));
                mHarness.addAttribute<openvdb::math::Vec2<float>>("testvec2f2",
                    openvdb::math::Vec2<float>(-1.1f,1.1f), openvdb::math::Vec2<float>(-1.1f-1.0f, 1.1f-1.0f));

                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d1", openvdb::math::Vec2<double>(-1.1-1.0, 1.1));
                mHarness.addAttribute<openvdb::math::Vec2<double>>("testvec2d2",
                    openvdb::math::Vec2<double>(-1.1,1.1), openvdb::math::Vec2<double>(-1.1-1.0, 1.1-1.0));


                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i1", openvdb::math::Vec3<int32_t>(-2,1,0));
                mHarness.addAttribute<openvdb::math::Vec3<int32_t>>("testvec3i2",
                    openvdb::math::Vec3<int32_t>(-1,1,0), openvdb::math::Vec3<int32_t>(-2,0,0));

                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f1", openvdb::math::Vec3<float>(-1.1f-1.0f, 1.1f, 0.0f));
                mHarness.addAttribute<openvdb::math::Vec3<float>>("testvec3f2",
                    openvdb::math::Vec3<float>(-1.1f,1.1f,0.0f), openvdb::math::Vec3<float>(-1.1f-1.0f, 1.1f-1.0f, 0.0f));

                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d1", openvdb::math::Vec3<double>(-1.1-1.0, 1.1, 0.0));
                mHarness.addAttribute<openvdb::math::Vec3<double>>("testvec3d2",
                    openvdb::math::Vec3<double>(-1.1,1.1,0.0), openvdb::math::Vec3<double>(-1.1-1.0, 1.1-1.0 ,0.0));


                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i1", openvdb::math::Vec4<int32_t>(-2,1,0,0));
                mHarness.addAttribute<openvdb::math::Vec4<int32_t>>("testvec4i2",
                    openvdb::math::Vec4<int32_t>(-1,1,0,0), openvdb::math::Vec4<int32_t>(-2,0,0,0));

                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f1", openvdb::math::Vec4<float>(-1.1f-1.0f, 1.1f, 0.0f, 0.0f));
                mHarness.addAttribute<openvdb::math::Vec4<float>>("testvec4f2",
                    openvdb::math::Vec4<float>(-1.1f,1.1f,0.0f,0.0f), openvdb::math::Vec4<float>(-1.1f-1.0f, 1.1f-1.0f, 0.0f, 0.0f));

                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d1", openvdb::math::Vec4<double>(-1.1-1.0, 1.1, 0.0, 0.0));
                mHarness.addAttribute<openvdb::math::Vec4<double>>("testvec4d2",
                    openvdb::math::Vec4<double>(-1.1,1.1,0.0,0.0), openvdb::math::Vec4<double>(-1.1-1.0, 1.1-1.0, 0.0, 0.0));


                auto getmat = [](auto x, auto a, auto b) -> decltype(x) {
                    x = decltype(x)::zero();
                    x(0,0) = a;
                    x(0,1) = b;
                    return x;
                };

                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f1", getmat(openvdb::math::Mat3<float>(), -1.1f-1.0f, 1.1f));
                mHarness.addAttribute<openvdb::math::Mat3<float>>("testmat3f2",
                    getmat(openvdb::math::Mat3<float>(),-1.1f,1.1f),
                    getmat(openvdb::math::Mat3<float>(),-1.1f-1.0f,1.1f-1.0f));

                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d1", getmat(openvdb::math::Mat3<double>(), -1.1-1.0, 1.1));
                mHarness.addAttribute<openvdb::math::Mat3<double>>("testmat3d2",
                    getmat(openvdb::math::Mat3<double>(),-1.1,1.1),
                    getmat(openvdb::math::Mat3<double>(),-1.1-1.0, 1.1-1.0));


                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f1", getmat(openvdb::math::Mat4<float>(), -1.1f-1.0f, 1.1f));
                mHarness.addAttribute<openvdb::math::Mat4<float>>("testmat4f2",
                    getmat(openvdb::math::Mat4<float>(),-1.1f,1.1f),
                    getmat(openvdb::math::Mat4<float>(),-1.1f-1.0f,1.1f-1.0f));

                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d1", getmat(openvdb::math::Mat4<double>(), -1.1-1.0, 1.1));
                mHarness.addAttribute<openvdb::math::Mat4<double>>("testmat4d2",
                    getmat(openvdb::math::Mat4<double>(),-1.1,1.1),
                    getmat(openvdb::math::Mat4<double>(),-1.1-1.0, 1.1-1.0));
            }
        }
    };

    for (const auto& expc : expected) {
        mHarness.reset();
        expc.second.operator()();
        this->execute("crement_" + expc.first + ".component.ax");
    }
}


// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
