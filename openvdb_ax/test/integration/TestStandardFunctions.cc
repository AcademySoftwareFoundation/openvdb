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

#include <iostream>
#include <cstdlib>
#include <cmath>

#include "TestHarness.h"

#include <openvdb_ax/test/util.h>
#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/math/OpenSimplexNoise.h>
#include <openvdb_ax/compiler/PointExecutable.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>

#include <openvdb/points/PointConversion.h>
#include <openvdb/util/CpuTimer.h>

#include <cppunit/extensions/HelperMacros.h>

#include <boost/functional/hash.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

using namespace openvdb::points;
using namespace openvdb::ax;

class TestStandardFunctions : public unittest_util::AXTestCase
{
public:
#ifdef PROFILE
    void setUp() override {
        // if PROFILE, generate more data for each test
        mHarness.reset(/*ppv*/8, openvdb::CoordBBox({0,0,0},{50,50,50}));
    }
#endif

    CPPUNIT_TEST_SUITE(TestStandardFunctions);
    CPPUNIT_TEST(abs);
    CPPUNIT_TEST(acos);
    CPPUNIT_TEST(asin);
    CPPUNIT_TEST(atan);
    CPPUNIT_TEST(atan2);
    CPPUNIT_TEST(atof);
    CPPUNIT_TEST(atoi);
    CPPUNIT_TEST(cbrt);
    CPPUNIT_TEST(clamp);
    CPPUNIT_TEST(cosh);
    CPPUNIT_TEST(cross);
    CPPUNIT_TEST(curlsimplexnoise);
    CPPUNIT_TEST(determinant);
    CPPUNIT_TEST(diag);
    CPPUNIT_TEST(dot);
    CPPUNIT_TEST(external);
    CPPUNIT_TEST(fit);
    CPPUNIT_TEST(hash);
    CPPUNIT_TEST(identity3);
    CPPUNIT_TEST(identity4);
    CPPUNIT_TEST(intrinsic);
    CPPUNIT_TEST(length);
    CPPUNIT_TEST(lengthsq);
    CPPUNIT_TEST(lerp);
    CPPUNIT_TEST(max);
    CPPUNIT_TEST(min);
    CPPUNIT_TEST(normalize);
    CPPUNIT_TEST(polardecompose);
    CPPUNIT_TEST(postscale);
    CPPUNIT_TEST(pow);
    CPPUNIT_TEST(prescale);
    CPPUNIT_TEST(pretransform);
    CPPUNIT_TEST(print);
    CPPUNIT_TEST(rand);
    CPPUNIT_TEST(signbit);
    CPPUNIT_TEST(simplexnoise);
    CPPUNIT_TEST(sinh);
    CPPUNIT_TEST(tan);
    CPPUNIT_TEST(tanh);
    CPPUNIT_TEST(trace);
    CPPUNIT_TEST(transform);
    CPPUNIT_TEST(transpose);
    CPPUNIT_TEST_SUITE_END();

    void abs();
    void acos();
    void asin();
    void atan();
    void atan2();
    void atof();
    void atoi();
    void cbrt();
    void clamp();
    void cosh();
    void cross();
    void curlsimplexnoise();
    void determinant();
    void diag();
    void dot();
    void external();
    void fit();
    void hash();
    void identity3();
    void identity4();
    void intrinsic();
    void length();
    void lengthsq();
    void lerp();
    void max();
    void min();
    void normalize();
    void polardecompose();
    void postscale();
    void pow();
    void prescale();
    void pretransform();
    void print();
    void rand();
    void signbit();
    void simplexnoise();
    void sinh();
    void tan();
    void tanh();
    void trace();
    void transform();
    void transpose();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestStandardFunctions);

inline void testFunctionOptions(unittest_util::AXTestHarness& harness,
                                const std::string& name,
                                CustomData::Ptr data = CustomData::create())
{
    const std::string file = "test/snippets/function/" + name;

#ifdef PROFILE
    struct Timer : public openvdb::util::CpuTimer {} timer;

    const std::string code = unittest_util::loadText(file);
    timer.start(std::string("\n") + name + std::string(": Parsing"));
    const ast::Tree::Ptr syntaxTree = ast::parse(code.c_str());
    timer.stop();

    // @warning  the first execution can take longer due to some llvm startup
    //           so if you're profiling a single function be aware of this.
    //           This also profiles execution AND compilation.

    auto profile = [&syntaxTree, &timer, &data]
        (const openvdb::ax::CompilerOptions& opts,
        std::vector<openvdb::points::PointDataGrid::Ptr>& points,
        openvdb::GridPtrVec& volumes,
        const bool doubleCompile = true)
    {
        if (!points.empty())
        {
            openvdb::ax::Compiler compiler(opts);
            if (doubleCompile) {
                compiler.compile<PointExecutable>(*syntaxTree, data);
            }
            {
                timer.start("    Points/Compilation  ");
                PointExecutable::Ptr executable =
                    compiler.compile<PointExecutable>(*syntaxTree, data);
                timer.stop();
                timer.start("    Points/Execution    ");
                executable->execute(*points.front());
                timer.stop();
            }
        }

        if (!volumes.empty())
        {
            openvdb::ax::Compiler compiler(opts);
            if (doubleCompile) {
                compiler.compile<VolumeExecutable>(*syntaxTree, data);
            }
            {
                timer.start("    Volumes/Compilation ");
                VolumeExecutable::Ptr executable =
                    compiler.compile<VolumeExecutable>(*syntaxTree, data);
                timer.stop();
                timer.start("    Volumes/Execution   ");
                executable->execute(volumes);
                timer.stop();
            }
        }
    };
#endif

    openvdb::ax::CompilerOptions opts;
    opts.mFunctionOptions.mConstantFoldCBindings = false;
    opts.mFunctionOptions.mPrioritiseIR = false;
#ifdef PROFILE
    std::cerr << "  C Bindings" << std::endl;
    profile(opts, harness.mInputPointGrids, harness.mInputVolumeGrids);
#else
    harness.mOpts = opts;
    harness.mCustomData = data;
    harness.executeCode(file);
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
#endif

    harness.resetInputsToZero();

    opts.mFunctionOptions.mConstantFoldCBindings = false;
    opts.mFunctionOptions.mPrioritiseIR = true;
#ifdef PROFILE
    std::cerr << "  IR Functions " << std::endl;
    profile(opts, harness.mInputPointGrids, harness.mInputVolumeGrids);
#else
    harness.mOpts = opts;
    harness.mCustomData = data;
    harness.executeCode(file);
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
#endif

    harness.resetInputsToZero();

    opts.mFunctionOptions.mConstantFoldCBindings = true;
    opts.mFunctionOptions.mPrioritiseIR = false;
#ifdef PROFILE
    std::cerr << "  C Folding   " << std::endl;
    profile(opts, harness.mInputPointGrids, harness.mInputVolumeGrids);
#else
    harness.mOpts = opts;
    harness.mCustomData = data;
    harness.executeCode(file);
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
#endif
}

void
TestStandardFunctions::abs()
{
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("test", 3), {
        std::abs(-3), std::abs(3), std::abs(0)
    });
    mHarness.addAttribute<int64_t>("test4", std::abs(-2147483649l));
    mHarness.addAttribute<float>("test5", std::abs(0.3f));
    mHarness.addAttribute<float>("test6", std::abs(-0.3f));
    mHarness.addAttribute<double>("test7", std::abs(1.79769e+308));
    mHarness.addAttribute<double>("test8", std::abs(-1.79769e+308));
    testFunctionOptions(mHarness, "abs");
}

void
TestStandardFunctions::acos()
{
    volatile double arg = 0.5;
    volatile float argf = 0.5f;
    mHarness.addAttribute<double>("test1", std::acos(arg));
    mHarness.addAttribute<float>("test2", std::acos(argf));
    testFunctionOptions(mHarness, "acos");
}

void
TestStandardFunctions::asin()
{
    mHarness.addAttribute<double>("test1", std::asin(-0.5));
    mHarness.addAttribute<float>("test2", std::asin(-0.5f));
    testFunctionOptions(mHarness, "asin");
}

void
TestStandardFunctions::atan()
{
    mHarness.addAttribute<double>("test1", std::atan(1.0));
    mHarness.addAttribute<float>("test2", std::atan(1.0f));
    testFunctionOptions(mHarness, "atan");
}

void
TestStandardFunctions::atan2()
{
    mHarness.addAttribute<double>("test1", std::atan2(1.0, 1.0));
    mHarness.addAttribute<float>("test2", std::atan2(1.0f, 1.0f));
    testFunctionOptions(mHarness, "atan2");
}

void
TestStandardFunctions::atoi()
{
    const std::vector<int32_t> values {
        std::atoi(""),
        std::atoi("-0"),
        std::atoi("+0"),
        std::atoi("-1"),
        std::atoi("1"),
        std::atoi("1s"),
        std::atoi("1s"),
        std::atoi(" 1"),
        std::atoi("1s1"),
        std::atoi("1 1"),
        std::atoi("11"),
        std::atoi("2147483647"), // int max
        std::atoi("-2147483648")
    };

    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("test", 13), values);
    testFunctionOptions(mHarness, "atoi");
}

void
TestStandardFunctions::atof()
{
    const std::vector<double> values {
        std::atof(""),
        std::atof("-0.0"),
        std::atof("+0.0"),
        std::atof("-1.1"),
        std::atof("1.5"),
        std::atof("1.s9"),
        std::atof("1s.9"),
        std::atof(" 1.6"),
        std::atof("1.5s1"),
        std::atof("1. 1.3"),
        std::atof("11.11"),
        std::atof("1.79769e+308"),
        std::atof("2.22507e-308")
    };

    mHarness.addAttributes<double>(unittest_util::nameSequence("test", 13), values);
    testFunctionOptions(mHarness, "atof");
}

void
TestStandardFunctions::cbrt()
{
    volatile double arg = 729.0;
    volatile float argf = 729.0f;
    mHarness.addAttribute<double>("test1", std::cbrt(arg));
    mHarness.addAttribute<float>("test2", std::cbrt(argf));
    testFunctionOptions(mHarness, "cbrt");
}

void
TestStandardFunctions::clamp()
{
    mHarness.addAttributes<double>(unittest_util::nameSequence("double_test", 3), {-1.5, 0.0, 1.5});
    testFunctionOptions(mHarness, "clamp");
}

void
TestStandardFunctions::cosh()
{
    volatile float arg = 1.0f;
    mHarness.addAttribute<double>("test1", std::cosh(1.0));
    mHarness.addAttribute<float>("test2",  std::cosh(arg));
    testFunctionOptions(mHarness, "cosh");
}

void
TestStandardFunctions::cross()
{
    const openvdb::Vec3d ad(1.0,2.2,3.4), bd(4.1,5.3,6.2);
    const openvdb::Vec3f af(1.0f,2.2f,3.4f), bf(4.1f,5.3f,6.2f);
    const openvdb::Vec3i ai(1,2,3), bi(4,5,6);
    mHarness.addAttribute<openvdb::Vec3d>("test1", ad.cross(bd));
    mHarness.addAttribute<openvdb::Vec3f>("test2", af.cross(bf));
    mHarness.addAttribute<openvdb::Vec3i>("test3", ai.cross(bi));
    testFunctionOptions(mHarness, "cross");
}

void
TestStandardFunctions::curlsimplexnoise()
{
    struct Local {
        static inline double noise(double x, double y, double z) {
            const OSN::OSNoise gen;
            const double result = gen.eval<double>(x, y, z);
            return (result + 1.0) * 0.5;
        }
    };

    double result[3];
    openvdb::ax::math::curlnoise<Local>(&result, 4.3, 5.7, -6.2);
    const openvdb::Vec3d expected(result[0], result[1], result[2]);

    mHarness.addAttributes<openvdb::Vec3d>
        (unittest_util::nameSequence("test", 2), {expected,expected});
    testFunctionOptions(mHarness, "curlsimplexnoise");
}

void
TestStandardFunctions::determinant()
{
    mHarness.addAttribute<float>("det3_float",  600.0f);
    mHarness.addAttribute<double>("det3_double", 600.0);
    mHarness.addAttribute<float>("det4_float",  24.0f);
    mHarness.addAttribute<double>("det4_double",  2400.0);
    testFunctionOptions(mHarness, "determinant");
}

void
TestStandardFunctions::diag()
{
    mHarness.addAttribute<openvdb::math::Mat3<double>>
        ("test1", openvdb::math::Mat3<double>(-1,0,0, 0,-2,0, 0,0,-3));
    mHarness.addAttribute<openvdb::math::Mat3<float>>
        ("test2", openvdb::math::Mat3<float>(-1,0,0, 0,-2,0, 0,0,-3));
    mHarness.addAttribute<openvdb::math::Mat4<double>>
        ("test3", openvdb::math::Mat4<double>(-1,0,0,0, 0,-2,0,0, 0,0,-3,0, 0,0,0,-4));
    mHarness.addAttribute<openvdb::math::Mat4<float>>
        ("test4", openvdb::math::Mat4<float>(-1,0,0,0, 0,-2,0,0, 0,0,-3,0, 0,0,0,-4));
    mHarness.addAttribute<openvdb::math::Vec3<double>>("test5", openvdb::math::Vec3<float>(-1,-5,-9));
    mHarness.addAttribute<openvdb::math::Vec3<float>>("test6", openvdb::math::Vec3<float>(-1,-5,-9));
    mHarness.addAttribute<openvdb::math::Vec4<double>>("test7", openvdb::math::Vec4<double>(-1,-6,-11,-16));
    mHarness.addAttribute<openvdb::math::Vec4<float>>("test8", openvdb::math::Vec4<float>(-1,-6,-11,-16));
    testFunctionOptions(mHarness, "diag");
}

void
TestStandardFunctions::dot()
{
    const openvdb::Vec3d ad(1.0,2.2,3.4), bd(4.1,5.3,6.2);
    const openvdb::Vec3f af(1.0f,2.2f,3.4f), bf(4.1f,5.3f,6.2f);
    const openvdb::Vec3i ai(1,2,3), bi(4,5,6);
    mHarness.addAttribute<double>("test1", ad.dot(bd));
    mHarness.addAttribute<float>("test2", af.dot(bf));
    mHarness.addAttribute<int32_t>("test3", ai.dot(bi));
    testFunctionOptions(mHarness, "dot");
}

void
TestStandardFunctions::external()
{
    mHarness.addAttribute<float>("foo", 2.0f);
    mHarness.addAttribute<openvdb::Vec3f>("v", openvdb::Vec3f(1.0f, 2.0f, 3.0f));

    using FloatMeta = openvdb::TypedMetadata<float>;
    using VectorFloatMeta = openvdb::TypedMetadata<openvdb::math::Vec3<float>>;

    FloatMeta customFloatData(2.0f);
    VectorFloatMeta customVecData(openvdb::math::Vec3<float>(1.0f, 2.0f, 3.0f));

    // test initialising the data before compile

    CustomData::Ptr data = CustomData::create();
    data->insertData("float1", customFloatData.copy());
    data->insertData("vector1", customVecData.copy());

    testFunctionOptions(mHarness, "external", data);

    mHarness.reset();
    mHarness.addAttribute<float>("foo", 2.0f);
    mHarness.addAttribute<openvdb::Vec3f>("v", openvdb::Vec3f(1.0f, 2.0f, 3.0f));

    // test post compilation

    data->reset();

    const std::string code = unittest_util::loadText("test/snippets/function/external");
    Compiler compiler;
    PointExecutable::Ptr pointExecutable = compiler.compile<PointExecutable>(code, data);
    VolumeExecutable::Ptr volumeExecutable = compiler.compile<VolumeExecutable>(code, data);

    data->insertData("float1", customFloatData.copy());

    VectorFloatMeta::Ptr customTypedVecData =
        openvdb::StaticPtrCast<VectorFloatMeta>(customVecData.copy());
    data->insertData<VectorFloatMeta>("vector1", customTypedVecData);

    for (auto& grid : mHarness.mInputPointGrids) {
        pointExecutable->execute(*grid);
    }

    volumeExecutable->execute(mHarness.mInputVolumeGrids);

    AXTESTS_STANDARD_ASSERT()
}

void
TestStandardFunctions::fit()
{
    std::vector<double> values{23.0, -23.0, -25.0, -15.0, -15.0, -18.0, -24.0, 0.0, 10.0,
        -5.0, 0.0, -1.0, 4.5, 4.5, 4.5, 4.5, 4.5};
    mHarness.addAttributes<double>(unittest_util::nameSequence("double_test", 17), values);
    testFunctionOptions(mHarness, "fit");
}

void
TestStandardFunctions::hash()
{
    const std::vector<int64_t> values{
        static_cast<int64_t>(std::hash<std::string>{}("")),
        static_cast<int64_t>(std::hash<std::string>{}("0")),
        static_cast<int64_t>(std::hash<std::string>{}("abc")),
        static_cast<int64_t>(std::hash<std::string>{}("123")),
    };
    mHarness.addAttributes<int64_t>(unittest_util::nameSequence("test", 4), values);
    testFunctionOptions(mHarness, "hash");
}

void
TestStandardFunctions::identity3()
{
    mHarness.addAttribute<openvdb::Mat3d>("test", openvdb::Mat3d::identity());
    testFunctionOptions(mHarness, "identity3");
}

void
TestStandardFunctions::identity4()
{
    mHarness.addAttribute<openvdb::Mat4d>("test", openvdb::Mat4d::identity());
    testFunctionOptions(mHarness, "identity4");
}

void
TestStandardFunctions::intrinsic()
{
    mHarness.addAttributes<double>(unittest_util::nameSequence("dtest", 12), {
        std::sqrt(9.0),
        std::cos(0.0),
        std::sin(0.0),
        std::log(1.0),
        std::log10(1.0),
        std::log2(2.0),
        std::exp(0.0),
        std::exp2(4.0),
        std::fabs(-10.321),
        std::floor(2194.213),
        std::ceil(2194.213),
        std::round(0.5)
    });

    mHarness.addAttributes<float>(unittest_util::nameSequence("ftest", 12), {
        std::sqrt(9.0f),
        std::cos(0.0f),
        std::sin(0.0f),
        std::log(1.0f),
        std::log10(1.0f),
        std::log2(2.0f),
        std::exp(0.0f),
        std::exp2(4.0f),
        std::fabs(-10.321f),
        std::floor(2194.213f),
        std::ceil(2194.213f),
        std::round(0.5f)
    });

    testFunctionOptions(mHarness, "intrinsic");
}

void
TestStandardFunctions::length()
{
    mHarness.addAttribute("test1", openvdb::Vec2d(2.2, 3.3).length());
    mHarness.addAttribute("test2", openvdb::Vec2f(2.2f, 3.3f).length());
    mHarness.addAttribute("test3", std::sqrt(double(openvdb::Vec2i(2, 3).lengthSqr())));

    mHarness.addAttribute("test4", openvdb::Vec3d(2.2, 3.3, 6.6).length());
    mHarness.addAttribute("test5", openvdb::Vec3f(2.2f, 3.3f, 6.6f).length());
    mHarness.addAttribute("test6", std::sqrt(double(openvdb::Vec3i(2, 3, 6).lengthSqr())));

    mHarness.addAttribute("test7", openvdb::Vec4d(2.2, 3.3, 6.6, 7.7).length());
    mHarness.addAttribute("test8", openvdb::Vec4f(2.2f, 3.3f, 6.6f, 7.7f).length());
    mHarness.addAttribute("test9", std::sqrt(double(openvdb::Vec4i(2, 3, 6, 7).lengthSqr())));
    testFunctionOptions(mHarness, "length");
}

void
TestStandardFunctions::lengthsq()
{
    mHarness.addAttribute("test1", openvdb::Vec2d(2.2, 3.3).lengthSqr());
    mHarness.addAttribute("test2", openvdb::Vec2f(2.2f, 3.3f).lengthSqr());
    mHarness.addAttribute("test3", openvdb::Vec2i(2, 3).lengthSqr());

    mHarness.addAttribute("test4", openvdb::Vec3d(2.2, 3.3, 6.6).lengthSqr());
    mHarness.addAttribute("test5", openvdb::Vec3f(2.2f, 3.3f, 6.6f).lengthSqr());
    mHarness.addAttribute("test6", openvdb::Vec3i(2, 3, 6).lengthSqr());

    mHarness.addAttribute("test7", openvdb::Vec4d(2.2, 3.3, 6.6, 7.7).lengthSqr());
    mHarness.addAttribute("test8", openvdb::Vec4f(2.2f, 3.3f, 6.6f, 7.7f).lengthSqr());
    mHarness.addAttribute("test9", openvdb::Vec4i(2, 3, 6, 7).lengthSqr());
    testFunctionOptions(mHarness, "lengthsq");
}

void
TestStandardFunctions::lerp()
{
    mHarness.addAttributes<double>(unittest_util::nameSequence("test", 3), {6.0, 21.0, -19.0});
    mHarness.addAttribute<float>("test4", 6.0f);
    testFunctionOptions(mHarness, "lerp");
}

void
TestStandardFunctions::max()
{
    mHarness.addAttribute("test1", std::max(-1.5, 1.5));
    mHarness.addAttribute("test2", std::max(-1.5f, 1.5f));
    mHarness.addAttribute("test3", std::max(-1, 1));
    testFunctionOptions(mHarness, "max");
}

void
TestStandardFunctions::min()
{
    mHarness.addAttribute("test1", std::min(-1.5, 1.5));
    mHarness.addAttribute("test2", std::min(-1.5f, 1.5f));
    mHarness.addAttribute("test3", std::min(-1, 1));
    testFunctionOptions(mHarness, "min");
}

void
TestStandardFunctions::normalize()
{
    openvdb::Vec3f expectedf(1.f, 2.f, 3.f);
    openvdb::Vec3d expectedd(1., 2., 3.);
    openvdb::Vec3d expectedi(1, 2, 3);
    expectedf.normalize();
    expectedd.normalize();
    expectedi.normalize();

    mHarness.addAttribute("test1", expectedf);
    mHarness.addAttribute("test2", expectedd);
    mHarness.addAttribute("test3", expectedi);
    testFunctionOptions(mHarness, "normalize");
}

void
TestStandardFunctions::polardecompose()
{
    // See snippet/polardecompose for details
    const openvdb::Mat3d composite(
        1.41421456236949,  0.0,  -5.09116882455613,
        0.0,               3.3,  0.0,
        -1.41421356237670, 0.0, -5.09116882453015);

    openvdb::Mat3d rot, symm;
    openvdb::math::polarDecomposition(composite, rot, symm);

    mHarness.addAttribute<openvdb::Mat3d>("rotation", rot);
    mHarness.addAttribute<openvdb::Mat3d>("symm", symm);
    testFunctionOptions(mHarness, "polardecompose");
}

void
TestStandardFunctions::postscale()
{

    mHarness.addAttributes<openvdb::math::Mat4<float>>
        ({"mat1", "mat3", "mat5"}, {
            openvdb::math::Mat4<float>(
                10.0f, 22.0f, 36.0f, 4.0f,
                50.0f, 66.0f, 84.0f, 8.0f,
                90.0f, 110.0f,132.0f,12.0f,
                130.0f,154.0f,180.0f,16.0f),
            openvdb::math::Mat4<float>(
                -1.0f, -4.0f, -9.0f, 4.0f,
                -5.0f, -12.0f,-21.0f, 8.0f,
                -9.0f, -20.0f,-33.0f,12.0f,
                -13.0f,-28.0f,-45.0f,16.0f),
            openvdb::math::Mat4<float>(
                0.0f, 100.0f, 200.0f, 100.0f,
                0.0f, 200.0f, 400.0f, 200.0f,
                0.0f, 300.0f, 600.0f, 300.0f,
                0.0f, 400.0f, 800.0f, 400.0f)
        });

    mHarness.addAttributes<openvdb::math::Mat4<double>>
        ({"mat2", "mat4", "mat6"}, {
            openvdb::math::Mat4<double>(
                10.0, 22.0, 36.0, 4.0,
                50.0, 66.0, 84.0, 8.0,
                90.0, 110.0,132.0,12.0,
                130.0,154.0,180.0,16.0),
            openvdb::math::Mat4<double>(
                -1.0, -4.0, -9.0, 4.0,
                -5.0, -12.0,-21.0, 8.0,
                -9.0, -20.0,-33.0,12.0,
                -13.0,-28.0,-45.0,16.0),
            openvdb::math::Mat4<double>(
                0.0, 100.0, 200.0, 100.0,
                0.0, 200.0, 400.0, 200.0,
                0.0, 300.0, 600.0, 300.0,
                0.0, 400.0, 800.0, 400.0)
        });

    testFunctionOptions(mHarness, "postscale");
}

void
TestStandardFunctions::pow()
{
    mHarness.addAttributes<float>(unittest_util::nameSequence("float_test", 5),{
        1.0f,
        static_cast<float>(std::pow(3.0, -2.1)),
        std::pow(4.7f, -4.3f),
        static_cast<float>(std::pow(4.7f, 3)),
        0.00032f
    });

    mHarness.addAttribute<int>("int_test1", static_cast<int>(std::pow(3, 5)));
    testFunctionOptions(mHarness, "pow");
}

void
TestStandardFunctions::prescale()
{

    mHarness.addAttributes<openvdb::math::Mat4<float>>
        ({"mat1", "mat3", "mat5"}, {
            openvdb::math::Mat4<float>(
                10.0f, 20.0f, 30.0f, 40.0f,
                55.0f, 66.0f, 77.0f, 88.0f,
                108.0f, 120.0f,132.0f,144.0f,
                13.0f,14.0f,15.0f,16.0f),
            openvdb::math::Mat4<float>(
                -1.0f,-2.0f,-3.0f,-4.0f,
                -10.0f,-12.0f,-14.0f,-16.0f,
                -27.0f,-30.0f,-33.0f,-36.0f,
                13.0f,14.0f,15.0f,16.0f),
            openvdb::math::Mat4<float>(
                0.0f, 0.0f, 0.0f, 0.0f,
                200.0f, 200.0f, 200.0f, 200.0f,
                600.0f, 600.0f, 600.0f, 600.0f,
                400.0f, 400.0f, 400.0f, 400.0f)
        });

    mHarness.addAttributes<openvdb::math::Mat4<double>>
        ({"mat2", "mat4", "mat6"}, {
            openvdb::math::Mat4<double>(
                10.0, 20.0, 30.0, 40.0,
                55.0, 66.0, 77.0, 88.0,
                108.0, 120.0,132.0,144.0,
                13.0,14.0,15.0,16.0),
            openvdb::math::Mat4<double>(
                -1.0,-2.0,-3.0,-4.0,
                -10.0,-12.0,-14.0,-16.0,
                -27.0,-30.0,-33.0,-36.0,
                13.0,14.0,15.0,16.0),
            openvdb::math::Mat4<double>(
                0.0, 0.0, 0.0, 0.0,
                200.0, 200.0, 200.0, 200.0,
                600.0, 600.0, 600.0, 600.0,
                400.0, 400.0, 400.0, 400.0)
        });

    testFunctionOptions(mHarness, "prescale");
}

void
TestStandardFunctions::pretransform()
{
    mHarness.addAttributes<openvdb::math::Vec3<double>>
        ({"test1", "test3", "test7"}, {
            openvdb::math::Vec3<double>(14.0, 32.0, 50.0),
            openvdb::math::Vec3<double>(18.0, 46.0, 74.0),
            openvdb::math::Vec3<double>(18.0, 46.0, 74.0),
        });

    mHarness.addAttribute<openvdb::math::Vec4<double>>("test5",
        openvdb::math::Vec4<double>(30.0, 70.0, 110.0, 150.0));

    mHarness.addAttributes<openvdb::math::Vec3<float>>
        ({"test2", "test4", "test8"}, {
            openvdb::math::Vec3<float>(14.0f, 32.0f, 50.0f),
            openvdb::math::Vec3<float>(18.0f, 46.0f, 74.0f),
            openvdb::math::Vec3<float>(18.0f, 46.0f, 74.0f),
        });

    mHarness.addAttribute<openvdb::math::Vec4<float>>("test6",
        openvdb::math::Vec4<float>(30.0f, 70.0f, 110.0f, 150.0f));

    testFunctionOptions(mHarness, "pretransform");
}

void
TestStandardFunctions::print()
{
    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform();
    const std::vector<openvdb::Vec3d> single = {
        openvdb::Vec3d::zero()
    };

    openvdb::points::PointDataGrid::Ptr grid =
        openvdb::points::createPointDataGrid
            <openvdb::points::NullCodec, openvdb::points::PointDataGrid>
                (single, *transform);

    const std::string code = unittest_util::loadText("test/snippets/function/print");

    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    openvdb::ax::PointExecutable::Ptr executable =
        compiler->compile<openvdb::ax::PointExecutable>(code);

    std::streambuf* sbuf = std::cout.rdbuf();

    try {
        // Redirect cout
        std::stringstream buffer;
        std::cout.rdbuf(buffer.rdbuf());

        executable->execute(*grid);
        const std::string& result = buffer.str();

        std::string expected = "a\n1\n2e-10\n";
        expected += openvdb::Vec4i(3,4,5,6).str() + "\n";
        expected += "bcd\n";

        CPPUNIT_ASSERT_EQUAL(expected, result);
    }
    catch (...) {
        std::cout.rdbuf(sbuf);
        throw;
    }

    std::cout.rdbuf(sbuf);
}

void
TestStandardFunctions::rand()
{
    auto hashToSeed = [](size_t hash) -> uint32_t {
        unsigned int seed = 0;
        do {
            seed ^= (uint32_t) hash;
        } while (hash >>= sizeof(uint32_t) * 8);
        return seed;
    };

    boost::uniform_01<double> uniform_01;
    size_t hash = boost::hash<double>()(2.0);
    boost::mt19937 engine(static_cast<boost::mt19937::result_type>(hashToSeed(hash)));

    const double expected1 = uniform_01(engine);

    hash = boost::hash<double>()(3.0);
    engine.seed(static_cast<boost::mt19937::result_type>(hashToSeed(hash)));
    const double expected2 = uniform_01(engine);
    const double expected3 = uniform_01(engine);

    mHarness.addAttributes<double>({"test0", "test1", "test2", "test3"},
        {expected1, expected1, expected2, expected3});
    testFunctionOptions(mHarness, "rand");
}

void
TestStandardFunctions::signbit()
{
    mHarness.addAttributes<bool>(unittest_util::nameSequence("test", 5), {true,false,true,false,false});
    testFunctionOptions(mHarness, "signbit");
}

void
TestStandardFunctions::simplexnoise()
{
    const OSN::OSNoise noiseGenerator;

    const double noise1 = noiseGenerator.eval<double>(1.0, 2.0, 3.0);
    const double noise2 = noiseGenerator.eval<double>(1.0, 2.0, 0.0);
    const double noise3 = noiseGenerator.eval<double>(1.0, 0.0, 0.0);
    const double noise4 = noiseGenerator.eval<double>(4.0, 14.0, 114.0);

    mHarness.addAttribute<double>("noise1", (noise1 + 1.0) * 0.5);
    mHarness.addAttribute<double>("noise2", (noise2 + 1.0) * 0.5);
    mHarness.addAttribute<double>("noise3", (noise3 + 1.0) * 0.5);
    mHarness.addAttribute<double>("noise4", (noise4 + 1.0) * 0.5);

    testFunctionOptions(mHarness, "simplexnoise");
}

void
TestStandardFunctions::sinh()
{
    mHarness.addAttribute<double>("test1", std::sinh(1.0));
    mHarness.addAttribute<float>("test2", std::sinh(1.0f));
    testFunctionOptions(mHarness, "sinh");
}

void
TestStandardFunctions::tan()
{
    mHarness.addAttribute<double>("test1", std::tan(1.0));
    mHarness.addAttribute<float>("test2", std::tan(1.0f));
    testFunctionOptions(mHarness, "tan");
}

void
TestStandardFunctions::tanh()
{
    mHarness.addAttribute<double>("test1", std::tanh(1.0));
    mHarness.addAttribute<float>("test2", std::tanh(1.0f));
    testFunctionOptions(mHarness, "tanh");
}

void
TestStandardFunctions::trace()
{
    mHarness.addAttribute<double>("test1", 6.0);
    mHarness.addAttribute<float>("test2", 6.0f);
    testFunctionOptions(mHarness, "trace");
}

void
TestStandardFunctions::transform()
{
    mHarness.addAttributes<openvdb::math::Vec3<double>>
        ({"test1", "test3", "test7"}, {
            openvdb::math::Vec3<double>(30.0, 36.0, 42.0),
            openvdb::math::Vec3<double>(51.0, 58, 65.0),
            openvdb::math::Vec3<double>(51.0, 58, 65.0),
        });

    mHarness.addAttribute<openvdb::math::Vec4<double>>("test5",
        openvdb::math::Vec4<double>(90.0, 100.0, 110.0, 120.0));

    mHarness.addAttributes<openvdb::math::Vec3<float>>
        ({"test2", "test4", "test8"}, {
            openvdb::math::Vec3<float>(30.0f, 36.0f, 42.0f),
            openvdb::math::Vec3<float>(51.0f, 58.0f, 65.0f),
            openvdb::math::Vec3<float>(51.0f, 58.0f, 65.0f),
        });

    mHarness.addAttribute<openvdb::math::Vec4<float>>("test6",
        openvdb::math::Vec4<float>(90.0f, 100.0f, 110.0f, 120.0f));

    testFunctionOptions(mHarness, "transform");
}

void
TestStandardFunctions::transpose()
{

    mHarness.addAttribute("test1",
        openvdb::math::Mat3<double>(
            1.0, 4.0, 7.0,
            2.0, 5.0, 8.0,
            3.0, 6.0, 9.0));
    mHarness.addAttribute("test2",
        openvdb::math::Mat3<float>(
            1.0f, 4.0f, 7.0f,
            2.0f, 5.0f, 8.0f,
            3.0f, 6.0f, 9.0f));
    mHarness.addAttribute("test3",
        openvdb::math::Mat4<double>(
            1.0, 5.0, 9.0,13.0,
            2.0, 6.0,10.0,14.0,
            3.0, 7.0,11.0,15.0,
            4.0, 8.0,12.0,16.0));
    mHarness.addAttribute("test4",
        openvdb::math::Mat4<float>(
            1.0f, 5.0f, 9.0f,13.0f,
            2.0f, 6.0f,10.0f,14.0f,
            3.0f, 7.0f,11.0f,15.0f,
            4.0f, 8.0f,12.0f,16.0f));

    testFunctionOptions(mHarness, "transpose");
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
