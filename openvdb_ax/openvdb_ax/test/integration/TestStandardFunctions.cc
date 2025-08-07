// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"

#include "../util.h"

#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/math/OpenSimplexNoise.h>
#include <openvdb_ax/compiler/PointExecutable.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>

#include <openvdb/points/PointConversion.h>
#include <openvdb/util/CpuTimer.h>

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <random>

using namespace openvdb::points;
using namespace openvdb::ax;

class TestStandardFunctions : public unittest_util::AXTestCase
{
public:
#ifdef PROFILE
    void SetUp() override {
        // if PROFILE, generate more data for each test
        mHarness.reset(/*ppv*/8, openvdb::CoordBBox({0,0,0},{50,50,50}));
    }
#endif
};


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

    ASSERT_TRUE(syntaxTree) << "Invalid AX passed to testFunctionOptions.";

    // @warning  the first execution can take longer due to some llvm startup
    //           so if you're profiling a single function be aware of this.
    //           This also profiles execution AND compilation.

    auto profile = [&syntaxTree, &timer, &data]
        (const openvdb::ax::CompilerOptions& opts,
        std::vector<openvdb::points::PointDataGrid::Ptr>& points,
        openvdb::GridPtrVec& sparsevols,
        openvdb::GridPtrVec& densevols,
        const bool doubleCompile = true) // warmup if true
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

        if (!sparsevols.empty())
        {
            openvdb::ax::Compiler compiler(opts);
            if (doubleCompile) {
                compiler.compile<VolumeExecutable>(*syntaxTree, data);
            }
            {
                timer.start("    Sparse Volumes/Compilation ");
                VolumeExecutable::Ptr executable =
                    compiler.compile<VolumeExecutable>(*syntaxTree, data);
                timer.stop();
                timer.start("    Sparse Volumes/Execution   ");
                executable->execute(sparsevols);
                timer.stop();
            }
        }

        if (!densevols.empty())
        {
            openvdb::ax::Compiler compiler(opts);
            if (doubleCompile) {
                compiler.compile<VolumeExecutable>(*syntaxTree, data);
            }
            {
                timer.start("    Dense Volumes/Compilation ");
                VolumeExecutable::Ptr executable =
                    compiler.compile<VolumeExecutable>(*syntaxTree, data);
                timer.stop();
                timer.start("    Dense Volumes/Execution   ");
                executable->execute(densevols);
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
    profile(opts, harness.mInputPointGrids, harness.mInputSparseVolumeGrids, harness.mInputDenseVolumeGrids);
#else
    harness.mOpts = opts;
    harness.mCustomData = data;
    bool success = harness.executeCode(file);
    ASSERT_TRUE(success) << ("error thrown during test: " + file + "\n" + harness.errors());
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
#endif

    harness.resetInputsToZero();

    opts.mFunctionOptions.mConstantFoldCBindings = false;
    opts.mFunctionOptions.mPrioritiseIR = true;
#ifdef PROFILE
    std::cerr << "  IR Functions " << std::endl;
    profile(opts, harness.mInputPointGrids, harness.mInputSparseVolumeGrids, harness.mInputDenseVolumeGrids);
#else
    harness.mOpts = opts;
    harness.mCustomData = data;
    success = harness.executeCode(file);
    ASSERT_TRUE(success) << ("error thrown during test: " + file + "\n" + harness.errors());
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
#endif

    harness.resetInputsToZero();

    opts.mFunctionOptions.mConstantFoldCBindings = true;
    opts.mFunctionOptions.mPrioritiseIR = false;
#ifdef PROFILE
    std::cerr << "  C Folding   " << std::endl;
    profile(opts, harness.mInputPointGrids, harness.mInputSparseVolumeGrids, harness.mInputDenseVolumeGrids);
#else
    harness.mOpts = opts;
    harness.mCustomData = data;
    success = harness.executeCode(file);
    ASSERT_TRUE(success) << ("error thrown during test: " + file + "\n" + harness.errors());
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
#endif
}

TEST_F(TestStandardFunctions, abs)
{
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("test", 3), {
        std::abs(-3), std::abs(3), std::abs(0)
    });
    mHarness.addAttribute<int64_t>("test4", std::llabs(-2147483649LL));
    mHarness.addAttribute<float>("test5", std::abs(0.3f));
    mHarness.addAttribute<float>("test6", std::abs(-0.3f));
    mHarness.addAttribute<double>("test7", std::abs(1.79769e+308));
    mHarness.addAttribute<double>("test8", std::abs(-1.79769e+308));
    testFunctionOptions(mHarness, "abs");
}

TEST_F(TestStandardFunctions, acos)
{
    volatile double arg = 0.5;
    volatile float argf = 0.5f;
    mHarness.addAttribute<double>("test1", std::acos(arg));
    mHarness.addAttribute<float>("test2", std::acos(argf));
    testFunctionOptions(mHarness, "acos");
}

TEST_F(TestStandardFunctions, adjoint)
{
    const openvdb::math::Mat3<double> inputd(
            1.0, -1.0, 0.0,
            2.0, 2.0, 0.0,
            0.0, 0.0, -1.0);

    openvdb::math::Mat3<double> add = inputd.adjoint();

    const openvdb::math::Mat3<float> inputf(
            1.0f, -1.0f, 0.0f,
            2.0f, 2.0f, 0.0f,
            0.0f, 0.0f, -1.0f);

    openvdb::math::Mat3<float> adf = inputf.adjoint();

    mHarness.addAttribute<openvdb::math::Mat3<double>>("test1", add);
    mHarness.addAttribute<openvdb::math::Mat3<float>>("test2", adf);
    testFunctionOptions(mHarness, "adjoint");
}

TEST_F(TestStandardFunctions, argsort)
{
    // const openvdb::Vec3d input3d(1.0, -1.0, 0.0);
    // const openvdb::Vec3f input3f(1.0f, -1.0f, 0.0f);
    // const openvdb::Vec3i input3i(1, -1, 0);

    // const openvdb::Vec4d input4d(1.0, -1.0, 0.0, 5.0);
    // const openvdb::Vec4f input4f(1.0f, -1.0f, 0.0f, 5.0f);
    // const openvdb::Vec4i input4i(1, -1, 0, 5);

    const openvdb::Vec3i arg3d(1,2,0);
    const openvdb::Vec3i arg3f(1,2,0);
    const openvdb::Vec3i arg3i(1,2,0);

    const openvdb::Vec4i arg4d(1,2,0,3);
    const openvdb::Vec4i arg4f(1,2,0,3);
    const openvdb::Vec4i arg4i(1,2,0,3);

    mHarness.addAttribute<openvdb::Vec3i>("test1", arg3d);
    mHarness.addAttribute<openvdb::Vec3i>("test2", arg3f);
    mHarness.addAttribute<openvdb::Vec3i>("test3", arg3i);
    mHarness.addAttribute<openvdb::Vec4i>("test4", arg4d);
    mHarness.addAttribute<openvdb::Vec4i>("test5", arg4f);
    mHarness.addAttribute<openvdb::Vec4i>("test6", arg4i);

    testFunctionOptions(mHarness, "argsort");
}


TEST_F(TestStandardFunctions, asin)
{
    mHarness.addAttribute<double>("test1", std::asin(-0.5));
    mHarness.addAttribute<float>("test2", std::asin(-0.5f));
    testFunctionOptions(mHarness, "asin");
}

TEST_F(TestStandardFunctions, atan)
{
    mHarness.addAttribute<double>("test1", std::atan(1.0));
    mHarness.addAttribute<float>("test2", std::atan(1.0f));
    testFunctionOptions(mHarness, "atan");
}

TEST_F(TestStandardFunctions, atan2)
{
    mHarness.addAttribute<double>("test1", std::atan2(1.0, 1.0));
    mHarness.addAttribute<float>("test2", std::atan2(1.0f, 1.0f));
    testFunctionOptions(mHarness, "atan2");
}

TEST_F(TestStandardFunctions, atoi)
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

TEST_F(TestStandardFunctions, atof)
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

TEST_F(TestStandardFunctions, cbrt)
{
    volatile double arg = 729.0;
    volatile float argf = 729.0f;
    mHarness.addAttribute<double>("test1", std::cbrt(arg));
    mHarness.addAttribute<float>("test2", std::cbrt(argf));
    testFunctionOptions(mHarness, "cbrt");
}

TEST_F(TestStandardFunctions, clamp)
{
    mHarness.addAttributes<double>(unittest_util::nameSequence("double_test", 3), {-1.5, 0.0, 1.5});
    testFunctionOptions(mHarness, "clamp");
}

TEST_F(TestStandardFunctions, cofactor)
{
    const openvdb::math::Mat3<double> inputd(
            1.0, -1.0, 0.0,
            2.0, 2.0, 0.0,
            0.0, 0.0, -1.0);

    openvdb::math::Mat3<double> cd = inputd.cofactor();

    const openvdb::math::Mat3<float> inputf(
            1.0f, -1.0f, 0.0f,
            2.0f, 2.0f, 0.0f,
            0.0f, 0.0f, -1.0f);

    openvdb::math::Mat3<float> cf = inputf.cofactor();

    mHarness.addAttribute<openvdb::math::Mat3<double>>("test1", cd);
    mHarness.addAttribute<openvdb::math::Mat3<float>>("test2", cf);
    testFunctionOptions(mHarness, "cofactor");
}

TEST_F(TestStandardFunctions, cosh)
{
    volatile float arg = 1.0f;
    mHarness.addAttribute<double>("test1", std::cosh(1.0));
    mHarness.addAttribute<float>("test2",  std::cosh(arg));
    testFunctionOptions(mHarness, "cosh");
}

TEST_F(TestStandardFunctions, cross)
{
    const openvdb::Vec3d ad(1.0,2.2,3.4), bd(4.1,5.3,6.2);
    const openvdb::Vec3f af(1.0f,2.2f,3.4f), bf(4.1f,5.3f,6.2f);
    const openvdb::Vec3i ai(1,2,3), bi(4,5,6);
    mHarness.addAttribute<openvdb::Vec3d>("test1", ad.cross(bd));
    mHarness.addAttribute<openvdb::Vec3f>("test2", af.cross(bf));
    mHarness.addAttribute<openvdb::Vec3i>("test3", ai.cross(bi));
    testFunctionOptions(mHarness, "cross");
}

TEST_F(TestStandardFunctions, curlsimplexnoise)
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

TEST_F(TestStandardFunctions, degrees)
{
    mHarness.addAttribute<double>("test1", 1.5708 * (180.0 / openvdb::math::pi<double>()));
    mHarness.addAttribute<float>("test2", -1.1344f * (180.0f / openvdb::math::pi<float>()));
    testFunctionOptions(mHarness, "degrees");
}

TEST_F(TestStandardFunctions, determinant)
{
    mHarness.addAttribute<float>("det3_float",  600.0f);
    mHarness.addAttribute<double>("det3_double", 600.0);
    mHarness.addAttribute<float>("det4_float",  24.0f);
    mHarness.addAttribute<double>("det4_double",  2400.0);
    testFunctionOptions(mHarness, "determinant");
}

TEST_F(TestStandardFunctions, diag)
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

TEST_F(TestStandardFunctions, dot)
{
    const openvdb::Vec3d ad(1.0,2.2,3.4), bd(4.1,5.3,6.2);
    const openvdb::Vec3f af(1.0f,2.2f,3.4f), bf(4.1f,5.3f,6.2f);
    const openvdb::Vec3i ai(1,2,3), bi(4,5,6);
    mHarness.addAttribute<double>("test1", ad.dot(bd));
    mHarness.addAttribute<float>("test2", af.dot(bf));
    mHarness.addAttribute<int32_t>("test3", ai.dot(bi));
    testFunctionOptions(mHarness, "dot");
}

TEST_F(TestStandardFunctions, euclideanmod)
{
    static auto emod = [](auto D, auto d) -> auto {
        using ValueType = decltype(D);
        return ValueType(D - d * (d < 0 ? std::ceil(D/double(d)) : std::floor(D/double(d))));
    };

    // @note these also test that these match % op
    const std::vector<int32_t> ivalues{ emod(7, 5), emod(-7, 5), emod(7,-5), emod(-7,-5) };
    const std::vector<float> fvalues{ emod(7.2f, 5.7f), emod(-7.2f, 5.7f), emod(7.2f, -5.7f), emod(-7.2f, -5.7f) };
    const std::vector<double> dvalues{ emod(7.2, 5.7), emod(-7.2, 5.7), emod(7.2, -5.7), emod(-7.2, -5.7) };
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("itest", 4), ivalues);
    mHarness.addAttributes<float>(unittest_util::nameSequence("ftest", 4), fvalues);
    mHarness.addAttributes<double>(unittest_util::nameSequence("dtest", 4), dvalues);
    testFunctionOptions(mHarness, "euclideanmod");
}

TEST_F(TestStandardFunctions, external)
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

    volumeExecutable->execute(mHarness.mInputSparseVolumeGrids);
    volumeExecutable->execute(mHarness.mInputDenseVolumeGrids);

    AXTESTS_STANDARD_ASSERT()
}

TEST_F(TestStandardFunctions, fit)
{
    std::vector<double> values{23.0, -23.0, -25.0, -15.0, -15.0, -18.0, -24.0, 0.0, 10.0,
        -5.0, 0.0, -1.0, 4.5, 4.5, 4.5, 4.5, 4.5};
    mHarness.addAttributes<double>(unittest_util::nameSequence("double_test", 17), values);
    testFunctionOptions(mHarness, "fit");
}

TEST_F(TestStandardFunctions, floormod)
{
    auto axmod = [](auto D, auto d) -> auto {
        auto r = std::fmod(D, d);
        if ((r > 0 && d < 0) || (r < 0 && d > 0)) r = r+d;
        return r;
    };

    // @note these also test that these match % op
    const std::vector<int32_t> ivalues{ 2,2, 3,3, -3,-3, -2,-2 };
    const std::vector<float> fvalues{ axmod(7.2f,5.7f),axmod(7.2f,5.7f),
        axmod(-7.2f,5.7f),axmod(-7.2f,5.7f),
        axmod(7.2f,-5.7f),axmod(7.2f,-5.7f),
        axmod(-7.2f,-5.7f),axmod(-7.2f,-5.7f)
    };
    const std::vector<double> dvalues{ axmod(7.2,5.7),axmod(7.2,5.7),
        axmod(-7.2,5.7),axmod(-7.2,5.7),
        axmod(7.2,-5.7),axmod(7.2,-5.7),
        axmod(-7.2,-5.7),axmod(-7.2,-5.7)
    };
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("itest", 8), ivalues);
    mHarness.addAttributes<float>(unittest_util::nameSequence("ftest", 8), fvalues);
    mHarness.addAttributes<double>(unittest_util::nameSequence("dtest", 8), dvalues);
    testFunctionOptions(mHarness, "floormod");
}

TEST_F(TestStandardFunctions, hash)
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

TEST_F(TestStandardFunctions, hsvtorgb)
{
    auto axmod = [](auto D, auto d) -> auto {
        auto r = std::fmod(D, d);
        if ((r > 0 && d < 0) || (r < 0 && d > 0)) r = r+d;
        return r;
    };

    // HSV to RGB conversion. Taken from OpenEXR's ImathColorAlgo
    // @note  AX adds flooredmod of input hue to wrap to [0,1] domain
    // @note  AX also clamp saturation to [0,1]
    auto convert = [&](const openvdb::Vec3d& hsv) {
        double hue = hsv.x();
        double sat = hsv.y();
        double val = hsv.z();
        openvdb::Vec3d rgb(0.0);

        // additions
        hue = axmod(hue, 1.0);
        sat = std::max(0.0, sat);
        sat = std::min(1.0, sat);
        //

        if (hue == 1) hue = 0;
        else          hue *= 6;

        int i = int(std::floor(hue));
        double f = hue - i;
        double p = val * (1 - sat);
        double q = val * (1 - (sat * f));
        double t = val * (1 - (sat * (1 - f)));

        switch (i) {
            case 0:
                rgb[0] = val; rgb[1] = t; rgb[2] = p;
                break;
            case 1:
                rgb[0] = q; rgb[1] = val; rgb[2] = p;
                break;
            case 2:
                rgb[0] = p; rgb[1] = val; rgb[2] = t;
                break;
            case 3:
                rgb[0] = p; rgb[1] = q; rgb[2] = val;
                break;
            case 4:
                rgb[0] = t; rgb[1] = p; rgb[2] = val;
                break;
            case 5:
                rgb[0] = val; rgb[1] = p; rgb[2] = q;
                break;
        }

        return rgb;
    };

    const std::vector<openvdb::Vec3d> values{
        convert({0,0,0}),
        convert({1,1,1}),
        convert({5.8,1,1}),
        convert({-0.1,-0.5,10}),
        convert({-5.1,10.5,-5}),
        convert({-7,-11.5,5}),
        convert({0.5,0.5,0.5}),
        convert({0.3,1.0,10.0})
    };
    mHarness.addAttributes<openvdb::Vec3d>(unittest_util::nameSequence("test", 8), values);
    testFunctionOptions(mHarness, "hsvtorgb");
}

TEST_F(TestStandardFunctions, identity3)
{
    mHarness.addAttribute<openvdb::Mat3d>("test", openvdb::Mat3d::identity());
    testFunctionOptions(mHarness, "identity3");
}

TEST_F(TestStandardFunctions, identity4)
{
    mHarness.addAttribute<openvdb::Mat4d>("test", openvdb::Mat4d::identity());
    testFunctionOptions(mHarness, "identity4");
}

TEST_F(TestStandardFunctions, intrinsic)
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

TEST_F(TestStandardFunctions, inverse)
{
    const openvdb::math::Mat3<double> inputd(
            1.0, -1.0, 0.0,
            2.0, 2.0, 0.0,
            0.0, 0.0, -1.0);

    const openvdb::math::Mat3<double> singulard(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0);

    const openvdb::math::Mat3<float> inputf(
            1.0f, -1.0f, 0.0f,
            2.0f, 2.0f, 0.0f,
            0.0f, 0.0f, -1.0f);

    const openvdb::math::Mat3<float> singularf(
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f);

    openvdb::math::Mat3<double> invd = inputd.inverse();
    openvdb::math::Mat3<float> invf = inputf.inverse();

    mHarness.addAttribute<openvdb::math::Mat3<double>>("test1", invd);
    mHarness.addAttribute<openvdb::math::Mat3<float>>("test2", invf);

    // inverse(singular) returns the original matrix
    mHarness.addAttribute<openvdb::math::Mat3<double>>("test3", singulard);
    mHarness.addAttribute<openvdb::math::Mat3<float>>("test4", singularf);

    testFunctionOptions(mHarness, "inverse");
}

TEST_F(TestStandardFunctions, isfinite)
{
    mHarness.addAttributes<bool>(
        {"test1","test2","test3","test4","test5","test6","test7","test8","test9","test10", "test11","test12",
        "test13","test14","test15","test16","test17","test18","test19","test20", "test21","test22", "test23","test24"},
        {true, true, true, true, true, true, true, true, true, true, true, true,
        false, false, false, false, false, false, false, false, false, false, false, false});

    testFunctionOptions(mHarness, "isfinite");
}

TEST_F(TestStandardFunctions, isinf)
{
    mHarness.addAttributes<bool>(
        {"test1","test2","test3","test4","test5","test6","test7","test8","test9","test10", "test11","test12",
        "test13","test14","test15","test16","test17","test18","test19","test20", "test21","test22", "test23","test24"},
        {false, false, false, false, false, false, false, false, false, false, false, false,
         true, true, true, true, true, true, true, true, true, true, true, true});

    testFunctionOptions(mHarness, "isinf");
}

TEST_F(TestStandardFunctions, isnan)
{
    mHarness.addAttributes<bool>(
        {"test1","test2","test3","test4","test5","test6","test7","test8","test9","test10", "test11","test12",
        "test13","test14","test15","test16","test17","test18","test19","test20", "test21","test22", "test23","test24"},
        {false, false, false, false, false, false, false, false, false, false, false, false,
         true, true, true, true, true, true, true, true, true, true, true, true});

    testFunctionOptions(mHarness, "isnan");
}


TEST_F(TestStandardFunctions, length)
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

TEST_F(TestStandardFunctions, lengthsq)
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

TEST_F(TestStandardFunctions, lerp)
{
    mHarness.addAttributes<double>(unittest_util::nameSequence("test", 9),
        {-1.1, 1.0000001, 1.0000001, -1.0000001, 1.1, -1.1, 6.0, 21.0, -19.0});
    mHarness.addAttribute<float>("test10", 6.0f);
    testFunctionOptions(mHarness, "lerp");
}

TEST_F(TestStandardFunctions, max)
{
    mHarness.addAttribute("test1", std::max(-1.5, 1.5));
    mHarness.addAttribute("test2", std::max(-1.5f, 1.5f));
    mHarness.addAttribute("test3", std::max(-1, 1));
    testFunctionOptions(mHarness, "max");
}

TEST_F(TestStandardFunctions, min)
{
    mHarness.addAttribute("test1", std::min(-1.5, 1.5));
    mHarness.addAttribute("test2", std::min(-1.5f, 1.5f));
    mHarness.addAttribute("test3", std::min(-1, 1));
    testFunctionOptions(mHarness, "min");
}

TEST_F(TestStandardFunctions, normalize)
{
    openvdb::Vec3f expected3f(1.f, 2.f, 3.f);
    openvdb::Vec3d expected3d(1., 2., 3.);
    openvdb::Vec3d expected3i(1, 2, 3);

    openvdb::Vec4f expected4f(1.f, 2.f, 3.f, 4.f);
    openvdb::Vec4d expected4d(1., 2., 3., 4.);
    openvdb::Vec4d expected4i(1, 2, 3, 4);

    expected3f.normalize();
    expected3d.normalize();
    expected3i.normalize();
    expected4f.normalize();
    expected4d.normalize();
    expected4i.normalize();

    mHarness.addAttribute("test1", expected3f);
    mHarness.addAttribute("test2", expected3d);
    mHarness.addAttribute("test3", expected3i);
    mHarness.addAttribute("test4", expected4f);
    mHarness.addAttribute("test5", expected4d);
    mHarness.addAttribute("test6", expected4i);
    testFunctionOptions(mHarness, "normalize");
}

TEST_F(TestStandardFunctions, polardecompose)
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

TEST_F(TestStandardFunctions, postscale)
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

TEST_F(TestStandardFunctions, pow)
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

TEST_F(TestStandardFunctions, prescale)
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

TEST_F(TestStandardFunctions, pretransform)
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

TEST_F(TestStandardFunctions, print)
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

        ASSERT_EQ(expected, result);
    }
    catch (...) {
        std::cout.rdbuf(sbuf);
        throw;
    }

    std::cout.rdbuf(sbuf);
}

TEST_F(TestStandardFunctions, radians)
{
    mHarness.addAttribute<double>("test1", 90.0 * (openvdb::math::pi<double>() / 180.0));
    mHarness.addAttribute<float>("test2", -65.0f * (openvdb::math::pi<float>() / 180.0f ));
    testFunctionOptions(mHarness, "radians");
}

TEST_F(TestStandardFunctions, rand)
{
    std::mt19937_64 engine;
    std::uniform_real_distribution<double> uniform(0.0,1.0);

    size_t hash = std::hash<double>()(2.0);
    engine.seed(hash);

    const double expected1 = uniform(engine);

    hash = std::hash<double>()(3.0);
    engine.seed(hash);

    const double expected2 = uniform(engine);
    const double expected3 = uniform(engine);

    mHarness.addAttributes<double>({"test0", "test1", "test2", "test3"},
        {expected1, expected1, expected2, expected3});
    testFunctionOptions(mHarness, "rand");
}

TEST_F(TestStandardFunctions, rand32)
{
    auto hashToSeed = [](uint64_t hash) ->
        std::mt19937::result_type
    {
        unsigned int seed = 0;
        do {
            seed ^= (uint32_t) hash;
        } while (hash >>= sizeof(uint32_t) * 8);
        return std::mt19937::result_type(seed);
    };

    std::mt19937 engine;
    std::uniform_real_distribution<double> uniform(0.0,1.0);

    size_t hash = std::hash<double>()(2.0);
    engine.seed(hashToSeed(hash));

    const double expected1 = uniform(engine);

    hash = std::hash<double>()(3.0);
    engine.seed(hashToSeed(hash));

    const double expected2 = uniform(engine);
    const double expected3 = uniform(engine);

    mHarness.addAttributes<double>({"test0", "test1", "test2", "test3"},
        {expected1, expected1, expected2, expected3});
    testFunctionOptions(mHarness, "rand32");
}

TEST_F(TestStandardFunctions, rgbtohsv)
{
    // RGB to HSV conversion. Taken from OpenEXR's ImathColorAlgo
    auto convert = [](const openvdb::Vec3d& rgb) {
        const double& x = rgb.x();
        const double& y = rgb.y();
        const double& z = rgb.z();

        double max   = (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
        double min   = (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
        double range = max - min;
        double val   = max;
        double sat   = 0;
        double hue   = 0;

        if (max != 0) sat = range / max;
        if (sat != 0)
        {
            double h;
            if (x == max)       h = (y - z) / range;
            else if (y == max)  h = 2 + (z - x) / range;
            else                h = 4 + (x - y) / range;
            hue = h / 6.;
            if (hue < 0.) hue += 1.0;
        }

        return openvdb::Vec3d(hue, sat, val);
    };

    const std::vector<openvdb::Vec3d> values{
        convert({0,0,0}),
        convert({1,1,1}),
        convert({20.5,40.3,100.1}),
        convert({-10,1.3,0.25})
    };
    mHarness.addAttributes<openvdb::Vec3d>(unittest_util::nameSequence("test", 4), values);
    testFunctionOptions(mHarness, "rgbtohsv");
}

TEST_F(TestStandardFunctions, sign)
{
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("test", 13),
        { 0,0,0,0,0,0,0, -1,-1,-1, 1,1,1 });
    testFunctionOptions(mHarness, "sign");
}

TEST_F(TestStandardFunctions, signbit)
{
    mHarness.addAttributes<bool>(unittest_util::nameSequence("test", 5), {true,false,true,false,false});
    testFunctionOptions(mHarness, "signbit");
}

TEST_F(TestStandardFunctions, simplexnoise)
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

TEST_F(TestStandardFunctions, sinh)
{
    mHarness.addAttribute<double>("test1", std::sinh(1.0));
    mHarness.addAttribute<float>("test2", std::sinh(1.0f));
    testFunctionOptions(mHarness, "sinh");
}

TEST_F(TestStandardFunctions, sort)
{
    // const openvdb::Vec3d input3d(1.0, -1.0, 0.0);
    // const openvdb::Vec3f input3f(1.0f, -1.0f, 0.0f);
    // const openvdb::Vec3i input3i(1, -1, 0);

    // const openvdb::Vec4d input4d(1.0, -1.0, 0.0, 5.0);
    // const openvdb::Vec4f input4f(1.0f, -1.0f, 0.0f, 5.0f);
    // const openvdb::Vec4i input4i(1, -1, 0, 5);

    const openvdb::Vec3d sorted3d(-1.0,0.0,1.0);
    const openvdb::Vec3f sorted3f(-1.0f,0.0f,1.0f);
    const openvdb::Vec3i sorted3i(-1,0,1);

    const openvdb::Vec4d sorted4d(-1.0,0.0,1.0,5.0);
    const openvdb::Vec4f sorted4f(-1.0f,0.0f,1.0f,5.0f);
    const openvdb::Vec4i sorted4i(-1,0,1,5);

    mHarness.addAttribute<openvdb::Vec3d>("test1", sorted3d);
    mHarness.addAttribute<openvdb::Vec3f>("test2", sorted3f);
    mHarness.addAttribute<openvdb::Vec3i>("test3", sorted3i);
    mHarness.addAttribute<openvdb::Vec4d>("test4", sorted4d);
    mHarness.addAttribute<openvdb::Vec4f>("test5", sorted4f);
    mHarness.addAttribute<openvdb::Vec4i>("test6", sorted4i);

    testFunctionOptions(mHarness, "sort");
}

TEST_F(TestStandardFunctions, tan)
{
    mHarness.addAttribute<double>("test1", std::tan(1.0));
    mHarness.addAttribute<float>("test2", std::tan(1.0f));
    testFunctionOptions(mHarness, "tan");
}

TEST_F(TestStandardFunctions, tanh)
{
    mHarness.addAttribute<double>("test1", std::tanh(1.0));
    mHarness.addAttribute<float>("test2", std::tanh(1.0f));
    testFunctionOptions(mHarness, "tanh");
}

TEST_F(TestStandardFunctions, trace)
{
    mHarness.addAttribute<double>("test1", 6.0);
    mHarness.addAttribute<float>("test2", 6.0f);
    testFunctionOptions(mHarness, "trace");
}

TEST_F(TestStandardFunctions, truncatemod)
{
    // @note these also test that these match % op
    const std::vector<int32_t> ivalues{ 2,-2,2,-2, };
    const std::vector<float> fvalues{ std::fmod(7.2f, 5.7f), std::fmod(-7.2f, 5.7f), std::fmod(7.2f, -5.7f), std::fmod(-7.2f, -5.7f) };
    const std::vector<double> dvalues{ std::fmod(7.2, 5.7), std::fmod(-7.2, 5.7), std::fmod(7.2, -5.7), std::fmod(-7.2, -5.7) };
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("itest", 4), ivalues);
    mHarness.addAttributes<float>(unittest_util::nameSequence("ftest", 4), fvalues);
    mHarness.addAttributes<double>(unittest_util::nameSequence("dtest", 4), dvalues);
    testFunctionOptions(mHarness, "truncatemod");
}

TEST_F(TestStandardFunctions, transform)
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

TEST_F(TestStandardFunctions, transpose)
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

