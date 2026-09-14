// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/version.h>
 // @note  We leave it to our CI to test VCL On/Off - but if you have a VCL
//  build, you can simply uncomment these to test the tuple interface.
// #undef OPENVDB_USE_VCL
// #undef INSTRSET
#include <openvdb/Types.h>
#include <openvdb/simd/Simd.h>

#include <gtest/gtest.h>
#include <unordered_set>

using namespace openvdb;

template <typename S, typename T, size_t E>
struct SimdConfig
{
    using SimdT = S;
    using ExpectedElementT = T;
    static constexpr size_t ExpectedSize = E;
};

// List of integer and float types to tests. broad masks are tested implicitly
// via each arithmetic type's test
using TestTypes = ::testing::Types<
    // int8
    SimdConfig<simd::Vec16c, int8_t, 16>,
    SimdConfig<simd::Vec32c, int8_t, 32>,
    SimdConfig<simd::Vec64c, int8_t, 64>,
    // int16
    SimdConfig<simd::Vec8s, int16_t, 8>,
    SimdConfig<simd::Vec16s, int16_t, 16>,
    SimdConfig<simd::Vec32s, int16_t, 32>,
    // int32
    SimdConfig<simd::Vec4i, int32_t, 4>,
    SimdConfig<simd::Vec8i, int32_t, 8>,
    SimdConfig<simd::Vec16i, int32_t, 16>,
    // int64
    SimdConfig<simd::Vec2q, int64_t, 2>,
    SimdConfig<simd::Vec4q, int64_t, 4>,
    SimdConfig<simd::Vec8q, int64_t, 8>,
    // uint8
    SimdConfig<simd::Vec16uc, uint8_t, 16>,
    SimdConfig<simd::Vec32uc, uint8_t, 32>,
    SimdConfig<simd::Vec64uc, uint8_t, 64>,
    // uint16
    SimdConfig<simd::Vec8us, uint16_t, 8>,
    SimdConfig<simd::Vec16us, uint16_t, 16>,
    SimdConfig<simd::Vec32us, uint16_t, 32>,
    // uint32
    SimdConfig<simd::Vec4ui, uint32_t, 4>,
    SimdConfig<simd::Vec8ui, uint32_t, 8>,
    SimdConfig<simd::Vec16ui, uint32_t, 16>,
    // uint64
    SimdConfig<simd::Vec2uq, uint64_t, 2>,
    SimdConfig<simd::Vec4uq, uint64_t, 4>,
    SimdConfig<simd::Vec8uq, uint64_t, 8>,
    /*
    SimdConfig<simd::Vec8h, math::half, 8>,
    SimdConfig<simd::Vec16h, math::half, 16>,
    SimdConfig<simd::Vec32h, math::half, 32>,
    */
    // float
    SimdConfig<simd::Vec4f, float, 4>,
    SimdConfig<simd::Vec8f, float, 8>,
    SimdConfig<simd::Vec16f, float, 16>,
    // double
    SimdConfig<simd::Vec2d, double, 2>,
    SimdConfig<simd::Vec4d, double, 4>,
    SimdConfig<simd::Vec8d, double, 8>
>;

template<typename Config>
class TestSimd : public ::testing::Test {};

TYPED_TEST_SUITE(TestSimd, TestTypes);

struct TestSimdOperators
{
    static const std::array<double, 128>& input()
    {
        static const std::array<double, 128> data = [](){
            std::array<double, 128> tmp;
            // simple trigonometric pattern
            for (size_t i = 0; i < 128; ++i) {
                tmp[i] = std::abs(std::sin(double(i)) * 10.0);
            }
            // flip the sign of every second value in the second set
            for (size_t i = 64; i < 128; i+=2) {
                tmp[i] = -tmp[i];
            }
            return tmp;
        }();
        return data;
    }

    template <typename T, size_t Size>
    static void makeUnique(std::array<T, Size>& a, std::array<T, Size>& b)
    {
        std::unordered_set<T> seen;
        auto op = [&](std::array<T, Size>& data) {
            for (size_t i = 0; i < Size; ++i) {
                while (seen.count(data[i])) {
                    if constexpr (openvdb::is_floating_point<T>::value) {
                        data[i] = std::nextafter(data[i], std::numeric_limits<T>::infinity());
                    }
                    else {
                        data[i] += static_cast<T>(i + 1);
                    }
                }
                seen.insert(data[i]);
            }
        };
        op(a);
        op(b);
    }

    template <typename T>
    static void ExpectNear(T a, T b)
    {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(a, b, static_cast<T>(1e-3));
        } else {
            EXPECT_EQ(a, b);
        }
    }

    template <typename SimdT>
    static void test()
    {
        using T = typename simd::SimdTraits<SimdT>::ElementT;
        constexpr int Size = simd::SimdTraits<SimdT>::size;
        const auto& in = input();

        std::array<T, Size> data_a;
        std::array<T, Size> data_b;
        std::transform(in.data(),      in.data() + Size,      data_a.begin(), [](double x) { return T(x); });
        std::transform(in.data() + 64, in.data() + Size + 64, data_b.begin(), [](double x) { return T(x); });

        makeUnique(data_a, data_b);

        // test load
        const SimdT a = simd::load<Size>(data_a.data());
        const SimdT b = simd::load<Size>(data_b.data());
        for (int i = 0; i < Size; ++i) EXPECT_EQ(a[i], data_a[i]);
        for (int i = 0; i < Size; ++i) EXPECT_EQ(b[i], data_b[i]);

        // ops
        auto c = a + b;
        for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], T(data_a[i] + data_b[i]));
        c = b - a;
        for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], T(data_b[i] - data_a[i]));
        c = a * b;
        for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], T(data_a[i] * data_b[i]));

        // interface
        c = simd::min(a, b);
        for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], std::min(data_a[i], data_b[i]));
        c = simd::max(a, b);
        for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], std::max(data_a[i], data_b[i]));
        c = simd::pow2(a);
        for (int i = 0; i < Size; ++i) ExpectNear(c[i], T(data_a[i] * data_a[i]));
        c = simd::pow3(a);
        for (int i = 0; i < Size; ++i) ExpectNear(c[i], T(data_a[i] * data_a[i] * data_a[i]));

        // reductions
        auto r1 = simd::horizontal_min(b);
        auto r2 = data_b[0];
        for (int i = 1; i < Size; ++i) r2 = std::min(r2, data_b[i]);
        EXPECT_EQ(r1, r2);

        r1 = simd::horizontal_max(b);
        r2 = data_b[0];
        for (int i = 1; i < Size; ++i) r2 = std::max(r2, data_b[i]);
        EXPECT_EQ(r1, r2);

        r1 = T(simd::horizontal_add(b));
        r2 = data_b[0];
        for (int i = 1; i < Size; ++i) r2 += data_b[i];
        ExpectNear(r1, r2);

        // signeds ops
        if constexpr (std::is_signed_v<T>)
        {
            c = simd::abs(b);
            for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], T(std::abs(data_b[i])));
        }

        // floating point interface
        if constexpr (openvdb::is_floating_point<T>::value)
        {
            c = simd::sqrt(a);
            for (int i = 0; i < Size; ++i) ExpectNear(c[i], std::sqrt(data_a[i]));
            c = a / b;
            for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], data_a[i] / data_b[i]);

            EXPECT_TRUE(simd::horizontal_and(simd::is_finite(a)));
            EXPECT_TRUE(simd::horizontal_and(simd::is_finite(b)));
            EXPECT_TRUE(simd::horizontal_and(simd::is_finite(c)));
        }

        // comparisons/masks
        std::array<bool, Size> expected;
        auto mask = a == b;
        for (int i = 0; i < Size; ++i) expected[i] = data_a[i] == data_b[i];
        for (int i = 0; i < Size; ++i) EXPECT_EQ(mask[i], expected[i]);

        mask = SimdT(simd::horizontal_min(a)) == a;
        r2 = data_a[0];
        for (int i = 1; i < Size; ++i) r2 = std::min(r2, data_a[i]);
        for (int i = 0; i < Size; ++i) expected[i] = r2 == data_a[i];
        for (int i = 0; i < Size; ++i) EXPECT_EQ(mask[i], expected[i]);

        // lt, lte, gt, gte
        mask = a < b;
        for (int i = 0; i < Size; ++i) expected[i] = data_a[i] < data_b[i];
        for (int i = 0; i < Size; ++i) EXPECT_EQ(mask[i], expected[i]);

        mask = b <= a;
        for (int i = 0; i < Size; ++i) expected[i] = data_b[i] <= data_a[i];
        for (int i = 0; i < Size; ++i) EXPECT_EQ(mask[i], expected[i]);

        mask = a > b;
        for (int i = 0; i < Size; ++i) expected[i] = data_a[i] > data_b[i];
        for (int i = 0; i < Size; ++i) EXPECT_EQ(mask[i], expected[i]);

        mask = b >= a;
        for (int i = 0; i < Size; ++i) expected[i] = data_b[i] >= data_a[i];
        for (int i = 0; i < Size; ++i) EXPECT_EQ(mask[i], expected[i]);

        // selects, blends, horizontal mask ops
        std::array<T, Size> data_blend;
        for (int i = 0; i < Size; ++i) data_blend[i] = i % 2 ? data_a[i] : data_b[i];
        const SimdT blend = simd::load<Size>(data_blend.data());

        mask = blend == a;
        EXPECT_FALSE(simd::horizontal_and(mask));
        EXPECT_TRUE(simd::horizontal_or(mask));

        mask = blend == b;
        EXPECT_FALSE(simd::horizontal_and(mask));
        EXPECT_TRUE(simd::horizontal_or(mask));

        EXPECT_EQ(simd::horizontal_count(mask), Size/2);

        c = simd::select(mask, b, a);
        for (int i = 0; i < Size; ++i) EXPECT_EQ(c[i], data_blend[i]);

        mask = SimdT(data_a[Size-1]) == a;
        EXPECT_EQ(simd::horizontal_find_first(mask), Size-1);

        mask = b == a;
        EXPECT_EQ(simd::horizontal_find_first(mask), -1);
    }
};

TYPED_TEST(TestSimd, Operators)
{
    using SimdT = typename TypeParam::SimdT;
    using ExpectedElementT = typename TypeParam::ExpectedElementT;
    constexpr size_t ExpectedSize = TypeParam::ExpectedSize;

    // coule be static, but nicer to fail at test runtime
    EXPECT_TRUE((std::is_same_v<typename simd::SimdTraits<SimdT>::ElementT, ExpectedElementT>));
    EXPECT_EQ(simd::SimdTraits<SimdT>::size, ExpectedSize);

    TestSimdOperators::test<SimdT>();
}
