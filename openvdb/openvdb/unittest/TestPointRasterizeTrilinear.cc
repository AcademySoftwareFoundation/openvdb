// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointRasterizeTrilinear.h>
#include <openvdb/util/Assert.h>

#include "PointBuilder.h"

#include <gtest/gtest.h>

using namespace openvdb;

class TestPointRasterize: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointRasterize

template <bool S, typename V>
using RasterizeTrilinearT =
    decltype(points::rasterizeTrilinear<S, V, points::NullFilter, points::PointDataTree>(
        std::declval<const points::PointDataTree&>(),
        std::declval<const std::string&>(),
        std::declval<const points::NullFilter&>()));

// Assert some expected tree types from invoking rasterizeTrilinear
static_assert(std::is_same_v<RasterizeTrilinearT<false, int32_t>, FloatTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  int32_t>, Vec3fTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<false, int64_t>, DoubleTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  int64_t>, Vec3dTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<false, uint32_t>, FloatTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  uint32_t>, Vec3fTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<false, uint64_t>, DoubleTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  uint64_t>, Vec3dTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<false, float>, FloatTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  float>, Vec3fTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<false, double>, DoubleTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  double>, Vec3dTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<false, Vec3f>, Vec3fTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  Vec3f>, Vec3fTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<false, Vec3d>, Vec3dTree::Ptr>);
static_assert(std::is_same_v<RasterizeTrilinearT<true,  Vec3d>, Vec3dTree::Ptr>);

inline double kweight(const Vec3d& dist)
{
    return
        points::rasterize_trilinear_internal::value(dist[0]) *
        points::rasterize_trilinear_internal::value(dist[1]) *
        points::rasterize_trilinear_internal::value(dist[2]);
}

template <typename T>
inline void TestTrilinearScalar()
{
    using ElemT = typename ValueTraits<T>::ElementType;
    using FltT = typename types_internal::flt_t<sizeof(ElemT)*CHAR_BIT>::type;

    // Test single point at the origin (center of a voxel)
    auto points = PointBuilder({Vec3f(0)}).attribute(T(111.0), "test").get();
    {
        auto tree = points::rasterizeTrilinear<false, T>(points->tree(), "test");
        static_assert(
            (sizeof(T) == 4 && std::is_same_v<FloatTree::Ptr, decltype(tree)>) ||
            (sizeof(T) == 8 && std::is_same_v<DoubleTree::Ptr, decltype(tree)>));
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount()); // should probably by 8 but we don't deactivate
        for (auto iter = tree->cbeginValueAll(); iter; ++iter) {
            // we're at the origin, expect either full weight or no weight
            if (iter.getCoord() == Coord(0,0,0)) EXPECT_NEAR(FltT(111.0), *iter, 1e-6f);
            else                                 EXPECT_EQ(FltT(0.0), *iter);
        }
    }
    {
        auto tree = points::rasterizeTrilinear<true, T>(points->tree(), "test");
        static_assert(
            (sizeof(T) == 4 && std::is_same_v<Vec3fTree::Ptr, decltype(tree)>) ||
            (sizeof(T) == 8 && std::is_same_v<Vec3dTree::Ptr, decltype(tree)>));
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            const Vec3d dx = iter.getCoord().asVec3d() - Vec3d(0.5,0,0);
            const Vec3d dy = iter.getCoord().asVec3d() - Vec3d(0,0.5,0);
            const Vec3d dz = iter.getCoord().asVec3d() - Vec3d(0,0,0.5);
            // we know we're at the origin, so we expect a |hat| function
            // here, where the weight is either exactly zero or one
            EXPECT_NEAR(kweight(dx) > 0 ? FltT(111.0) : 0.0f, (*iter)[0], 1e-6f);
            EXPECT_NEAR(kweight(dy) > 0 ? FltT(111.0) : 0.0f, (*iter)[1], 1e-6f);
            EXPECT_NEAR(kweight(dz) > 0 ? FltT(111.0) : 0.0f, (*iter)[2], 1e-6f);
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(iter->zero(), *iter);
    }

    // Test eight point at the origin (all overlapping). Result should be evenly weighted
    auto positions = getBoxPoints(/*scale*/0.0f); // 8 positions
    const std::vector<T> values { T(-1.0), T(0.0), T(2.3), T(5.4), T(8.4), T(-9.1), T(0.0), T(0.1) };
    const FltT expected = [&]() {
        FltT r = 0;
        for (auto& v : values) r += FltT(v);
        return r / FltT(8);
    }();

    points = PointBuilder(positions).attribute(values, "test").get();
    {
        auto tree = points::rasterizeTrilinear<false, T>(points->tree(), "test");
        static_assert(
            (sizeof(T) == 4 && std::is_same_v<FloatTree::Ptr, decltype(tree)>) ||
            (sizeof(T) == 8 && std::is_same_v<DoubleTree::Ptr, decltype(tree)>));
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueAll(); iter; ++iter) {
            // we're at the origin, expect either full weight or no weight
            if (iter.getCoord() == Coord(0,0,0)) EXPECT_NEAR(expected, *iter, 1e-6f);
            else                                 EXPECT_EQ(FltT(0.0), *iter);
        }
    }
    {
        auto tree = points::rasterizeTrilinear<true, T>(points->tree(), "test");
        static_assert(
            (sizeof(T) == 4 && std::is_same_v<Vec3fTree::Ptr, decltype(tree)>) ||
            (sizeof(T) == 8 && std::is_same_v<Vec3dTree::Ptr, decltype(tree)>));
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount());

        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            const Vec3d dx = iter.getCoord().asVec3d() - Vec3d(0.5,0,0);
            const Vec3d dy = iter.getCoord().asVec3d() - Vec3d(0,0.5,0);
            const Vec3d dz = iter.getCoord().asVec3d() - Vec3d(0,0,0.5);
            // we know we're at the origin, so we expect a |hat| function
            // here, where the weight is either exactly zero or one
            EXPECT_NEAR(kweight(dx) > 0 ? expected : 0.0f, (*iter)[0], 1e-6f);
            EXPECT_NEAR(kweight(dy) > 0 ? expected : 0.0f, (*iter)[1], 1e-6f);
            EXPECT_NEAR(kweight(dz) > 0 ? expected : 0.0f, (*iter)[2], 1e-6f);
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(iter->zero(), *iter);
    }

    // Test eight points
    positions = getBoxPoints(); // 8 positions
    points = PointBuilder(positions).attribute(values, "test").get();
    // positions to index space for the test check
    for (auto& p : positions) p = Vec3f(points->transform().worldToIndex(p));
    {
        auto tree = points::rasterizeTrilinear<false, T>(points->tree(), "test");
        static_assert(
            (sizeof(T) == 4 && std::is_same_v<FloatTree::Ptr, decltype(tree)>) ||
            (sizeof(T) == 8 && std::is_same_v<DoubleTree::Ptr, decltype(tree)>));
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(216), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            FltT expected(0.0), weight(0.0);
            for (size_t i = 0; i < 8; ++i) {
                FltT w = FltT(kweight(positions[i] - iter.getCoord().asVec3d()));
                weight += w;
                expected += FltT(values[i]) * w;
            }
            EXPECT_GE(weight, 0.0);
            if (bool(weight)) expected /= weight;
            EXPECT_NEAR(FltT(expected), *iter, 1e-6f);
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(T(0), *iter);
    }
    {
        auto tree = points::rasterizeTrilinear<true, T>(points->tree(), "test");
        static_assert(
            (sizeof(T) == 4 && std::is_same_v<Vec3fTree::Ptr, decltype(tree)>) ||
            (sizeof(T) == 8 && std::is_same_v<Vec3dTree::Ptr, decltype(tree)>));
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(216), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            const Vec3d ijk = iter.getCoord().asVec3d();
            math::Vec3<FltT> expected(0.0f), weight(0.0f);
            for (size_t i = 0; i < 8; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    Vec3d offset(0.0);
                    offset[j] = 0.5;
                    FltT w = FltT(kweight(positions[i] - (ijk - offset)));
                    weight[j] += w;
                    expected[j] += FltT(values[i]) * w;
                }
            }
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_GE(weight[j], FltT(0.0)) << j;
                if (bool(weight[j])) expected[j] /= weight[j];
            }
            EXPECT_NEAR(expected[0], (*iter)[0], 1e-6f);
            EXPECT_NEAR(expected[1], (*iter)[1], 1e-6f);
            EXPECT_NEAR(expected[2], (*iter)[2], 1e-6f);
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(iter->zero(), *iter);
    }
}

TEST_F(TestPointRasterize, testTrilinearRasterizeScalar)
{
    TestTrilinearScalar<float>();
    TestTrilinearScalar<double>();
    TestTrilinearScalar<int32_t>();
    TestTrilinearScalar<int64_t>();
}


template <typename Vec3T>
inline void TestTrilinearVec()
{
    using ElemT = typename Vec3T::ValueType;
    using FltT = typename types_internal::flt_t<sizeof(ElemT)*CHAR_BIT>::type;
    using TreeT = typename openvdb::points::PointDataGrid::ValueConverter<math::Vec3<FltT>>::Type::TreeType;

    // Test single point at the origin (center of a voxel)
    auto points = PointBuilder({Vec3T(0)}).attribute(Vec3T(1.0, 2.0, 3.0), "test").get();
    {
        auto tree = points::rasterizeTrilinear<false, Vec3T>(points->tree(), "test");
        static_assert(std::is_same_v<typename TreeT::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount()); // should probably by 8 but we don't deactivate
        for (auto iter = tree->cbeginValueAll(); iter; ++iter) {
            // we're at the origin, expect either full weight or no weight
            if (iter.getCoord() == Coord(0,0,0)) {
                EXPECT_NEAR(1.0f, (*iter)[0], 1e-6f);
                EXPECT_NEAR(2.0f, (*iter)[1], 1e-6f);
                EXPECT_NEAR(3.0f, (*iter)[2], 1e-6f);
            }
            else {
                EXPECT_EQ(iter->zero(), *iter);
            }
        }
    }
    {
        auto tree = points::rasterizeTrilinear<true, Vec3T>(points->tree(), "test");
        static_assert(std::is_same_v<typename TreeT::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            const Vec3d dx = iter.getCoord().asVec3d() - Vec3d(0.5,0,0);
            const Vec3d dy = iter.getCoord().asVec3d() - Vec3d(0,0.5,0);
            const Vec3d dz = iter.getCoord().asVec3d() - Vec3d(0,0,0.5);
            // we know we're at the origin, so we expect a |hat| function
            // here, where the weight is either exactly zero or one
            EXPECT_NEAR(kweight(dx) > 0 ? 1.0f : 0.0f, (*iter)[0], 1e-6f) << iter.getCoord();
            EXPECT_NEAR(kweight(dy) > 0 ? 2.0f : 0.0f, (*iter)[1], 1e-6f) << iter.getCoord();
            EXPECT_NEAR(kweight(dz) > 0 ? 3.0f : 0.0f, (*iter)[2], 1e-6f) << iter.getCoord();
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(iter->zero(), *iter);
    }

    // Test eight point at the origin (all overlapping). Result should be evenly weighted
    auto positions = getBoxPoints(/*scale*/0.0f); // 8 positions
    const std::vector<Vec3T> values {
        Vec3T(ElemT(-1.0f), ElemT(0.0f), ElemT(2.3f)),
        Vec3T(ElemT(5.4f),  ElemT(8.4f), ElemT(-9.1f)),
        Vec3T(ElemT(0.0f),  ElemT(0.1f), ElemT(0.0f)),
        Vec3T(ElemT(8.2f),  ElemT(3.1f), ElemT(0.0f)),
        Vec3T(ElemT(0.0f),  ElemT(0.0f), ElemT(0.0f)),
        Vec3T(ElemT(-9.0f), ElemT(0.0f), ElemT(-3.0f)),
        Vec3T(ElemT(0.5f),  ElemT(0.5f), ElemT(0.5f)),
        Vec3T(ElemT(0.0f),  ElemT(0.1f), ElemT(0.0f))
    };
    const math::Vec3<FltT> expected = [&]() {
        math::Vec3<FltT> r(0.0f);
        for (auto& v : values) r += math::Vec3<FltT>(v);
        return r / float(8);
    }();

    points = PointBuilder(positions).attribute(values, "test").get();
    {
        auto tree = points::rasterizeTrilinear<false, Vec3T>(points->tree(), "test");
        static_assert(std::is_same_v<typename TreeT::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueAll(); iter; ++iter) {
            // we're at the origin, expect either full weight or no weight
            if (iter.getCoord() == Coord(0,0,0)) {
                EXPECT_NEAR(expected[0], (*iter)[0], 1e-6f);
                EXPECT_NEAR(expected[1], (*iter)[1], 1e-6f);
                EXPECT_NEAR(expected[2], (*iter)[2], 1e-6f);
            }
            else {
                EXPECT_EQ(iter->zero(), *iter);
            }
        }
    }
    {
        auto tree = points::rasterizeTrilinear<true, Vec3T>(points->tree(), "test");
        static_assert(std::is_same_v<typename TreeT::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount());

        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            const Vec3d dx = iter.getCoord().asVec3d() - Vec3d(0.5,0,0);
            const Vec3d dy = iter.getCoord().asVec3d() - Vec3d(0,0.5,0);
            const Vec3d dz = iter.getCoord().asVec3d() - Vec3d(0,0,0.5);
            // we know we're at the origin, so we expect a |hat| function
            // here, where the weight is either exactly zero or one
            EXPECT_NEAR(kweight(dx) > 0 ? expected[0] : 0.0f, (*iter)[0], 1e-6f) << iter.getCoord();
            EXPECT_NEAR(kweight(dy) > 0 ? expected[1] : 0.0f, (*iter)[1], 1e-6f) << iter.getCoord();
            EXPECT_NEAR(kweight(dz) > 0 ? expected[2] : 0.0f, (*iter)[2], 1e-6f) << iter.getCoord();
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(iter->zero(), *iter);
    }

    // Test eight points
    positions = getBoxPoints(); // 8 positions
    points = PointBuilder(positions).attribute(values, "test").get();
    // positions to index space for the test check
    for (auto& p : positions) p = Vec3f(points->transform().worldToIndex(p));
    {
        auto tree = points::rasterizeTrilinear<false, Vec3T>(points->tree(), "test");
        static_assert(std::is_same_v<typename TreeT::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(216), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            math::Vec3<FltT> expected(0.0f);
            FltT weight = 0.0f;
            for (size_t i = 0; i < 8; ++i) {
                FltT w = FltT(kweight(positions[i] - iter.getCoord().asVec3d()));
                weight += w;
                expected += values[i] * w;
            }
            EXPECT_GE(weight, 0.0f);
            if (bool(weight)) expected /= weight;
            EXPECT_NEAR(expected[0], (*iter)[0], 1e-6f);
            EXPECT_NEAR(expected[1], (*iter)[1], 1e-6f);
            EXPECT_NEAR(expected[2], (*iter)[2], 1e-6f);
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(iter->zero(), *iter);
    }
    {
        auto tree = points::rasterizeTrilinear<true, Vec3T>(points->tree(), "test");
        static_assert(std::is_same_v<typename TreeT::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(216), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            const Vec3d ijk = iter.getCoord().asVec3d();
            math::Vec3<FltT> expected(0.0f), weight(0.0f);
            for (size_t i = 0; i < 8; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    Vec3d offset(0.0);
                    offset[j] = 0.5;
                    FltT w = FltT(kweight(positions[i] - (ijk - offset)));
                    weight[j] += w;
                    expected[j] += FltT(values[i][j]) * w;
                }
            }
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_GE(weight[j], 0.0f) << j;
                if (bool(weight[j])) expected[j] /= weight[j];
            }
            EXPECT_NEAR(expected[0], (*iter)[0], 1e-6f);
            EXPECT_NEAR(expected[1], (*iter)[1], 1e-6f);
            EXPECT_NEAR(expected[2], (*iter)[2], 1e-6f);
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(iter->zero(), *iter);
    }
}


TEST_F(TestPointRasterize, testTrilinearRasterizeVec)
{
    TestTrilinearVec<Vec3f>();
    TestTrilinearVec<Vec3i>();
    TestTrilinearVec<Vec3d>();
}


TEST_F(TestPointRasterize, testTrilinearRasterizeMat)
{
    using FltT = float;
    using Mat3sTree = tree::Tree4<Mat3s, 5, 4, 3>::Type;

    // Test single point at the origin (center of a voxel)
    auto points = PointBuilder({Vec3f(0)}).attribute(Mat3s(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f), "test").get();
    {
        auto tree = points::rasterizeTrilinear<false, Mat3s>(points->tree(), "test");
        static_assert(std::is_same_v<Mat3sTree::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount()); // should probably by 8 but we don't deactivate
        for (auto iter = tree->cbeginValueAll(); iter; ++iter) {
            // we're at the origin, expect either full weight or no weight
            if (iter.getCoord() == Coord(0,0,0)) {
                for (size_t i = 0; i < 9; ++i) EXPECT_NEAR(float(i+1), (*iter).asPointer()[i], 1e-6f);
            }
            else {
                EXPECT_EQ(Mat3s::zero(), *iter);
            }
        }
    }
    // Test eight point at the origin (all overlapping). Result should be evenly weighted
    auto positions = getBoxPoints(/*scale*/0.0f); // 8 positions
    const std::vector<Mat3s> values {
        Mat3s(-1.0f, 0.0f, 2.3f,  5.4f, 0.0f, -14.0f,  0.24f, 0.2f, 1.0f),
        Mat3s(5.4f, 8.4f, -9.1f,  0.5f, 0.5f, 0.5f,    0.5f, 0.5f, 0.5f),
        Mat3s::zero(),
        Mat3s(8.2f, 3.1f, 0.0f,  -1.0f, 0.0f, 2.3f,    1.0f, 1.0f, 1.0f),
        Mat3s(0.0f, 0.0f, 0.0f,   5.1f, 7.2f, 9.0f,    0.0f, 0.0f, 0.0f),
        Mat3s::identity(),
        Mat3s(0.5f, 0.5f, 0.5f,   0.5f, 0.5f, 0.5f,    0.5f, 0.5f, 0.5f),
        Mat3s::zero(),
    };
    const Mat3s expected = [&]() {
        Mat3s r = Mat3s::zero();
        for (auto& v : values) r += v;
        for (size_t i = 0; i < 9; ++i) r.asPointer()[i] /= float(8);
        return r;
    }();

    points = PointBuilder(positions).attribute(values, "test").get();
    {
        auto tree = points::rasterizeTrilinear<false, Mat3s>(points->tree(), "test");
        static_assert(std::is_same_v<Mat3sTree::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(27), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueAll(); iter; ++iter) {
            // we're at the origin, expect either full weight or no weight
            if (iter.getCoord() == Coord(0,0,0)) {
                for (size_t i = 0; i < 9; ++i) {
                    EXPECT_NEAR(expected.asPointer()[i], (*iter).asPointer()[i], 1e-6f);
                }
            }
            else {
                EXPECT_EQ(Mat3s::zero(), *iter);
            }
        }
    }
    // Test eight points
    positions = getBoxPoints(); // 8 positions
    points = PointBuilder(positions).attribute(values, "test").get();
    // positions to index space for the test check
    for (auto& p : positions) p = Vec3f(points->transform().worldToIndex(p));
    {
        auto tree = points::rasterizeTrilinear<false, Mat3s>(points->tree(), "test");
        static_assert(std::is_same_v<Mat3sTree::Ptr, decltype(tree)>);
        EXPECT_EQ(Index64(8), tree->leafCount());
        EXPECT_EQ(Index64(0), tree->activeTileCount());
        EXPECT_EQ(Index64(216), tree->activeVoxelCount());
        for (auto iter = tree->cbeginValueOn(); iter; ++iter) {
            Mat3s expected = Mat3s::zero();
            float weight = 0.0f;
            for (size_t i = 0; i < 8; ++i) {
                FltT w = FltT(kweight(positions[i] - iter.getCoord().asVec3d()));
                weight += w;
                expected += values[i] * w;
            }
            EXPECT_GE(weight, 0.0f);
            if (bool(weight)) {
                for (size_t i = 0; i < 9; ++i) expected.asPointer()[i] /= weight;
            }
            for (size_t i = 0; i < 9; ++i) {
                EXPECT_NEAR(expected.asPointer()[i], (*iter).asPointer()[i], 1e-6f);
            }
        }
        for (auto iter = tree->cbeginValueOff(); iter; ++iter) EXPECT_EQ(Mat3s::zero(), *iter);
    }
}
