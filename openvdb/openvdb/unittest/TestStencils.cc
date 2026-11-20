// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/VelocityFields.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace openvdb;

using DefaultGrid = FloatGrid;                                      // 8^3 leaves
using SmallLeafGrid = Grid<tree::Tree4<float, 3, 3, 2>::Type>;      // 4^3 leaves
using TinyLeafGrid = Grid<tree::Tree4<float, 3, 3, 1>::Type>;       // 2^3 leaves

// Leaves next to each other, active and inactive tiles, and background
template<typename GridT>
void populate(GridT& grid, unsigned seed)
{
    auto& tree = grid.tree();
    std::mt19937 generator(seed);
    auto randomCoord = [&](int range) {
        std::uniform_int_distribution<int> distribution(-range, range);
        return Coord(distribution(generator), distribution(generator), distribution(generator));
    };
    for (int i = 0; i < 1500; ++i) tree.setValueOn(randomCoord(200), float(i) * 0.25f);
    tree.fill(CoordBBox(Coord(-9), Coord(9)), 1.5f, true);
    for (int i = 0; i < 400; ++i) {
        tree.setValueOn(Coord(i % 23 - 11, i % 13 - 6, i % 7 - 3), float(i) - 0.5f);
    }
    for (Index level = 1; level <= GridT::TreeType::RootNodeType::LEVEL; ++level) {
        for (int k = 0; k < 6; ++k) {
            tree.addTile(level, randomCoord(k < 3 ? 200 : 2000), float(level * 10 + k), (k & 1) != 0);
        }
    }
}

std::vector<Coord> makeCoords(unsigned seed, int count, int range)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(-range, range);
    std::vector<Coord> coords;
    for (int i = 0; i < count; ++i) {
        coords.emplace_back(distribution(generator), distribution(generator), distribution(generator));
    }
    return coords;
}

// Scan order, random order, and alternating between distant places
std::vector<Coord> makeCoordinateOrders()
{
    std::vector<Coord> coords;
    for (int x = -12; x <= 12; ++x)
        for (int y = -12; y <= 12; ++y)
            for (int z = -12; z <= 12; ++z)
                coords.emplace_back(x, y, z);
    const std::vector<Coord> random = makeCoords(1, 6000, 250);
    coords.insert(coords.end(), random.begin(), random.end());
    const std::vector<Coord> far = makeCoords(2, 3000, 2500);
    for (size_t i = 0; i < far.size(); ++i) {
        coords.push_back(far[i]);
        coords.push_back(Coord(int(i % 17), int((i / 17) % 13), int(i % 5)));
    }
    return coords;
}

// A stencil that reads through a registered accessor and one that reads
// through a NeighborAccessor must hold the same values
template<typename SafeT, typename FastT>
void checkSame(SafeT& safe, FastT& fast)
{
    ASSERT_EQ(safe.size(), fast.size());
    ASSERT_EQ(safe.getCenterCoord(), fast.getCenterCoord());
    for (int pos = 0; pos < safe.size(); ++pos) {
        ASSERT_EQ(safe.getValue(pos), fast.getValue(pos))
            << "position " << pos << " at " << fast.getCenterCoord();
    }
}

// DenseStencil hides the moveTo that takes a center value
template<typename StencilT, typename = void>
struct HasCenterValueMove: std::false_type {};
template<typename StencilT>
struct HasCenterValueMove<StencilT, std::void_t<decltype(std::declval<StencilT&>().moveTo(
    std::declval<const Coord&>(), std::declval<const typename StencilT::ValueType&>()))>>
    : std::true_type {};

template<template<typename, bool> class StencilT, typename GridT, typename... Args>
void checkStencil(const GridT& grid, const std::vector<Coord>& coords, Args... args)
{
    StencilT<GridT, true> safe(grid, args...);
    StencilT<GridT, false> fast(grid, args...);

    // By coordinate
    for (const Coord& xyz : coords) {
        safe.moveTo(xyz);
        fast.moveTo(xyz);
        checkSame(safe, fast);
    }

    // By coordinate and a known center value
    if constexpr (HasCenterValueMove<decltype(safe)>::value) {
        for (const Coord& xyz : coords) {
            const auto center = grid.tree().getValue(xyz);
            safe.moveTo(xyz, center);
            fast.moveTo(xyz, center);
            checkSame(safe, fast);
        }
    }

    // By iterator, over the active voxels of every leaf
    for (auto leafIter = grid.tree().cbeginLeaf(); leafIter; ++leafIter) {
        for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
            safe.moveTo(iter);
            fast.moveTo(iter);
            checkSame(safe, fast);
        }
    }
}

template<typename GridT>
void checkAllStencils(unsigned seed)
{
    GridT grid(0.0f);
    populate(grid, seed);
    const std::vector<Coord> coords = makeCoordinateOrders();

    checkStencil<math::SevenPointStencil>(grid, coords);
    checkStencil<math::BoxStencil>(grid, coords);
    checkStencil<math::SecondOrderDenseStencil>(grid, coords);
    checkStencil<math::ThirteenPointStencil>(grid, coords);
    checkStencil<math::FourthOrderDenseStencil>(grid, coords);
    checkStencil<math::NineteenPointStencil>(grid, coords);
    checkStencil<math::SixthOrderDenseStencil>(grid, coords);
    checkStencil<math::GradStencil>(grid, coords);
    checkStencil<math::WenoStencil>(grid, coords);
    checkStencil<math::CurvatureStencil>(grid, coords);
}

} // anonymous namespace


class TestStencils : public ::testing::Test
{
};

TEST_F(TestStencils, testMinMax)
{
    using namespace openvdb;

    Int32Grid grid;
    grid.tree().setValue(math::Coord(0,0,0), -3);
    grid.tree().setValue(math::Coord(0,0,1), -2);
    grid.tree().setValue(math::Coord(0,1,0), -1);
    grid.tree().setValue(math::Coord(1,0,0),  0);
    grid.tree().setValue(math::Coord(1,1,0),  1);
    grid.tree().setValue(math::Coord(0,1,1),  2);
    grid.tree().setValue(math::Coord(1,0,1),  3);
    grid.tree().setValue(math::Coord(1,1,1),  4);
    math::BoxStencil<Int32Grid> stencil(grid);

    stencil.moveTo(Coord(0,0,0));
    EXPECT_EQ(stencil.min(), -3);
    EXPECT_EQ(stencil.max(), 4);

    stencil.moveTo(Coord(1,1,1));
    EXPECT_EQ(stencil.min(), 0);
    EXPECT_EQ(stencil.max(), 4);

    stencil.moveTo(Coord(0,0,1));
    EXPECT_EQ(stencil.min(), -2);
    EXPECT_EQ(stencil.max(),  4);
}


TEST_F(TestStencils, testUnsafeStencils)
{
    checkAllStencils<DefaultGrid>(11);
    // The 19 point stencils reach past a 4^3 leaf
    checkAllStencils<SmallLeafGrid>(12);
    // and past two 2^3 leaves
    checkAllStencils<TinyLeafGrid>(13);
}

TEST_F(TestStencils, testUnsafeDenseStencil)
{
    const std::vector<Coord> coords = makeCoords(21, 3000, 60);
    DefaultGrid grid(0.0f);
    populate(grid, 22);
    // A half width of 9 is wider than an 8^3 leaf
    for (int halfWidth : {1, 2, 3, 9}) {
        checkStencil<math::DenseStencil>(grid, coords, halfWidth);
    }
    SmallLeafGrid small(0.0f);
    populate(small, 23);
    for (int halfWidth : {1, 3, 5}) {
        checkStencil<math::DenseStencil>(small, coords, halfWidth);
    }
}

TEST_F(TestStencils, testUnsafeBoxStencilPositions)
{
    DefaultGrid grid(0.0f);
    populate(grid, 31);
    math::BoxStencil<DefaultGrid, true> safe(grid);
    math::BoxStencil<DefaultGrid, false> fast(grid);

    std::mt19937 generator(32);
    std::uniform_real_distribution<double> distribution(-60.0, 60.0);
    for (int i = 0; i < 5000; ++i) {
        const Vec3d xyz(distribution(generator), distribution(generator), distribution(generator));
        safe.moveTo(xyz);
        fast.moveTo(xyz);
        checkSame(safe, fast);
        const Vec3f position(xyz);
        ASSERT_EQ(safe.interpolation(position), fast.interpolation(position));
        ASSERT_EQ(safe.gradient(position), fast.gradient(position));
    }
}

TEST_F(TestStencils, testUnsafeVectorValues)
{
    Vec3SGrid grid(Vec3f(0.0f));
    for (int i = 0; i < 2000; ++i) {
        grid.tree().setValueOn(Coord(i % 31 - 15, (i / 3) % 29 - 14, (i / 7) % 23 - 11),
            Vec3f(float(i), float(-i), 0.5f));
    }
    const std::vector<Coord> coords = makeCoords(41, 3000, 25);
    checkStencil<math::SevenPointStencil>(grid, coords);
}

TEST_F(TestStencils, testUnsafeStencilClear)
{
    DefaultGrid grid(0.0f);
    const Coord xyz(100, 100, 100);
    math::SevenPointStencil<DefaultGrid, false> stencil(grid);

    // The first read caches the background around xyz
    stencil.moveTo(xyz);
    EXPECT_EQ(0.0f, (stencil.getValue<1, 0, 0>()));

    // A leaf appears. clear() drops what was cached
    grid.tree().setValueOn(xyz, 2.0f);
    grid.tree().setValueOn(xyz.offsetBy(1, 0, 0), 3.0f);
    stencil.clear();
    stencil.moveTo(xyz);
    EXPECT_EQ(2.0f, stencil.getCenterValue());
    EXPECT_EQ(3.0f, (stencil.getValue<1, 0, 0>()));

    // clear() also forgets the center, so a floating point move to the same voxel reads the tree
    grid.tree().clear();
    stencil.clear();
    stencil.moveTo(Vec3d(100.0, 100.0, 100.0));
    EXPECT_EQ(0.0f, stencil.getCenterValue());
    EXPECT_EQ(0.0f, (stencil.getValue<1, 0, 0>()));
}


namespace {

// Random voxels, plus tiles of both states next to leaves
template<typename GridT>
typename GridT::Ptr
makeStencilGrid(unsigned seed)
{
    using namespace openvdb;
    auto grid = GridT::create(typename GridT::ValueType(10));
    auto& tree = grid->tree();
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(-40, 40);
    for (int i = 0; i < 3000; ++i) {
        const Coord xyz(distribution(generator), distribution(generator), distribution(generator));
        tree.setValueOn(xyz, typename GridT::ValueType(i % 97) * 0.25f);
    }
    // Neighboring leaves
    for (int x = 0; x < 24; ++x)
        for (int y = 0; y < 24; ++y)
            for (int z = 0; z < 24; ++z)
                if (((x * 5 + y * 3 + z) % 5) != 0) {
                    tree.setValueOn(Coord(x, y, z), typename GridT::ValueType(x - y + 2 * z));
                }
    tree.fill(CoordBBox(Coord(24, 0, 0), Coord(47, 23, 23)), typename GridT::ValueType(-7), false);
    tree.fill(CoordBBox(Coord(0, 24, 0), Coord(23, 47, 23)), typename GridT::ValueType(-8), true);
    return grid;
}

template<openvdb::math::BiasedGradientScheme Scheme>
void checkBiasedGradient(const openvdb::FloatGrid& grid)
{
    using namespace openvdb;
    using SchemeT = math::BIAS_SCHEME<Scheme>;
    using SafeT = typename SchemeT::template ISStencil<FloatGrid, true>::StencilType;
    using FastT = typename SchemeT::template ISStencil<FloatGrid, false>::StencilType;
    using GradientT = math::GradientBiased<math::UniformScaleMap, Scheme>;

    const math::UniformScaleMap map(0.5);
    const Vec3f velocity(0.3f, -1.0f, 0.7f);
    SafeT safe(grid);
    FastT fast(grid);
    for (auto leafIter = grid.tree().cbeginLeaf(); leafIter; ++leafIter) {
        for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
            safe.moveTo(iter.getCoord());
            fast.moveTo(iter);
            EXPECT_EQ(math::ISGradientNormSqrd<Scheme>::result(safe),
                math::ISGradientNormSqrd<Scheme>::result(fast));
            EXPECT_EQ(GradientT::result(map, safe, velocity),
                GradientT::result(map, fast, velocity));
            if (::testing::Test::HasFailure()) return;
        }
    }
}

} // anonymous namespace


TEST_F(TestStencils, testUnsafeBiasedGradient)
{
    using namespace openvdb;
    auto grid = makeStencilGrid<FloatGrid>(5);
    checkBiasedGradient<math::FIRST_BIAS>(*grid);
    checkBiasedGradient<math::SECOND_BIAS>(*grid);
    checkBiasedGradient<math::THIRD_BIAS>(*grid);
    checkBiasedGradient<math::WENO5_BIAS>(*grid);
    checkBiasedGradient<math::HJWENO5_BIAS>(*grid);
}


namespace {

template<typename GridT>
typename GridT::Ptr
makeDistortedSphere(float voxelSize)
{
    using namespace openvdb;
    auto grid = tools::createLevelSetSphere<GridT>(0.15f, Vec3f(0.35f), voxelSize, 3.0f);
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        const Coord xyz = iter.getCoord();
        const float noise = float(((xyz[0] * 7 + xyz[1] * 13 + xyz[2] * 29) % 11) - 5) / 5.0f;
        iter.setValue(*iter + 0.4f * voxelSize * noise);
    }
    return grid;
}

template<typename GridA, typename GridB>
void expectSameValues(const GridA& expected, const GridB& actual, float tolerance)
{
    EXPECT_EQ(expected.activeVoxelCount(), actual.activeVoxelCount());
    auto accessor = actual.getConstAccessor();
    for (auto iter = expected.cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord xyz = iter.getCoord();
        ASSERT_TRUE(accessor.isValueOn(xyz)) << xyz;
        ASSERT_NEAR(*iter, accessor.getValue(xyz), tolerance) << xyz;
    }
}

// One Euler step of normalization with the math:: stencils
template<openvdb::math::BiasedGradientScheme Scheme>
void normalizeWithStencils(openvdb::FloatGrid& grid, float voxelSize, const openvdb::MaskTree* mask)
{
    using namespace openvdb;
    using StencilT = typename math::BIAS_SCHEME<Scheme>::template ISStencil<FloatGrid>::StencilType;

    const FloatGrid::Ptr source = grid.deepCopy();
    StencilT stencil(*source);
    const float dt = voxelSize * 0.3f;
    const float inverseVoxelSize = 1.0f / voxelSize;
    for (auto iter = source->cbeginValueOn(); iter; ++iter) {
        if (mask && !mask->isValueOn(iter.getCoord())) continue;
        stencil.moveTo(iter);
        const float normSqGradient = math::ISGradientNormSqrd<Scheme>::result(stencil);
        const float phi = stencil.getValue();
        float v = phi / (math::Sqrt(math::Pow2(phi) + normSqGradient) + math::Tolerance<float>::value());
        v = phi - dt * v * (math::Sqrt(normSqGradient) * inverseVoxelSize - 1.0f);
        grid.tree().setValueOnly(iter.getCoord(), v);
    }
}

template<openvdb::math::BiasedGradientScheme Scheme>
void checkNormalize(float voxelSize, const openvdb::MaskTree* mask)
{
    using namespace openvdb;
    SCOPED_TRACE(::testing::Message() << "spatial " << int(Scheme) << (mask ? " masked" : ""));
    auto actual = makeDistortedSphere<FloatGrid>(voxelSize);
    auto expected = makeDistortedSphere<FloatGrid>(voxelSize);
    tools::LevelSetTracker<FloatGrid> tracker(*actual);
    tracker.setSpatialScheme(Scheme);
    tracker.setTemporalScheme(math::TVD_RK1);
    tracker.setNormCount(1);
    tracker.normalize(mask);
    normalizeWithStencils<Scheme>(*expected, voxelSize, mask);
    expectSameValues(*expected, *actual, 1.0e-4f * voxelSize);
}

} // anonymous namespace


TEST_F(TestStencils, testLevelSetTrackerNormalize)
{
    using namespace openvdb;

    const float voxelSize = 1.0f / 48.0f;

    // A mask of every other column of voxels
    MaskTree mask;
    {
        auto grid = makeDistortedSphere<FloatGrid>(voxelSize);
        for (auto iter = grid->cbeginValueOn(); iter; ++iter) {
            if (iter.getCoord()[0] % 2 == 0) mask.setValueOn(iter.getCoord());
        }
    }

    const MaskTree* masks[] = {nullptr, &mask};
    for (const MaskTree* maskPointer : masks) {
        checkNormalize<math::FIRST_BIAS>(voxelSize, maskPointer);
        checkNormalize<math::SECOND_BIAS>(voxelSize, maskPointer);
        checkNormalize<math::THIRD_BIAS>(voxelSize, maskPointer);
        checkNormalize<math::WENO5_BIAS>(voxelSize, maskPointer);
        checkNormalize<math::HJWENO5_BIAS>(voxelSize, maskPointer);
    }
}


TEST_F(TestStencils, testLevelSetAdvection)
{
    using namespace openvdb;

    const float voxelSize = 1.0f / 48.0f;
    for (math::BiasedGradientScheme scheme : {math::FIRST_BIAS, math::HJWENO5_BIAS}) {
        auto grid = makeDistortedSphere<FloatGrid>(voxelSize);
        tools::EnrightField<float> field;
        tools::LevelSetAdvection<FloatGrid, tools::EnrightField<float>> advection(*grid, field);
        advection.setSpatialScheme(scheme);
        advection.setTrackerSpatialScheme(scheme);
        EXPECT_TRUE(advection.advect(0.0f, 0.02f) > 0);
        EXPECT_TRUE(grid->activeVoxelCount() > 0);
        for (auto iter = grid->cbeginValueOn(); iter; ++iter) {
            ASSERT_TRUE(std::isfinite(*iter)) << iter.getCoord();
        }
    }
}
