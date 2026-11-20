// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file unittest/TestNeighborSampling.cc
/// @brief The tools that sample through a tree::NeighborAccessor, against the
/// same tools through a ValueAccessor

#include <openvdb/openvdb.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointSample.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetMorph.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/MultiResGrid.h>
#include <openvdb/tools/VelocityFields.h>
#include <openvdb/tools/VolumeAdvect.h>
#include <openvdb/tree/NeighborAccessor.h>

#include <gtest/gtest.h>

#include <tbb/global_control.h>

#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace openvdb;

class TestNeighborSampling: public ::testing::Test
{
public:
    void SetUp() override { initialize(); }
    void TearDown() override { uninitialize(); }
};


namespace {

// A sampler with no NeighborAccessor overloads, so that the tools read through a ValueAccessor
template<typename SamplerT>
struct AccessorOnlySampler
{
    static const char* name() { return SamplerT::name(); }
    static int radius() { return SamplerT::radius(); }
    static bool mipmap() { return SamplerT::mipmap(); }
    static bool consistent() { return SamplerT::consistent(); }
    static bool staggered() { return SamplerT::staggered(); }
    static size_t order() { return SamplerT::order(); }

    template<class TreeT>
    static bool sample(const TreeT& tree, const Vec3R& xyz, typename TreeT::ValueType& result)
    {
        return SamplerT::sample(tree, xyz, result);
    }

    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& tree, const Vec3R& xyz)
    {
        return SamplerT::sample(tree, xyz);
    }
};

static_assert(tools::HasNeighborAccessorSample<tools::BoxSampler, const FloatTree>::value);
static_assert(tools::HasNeighborAccessorSample<tools::Sampler<2, true>, const Vec3STree>::value);
static_assert(!tools::HasNeighborAccessorSample<
    AccessorOnlySampler<tools::BoxSampler>, const FloatTree>::value);

// The point sampler of points::sampleGrid() with no NeighborAccessor overload
struct AccessorOnlySampleWithRounding
{
    template<typename ValueT, typename SamplerT, typename AccessorT>
    ValueT sample(const AccessorT& accessor, const Vec3d& position) const
    {
        return points::SampleWithRounding().template sample<ValueT, SamplerT, AccessorT>(
            accessor, position);
    }
};

template<typename ValueT>
ValueT
fieldValue(const Coord& xyz)
{
    const double x = xyz.x(), y = xyz.y(), z = xyz.z();
    if constexpr (VecTraits<ValueT>::IsVec) {
        using ElementT = typename VecTraits<ValueT>::ElementType;
        return ValueT(ElementT(0.25 * std::sin(0.3 * y) - 0.05 * z),
            ElementT(0.2 * std::cos(0.2 * x) + 0.1),
            ElementT(0.15 * std::sin(0.25 * (x + y))));
    } else {
        return ValueT(std::sin(0.2 * x) * std::cos(0.15 * y) + 0.05 * z);
    }
}

// A ball of voxels with a hole and a few inactive voxels, next to active and
// inactive tiles and the background
template<typename GridT>
typename GridT::Ptr
makeField(const math::Transform::Ptr& transform)
{
    using ValueT = typename GridT::ValueType;
    constexpr int dim = int(GridT::TreeType::LeafNodeType::DIM);
    typename GridT::Ptr grid = GridT::create(fieldValue<ValueT>(Coord(40, -30, 20)));
    grid->setTransform(transform);
    auto& tree = grid->tree();
    for (int x = -20; x <= 20; ++x) {
        for (int y = -20; y <= 20; ++y) {
            for (int z = -20; z <= 20; ++z) {
                if (x * x + y * y + z * z > 400) continue;
                if (x > -3 && x < 4 && y > 2 && y < 7) continue;
                const Coord xyz(x, y, z);
                if ((x + 2 * y + z) % 7 == 0) tree.setValueOff(xyz, fieldValue<ValueT>(xyz));
                else tree.setValueOn(xyz, fieldValue<ValueT>(xyz));
            }
        }
    }
    tree.addTile(1, Coord(3 * dim, 0, 0), fieldValue<ValueT>(Coord(-7, 3, 2)), true);
    tree.addTile(1, Coord(-4 * dim, 0, 0), fieldValue<ValueT>(Coord(5, -4, 1)), false);
    tree.addTile(1, Coord(0, 3 * dim, dim), fieldValue<ValueT>(Coord(2, 9, -6)), true);
    return grid;
}

// Index-space positions in and around the field, in order along a line and at random
std::vector<Vec3R>
makePositions()
{
    std::vector<Vec3R> positions;
    for (double x = -26.3; x < 26.0; x += 0.37) positions.emplace_back(x, 1.6 * std::sin(x), 0.3 * x);
    std::mt19937 generator(7);
    std::uniform_real_distribution<double> distribution(-26.0, 26.0);
    for (int i = 0; i < 2000; ++i) {
        const double x = distribution(generator);
        const double y = distribution(generator);
        const double z = distribution(generator);
        positions.emplace_back(x, y, z);
    }
    return positions;
}

template<typename GridT>
void
checkSameGrid(const GridT& expected, const GridT& grid)
{
    EXPECT_TRUE(expected.tree().hasSameTopology(grid.tree()));
    EXPECT_EQ(expected.activeVoxelCount(), grid.activeVoxelCount());
    tree::ValueAccessor<const typename GridT::TreeType> acc(grid.tree());
    // The value of an inactive voxel depends on which thread writes it last, since
    // overlapping bounding boxes sample the same output voxel
    for (auto iter = expected.cbeginValueOn(); iter; ++iter) {
        EXPECT_EQ(*iter, acc.getValue(iter.getCoord())) << " at " << iter.getCoord();
    }
}


////////////////////////////////////////


template<typename SamplerT, typename GridT>
void
checkTransformer(const GridT& input)
{
    const std::vector<tools::GridTransformer> transformers{
        // A large downscale, which halves the input with mipmaps first
        tools::GridTransformer(Vec3R(0), Vec3R(0.3), Vec3R(0), Vec3R(0)),
        tools::GridTransformer(Vec3R(1.5, -2.0, 0.5), Vec3R(1.7, 1.2, 0.8),
            Vec3R(0.3, 0.2, 0.1), Vec3R(0.5, -1.25, 2.0)),
        tools::GridTransformer(Vec3R(0), Vec3R(2.0), Vec3R(0), Vec3R(0.25, 0.0, 0.5)),
    };
    for (const tools::GridTransformer& transformer : transformers) {
        typename GridT::Ptr expected = GridT::create(input.background());
        transformer.transformGrid<AccessorOnlySampler<SamplerT>>(input, *expected);
        typename GridT::Ptr grid = GridT::create(input.background());
        transformer.transformGrid<SamplerT>(input, *grid);
        checkSameGrid(*expected, *grid);
    }

    // Into the index space of a grid with another transform
    typename GridT::Ptr expected = GridT::create(input.background());
    expected->setTransform(math::Transform::createLinearTransform(0.7));
    typename GridT::Ptr grid = expected->deepCopy();
    tools::resampleToMatch<AccessorOnlySampler<SamplerT>>(input, *expected);
    tools::resampleToMatch<SamplerT>(input, *grid);
    checkSameGrid(*expected, *grid);
}


template<typename GridT>
void
checkPointSample(const GridT& source, const math::Transform::Ptr& pointsTransform,
    const std::vector<Vec3R>& positions)
{
    using ValueT = typename GridT::ValueType;
    std::vector<Vec3f> worldPositions;
    for (const Vec3R& xyz : positions) worldPositions.emplace_back(source.indexToWorld(xyz));
    points::PointDataGrid::Ptr points =
        points::createPointDataGrid<points::NullCodec, points::PointDataGrid, Vec3f>(
            worldPositions, *pointsTransform);

    for (size_t order = 0; order <= 2; ++order) {
        const std::string expectedName = "expected" + std::to_string(order);
        const std::string name = "value" + std::to_string(order);
        points::sampleGrid(order, *points, source, expectedName, points::NullFilter(),
            AccessorOnlySampleWithRounding());
        points::sampleGrid(order, *points, source, name);

        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {
            auto expected = points::AttributeHandle<ValueT>::create(
                leaf->constAttributeArray(expectedName));
            auto values = points::AttributeHandle<ValueT>::create(leaf->constAttributeArray(name));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                EXPECT_EQ(expected->get(*iter), values->get(*iter));
            }
        }
    }

    if constexpr (!VecTraits<ValueT>::IsVec) {
        // Rounded to an integer attribute
        points::sampleGrid<points::PointDataGrid, GridT, int>(1, *points, source, "expectedInt",
            points::NullFilter(), AccessorOnlySampleWithRounding());
        points::sampleGrid<points::PointDataGrid, GridT, int>(1, *points, source, "valueInt");
        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {
            auto expected = points::AttributeHandle<int>::create(
                leaf->constAttributeArray("expectedInt"));
            auto values = points::AttributeHandle<int>::create(leaf->constAttributeArray("valueInt"));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                EXPECT_EQ(expected->get(*iter), values->get(*iter));
            }
        }
    }
}


// The restriction and prolongation of MultiResGrid, through a ValueAccessor
template<typename TreeT>
struct ReferenceMultiRes
{
    using ValueT = typename TreeT::ValueType;
    using AccessorT = tree::ValueAccessor<const TreeT>;

    static ValueT restrict(Coord ijk, const AccessorT& acc)
    {
        ijk <<= 1;
        ValueT v = 8*acc.getValue(ijk);
        v += 4*(acc.getValue(ijk.offsetBy(-1, 0, 0)) + acc.getValue(ijk.offsetBy( 1, 0, 0)) +
                acc.getValue(ijk.offsetBy( 0,-1, 0)) + acc.getValue(ijk.offsetBy( 0, 1, 0)) +
                acc.getValue(ijk.offsetBy( 0, 0,-1)) + acc.getValue(ijk.offsetBy( 0, 0, 1)));
        v += 2*(acc.getValue(ijk.offsetBy(-1,-1, 0)) + acc.getValue(ijk.offsetBy(-1, 1, 0)) +
                acc.getValue(ijk.offsetBy( 1,-1, 0)) + acc.getValue(ijk.offsetBy( 1, 1, 0)) +
                acc.getValue(ijk.offsetBy(-1, 0,-1)) + acc.getValue(ijk.offsetBy(-1, 0, 1)) +
                acc.getValue(ijk.offsetBy( 1, 0,-1)) + acc.getValue(ijk.offsetBy( 1, 0, 1)) +
                acc.getValue(ijk.offsetBy( 0,-1,-1)) + acc.getValue(ijk.offsetBy( 0,-1, 1)) +
                acc.getValue(ijk.offsetBy( 0, 1,-1)) + acc.getValue(ijk.offsetBy( 0, 1, 1)));
        for (int i=-1; i<=1; i+=2) {
            for (int j=-1; j<=1; j+=2) {
                for (int k=-1; k<=1; k+=2) v += acc.getValue(ijk.offsetBy(i,j,k));
            }
        }
        v *= ValueT(1.0f/64.0f);
        return v;
    }

    static ValueT prolongate(const Coord& ijk, const AccessorT& acc)
    {
        switch ( (ijk[0] & 1) | ((ijk[1] & 1) << 1) | ((ijk[2] & 1) << 2) ) {
        case 0:
            return acc.getValue(ijk>>1);
        case 1:
            return ValueT(0.5)*(acc.getValue(ijk.offsetBy(-1,0,0)>>1) +
                                acc.getValue(ijk.offsetBy( 1,0,0)>>1));
        case 2:
            return ValueT(0.5)*(acc.getValue(ijk.offsetBy(0,-1,0)>>1) +
                                acc.getValue(ijk.offsetBy(0, 1,0)>>1));
        case 3:
            return ValueT(0.25)*(acc.getValue(ijk.offsetBy(-1,-1,0)>>1) +
                                 acc.getValue(ijk.offsetBy(-1, 1,0)>>1) +
                                 acc.getValue(ijk.offsetBy( 1,-1,0)>>1) +
                                 acc.getValue(ijk.offsetBy( 1, 1,0)>>1));
        case 4:
            return ValueT(0.5)*(acc.getValue(ijk.offsetBy(0,0,-1)>>1) +
                                acc.getValue(ijk.offsetBy(0,0, 1)>>1));
        case 5:
            return ValueT(0.25)*(acc.getValue(ijk.offsetBy(-1,0,-1)>>1) +
                                 acc.getValue(ijk.offsetBy(-1,0, 1)>>1) +
                                 acc.getValue(ijk.offsetBy( 1,0,-1)>>1) +
                                 acc.getValue(ijk.offsetBy( 1,0, 1)>>1));
        case 6:
            return ValueT(0.25)*(acc.getValue(ijk.offsetBy(0,-1,-1)>>1) +
                                 acc.getValue(ijk.offsetBy(0,-1, 1)>>1) +
                                 acc.getValue(ijk.offsetBy(0, 1,-1)>>1) +
                                 acc.getValue(ijk.offsetBy(0, 1, 1)>>1));
        }
        ValueT v = zeroVal<ValueT>();
        for (int i=-1; i<=1; i+=2) {
            for (int j=-1; j<=1; j+=2) {
                for (int k=-1; k<=1; k+=2) v += acc.getValue(ijk.offsetBy(i,j,k)>>1);
            }
        }
        return ValueT(0.125) * v;
    }
};

template<typename TreeT>
void
checkMultiResGrid(const Grid<TreeT>& grid)
{
    using ValueT = typename TreeT::ValueType;
    using ReferenceT = ReferenceMultiRes<TreeT>;
    using AccessorT = typename ReferenceT::AccessorT;

    tools::MultiResGrid<TreeT> mrg(/*levels=*/4, grid);

    // Each level is restricted from the level above it
    for (size_t level = 1; level < mrg.numLevels(); ++level) {
        const AccessorT fine(mrg.constTree(level - 1));
        size_t count = 0;
        for (auto iter = mrg.constTree(level).cbeginValueOn(); iter; ++iter, ++count) {
            EXPECT_EQ(ReferenceT::restrict(iter.getCoord(), fine), *iter);
            EXPECT_EQ(*iter, mrg.restrictVoxel(iter.getCoord(), level));
        }
        EXPECT_TRUE(count > 0);
    }

    // Prolongation of single voxels and of all the active voxels of a level
    for (size_t level = 0; level + 1 < mrg.numLevels(); ++level) {
        const AccessorT coarse(mrg.constTree(level + 1));
        std::vector<ValueT> expected;
        for (auto iter = mrg.constTree(level).cbeginValueOn(); iter; ++iter) {
            expected.push_back(ReferenceT::prolongate(iter.getCoord(), coarse));
            EXPECT_EQ(expected.back(), mrg.prolongateVoxel(iter.getCoord(), level));
        }
        mrg.prolongateActiveVoxels(level);
        size_t n = 0;
        for (auto iter = mrg.constTree(level).cbeginValueOn(); iter; ++iter) {
            EXPECT_EQ(expected[n++], *iter);
        }
    }

    // A level between two levels, from samples of both
    for (float level : {0.5f, 1.25f, 2.75f}) {
        typename Grid<TreeT>::Ptr fraction = mrg.template createGrid<1>(level);
        const AccessorT acc0(mrg.constTree(size_t(std::floor(level))));
        const AccessorT acc1(mrg.constTree(size_t(std::ceil(level))));
        const float b = math::FractionalPart(level), a = 1.0f - b;
        const float scale0 = math::Pow( 2.0f, b );
        const float scale1 = math::Pow( 2.0f,-a );
        size_t count = 0;
        for (auto iter = fraction->tree().cbeginValueOn(); iter; ++iter, ++count) {
            const Vec3R xyz = Vec3R(iter.getCoord().data());
            const ValueT v0 = tools::Sampler<1>::sample(acc0, xyz * scale0);
            const ValueT v1 = tools::Sampler<1>::sample(acc1, xyz * scale1);
            // The compiler may fuse the multiply and add differently here and in MultiResGrid
            EXPECT_NEAR(ValueT(a*v0 + b*v1), *iter, 1e-5);
        }
        EXPECT_TRUE(count > 0);
    }
}

} // unnamed namespace


////////////////////////////////////////


TEST_F(TestNeighborSampling, testGridTransformer)
{
    // The resampler writes overlapping output voxels from separate threads, and the
    // value that survives depends on the order, so use one thread to compare exactly
    tbb::global_control singleThread(tbb::global_control::max_allowed_parallelism, 1);

    const math::Transform::Ptr transform = math::Transform::createLinearTransform(0.5);
    const FloatGrid::Ptr scalarGrid = makeField<FloatGrid>(transform);
    const Vec3SGrid::Ptr vectorGrid = makeField<Vec3SGrid>(transform);

    checkTransformer<tools::PointSampler>(*scalarGrid);
    checkTransformer<tools::BoxSampler>(*scalarGrid);
    checkTransformer<tools::QuadraticSampler>(*scalarGrid);
    checkTransformer<tools::BoxSampler>(*vectorGrid);
    checkTransformer<tools::StaggeredBoxSampler>(*vectorGrid);
}


TEST_F(TestNeighborSampling, testVolumeAdvect)
{
    const math::Transform::Ptr transform = math::Transform::createLinearTransform(0.5);
    const FloatGrid::Ptr density = makeField<FloatGrid>(transform);
    const Vec3SGrid::Ptr velocity = makeField<Vec3SGrid>(transform);

    using AdvectionT = tools::VolumeAdvection<Vec3SGrid>;
    using SamplerT = tools::Sampler<1, false>;
    AdvectionT advection(*velocity);
    const double maxVelocity = advection.getMaxVelocity();
    EXPECT_TRUE(maxVelocity > 0.0);

    const std::vector<std::pair<tools::Scheme::SemiLagrangian, tools::Scheme::Limiter>> schemes{
        {tools::Scheme::SEMI, tools::Scheme::NO_LIMITER},
        {tools::Scheme::MID, tools::Scheme::NO_LIMITER},
        {tools::Scheme::RK3, tools::Scheme::NO_LIMITER},
        {tools::Scheme::RK4, tools::Scheme::NO_LIMITER},
        {tools::Scheme::MAC, tools::Scheme::CLAMP},
        {tools::Scheme::BFECC, tools::Scheme::REVERT},
    };
    // Time steps that move the density about half a voxel and four voxels
    for (double cfl : {0.5, 4.0}) {
        const double dt = cfl * density->voxelSize()[0] / maxVelocity;
        for (const auto& scheme : schemes) {
            advection.setIntegrator(scheme.first);
            advection.setLimiter(scheme.second);
            const FloatGrid::Ptr expected =
                advection.advect<FloatGrid, AccessorOnlySampler<SamplerT>>(*density, dt);
            const FloatGrid::Ptr grid = advection.advect<FloatGrid, SamplerT>(*density, dt);
            checkSameGrid(*expected, *grid);
        }
    }
}


TEST_F(TestNeighborSampling, testVelocityFields)
{
    const math::Transform::Ptr transform = math::Transform::createLinearTransform(0.5);
    const Vec3SGrid::Ptr velocity = makeField<Vec3SGrid>(transform);
    using NeighborsT = tree::NeighborAccessor<const Vec3STree>;
    const std::vector<Vec3R> positions = makePositions();

    {
        const tools::DiscreteField<Vec3SGrid> expected(*velocity);
        const tools::DiscreteField<Vec3SGrid, tools::BoxSampler, NeighborsT> field(*velocity);
        for (const Vec3R& xyz : positions) {
            const Vec3d world = velocity->indexToWorld(xyz);
            EXPECT_EQ(expected(world, 0.0f), field(world, 0.0f));
            EXPECT_EQ(expected(Coord::round(xyz), 0.0f), field(Coord::round(xyz), 0.0f));
        }
    }

    const tools::VelocityIntegrator<Vec3SGrid, false, 1> expected(*velocity);
    const tools::VelocityIntegrator<Vec3SGrid, false, 1, NeighborsT> integrator(*velocity);
    const tools::VelocityIntegrator<Vec3SGrid, true, 2> expectedStaggered(*velocity);
    const tools::VelocityIntegrator<Vec3SGrid, true, 2, NeighborsT> staggered(*velocity);
    for (const Vec3R& xyz : positions) {
        const Vec3d world = velocity->indexToWorld(xyz);
        Vec3d expectedWorld = world, integrated = world;
        expected.rungeKutta<1>(0.5f, expectedWorld);
        integrator.rungeKutta<1>(0.5f, integrated);
        EXPECT_EQ(expectedWorld, integrated);
        expected.rungeKutta<4>(0.5f, expectedWorld);
        integrator.rungeKutta<4>(0.5f, integrated);
        EXPECT_EQ(expectedWorld, integrated);
        expectedStaggered.rungeKutta<2>(0.5f, expectedWorld);
        staggered.rungeKutta<2>(0.5f, integrated);
        EXPECT_EQ(expectedWorld, integrated);
    }
}


TEST_F(TestNeighborSampling, testPointSample)
{
    const math::Transform::Ptr transform = math::Transform::createLinearTransform(0.5);
    const FloatGrid::Ptr scalarGrid = makeField<FloatGrid>(transform);
    const Vec3SGrid::Ptr vectorGrid = makeField<Vec3SGrid>(transform);
    const Vec3SGrid::Ptr staggeredGrid = vectorGrid->deepCopy();
    staggeredGrid->setGridClass(GRID_STAGGERED);
    const std::vector<Vec3R> positions = makePositions();

    // Points in the transform of the grids, and in another one
    for (const math::Transform::Ptr& pointsTransform :
            {transform, math::Transform::createLinearTransform(1.3)}) {
        checkPointSample(*scalarGrid, pointsTransform, positions);
        checkPointSample(*vectorGrid, pointsTransform, positions);
        checkPointSample(*staggeredGrid, pointsTransform, positions);
    }
}


TEST_F(TestNeighborSampling, testLevelSetMorph)
{
    // The target and mask sampling of LevelSetMorphing in another transform
    const math::Transform::Ptr transform = math::Transform::createLinearTransform(0.5);
    const FloatGrid::Ptr target = makeField<FloatGrid>(transform);
    const math::Transform& xform = target->transform();
    const FloatGrid::ConstAccessor acc = target->getConstAccessor();
    const tools::GridSampler<FloatGrid::ConstAccessor, tools::BoxSampler> expected(acc, xform);
    tree::NeighborAccessor<const FloatTree> neighbors(target->tree());
    for (const Vec3R& xyz : makePositions()) {
        const Vec3d world = xform.indexToWorld(xyz);
        EXPECT_EQ(expected.wsSample(world),
            tools::BoxSampler::sample(neighbors, xform.worldToIndex(world)));
    }

    // Morph a sphere towards a sphere in another transform
    FloatGrid::Ptr source = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(0.0f), 0.25f);
    const FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(1.0f), 0.3f);
    tools::LevelSetMorphing<FloatGrid> morph(*source, *sphere);
    morph.setAlphaMask(*sphere);
    morph.advect(0.0f, 0.25f);
    for (auto iter = source->cbeginValueOn(); iter; ++iter) EXPECT_TRUE(std::isfinite(*iter));
}


TEST_F(TestNeighborSampling, testMultiResGrid)
{
    const math::Transform::Ptr transform = math::Transform::createLinearTransform(0.5);
    checkMultiResGrid(*makeField<FloatGrid>(transform));
    // Smaller internal nodes. MultiResGrid dilates leaf masks, which needs 8^3 leaves.
    checkMultiResGrid(*makeField<Grid<tree::Tree4<float, 4, 3, 3>::Type>>(transform));
}
