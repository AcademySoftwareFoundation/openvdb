// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/VelocityFields.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>

namespace {

// Has the same 8^3 leaves as openvdb::FloatGrid but is a distinct type, because its internal
// nodes differ. The specializations below keep it on the math:: stencils, which makes it the
// reference that the leaf stencil is compared against.
using ReferenceGrid = openvdb::Grid<openvdb::tree::Tree4<float, 4, 5, 3>::Type>;

} // namespace

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {
namespace lstrack {
namespace internal {
template<>
struct LeafStencil<ReferenceGrid, math::SevenPointStencil<ReferenceGrid>, true>
    : LeafStencil<ReferenceGrid, math::SevenPointStencil<ReferenceGrid>, false> {};
template<>
struct LeafStencil<ReferenceGrid, math::ThirteenPointStencil<ReferenceGrid>, true>
    : LeafStencil<ReferenceGrid, math::ThirteenPointStencil<ReferenceGrid>, false> {};
template<>
struct LeafStencil<ReferenceGrid, math::NineteenPointStencil<ReferenceGrid>, true>
    : LeafStencil<ReferenceGrid, math::NineteenPointStencil<ReferenceGrid>, false> {};
} // namespace internal
} // namespace lstrack
} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


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

namespace {

using openvdb::Coord;
using openvdb::FloatGrid;
using openvdb::Index64;
namespace math = openvdb::math;
namespace tools = openvdb::tools;
namespace lstrack = openvdb::tools::lstrack::internal;

template<typename ValueT>
bool bitsDiffer(const ValueT& a, const ValueT& b)
{
    return std::memcmp(&a, &b, sizeof(ValueT)) != 0;
}

// Whether two stencils read different values at the offset (I, J, K).
template<int I, int J, int K, typename StencilA, typename StencilB>
bool valueDiffers(const StencilA& a, const StencilB& b)
{
    return bitsDiffer(a.template getValue<I, J, K>(), b.template getValue<I, J, K>());
}

// Number of offsets along the axes, up to distance D, where two stencils read different values.
template<int D, typename StencilA, typename StencilB>
Index64 axisMismatches(const StencilA& a, const StencilB& b)
{
    Index64 n = 0;
    n += valueDiffers< D, 0, 0>(a, b);
    n += valueDiffers<-D, 0, 0>(a, b);
    n += valueDiffers< 0, D, 0>(a, b);
    n += valueDiffers< 0,-D, 0>(a, b);
    n += valueDiffers< 0, 0, D>(a, b);
    n += valueDiffers< 0, 0,-D>(a, b);
    if constexpr (D > 1) n += axisMismatches<D - 1>(a, b);
    return n;
}

// Grid of random values whose leaves have every kind of face neighbor: another leaf, the
// background, and active and inactive tiles. It spans negative coordinates.
template<typename GridT>
typename GridT::Ptr makeRandomGrid()
{
    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using NodeT = typename TreeT::RootNodeType::ChildNodeType::ChildNodeType;
    using ValueT = typename GridT::ValueType;
    const int leafDim = static_cast<int>(LeafT::DIM);
    const int nodeDim = static_cast<int>(NodeT::DIM);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> randomValue(-1.0, 1.0);
    std::bernoulli_distribution isActive(0.5), isHole(0.25);

    // Create a new grid with background value 7
    typename GridT::Ptr grid = GridT::create(ValueT(7));
    TreeT& tree = grid->tree();

    // All leaves are in the node at (-nodeDim)^3. Each of its six faces touches a tile with its
    // own value, so reading the wrong tile is detected. The tiles are as big as the node, so they
    // are added to its parent, one level up.
    const Coord node(-nodeDim);
    int tileValue = 11;
    for (int axis = 0; axis < 3; ++axis) {
        for (const int side : { -1, 1 }) {
            Coord tile = node;
            tile[axis] += side * nodeDim;
            tree.addTile(NodeT::LEVEL + 1, tile, ValueT(tileValue++), /*active=*/side < 0);
        }
    }

    // Two blocks of leaves, with holes, in opposite corners of the node, so they touch the tiles.
    const int blockDim = 4 * leafDim;
    for (const Coord& corner : { node, node.offsetBy(nodeDim - blockDim) }) {
        for (int i = 0; i < blockDim; i += leafDim) {
            for (int j = 0; j < blockDim; j += leafDim) {
                for (int k = 0; k < blockDim; k += leafDim) {
                    if (!isHole(rng)) tree.touchLeaf(corner.offsetBy(i, j, k));
                }
            }
        }
    }

    for (auto leaf = tree.beginLeaf(); leaf; ++leaf) {
        for (openvdb::Index n = 0; n < LeafT::SIZE; ++n) {
            leaf->setValueOnly(n, ValueT(randomValue(rng)));
            leaf->setActiveState(n, isActive(rng));
        }
    }
    return grid;
}

// Asserts that the leaf stencil of the given radius replaces MathStencilT, and that both read
// identical values at every voxel of every leaf of the grid, active or not.
template<typename GridT, typename MathStencilT, int Radius>
void checkGather(const GridT& grid)
{
    using StarT = lstrack::LeafStarStencil<GridT, Radius>;
    static_assert(std::is_same_v<typename lstrack::LeafStencil<GridT, MathStencilT>::Type, StarT>,
        "the selector must replace this stencil");

    const auto mismatches = [](const StarT& star, const MathStencilT& reference) {
        return Index64(star.getCenterCoord() != reference.getCenterCoord())
            + bitsDiffer(star.getValue(), reference.getValue())
            + valueDiffers<0, 0, 0>(star, reference)
            + axisMismatches<Radius>(star, reference);
    };

    MathStencilT reference(grid);
    StarT star(grid);
    Index64 voxels = 0, total = 0;
    for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
        star.setLeaf(*leaf);
        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel, ++voxels) {
            reference.moveTo(voxel);
            star.moveTo(voxel); // the iterator overload
            total += mismatches(star, reference);
            star.moveTo(voxel.getCoord(), *voxel); // the coordinate overload
            total += mismatches(star, reference);
        }
    }
    EXPECT_GT(voxels, Index64(0));
    EXPECT_EQ(total, Index64(0));
}

template<typename GridT>
void checkAllGathers()
{
    const typename GridT::Ptr grid = makeRandomGrid<GridT>();
    checkGather<GridT, math::SevenPointStencil<GridT>, 1>(*grid);
    checkGather<GridT, math::ThirteenPointStencil<GridT>, 2>(*grid);
    checkGather<GridT, math::NineteenPointStencil<GridT>, 3>(*grid);
}

// A sphere that is no longer a signed distance field, so the tracker has work to do.
template<typename GridT>
typename GridT::Ptr makeDistortedSphere()
{
    typename GridT::Ptr grid = tools::createLevelSetSphere<GridT>(
        0.15f, openvdb::Vec3f(0.35f), 1.0f / 64.0f);
    for (auto voxel = grid->beginValueOn(); voxel; ++voxel) {
        const Coord c = voxel.getCoord();
        const double distortion =
            std::sin(0.7 * c.x()) * std::cos(0.9 * c.y()) * std::sin(1.1 * c.z());
        voxel.setValue(float(double(*voxel) * (1.0 + 0.2 * distortion)));
    }
    return grid;
}

// Number of voxels in the leaves of a whose active state differs in b, or whose value differs
// by more than 1e-5 of a voxel. Exact equality is not expected here: compilers can fuse
// multiply-adds differently in the two instantiations, which changes the last bits.
template<typename GridT>
Index64 voxelMismatches(const FloatGrid& a, const GridT& b)
{
    const double tolerance = 1e-5 * a.voxelSize()[0];
    Index64 n = 0;
    auto accessor = b.getConstAccessor();
    for (auto leaf = a.tree().cbeginLeaf(); leaf; ++leaf) {
        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel) {
            const Coord c = voxel.getCoord();
            n += std::abs(double(*voxel) - double(accessor.getValue(c))) > tolerance
                || voxel.isValueOn() != accessor.isValueOn(c);
        }
    }
    return n;
}

// Asserts that both grids hold the same level set, and that it is no longer the input sphere,
// which would make the comparison trivially true.
void expectSameLevelSet(const FloatGrid& star, const ReferenceGrid& reference)
{
    EXPECT_EQ(star.tree().leafCount(), reference.tree().leafCount());
    EXPECT_EQ(star.activeVoxelCount(), reference.activeVoxelCount());
    EXPECT_EQ(voxelMismatches(star, reference), Index64(0));
    EXPECT_GT(voxelMismatches(star, *makeDistortedSphere<FloatGrid>()), Index64(0));
}

template<typename GridT>
void normalize(GridT& grid, math::BiasedGradientScheme spatial,
    math::TemporalIntegrationScheme temporal, bool useMask)
{
    using MaskTreeT = typename GridT::TreeType::template ValueConverter<openvdb::ValueMask>::Type;
    tools::LevelSetTracker<GridT> tracker(grid);
    tracker.setSpatialScheme(spatial);
    tracker.setTemporalScheme(temporal);
    tracker.setNormCount(2);
    if (useMask) {
        MaskTreeT mask;
        for (auto voxel = grid.cbeginValueOn(); voxel; ++voxel) {
            if (voxel.getCoord().x() % 3 != 0) mask.setValueOn(voxel.getCoord());
        }
        tracker.template normalize<MaskTreeT>(&mask);
    } else {
        tracker.normalize();
    }
}

// Returns the number of CFL steps taken.
template<typename GridT>
size_t advect(GridT& grid, math::BiasedGradientScheme spatial,
    math::TemporalIntegrationScheme temporal)
{
    using FieldT = tools::EnrightField<float>;
    FieldT field;
    tools::LevelSetAdvection<GridT, FieldT> advection(grid, field);
    advection.setSpatialScheme(spatial);
    advection.setTemporalScheme(temporal);
    advection.setTrackerSpatialScheme(spatial);
    return advection.advect(0.0f, 0.05f);
}

void checkNormalize(math::BiasedGradientScheme spatial,
    math::TemporalIntegrationScheme temporal, bool useMask)
{
    SCOPED_TRACE(useMask ? "normalize with mask" : "normalize");
    FloatGrid::Ptr star = makeDistortedSphere<FloatGrid>();
    ReferenceGrid::Ptr reference = makeDistortedSphere<ReferenceGrid>();
    normalize(*star, spatial, temporal, useMask);
    normalize(*reference, spatial, temporal, useMask);
    expectSameLevelSet(*star, *reference);
}

void checkAdvect(math::BiasedGradientScheme spatial, math::TemporalIntegrationScheme temporal)
{
    SCOPED_TRACE("advect");
    FloatGrid::Ptr star = makeDistortedSphere<FloatGrid>();
    ReferenceGrid::Ptr reference = makeDistortedSphere<ReferenceGrid>();
    const size_t steps = advect(*star, spatial, temporal);
    EXPECT_GT(steps, size_t(0));
    EXPECT_EQ(steps, advect(*reference, spatial, temporal));
    expectSameLevelSet(*star, *reference);
}

} // namespace

// Compares the neighbor lookup of LeafStarStencil with the math:: stencils it replaces. The
// risk is reading the wrong buffer or offset near a leaf boundary, so the grids have every kind
// of face neighbor (see makeRandomGrid), and 4^3 leaves make offsets of +-3 cross most of a leaf.
TEST_F(TestStencils, testLeafStarStencilGather)
{
    using lstrack::LeafStencil;
    using FloatGrid4 = openvdb::Grid<openvdb::tree::Tree4<float, 5, 4, 2>::Type>; // 4^3 leaves
    using FloatGrid2 = openvdb::Grid<openvdb::tree::Tree4<float, 5, 4, 1>::Type>; // 2^3 leaves

    // Leaves narrower than 4 voxels: offsets of +-3 would reach past the face neighbors.
    static_assert(std::is_same_v<
        LeafStencil<FloatGrid2, math::SevenPointStencil<FloatGrid2>>::Type,
        math::SevenPointStencil<FloatGrid2>>, "narrow leaves keep the math:: stencil");
    // Only star stencils can be replaced.
    static_assert(std::is_same_v<
        LeafStencil<FloatGrid, math::BoxStencil<FloatGrid>>::Type,
        math::BoxStencil<FloatGrid>>, "other stencils pass through");

    { SCOPED_TRACE("float, 8^3 leaves");  checkAllGathers<FloatGrid>(); }
    { SCOPED_TRACE("double, 8^3 leaves"); checkAllGathers<openvdb::DoubleGrid>(); }
    { SCOPED_TRACE("float, 4^3 leaves");  checkAllGathers<FloatGrid4>(); }
}

// Compares LevelSetTracker and LevelSetAdvection on a FloatGrid, which uses the leaf stencil,
// and on a ReferenceGrid, which uses the math:: stencils, for all schemes. This catches what the
// test above cannot see, such as a kernel that does not call setLeaf(), or that reads the wrong
// buffer between RK stages.
TEST_F(TestStencils, testLeafStarStencilTrackerAndAdvection)
{
    static_assert(std::is_same_v<
        lstrack::LeafStencil<ReferenceGrid, math::NineteenPointStencil<ReferenceGrid>>::Type,
        math::NineteenPointStencil<ReferenceGrid>>, "ReferenceGrid uses the math:: stencil");

    for (const math::BiasedGradientScheme spatial : { math::FIRST_BIAS, math::SECOND_BIAS,
        math::THIRD_BIAS, math::WENO5_BIAS, math::HJWENO5_BIAS })
    {
        for (const math::TemporalIntegrationScheme temporal :
            { math::TVD_RK1, math::TVD_RK2, math::TVD_RK3 })
        {
            SCOPED_TRACE(math::biasedGradientSchemeToString(spatial) + ", "
                + math::temporalIntegrationSchemeToString(temporal));

            checkNormalize(spatial, temporal, /*useMask=*/false);
            checkNormalize(spatial, temporal, /*useMask=*/true);
            checkAdvect(spatial, temporal);
        }
    }
}
