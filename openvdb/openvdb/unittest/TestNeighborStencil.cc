// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Maps.h>
#include <openvdb/math/Operators.h>
#include <openvdb/math/Stencils.h>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <typeinfo>

using namespace openvdb;

namespace {

template<typename ValueT> ValueT fieldValue(int x, int y, int z);

template<>
float fieldValue<float>(int x, int y, int z)
{
    return float(std::sin(0.3 * x) * std::cos(0.2 * y) + 0.01 * z * z - 0.05 * x);
}

template<>
Vec3f fieldValue<Vec3f>(int x, int y, int z)
{
    return Vec3f(float(std::sin(0.3 * x + 0.1 * y)), float(std::cos(0.2 * y - 0.1 * z)),
        float(0.01 * x * z));
}

// Smooth values in a ball of voxels with inactive gaps, leaf-sized tiles of
// both states beside it, and background
template<typename TreeT>
void fillField(TreeT& tree)
{
    using ValueT = typename TreeT::ValueType;
    constexpr int dim = int(TreeT::LeafNodeType::DIM);
    for (int x = -10; x <= 10; ++x) {
        for (int y = -10; y <= 10; ++y) {
            for (int z = -10; z <= 10; ++z) {
                if (x * x + y * y + z * z <= 100 && (x * 7 + y * 3 + z) % 5 != 0) {
                    tree.setValueOn(Coord(x, y, z), fieldValue<ValueT>(x, y, z));
                }
            }
        }
    }
    for (int i = 0; i < 12; ++i) {
        const int side = (i & 2) ? 1 : -1;
        const Coord xyz(side * (18 + (i % 3) * dim), (i / 3) * dim - 16, (i % 2) * dim);
        tree.addTile(1, xyz, fieldValue<ValueT>(i, -i, 2 * i), (i & 1) != 0);
    }
}

// The stencil interface that the operators use, read through a ValueAccessor
template<typename TreeT>
class AccessorStencil
{
public:
    using ValueType = typename TreeT::ValueType;

    explicit AccessorStencil(const TreeT& tree): mAccessor(tree) {}

    void moveTo(const Coord& ijk) { mCenter = ijk; }

    template<int I, int J, int K>
    const ValueType& getValue() const { return mAccessor.getValue(mCenter.offsetBy(I, J, K)); }

    const Coord& getCenterCoord() const { return mCenter; }

private:
    tree::ValueAccessor<const TreeT> mAccessor;
    Coord mCenter;
};

bool isSame(double expected, double actual)
{
    if (std::isnan(expected) || std::isnan(actual)) return std::isnan(expected) && std::isnan(actual);
    return expected == actual;
}

template<typename T>
bool isSame(const math::Vec3<T>& expected, const math::Vec3<T>& actual)
{
    return isSame(expected[0], actual[0]) && isSame(expected[1], actual[1]) &&
        isSame(expected[2], actual[2]);
}

// The stencil overload of an operator through a NeighborStencil must give the
// result through a stencil that reads with a ValueAccessor, bit for bit, and
// the result of the accessor overload if @a sameAsAccessor. Centers are the
// active voxels by leaf iterator, and voxels of leaves, tiles and the
// background by coordinate.
template<typename TreeT, typename StencilResultT, typename AccessorResultT>
void checkOperator(const TreeT& tree, const char* name, const StencilResultT& stencilResult,
    const AccessorResultT& accessorResult, bool sameAsAccessor)
{
    tree::ValueAccessor<const TreeT> accessor(tree);
    AccessorStencil<TreeT> reference(tree);
    math::NeighborStencil<const TreeT> stencil(tree);

    auto check = [&](const Coord& xyz) {
        ASSERT_EQ(xyz, stencil.getCenterCoord());
        reference.moveTo(xyz);
        const auto actual = stencilResult(stencil);
        const auto expected = stencilResult(reference);
        ASSERT_TRUE(isSame(expected, actual))
            << name << " at " << xyz << ": " << expected << " and " << actual;
        if (sameAsAccessor) {
            const auto expectedFromAccessor = accessorResult(accessor, xyz);
            ASSERT_TRUE(isSame(expectedFromAccessor, actual)) << name << " accessor overload at "
                << xyz << ": " << expectedFromAccessor << " and " << actual;
        }
    };

    for (auto leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        for (auto iter = leafIter->cbeginValueOn(); iter; ++iter) {
            stencil.moveTo(iter);
            check(iter.getCoord());
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
    for (int x = -24; x <= 24; x += 4) {
        for (int y = -24; y <= 24; y += 4) {
            for (int z = -24; z <= 24; z += 4) {
                stencil.moveTo(Coord(x, y, z));
                check(Coord(x, y, z));
                if (::testing::Test::HasFatalFailure()) return;
            }
        }
    }
}

template<typename OpT, typename TreeT>
void checkIndex(const TreeT& tree, bool sameAsAccessor = true)
{
    checkOperator(tree, typeid(OpT).name(),
        [](const auto& stencil) { return OpT::result(stencil); },
        [](const auto& accessor, const Coord& xyz) { return OpT::result(accessor, xyz); },
        sameAsAccessor);
}

template<typename OpT, typename TreeT>
void checkIndexBiased(const TreeT& tree)
{
    for (const Vec3f& velocity : {Vec3f(0.3f, -1.0f, 0.7f), Vec3f(-0.3f, 1.0f, -0.7f)}) {
        checkOperator(tree, typeid(OpT).name(),
            [&](const auto& stencil) { return OpT::result(stencil, velocity); },
            [&](const auto& accessor, const Coord& xyz) { return OpT::result(accessor, xyz, velocity); },
            true);
    }
}

// The result is returned as (alpha, beta, nonzero gradient)
template<typename OpT, typename TreeT>
void checkIndexMeanCurvature(const TreeT& tree, bool sameAsAccessor = true)
{
    checkOperator(tree, typeid(OpT).name(),
        [](const auto& stencil) {
            float alpha = 0.0f, beta = 0.0f;
            const bool nonzero = OpT::result(stencil, alpha, beta);
            return Vec3d(alpha, beta, nonzero ? 1.0 : 0.0);
        },
        [](const auto& accessor, const Coord& xyz) {
            float alpha = 0.0f, beta = 0.0f;
            const bool nonzero = OpT::result(accessor, xyz, alpha, beta);
            return Vec3d(alpha, beta, nonzero ? 1.0 : 0.0);
        },
        sameAsAccessor);
}

template<typename OpT, typename TreeT, typename MapT>
void checkMap(const TreeT& tree, const MapT& map, bool sameAsAccessor = true)
{
    checkOperator(tree, typeid(OpT).name(),
        [&](const auto& stencil) { return OpT::result(map, stencil); },
        [&](const auto& accessor, const Coord& xyz) { return OpT::result(map, accessor, xyz); },
        sameAsAccessor);
}

template<typename OpT, typename TreeT, typename MapT>
void checkMapBiased(const TreeT& tree, const MapT& map)
{
    for (const Vec3f& velocity : {Vec3f(0.3f, -1.0f, 0.7f), Vec3f(-0.3f, 1.0f, -0.7f)}) {
        checkOperator(tree, typeid(OpT).name(),
            [&](const auto& stencil) { return OpT::result(map, stencil, velocity); },
            [&](const auto& accessor, const Coord& xyz) {
                return OpT::result(map, accessor, xyz, velocity);
            },
            true);
    }
}

template<typename OpT, typename TreeT, typename MapT>
void checkMapNormGrad(const TreeT& tree, const MapT& map, bool sameAsAccessor = true)
{
    checkOperator(tree, typeid(OpT).name(),
        [&](const auto& stencil) { return OpT::normGrad(map, stencil); },
        [&](const auto& accessor, const Coord& xyz) { return OpT::normGrad(map, accessor, xyz); },
        sameAsAccessor);
}

// The accessor and stencil overloads of the fourth and sixth order cross
// derivatives in math/FiniteDifference.h add the same terms in another order,
// so the operators that use them are only compared with the reference stencil.
constexpr bool CrossOrderDiffers = false;

template<typename TreeT, typename VectorTreeT>
void checkIndexOperators(const TreeT& tree, const VectorTreeT& vectorTree)
{
    using namespace math;

    checkIndex<ISGradient<CD_2NDT>>(tree);
    checkIndex<ISGradient<CD_2ND>>(tree);
    checkIndex<ISGradient<CD_4TH>>(tree);
    checkIndex<ISGradient<CD_6TH>>(tree);
    checkIndex<ISGradient<FD_1ST>>(tree);
    checkIndex<ISGradient<FD_2ND>>(tree);
    checkIndex<ISGradient<FD_3RD>>(tree);
    checkIndex<ISGradient<BD_1ST>>(tree);
    checkIndex<ISGradient<BD_2ND>>(tree);
    checkIndex<ISGradient<BD_3RD>>(tree);
    checkIndex<ISGradient<FD_WENO5>>(tree);
    checkIndex<ISGradient<BD_WENO5>>(tree);
    checkIndex<ISGradient<FD_HJWENO5>>(tree);
    checkIndex<ISGradient<BD_HJWENO5>>(tree);

    checkIndexBiased<ISGradientBiased<FIRST_BIAS, Vec3f>>(tree);
    checkIndexBiased<ISGradientBiased<SECOND_BIAS, Vec3f>>(tree);
    checkIndexBiased<ISGradientBiased<THIRD_BIAS, Vec3f>>(tree);
    checkIndexBiased<ISGradientBiased<WENO5_BIAS, Vec3f>>(tree);
    checkIndexBiased<ISGradientBiased<HJWENO5_BIAS, Vec3f>>(tree);

    checkIndex<ISGradientNormSqrd<FIRST_BIAS>>(tree);
    checkIndex<ISGradientNormSqrd<SECOND_BIAS>>(tree);
    checkIndex<ISGradientNormSqrd<THIRD_BIAS>>(tree);
    checkIndex<ISGradientNormSqrd<WENO5_BIAS>>(tree);
    checkIndex<ISGradientNormSqrd<HJWENO5_BIAS>>(tree);

    checkIndex<ISLaplacian<CD_SECOND>>(tree);
    checkIndex<ISLaplacian<CD_FOURTH>>(tree);
    checkIndex<ISLaplacian<CD_SIXTH>>(tree);

    checkIndexMeanCurvature<ISMeanCurvature<CD_SECOND, CD_2ND>>(tree);
    checkIndexMeanCurvature<ISMeanCurvature<CD_FOURTH, CD_4TH>>(tree, CrossOrderDiffers);
    checkIndexMeanCurvature<ISMeanCurvature<CD_SIXTH, CD_6TH>>(tree, CrossOrderDiffers);

    checkIndex<ISDivergence<CD_2NDT>>(vectorTree);
    checkIndex<ISDivergence<CD_2ND>>(vectorTree);
    checkIndex<ISDivergence<CD_4TH>>(vectorTree);
    checkIndex<ISDivergence<CD_6TH>>(vectorTree);
    checkIndex<ISDivergence<FD_1ST>>(vectorTree);
    checkIndex<ISDivergence<FD_2ND>>(vectorTree);
    checkIndex<ISDivergence<FD_3RD>>(vectorTree);
    checkIndex<ISDivergence<BD_1ST>>(vectorTree);
    checkIndex<ISDivergence<BD_2ND>>(vectorTree);
    checkIndex<ISDivergence<BD_3RD>>(vectorTree);

    checkIndex<ISCurl<CD_2NDT>>(vectorTree);
    checkIndex<ISCurl<CD_2ND>>(vectorTree);
    checkIndex<ISCurl<CD_4TH>>(vectorTree);
    checkIndex<ISCurl<CD_6TH>>(vectorTree);
    checkIndex<ISCurl<FD_1ST>>(vectorTree);
    checkIndex<ISCurl<BD_3RD>>(vectorTree);
}

// The operators with each specialization for the map, and the general form
template<typename TreeT, typename VectorTreeT, typename MapT>
void checkMapOperators(const TreeT& tree, const VectorTreeT& vectorTree, const MapT& map)
{
    using namespace math;
    SCOPED_TRACE(MapT::mapType());

    checkMap<Gradient<MapT, CD_2ND>>(tree, map);
    checkMap<Gradient<MapT, CD_4TH>>(tree, map);
    checkMap<Gradient<MapT, FD_1ST>>(tree, map);
    checkMap<Gradient<MapT, BD_WENO5>>(tree, map);

    checkMapBiased<GradientBiased<MapT, FIRST_BIAS>>(tree, map);
    checkMapBiased<GradientBiased<MapT, HJWENO5_BIAS>>(tree, map);

    checkMap<GradientNormSqrd<MapT, FIRST_BIAS>>(tree, map);
    checkMap<GradientNormSqrd<MapT, HJWENO5_BIAS>>(tree, map);

    // The general form reads the cross derivatives, the specializations do not
    constexpr bool crossLaplacian = std::is_same_v<MapT, AffineMap> ||
        std::is_same_v<MapT, NonlinearFrustumMap>;
    checkMap<Laplacian<MapT, CD_SECOND>>(tree, map);
    checkMap<Laplacian<MapT, CD_FOURTH>>(tree, map, !crossLaplacian);
    checkMap<Laplacian<MapT, CD_SIXTH>>(tree, map, !crossLaplacian);

    checkMap<CPT<MapT, CD_2ND>>(tree, map);
    checkMap<CPT<MapT, FD_1ST>>(tree, map);
    checkMap<CPT_RANGE<MapT, CD_2ND>>(tree, map);

    checkMap<MeanCurvature<MapT, CD_SECOND, CD_2ND>>(tree, map);
    checkMapNormGrad<MeanCurvature<MapT, CD_SECOND, CD_2ND>>(tree, map);
    checkMap<MeanCurvature<MapT, CD_FOURTH, CD_4TH>>(tree, map, CrossOrderDiffers);
    checkMapNormGrad<MeanCurvature<MapT, CD_FOURTH, CD_4TH>>(tree, map, CrossOrderDiffers);

    checkMap<Divergence<MapT, CD_2ND>>(vectorTree, map);
    checkMap<Divergence<MapT, CD_4TH>>(vectorTree, map);
    checkMap<Divergence<MapT, FD_1ST>>(vectorTree, map);

    checkMap<Curl<MapT, CD_2ND>>(vectorTree, map);
    checkMap<Curl<MapT, CD_4TH>>(vectorTree, map);
    checkMap<Curl<MapT, BD_1ST>>(vectorTree, map);
}

template<typename TreeT, typename VectorTreeT>
void checkOperators()
{
    TreeT tree(0.0f);
    fillField(tree);
    VectorTreeT vectorTree(Vec3f(0.0f));
    fillField(vectorTree);

    checkIndexOperators(tree, vectorTree);

    checkMapOperators(tree, vectorTree, math::UniformScaleMap(0.5));
    checkMapOperators(tree, vectorTree, math::UniformScaleTranslateMap(0.5, Vec3d(1, -2, 3)));
    checkMapOperators(tree, vectorTree, math::ScaleMap(Vec3d(0.5, 0.25, 2.0)));
    checkMapOperators(tree, vectorTree, math::ScaleTranslateMap(Vec3d(0.5, 0.25, 2.0), Vec3d(1, -2, 3)));
    checkMapOperators(tree, vectorTree, math::TranslationMap(Vec3d(1, -2, 3)));
    checkMapOperators(tree, vectorTree, math::UnitaryMap(math::X_AXIS, 0.7));
    checkMapOperators(tree, vectorTree,
        math::AffineMap(Mat3d(1.0, 0.2, 0.0, 0.1, 0.5, 0.3, 0.0, -0.2, 2.0)));
    // The taper stays positive over the centers and their neighbors
    checkMapOperators(tree, vectorTree, math::NonlinearFrustumMap(BBoxd(Vec3d(-60), Vec3d(60)), 0.5, 2.0));
}

} // anonymous namespace


class TestNeighborStencil: public ::testing::Test {};


TEST_F(TestNeighborStencil, testOperatorsDefaultLeaf)
{
    checkOperators<FloatTree, Vec3fTree>();
}

TEST_F(TestNeighborStencil, testOperatorsSmallLeaf)
{
    // 4^3 leaves, so that most reads cross into a neighbor leaf
    checkOperators<tree::Tree4<float, 3, 3, 2>::Type, tree::Tree4<Vec3f, 3, 3, 2>::Type>();
}

TEST_F(TestNeighborStencil, testOperatorsTinyLeaf)
{
    // 2^3 leaves, so that the sixth order schemes read past the neighbor leaves
    checkOperators<tree::Tree4<float, 3, 3, 1>::Type, tree::Tree4<Vec3f, 3, 3, 1>::Type>();
}

TEST_F(TestNeighborStencil, testInterface)
{
    using LeafT = FloatTree::LeafNodeType;
    FloatTree tree(-1.0f);
    tree.setValueOn(Coord(1, 2, 3), 4.0f);
    tree.setValueOn(Coord(2, 2, 3), 5.0f);
    tree.setValueOn(Coord(1, 2, 3 + 20), 6.0f);

    // The tree type is deduced and made const
    math::NeighborStencil stencil(tree);
    static_assert(std::is_same_v<decltype(stencil), math::NeighborStencil<const FloatTree>>);
    static_assert(std::is_same_v<math::NeighborStencil<FloatTree>::TreeType, FloatTree>);
    static_assert(std::is_same_v<math::NeighborStencil<FloatTree>::ValueType, float>);
    EXPECT_EQ(&tree, &stencil.tree());
    EXPECT_EQ(Coord::max(), stencil.getCenterCoord());

    stencil.moveTo(Coord(1, 2, 3));
    EXPECT_EQ(Coord(1, 2, 3), stencil.getCenterCoord());
    EXPECT_EQ(4.0f, stencil.getCenterValue());
    EXPECT_EQ(5.0f, (stencil.getValue<1, 0, 0>()));
    EXPECT_EQ(5.0f, stencil.getValue(1, 0, 0));
    EXPECT_EQ(-1.0f, (stencil.getValue<0, -1, 0>()));
    EXPECT_EQ(6.0f, (stencil.getValue<0, 0, 20>()));
    EXPECT_EQ(6.0f, stencil.getValue(0, 0, 20));

    // A value in a leaf is the element of the leaf buffer
    const LeafT* leaf = tree.probeConstLeaf(Coord(1, 2, 3));
    ASSERT_TRUE(leaf != nullptr);
    EXPECT_EQ(&leaf->getValue(Coord(2, 2, 3)), &(stencil.getValue<1, 0, 0>()));

    // Centered by the iterator of another tree, with the values of this tree
    FloatTree other(0.0f);
    other.setValueOn(Coord(2, 2, 3), 9.0f);
    auto iter = other.cbeginLeaf()->cbeginValueOn();
    ASSERT_TRUE(iter);
    stencil.moveTo(iter);
    EXPECT_EQ(Coord(2, 2, 3), stencil.getCenterCoord());
    EXPECT_EQ(5.0f, stencil.getCenterValue());
    EXPECT_EQ(4.0f, (stencil.getValue<-1, 0, 0>()));

    // A copy reads the same values and moves on its own
    math::NeighborStencil<const FloatTree> copy(stencil);
    EXPECT_EQ(5.0f, copy.getCenterValue());
    copy.moveTo(Coord(1, 2, 3));
    EXPECT_EQ(4.0f, copy.getCenterValue());
    EXPECT_EQ(5.0f, stencil.getCenterValue());

    // After a topology change, clear() and moveTo() read the new tree
    tree.setValueOn(Coord(100, 100, 100), 7.0f);
    stencil.clear();
    EXPECT_EQ(Coord::max(), stencil.getCenterCoord());
    stencil.moveTo(Coord(100, 100, 99));
    EXPECT_EQ(7.0f, (stencil.getValue<0, 0, 1>()));
}
