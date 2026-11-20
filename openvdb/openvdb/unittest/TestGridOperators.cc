// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Operators.h>
#include <openvdb/tools/GridOperators.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace openvdb;

class TestGridOperators: public ::testing::Test
{
};


namespace {

template<typename ValueT>
ValueT
fieldValue(const Coord& xyz)
{
    const float x = float(xyz.x()), y = float(xyz.y()), z = float(xyz.z());
    if constexpr (VecTraits<ValueT>::IsVec) {
        return ValueT(0.125f * x * y - z, 0.0625f * z * z + x, 0.25f * x - 0.125f * y * z);
    } else {
        return ValueT(0.0625f * (x * x + 2.0f * y * y + z * z) - 0.25f * x * z - 3.0f);
    }
}

// A block of 3x3x3 leaves with some inactive voxels, next to active and inactive
// tiles of the leaf size and the background
template<typename GridT>
typename GridT::Ptr
makeGrid(const math::Transform::Ptr& transform)
{
    using ValueT = typename GridT::ValueType;
    constexpr int dim = int(GridT::TreeType::LeafNodeType::DIM);
    typename GridT::Ptr grid = GridT::create(fieldValue<ValueT>(Coord(40, -30, 20)));
    grid->setTransform(transform);
    auto& tree = grid->tree();
    for (int x = -dim; x < 2 * dim; ++x) {
        for (int y = -dim; y < 2 * dim; ++y) {
            for (int z = -dim; z < 2 * dim; ++z) {
                const Coord xyz(x, y, z);
                if ((x + 2 * y + z) % 5 == 0) tree.setValueOff(xyz, fieldValue<ValueT>(xyz));
                else tree.setValueOn(xyz, fieldValue<ValueT>(xyz));
            }
        }
    }
    tree.addTile(1, Coord(2 * dim, 0, 0), fieldValue<ValueT>(Coord(-7, 3, 2)), true);
    tree.addTile(1, Coord(-2 * dim, 0, 0), fieldValue<ValueT>(Coord(5, -4, 1)), false);
    tree.addTile(1, Coord(0, 2 * dim, dim), fieldValue<ValueT>(Coord(2, 9, -6)), true);
    tree.addTile(1, Coord(dim, 0, -2 * dim), fieldValue<ValueT>(Coord(-3, -3, 8)), false);
    return grid;
}

MaskGrid::Ptr
makeMask(const math::Transform::Ptr& transform)
{
    constexpr int dim = 8;
    MaskGrid::Ptr mask = MaskGrid::create();
    mask->setTransform(transform);
    mask->fill(CoordBBox(Coord(-dim, -dim, -dim), Coord(dim + dim / 2, 3 * dim, dim)), true);
    return mask;
}

bool isSame(double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); }

template<typename T>
bool isSame(const math::Vec3<T>& a, const math::Vec3<T>& b)
{
    return isSame(a[0], b[0]) && isSame(a[1], b[1]) && isSame(a[2], b[2]);
}

// Compare every active value of @a output with OpT applied to @a input through a
// ValueAccessor. A tile of an operator that does not densify holds the result at its origin.
template<template<typename> class OpT, typename InGridT, typename OutGridT>
struct CheckOutput
{
    CheckOutput(const InGridT& in, const OutGridT& out, bool atOrigin)
        : input(in), output(out), tilesAtOrigin(atOrigin) {}

    template<typename MapT>
    void operator()(const MapT& map)
    {
        tree::ValueAccessor<const typename InGridT::TreeType> acc(input.tree());
        for (auto iter = output.cbeginValueOn(); iter; ++iter) {
            if (iter.isVoxelValue() || tilesAtOrigin) {
                this->check(OpT<MapT>::result(map, acc, iter.getCoord()), *iter, iter.getCoord());
            } else {
                const CoordBBox bbox = iter.getBoundingBox();
                for (auto ijk = bbox.begin(); ijk; ++ijk) {
                    this->check(OpT<MapT>::result(map, acc, *ijk), *iter, *ijk);
                }
            }
        }
        EXPECT_TRUE(count > 0);
    }

    template<typename ValueT>
    void check(const ValueT& expected, const typename OutGridT::ValueType& value, const Coord& ijk)
    {
        EXPECT_TRUE(isSame(typename OutGridT::ValueType(expected), value))
            << expected << " and " << value << " at " << ijk;
        ++count;
    }

    const InGridT& input;
    const OutGridT& output;
    const bool tilesAtOrigin;
    Index64 count = 0;
};

template<template<typename> class OpT, typename InGridT, typename OutGridT>
void
checkOutput(const InGridT& input, const OutGridT& output, bool tilesAtOrigin)
{
    CheckOutput<OpT, InGridT, OutGridT> check(input, output, tilesAtOrigin);
    math::processTypedMap(input.transform(), check);
}

template<typename MapT> using CptOp = math::CPT<MapT, math::CD_2ND>;
template<typename MapT> using CptRangeOp = math::CPT_RANGE<MapT, math::CD_2ND>;
template<typename MapT> using CurlOp = math::Curl<MapT, math::CD_2ND>;
template<typename MapT> using DivergenceOp = math::Divergence<MapT, math::CD_2ND>;
template<typename MapT> using StaggeredDivergenceOp = math::Divergence<MapT, math::FD_1ST>;
template<typename MapT> using GradientOp = math::Gradient<MapT, math::CD_2ND>;
template<typename MapT> using LaplacianOp = math::Laplacian<MapT, math::CD_SECOND>;
template<typename MapT> using MeanCurvatureOp = math::MeanCurvature<MapT, math::CD_SECOND, math::CD_2ND>;

template<typename MapT>
struct MagnitudeOp
{
    template<typename AccessorT>
    static float result(const MapT&, const AccessorT& acc, const Coord& ijk)
    {
        return acc.getValue(ijk).length();
    }
};

template<typename MapT>
struct NormalizeOp
{
    template<typename AccessorT>
    static Vec3f result(const MapT&, const AccessorT& acc, const Coord& ijk)
    {
        Vec3f vec = acc.getValue(ijk);
        if (!vec.normalize()) vec.setZero();
        return vec;
    }
};

// An operator with no stencil overload, which GridOperator reads through its accessor
template<typename MapT>
struct AccessorOnlyGradientOp
{
    template<typename AccessorT>
    static Vec3f result(const MapT& map, const AccessorT& acc, const Coord& ijk)
    {
        return math::Gradient<MapT, math::FD_1ST>::result(map, acc, ijk);
    }
};

// GridOperator itself, with densify on and off
template<template<typename> class OpT>
struct CheckGridOperator
{
    template<typename MapT>
    void operator()(const MapT& map)
    {
        for (bool densify : {true, false}) {
            tools::gridop::GridOperator<FloatGrid, MaskGrid, Vec3SGrid, MapT, OpT<MapT>>
                op(*grid, mask, map, nullptr, densify);
            const Vec3SGrid::Ptr output = op.process(/*threaded=*/true);
            CheckOutput<OpT, FloatGrid, Vec3SGrid> check(*grid, *output, !densify);
            check(map);
        }
    }

    const FloatGrid* grid;
    const MaskGrid* mask;
};

void
checkOperators(const math::Transform::Ptr& transform)
{
    const FloatGrid::Ptr scalarGrid = makeGrid<FloatGrid>(transform);
    const Vec3SGrid::Ptr vectorGrid = makeGrid<Vec3SGrid>(transform);
    const MaskGrid::Ptr mask = makeMask(transform);

    for (const MaskGrid* maskGrid : {static_cast<const MaskGrid*>(nullptr),
            static_cast<const MaskGrid*>(mask.get())}) {
        {
            using CptT = tools::Cpt<FloatGrid, MaskGrid>;
            CptT op = maskGrid ? CptT(*scalarGrid, *maskGrid) : CptT(*scalarGrid);
            checkOutput<CptRangeOp>(*scalarGrid, *op.process(true, /*useWorldTransform=*/true), true);
            checkOutput<CptOp>(*scalarGrid, *op.process(true, /*useWorldTransform=*/false), true);
        }
        if (maskGrid) {
            checkOutput<CurlOp>(*vectorGrid, *tools::curl(*vectorGrid, *maskGrid), false);
            checkOutput<DivergenceOp>(*vectorGrid, *tools::divergence(*vectorGrid, *maskGrid), false);
            checkOutput<GradientOp>(*scalarGrid, *tools::gradient(*scalarGrid, *maskGrid), false);
            checkOutput<LaplacianOp>(*scalarGrid, *tools::laplacian(*scalarGrid, *maskGrid), false);
            checkOutput<MeanCurvatureOp>(*scalarGrid,
                *tools::meanCurvature(*scalarGrid, *maskGrid), false);
            checkOutput<MagnitudeOp>(*vectorGrid, *tools::magnitude(*vectorGrid, *maskGrid), true);
            checkOutput<NormalizeOp>(*vectorGrid, *tools::normalize(*vectorGrid, *maskGrid), true);
        } else {
            checkOutput<CurlOp>(*vectorGrid, *tools::curl(*vectorGrid), false);
            checkOutput<DivergenceOp>(*vectorGrid, *tools::divergence(*vectorGrid), false);
            checkOutput<GradientOp>(*scalarGrid, *tools::gradient(*scalarGrid), false);
            checkOutput<LaplacianOp>(*scalarGrid, *tools::laplacian(*scalarGrid), false);
            checkOutput<MeanCurvatureOp>(*scalarGrid, *tools::meanCurvature(*scalarGrid), false);
            checkOutput<MagnitudeOp>(*vectorGrid, *tools::magnitude(*vectorGrid), true);
            checkOutput<NormalizeOp>(*vectorGrid, *tools::normalize(*vectorGrid), true);
        }

        CheckGridOperator<GradientOp> checkGradient{scalarGrid.get(), maskGrid};
        math::processTypedMap(scalarGrid->transform(), checkGradient);
        CheckGridOperator<AccessorOnlyGradientOp> checkAccessorOnly{scalarGrid.get(), maskGrid};
        math::processTypedMap(scalarGrid->transform(), checkAccessorOnly);
    }

    // A staggered grid uses forward differences for the divergence
    const Vec3SGrid::Ptr staggeredGrid = vectorGrid->deepCopy();
    staggeredGrid->setGridClass(GRID_STAGGERED);
    checkOutput<StaggeredDivergenceOp>(*staggeredGrid, *tools::divergence(*staggeredGrid), false);
}

} // unnamed namespace


TEST_F(TestGridOperators, testNeighborStencilUniform)
{
    checkOperators(math::Transform::createLinearTransform(0.5));
}

TEST_F(TestGridOperators, testNeighborStencilScale)
{
    checkOperators(math::Transform::Ptr(new math::Transform(
        math::MapBase::Ptr(new math::ScaleMap(Vec3d(0.5, 0.25, 2.0))))));
}

TEST_F(TestGridOperators, testNeighborStencilFrustum)
{
    checkOperators(math::Transform::Ptr(new math::Transform(math::MapBase::Ptr(
        new math::NonlinearFrustumMap(BBoxd(Vec3d(-60.0), Vec3d(60.0)), 0.5, 2.0)))));
}
