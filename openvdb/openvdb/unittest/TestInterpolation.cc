// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tree/NeighborAccessor.h>

#include <gtest/gtest.h>

#include <random>
#include <vector>

using namespace openvdb;

class TestInterpolation: public ::testing::Test
{
};


namespace {

template<typename ValueT>
ValueT
fieldValue(const Coord& xyz)
{
    const float x = float(xyz.x()), y = float(xyz.y()), z = float(xyz.z());
    if constexpr (VecTraits<ValueT>::IsVec) {
        return ValueT(0.125f * x * x - 0.5f * y + 1.0f, 0.25f * y * z - x, 0.0625f * z * z + 0.5f * x);
    } else {
        return ValueT(0.125f * x * x - 0.5f * y + 0.25f * z * x + 1.0f);
    }
}

// A block of 3x3x3 leaves with some inactive voxels, next to active and inactive
// tiles and the background, all with different values
template<typename TreeT>
TreeT
makeTree()
{
    using ValueT = typename TreeT::ValueType;
    constexpr int dim = int(TreeT::LeafNodeType::DIM);
    TreeT tree(fieldValue<ValueT>(Coord(100, 50, -25)));
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
    return tree;
}

// Positions on and near the leaf faces, edges and corners of the block, a walk
// through it in order and random positions around it
std::vector<Vec3R>
makePositions(int dim)
{
    std::vector<Vec3R> positions;
    std::vector<double> nearFaces;
    for (int face = -1; face <= 2; ++face) {
        for (double delta : {-1.0, -0.5, -0.25, 0.0, 0.25, 0.5}) {
            nearFaces.push_back(face * dim + delta);
        }
    }
    for (double x : nearFaces) {
        for (double y : nearFaces) {
            for (double z : nearFaces) positions.emplace_back(x, y, z);
        }
    }
    for (double x = -2.0 * dim - 0.3; x < 3.0 * dim; x += 0.37) {
        positions.emplace_back(x, 0.6 * dim - 0.2, 1.0 * dim - 0.7);
    }
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(-2.0 * dim - 1.5, 3.0 * dim + 1.5);
    for (int i = 0; i < 1000; ++i) {
        const double x = distribution(generator);
        const double y = distribution(generator);
        const double z = distribution(generator);
        positions.emplace_back(x, y, z);
    }
    return positions;
}

// Both overloads of @a SamplerT through a NeighborAccessor, against the same
// overloads through a ValueAccessor and the tree
template<typename SamplerT, typename TreeT>
void
checkSampler(const TreeT& tree, const std::vector<Vec3R>& positions)
{
    using ValueT = typename TreeT::ValueType;
    tree::ValueAccessor<const TreeT> valueAccessor(tree);
    tree::NeighborAccessor<const TreeT> neighbors(tree);
    size_t activeCount = 0;
    for (const Vec3R& xyz : positions) {
        const ValueT expected = SamplerT::sample(valueAccessor, xyz);
        EXPECT_EQ(expected, SamplerT::sample(tree, xyz));
        EXPECT_EQ(expected, SamplerT::sample(neighbors, xyz)) << SamplerT::name() << " at " << xyz;

        ValueT expectedProbed = zeroVal<ValueT>(), probed = zeroVal<ValueT>();
        const bool expectedActive = SamplerT::sample(valueAccessor, xyz, expectedProbed);
        const bool active = SamplerT::sample(neighbors, xyz, probed);
        EXPECT_EQ(expectedActive, active) << SamplerT::name() << " at " << xyz;
        EXPECT_EQ(expectedProbed, probed) << SamplerT::name() << " at " << xyz;
        if (active) ++activeCount;
    }
    // Some samples are in the background and some are not
    EXPECT_TRUE(activeCount > 0);
    EXPECT_TRUE(activeCount < positions.size());
}

template<typename TreeT>
void
checkBoxValues(const TreeT& tree, const std::vector<Vec3R>& positions)
{
    using ValueT = typename TreeT::ValueType;
    tree::ValueAccessor<const TreeT> valueAccessor(tree);
    tree::NeighborAccessor<const TreeT> neighbors(tree);
    for (const Vec3R& xyz : positions) {
        const Coord ijk = Coord::floor(xyz);
        ValueT expected[2][2][2], values[2][2][2];
        tools::BoxSampler::getValues(expected, valueAccessor, ijk);
        tools::BoxSampler::getValues(values, neighbors, ijk);
        for (int i = 0; i < 8; ++i) {
            EXPECT_EQ(expected[i >> 2][(i >> 1) & 1][i & 1], values[i >> 2][(i >> 1) & 1][i & 1]);
        }
        const bool expectedActive = tools::BoxSampler::probeValues(expected, valueAccessor, ijk);
        EXPECT_EQ(expectedActive, tools::BoxSampler::probeValues(values, neighbors, ijk));
        for (int i = 0; i < 8; ++i) {
            EXPECT_EQ(expected[i >> 2][(i >> 1) & 1][i & 1], values[i >> 2][(i >> 1) & 1][i & 1]);
        }
    }
}

template<typename ScalarTreeT, typename VectorTreeT>
void
checkSamplers()
{
    const ScalarTreeT scalarTree = makeTree<ScalarTreeT>();
    const VectorTreeT vectorTree = makeTree<VectorTreeT>();
    const std::vector<Vec3R> positions = makePositions(int(ScalarTreeT::LeafNodeType::DIM));

    checkSampler<tools::PointSampler>(scalarTree, positions);
    checkSampler<tools::BoxSampler>(scalarTree, positions);
    checkSampler<tools::QuadraticSampler>(scalarTree, positions);

    checkSampler<tools::PointSampler>(vectorTree, positions);
    checkSampler<tools::BoxSampler>(vectorTree, positions);
    checkSampler<tools::QuadraticSampler>(vectorTree, positions);

    checkSampler<tools::StaggeredPointSampler>(vectorTree, positions);
    checkSampler<tools::StaggeredBoxSampler>(vectorTree, positions);
    checkSampler<tools::StaggeredQuadraticSampler>(vectorTree, positions);

    // Through the Sampler template
    checkSampler<tools::Sampler<1, false>>(scalarTree, positions);
    checkSampler<tools::Sampler<2, true>>(vectorTree, positions);

    checkBoxValues(scalarTree, positions);
    checkBoxValues(vectorTree, positions);
}

} // unnamed namespace


TEST_F(TestInterpolation, testNeighborAccessorDefaultLeaf)
{
    checkSamplers<FloatTree, Vec3STree>();
}

TEST_F(TestInterpolation, testNeighborAccessorSmallLeaf)
{
    checkSamplers<tree::Tree4<float, 3, 3, 2>::Type, tree::Tree4<Vec3f, 3, 3, 2>::Type>();
}

TEST_F(TestInterpolation, testNeighborAccessorTinyLeaf)
{
    // Leaves smaller than the footprint of the cubic samplers
    checkSamplers<tree::Tree4<float, 3, 3, 1>::Type, tree::Tree4<Vec3f, 3, 3, 1>::Type>();
}
