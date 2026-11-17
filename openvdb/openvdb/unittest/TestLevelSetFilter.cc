// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Composite.h>        // for csgUnion()

#include <gtest/gtest.h>

using namespace openvdb;

class TestLevelSetFilter: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }

    void testLevelSetFillet();
}; // class TestLevelSetFilter


////////////////////////////////////////


TEST_F(TestLevelSetFilter, testLevelSetFillet)
{
    using GridT = FloatGrid;
    using FilterT = tools::LevelSetFilter<GridT>;

    const float radius = 5.0f;
    const float voxelSize = 1.0f;
    const float halfWidth = 3.0f;
    typename GridT::Ptr sdfGrid = tools::createLevelSetSphere<GridT>(/*radius=*/radius,
            /*center=*/Vec3f(-radius, 0.0f, 0.0f),
            /*dx=*/voxelSize, /*halfWidth*/ halfWidth);
    typename GridT::Ptr sdfGridB = tools::createLevelSetSphere<GridT>(/*radius=*/radius,
            /*center=*/Vec3f(radius, 0.0f, 0.0f),
            /*dx=*/voxelSize, /*halfWidth*/ halfWidth);
    typename GridT::Accessor acc = sdfGrid->getAccessor();

    EXPECT_TRUE(sdfGrid);
    EXPECT_TRUE(sdfGridB);

    tools::csgUnion(*sdfGrid, *sdfGridB);

    {
        EXPECT_TRUE(sdfGrid);

        Coord ijk(0, 3, 0);

        // We expect that the intersection between the two spheres are at (0, 0, 0)
        // so the SDF value of the union in these offsets locations should be > 0
        EXPECT_TRUE(acc.getValue(ijk) > 0.f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0, 1)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0,-1)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 0)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 1)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1,-1)) > 0.0f);
    }

    FilterT filter(*sdfGrid);
    filter.fillet();

    {
        EXPECT_TRUE(sdfGrid);

        Coord ijk(0, 3, 0);

        // After the fillet operation, we expect that the zero-isocontour is
        // pushed outward.
        EXPECT_TRUE(acc.getValue(ijk) < 0.f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0, 1)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0,-1)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 0)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 1)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1,-1)) < 0.0f);
    }
}


namespace {

// Copy the values, active states and tiles of @a source into a level set of another tree type
template<typename GridT>
typename GridT::Ptr
copyLevelSet(const FloatGrid& source)
{
    typename GridT::Ptr grid = GridT::create(source.background());
    grid->setTransform(source.transform().copy());
    grid->setGridClass(GRID_LEVEL_SET);
    for (auto iter = source.cbeginValueAll(); iter; ++iter) {
        if (iter.isVoxelValue()) {
            if (iter.isValueOn()) grid->tree().setValueOn(iter.getCoord(), *iter);
            else grid->tree().setValueOff(iter.getCoord(), *iter);
        } else {
            grid->tree().fill(iter.getBoundingBox(), *iter, iter.isValueOn());
        }
    }
    return grid;
}

// One pass of the box filter of tools::LevelSetFilter along @a axis, through a ValueAccessor
template<typename GridT>
typename GridT::Ptr
referenceBoxPass(const GridT& input, int axis, int width)
{
    using ValueT = typename GridT::ValueType;
    typename GridT::Ptr output = input.deepCopy();
    tree::ValueAccessor<const typename GridT::TreeType> acc(input.tree());
    const ValueT frac = 1 / ValueT(2 * width + 1);
    for (auto leaf = output->tree().beginLeaf(); leaf; ++leaf) {
        for (auto iter = leaf->beginValueOn(); iter; ++iter) {
            const Coord xyz = iter.getCoord();
            Coord ijk = xyz;
            ValueT sum = zeroVal<ValueT>();
            for (int d = -width; d <= width; ++d) {
                ijk[axis] = xyz[axis] + d;
                sum += acc.getValue(ijk);
            }
            iter.setValue(sum * frac);
        }
    }
    return output;
}

template<typename GridT>
void
checkSameGrid(const GridT& expected, const GridT& grid)
{
    EXPECT_TRUE(expected.tree().hasSameTopology(grid.tree()));
    tree::ValueAccessor<const typename GridT::TreeType> acc(grid.tree());
    for (auto iter = expected.cbeginValueAll(); iter; ++iter) {
        EXPECT_EQ(*iter, acc.getValue(iter.getCoord())) << " at " << iter.getCoord();
    }
}

template<typename GridT>
void
checkMeanFilter(const FloatGrid& sphere)
{
    using FilterT = tools::LevelSetFilter<GridT>;
    const typename GridT::Ptr input = copyLevelSet<GridT>(sphere);

    for (int width : {1, 4, 8, 9}) {
        typename GridT::Ptr grid = input->deepCopy();
        FilterT filter(*grid);
        filter.mean(width);

        // The passes run x, then z, then y, and the filter then tracks the interface.
        typename GridT::Ptr expected = input->deepCopy();
        for (int axis : {0, 2, 1}) expected = referenceBoxPass(*expected, axis, width);
        tools::LevelSetTracker<GridT> tracker(*expected);
        tracker.setState(filter.getState());
        tracker.setTrimming(filter.trimming());
        tracker.track();

        checkSameGrid(*expected, *grid);
    }
}

} // unnamed namespace


TEST_F(TestLevelSetFilter, testMeanNeighbors)
{
    const FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(/*radius=*/8.0f,
        /*center=*/Vec3f(0.3f, -0.7f, 1.1f), /*dx=*/0.5f, /*halfWidth=*/3.0f);
    checkMeanFilter<FloatGrid>(*sphere);
    // Smaller internal nodes, so that wide filters cross more of them. The leaves
    // stay at the size that Morphology supports.
    checkMeanFilter<Grid<tree::Tree4<float, 4, 3, 3>::Type>>(*sphere);
}
