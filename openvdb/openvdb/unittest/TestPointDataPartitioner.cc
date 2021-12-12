// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/PointDataPartitioner.h>
#include <openvdb/points/PointConversion.h>

#include <gtest/gtest.h>

#include <vector>

using namespace openvdb;
using namespace openvdb::points;


class TestPointDataPartitioner: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


////////////////////////////////////////

namespace {

PointDataGrid::Ptr
positionsToGrid(const std::vector<Vec3s>& positions, const float voxelSize = 1.0)
{
    const PointAttributeVector<Vec3s> pointList(positions);

    openvdb::math::Transform::Ptr transform(
        openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                          pointList, *transform);

    return points;
}

} // namespace

////////////////////////////////////////


TEST_F(TestPointDataPartitioner, testPartitioner)
{
    using LeafArray = PointDataLeafArray<PointDataGrid>;
    using Partitioner = PointDataPartitioner<PointDataGrid>;

    { // single point, no deformer, same transform
        Partitioner partitioner;
        EXPECT_EQ(size_t(0), partitioner.size());

        std::vector<Vec3s> positions;
        positions.emplace_back(0, 1, 0);

        auto points = positionsToGrid(positions);

        LeafArray leafArray(*points, points->constTransform());

        partitioner.construct(leafArray, NullDeformer());

        EXPECT_EQ(size_t(1), partitioner.size());
        EXPECT_EQ(size_t(1), partitioner.size(0));
        EXPECT_EQ(Coord(0), partitioner.origin(0));

        auto iter = partitioner.indices(0);

        EXPECT_TRUE(iter);
        EXPECT_EQ(Index(0), iter.sourceBufferIndex());
        EXPECT_EQ(Index(0), iter.sourceIndex());

        ++iter;
        EXPECT_TRUE(!iter);

        partitioner.clear();

        EXPECT_EQ(size_t(0), partitioner.size());
    }
}

TEST_F(TestPointDataPartitioner, testMerge)
{
    using LeafArray = PointDataLeafArray<PointDataGrid>;
    using Partitioner = PointDataPartitioner<PointDataGrid>;

    { // two points, two grids, no deformer, identical transform
        Partitioner partitioner;
        EXPECT_EQ(size_t(0), partitioner.size());

        std::vector<Vec3s> positions;
        positions.emplace_back(0, 1, 0);

        auto points1 = positionsToGrid(positions);

        positions.clear();
        positions.emplace_back(0, 2, 0);
        auto points2 = positionsToGrid(positions);

        std::vector<PointDataGrid::Ptr> otherPoints;
        otherPoints.push_back(points2);

        LeafArray leafArray(*points1, points1->constTransform(), otherPoints);

        partitioner.construct(leafArray, NullDeformer(), NullFilter(), true);

        EXPECT_EQ(size_t(1), partitioner.size());
        EXPECT_EQ(size_t(2), partitioner.size(0));
        EXPECT_EQ(Coord(0), partitioner.origin(0));

        auto iter = partitioner.indices(0);

        EXPECT_TRUE(iter);
        EXPECT_EQ(Index(0), iter.sourceBufferIndex());
        EXPECT_EQ(Index(0), iter.sourceIndex());

        ++iter;
        EXPECT_TRUE(iter);
        EXPECT_EQ(Index(1), iter.sourceBufferIndex());
        EXPECT_EQ(Index(0), iter.sourceIndex());

        ++iter;
        EXPECT_TRUE(!iter);
    }
}
