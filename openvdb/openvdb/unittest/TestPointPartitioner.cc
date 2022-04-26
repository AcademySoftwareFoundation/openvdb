// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/PointPartitioner.h>
#include <gtest/gtest.h>

#include <vector>


class TestPointPartitioner: public ::testing::Test
{
};


////////////////////////////////////////

namespace {

struct PointList {
    typedef openvdb::Vec3s PosType;

    PointList(const std::vector<PosType>& points) : mPoints(&points) {}

    size_t size() const { return mPoints->size(); }

    void getPos(size_t n, PosType& xyz) const { xyz = (*mPoints)[n]; }

protected:
    std::vector<PosType> const * const mPoints;
}; // PointList

} // namespace

////////////////////////////////////////


TEST_F(TestPointPartitioner, testPartitioner)
{
    const size_t pointCount = 10000;
    const float voxelSize = 0.1f;

    std::vector<openvdb::Vec3s> points(pointCount, openvdb::Vec3s(0.f));
    for (size_t n = 1; n < pointCount; ++n) {
        points[n].x() = points[n-1].x() + voxelSize;
    }

    PointList pointList(points);

    const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

    typedef openvdb::tools::UInt32PointPartitioner PointPartitioner;

    PointPartitioner::Ptr partitioner =
            PointPartitioner::create(pointList, *transform);

    EXPECT_TRUE(!partitioner->empty());

    // The default interpretation should be cell-centered.
    EXPECT_TRUE(partitioner->usingCellCenteredTransform());

    const size_t expectedPageCount = pointCount / (1u << PointPartitioner::LOG2DIM);

    EXPECT_EQ(expectedPageCount, partitioner->size());
    EXPECT_EQ(openvdb::Coord(0), partitioner->origin(0));

    PointPartitioner::IndexIterator it = partitioner->indices(0);

    EXPECT_TRUE(it.test());
    EXPECT_EQ(it.size(), size_t(1 << PointPartitioner::LOG2DIM));

    PointPartitioner::IndexIterator itB = partitioner->indices(0);

    EXPECT_EQ(++it, ++itB);
    EXPECT_TRUE(it != ++itB);

    std::vector<PointPartitioner::IndexType> indices;

    for (it.reset(); it; ++it) {
        indices.push_back(*it);
    }

    EXPECT_EQ(it.size(), indices.size());

    size_t idx = 0;
    for (itB.reset(); itB; ++itB) {
        EXPECT_EQ(indices[idx++], *itB);
    }
}

