// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/PointPartitioner.h>

#include <openvdb/math/Math.h> // math::Rand01
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

TEST_F(TestPointPartitioner, testDeterminism)
{
    const size_t pointCount = 1000*1000;
    const float voxelSize = 0.1f;

    // create a box of points positioned randomly between -1.0 and 1.0

    openvdb::math::Rand01<float> rand01(0);

    std::vector<openvdb::Vec3s> points(pointCount, openvdb::Vec3s(0.f));
    for (size_t n = 0; n < pointCount; ++n) {
        points[n].x() = (rand01() * 2.0f) - 1.0f;
        points[n].y() = (rand01() * 2.0f) - 1.0f;
        points[n].z() = (rand01() * 2.0f) - 1.0f;
    }

    PointList pointList(points);

    const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

    typedef openvdb::tools::UInt32PointPartitioner PointPartitioner;

    std::vector<PointPartitioner::Ptr> partitioners;

    // create 10 point partitioners

    for (size_t i = 0; i < 10; i++) {
        if (i == 0) {
            // first point partitioner has threading disabled
            partitioners.push_back(PointPartitioner::create(pointList, *transform,
                /*voxelOrder=*/false, /*recordVoxelOffsets=*/false,
                /*cellCenteredTransform=*/true, /*threaded=*/false));
        } else {
            partitioners.push_back(PointPartitioner::create(pointList, *transform));
        }
    }

    // verify their sizes are the same

    for (size_t i = 1; i < partitioners.size(); i++) {
        EXPECT_EQ(partitioners[0]->size(), partitioners[i]->size());
    }

    // verify their contents are the same

    for (size_t i = 1; i < partitioners.size(); i++) {
        for (size_t n = 1; n < partitioners[0]->size(); n++) {
            EXPECT_EQ(partitioners[0]->origin(n), partitioners[i]->origin(n));

            auto it1 = partitioners[0]->indices(n);
            auto it2 = partitioners[i]->indices(n);

            size_t bucketSize = it1.size();
            EXPECT_EQ(bucketSize, it2.size());

            for (size_t j = 0; j < bucketSize; j++) {
                EXPECT_EQ(*it1, *it2);

                ++it1;
                ++it2;
            }
        }
    }
}

TEST_F(TestPointPartitioner, testNonLinearArray)
{
    using namespace openvdb;

    const Index arrayCount = 1000;
    const Index pointCount = 1000;
    const Index numPoints = arrayCount * pointCount;
    const float voxelSize = 0.1f;

    // create an array of arrays of points positioned randomly between -1.0 and 1.0

    openvdb::math::Rand01<float> rand01(0);

    std::vector<std::vector<openvdb::Vec3s>> pointArrays(arrayCount);

    for (auto& array : pointArrays) {
        array.reserve(pointCount);
        for (Index i = 0; i < pointCount; i++) {
            float x = (rand01() * 2.0f) - 1.0f;
            float y = (rand01() * 2.0f) - 1.0f;
            float z = (rand01() * 2.0f) - 1.0f;
            array.emplace_back(x, y, z);
        }
    }

    const openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(voxelSize);

    // build a thread-local bin for PointPartitioner (single-threaded)

    static constexpr Index BucketLog2Dim = 3u;
    static constexpr Index BinLog2Dim = 5u;

    using PointPartitioner = openvdb::tools::PointPartitioner<
        std::pair<uint32_t, uint32_t>, BucketLog2Dim, uint32_t>;

    PointPartitioner pointPartitioner;

    using IndexType = PointPartitioner::IndexType;
    using OffsetType = PointPartitioner::OffsetType;
    using VoxelOffsetType = PointPartitioner::VoxelOffsetType;
    using ThreadLocalBin = PointPartitioner::ThreadLocalBin;
    using ThreadLocalBins = PointPartitioner::ThreadLocalBins;
    using IndexPairList = ThreadLocalBin::IndexPairList;
    using IndexPairListPtr = ThreadLocalBin::IndexPairListPtr;

    static constexpr Index BucketLog2Dim2 = 2 * BucketLog2Dim;
    static constexpr Index BucketMask = (1u << BucketLog2Dim) - 1u;
    static constexpr Index BinLog2dim2 = 2 * BinLog2Dim;
    static constexpr Index BinMask = (1u << (BucketLog2Dim + BinLog2Dim)) - 1u;
    static constexpr Index InvBinMask = ~BinMask;

    std::unique_ptr<VoxelOffsetType[]> voxelOffsets;
    voxelOffsets.reset(new VoxelOffsetType[numPoints]);

    ThreadLocalBins bins(1);

    for (Index i = 0; i < arrayCount; i++) {

        for (Index j = 0; j < pointCount; j++) {

            Vec3s pos = pointArrays[i][j];

            Coord ijk = transform->worldToIndexCellCentered(pos);

            VoxelOffsetType voxelOffset(static_cast<VoxelOffsetType>(
                ((ijk[0] & BucketMask) << BucketLog2Dim2) +
                ((ijk[1] & BucketMask) << BucketLog2Dim) +
                (ijk[2] & BucketMask)));

            Coord binCoord(ijk[0] & InvBinMask,
                           ijk[1] & InvBinMask,
                           ijk[2] & InvBinMask);

            ijk[0] &= BinMask;
            ijk[1] &= BinMask;
            ijk[2] &= BinMask;

            ijk[0] >>= BucketLog2Dim;
            ijk[1] >>= BucketLog2Dim;
            ijk[2] >>= BucketLog2Dim;

            OffsetType bucketOffset(
                (ijk[0] << BinLog2dim2) +
                (ijk[1] << BinLog2Dim) +
                ijk[2]);

            IndexPairListPtr& idxPtr = bins[0].map()[binCoord];
            if (!idxPtr)    idxPtr.reset(new IndexPairList);

            idxPtr->emplace_back(IndexType(i, j), bucketOffset);
            voxelOffsets[i*pointCount+j] = voxelOffset;
        }
    }

    // create the point partitioner

    PointPartitioner::Ptr partitioner1 =
            PointPartitioner::create(bins, numPoints);

    EXPECT_TRUE(!partitioner1->empty());
    EXPECT_TRUE(partitioner1->usingCellCenteredTransform());

    // reset seed

    rand01.setSeed(0);

    // create a linear array of points

    std::vector<openvdb::Vec3s> points(numPoints, openvdb::Vec3s(0.f));
    for (size_t n = 0; n < numPoints; ++n) {
        points[n].x() = (rand01() * 2.0f) - 1.0f;
        points[n].y() = (rand01() * 2.0f) - 1.0f;
        points[n].z() = (rand01() * 2.0f) - 1.0f;
    }

    PointList pointList(points);

    using LinearPointPartitioner = openvdb::tools::UInt32PointPartitioner;

    LinearPointPartitioner::Ptr partitioner2 =
        LinearPointPartitioner::create(pointList, *transform);

    // verify partitioners match

    EXPECT_EQ(partitioner1->size(), partitioner2->size());

    for (size_t i = 0; i < partitioner1->size(); i++) {
        EXPECT_EQ(partitioner1->origin(i), partitioner2->origin(i));

        auto it1 = partitioner1->indices(i);
        auto it2 = partitioner2->indices(i);

        size_t bucketSize = it1.size();
        EXPECT_EQ(bucketSize, it2.size());

        for (size_t j = 0; j < bucketSize; j++) {
            const std::pair<uint32_t, uint32_t>& index1 = *it1;
            const uint32_t index1Combined = index1.first * pointCount + index1.second;
            const uint32_t& index2 = *it2;

            EXPECT_EQ(index1Combined, index2);
            EXPECT_EQ(pointArrays[index1.first][index1.second], points[index2]);

            ++it1;
            ++it2;
        }
    }
}
