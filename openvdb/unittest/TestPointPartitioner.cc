// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/tools/PointPartitioner.h>

#include <openvdb/math/Math.h> // math::Rand01

#include <vector>


class TestPointPartitioner: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestPointPartitioner);
    CPPUNIT_TEST(testPartitioner);
    CPPUNIT_TEST(testDeterminism);
    CPPUNIT_TEST_SUITE_END();

    void testPartitioner();
    void testDeterminism();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointPartitioner);

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


void
TestPointPartitioner::testPartitioner()
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

    CPPUNIT_ASSERT(!partitioner->empty());

    // The default interpretation should be cell-centered.
    CPPUNIT_ASSERT(partitioner->usingCellCenteredTransform());

    const size_t expectedPageCount = pointCount / (1u << PointPartitioner::LOG2DIM);

    CPPUNIT_ASSERT_EQUAL(expectedPageCount, partitioner->size());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0), partitioner->origin(0));

    PointPartitioner::IndexIterator it = partitioner->indices(0);

    CPPUNIT_ASSERT(it.test());
    CPPUNIT_ASSERT_EQUAL(it.size(), size_t(1 << PointPartitioner::LOG2DIM));

    PointPartitioner::IndexIterator itB = partitioner->indices(0);

    CPPUNIT_ASSERT_EQUAL(++it, ++itB);
    CPPUNIT_ASSERT(it != ++itB);

    std::vector<PointPartitioner::IndexType> indices;

    for (it.reset(); it; ++it) {
        indices.push_back(*it);
    }

    CPPUNIT_ASSERT_EQUAL(it.size(), indices.size());

    size_t idx = 0;
    for (itB.reset(); itB; ++itB) {
        CPPUNIT_ASSERT_EQUAL(indices[idx++], *itB);
    }
}

void
TestPointPartitioner::testDeterminism()
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
        partitioners.push_back(PointPartitioner::create(pointList, *transform));
    }

    // verify their sizes are the same

    for (size_t i = 1; i < partitioners.size(); i++) {
        CPPUNIT_ASSERT_EQUAL(partitioners[0]->size(), partitioners[i]->size());
    }

    // verify their contents are the same

    for (size_t i = 1; i < partitioners.size(); i++) {
        for (size_t n = 1; n < partitioners[0]->size(); n++) {
            CPPUNIT_ASSERT_EQUAL(partitioners[0]->origin(n), partitioners[i]->origin(n));

            auto it1 = partitioners[0]->indices(n);
            auto it2 = partitioners[i]->indices(n);

            size_t bucketSize = it1.size();
            CPPUNIT_ASSERT_EQUAL(bucketSize, it2.size());

            for (size_t j = 0; j < bucketSize; j++) {
                CPPUNIT_ASSERT_EQUAL(*it1, *it2);

                ++it1;
                ++it2;
            }
        }
    }
}
