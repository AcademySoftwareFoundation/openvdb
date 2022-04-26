// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/ParticleAtlas.h>
#include <openvdb/math/Math.h>

#include <gtest/gtest.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include "util.h" // for genPoints


struct TestParticleAtlas: public ::testing::Test
{
};


////////////////////////////////////////

namespace {

class ParticleList
{
public:
    typedef openvdb::Vec3R          PosType;
    typedef PosType::value_type     ScalarType;

    ParticleList(const std::vector<PosType>& points,
        const std::vector<ScalarType>& radius)
        : mPoints(&points)
        , mRadius(&radius)
    {
    }

    // Return the number of points in the array
    size_t size() const {
        return mPoints->size();
    }

    // Return the world-space position for the nth particle.
    void getPos(size_t n, PosType& xyz) const {
        xyz = (*mPoints)[n];
    }

    // Return the world-space radius for the nth particle.
    void getRadius(size_t n, ScalarType& radius) const {
        radius = (*mRadius)[n];
    }

protected:
    std::vector<PosType>    const * const mPoints;
    std::vector<ScalarType> const * const mRadius;
}; // ParticleList


template<typename T>
bool hasDuplicates(const std::vector<T>& items)
{
    std::vector<T> vec(items);
    std::sort(vec.begin(), vec.end());

    size_t duplicates = 0;
    for (size_t n = 1, N = vec.size(); n < N; ++n) {
        if (vec[n] == vec[n-1]) ++duplicates;
    }
    return duplicates != 0;
}

} // namespace



////////////////////////////////////////


TEST_F(TestParticleAtlas, testParticleAtlas)
{
    // generate points

    const size_t numParticle = 40000;
    const double minVoxelSize = 0.01;

    std::vector<openvdb::Vec3R> points;
    unittest_util::genPoints(numParticle, points);

    std::vector<double> radius;
    for (size_t n = 0, N = points.size() / 2; n < N; ++n) {
        radius.push_back(minVoxelSize);
    }

    for (size_t n = points.size() / 2, N = points.size(); n < N; ++n) {
        radius.push_back(minVoxelSize * 2.0);
    }

    ParticleList particles(points, radius);

    // construct data structure

    typedef openvdb::tools::ParticleAtlas<> ParticleAtlas;

    ParticleAtlas atlas;

    EXPECT_TRUE(atlas.empty());
    EXPECT_TRUE(atlas.levels() == 0);

    atlas.construct(particles, minVoxelSize);

    EXPECT_TRUE(!atlas.empty());
    EXPECT_TRUE(atlas.levels() == 2);

    EXPECT_TRUE(
        openvdb::math::isApproxEqual(atlas.minRadius(0), minVoxelSize));

    EXPECT_TRUE(
        openvdb::math::isApproxEqual(atlas.minRadius(1), minVoxelSize * 2.0));

    typedef openvdb::tools::ParticleAtlas<>::Iterator ParticleAtlasIterator;

    ParticleAtlasIterator it(atlas);

    EXPECT_TRUE(atlas.levels() == 2);

    std::vector<uint32_t> indices;
    indices.reserve(numParticle);

    it.updateFromLevel(0);

    EXPECT_TRUE(it);
    EXPECT_EQ(it.size(), numParticle - (points.size() / 2));


    for (; it; ++it) {
        indices.push_back(*it);
    }

    it.updateFromLevel(1);

    EXPECT_TRUE(it);
    EXPECT_EQ(it.size(), (points.size() / 2));


    for (; it; ++it) {
        indices.push_back(*it);
    }

    EXPECT_EQ(numParticle, indices.size());

    EXPECT_TRUE(!hasDuplicates(indices));


    openvdb::Vec3R center = points[0];
    double searchRadius = minVoxelSize * 10.0;

    it.worldSpaceSearchAndUpdate(center, searchRadius, particles);
    EXPECT_TRUE(it);

    indices.clear();
    for (; it; ++it) {
        indices.push_back(*it);
    }

    EXPECT_EQ(it.size(), indices.size());
    EXPECT_TRUE(!hasDuplicates(indices));


    openvdb::BBoxd bbox;
    for (size_t n = 0, N = points.size() / 2; n < N; ++n) {
        bbox.expand(points[n]);
    }

    it.worldSpaceSearchAndUpdate(bbox, particles);
    EXPECT_TRUE(it);

    indices.clear();
    for (; it; ++it) {
        indices.push_back(*it);
    }

    EXPECT_EQ(it.size(), indices.size());
    EXPECT_TRUE(!hasDuplicates(indices));
}

