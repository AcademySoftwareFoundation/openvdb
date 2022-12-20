// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "PointBuilder.h"

#include <openvdb/openvdb.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointReplicate.h>

#include <gtest/gtest.h>

using namespace openvdb;
using namespace openvdb::points;

class TestPointReplicate: public ::testing::Test
{
public:
    void SetUp() override { initialize(); }
    void TearDown() override { uninitialize(); }
}; // class TestPointReplicate


////////////////////////////////////////

template <typename ValueT>
inline void
getAttribute(const PointDataGrid& grid,
            const std::string& attr,
            std::vector<ValueT>& values)
{
    for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
        AttributeHandle<ValueT> handle(leaf->constAttributeArray(attr));
        for (auto iter = leaf->beginIndexAll(); iter; ++iter)
            values.emplace_back(handle.get(*iter));
    }
}

inline void
getP(const PointDataGrid& grid,
     std::vector<Vec3f>& values)
{
    for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
        AttributeHandle<Vec3f> handle(leaf->constAttributeArray("P"));
        auto iter = leaf->beginIndexAll();
        for (; iter; ++iter) {
            auto pos = Vec3d(handle.get(*iter)) + iter.getCoord().asVec3d();
            values.emplace_back(grid.indexToWorld(pos));
        }
    }
}

template <typename ValueT>
inline void
checkReplicatedAttribute(const PointDataGrid& grid,
            const std::string& attr,
            const std::vector<ValueT>& originals, // original values
            const size_t ppp)                     // points per point
{
    std::vector<ValueT> results;
    getAttribute(grid, attr, results);
    EXPECT_EQ(results.size(), originals.size() * ppp);

    auto iter = results.begin();
    for (const auto& o : originals) {
        for (size_t i = 0; i < ppp; ++i, ++iter) {
            EXPECT_TRUE(iter != results.end());
            EXPECT_EQ(o, *iter);
        }
    }
}

template <typename ValueT>
inline void
checkReplicatedAttribute(const PointDataGrid& grid,
            const std::string& attr,
            const std::vector<ValueT>& originals)
{
    std::vector<ValueT> results;
    getAttribute(grid, attr, results);
    EXPECT_EQ(results.size(), originals.size());

    auto iter1 = results.begin();
    auto iter2 = originals.begin();
    for (; iter1 != results.end(); ++iter1, ++iter2) {
        EXPECT_EQ(*iter1, *iter2);
    }
}

template <typename ValueT>
inline void
checkReplicatedAttribute(const PointDataGrid& grid,
            const std::string& attr,
            const ValueT& v)
{
    auto count = points::pointCount(grid.tree());
    const std::vector<ValueT> filled(count, v);
    checkReplicatedAttribute<ValueT>(grid, attr, filled, 1);
}

inline void
checkReplicatedP(const PointDataGrid& grid,
            const std::vector<Vec3f>& originals, // original values
            const size_t ppp)                    // points per point
{
    std::vector<Vec3f> results;
    getP(grid, results);
    EXPECT_EQ(results.size(), originals.size() * ppp);

    auto iter = results.begin();
    for (const auto& o : originals) {
        for (size_t i = 0; i < ppp; ++i, ++iter) {
            EXPECT_TRUE(iter != results.end());
            EXPECT_EQ(o, *iter);
        }
    }
}

inline void
checkReplicatedP(const PointDataGrid& grid,
            const Vec3f& v)
{
    auto count = points::pointCount(grid.tree());
    const std::vector<Vec3f> filled(count, v);
    checkReplicatedP(grid, filled, 1);
}

TEST_F(TestPointReplicate, testReplicate)
{
    // Test no points
    {
        const auto points = PointBuilder({}).get();
        const auto repl = points::replicate(*points, 2);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->empty());
    }

    // Test 1 to many, only position attribute
    {
        const auto points = PointBuilder({Vec3f(1.0f,-2.0f, 3.0f)}).get();

        // 2 points
        auto repl = points::replicate(*points, 2);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().isValueOn({10,-20,30}));
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_EQ(Index64(1), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(2), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, Vec3f(1.0f,-2.0f, 3.0f));

        // 10 points
        repl = points::replicate(*points, 10);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().isValueOn({10,-20,30}));
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_EQ(Index64(1), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(10), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, Vec3f(1.0f,-2.0f, 3.0f));
    }

    // Test 1 to many, arbitrary attributes
    {
        const auto points = PointBuilder({Vec3f(1.0f,-2.0f, 3.0f)})
            .attribute<int32_t>(5, "inttest")
            .attribute<float>(0.3f, "floattest")
            .attribute<double>(-1.3, "doubletest")
            .get();

        // 2 points
        auto repl = points::replicate(*points, 2);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().isValueOn({10,-20,30}));
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_EQ(Index64(1), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(2), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, { Vec3f(1.0f,-2.0f, 3.0f) });
        checkReplicatedAttribute(*repl, "inttest", int32_t(5));
        checkReplicatedAttribute(*repl, "floattest", float(0.3f));
        checkReplicatedAttribute(*repl, "doubletest", double(-1.3));

        // 10 points
        repl = points::replicate(*points, 10);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().isValueOn({10,-20,30}));
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_EQ(Index64(1), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(10), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, Vec3f(1.0f,-2.0f, 3.0f));
        checkReplicatedAttribute(*repl, "inttest", int32_t(5));
        checkReplicatedAttribute(*repl, "floattest", float(0.3f));
        checkReplicatedAttribute(*repl, "doubletest", double(-1.3));
    }

    // Test box points, arbitrary attributes
    {
        const std::vector<int32_t> int1 = {-3,2,1,0,3,-2,-1,0};
        const std::vector<int32_t> int2 = {-10,-5,-9,-1,-2,-2,-1,-2};
        const std::vector<float> float1 = {-4.3f,5.1f,-1.1f,0.0f,9.5f,-10.2f,3.4f,6.2f};
        const std::vector<Vec3f> vec = {
            Vec3f(0.0f), Vec3f(-0.0f), Vec3f(0.3f),
            Vec3f(1.0f,-0.5f,-0.2f), Vec3f(0.2f),
            Vec3f(0.2f, 0.5f, 0.1f), Vec3f(-0.1f),
            Vec3f(0.1f),
        };
        const auto positions = getBoxPoints();

        const auto points = PointBuilder(positions) // 8 points
            .attribute<int32_t>(int1, "inttest1")
            .attribute<int32_t>(int2, "inttest2")
            .attribute<float>(float1, "floattest")
            .attribute<Vec3f>(vec, "vectest")
            .get();

        // 2 points
        auto repl = points::replicate(*points, 2);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_EQ(Index64(8), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(16), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, positions, 2);
        checkReplicatedAttribute(*repl, "inttest1", int1, 2);
        checkReplicatedAttribute(*repl, "inttest2", int2, 2);
        checkReplicatedAttribute(*repl, "floattest", float1, 2);
        checkReplicatedAttribute(*repl, "vectest", vec, 2);

        // 10 points
        repl = points::replicate(*points, 10);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_EQ(Index64(8), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(80), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, positions, 10);
        checkReplicatedAttribute(*repl, "inttest1", int1, 10);
        checkReplicatedAttribute(*repl, "inttest2", int2, 10);
        checkReplicatedAttribute(*repl, "floattest", float1, 10);
        checkReplicatedAttribute(*repl, "vectest", vec, 10);

        // 10 points, specific attributes
        repl = points::replicate(*points, 10, std::vector<std::string>());
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_TRUE(repl->tree().cbeginLeaf());
        EXPECT_EQ(size_t(1), repl->tree().cbeginLeaf()->attributeSet().size());
        EXPECT_EQ(Index64(8), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(80), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, positions, 10);

        // 10 points, specific attributes
        const std::vector<std::string> attrs = { "P", "floattest" };
        repl = points::replicate(*points, 10, attrs);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().hasSameTopology(points->tree()));
        EXPECT_TRUE(repl->tree().cbeginLeaf());
        EXPECT_EQ(size_t(2), repl->tree().cbeginLeaf()->attributeSet().size());
        EXPECT_EQ(Index64(8), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(80), points::pointCount(repl->tree()));
        checkReplicatedP(*repl, positions, 10);
        checkReplicatedAttribute(*repl, "floattest", float1, 10);
    }
}


TEST_F(TestPointReplicate, testReplicateScale)
{
    // Test box points, arbitrary attributes
    {
        const std::vector<float> scales = {-3,2,1,0,3,-2,-1,0};
        const std::vector<int32_t> int2 = {-10,-5,-9,-1,-2,-2,-1,-2};
        const std::vector<float> float1 = {-4.3f,5.1f,-1.1f,0.0f,9.5f,-10.2f,3.4f,6.2f};
        const auto positions = getBoxPoints();

        const auto points = PointBuilder(positions) // 8 points
            .attribute<float>(scales, "scale")
            .attribute<int32_t>(int2, "inttest1")
            .attribute<float>(float1, "floattest")
            .get();

        // 2 points
        auto repl = points::replicate(*points, 2, "scale");
        size_t expectedTotal = 0;
        for (auto& scale : scales) expectedTotal += 2*size_t(scale < 0 ? 0 : scale);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().isValueOn({10,-10,-10}));
        EXPECT_TRUE(repl->tree().isValueOn({-10,10,-10}));
        EXPECT_TRUE(repl->tree().isValueOn({-10,-10,10}));
        EXPECT_EQ(Index64(3), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(expectedTotal), points::pointCount(repl->tree()));
        checkReplicatedAttribute<int32_t>(*repl, "inttest1", {-5,-5,-5,-5,-9,-9,-2,-2,-2,-2,-2,-2});
        checkReplicatedAttribute<float>(*repl, "scale", {2,2,2,2,1,1,3,3,3,3,3,3});
        checkReplicatedAttribute<float>(*repl, "floattest",
            {5.1f,5.1f,5.1f,5.1f,-1.1f,-1.1f,9.5f,9.5f,9.5f,9.5f,9.5f,9.5f});

        // 3 points with repl id
        repl = points::replicate(*points, 3, {"inttest1"}, "scale", "replid");
        expectedTotal = 0;
        for (auto& scale : scales) expectedTotal += 3*size_t(scale < 0 ? 0 : scale);
        EXPECT_TRUE(repl);
        EXPECT_TRUE(repl->tree().isValueOn({10,-10,-10}));
        EXPECT_TRUE(repl->tree().isValueOn({-10,10,-10}));
        EXPECT_TRUE(repl->tree().isValueOn({-10,-10,10}));
        EXPECT_TRUE(repl->tree().cbeginLeaf());
        EXPECT_EQ(size_t(3), repl->tree().cbeginLeaf()->attributeSet().size());
        EXPECT_TRUE(repl->tree().cbeginLeaf()->attributeSet().getConst("P"));
        EXPECT_TRUE(repl->tree().cbeginLeaf()->attributeSet().getConst("inttest1"));
        EXPECT_TRUE(repl->tree().cbeginLeaf()->attributeSet().getConst("replid"));
        EXPECT_EQ(Index64(3), repl->tree().activeVoxelCount());
        EXPECT_EQ(Index64(expectedTotal), points::pointCount(repl->tree()));
        checkReplicatedAttribute<int32_t>(*repl, "inttest1",
            {-5,-5,-5,-5,-5,-5,-9,-9,-9,-2,-2,-2,-2,-2,-2,-2,-2,-2});
        // check the repl id based on which points were replicated
        checkReplicatedAttribute<int32_t>(*repl, "replid",
            {0,1,2,3,4,5,0,1,2,0,1,2,3,4,5,6,7,8});
    }
}

TEST_F(TestPointReplicate, testReplicateZero)
{
    // Test box points, arbitrary attributes
    {
        const std::vector<float> scales = {0.4f,0.4f};
        const std::vector<openvdb::Vec3f> positions = {
            openvdb::Vec3f(0.0f, 0.0f, 0.0f),
            openvdb::Vec3f(0.0f, 0.0f, 0.0f)
        };

        const auto points = PointBuilder(positions) // 2 points
            .attribute<float>(scales, "scale")
            .get();

        //2 points
        auto repl = points::replicate(*points, 0);
        const size_t expectedTotal = 0;
        EXPECT_TRUE(repl);
        EXPECT_EQ(Index64(expectedTotal), points::pointCount(repl->tree()));

        repl = points::replicate(*points, 1, "scale");
        EXPECT_TRUE(repl);
        EXPECT_EQ(Index64(expectedTotal), points::pointCount(repl->tree()));
    }
}

