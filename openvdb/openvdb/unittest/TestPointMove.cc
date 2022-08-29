// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointMove.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include "util.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointMove: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointMove


////////////////////////////////////////


namespace {

struct OffsetDeformer
{
    OffsetDeformer(const Vec3d& _offset)
        : offset(_offset){ }

    template <typename LeafT>
    void reset(const LeafT&, size_t /*idx*/) { }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        position += offset;
    }

    Vec3d offset;
}; // struct OffsetDeformer

template <typename FilterT>
struct OffsetFilteredDeformer
{
    OffsetFilteredDeformer(const Vec3d& _offset,
                    const FilterT& _filter)
        : offset(_offset)
        , filter(_filter) { }

    template <typename LeafT>
    void reset(const LeafT& leaf, size_t /*idx*/)
    {
        filter.template reset<LeafT>(leaf);
    }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT& iter) const
    {
        if (!filter.template valid<IndexIterT>(iter))    return;
        position += offset;
    }

    //void finalize() const { }

    Vec3d offset;
    FilterT filter;
}; // struct OffsetFilteredDeformer


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

    // assign point 3 to new group "test" if more than 3 points

    if (positions.size() > 3) {
        appendGroup(points->tree(), "test");

        std::vector<short> groups(positions.size(), 0);
        groups[2] = 1;

        setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");
    }

    return points;
}


std::vector<Vec3s>
gridToPositions(const PointDataGrid::Ptr& points, bool sort = true)
{
    std::vector<Vec3s> positions;

    for (auto leaf = points->tree().beginLeaf(); leaf; ++leaf) {

        const openvdb::points::AttributeArray& positionArray =
            leaf->constAttributeArray("P");
        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);

        for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
            openvdb::Vec3f voxelPosition = positionHandle.get(*iter);
            openvdb::Vec3d xyz = iter.getCoord().asVec3d();
            openvdb::Vec3f worldPosition = points->transform().indexToWorld(voxelPosition + xyz);

            positions.push_back(worldPosition);
        }
    }

    if (sort)   std::sort(positions.begin(), positions.end());
    return positions;
}


std::vector<Vec3s>
applyOffset(const std::vector<Vec3s>& positions, const Vec3s& offset)
{
    std::vector<Vec3s> newPositions;

    for (const auto& it : positions) {
        newPositions.emplace_back(it + offset);
    }

    std::sort(newPositions.begin(), newPositions.end());
    return newPositions;
}


template<typename T>
inline void
ASSERT_APPROX_EQUAL(const std::vector<T>& a, const std::vector<T>& b,
                    const Index lineNumber, const double /*tolerance*/ = 1e-6)
{
    std::stringstream ss;
    ss << "Assertion Line Number: " << lineNumber;

    if (a.size() != b.size())   FAIL() << ss.str();

    for (int i = 0; i < a.size(); i++) {
        if (!math::isApproxEqual(a[i], b[i]))   FAIL() << ss.str();
    }
}


template<typename T>
inline void
ASSERT_APPROX_EQUAL(const std::vector<math::Vec3<T>>& a, const std::vector<math::Vec3<T>>& b,
                    const Index lineNumber, const double /*tolerance*/ = 1e-6)
{
    std::stringstream ss;
    ss << "Assertion Line Number: " << lineNumber;

    if (a.size() != b.size())   FAIL() << ss.str();

    for (size_t i = 0; i < a.size(); i++) {
        if (!math::isApproxEqual(a[i], b[i]))   FAIL() << ss.str();
    }
}

struct NullObject { };

// A dummy iterator that can be used to match LeafIter and IndexIter interfaces
struct DummyIter
{
    DummyIter(Index _index): index(_index) { }
    //Index pos() const { return index; }
    Index operator*() const { return index; }
    Index index;
};

struct OddIndexFilter
{
    static bool initialized() { return true; }
    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT&) { }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return ((*iter) % 2) == 1;
    }
};

} // namespace


TEST_F(TestPointMove, testMoveLocal)
{
    // This test is for points that only move locally, meaning that
    // they remain in the leaf from which they originated

    { // single point, y offset, same voxel
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // two points, y offset, same voxel
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 10.1f, 10},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // two points, y offset, different voxels
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 11, 10},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, y offset, same voxel, only third point is kept
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 10.1f, 10},
                                                    {10, 10.2f, 10},
                                                    {10, 10.3f, 10},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]+offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, y offset, different voxels, only third point is kept
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 11, 10},
                                                    {10, 12, 10},
                                                    {10, 13, 10},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]+offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, y offset, different voxels, only third point is moved
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 11, 10},
                                                    {10, 12, 10},
                                                    {10, 13, 10},
                                                };

        std::vector<Vec3s> desiredPositions(positions);
        desiredPositions[2] = Vec3s(positions[2] + offset);

        std::sort(desiredPositions.begin(), desiredPositions.end());

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        OffsetFilteredDeformer<MultiGroupFilter> deformer(offset, filter);
        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }
}


TEST_F(TestPointMove, testMoveGlobal)
{
    { // four points, all different leafs
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, only third point is kept
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]+offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, third point is deleted
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[0]+offset);
        desiredPositions.emplace_back(positions[1]+offset);
        desiredPositions.emplace_back(positions[3]+offset);

        std::vector<std::string> includeGroups;
        std::vector<std::string> excludeGroups{"test"};

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // six points, some same leaf, some different
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1,     1,     1},
                                                    {1.01f, 1.01f, 1.01f},
                                                    {1,     5.05f, 1},
                                                    {2,     1,     1},
                                                    {2.01f, 1.01f, 1.01f},
                                                    {2,     2,     1},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, only third point is moved
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions(positions);
        desiredPositions[2] = Vec3s(positions[2] + offset);

        std::sort(desiredPositions.begin(), desiredPositions.end());

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        OffsetFilteredDeformer<MultiGroupFilter> deformer(offset, filter);
        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, only third point is kept but not moved
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]);

        std::sort(desiredPositions.begin(), desiredPositions.end());

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        // these groups mark which points are kept

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        // these groups mark which points are moved

        std::vector<std::string> moveIncludeGroups;
        std::vector<std::string> moveExcludeGroups{"test"};

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter moveFilter(moveIncludeGroups, moveExcludeGroups, leaf->attributeSet());
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        OffsetFilteredDeformer<MultiGroupFilter> deformer(offset, moveFilter);
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }
}


namespace {

// Custom Deformer with reset and apply counters
struct CustomDeformer
{
    using LeafT = PointDataGrid::TreeType::LeafNodeType;

    CustomDeformer(const openvdb::Vec3d& offset,
                   std::atomic<int>& resetCalls,
                   std::atomic<int>& applyCalls)
        : mOffset(offset)
        , mResetCalls(resetCalls)
        , mApplyCalls(applyCalls) { }

    template <typename LeafT>
    void reset(const LeafT& /*leaf*/, size_t /*idx*/)
    {
        mResetCalls++;
    }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        // ensure reset has been called at least once
        if (mResetCalls > 0) {
            position += mOffset;
        }
        mApplyCalls++;
    }

    const openvdb::Vec3d mOffset;
    std::atomic<int>& mResetCalls;
    std::atomic<int>& mApplyCalls;
}; // struct CustomDeformer

// Custom Deformer that always returns the position supplied in the constructor
struct StaticDeformer
{
    StaticDeformer(const openvdb::Vec3d& position)
        : mPosition(position) { }

    template <typename LeafT>
    void reset(const LeafT& /*leaf*/, size_t /*idx*/) { }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        position = mPosition;
    }

    const openvdb::Vec3d mPosition;
}; // struct StaticDeformer

} // namespace

TEST_F(TestPointMove, testCustomDeformer)
{
    { // four points, some same leaf, some different, custom deformer
        const float voxelSize = 1.0f;
        Vec3d offset(4.5,3.2,1.85);

        std::vector<Vec3s> positions =          {
                                                    {5, 2, 3},
                                                    {2, 4, 1},
                                                    // {50, 5, 1},
                                                    // {3, 20, 1},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);
        PointDataGrid::Ptr cachedPoints = points->deepCopy();

        const int leafCount = points->tree().leafCount();
        const int pointCount = int(positions.size());

        std::atomic<int> resetCalls, applyCalls;
        resetCalls = 0;
        applyCalls = 0;

        // this deformer applies an offset and tracks the number of calls

        CustomDeformer deformer(offset, resetCalls, applyCalls);

        movePoints(*points, deformer);

        EXPECT_TRUE(leafCount == resetCalls);
        EXPECT_TRUE(pointCount == applyCalls);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);

        resetCalls = 0;
        applyCalls = 0;

        movePoints(*cachedPoints, deformer);

        EXPECT_TRUE(leafCount == resetCalls);
        EXPECT_TRUE(pointCount == applyCalls);

        std::vector<Vec3s> cachedPositions = gridToPositions(cachedPoints);

        ASSERT_APPROX_EQUAL(desiredPositions, cachedPositions, __LINE__);
    }

    {
        { // four points, some same leaf, some different, static deformer
            const float voxelSize = 1.0f;
            Vec3d newPosition(15.2,18.3,-100.9);

            std::vector<Vec3s> positions =          {
                                                        {5, 2, 3},
                                                        {2, 4, 1},
                                                        {50, 5, 1},
                                                        {3, 20, 1},
                                                    };

            std::vector<Vec3s> desiredPositions(positions.size(), newPosition);

            PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

            StaticDeformer deformer(newPosition);

            movePoints(*points, deformer);

            std::vector<Vec3s> actualPositions = gridToPositions(points);

            ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
        }
    }
}


namespace {

// Custom deformer that stores a map of current positions to new positions
struct AssignDeformer
{
    AssignDeformer(const std::map<Vec3d, Vec3d>& _values)
        : values(_values) { }

    template <typename LeafT>
    void reset(const LeafT&, size_t /*idx*/) { }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        position = values.at(position);
    }

    std::map<Vec3d, Vec3d> values;
}; // struct AssignDeformer

}


TEST_F(TestPointMove, testPointData)
{
    // four points, some same leaf, some different
    // spatial order is (1, 0, 3, 2)

    const float voxelSize = 1.0f;
    std::vector<Vec3s> positions =  {
                                    {5, 2, 3},
                                    {2, 4, 1},
                                    {50, 5, 1},
                                    {3, 20, 1},
                                };

    // simple reversing deformer

    std::map<Vec3d, Vec3d> remap;
    remap.insert({positions[0], positions[3]});
    remap.insert({positions[1], positions[2]});
    remap.insert({positions[2], positions[1]});
    remap.insert({positions[3], positions[0]});

    AssignDeformer deformer(remap);

    { // reversing point positions results in the same iteration order due to spatial organisation
        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<Vec3s> initialPositions = gridToPositions(points, /*sort=*/false);

        std::vector<std::string> includeGroups;
        std::vector<std::string> excludeGroups;

        movePoints(*points, deformer);

        auto leaf = points->tree().cbeginLeaf();
        const auto& descriptor = leaf->attributeSet().descriptor();

        // ensure that the test group and it's values have been preserved as expected
        EXPECT_TRUE(descriptor.hasGroup("test"));
        size_t groupCount = 0;
        for (; leaf; ++leaf) {
            GroupHandle handle = leaf->groupHandle("test");
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                groupCount += handle.get(*iter) ? size_t(1) : size_t(0);
            }
        }
        EXPECT_EQ(size_t(1), groupCount);

        std::vector<Vec3s> finalPositions1 = gridToPositions(points, /*sort=*/false);

        ASSERT_APPROX_EQUAL(initialPositions, finalPositions1, __LINE__);

        // now we delete the third point while sorting, using the test group

        excludeGroups.push_back("test");

        leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[0]);
        desiredPositions.emplace_back(positions[1]);
        desiredPositions.emplace_back(positions[3]);

        std::vector<Vec3s> finalPositions2 = gridToPositions(points, /*sort=*/false);

        std::sort(desiredPositions.begin(), desiredPositions.end());
        std::sort(finalPositions2.begin(), finalPositions2.end());

        ASSERT_APPROX_EQUAL(desiredPositions, finalPositions2, __LINE__);
    }

    { // additional point data - integer "id", float "pscale", "odd" and "even" groups

        std::vector<int> id;
        id.push_back(0);
        id.push_back(1);
        id.push_back(2);
        id.push_back(3);

        std::vector<float> radius;
        radius.push_back(0.1f);
        radius.push_back(0.15f);
        radius.push_back(0.2f);
        radius.push_back(0.5f);

        // manually construct point data grid instead of using positionsToGrid()

        const PointAttributeVector<Vec3s> pointList(positions);

        openvdb::math::Transform::Ptr transform(
            openvdb::math::Transform::createLinearTransform(voxelSize));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
                createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                              pointList, *transform);
        auto idAttributeType =
            openvdb::points::TypedAttributeArray<int>::attributeType();
        openvdb::points::appendAttribute(points->tree(), "id", idAttributeType);

        // create a wrapper around the id vector
        openvdb::points::PointAttributeVector<int> idWrapper(id);

        openvdb::points::populateAttribute<openvdb::points::PointDataTree,
            openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<int>>(
                points->tree(), pointIndexGrid->tree(), "id", idWrapper);

        // use fixed-point codec for radius
        // note that this attribute type is not registered by default so needs to be
        // explicitly registered.
        using Codec = openvdb::points::FixedPointCodec</*1-byte=*/false,
                openvdb::points::UnitRange>;
        openvdb::points::TypedAttributeArray<float, Codec>::registerType();
        auto radiusAttributeType =
            openvdb::points::TypedAttributeArray<float, Codec>::attributeType();
        openvdb::points::appendAttribute(points->tree(), "pscale", radiusAttributeType);

        // create a wrapper around the radius vector
        openvdb::points::PointAttributeVector<float> radiusWrapper(radius);

        openvdb::points::populateAttribute<openvdb::points::PointDataTree,
            openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<float>>(
                points->tree(), pointIndexGrid->tree(), "pscale", radiusWrapper);

        appendGroup(points->tree(), "odd");
        appendGroup(points->tree(), "even");
        appendGroup(points->tree(), "nonzero");

        std::vector<short> oddGroups(positions.size(), 0);
        oddGroups[1] = 1;
        oddGroups[3] = 1;
        std::vector<short> evenGroups(positions.size(), 0);
        evenGroups[0] = 1;
        evenGroups[2] = 1;
        std::vector<short> nonZeroGroups(positions.size(), 1);
        nonZeroGroups[0] = 0;

        setGroup(points->tree(), pointIndexGrid->tree(), evenGroups, "even");
        setGroup(points->tree(), pointIndexGrid->tree(), oddGroups, "odd");
        setGroup(points->tree(), pointIndexGrid->tree(), nonZeroGroups, "nonzero");

        movePoints(*points, deformer);

        // extract data

        std::vector<int> id2;
        std::vector<float> radius2;
        std::vector<short> oddGroups2;
        std::vector<short> evenGroups2;
        std::vector<short> nonZeroGroups2;

        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {

            AttributeHandle<int> idHandle(leaf->constAttributeArray("id"));
            AttributeHandle<float> pscaleHandle(leaf->constAttributeArray("pscale"));
            GroupHandle oddHandle(leaf->groupHandle("odd"));
            GroupHandle evenHandle(leaf->groupHandle("even"));
            GroupHandle nonZeroHandle(leaf->groupHandle("nonzero"));

            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                id2.push_back(idHandle.get(*iter));
                radius2.push_back(pscaleHandle.get(*iter));
                oddGroups2.push_back(oddHandle.get(*iter) ? 1 : 0);
                evenGroups2.push_back(evenHandle.get(*iter) ? 1 : 0);
                nonZeroGroups2.push_back(nonZeroHandle.get(*iter) ? 1 : 0);
            }
        }

        // new reversed order is (2, 3, 0, 1)

        EXPECT_EQ(2, id2[0]);
        EXPECT_EQ(3, id2[1]);
        EXPECT_EQ(0, id2[2]);
        EXPECT_EQ(1, id2[3]);

        EXPECT_TRUE(math::isApproxEqual(radius[0], radius2[2], 1e-3f));
        EXPECT_TRUE(math::isApproxEqual(radius[1], radius2[3], 1e-3f));
        EXPECT_TRUE(math::isApproxEqual(radius[2], radius2[0], 1e-3f));
        EXPECT_TRUE(math::isApproxEqual(radius[3], radius2[1], 1e-3f));

        EXPECT_EQ(short(0), oddGroups2[0]);
        EXPECT_EQ(short(1), oddGroups2[1]);
        EXPECT_EQ(short(0), oddGroups2[2]);
        EXPECT_EQ(short(1), oddGroups2[3]);

        EXPECT_EQ(short(1), evenGroups2[0]);
        EXPECT_EQ(short(0), evenGroups2[1]);
        EXPECT_EQ(short(1), evenGroups2[2]);
        EXPECT_EQ(short(0), evenGroups2[3]);

        EXPECT_EQ(short(1), nonZeroGroups2[0]);
        EXPECT_EQ(short(1), nonZeroGroups2[1]);
        EXPECT_EQ(short(0), nonZeroGroups2[2]);
        EXPECT_EQ(short(1), nonZeroGroups2[3]);
    }

    { // larger data set with a cached deformer and group filtering
        std::vector<openvdb::Vec3R> newPositions;
        const int count = 10000;
        unittest_util::genPoints(count, newPositions);

        // manually construct point data grid instead of using positionsToGrid()

        const PointAttributeVector<openvdb::Vec3R> pointList(newPositions);

        openvdb::math::Transform::Ptr transform(
            openvdb::math::Transform::createLinearTransform(/*voxelSize=*/0.1));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
                createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                              pointList, *transform);

        appendGroup(points->tree(), "odd");

        std::vector<short> oddGroups(newPositions.size(), 0);
        for (size_t i = 1; i < newPositions.size(); i += 2) {
            oddGroups[i] = 1;
        }

        setGroup(points->tree(), pointIndexGrid->tree(), oddGroups, "odd");

        std::vector<std::string> includeGroups{"odd"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter advectFilter(includeGroups, excludeGroups, leaf->attributeSet());
        OffsetDeformer offsetDeformer(Vec3d(0, 1, 0));

        double ySumBefore = 0.0;
        double ySumAfter = 0.0;

        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {
            AttributeHandle<Vec3f> handle(leaf->constAttributeArray("P"));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                Vec3d position = handle.get(*iter) + iter.getCoord().asVec3d();
                position = transform->indexToWorld(position);
                ySumBefore += position.y();
            }
        }

        movePoints(*points, offsetDeformer, advectFilter);

        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {
            AttributeHandle<Vec3f> handle(leaf->constAttributeArray("P"));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                Vec3d position = handle.get(*iter) + iter.getCoord().asVec3d();
                position = transform->indexToWorld(position);
                ySumAfter += position.y();
            }
        }

        // total increase in Y should be approximately count / 2
        // (only odd points are being moved 1.0 in Y)
        double increaseInY = ySumAfter - ySumBefore;

        EXPECT_NEAR(static_cast<double>(count) / 2.0, increaseInY,
            /*tolerance=*/double(1.0));
    }
}


TEST_F(TestPointMove, testPointOrder)
{
    struct Local
    {
        using GridT = points::PointDataGrid;

        static void populate(std::vector<Vec3s>& positions, const GridT& points,
            const math::Transform& transform, bool /*threaded*/)
        {
            auto newPoints1 = points.deepCopy();

            points::NullDeformer nullDeformer;
            points::NullFilter nullFilter;
            points::movePoints(*newPoints1, transform, nullDeformer, nullFilter);

            size_t totalPoints = points::pointCount(newPoints1->tree());

            positions.reserve(totalPoints);

            for (auto leaf = newPoints1->tree().cbeginLeaf(); leaf; ++leaf) {
                AttributeHandle<Vec3f> handle(leaf->constAttributeArray("P"));
                for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                    positions.push_back(handle.get(*iter));
                }
            }
        }
    };

    auto sourceTransform = math::Transform::createLinearTransform(/*voxelSize=*/0.1);
    auto targetTransform = math::Transform::createLinearTransform(/*voxelSize=*/1.0);

    auto mask = MaskGrid::create();
    mask->setTransform(sourceTransform);
    mask->denseFill(CoordBBox(Coord(-20,-20,-20), Coord(20,20,20)), true);

    auto points = points::denseUniformPointScatter(*mask, /*pointsPerVoxel=*/8);

    // three copies of the points, two multi-threaded and one single-threaded
    std::vector<Vec3s> positions1;
    std::vector<Vec3s> positions2;
    std::vector<Vec3s> positions3;

    Local::populate(positions1, *points, *targetTransform, true);
    Local::populate(positions2, *points, *targetTransform, true);
    Local::populate(positions3, *points, *targetTransform, false);

    // verify all sequences are identical to confirm that points are ordered deterministically

    ASSERT_APPROX_EQUAL(positions1, positions2, __LINE__);
    ASSERT_APPROX_EQUAL(positions1, positions3, __LINE__);
}
