// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/IndexIterator.h>
#include <openvdb/points/IndexFilter.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointCount.h>

#include <gtest/gtest.h>

#include <sstream>
#include <iostream>
#include <utility>

using namespace openvdb;
using namespace openvdb::points;

class TestIndexFilter: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }

    void testRandomLeafFilterImpl();
}; // class TestIndexFilter


////////////////////////////////////////


struct OriginLeaf
{
    OriginLeaf(const openvdb::Coord& _leafOrigin, const size_t _size = size_t(0)):
        leafOrigin(_leafOrigin), size(_size) { }
    openvdb::Coord origin() const { return leafOrigin; }
    size_t pointCount() const { return size; }
    const openvdb::Coord leafOrigin;
    const size_t size;
};


struct SimpleIter
{
    SimpleIter() : i(0) { }
    int operator*() const { return i; }
    void operator++() { i++; }
    openvdb::Coord getCoord() const { return coord; }
    int i;
    openvdb::Coord coord;
};


template <bool LessThan>
class ThresholdFilter
{
public:
    ThresholdFilter(const int threshold)
        : mThreshold(threshold) { }

    bool isPositiveInteger() const { return mThreshold > 0; }
    bool isMax() const { return mThreshold == std::numeric_limits<int>::max(); }

    static bool initialized() { return true; }
    inline index::State state() const
    {
        if (LessThan) {
            if (isMax())                    return index::ALL;
            else if (!isPositiveInteger())  return index::NONE;
        }
        else {
            if (isMax())                    return index::NONE;
            else if (!isPositiveInteger())  return index::ALL;
        }
        return index::PARTIAL;
    }

    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT&) { }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        return LessThan ? *iter < mThreshold : *iter > mThreshold;
    }

private:
    const int mThreshold;
}; // class ThresholdFilter


/// @brief Generates the signed distance to a sphere located at @a center
/// and with a specified @a radius (both in world coordinates). Only voxels
/// in the domain [0,0,0] -> @a dim are considered. Also note that the
/// level set is either dense, dense narrow-band or sparse narrow-band.
///
/// @note This method is VERY SLOW and should only be used for debugging purposes!
/// However it works for any transform and even with open level sets.
/// A faster approch for closed narrow band generation is to only set voxels
/// sparsely and then use grid::signedFloodFill to define the sign
/// of the background values and tiles! This is implemented in openvdb/tools/LevelSetSphere.h
template<class GridType>
inline void
makeSphere(const openvdb::Coord& dim, const openvdb::Vec3f& center, float radius, GridType& grid)
{
    using ValueT = typename GridType::ValueType;
    const ValueT zero = openvdb::zeroVal<ValueT>();

    typename GridType::Accessor acc = grid.getAccessor();
    openvdb::Coord xyz;
    for (xyz[0]=0; xyz[0]<dim[0]; ++xyz[0]) {
        for (xyz[1]=0; xyz[1]<dim[1]; ++xyz[1]) {
            for (xyz[2]=0; xyz[2]<dim[2]; ++xyz[2]) {
                const openvdb::Vec3R p =  grid.transform().indexToWorld(xyz);
                const float dist = float((p-center).length() - radius);
                ValueT val = ValueT(zero + dist);
                acc.setValue(xyz, val);
            }
        }
    }
}


template <typename LeafT>
bool
multiGroupMatches(  const LeafT& leaf, const Index32 size,
                    const std::vector<Name>& include, const std::vector<Name>& exclude,
                    const std::vector<int>& indices)
{
    using IndexGroupIter = IndexIter<ValueVoxelCIter, MultiGroupFilter>;
    ValueVoxelCIter indexIter(0, size);
    MultiGroupFilter filter(include, exclude, leaf.attributeSet());
    filter.reset(leaf);
    IndexGroupIter iter(indexIter, filter);
    for (unsigned i = 0; i < indices.size(); ++i, ++iter) {
        if (!iter)                                  return false;
        if (*iter != Index32(indices[i]))           return false;
    }
    return !iter;
}


TEST_F(TestIndexFilter, testActiveFilter)
{
    // create a point grid, three points are stored in two leafs

    PointDataGrid::Ptr points;
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 2, 1}, {10.1f, 10, 1}};

    const double voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

    // check there are two leafs

    EXPECT_EQ(Index32(2), points->tree().leafCount());

    ActiveFilter activeFilter;
    InactiveFilter inActiveFilter;

    EXPECT_EQ(index::PARTIAL, activeFilter.state());
    EXPECT_EQ(index::PARTIAL, inActiveFilter.state());

    { // test default active / inactive values
        auto leafIter = points->tree().cbeginLeaf();

        EXPECT_EQ(index::PARTIAL, activeFilter.state(*leafIter));
        EXPECT_EQ(index::PARTIAL, inActiveFilter.state(*leafIter));

        auto indexIter = leafIter->beginIndexAll();
        activeFilter.reset(*leafIter);
        inActiveFilter.reset(*leafIter);

        EXPECT_TRUE(activeFilter.valid(indexIter));
        EXPECT_TRUE(!inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(activeFilter.valid(indexIter));
        EXPECT_TRUE(!inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndexAll();
        activeFilter.reset(*leafIter);
        inActiveFilter.reset(*leafIter);

        EXPECT_TRUE(activeFilter.valid(indexIter));
        EXPECT_TRUE(!inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
    }

    auto firstLeaf = points->tree().beginLeaf();

    { // set all voxels to be inactive in the first leaf
        firstLeaf->getValueMask().set(false);

        auto leafIter = points->tree().cbeginLeaf();

        EXPECT_EQ(index::NONE, activeFilter.state(*leafIter));
        EXPECT_EQ(index::ALL, inActiveFilter.state(*leafIter));

        auto indexIter = leafIter->beginIndexAll();
        activeFilter.reset(*leafIter);
        inActiveFilter.reset(*leafIter);

        EXPECT_TRUE(!activeFilter.valid(indexIter));
        EXPECT_TRUE(inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!activeFilter.valid(indexIter));
        EXPECT_TRUE(inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndexAll();
        activeFilter.reset(*leafIter);
        inActiveFilter.reset(*leafIter);

        EXPECT_TRUE(activeFilter.valid(indexIter));
        EXPECT_TRUE(!inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
    }

    { // set all voxels to be active in the first leaf
        firstLeaf->getValueMask().set(true);

        auto leafIter = points->tree().cbeginLeaf();

        EXPECT_EQ(index::ALL, activeFilter.state(*leafIter));
        EXPECT_EQ(index::NONE, inActiveFilter.state(*leafIter));

        auto indexIter = leafIter->beginIndexAll();
        activeFilter.reset(*leafIter);
        inActiveFilter.reset(*leafIter);

        EXPECT_TRUE(activeFilter.valid(indexIter));
        EXPECT_TRUE(!inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(activeFilter.valid(indexIter));
        EXPECT_TRUE(!inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndexAll();
        activeFilter.reset(*leafIter);
        inActiveFilter.reset(*leafIter);

        EXPECT_TRUE(activeFilter.valid(indexIter));
        EXPECT_TRUE(!inActiveFilter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
    }
}

TEST_F(TestIndexFilter, testMultiGroupFilter)
{
    using LeafNode          = PointDataTree::LeafNodeType;
    using AttributeVec3f    = TypedAttributeArray<Vec3f>;

    PointDataTree tree;
    LeafNode* leaf = tree.touchLeaf(openvdb::Coord(0, 0, 0));

    using Descriptor = AttributeSet::Descriptor;
    Descriptor::Ptr descriptor = Descriptor::create(AttributeVec3f::attributeType());

    const Index size = 5;

    leaf->initializeAttributes(descriptor, size);

    appendGroup(tree, "even");
    appendGroup(tree, "odd");
    appendGroup(tree, "all");
    appendGroup(tree, "first");

    { // construction, copy construction
        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups;
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        EXPECT_TRUE(!filter.initialized());
        MultiGroupFilter filter2 = filter;
        EXPECT_TRUE(!filter2.initialized());

        filter.reset(*leaf);
        EXPECT_TRUE(filter.initialized());
        MultiGroupFilter filter3 = filter;
        EXPECT_TRUE(filter3.initialized());
    }

    // group population

    { // even
        GroupWriteHandle groupHandle = leaf->groupWriteHandle("even");
        groupHandle.set(0, true);
        groupHandle.set(2, true);
        groupHandle.set(4, true);
    }

    { // odd
        GroupWriteHandle groupHandle = leaf->groupWriteHandle("odd");
        groupHandle.set(1, true);
        groupHandle.set(3, true);
    }

    setGroup(tree, "all", true);

    { // first
        GroupWriteHandle groupHandle = leaf->groupWriteHandle("first");
        groupHandle.set(0, true);
    }

    { // test state()
        std::vector<Name> include;
        std::vector<Name> exclude;
        MultiGroupFilter filter(include, exclude, leaf->attributeSet());
        EXPECT_EQ(filter.state(), index::ALL);
        include.push_back("all");
        MultiGroupFilter filter2(include, exclude, leaf->attributeSet());
        EXPECT_EQ(filter2.state(), index::PARTIAL);
    }

    // test multi group iteration

    { // all (implicit, no include or exclude)
        std::vector<Name> include;
        std::vector<Name> exclude;
        std::vector<int> indices{0, 1, 2, 3, 4};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all include
        std::vector<Name> include{"all"};
        std::vector<Name> exclude;
        std::vector<int> indices{0, 1, 2, 3, 4};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all exclude
        std::vector<Name> include;
        std::vector<Name> exclude{"all"};
        std::vector<int> indices;
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all include and exclude
        std::vector<Name> include{"all"};
        std::vector<Name> exclude{"all"};
        std::vector<int> indices;
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // even include
        std::vector<Name> include{"even"};
        std::vector<Name> exclude;
        std::vector<int> indices{0, 2, 4};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd include
        std::vector<Name> include{"odd"};
        std::vector<Name> exclude;
        std::vector<int> indices{1, 3};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd include and exclude
        std::vector<Name> include{"odd"};
        std::vector<Name> exclude{"odd"};
        std::vector<int> indices;
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd and first include
        std::vector<Name> include{"odd", "first"};
        std::vector<Name> exclude;
        std::vector<int> indices{0, 1, 3};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // even include, first exclude
        std::vector<Name> include{"even"};
        std::vector<Name> exclude{"first"};
        std::vector<int> indices{2, 4};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all include, first and odd exclude
        std::vector<Name> include{"all"};
        std::vector<Name> exclude{"first", "odd"};
        std::vector<int> indices{2, 4};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd and first include, even exclude
        std::vector<Name> include{"odd", "first"};
        std::vector<Name> exclude{"even"};
        std::vector<int> indices{1, 3};
        EXPECT_TRUE(multiGroupMatches(*leaf, size, include, exclude, indices));
    }
}


void
TestIndexFilter::testRandomLeafFilterImpl()
{
    { // generateRandomSubset
        std::vector<int> values = index_filter_internal::generateRandomSubset<std::mt19937, int>(
            /*seed*/unsigned(0), 1, 20);

        EXPECT_EQ(values.size(), size_t(1));

        // different seed

        std::vector<int> values2 = index_filter_internal::generateRandomSubset<std::mt19937, int>(
            /*seed*/unsigned(1), 1, 20);

        EXPECT_EQ(values2.size(), size_t(1));
        EXPECT_TRUE(values[0] != values2[0]);

        // different integer type

        std::vector<long> values3 = index_filter_internal::generateRandomSubset<std::mt19937, long>(
            /*seed*/unsigned(0), 1, 20);

        EXPECT_EQ(values3.size(), size_t(1));
        EXPECT_TRUE(values[0] == values3[0]);

        // different random number generator

        values = index_filter_internal::generateRandomSubset<std::mt19937_64, int>(
            /*seed*/unsigned(1), 1, 20);

        EXPECT_EQ(values.size(), size_t(1));
        EXPECT_TRUE(values[0] != values2[0]);

        // no values

        values = index_filter_internal::generateRandomSubset<std::mt19937, int>(
            /*seed*/unsigned(0), 0, 20);

        EXPECT_EQ(values.size(), size_t(0));

        // all values

        values = index_filter_internal::generateRandomSubset<std::mt19937, int>(
            /*seed*/unsigned(0), 1000, 1000);

        EXPECT_EQ(values.size(), size_t(1000));

        // ensure all numbers are represented

        std::sort(values.begin(), values.end());

        for (int i = 0; i < 1000; i++) {
            EXPECT_EQ(values[i], i);
        }
    }

    { // RandomLeafFilter
        using RandFilter = RandomLeafFilter<PointDataTree, std::mt19937>;

        PointDataTree tree;

        RandFilter filter(tree, 0);

        EXPECT_TRUE(filter.state() == index::PARTIAL);

        filter.mLeafMap[Coord(0, 0, 0)] = std::make_pair(0, 10);
        filter.mLeafMap[Coord(0, 0, 8)] = std::make_pair(1, 1);
        filter.mLeafMap[Coord(0, 8, 0)] = std::make_pair(2, 50);

        { // construction, copy construction
            EXPECT_TRUE(filter.initialized());
            RandFilter filter2 = filter;
            EXPECT_TRUE(filter2.initialized());

            filter.reset(OriginLeaf(Coord(0, 0, 0), 10));
            EXPECT_TRUE(filter.initialized());
            RandFilter filter3 = filter;
            EXPECT_TRUE(filter3.initialized());
        }

        { // all 10 values
            filter.reset(OriginLeaf(Coord(0, 0, 0), 10));
            std::vector<int> values;

            for (SimpleIter iter; *iter < 100; ++iter) {
                if (filter.valid(iter))     values.push_back(*iter);
            }

            EXPECT_EQ(values.size(), size_t(10));

            for (int i = 0; i < 10; i++) {
                EXPECT_EQ(values[i], i);
            }
        }

        { // 50 of 100
            filter.reset(OriginLeaf(Coord(0, 8, 0), 100));
            std::vector<int> values;

            for (SimpleIter iter; *iter < 100; ++iter) {
                if (filter.valid(iter))     values.push_back(*iter);
            }

            EXPECT_EQ(values.size(), size_t(50));

            // ensure no duplicates

            std::sort(values.begin(), values.end());
            auto it = std::adjacent_find(values.begin(), values.end());

            EXPECT_TRUE(it == values.end());
        }
    }
}
TEST_F(TestIndexFilter, testRandomLeafFilter) { testRandomLeafFilterImpl(); }


inline void
setId(PointDataTree& tree, const size_t index, const std::vector<int>& ids)
{
    int offset = 0;
    for (auto leafIter = tree.beginLeaf(); leafIter; ++leafIter) {
        auto id = AttributeWriteHandle<int>::create(leafIter->attributeArray(index));

        for (auto iter = leafIter->beginIndexAll(); iter; ++iter) {
            if (offset >= int(ids.size()))   throw std::runtime_error("Out of range");

            id->set(*iter, ids[offset++]);
        }
    }
}


TEST_F(TestIndexFilter, testAttributeHashFilter)
{
    std::vector<Vec3s> positions{{1, 1, 1}, {2, 2, 2}, {11, 11, 11}, {12, 12, 12}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // four points, two leafs

    EXPECT_EQ(tree.leafCount(), Index32(2));

    appendAttribute<int>(tree, "id");

    const size_t index = tree.cbeginLeaf()->attributeSet().descriptor().find("id");

    // ascending integers, block one
    std::vector<int> ids{1, 2, 3, 4};
    setId(tree, index, ids);

    using HashFilter = AttributeHashFilter<std::mt19937, int>;

    { // construction, copy construction
        HashFilter filter(index, 0.0f);
        EXPECT_TRUE(filter.state() == index::PARTIAL);
        EXPECT_TRUE(!filter.initialized());
        HashFilter filter2 = filter;
        EXPECT_TRUE(!filter2.initialized());

        filter.reset(*tree.cbeginLeaf());
        EXPECT_TRUE(filter.initialized());
        HashFilter filter3 = filter;
        EXPECT_TRUE(filter3.initialized());
    }

    { // zero percent
        HashFilter filter(index, 0.0f);

        auto leafIter = tree.cbeginLeaf();

        auto indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);

        EXPECT_TRUE(!filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);
        EXPECT_TRUE(!filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
    }

    { // one hundred percent
        HashFilter filter(index, 100.0f);

        auto leafIter = tree.cbeginLeaf();

        auto indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);

        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
    }

    { // fifty percent
        HashFilter filter(index, 50.0f);

        auto leafIter = tree.cbeginLeaf();

        auto indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);

        EXPECT_TRUE(!filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
    }

    { // fifty percent, new seed
        HashFilter filter(index, 50.0f, /*seed=*/100);

        auto leafIter = tree.cbeginLeaf();

        auto indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);

        EXPECT_TRUE(!filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndexAll();
        filter.reset(*leafIter);
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(filter.valid(indexIter));
        ++indexIter;
        EXPECT_TRUE(!indexIter);
    }
}


TEST_F(TestIndexFilter, testLevelSetFilter)
{
    // create a point grid

    PointDataGrid::Ptr points;

    {
        std::vector<Vec3s> positions{{1, 1, 1}, {1, 2, 1}, {10.1f, 10, 1}};

        const double voxelSize(1.0);
        math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

        points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    }

    // create a sphere levelset

    FloatGrid::Ptr sphere;

    {
        double voxelSize = 0.5;
        sphere = FloatGrid::create(/*backgroundValue=*/5.0);
        sphere->setTransform(math::Transform::createLinearTransform(voxelSize));

        const openvdb::Coord dim(10, 10, 10);
        const openvdb::Vec3f center(0.0f, 0.0f, 0.0f);
        const float radius = 2;
        makeSphere<FloatGrid>(dim, center, radius, *sphere);
    }

    using LSFilter = LevelSetFilter<FloatGrid>;

    { // construction, copy construction
        LSFilter filter(*sphere, points->transform(), -4.0f, 4.0f);
        EXPECT_TRUE(filter.state() == index::PARTIAL);
        EXPECT_TRUE(!filter.initialized());
        LSFilter filter2 = filter;
        EXPECT_TRUE(!filter2.initialized());

        filter.reset(* points->tree().cbeginLeaf());
        EXPECT_TRUE(filter.initialized());
        LSFilter filter3 = filter;
        EXPECT_TRUE(filter3.initialized());
    }

    { // capture both points near origin
        LSFilter filter(*sphere, points->transform(), -4.0f, 4.0f);
        auto leafIter = points->tree().cbeginLeaf();
        auto iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(filter.valid(iter));
        ++iter;
        EXPECT_TRUE(filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(iter);
        EXPECT_TRUE(!filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);
    }

    { // capture just the inner-most point
        LSFilter filter(*sphere, points->transform(), -0.3f, -0.25f);
        auto leafIter = points->tree().cbeginLeaf();
        auto iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(iter);
        EXPECT_TRUE(!filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);
    }

    { // capture everything but the second point (min > max)
        LSFilter filter(*sphere, points->transform(), -0.25f, -0.3f);
        auto leafIter = points->tree().cbeginLeaf();
        auto iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(!filter.valid(iter));
        ++iter;
        EXPECT_TRUE(filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(iter);
        EXPECT_TRUE(filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);
    }

    {
        std::vector<Vec3s> positions{{1, 1, 1}, {1, 2, 1}, {10.1f, 10, 1}};

        const double voxelSize(0.25);
        math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

        points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    }

    {
        double voxelSize = 1.0;
        sphere = FloatGrid::create(/*backgroundValue=*/5.0);
        sphere->setTransform(math::Transform::createLinearTransform(voxelSize));

        const openvdb::Coord dim(40, 40, 40);
        const openvdb::Vec3f center(10.0f, 10.0f, 0.1f);
        const float radius = 0.2f;
        makeSphere<FloatGrid>(dim, center, radius, *sphere);
    }

    { // capture only the last point using a different transform and a new sphere
        LSFilter filter(*sphere, points->transform(), 0.5f, 1.0f);
        auto leafIter = points->tree().cbeginLeaf();
        auto iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(!filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(!filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter.reset(*leafIter);

        EXPECT_TRUE(iter);
        EXPECT_TRUE(filter.valid(iter));
        ++iter;
        EXPECT_TRUE(!iter);
    }
}


TEST_F(TestIndexFilter, testBBoxFilter)
{
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 2, 1}, {10.1f, 10, 1}};

    const float voxelSize(0.5);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    EXPECT_EQ(tree.leafCount(), Index32(2));

    // build some bounding box filters to test

    BBoxFilter filter1(*transform, BBoxd({0.5, 0.5, 0.5}, {1.5, 1.5, 1.5}));
    BBoxFilter filter2(*transform, BBoxd({0.5, 0.5, 0.5}, {1.5, 2.01, 1.5}));
    BBoxFilter filter3(*transform, BBoxd({0.5, 0.5, 0.5}, {11, 11, 1.5}));
    BBoxFilter filter4(*transform, BBoxd({-10, 0, 0}, {11, 1.2, 1.2}));

    { // construction, copy construction
        EXPECT_TRUE(!filter1.initialized());
        BBoxFilter filter5 = filter1;
        EXPECT_TRUE(!filter5.initialized());

        filter1.reset(*tree.cbeginLeaf());
        EXPECT_TRUE(filter1.initialized());
        BBoxFilter filter6 = filter1;
        EXPECT_TRUE(filter6.initialized());
    }

    // leaf 1

    auto leafIter = tree.cbeginLeaf();

    {
        auto iter(leafIter->beginIndexOn());

        // point 1

        filter1.reset(*leafIter);
        EXPECT_TRUE(filter1.valid(iter));
        filter2.reset(*leafIter);
        EXPECT_TRUE(filter2.valid(iter));
        filter3.reset(*leafIter);
        EXPECT_TRUE(filter3.valid(iter));
        filter4.reset(*leafIter);
        EXPECT_TRUE(filter4.valid(iter));

        ++iter;

        // point 2

        filter1.reset(*leafIter);
        EXPECT_TRUE(!filter1.valid(iter));
        filter2.reset(*leafIter);
        EXPECT_TRUE(filter2.valid(iter));
        filter3.reset(*leafIter);
        EXPECT_TRUE(filter3.valid(iter));
        filter4.reset(*leafIter);
        EXPECT_TRUE(!filter4.valid(iter));

        ++iter;
        EXPECT_TRUE(!iter);
    }

    ++leafIter;

    // leaf 2

    {
        auto iter(leafIter->beginIndexOn());

        // point 3

        filter1.reset(*leafIter);
        EXPECT_TRUE(!filter1.valid(iter));
        filter2.reset(*leafIter);
        EXPECT_TRUE(!filter2.valid(iter));
        filter3.reset(*leafIter);
        EXPECT_TRUE(filter3.valid(iter));
        filter4.reset(*leafIter);
        EXPECT_TRUE(!filter4.valid(iter));

        ++iter;
        EXPECT_TRUE(!iter);
    }
}


struct NeedsInitializeFilter
{
    inline bool initialized() const { return mInitialized; }
    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    inline index::State state(const LeafT&) { return index::PARTIAL; }
    template <typename LeafT>
    void reset(const LeafT&) { mInitialized = true; }
private:
    bool mInitialized = false;
};


TEST_F(TestIndexFilter, testBinaryFilter)
{
    const int intMax = std::numeric_limits<int>::max();

    { // construction, copy construction
        using InitializeBinaryFilter = BinaryFilter<NeedsInitializeFilter, NeedsInitializeFilter, /*And=*/true>;

        NeedsInitializeFilter needs1;
        NeedsInitializeFilter needs2;
        InitializeBinaryFilter filter(needs1, needs2);
        EXPECT_TRUE(filter.state() == index::PARTIAL);
        EXPECT_TRUE(!filter.initialized());
        InitializeBinaryFilter filter2 = filter;
        EXPECT_TRUE(!filter2.initialized());

        filter.reset(OriginLeaf(Coord(0, 0, 0)));
        EXPECT_TRUE(filter.initialized());
        InitializeBinaryFilter filter3 = filter;
        EXPECT_TRUE(filter3.initialized());
    }

    using LessThanFilter    = ThresholdFilter<true>;
    using GreaterThanFilter = ThresholdFilter<false>;

    { // less than
        LessThanFilter zeroFilter(0); // all invalid
        EXPECT_TRUE(zeroFilter.state() == index::NONE);
        LessThanFilter maxFilter(intMax); // all valid
        EXPECT_TRUE(maxFilter.state() == index::ALL);

        LessThanFilter filter(5);
        filter.reset(OriginLeaf(Coord(0, 0, 0)));
        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        EXPECT_EQ(values.size(), size_t(5));

        for (int i = 0; i < 5; i++) {
            EXPECT_EQ(values[i], i);
        }
    }

    { // greater than
        GreaterThanFilter zeroFilter(0); // all valid
        EXPECT_TRUE(zeroFilter.state() == index::ALL);
        GreaterThanFilter maxFilter(intMax); // all invalid
        EXPECT_TRUE(maxFilter.state() == index::NONE);

        GreaterThanFilter filter(94);
        filter.reset(OriginLeaf(Coord(0, 0, 0)));
        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        EXPECT_EQ(values.size(), size_t(5));

        int offset = 0;
        for (int i = 95; i < 100; i++) {
            EXPECT_EQ(values[offset++], i);
        }
    }

    { // binary and
        using RangeFilter = BinaryFilter<LessThanFilter, GreaterThanFilter, /*And=*/true>;

        RangeFilter zeroFilter(LessThanFilter(0), GreaterThanFilter(10)); // all invalid
        EXPECT_TRUE(zeroFilter.state() == index::NONE);
        RangeFilter maxFilter(LessThanFilter(intMax), GreaterThanFilter(0)); // all valid
        EXPECT_TRUE(maxFilter.state() == index::ALL);

        RangeFilter filter(LessThanFilter(55), GreaterThanFilter(45));
        EXPECT_TRUE(filter.state() == index::PARTIAL);

        filter.reset(OriginLeaf(Coord(0, 0, 0)));

        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        EXPECT_EQ(values.size(), size_t(9));

        int offset = 0;
        for (int i = 46; i < 55; i++) {
            EXPECT_EQ(values[offset++], i);
        }
    }

    { // binary or
        using HeadTailFilter = BinaryFilter<LessThanFilter, GreaterThanFilter, /*And=*/false>;

        HeadTailFilter zeroFilter(LessThanFilter(0), GreaterThanFilter(10)); // some valid
        EXPECT_TRUE(zeroFilter.state() == index::PARTIAL);
        HeadTailFilter maxFilter(LessThanFilter(intMax), GreaterThanFilter(0)); // all valid
        EXPECT_TRUE(maxFilter.state() == index::ALL);

        HeadTailFilter filter(LessThanFilter(5), GreaterThanFilter(95));
        filter.reset(OriginLeaf(Coord(0, 0, 0)));

        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        EXPECT_EQ(values.size(), size_t(9));

        int offset = 0;
        for (int i = 0; i < 5; i++) {
            EXPECT_EQ(values[offset++], i);
        }
        for (int i = 96; i < 100; i++) {
            EXPECT_EQ(values[offset++], i);
        }
    }
}
