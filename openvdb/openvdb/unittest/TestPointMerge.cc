// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0


#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointMerge.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/tools/LevelSetSphere.h>

#include <gtest/gtest.h>

#include <algorithm> // std::count
#include <vector>

using namespace openvdb;
using namespace openvdb::points;


class TestPointMerge: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


////////////////////////////////////////

TEST_F(TestPointMerge, testMerge)
{
    const float voxelSize = 0.1f;
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));
    std::vector<Vec3f> points1{{0,0,0}};
    std::vector<Vec3f> points2{{10,0,0}};

    PointDataGrid::Ptr grid1 = createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(points1, *transform);
    PointDataGrid::Ptr grid2 = createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(points2, *transform);

    std::vector<PointDataGrid::Ptr> grids;
    grids.push_back(grid1);
    grids.push_back(grid2);

    PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

    // Ensure point in B was merged into A

    math::Coord coord = transform->worldToIndexCellCentered(points2[0]);
    EXPECT_TRUE(result->tree().probeLeaf(coord)->pointCount() == 1);

    // Ensure original point in A still exists

    coord = transform->worldToIndexCellCentered(points1[0]);
    EXPECT_TRUE(result->tree().probeLeaf(coord)->pointCount() == 1);
}

TEST_F(TestPointMerge, testGroupMerge)
{
    const float voxelSize = 0.1f;
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));
    std::vector<Vec3f> points{{0,0,0}};

    {
        PointDataGrid::Ptr grid1 = createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(points, *transform);
        PointDataGrid::Ptr grid2 = createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(points, *transform);
        appendGroup(grid2->tree(), "a1");
        setGroup(grid2->tree(), "a1", true);

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());

        const auto leafIter = result->tree().cbeginLeaf();
        const auto& desc = leafIter->attributeSet().descriptor();

        EXPECT_TRUE(leafIter);
        EXPECT_TRUE(desc.hasGroup("a1"));
        EXPECT_EQ(leafIter->pointCount(), Index64(2));
        GroupHandle handle(leafIter->groupHandle("a1"));
        EXPECT_EQ(handle.get(0), false);
        EXPECT_EQ(handle.get(1), true);
    }

    {
        PointDataGrid::Ptr grid1 = createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(points, *transform);
        PointDataGrid::Ptr grid2 = createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(points, *transform);
        appendGroup(grid1->tree(), "a1");
        setGroup(grid1->tree(), "a1", true);

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());

        const auto leafIter = result->tree().cbeginLeaf();
        const auto& desc = leafIter->attributeSet().descriptor();

        EXPECT_TRUE(leafIter);
        EXPECT_TRUE(desc.hasGroup("a1"));
        EXPECT_EQ(leafIter->pointCount(), Index64(2));
        GroupHandle handle(leafIter->groupHandle("a1"));
        EXPECT_EQ(handle.get(0), true);
        EXPECT_EQ(handle.get(1), false);
    }
}

TEST_F(TestPointMerge, testMultiAttributeMerge)
{
    // five points across four leafs with transform1, all the in same leaf with
    // transform2

    math::Transform::Ptr transform1(math::Transform::createLinearTransform(1.0));
    math::Transform::Ptr transform2(math::Transform::createLinearTransform(10.0));
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 3, 1}, {2, 5, 1}, {5, 1, 1}, {5, 5, 1}};
    const Index64 totalPointCount(positions.size() * 3);

    {
        PointDataGrid::Ptr grid1;
        PointDataGrid::Ptr grid2;
        PointDataGrid::Ptr grid3;
        {
            grid1 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
            PointDataTree& tree1 = grid1->tree();

            appendAttribute<int16_t>(tree1, "a1");
            appendAttribute<Vec3f>(tree1, "a2");
            appendAttribute<double>(tree1, "a5");
            appendAttribute<Vec3i>(tree1, "a6");

            grid2 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
            PointDataTree& tree2 = grid2->tree();

            appendAttribute<int64_t>(tree2, "a3");
            appendAttribute<double>(tree2, "a5");
            appendAttribute<int16_t>(tree2, "a1");
            appendAttribute<Vec3f>(tree2, "a2");

            grid3 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
            PointDataTree& tree3 = grid3->tree();

            appendAttribute<int16_t>(tree3, "a4");
            appendAttribute<Vec3f>(tree3, "a2");
            appendAttribute<int64_t>(tree3, "a3");
            appendAttribute<int16_t>(tree3, "a1");
        }

        //   grid1 has:
        //    - a1: short, a2: vec3f, a5: double, a6: vec3short
        //   grid2 has:
        //    - a3: long, a5: double, a1: short, a2: vec3f
        //   grid3 has:
        //    - a4: short, a2: vec3f, a3: long, a1: short

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);
        grids.push_back(grid3);

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());
        EXPECT_TRUE(!grid3->tree().cbeginLeaf());
        EXPECT_EQ(totalPointCount, pointCount(result->tree()));

        const auto leafIter = result->tree().cbeginLeaf();

        EXPECT_TRUE(leafIter);
        EXPECT_TRUE(leafIter->hasAttribute("a1"));
        EXPECT_TRUE(leafIter->attributeArray("a1").hasValueType<int16_t>());
        EXPECT_TRUE(leafIter->hasAttribute("a2"));
        EXPECT_TRUE(leafIter->attributeArray("a2").hasValueType<Vec3f>());
        EXPECT_TRUE(leafIter->hasAttribute("a3"));
        EXPECT_TRUE(leafIter->attributeArray("a3").hasValueType<int64_t>());
        EXPECT_TRUE(leafIter->hasAttribute("a4"));
        EXPECT_TRUE(leafIter->attributeArray("a4").hasValueType<int16_t>());
        EXPECT_TRUE(leafIter->hasAttribute("a5"));
        EXPECT_TRUE(leafIter->attributeArray("a5").hasValueType<double>());
        EXPECT_TRUE(leafIter->hasAttribute("a6"));
        EXPECT_TRUE(leafIter->attributeArray("a6").hasValueType<Vec3i>());
    }

    {
        PointDataGrid::Ptr grid1;
        PointDataGrid::Ptr grid2;
        PointDataGrid::Ptr grid3;
        {
            grid1 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform2);
            PointDataTree& tree1 = grid1->tree();

            appendAttribute<int16_t>(tree1, "a1");
            appendAttribute<Vec3f>(tree1, "a2");
            appendAttribute<double>(tree1, "a5");
            appendAttribute<Vec3i>(tree1, "a6");

            grid2 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform2);
            PointDataTree& tree2 = grid2->tree();

            appendAttribute<int64_t>(tree2, "a3");
            appendAttribute<double>(tree2, "a5");
            appendAttribute<int16_t>(tree2, "a1");
            appendAttribute<Vec3f>(tree2, "a2");

            grid3 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform2);
            PointDataTree& tree3 = grid3->tree();

            appendAttribute<int16_t>(tree3, "a4");
            appendAttribute<Vec3f>(tree3, "a2");
            appendAttribute<int64_t>(tree3, "a3");
            appendAttribute<int16_t>(tree3, "a1");
        }

        //   grid1 has:
        //    - a1: short, a2: vec3f, a5: double, a6: vec3short
        //   grid2 has:
        //    - a3: long, a5: double, a1: short, a2: vec3f
        //   grid3 has:
        //    - a4: short, a2: vec3f, a3: long, a1: short

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);
        grids.push_back(grid3);

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());
        EXPECT_TRUE(!grid3->tree().cbeginLeaf());
        EXPECT_EQ(totalPointCount, pointCount(result->tree()));

        const auto leafIter = result->tree().cbeginLeaf();

        EXPECT_TRUE(leafIter);
        EXPECT_TRUE(leafIter->hasAttribute("a1"));
        EXPECT_TRUE(leafIter->attributeArray("a1").hasValueType<int16_t>());
        EXPECT_TRUE(leafIter->hasAttribute("a2"));
        EXPECT_TRUE(leafIter->attributeArray("a2").hasValueType<Vec3f>());
        EXPECT_TRUE(leafIter->hasAttribute("a3"));
        EXPECT_TRUE(leafIter->attributeArray("a3").hasValueType<int64_t>());
        EXPECT_TRUE(leafIter->hasAttribute("a4"));
        EXPECT_TRUE(leafIter->attributeArray("a4").hasValueType<int16_t>());
        EXPECT_TRUE(leafIter->hasAttribute("a5"));
        EXPECT_TRUE(leafIter->attributeArray("a5").hasValueType<double>());
        EXPECT_TRUE(leafIter->hasAttribute("a6"));
        EXPECT_TRUE(leafIter->attributeArray("a6").hasValueType<Vec3i>());
    }
}

TEST_F(TestPointMerge, testMultiGroupMerge)
{
    // five points across four leafs with transform1, all the in same leaf with
    // transform2

    math::Transform::Ptr transform1(math::Transform::createLinearTransform(1.0));
    math::Transform::Ptr transform2(math::Transform::createLinearTransform(10.0));
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 3, 1}, {2, 5, 1}, {5, 1, 1}, {5, 5, 1}};
    const Index64 totalPointCount(positions.size() * 3);

    {
        PointDataGrid::Ptr grid1;
        PointDataGrid::Ptr grid2;
        PointDataGrid::Ptr grid3;

        {
            grid1 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
            PointDataTree& tree1 = grid1->tree();

            appendGroup(tree1, "a1");
            appendGroup(tree1, "a2");
            appendGroup(tree1, "a5");
            appendGroup(tree1, "a6");
            appendGroup(tree1, "a7");

            grid2 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
            PointDataTree& tree2 = grid2->tree();

            appendGroup(tree2, "a3");
            appendGroup(tree2, "a5");
            appendGroup(tree2, "a1");
            appendGroup(tree2, "a2");
            appendGroup(tree2, "a8");

            grid3 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
            PointDataTree& tree3 = grid3->tree();

            appendGroup(tree3, "a4");
            appendGroup(tree3, "a2");
            appendGroup(tree3, "a3");
            appendGroup(tree3, "a1");
            appendGroup(tree3, "a9");
        }

        //   grid1 has:
        //    - a1, a2, a5, a6, a7
        //   grid2 has:
        //    - a3, a5, a1, a2, a8
        //   grid3 has:
        //    - a4, a2, a3, a1, a9

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);
        grids.push_back(grid3);

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());
        EXPECT_TRUE(!grid3->tree().cbeginLeaf());
        EXPECT_EQ(totalPointCount, pointCount(result->tree()));

        const auto leafIter = result->tree().cbeginLeaf();
        EXPECT_TRUE(leafIter);

        const auto& desc = leafIter->attributeSet().descriptor();

        EXPECT_TRUE(desc.hasGroup("a1"));
        EXPECT_TRUE(desc.hasGroup("a2"));
        EXPECT_TRUE(desc.hasGroup("a3"));
        EXPECT_TRUE(desc.hasGroup("a4"));
        EXPECT_TRUE(desc.hasGroup("a5"));
        EXPECT_TRUE(desc.hasGroup("a6"));
        EXPECT_TRUE(desc.hasGroup("a7"));
        EXPECT_TRUE(desc.hasGroup("a8"));
        EXPECT_TRUE(desc.hasGroup("a9"));
    }

    {
        PointDataGrid::Ptr grid1;
        PointDataGrid::Ptr grid2;
        PointDataGrid::Ptr grid3;

        {
            grid1 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform2);
            PointDataTree& tree1 = grid1->tree();

            appendGroup(tree1, "a1");
            appendGroup(tree1, "a2");
            appendGroup(tree1, "a5");
            appendGroup(tree1, "a6");
            appendGroup(tree1, "a7");

            grid2 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform2);
            PointDataTree& tree2 = grid2->tree();

            appendGroup(tree2, "a3");
            appendGroup(tree2, "a5");
            appendGroup(tree2, "a1");
            appendGroup(tree2, "a2");
            appendGroup(tree2, "a8");

            grid3 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform2);
            PointDataTree& tree3 = grid3->tree();

            appendGroup(tree3, "a4");
            appendGroup(tree3, "a2");
            appendGroup(tree3, "a3");
            appendGroup(tree3, "a1");
            appendGroup(tree3, "a9");
        }

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);
        grids.push_back(grid3);

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());
        EXPECT_TRUE(!grid3->tree().cbeginLeaf());
        EXPECT_EQ(totalPointCount, pointCount(result->tree()));

        const auto leafIter = result->tree().cbeginLeaf();
        EXPECT_TRUE(leafIter);

        const auto& desc = leafIter->attributeSet().descriptor();

        EXPECT_TRUE(desc.hasGroup("a1"));
        EXPECT_TRUE(desc.hasGroup("a2"));
        EXPECT_TRUE(desc.hasGroup("a3"));
        EXPECT_TRUE(desc.hasGroup("a4"));
        EXPECT_TRUE(desc.hasGroup("a5"));
        EXPECT_TRUE(desc.hasGroup("a6"));
        EXPECT_TRUE(desc.hasGroup("a7"));
        EXPECT_TRUE(desc.hasGroup("a8"));
        EXPECT_TRUE(desc.hasGroup("a9"));
    }
}

TEST_F(TestPointMerge, testCompressionMerge)
{
    // five points across four leafs with transform1, all the in same leaf with
    // transform2

    math::Transform::Ptr transform1(math::Transform::createLinearTransform(1.0));
    math::Transform::Ptr transform2(math::Transform::createLinearTransform(10.0));
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 3, 1}, {2, 5, 1}, {5, 1, 1}, {5, 5, 1}};
    const Index64 totalPointCount(positions.size() * 2);

    using PositionType = TypedAttributeArray<Vec3f, FixedPointCodec<false>>;

    {
        PointDataGrid::Ptr grid1 = createPointDataGrid<FixedPointCodec<false>, PointDataGrid>(positions, *transform1);
        PointDataGrid::Ptr grid2 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);
        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());
        EXPECT_EQ(totalPointCount, pointCount(result->tree()));

        const auto leafIter = result->tree().cbeginLeaf();
        EXPECT_TRUE(leafIter);
        EXPECT_TRUE(leafIter->hasAttribute("P"));
        EXPECT_TRUE(leafIter->attributeArray("P").isType<PositionType>());
    }

    {
        PointDataGrid::Ptr grid1 = createPointDataGrid<FixedPointCodec<false>, PointDataGrid>(positions, *transform2);
        PointDataGrid::Ptr grid2 = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform2);

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(grid1);
        grids.push_back(grid2);
        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

        EXPECT_TRUE(result->tree().cbeginLeaf());
        EXPECT_TRUE(!grid2->tree().cbeginLeaf());
        EXPECT_EQ(totalPointCount, pointCount(result->tree()));

        const auto leafIter = result->tree().cbeginLeaf();
        EXPECT_TRUE(leafIter);
        EXPECT_TRUE(leafIter->hasAttribute("P"));
        EXPECT_TRUE(leafIter->attributeArray("P").isType<PositionType>());
    }
}

TEST_F(TestPointMerge, testStringMerge)
{
    std::vector<Vec3s> positions1{{1, 1, 1}, {1, 3, 1}, {2, 5, 1}};
    std::vector<Vec3s> positions2{{1, 2, 1}, {100, 3, 1}, {5, 2, 8}};

    std::vector<std::string> str1{"abc", "def", "foo"};
    std::vector<std::string> str2{"bar", "ijk", "def"};

    math::Transform::Ptr transform(math::Transform::createLinearTransform(1.0));

    PointAttributeVector<Vec3s> posWrapper1(positions1);
    PointAttributeVector<Vec3s> posWrapper2(positions2);

    tools::PointIndexGrid::Ptr indexGrid1 = tools::createPointIndexGrid<tools::PointIndexGrid>(posWrapper1, *transform);
    PointDataGrid::Ptr grid1 = createPointDataGrid<NullCodec, PointDataGrid>(*indexGrid1, posWrapper1, *transform);

    tools::PointIndexGrid::Ptr indexGrid2 = tools::createPointIndexGrid<tools::PointIndexGrid>(posWrapper2, *transform);
    PointDataGrid::Ptr grid2 = createPointDataGrid<NullCodec, PointDataGrid>(*indexGrid2, posWrapper2, *transform);

    appendAttribute<std::string>(grid1->tree(), "test");
    appendAttribute<std::string>(grid2->tree(), "test");

    PointAttributeVector<std::string> strWrapper1(str1);
    PointAttributeVector<std::string> strWrapper2(str2);

    populateAttribute<PointDataTree, tools::PointIndexTree>(
        grid1->tree(), indexGrid1->tree(), "test", strWrapper1);
    populateAttribute<PointDataTree, tools::PointIndexTree>(
        grid2->tree(), indexGrid2->tree(), "test", strWrapper2);

    std::vector<PointDataGrid::Ptr> grids;
    grids.push_back(grid1);
    grids.push_back(grid2);
    PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);

    auto leaf = result->tree().cbeginLeaf();
    const MetaMap& meta = leaf->attributeSet().descriptor().getMetadata();
    StringMetaCache metaCache(meta);
    const auto& cacheMap = metaCache.map();

    // each expected string occurs once and once only

    EXPECT_EQ(metaCache.size(), size_t(5));
    EXPECT_TRUE(cacheMap.find("abc") != cacheMap.end());
    EXPECT_TRUE(cacheMap.find("def") != cacheMap.end());
    EXPECT_TRUE(cacheMap.find("foo") != cacheMap.end());
    EXPECT_TRUE(cacheMap.find("bar") != cacheMap.end());
    EXPECT_TRUE(cacheMap.find("ijk") != cacheMap.end());

    // create array of strings in use

    std::vector<std::string> stringValues;

    for (auto leafIter = result->tree().cbeginLeaf(); leafIter; ++leafIter) {
        StringAttributeHandle handle(leafIter->constAttributeArray("test"), leafIter->attributeSet().descriptor().getMetadata());
        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            std::string str = handle.get(*indexIter);
            stringValues.push_back(str);
        }
    }

    EXPECT_EQ(stringValues.size(), size_t(6));
    EXPECT_EQ(Index64(1), Index64(std::count(stringValues.cbegin(), stringValues.cend(), "abc")));
    EXPECT_EQ(Index64(2), Index64(std::count(stringValues.cbegin(), stringValues.cend(), "def")));
    EXPECT_EQ(Index64(1), Index64(std::count(stringValues.cbegin(), stringValues.cend(), "foo")));
    EXPECT_EQ(Index64(1), Index64(std::count(stringValues.cbegin(), stringValues.cend(), "bar")));
    EXPECT_EQ(Index64(1), Index64(std::count(stringValues.cbegin(), stringValues.cend(), "ijk")));
}

TEST_F(TestPointMerge, testSphereMerge)
{
    { // merge ten spheres with points scattered inside

        Vec3s center(0, 0, 0);
        float radius = 10;
        float voxelSize = 0.2f;

        const math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);

        FloatGrid::Ptr sphere =
            tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);

        PointDataGrid::Ptr points =
            points::denseUniformPointScatter(*sphere, /*pointsPerVoxel=*/8);

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(points);

        for (int i = 1; i < 10; i++) {
            FloatGrid::Ptr otherSphere =
                tools::createLevelSetSphere<FloatGrid>(radius, center + Vec3s(0, static_cast<float>(i), 0), voxelSize);

            grids.push_back(
                points::denseUniformPointScatter(*otherSphere, /*pointsPerVoxel=*/8));
        }

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids);
    }

    { // merge two spheres where the first has id with non-zero default value, the second has no id
        Vec3s center(0, 0, 0);
        float radius = 1;
        float voxelSize = 0.2f;

        const math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);

        FloatGrid::Ptr sphere =
            tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);

        PointDataGrid::Ptr points =
            points::denseUniformPointScatter(*sphere, /*pointsPerVoxel=*/8);

        // append an id attribute with default value of 5

        TypedMetadata<int> meta(5);

        points::appendAttribute<int>(points->tree(), "id",
                                    /*uniformValue*/0,
                                    /*stride=*/1,
                                    /*constantStride=*/true,
                                    /*defaultValue*/&meta);

        int id(1);
        Index64 idTotal(0);

        for (auto leaf = points->tree().beginLeaf(); leaf; ++leaf) {
            AttributeWriteHandle<int> handle(leaf->attributeArray("id"));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                idTotal += id;
                handle.set(*iter, id++);
            }
        }

        std::vector<PointDataGrid::Ptr> grids;
        grids.push_back(points);

        FloatGrid::Ptr otherSphere =
            tools::createLevelSetSphere<FloatGrid>(radius, center + Vec3s(0, 1.0f, 0), voxelSize);

        grids.push_back(
            points::denseUniformPointScatter(*otherSphere, /*pointsPerVoxel=*/4));

        Index64 pointCount1 = points::pointCount(grids[0]->tree());
        Index64 pointCount2 = points::pointCount(grids[1]->tree());

        // append default value ids

        idTotal += pointCount2 * 5;

        std::vector<PointDataGrid::ConstPtr> constGrids;

        PointDataGrid::Ptr result = mergePoints<PointDataGrid>(grids, constGrids, false);

        Index64 pointCount3 = points::pointCount(result->tree());

        EXPECT_EQ(pointCount3, pointCount1+pointCount2);

        Index64 newIdTotal(0);

        for (auto leaf = result->tree().beginLeaf(); leaf; ++leaf) {
            AttributeHandle<int> handle(leaf->constAttributeArray("id"));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                int newId = handle.get(*iter);
                newIdTotal += newId;
            }
        }

        EXPECT_EQ(idTotal, newIdTotal);
    }
}
