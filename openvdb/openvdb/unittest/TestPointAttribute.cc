// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/AttributeArrayString.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>

#include <gtest/gtest.h>

#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointAttribute: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointAttribute


////////////////////////////////////////


TEST_F(TestPointAttribute, testAppendDrop)
{
    using AttributeI = TypedAttributeArray<int>;

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    EXPECT_EQ(tree.leafCount(), Index32(4));

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    ++leafIter;
    ++leafIter;
    ++leafIter;

    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    // check just one attribute exists (position)
    EXPECT_EQ(attributeSet.descriptor().size(), size_t(1));

    { // append an attribute, different initial values and collapse
        appendAttribute<int>(tree,  "id");

        EXPECT_TRUE(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array = tree.beginLeaf()->attributeArray("id");
        EXPECT_TRUE(array.isUniform());
        EXPECT_EQ(AttributeI::cast(array).get(0), zeroVal<AttributeI::ValueType>());

        dropAttribute(tree, "id");

        appendAttribute<int>(tree, "id", 10, /*stride*/1);

        EXPECT_TRUE(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array2 = tree.beginLeaf()->attributeArray("id");
        EXPECT_TRUE(array2.isUniform());
        EXPECT_EQ(AttributeI::cast(array2).get(0), AttributeI::ValueType(10));

        array2.expand();
        EXPECT_TRUE(!array2.isUniform());

        collapseAttribute<int>(tree, "id", 50);

        AttributeArray& array3 = tree.beginLeaf()->attributeArray("id");
        EXPECT_TRUE(array3.isUniform());
        EXPECT_EQ(AttributeI::cast(array3).get(0), AttributeI::ValueType(50));

        dropAttribute(tree, "id");

        appendAttribute<Name>(tree, "name", "test");

        AttributeArray& array4 = tree.beginLeaf()->attributeArray("name");
        EXPECT_TRUE(array4.isUniform());
        StringAttributeHandle handle(array4, attributeSet.descriptor().getMetadata());
        EXPECT_EQ(handle.get(0), Name("test"));

        dropAttribute(tree, "name");
    }

    { // append a strided attribute
        appendAttribute<int>(tree, "id", 0, /*stride=*/1);

        AttributeArray& array = tree.beginLeaf()->attributeArray("id");
        EXPECT_EQ(array.stride(), Index(1));

        dropAttribute(tree, "id");

        appendAttribute<int>(tree, "id", 0, /*stride=*/10);

        EXPECT_TRUE(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array2 = tree.beginLeaf()->attributeArray("id");
        EXPECT_EQ(array2.stride(), Index(10));

        dropAttribute(tree, "id");
    }

    { // append an attribute, check descriptors are as expected, default value test
        TypedMetadata<int> meta(10);
        appendAttribute<int>(tree,  "id",
                                /*uniformValue*/0,
                                /*stride=*/1,
                                /*constantStride=*/true,
                                /*defaultValue*/&meta,
                                /*hidden=*/false, /*transient=*/false);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(2));
        EXPECT_TRUE(attributeSet.descriptor() == attributeSet4.descriptor());
        EXPECT_TRUE(&attributeSet.descriptor() == &attributeSet4.descriptor());

        EXPECT_TRUE(attributeSet.descriptor().getMetadata()["default:id"]);

        AttributeArray& array = tree.beginLeaf()->attributeArray("id");
        EXPECT_TRUE(array.isUniform());

        AttributeHandle<int> handle(array);
        EXPECT_EQ(0, handle.get(0));
    }

    { // append three attributes, check ordering is consistent with insertion
        appendAttribute<float>(tree, "test3");
        appendAttribute<float>(tree, "test1");
        appendAttribute<float>(tree, "test2");

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(5));

        EXPECT_EQ(attributeSet.descriptor().find("P"), size_t(0));
        EXPECT_EQ(attributeSet.descriptor().find("id"), size_t(1));
        EXPECT_EQ(attributeSet.descriptor().find("test3"), size_t(2));
        EXPECT_EQ(attributeSet.descriptor().find("test1"), size_t(3));
        EXPECT_EQ(attributeSet.descriptor().find("test2"), size_t(4));
    }

    { // drop an attribute by index, check ordering remains consistent
        std::vector<size_t> indices{2};

        dropAttributes(tree, indices);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(4));

        EXPECT_EQ(attributeSet.descriptor().find("P"), size_t(0));
        EXPECT_EQ(attributeSet.descriptor().find("id"), size_t(1));
        EXPECT_EQ(attributeSet.descriptor().find("test1"), size_t(2));
        EXPECT_EQ(attributeSet.descriptor().find("test2"), size_t(3));
    }

    { // drop attributes by index, check ordering remains consistent
        std::vector<size_t> indices{1, 3};

        dropAttributes(tree, indices);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(2));

        EXPECT_EQ(attributeSet.descriptor().find("P"), size_t(0));
        EXPECT_EQ(attributeSet.descriptor().find("test1"), size_t(1));
    }

    { // drop last non-position attribute
        std::vector<size_t> indices{1};

        dropAttributes(tree, indices);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(1));
    }

    { // attempt (and fail) to drop position
        std::vector<size_t> indices{0};

        EXPECT_THROW(dropAttributes(tree, indices), openvdb::KeyError);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(1));
        EXPECT_TRUE(attributeSet.descriptor().find("P") != AttributeSet::INVALID_POS);
    }

    { // add back previous attributes
        appendAttribute<int>(tree, "id");
        appendAttribute<float>(tree, "test3");
        appendAttribute<float>(tree, "test1");
        appendAttribute<float>(tree, "test2");

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(5));
    }

    { // attempt (and fail) to drop non-existing attribute
        std::vector<Name> names{"test1000"};

        EXPECT_THROW(dropAttributes(tree, names), openvdb::KeyError);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(5));
    }

    { // drop by name
        std::vector<Name> names{"test1", "test2"};

        dropAttributes(tree, names);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(3));
        EXPECT_TRUE(attributeSet.descriptor() == attributeSet4.descriptor());
        EXPECT_TRUE(&attributeSet.descriptor() == &attributeSet4.descriptor());

        EXPECT_EQ(attributeSet.descriptor().find("P"), size_t(0));
        EXPECT_EQ(attributeSet.descriptor().find("id"), size_t(1));
        EXPECT_EQ(attributeSet.descriptor().find("test3"), size_t(2));
    }

    { // attempt (and fail) to drop position
        std::vector<Name> names{"P"};

        EXPECT_THROW(dropAttributes(tree, names), openvdb::KeyError);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(3));
        EXPECT_TRUE(attributeSet.descriptor().find("P") != AttributeSet::INVALID_POS);
    }

    { // drop one attribute by name
        dropAttribute(tree, "test3");

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(2));
        EXPECT_EQ(attributeSet.descriptor().find("P"), size_t(0));
        EXPECT_EQ(attributeSet.descriptor().find("id"), size_t(1));
    }

    { // drop one attribute by id
        dropAttribute(tree, 1);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(1));
        EXPECT_EQ(attributeSet.descriptor().find("P"), size_t(0));
    }

    { // attempt to add an attribute with a name that already exists
        appendAttribute<float>(tree, "test3");
        EXPECT_THROW(appendAttribute<float>(tree, "test3"), openvdb::KeyError);

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(2));
    }

    { // attempt to add an attribute with an unregistered type (Vec2R)
        EXPECT_THROW(appendAttribute<Vec2R>(tree, "unregistered"), openvdb::KeyError);
    }

    { // append attributes marked as hidden, transient, group and string
        appendAttribute<float>(tree, "testHidden", 0,
            /*stride=*/1, /*constantStride=*/true, nullptr, true, false);
        appendAttribute<float>(tree, "testTransient", 0,
            /*stride=*/1, /*constantStride=*/true, nullptr, false, true);
        appendAttribute<Name>(tree, "testString", "",
            /*stride=*/1, /*constantStride=*/true, nullptr, false, false);

        const AttributeArray& arrayHidden = leafIter->attributeArray("testHidden");
        const AttributeArray& arrayTransient = leafIter->attributeArray("testTransient");
        const AttributeArray& arrayString = leafIter->attributeArray("testString");

        EXPECT_TRUE(arrayHidden.isHidden());
        EXPECT_TRUE(!arrayTransient.isHidden());

        EXPECT_TRUE(!arrayHidden.isTransient());
        EXPECT_TRUE(arrayTransient.isTransient());
        EXPECT_TRUE(!arrayString.isTransient());

        EXPECT_TRUE(!isGroup(arrayHidden));
        EXPECT_TRUE(!isGroup(arrayTransient));
        EXPECT_TRUE(!isGroup(arrayString));

        EXPECT_TRUE(!isString(arrayHidden));
        EXPECT_TRUE(!isString(arrayTransient));
        EXPECT_TRUE(isString(arrayString));
    }

    { // collapsing non-existing attribute throws exception
        EXPECT_THROW(collapseAttribute<int>(tree, "unknown", 0), openvdb::KeyError);
        EXPECT_THROW(collapseAttribute<Name>(tree, "unknown", "unknown"), openvdb::KeyError);
    }
}

TEST_F(TestPointAttribute, testRename)
{
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    EXPECT_EQ(tree.leafCount(), Index32(4));

    const openvdb::TypedMetadata<float> defaultValue(5.0f);

    appendAttribute<float>(tree, "test1", 0,
        /*stride=*/1, /*constantStride=*/true, &defaultValue);
    appendAttribute<int>(tree, "id");
    appendAttribute<float>(tree, "test2");

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();
    ++leafIter;
    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    { // rename one attribute
        renameAttribute(tree, "test1", "test1renamed");

        EXPECT_EQ(attributeSet.descriptor().size(), size_t(4));
        EXPECT_TRUE(attributeSet.descriptor().find("test1") == AttributeSet::INVALID_POS);
        EXPECT_TRUE(attributeSet.descriptor().find("test1renamed") != AttributeSet::INVALID_POS);

        EXPECT_EQ(attributeSet4.descriptor().size(), size_t(4));
        EXPECT_TRUE(attributeSet4.descriptor().find("test1") == AttributeSet::INVALID_POS);
        EXPECT_TRUE(attributeSet4.descriptor().find("test1renamed") != AttributeSet::INVALID_POS);

        renameAttribute(tree, "test1renamed", "test1");
    }

    { // rename non-existing, matching and existing attributes
        EXPECT_THROW(renameAttribute(tree, "nonexist", "newname"), openvdb::KeyError);
        EXPECT_THROW(renameAttribute(tree, "test1", "test1"), openvdb::KeyError);
        EXPECT_THROW(renameAttribute(tree, "test2", "test1"), openvdb::KeyError);
    }

    { // rename multiple attributes
        std::vector<Name> oldNames{"test1", "test2"};
        std::vector<Name> newNames{"test1renamed"};

        EXPECT_THROW(renameAttributes(tree, oldNames, newNames), openvdb::ValueError);

        newNames.push_back("test2renamed");
        renameAttributes(tree, oldNames, newNames);

        renameAttribute(tree, "test1renamed", "test1");
        renameAttribute(tree, "test2renamed", "test2");
    }

    { // rename an attribute with a default value
        EXPECT_TRUE(attributeSet.descriptor().hasDefaultValue("test1"));

        renameAttribute(tree, "test1", "test1renamed");

        EXPECT_TRUE(attributeSet.descriptor().hasDefaultValue("test1renamed"));
    }
}

TEST_F(TestPointAttribute, testBloscCompress)
{
    std::vector<Vec3s> positions;
    for (float i = 1.f; i < 6.f; i += 0.1f) {
        positions.emplace_back(1, i, 1);
        positions.emplace_back(1, 1, i);
        positions.emplace_back(10, i, 1);
        positions.emplace_back(10, 1, i);
    }

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check two leaves
    EXPECT_EQ(tree.leafCount(), Index32(2));

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.beginLeaf();
    auto leafIter2 = ++tree.beginLeaf();

    { // append an attribute, check descriptors are as expected
        appendAttribute<int>(tree, "compact");
        appendAttribute<int>(tree, "id");
        appendAttribute<int>(tree, "id2");
    }

    using AttributeHandleRWI = AttributeWriteHandle<int>;

    { // set some id values (leaf 1)
        AttributeHandleRWI handleCompact(leafIter->attributeArray("compact"));
        AttributeHandleRWI handleId(leafIter->attributeArray("id"));
        AttributeHandleRWI handleId2(leafIter->attributeArray("id2"));

        const int size = leafIter->attributeArray("id").size();

        EXPECT_EQ(size, 102);

        for (int i = 0; i < size; i++) {
            handleCompact.set(i, 5);
            handleId.set(i, i);
            handleId2.set(i, i);
        }
    }

    { // set some id values (leaf 2)
        AttributeHandleRWI handleCompact(leafIter2->attributeArray("compact"));
        AttributeHandleRWI handleId(leafIter2->attributeArray("id"));
        AttributeHandleRWI handleId2(leafIter2->attributeArray("id2"));

        const int size = leafIter2->attributeArray("id").size();

        EXPECT_EQ(size, 102);

        for (int i = 0; i < size; i++) {
            handleCompact.set(i, 10);
            handleId.set(i, i);
            handleId2.set(i, i);
        }
    }

    compactAttributes(tree);

    EXPECT_TRUE(leafIter->attributeArray("compact").isUniform());
    EXPECT_TRUE(leafIter2->attributeArray("compact").isUniform());
}
