///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////


#include <cppunit/extensions/HelperMacros.h>
#include <openvdb_points/tools/AttributeArrayString.h>
#include <openvdb_points/tools/PointAttribute.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/openvdb.h>

#include <iostream>
#include <sstream>

using namespace openvdb;
using namespace openvdb::tools;

class TestPointAttribute: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointAttribute);
    CPPUNIT_TEST(testAppendDrop);
    CPPUNIT_TEST(testRename);
    CPPUNIT_TEST(testBloscCompress);

    CPPUNIT_TEST_SUITE_END();

    void testAppendDrop();
    void testRename();
    void testBloscCompress();
}; // class TestPointAttribute

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointAttribute);

////////////////////////////////////////


void
TestPointAttribute::testAppendDrop()
{
    using AttributeF = TypedAttributeArray<float>;
    using AttributeI = TypedAttributeArray<int>;

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    ++leafIter;
    ++leafIter;
    ++leafIter;

    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    // check just one attribute exists (position)
    CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));

    { // append an attribute, different initial values and collapse
        appendAttribute<AttributeI>(tree,  "id");

        CPPUNIT_ASSERT(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(array.isUniform());
        CPPUNIT_ASSERT_EQUAL(AttributeI::cast(array).get(0), zeroVal<AttributeI::ValueType>());

        dropAttribute(tree, "id");

        appendAttribute<AttributeI>(tree,  "id", /*stride*/1,
                                    AttributeI::ValueType(10));

        CPPUNIT_ASSERT(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array2 = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(array2.isUniform());
        CPPUNIT_ASSERT_EQUAL(AttributeI::cast(array2).get(0), AttributeI::ValueType(10));

        array2.expand();
        CPPUNIT_ASSERT(!array2.isUniform());

        collapseAttribute<AttributeI>(tree, "id", AttributeI::ValueType(50));

        AttributeArray& array3 = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(array3.isUniform());
        CPPUNIT_ASSERT_EQUAL(AttributeI::cast(array3).get(0), AttributeI::ValueType(50));

        dropAttribute(tree, "id");
    }

    { // append a strided attribute
        appendAttribute<AttributeI>(tree, "id", /*stride=*/1);

        AttributeArray& array = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(!array.isStrided());
        CPPUNIT_ASSERT_EQUAL(array.stride(), Index(1));

        dropAttribute(tree, "id");

        appendAttribute<AttributeI>(tree, "id", /*stride=*/10);

        CPPUNIT_ASSERT(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array2 = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(array2.isStrided());
        CPPUNIT_ASSERT_EQUAL(array2.stride(), Index(10));

        dropAttribute(tree, "id");
    }

    { // append an attribute, check descriptors are as expected, default value test
        appendAttribute<AttributeI>(tree,  "id",
                                    /*stride=*/1,
                                    /*uniformValue*/zeroVal<AttributeI::ValueType>(),
                                    /*defaultValue*/TypedMetadata<AttributeI::ValueType>(AttributeI::ValueType(10)).copy(),
                                    /*hidden=*/false, /*transient=*/false);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
        CPPUNIT_ASSERT(attributeSet.descriptor() == attributeSet4.descriptor());
        CPPUNIT_ASSERT(&attributeSet.descriptor() == &attributeSet4.descriptor());

        CPPUNIT_ASSERT(attributeSet.descriptor().getMetadata()["default:id"]);
    }

    { // append three attributes, check ordering is consistent with insertion
        appendAttribute<AttributeF>(tree, "test3");
        appendAttribute<AttributeF>(tree, "test1");
        appendAttribute<AttributeF>(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test3"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(3));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test2"), size_t(4));
    }

    { // drop an attribute by index, check ordering remains consistent
        std::vector<size_t> indices{2};

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(4));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test2"), size_t(3));
    }

    { // drop attributes by index, check ordering remains consistent
        std::vector<size_t> indices{1, 3};

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(1));
    }

    { // drop last non-position attribute
        std::vector<size_t> indices{1};

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
    }

    { // attempt (and fail) to drop position
        std::vector<size_t> indices{0};

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, indices), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
        CPPUNIT_ASSERT(attributeSet.descriptor().find("P") != AttributeSet::INVALID_POS);
    }

    { // add back previous attributes
        appendAttribute<AttributeI>(tree, "id");
        appendAttribute<AttributeF>(tree, "test3");
        appendAttribute<AttributeF>(tree, "test1");
        appendAttribute<AttributeF>(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));
    }

    { // attempt (and fail) to drop non-existing attribute
        std::vector<Name> names{"test1000"};

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, names), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));
    }

    { // drop by name
        std::vector<Name> names{"test1", "test2"};

        dropAttributes(tree, names);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(3));
        CPPUNIT_ASSERT(attributeSet.descriptor() == attributeSet4.descriptor());
        CPPUNIT_ASSERT(&attributeSet.descriptor() == &attributeSet4.descriptor());

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test3"), size_t(2));
    }

    { // attempt (and fail) to drop position
        std::vector<Name> names{"P"};

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, names), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(3));
        CPPUNIT_ASSERT(attributeSet.descriptor().find("P") != AttributeSet::INVALID_POS);
    }

    { // drop one attribute by name
        dropAttribute(tree, "test3");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
    }

    { // drop one attribute by id
        dropAttribute(tree, 1);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
    }

    { // attempt to add an attribute with a name that already exists
        appendAttribute<AttributeF>(tree, "test3");
        CPPUNIT_ASSERT_THROW(appendAttribute<AttributeF>(tree, "test3"), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
    }

    { // append attributes marked as hidden, transient, group and string
        appendAttribute<AttributeF>(tree, "testHidden", /*stride=*/1, zeroVal<AttributeF::ValueType>(), Metadata::Ptr(), true, false);
        appendAttribute<AttributeF>(tree, "testTransient", /*stride=*/1, zeroVal<AttributeF::ValueType>(), Metadata::Ptr(), false, true);
        appendAttribute<GroupAttributeArray>(tree, "testGroup", /*stride=*/1, zeroVal<GroupAttributeArray::ValueType>(), Metadata::Ptr(), false, false);
        appendAttribute<StringAttributeArray>(tree, "testString", /*stride=*/1, zeroVal<StringAttributeArray::ValueType>(), Metadata::Ptr(), false, false);

        const AttributeArray& arrayHidden = leafIter->attributeArray("testHidden");
        const AttributeArray& arrayTransient = leafIter->attributeArray("testTransient");
        const AttributeArray& arrayGroup = leafIter->attributeArray("testGroup");
        const AttributeArray& arrayString = leafIter->attributeArray("testString");

        CPPUNIT_ASSERT(arrayHidden.isHidden());
        CPPUNIT_ASSERT(!arrayTransient.isHidden());
        CPPUNIT_ASSERT(!arrayGroup.isHidden());

        CPPUNIT_ASSERT(!arrayHidden.isTransient());
        CPPUNIT_ASSERT(arrayTransient.isTransient());
        CPPUNIT_ASSERT(!arrayGroup.isTransient());
        CPPUNIT_ASSERT(!arrayString.isTransient());

        CPPUNIT_ASSERT(!isGroup(arrayHidden));
        CPPUNIT_ASSERT(!isGroup(arrayTransient));
        CPPUNIT_ASSERT(isGroup(arrayGroup));
        CPPUNIT_ASSERT(!isGroup(arrayString));

        CPPUNIT_ASSERT(!isString(arrayHidden));
        CPPUNIT_ASSERT(!isString(arrayTransient));
        CPPUNIT_ASSERT(!isString(arrayGroup));
        CPPUNIT_ASSERT(isString(arrayString));
    }
}

void
TestPointAttribute::testRename()
{
    using AttributeF = TypedAttributeArray<float>;
    using AttributeI = TypedAttributeArray<int>;

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    const openvdb::TypedMetadata<float> defaultValue(5.0f);

    appendAttribute<AttributeF>(tree, "test1", /*stride=*/1, zeroVal<AttributeF::ValueType>(), defaultValue.copy());
    appendAttribute<AttributeI>(tree, "id");
    appendAttribute<AttributeF>(tree, "test2");

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();
    ++leafIter;
    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    { // rename one attribute
        renameAttribute(tree, "test1", "test1renamed");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(4));
        CPPUNIT_ASSERT(attributeSet.descriptor().find("test1") == AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(attributeSet.descriptor().find("test1renamed") != AttributeSet::INVALID_POS);

        CPPUNIT_ASSERT_EQUAL(attributeSet4.descriptor().size(), size_t(4));
        CPPUNIT_ASSERT(attributeSet4.descriptor().find("test1") == AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(attributeSet4.descriptor().find("test1renamed") != AttributeSet::INVALID_POS);

        renameAttribute(tree, "test1renamed", "test1");
    }

    { // rename non-existing, matching and existing attributes
        CPPUNIT_ASSERT_THROW(renameAttribute(tree, "nonexist", "newname"), openvdb::KeyError);
        CPPUNIT_ASSERT_THROW(renameAttribute(tree, "test1", "test1"), openvdb::KeyError);
        CPPUNIT_ASSERT_THROW(renameAttribute(tree, "test2", "test1"), openvdb::KeyError);
    }

    { // rename multiple attributes
        std::vector<Name> oldNames{"test1", "test2"};
        std::vector<Name> newNames{"test1renamed"};

        CPPUNIT_ASSERT_THROW(renameAttributes(tree, oldNames, newNames), openvdb::ValueError);

        newNames.push_back("test2renamed");
        renameAttributes(tree, oldNames, newNames);

        renameAttribute(tree, "test1renamed", "test1");
        renameAttribute(tree, "test2renamed", "test2");
    }

    { // don't rename group attributes
        appendAttribute<GroupAttributeArray>(tree, "testGroup", /*stride=*/1,
                                            zeroVal<GroupAttributeArray::ValueType>(), Metadata::Ptr());
        CPPUNIT_ASSERT_THROW(renameAttribute(tree, "testGroup", "testGroup2"), openvdb::KeyError);
    }

    { // rename an attribute with a default value
        CPPUNIT_ASSERT(attributeSet.descriptor().hasDefaultValue("test1"));

        renameAttribute(tree, "test1", "test1renamed");

        CPPUNIT_ASSERT(attributeSet.descriptor().hasDefaultValue("test1renamed"));
    }
}

void
TestPointAttribute::testBloscCompress()
{
    using AttributeI = TypedAttributeArray<int>;

    std::vector<Vec3s> positions;
    for (float i = 1; i < 6; i+= 0.1) {
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
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(2));

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.beginLeaf();
    auto leafIter2 = ++tree.beginLeaf();

    { // append an attribute, check descriptors are as expected
        appendAttribute<AttributeI>(tree, "compact");
        appendAttribute<AttributeI>(tree, "id");
        appendAttribute<AttributeI>(tree, "id2");
    }

    using AttributeHandleRWI = AttributeWriteHandle<int>;

    { // set some id values (leaf 1)
        AttributeHandleRWI handleCompact(leafIter->attributeArray("compact"));
        AttributeHandleRWI handleId(leafIter->attributeArray("id"));
        AttributeHandleRWI handleId2(leafIter->attributeArray("id2"));

        const int size = leafIter->attributeArray("id").size();

        CPPUNIT_ASSERT_EQUAL(size, 102);

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

        CPPUNIT_ASSERT_EQUAL(size, 102);

        for (int i = 0; i < size; i++) {
            handleCompact.set(i, 10);
            handleId.set(i, i);
            handleId2.set(i, i);
        }
    }

    compactAttributes(tree);

    CPPUNIT_ASSERT(leafIter->attributeArray("compact").isUniform());
    CPPUNIT_ASSERT(leafIter2->attributeArray("compact").isUniform());

    bloscCompressAttribute(tree, "id");

#ifdef OPENVDB_USE_BLOSC
    CPPUNIT_ASSERT(leafIter->attributeArray("id").isCompressed());
    CPPUNIT_ASSERT(!leafIter->attributeArray("id2").isCompressed());
    CPPUNIT_ASSERT(leafIter2->attributeArray("id").isCompressed());
    CPPUNIT_ASSERT(!leafIter2->attributeArray("id2").isCompressed());

    CPPUNIT_ASSERT(leafIter->attributeArray("id").memUsage() < leafIter->attributeArray("id2").memUsage());
#endif
}


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
