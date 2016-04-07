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
    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;
    typedef TypedAttributeArray<float>   AttributeF;
    typedef TypedAttributeArray<int>     AttributeI;

    typedef AttributeSet::Descriptor   Descriptor;

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(1, 10, 1));
    positions.push_back(Vec3s(10, 1, 1));
    positions.push_back(Vec3s(10, 10, 1));

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    ++leafIter;
    ++leafIter;
    ++leafIter;

    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    // check just one attribute exists (position)
    CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));

    { // append an attribute, check descriptors are as expected, default value test
        appendAttribute(tree,   Descriptor::NameAndType("id", AttributeI::attributeType()),
                                /*defaultValue*/TypedMetadata<AttributeI::ValueType>(AttributeI::ValueType(10)).copy(),
                                /*hidden=*/false, /*transient=*/false, /*group=*/false);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
        CPPUNIT_ASSERT(attributeSet.descriptor() == attributeSet4.descriptor());
        CPPUNIT_ASSERT(&attributeSet.descriptor() == &attributeSet4.descriptor());

        CPPUNIT_ASSERT(attributeSet.descriptor().getMetadata()["default:id"]);
    }

    { // append three attributes, check ordering is consistent with insertion
        appendAttribute(tree, Descriptor::NameAndType("test3", AttributeF::attributeType()));
        appendAttribute(tree, Descriptor::NameAndType("test1", AttributeF::attributeType()));
        appendAttribute(tree, Descriptor::NameAndType("test2", AttributeF::attributeType()));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test3"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(3));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test2"), size_t(4));
    }

    { // drop an attribute by index, check ordering remains consistent
        std::vector<size_t> indices;
        indices.push_back(2);

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(4));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test2"), size_t(3));
    }

    { // drop attributes by index, check ordering remains consistent
        std::vector<size_t> indices;
        indices.push_back(1);
        indices.push_back(3);

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(1));
    }

    { // drop last non-position attribute
        std::vector<size_t> indices;
        indices.push_back(1);

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
    }

    { // attempt (and fail) to drop position
        std::vector<size_t> indices;
        indices.push_back(0);

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, indices), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
        CPPUNIT_ASSERT(attributeSet.descriptor().find("P") != AttributeSet::INVALID_POS);
    }

    { // add back previous attributes
        appendAttribute(tree, Descriptor::NameAndType("id", AttributeI::attributeType()));
        appendAttribute(tree, Descriptor::NameAndType("test3", AttributeF::attributeType()));
        appendAttribute(tree, Descriptor::NameAndType("test1", AttributeF::attributeType()));
        appendAttribute(tree, Descriptor::NameAndType("test2", AttributeF::attributeType()));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));
    }

    { // attempt (and fail) to drop non-existing attribute
        std::vector<Name> names;
        names.push_back("test1000");

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, names), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));
    }

    { // drop by name
        std::vector<Name> names;
        names.push_back("test1");
        names.push_back("test2");

        dropAttributes(tree, names);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(3));
        CPPUNIT_ASSERT(attributeSet.descriptor() == attributeSet4.descriptor());
        CPPUNIT_ASSERT(&attributeSet.descriptor() == &attributeSet4.descriptor());

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test3"), size_t(2));
    }

    { // attempt (and fail) to drop position
        std::vector<Name> names;
        names.push_back("P");

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
        appendAttribute(tree, Descriptor::NameAndType("test3", AttributeF::attributeType()));
        CPPUNIT_ASSERT_THROW(appendAttribute(tree, Descriptor::NameAndType("test3", AttributeF::attributeType())), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
    }

    { // append attributes marked as hidden, transient and group
        appendAttribute(tree, Descriptor::NameAndType("testHidden", AttributeF::attributeType()), Metadata::Ptr(), true, false, false);
        appendAttribute(tree, Descriptor::NameAndType("testTransient", AttributeF::attributeType()), Metadata::Ptr(), false, true, false);
        appendAttribute(tree, Descriptor::NameAndType("testGroup", GroupAttributeArray::attributeType()), Metadata::Ptr(), false, false, true);

        const AttributeArray& arrayHidden = leafIter->attributeArray("testHidden");
        const AttributeArray& arrayTransient = leafIter->attributeArray("testTransient");
        const AttributeArray& arrayGroup = leafIter->attributeArray("testGroup");

        CPPUNIT_ASSERT(arrayHidden.isHidden());
        CPPUNIT_ASSERT(!arrayTransient.isHidden());
        CPPUNIT_ASSERT(!arrayGroup.isHidden());

        CPPUNIT_ASSERT(!arrayHidden.isTransient());
        CPPUNIT_ASSERT(arrayTransient.isTransient());
        CPPUNIT_ASSERT(!arrayGroup.isTransient());

        CPPUNIT_ASSERT(!GroupAttributeArray::isGroup(arrayHidden));
        CPPUNIT_ASSERT(!GroupAttributeArray::isGroup(arrayTransient));
        CPPUNIT_ASSERT(GroupAttributeArray::isGroup(arrayGroup));
    }
}

void
TestPointAttribute::testRename()
{
    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;
    typedef TypedAttributeArray<float>   AttributeF;
    typedef TypedAttributeArray<int>     AttributeI;

    typedef AttributeSet::Descriptor   Descriptor;

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(1, 10, 1));
    positions.push_back(Vec3s(10, 1, 1));
    positions.push_back(Vec3s(10, 10, 1));

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    const openvdb::TypedMetadata<float> defaultValue(5.0f);

    appendAttribute(tree, Descriptor::NameAndType("test1", AttributeF::attributeType()), defaultValue.copy());
    appendAttribute(tree, Descriptor::NameAndType("id", AttributeI::attributeType()));
    appendAttribute(tree, Descriptor::NameAndType("test2", AttributeF::attributeType()));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
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
        std::vector<Name> oldNames;
        std::vector<Name> newNames;
        oldNames.push_back("test1");
        oldNames.push_back("test2");
        newNames.push_back("test1renamed");

        CPPUNIT_ASSERT_THROW(renameAttributes(tree, oldNames, newNames), openvdb::ValueError);

        newNames.push_back("test2renamed");
        renameAttributes(tree, oldNames, newNames);

        renameAttribute(tree, "test1renamed", "test1");
        renameAttribute(tree, "test2renamed", "test2");
    }

    { // don't rename group attributes
        appendAttribute(tree, Descriptor::NameAndType("testGroup", GroupAttributeArray::attributeType()), Metadata::Ptr(), false, false, true);
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
    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;
    typedef TypedAttributeArray<int>     AttributeI;

    typedef AttributeSet::Descriptor   Descriptor;

    std::vector<Vec3s> positions;
    for (float i = 1; i < 6; i+= 0.1) {
        positions.push_back(Vec3s(1, i, 1));
        positions.push_back(Vec3s(1, 1, i));
        positions.push_back(Vec3s(10, i, 1));
        positions.push_back(Vec3s(10, 1, i));
    }

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // check two leaves
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(2));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafIter leafIter = tree.beginLeaf();
    PointDataTree::LeafIter leafIter2 = ++tree.beginLeaf();

    { // append an attribute, check descriptors are as expected
        appendAttribute(tree, Descriptor::NameAndType("compact", AttributeI::attributeType()));
        appendAttribute(tree, Descriptor::NameAndType("id", AttributeI::attributeType()));
        appendAttribute(tree, Descriptor::NameAndType("id2", AttributeI::attributeType()));
    }

    typedef AttributeWriteHandle<int> AttributeHandleRWI;

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
