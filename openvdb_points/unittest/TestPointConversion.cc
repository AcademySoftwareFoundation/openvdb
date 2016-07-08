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

#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointAttribute.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/tools/PointCount.h>
#include <openvdb_points/tools/PointGroup.h>
#include <openvdb_points/openvdb.h>

using namespace openvdb;
using namespace openvdb::tools;

class TestPointConversion: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointConversion);
    CPPUNIT_TEST(testPointConversion);

    CPPUNIT_TEST_SUITE_END();

    void testPointConversion();

}; // class TestPointConversion

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointConversion);


// Simple Attribute Wrapper
template <typename T>
struct AttributeWrapper
{
    typedef T ValueType;
    typedef T PosType;
    typedef T value_type;

    struct Handle
    {
        Handle(AttributeWrapper<T>& attribute)
            : mBuffer(attribute.mAttribute) { }

        template <typename ValueType>
        void set(openvdb::Index offset, const ValueType& value) {
            mBuffer[offset] = value;
        }

        template <typename ValueType>
        void set(openvdb::Index offset, const openvdb::math::Vec3<ValueType>& value) {
            mBuffer[offset] = value;
        }

    private:
        std::vector<T>& mBuffer;
    }; // struct Handle

    AttributeWrapper() { }

    void expand() { }
    void compact() { }

    void resize(const size_t n) { mAttribute.resize(n); }
    size_t size() const { return mAttribute.size(); }

    std::vector<T>& buffer() { return mAttribute; }

    template <typename ValueT>
    void get(size_t n, ValueT& value) const { value = mAttribute[n]; }
    template <typename ValueT>
    void getPos(size_t n, ValueT& value) const { this->get<ValueT>(n, value); }

private:
    std::vector<T> mAttribute;
}; // struct AttributeWrapper


struct GroupWrapper
{
    GroupWrapper() { }

    void setOffsetOn(openvdb::Index index) {
        mGroup[index] = short(1);
    }

    void finalize() { }

    void resize(const size_t n) { mGroup.resize(n, short(0)); }
    size_t size() const { return mGroup.size(); }

    std::vector<short>& buffer() { return mGroup; }

private:
    std::vector<short> mGroup;
}; // struct GroupWrapper


struct PointData
{
    int id;
    Vec3f position;
    float uniform;
    openvdb::Name string;
    short group;

    bool operator<(const PointData& other) const { return id < other.id; }
}; // PointData


// Generate random points by uniformly distributing points
// on a unit-sphere.
void genPoints( const int numPoints, const double scale,
                AttributeWrapper<Vec3f>& position,
                AttributeWrapper<int>& id,
                AttributeWrapper<float>& uniform,
                AttributeWrapper<openvdb::Name>& string,
                GroupWrapper& group)
{
    // init
    openvdb::math::Random01 randNumber(0);
    const int n = int(std::sqrt(double(numPoints)));
    const double xScale = (2.0 * M_PI) / double(n);
    const double yScale = M_PI / double(n);

    double x, y, theta, phi;
    openvdb::Vec3f pos;

    position.resize(n*n);
    id.resize(n*n);
    uniform.resize(n*n);
    string.resize(n*n);
    group.resize(n*n);

    AttributeWrapper<Vec3f>::Handle positionHandle(position);
    AttributeWrapper<int>::Handle idHandle(id);
    AttributeWrapper<float>::Handle uniformHandle(uniform);
    AttributeWrapper<openvdb::Name>::Handle stringHandle(string);

    int i = 0;

    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {

            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();

            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi   = x * xScale; // [0 to 2PI]

            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            pos[0] = std::sin(theta)*std::cos(phi)*scale;
            pos[1] = std::sin(theta)*std::sin(phi)*scale;
            pos[2] = std::cos(theta)*scale;

            positionHandle.set(i, pos);
            idHandle.set(i, i);
            uniformHandle.set(i, 100.0f);

            // add points with even id to the group
            if ((i % 2) == 0) {
                group.setOffsetOn(i);
                stringHandle.set(i, "testA");
            }
            else {
                stringHandle.set(i, "testB");
            }

            i++;
        }
    }
}


////////////////////////////////////////


void
TestPointConversion::testPointConversion()
{
    // Define and register some common attribute types
    typedef TypedAttributeArray<int32_t>        AttributeI;
    typedef TypedAttributeArray<float>          AttributeF;
    typedef TypedAttributeArray<openvdb::Vec3s> AttributeVec3s;

    AttributeI::registerType();
    AttributeF::registerType();
    AttributeVec3s::registerType();

    // generate points

    const unsigned long count(40000);

    AttributeWrapper<Vec3f> position;
    AttributeWrapper<int> id;
    AttributeWrapper<float> uniform;
    AttributeWrapper<openvdb::Name> string;
    GroupWrapper group;

    genPoints(count, /*scale=*/ 100.0, position, id, uniform, string, group);

    CPPUNIT_ASSERT_EQUAL(position.size(), count);
    CPPUNIT_ASSERT_EQUAL(id.size(), count);
    CPPUNIT_ASSERT_EQUAL(uniform.size(), count);
    CPPUNIT_ASSERT_EQUAL(string.size(), count);
    CPPUNIT_ASSERT_EQUAL(group.size(), count);

    // convert point positions into a Point Data Grid

    const float voxelSize = 1.0f;
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(voxelSize));

    PointIndexGrid::Ptr pointIndexGrid = createPointIndexGrid<PointIndexGrid>(position, *transform);
    PointDataGrid::Ptr pointDataGrid = createPointDataGrid<PointDataGrid>(*pointIndexGrid, position,
                                            AttributeVec3s::attributeType(), *transform);

    PointIndexTree& indexTree = pointIndexGrid->tree();
    PointDataTree& tree = pointDataGrid->tree();

    // add id and populate

    appendAttribute<AttributeI>(tree, "id");
    populateAttribute(tree, pointIndexGrid->tree(), "id", id);

    // add uniform and populate

    appendAttribute<AttributeF>(tree, "uniform");
    populateAttribute(tree, pointIndexGrid->tree(), "uniform", uniform);

    // add string and populate

    appendAttribute<StringAttributeArray>(tree, "string");

    // extract the metadata and reset the descriptors
    PointDataTree::LeafIter leafIter = tree.beginLeaf();
    const AttributeSet::Descriptor& descriptor = leafIter->attributeSet().descriptor();
    AttributeSet::Descriptor::Ptr newDescriptor(new AttributeSet::Descriptor(descriptor));
    MetaMap& metadata = newDescriptor->getMetadata();
    for (; leafIter; ++leafIter) {
        leafIter->resetDescriptor(newDescriptor);
    }

    // insert the required strings into the metadata
    StringMetaInserter inserter(metadata);
    inserter.insert("testA");
    inserter.insert("testB");

    populateAttribute(tree, pointIndexGrid->tree(), "string", string);

    // add group and set membership

    appendGroup(tree, "test");
    setGroup(tree, indexTree, group.buffer(), "test");

    CPPUNIT_ASSERT_EQUAL(indexTree.leafCount(), tree.leafCount());

    // create accessor and iterator for Point Data Tree

    PointDataTree::LeafCIter leafCIter = tree.cbeginLeaf();

    CPPUNIT_ASSERT_EQUAL((unsigned long) 5, leafCIter->attributeSet().size());

    CPPUNIT_ASSERT(leafCIter->attributeSet().find("id") != AttributeSet::INVALID_POS);
    CPPUNIT_ASSERT(leafCIter->attributeSet().find("uniform") != AttributeSet::INVALID_POS);
    CPPUNIT_ASSERT(leafCIter->attributeSet().find("P") != AttributeSet::INVALID_POS);
    CPPUNIT_ASSERT(leafCIter->attributeSet().find("string") != AttributeSet::INVALID_POS);

    const size_t idIndex = leafCIter->attributeSet().find("id");
    const size_t uniformIndex = leafCIter->attributeSet().find("uniform");
    const size_t stringIndex = leafCIter->attributeSet().find("string");
    const AttributeSet::Descriptor::GroupIndex groupIndex = leafCIter->attributeSet().groupIndex("test");

    // convert back into linear point attribute data

    AttributeWrapper<Vec3f> outputPosition;
    AttributeWrapper<int> outputId;
    AttributeWrapper<float> outputUniform;
    AttributeWrapper<openvdb::Name> outputString;
    GroupWrapper outputGroup;

    // test offset the whole point block by an arbitrary amount

    Index64 startOffset = 10;

    outputPosition.resize(startOffset + position.size());
    outputId.resize(startOffset + id.size());
    outputUniform.resize(startOffset + uniform.size());
    outputString.resize(startOffset + string.size());
    outputGroup.resize(startOffset + group.size());

    std::vector<Index64> pointOffsets;
    getPointOffsets(pointOffsets, tree);

    convertPointDataGridPosition(outputPosition, *pointDataGrid, pointOffsets, startOffset);
    convertPointDataGridAttribute(outputId, tree, pointOffsets, startOffset, idIndex);
    convertPointDataGridAttribute(outputUniform, tree, pointOffsets, startOffset, uniformIndex);
    convertPointDataGridAttribute(outputString, tree, pointOffsets, startOffset, stringIndex);
    convertPointDataGridGroup(outputGroup, tree, pointOffsets, startOffset, groupIndex);

    // pack and sort the new buffers based on id

    std::vector<PointData> pointData;

    pointData.resize(count);

    for (unsigned int i = 0; i < count; i++) {
        pointData[i].id = outputId.buffer()[startOffset + i];
        pointData[i].position = outputPosition.buffer()[startOffset + i];
        pointData[i].uniform = outputUniform.buffer()[startOffset + i];
        pointData[i].string = outputString.buffer()[startOffset + i];
        pointData[i].group = outputGroup.buffer()[startOffset + i];
    }

    std::sort(pointData.begin(), pointData.end());

    // compare old and new buffers

    for (unsigned int i = 0; i < count; i++)
    {
        CPPUNIT_ASSERT_EQUAL(id.buffer()[i], pointData[i].id);
        CPPUNIT_ASSERT_EQUAL(group.buffer()[i], pointData[i].group);
        CPPUNIT_ASSERT_EQUAL(uniform.buffer()[i], pointData[i].uniform);
        CPPUNIT_ASSERT_EQUAL(string.buffer()[i], pointData[i].string);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(position.buffer()[i].x(), pointData[i].position.x(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(position.buffer()[i].y(), pointData[i].position.y(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(position.buffer()[i].z(), pointData[i].position.z(), /*tolerance=*/1e-6);
    }

    // convert based on even group

    const unsigned long halfCount = count / 2;

    outputPosition.resize(startOffset + halfCount);
    outputId.resize(startOffset + halfCount);
    outputUniform.resize(startOffset + halfCount);
    outputString.resize(startOffset + halfCount);
    outputGroup.resize(startOffset + halfCount);

    std::vector<Name> includeGroups;
    includeGroups.push_back("test");

    pointOffsets.clear();
    getPointOffsets(pointOffsets, tree, includeGroups);

    convertPointDataGridPosition(outputPosition, *pointDataGrid, pointOffsets, startOffset, includeGroups);
    convertPointDataGridAttribute(outputId, tree, pointOffsets, startOffset, idIndex, includeGroups);
    convertPointDataGridAttribute(outputUniform, tree, pointOffsets, startOffset, uniformIndex, includeGroups);
    convertPointDataGridAttribute(outputString, tree, pointOffsets, startOffset, stringIndex, includeGroups);
    convertPointDataGridGroup(outputGroup, tree, pointOffsets, startOffset, groupIndex, includeGroups);

    CPPUNIT_ASSERT_EQUAL(size_t(outputPosition.size() - startOffset), size_t(halfCount));
    CPPUNIT_ASSERT_EQUAL(size_t(outputId.size() - startOffset), size_t(halfCount));
    CPPUNIT_ASSERT_EQUAL(size_t(outputUniform.size() - startOffset), size_t(halfCount));
    CPPUNIT_ASSERT_EQUAL(size_t(outputString.size() - startOffset), size_t(halfCount));
    CPPUNIT_ASSERT_EQUAL(size_t(outputGroup.size() - startOffset), size_t(halfCount));

    pointData.clear();

    for (unsigned int i = 0; i < halfCount; i++) {
        PointData data;
        data.id = outputId.buffer()[startOffset + i];
        data.position = outputPosition.buffer()[startOffset + i];
        data.uniform = outputUniform.buffer()[startOffset + i];
        data.string = outputString.buffer()[startOffset + i];
        data.group = outputGroup.buffer()[startOffset + i];
        pointData.push_back(data);
    }

    std::sort(pointData.begin(), pointData.end());

    // compare old and new buffers

    for (unsigned int i = 0; i < halfCount; i++)
    {
        CPPUNIT_ASSERT_EQUAL(id.buffer()[i*2], pointData[i].id);
        CPPUNIT_ASSERT_EQUAL(group.buffer()[i*2], pointData[i].group);
        CPPUNIT_ASSERT_EQUAL(uniform.buffer()[i*2], pointData[i].uniform);
        CPPUNIT_ASSERT_EQUAL(string.buffer()[i*2], pointData[i].string);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(position.buffer()[i*2].x(), pointData[i].position.x(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(position.buffer()[i*2].y(), pointData[i].position.y(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(position.buffer()[i*2].z(), pointData[i].position.z(), /*tolerance=*/1e-6);
    }
}

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
