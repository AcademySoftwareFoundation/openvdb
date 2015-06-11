///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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

#include <openvdb/tools/PointDataGrid.h>
#include <openvdb/openvdb.h>
#include <openvdb/openvdb_points.h>

class TestPointDataLeaf: public CppUnit::TestCase
{
public:

    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointDataLeaf);
    CPPUNIT_TEST(testEmptyLeaf);
    CPPUNIT_TEST(testOffsets);
    CPPUNIT_TEST(testSetValue);
    CPPUNIT_TEST(testMonotonicity);
    CPPUNIT_TEST(testPointCount);
    CPPUNIT_TEST(testAttributes);
    CPPUNIT_TEST(testTopologyCopy);
    CPPUNIT_TEST_SUITE_END();

    void testEmptyLeaf();
    void testOffsets();
    void testSetValue();
    void testMonotonicity();
    void testPointCount();
    void testAttributes();
    void testTopologyCopy();

}; // class TestPointDataLeaf

using openvdb::tools::PointDataTree;
using openvdb::tools::PointDataGrid;
typedef PointDataTree::LeafNodeType     LeafType;
typedef LeafType::ValueType             ValueType;
typedef LeafType::Buffer                BufferType;
using openvdb::Index;
using openvdb::Index64;

namespace {

bool
zeroLeafValues(const LeafType* leafNode)
{
    const LeafType::ValueType* data = leafNode->buffer().data();

    for (openvdb::Index i = 0; i < LeafType::SIZE; i++) {
        if (leafNode->buffer().getValue(i) != LeafType::ValueType(0))   return false;
    }

    return true;
}

bool
noAttributeData(const LeafType* leafNode)
{
    const openvdb::tools::AttributeSet& attributeSet = leafNode->attributeSet();

    return attributeSet.size() == 0 && attributeSet.descriptor().size() == 0;
}

bool
monotonicOffsets(const LeafType& leafNode)
{
    int previous = -1;

    for (LeafType::ValueOnCIter iter = leafNode.cbeginValueOn(); iter; ++iter) {
        if (previous > int(*iter))  return false;
        previous = int(*iter);
    }

    return true;
}

} // namespace

void
TestPointDataLeaf::testEmptyLeaf()
{
    using namespace openvdb::tools;

    // empty leaf construction

    {
        LeafType* leafNode = new LeafType();

        CPPUNIT_ASSERT(leafNode);
        CPPUNIT_ASSERT(leafNode->isEmpty());
        CPPUNIT_ASSERT(!leafNode->buffer().empty());
        CPPUNIT_ASSERT(zeroLeafValues(leafNode));
        CPPUNIT_ASSERT(noAttributeData(leafNode));
        CPPUNIT_ASSERT(leafNode->origin() == openvdb::Coord(0, 0, 0));

        delete leafNode;
    }

    // empty leaf with non-zero origin construction

    {
        openvdb::Coord coord(20, 30, 40);

        LeafType* leafNode = new LeafType(coord);

        CPPUNIT_ASSERT(leafNode);
        CPPUNIT_ASSERT(leafNode->isEmpty());
        CPPUNIT_ASSERT(!leafNode->buffer().empty());
        CPPUNIT_ASSERT(zeroLeafValues(leafNode));
        CPPUNIT_ASSERT(noAttributeData(leafNode));

        CPPUNIT_ASSERT(leafNode->origin() == openvdb::Coord(16, 24, 40));

        delete leafNode;
    }
}


void
TestPointDataLeaf::testOffsets()
{
    // offsets for one point per voxel (active = true)

    {
        LeafType* leafNode = new LeafType();

        for (openvdb::Index i = 0; i < LeafType::SIZE; i++) {
            leafNode->setOffsetOn(i, i);
        }

        CPPUNIT_ASSERT(leafNode->getValue(10) == 10);
        CPPUNIT_ASSERT(leafNode->isDense());
    }

    // offsets for one point per voxel (active = false)

    {
        LeafType* leafNode = new LeafType();

        for (openvdb::Index i = 0; i < LeafType::SIZE; i++) {
            leafNode->setOffsetOnly(i, i);
        }

        CPPUNIT_ASSERT(leafNode->getValue(10) == 10);
        CPPUNIT_ASSERT(leafNode->isEmpty());
    }
}


void
TestPointDataLeaf::testSetValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));

    openvdb::Coord xyz(0, 0, 0);
    openvdb::Index index(LeafType::coordToOffset(xyz));

    // the following tests are not run when in debug mode
    // due to assertions firing

#ifndef NDEBUG
    return;
#endif

    // ensure all non-modifiable operations are no-ops

    leaf.setValueOnly(xyz, 10);
    leaf.setValueOnly(index, 10);
    leaf.setValueOff(xyz, 10);
    leaf.setValueOff(index, 10);
    leaf.setValueOn(xyz, 10);
    leaf.setValueOn(index, 10);

    struct Local { static inline void op(unsigned int& n) { n = 10; } };

    leaf.modifyValue(xyz, Local::op);
    leaf.modifyValue(index, Local::op);
    leaf.modifyValueAndActiveState(xyz, Local::op);

    CPPUNIT_ASSERT_EQUAL(0, int(leaf.getValue(xyz)));
}


void
TestPointDataLeaf::testMonotonicity()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));

    // assign aggregate values and activate all non-even coordinate sums

    unsigned sum = 0;

    for (int i = 0; i < LeafType::DIM; i++) {
        for (int j = 0; j < LeafType::DIM; j++) {
            for (int k = 0; k < LeafType::DIM; k++) {
                if (((i + j + k) % 2) == 0)     continue;

                leaf.setOffsetOn(LeafType::coordToOffset(openvdb::Coord(i, j, k)), sum++);
            }
        }
    }

    CPPUNIT_ASSERT(monotonicOffsets(leaf));

    // manually change a value and ensure offsets become non-monotonic

    leaf.setOffsetOn(500, 4);

    CPPUNIT_ASSERT(!monotonicOffsets(leaf));
}


void
TestPointDataLeaf::testPointCount()
{
    using namespace openvdb::tools;

    LeafType leaf(openvdb::Coord(0, 0, 0));

    leaf.setOffsetOn(0, 4);
    leaf.setOffsetOn(1, 7);

    CPPUNIT_ASSERT_EQUAL(int(leaf.pointIndex(0).first), 0);
    CPPUNIT_ASSERT_EQUAL(int(leaf.pointIndex(0).second), 4);

    CPPUNIT_ASSERT_EQUAL(int(leaf.pointIndex(1).first), 4);
    CPPUNIT_ASSERT_EQUAL(int(leaf.pointIndex(1).second), 7);

    // one point per voxel

    for (int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, i);
    }

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::All), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::Active), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::Inactive), Index64(0));

    // manually de-activate two voxels

    leaf.setValueOff(100);
    leaf.setValueOff(101);

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::All), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::Active), Index64(LeafType::SIZE - 3));
    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::Inactive), Index64(2));

    // one point per every other voxel and de-activate empty voxels

    unsigned sum = 0;

    for (int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, sum);
        if (i % 2 == 0)     sum++;
    }

    leaf.deactivateEmptyVoxels();

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::All), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::Active), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(point_masks::Inactive), Index64(0));
}


void
TestPointDataLeaf::testAttributes()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>  AttributeS;
    typedef openvdb::tools::TypedAttributeArray<int>    AttributeI;

    AttributeS::registerType();
    AttributeI::registerType();

    // create a descriptor

    typedef openvdb::tools::AttributeSet::Descriptor Descriptor;

    Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());
    names.add("id", AttributeI::attributeType());

    Descriptor::Ptr descrA = Descriptor::create(names.vec);

    // create a leaf and initialize attributes using this descriptor

    LeafType leaf(openvdb::Coord(0, 0, 0));

    CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().size(), size_t(0));

    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

    CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().size(), size_t(2));

    {
        const AttributeArray* array = leaf.attributeSet().get(/*pos=*/0);

        CPPUNIT_ASSERT_EQUAL(array->size(), size_t(100));
    }

    // manually set a voxel

    leaf.setOffsetOn(LeafType::SIZE - 1, 10);

    CPPUNIT_ASSERT(!zeroLeafValues(&leaf));

    // clear the attributes and check voxel values have been zeroed

    leaf.clearAttributes();

    CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().size(), size_t(2));

    CPPUNIT_ASSERT(leaf.isDense());
    CPPUNIT_ASSERT(zeroLeafValues(&leaf));

    leaf.deactivateEmptyVoxels();

    CPPUNIT_ASSERT(leaf.isEmpty());

    // ensure arrays are uniform

    const AttributeArray* array0 = leaf.attributeSet().get(/*pos=*/0);
    const AttributeArray* array1 = leaf.attributeSet().get(/*pos=*/1);

    CPPUNIT_ASSERT_EQUAL(array0->size(), size_t(1));
    CPPUNIT_ASSERT_EQUAL(array1->size(), size_t(1));

    // test leaf returns expected result for hasAttribute()

    CPPUNIT_ASSERT(leaf.hasAttribute<AttributeS>(/*pos=*/0));
    CPPUNIT_ASSERT(!leaf.hasAttribute<AttributeI>(/*pos=*/0));
    CPPUNIT_ASSERT(leaf.hasAttribute<AttributeS>("density"));
    CPPUNIT_ASSERT(!leaf.hasAttribute<AttributeI>("density"));

    CPPUNIT_ASSERT(leaf.hasAttribute<AttributeI>(/*pos=*/1));
    CPPUNIT_ASSERT(!leaf.hasAttribute<AttributeS>(/*pos=*/1));
    CPPUNIT_ASSERT(leaf.hasAttribute<AttributeI>("id"));
    CPPUNIT_ASSERT(!leaf.hasAttribute<AttributeS>("id"));

    CPPUNIT_ASSERT(!leaf.hasAttribute<AttributeS>(/*pos=*/2));
    CPPUNIT_ASSERT(!leaf.hasAttribute<AttributeS>("test"));

    // test leaf can be successfully cast to TypedAttributeArray and check types

    CPPUNIT_ASSERT_EQUAL(leaf.typedAttributeArray<AttributeS>(/*pos=*/0).type(),
                         AttributeS::attributeType());
    CPPUNIT_ASSERT_EQUAL(leaf.typedAttributeArray<AttributeS>("density").type(),
                         AttributeS::attributeType());
    CPPUNIT_ASSERT_EQUAL(leaf.typedAttributeArray<AttributeI>(/*pos=*/1).type(),
                         AttributeI::attributeType());
    CPPUNIT_ASSERT_EQUAL(leaf.typedAttributeArray<AttributeI>("id").type(),
                         AttributeI::attributeType());

    const LeafType* constLeaf = &leaf;

    CPPUNIT_ASSERT_EQUAL(constLeaf->typedAttributeArray<AttributeS>(/*pos=*/0).type(),
                         AttributeS::attributeType());
    CPPUNIT_ASSERT_EQUAL(constLeaf->typedAttributeArray<AttributeS>("density").type(),
                         AttributeS::attributeType());
    CPPUNIT_ASSERT_EQUAL(constLeaf->typedAttributeArray<AttributeI>(/*pos=*/1).type(),
                         AttributeI::attributeType());
    CPPUNIT_ASSERT_EQUAL(constLeaf->typedAttributeArray<AttributeI>("id").type(),
                         AttributeI::attributeType());

    // check invalid type, pos or name throws

    CPPUNIT_ASSERT_THROW(leaf.typedAttributeArray<AttributeI>(/*pos=*/0), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.typedAttributeArray<AttributeI>("density"), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.typedAttributeArray<AttributeS>(/*pos=*/1), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.typedAttributeArray<AttributeS>("id"), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.typedAttributeArray<AttributeS>(/*pos=*/2), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.typedAttributeArray<AttributeS>("test"), openvdb::LookupError);

    CPPUNIT_ASSERT_THROW(constLeaf->typedAttributeArray<AttributeI>(/*pos=*/0), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->typedAttributeArray<AttributeI>("density"), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->typedAttributeArray<AttributeS>(/*pos=*/1), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->typedAttributeArray<AttributeS>("id"), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->typedAttributeArray<AttributeS>(/*pos=*/2), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->typedAttributeArray<AttributeS>("test"), openvdb::LookupError);

    // check memory usage = attribute set + base leaf

    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

    const LeafType::BaseLeaf& baseLeaf = static_cast<LeafType::BaseLeaf&>(leaf);

    const Index64 memUsage = baseLeaf.memUsage() + leaf.attributeSet().memUsage();

    CPPUNIT_ASSERT_EQUAL(memUsage, leaf.memUsage());
}


void
TestPointDataLeaf::testTopologyCopy()
{
    // create a float leaf and activate some values

    typedef openvdb::FloatTree::LeafNodeType FloatLeaf;

    FloatLeaf floatLeaf(openvdb::Coord(0, 0, 0));

    floatLeaf.setValueOn(1);
    floatLeaf.setValueOn(4);
    floatLeaf.setValueOn(7);
    floatLeaf.setValueOn(8);

    CPPUNIT_ASSERT_EQUAL(floatLeaf.onVoxelCount(), Index64(4));

    // validate construction of a PointDataLeaf using a TopologyCopy

    LeafType leaf(floatLeaf, openvdb::TopologyCopy());

    CPPUNIT_ASSERT_EQUAL(leaf.onVoxelCount(), Index64(4));

    LeafType leaf2(openvdb::Coord(8, 8, 8));

    leaf2.setValueOn(1);
    leaf2.setValueOn(4);
    leaf2.setValueOn(7);

    CPPUNIT_ASSERT(!leaf.hasSameTopology(&leaf2));

    leaf2.setValueOn(8);

    CPPUNIT_ASSERT(leaf.hasSameTopology(&leaf2));
}
CPPUNIT_TEST_SUITE_REGISTRATION(TestPointDataLeaf);


////////////////////////////////////////


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
