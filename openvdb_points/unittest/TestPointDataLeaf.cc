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
#include <openvdb_points/openvdb.h>
#include <openvdb/openvdb.h>

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
    CPPUNIT_TEST(testAttributes);
    CPPUNIT_TEST(testTopologyCopy);
    CPPUNIT_TEST(testEquivalence);
    CPPUNIT_TEST(testIterators);
    CPPUNIT_TEST(testReadWriteCompression);
    CPPUNIT_TEST(testIO);
    CPPUNIT_TEST(testSwap);
    CPPUNIT_TEST(testCopyOnWrite);
    CPPUNIT_TEST(testCopyDescriptor);
    CPPUNIT_TEST_SUITE_END();

    void testEmptyLeaf();
    void testOffsets();
    void testSetValue();
    void testMonotonicity();
    void testAttributes();
    void testTopologyCopy();
    void testEquivalence();
    void testIterators();
    void testReadWriteCompression();
    void testIO();
    void testSwap();
    void testCopyOnWrite();
    void testCopyDescriptor();

private:
    
}; // class TestPointDataLeaf

using openvdb::tools::PointDataTree;
using openvdb::tools::PointDataGrid;
using LeafType = PointDataTree::LeafNodeType;
using ValueType = LeafType::ValueType;
using BufferType = LeafType::Buffer;
using openvdb::Index;
using openvdb::Index64;

namespace {

bool
matchingNamePairs(const openvdb::NamePair& lhs,
                  const openvdb::NamePair& rhs)
{
    if (lhs.first != rhs.first)     return false;
    if (lhs.second != rhs.second)     return false;

    return true;
}

bool
zeroLeafValues(const LeafType* leafNode)
{
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

    for (auto iter = leafNode.cbeginValueOn(); iter; ++iter) {
        if (previous > int(*iter))  return false;
        previous = int(*iter);
    }

    return true;
}

// (borrowed from PointIndexGrid unit test)

class PointList
{
public:
    using PosType       = openvdb::Vec3R;
    using value_type    = openvdb::Vec3R;

    PointList(const std::vector<openvdb::Vec3R>& points)
        : mPoints(&points)
    {
    }

    size_t size() const {
        return mPoints->size();
    }

    void getPos(size_t n, openvdb::Vec3R& xyz) const {
        xyz = (*mPoints)[n];
    }

protected:
    std::vector<openvdb::Vec3R> const * const mPoints;
}; // PointList

// Generate random points by uniformly distributing points
// on a unit-sphere.
// (borrowed from PointIndexGrid unit test)
std::vector<openvdb::Vec3R> genPoints(const int numPoints)
{
    // init
    openvdb::math::Random01 randNumber(0);
    const int n = int(std::sqrt(double(numPoints)));
    const double xScale = (2.0 * M_PI) / double(n);
    const double yScale = M_PI / double(n);

    double x, y, theta, phi;

    std::vector<openvdb::Vec3R> points;
    points.reserve(n*n);

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
            points.emplace_back(    std::sin(theta)*std::cos(phi),
                                    std::sin(theta)*std::sin(phi),
                                    std::cos(theta) );
        }
    }

    return points;
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

        delete leafNode;
    }

    // offsets for one point per voxel (active = false)

    {
        LeafType* leafNode = new LeafType();

        for (openvdb::Index i = 0; i < LeafType::SIZE; i++) {
            leafNode->setOffsetOnly(i, i);
        }

        CPPUNIT_ASSERT(leafNode->getValue(10) == 10);
        CPPUNIT_ASSERT(leafNode->isEmpty());

        delete leafNode;
    }

    // test bulk offset replacement without activity mask update

    {
        LeafType* leafNode = new LeafType();

        for (openvdb::Index i = 0; i < LeafType::SIZE; ++i) {
            leafNode->setOffsetOn(i, 10);
        }

        std::vector<LeafType::ValueType> newOffsets(LeafType::SIZE);

        leafNode->setOffsets(newOffsets, /*updateValueMask*/false);

        const LeafType::NodeMaskType& valueMask = leafNode->getValueMask();
        for (openvdb::Index i = 0; i < LeafType::SIZE; ++i ) {
            CPPUNIT_ASSERT(valueMask.isOn(i));
        }

        delete leafNode;
    }

    // test bulk offset replacement with activity mask update

    {
        LeafType* leafNode = new LeafType();

        for (openvdb::Index i = 0; i < LeafType::SIZE; ++i) {
            leafNode->setOffsetOn(i, 10);
        }

        std::vector<LeafType::ValueType> newOffsets(LeafType::SIZE);

        leafNode->setOffsets(newOffsets, /*updateValueMask*/true);

        const LeafType::NodeMaskType& valueMask = leafNode->getValueMask();
        for (openvdb::Index i = 0; i < LeafType::SIZE; ++i ) {
            CPPUNIT_ASSERT(valueMask.isOff(i));
        }

        delete leafNode;
    }

    // ensure bulk offset replacement fails when vector size doesn't equal number of voxels

    {
        LeafType* leafNode = new LeafType();

        std::vector<LeafType::ValueType> newOffsets;
        CPPUNIT_ASSERT_THROW(leafNode->setOffsets(newOffsets), openvdb::ValueError);

        delete leafNode;
    }

    // test offset validation

    {
        using AttributeVec3s    = openvdb::tools::TypedAttributeArray<openvdb::Vec3s>;
        using AttributeS        = openvdb::tools::TypedAttributeArray<float>;
        using Descriptor        = openvdb::tools::AttributeSet::Descriptor;

        // empty Descriptor should throw on leaf node initialize
        auto emptyDescriptor = std::make_shared<Descriptor>();
        LeafType* emptyLeafNode = new LeafType();
        CPPUNIT_ASSERT_THROW(emptyLeafNode->initializeAttributes(emptyDescriptor, 5), openvdb::IndexError);

        // create a non-empty Descriptor
        Descriptor::Ptr descriptor = Descriptor::create(AttributeVec3s::attributeType());

        // ensure validateOffsets succeeds for monotonically increasing offsets that fully
        // utilise the underlying attribute arrays

        {
            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            descriptor = descriptor->duplicateAppend("density", AttributeS::attributeType());
            leafNode->appendAttribute(leafNode->attributeSet().descriptor(), descriptor, descriptor->find("density"));

            std::vector<LeafType::ValueType> offsets(LeafType::SIZE);
            offsets.back() = numAttributes;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_NO_THROW(leafNode->validateOffsets());
            delete leafNode;
        }

        // ensure validateOffsets detects non-monotonic offset values

        {
            LeafType* leafNode = new LeafType();

            std::vector<LeafType::ValueType> offsets(LeafType::SIZE);
            *offsets.begin() = 1;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_THROW(leafNode->validateOffsets(), openvdb::ValueError);
            delete leafNode;
        }

        // ensure validateOffsets detects inconsistent attribute array sizes

        {
            using openvdb::tools::AttributeSet;

            descriptor = Descriptor::create(AttributeVec3s::attributeType());

            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            descriptor = descriptor->duplicateAppend("density", AttributeS::attributeType());
            leafNode->appendAttribute(leafNode->attributeSet().descriptor(), descriptor, descriptor->find("density"));

            AttributeSet* newSet = new AttributeSet(leafNode->attributeSet(), numAttributes);
            newSet->replace("density", AttributeS::create(numAttributes+1));
            leafNode->swap(newSet);

            std::vector<LeafType::ValueType> offsets(LeafType::SIZE);
            offsets.back() = numAttributes;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_THROW(leafNode->validateOffsets(), openvdb::ValueError);
            delete leafNode;
        }

        // ensure validateOffsets detects unused attributes (e.g. final voxel offset not
        // equal to size of attribute arrays)

        {
            descriptor = Descriptor::create(AttributeVec3s::attributeType());

            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            descriptor = descriptor->duplicateAppend("density", AttributeS::attributeType());
            leafNode->appendAttribute(leafNode->attributeSet().descriptor(), descriptor, descriptor->find("density"));

            std::vector<LeafType::ValueType> offsets(LeafType::SIZE);
            offsets.back() = numAttributes - 1;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_THROW(leafNode->validateOffsets(), openvdb::ValueError);
            delete leafNode;
        }

        // ensure validateOffsets detects out-of-bounds offset values

        {
            descriptor = Descriptor::create(AttributeVec3s::attributeType());

            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            descriptor = descriptor->duplicateAppend("density", AttributeS::attributeType());
            leafNode->appendAttribute(leafNode->attributeSet().descriptor(), descriptor, descriptor->find("density"));

            std::vector<LeafType::ValueType> offsets(LeafType::SIZE);
            offsets.back() = numAttributes + 1;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_THROW(leafNode->validateOffsets(), openvdb::ValueError);
            delete leafNode;
        }
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

    for (unsigned int i = 0; i < LeafType::DIM; i++) {
        for (unsigned int j = 0; j < LeafType::DIM; j++) {
            for (unsigned int k = 0; k < LeafType::DIM; k++) {
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
TestPointDataLeaf::testAttributes()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types
    using AttributeVec3s    = openvdb::tools::TypedAttributeArray<openvdb::Vec3s>;
    using AttributeI        = openvdb::tools::TypedAttributeArray<int32_t>;

    AttributeVec3s::registerType();
    AttributeI::registerType();

    // create a descriptor

    using Descriptor = openvdb::tools::AttributeSet::Descriptor;

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3s::attributeType());

    // create a leaf and initialize attributes using this descriptor

    LeafType leaf(openvdb::Coord(0, 0, 0));

    CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().size(), size_t(0));

    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

    descrA = descrA->duplicateAppend("id", AttributeI::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("id"));

    CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().size(), size_t(2));

    {
        const AttributeArray* array = leaf.attributeSet().get(/*pos=*/0);

        CPPUNIT_ASSERT_EQUAL(array->size(), size_t(100));
    }

    // manually set a voxel

    leaf.setOffsetOn(LeafType::SIZE - 1, 10);

    CPPUNIT_ASSERT(!zeroLeafValues(&leaf));

    // neither dense nor empty

    CPPUNIT_ASSERT(!leaf.isDense());
    CPPUNIT_ASSERT(!leaf.isEmpty());

    // clear the attributes and check voxel values are zero but value mask is not touched

    leaf.clearAttributes(/*updateValueMask=*/ false);

    CPPUNIT_ASSERT(!leaf.isDense());
    CPPUNIT_ASSERT(!leaf.isEmpty());

    CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().size(), size_t(2));
    CPPUNIT_ASSERT(zeroLeafValues(&leaf));

    // call clearAttributes again, updating the value mask and check it is now inactive

    leaf.clearAttributes();

    CPPUNIT_ASSERT(leaf.isEmpty());

    // ensure arrays are uniform

    const AttributeArray* array0 = leaf.attributeSet().get(/*pos=*/0);
    const AttributeArray* array1 = leaf.attributeSet().get(/*pos=*/1);

    CPPUNIT_ASSERT_EQUAL(array0->size(), size_t(1));
    CPPUNIT_ASSERT_EQUAL(array1->size(), size_t(1));

    // test leaf returns expected result for hasAttribute()

    CPPUNIT_ASSERT(leaf.hasAttribute(/*pos*/0));
    CPPUNIT_ASSERT(leaf.hasAttribute("P"));

    CPPUNIT_ASSERT(leaf.hasAttribute(/*pos*/1));
    CPPUNIT_ASSERT(leaf.hasAttribute("id"));

    CPPUNIT_ASSERT(!leaf.hasAttribute(/*pos*/2));
    CPPUNIT_ASSERT(!leaf.hasAttribute("test"));

    // test underlying attributeArray can be accessed by name and index, and that their types are as expected.

    const LeafType* constLeaf = &leaf;

    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos*/0).type(), AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("P").type(), AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos*/1).type(), AttributeI::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("id").type(), AttributeI::attributeType()));

    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray(/*pos*/0).type(), AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray("P").type(), AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray(/*pos*/1).type(), AttributeI::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray("id").type(), AttributeI::attributeType()));

    // check invalid pos or name throws

    CPPUNIT_ASSERT_THROW(leaf.attributeArray(/*pos=*/3), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.attributeArray("not_there"), openvdb::LookupError);

    CPPUNIT_ASSERT_THROW(constLeaf->attributeArray(/*pos=*/3), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->attributeArray("not_there"), openvdb::LookupError);

    // test leaf can be successfully cast to TypedAttributeArray and check types

    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos=*/0).type(),
                         AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("P").type(),
                         AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos=*/1).type(),
                         AttributeI::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("id").type(),
                         AttributeI::attributeType()));

    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray(/*pos=*/0).type(),
                         AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray("P").type(),
                         AttributeVec3s::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray(/*pos=*/1).type(),
                         AttributeI::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray("id").type(),
                         AttributeI::attributeType()));

    // check invalid pos or name throws

    CPPUNIT_ASSERT_THROW(leaf.attributeArray(/*pos=*/2), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.attributeArray("test"), openvdb::LookupError);

    CPPUNIT_ASSERT_THROW(constLeaf->attributeArray(/*pos=*/2), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->attributeArray("test"), openvdb::LookupError);

    // check memory usage = attribute set + base leaf

    // leaf.initializeAttributes(descrA, /*arrayLength=*/100);

    const LeafType::BaseLeaf& baseLeaf = static_cast<LeafType::BaseLeaf&>(leaf);

    const Index64 memUsage = baseLeaf.memUsage() + leaf.attributeSet().memUsage();

    CPPUNIT_ASSERT_EQUAL(memUsage, leaf.memUsage());
}


void
TestPointDataLeaf::testTopologyCopy()
{
    // test topology copy from a float Leaf

    {
        using FloatLeaf = openvdb::FloatTree::LeafNodeType;

        // create a float leaf and activate some values

        FloatLeaf floatLeaf(openvdb::Coord(0, 0, 0));

        floatLeaf.setValueOn(1);
        floatLeaf.setValueOn(4);
        floatLeaf.setValueOn(7);
        floatLeaf.setValueOn(8);

        CPPUNIT_ASSERT_EQUAL(floatLeaf.onVoxelCount(), Index64(4));

        // validate construction of a PointDataLeaf using a TopologyCopy

        LeafType leaf(floatLeaf, 0, openvdb::TopologyCopy());

        CPPUNIT_ASSERT_EQUAL(leaf.onVoxelCount(), Index64(4));

        LeafType leaf2(openvdb::Coord(8, 8, 8));

        leaf2.setValueOn(1);
        leaf2.setValueOn(4);
        leaf2.setValueOn(7);

        CPPUNIT_ASSERT(!leaf.hasSameTopology(&leaf2));

        leaf2.setValueOn(8);

        CPPUNIT_ASSERT(leaf.hasSameTopology(&leaf2));

        // validate construction of a PointDataLeaf using an Off-On TopologyCopy

        LeafType leaf3(floatLeaf, 1, 2, openvdb::TopologyCopy());

        CPPUNIT_ASSERT_EQUAL(leaf3.onVoxelCount(), Index64(4));
    }

    // test topology copy from a PointIndexLeaf

    {
        // generate points
        // (borrowed from PointIndexGrid unit test)

        const float voxelSize = 0.01f;
        const openvdb::math::Transform::Ptr transform =
                openvdb::math::Transform::createLinearTransform(voxelSize);

        std::vector<openvdb::Vec3R> points = genPoints(40000);

        PointList pointList(points);

        // construct point index grid

        using PointIndexGrid = openvdb::tools::PointIndexGrid;

        PointIndexGrid::Ptr pointGridPtr =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);

        auto iter = pointGridPtr->tree().cbeginLeaf();

        CPPUNIT_ASSERT(iter);

        // check that the active voxel counts match for all leaves

        for ( ; iter; ++iter) {
            LeafType leaf(*iter);

            CPPUNIT_ASSERT_EQUAL(iter->onVoxelCount(), leaf.onVoxelCount());
        }
    }
}


void
TestPointDataLeaf::testEquivalence()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types

    using AttributeVec3s    = TypedAttributeArray<openvdb::Vec3s>;
    using AttributeF        = TypedAttributeArray<float>;
    using AttributeI        = TypedAttributeArray<int32_t>;

    AttributeVec3s::registerType();
    AttributeF::registerType();
    AttributeI::registerType();

    // create a descriptor

    using Descriptor = AttributeSet::Descriptor;

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3s::attributeType());

    // create a leaf and initialize attributes using this descriptor

    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

    descrA = descrA->duplicateAppend("density", AttributeF::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("density"));

    descrA = descrA->duplicateAppend("id", AttributeI::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("id"));

    // manually activate some voxels

    leaf.setValueOn(1);
    leaf.setValueOn(4);
    leaf.setValueOn(7);

    // manually change some values in the density array

    TypedAttributeArray<float>& attr = TypedAttributeArray<float>::cast(leaf.attributeArray("density"));

    attr.set(0, 5.0f);
    attr.set(50, 2.0f);
    attr.set(51, 8.1f);

    // check deep copy construction (topology and attributes)

    {
        LeafType leaf2(leaf);

        CPPUNIT_ASSERT_EQUAL(leaf.onVoxelCount(), leaf2.onVoxelCount());
        CPPUNIT_ASSERT(leaf.hasSameTopology(&leaf2));

        CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().size(), leaf2.attributeSet().size());
        CPPUNIT_ASSERT_EQUAL(leaf.attributeSet().get(0)->size(), leaf2.attributeSet().get(0)->size());
    }

    // check equivalence

    {
        LeafType leaf2(leaf);

        CPPUNIT_ASSERT(leaf == leaf2);

        leaf2.setOrigin(openvdb::Coord(0, 8, 0));

        CPPUNIT_ASSERT(leaf != leaf2);
    }

    {
        LeafType leaf2(leaf);

        CPPUNIT_ASSERT(leaf == leaf2);

        leaf2.setValueOn(10);

        CPPUNIT_ASSERT(leaf != leaf2);
    }
}


void
TestPointDataLeaf::testIterators()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    // Define and register some common attribute types

    using AttributeVec3s    = TypedAttributeArray<openvdb::Vec3s>;
    using AttributeF        = TypedAttributeArray<float>;

    AttributeVec3s::registerType();
    AttributeF::registerType();

    // create a descriptor

    using Descriptor = AttributeSet::Descriptor;

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3s::attributeType());

    // create a leaf and initialize attributes using this descriptor

    const size_t size = LeafType::NUM_VOXELS;

    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/size/2);

    descrA = descrA->duplicateAppend("density", AttributeF::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("density"));

    { // uniform monotonic offsets, only even active
        int offset = 0;

        for (size_t i = 0; i < size; i++)
        {
            if ((i % 2) == 0) {
                leaf.setOffsetOn(i, ++offset);
            }
            else {
                leaf.setOffsetOnly(i, ++offset);
                leaf.setValueOff(i);
            }
        }
    }

    { // test index on
        LeafType::IndexOnIter iterOn(leaf.beginIndexOn());
        CPPUNIT_ASSERT_EQUAL(iterCount(iterOn), Index64(size/2));
        for (int i = 0; iterOn; ++iterOn, i += 2)       CPPUNIT_ASSERT_EQUAL(*iterOn, Index32(i));
    }

    { // test index off
        LeafType::IndexOffIter iterOff(leaf.beginIndexOff());
        CPPUNIT_ASSERT_EQUAL(iterCount(iterOff), Index64(size/2));
        for (int i = 1; iterOff; ++iterOff, i += 2)     CPPUNIT_ASSERT_EQUAL(*iterOff, Index32(i));
    }

    { // test index all
        LeafType::IndexAllIter iterAll(leaf.beginIndexAll());
        CPPUNIT_ASSERT_EQUAL(iterCount(iterAll), Index64(size));
        for (int i = 0; iterAll; ++iterAll, ++i)        CPPUNIT_ASSERT_EQUAL(*iterAll, Index32(i));
    }

}


void
TestPointDataLeaf::testReadWriteCompression()
{
    using namespace openvdb;

    util::NodeMask<3> valueMask;
    util::NodeMask<3> childMask;

    {  // simple read/write test
        std::stringstream ss;

        Index count = 8*8*8;
        std::unique_ptr<PointDataIndex32[]> srcBuf(new PointDataIndex32[count]);

        for (Index i = 0; i < count; i++)  srcBuf[i] = i;

        {
            io::writeCompressedValues(ss, srcBuf.get(), count, valueMask, childMask, false);

            std::unique_ptr<PointDataIndex32[]> destBuf(new PointDataIndex32[count]);

            io::readCompressedValues(ss, destBuf.get(), count, valueMask, false);

            for (Index i = 0; i < count; i++) {
                CPPUNIT_ASSERT_EQUAL(srcBuf.get()[i], destBuf.get()[i]);
            }
        }

#ifndef OPENVDB_USE_BLOSC
        { // write to indicate Blosc compression
            ss.str("");

            uint16_t bytes16(100); // clamp to 16-bit unsigned integer
            ss.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));

            std::unique_ptr<PointDataIndex32[]> destBuf(new PointDataIndex32[count]);
            CPPUNIT_ASSERT_THROW(io::readCompressedValues(ss, destBuf.get(), count, valueMask, false), IoError);
        }
#endif

#ifdef OPENVDB_USE_BLOSC
        { // mis-matching destination bytes cause decompression failures
            std::unique_ptr<PointDataIndex32[]> destBuf(new PointDataIndex32[count]);

            ss.str("");
            io::writeCompressedValues(ss, srcBuf.get(), count, valueMask, childMask, false);
            CPPUNIT_ASSERT_THROW(io::readCompressedValues(ss, destBuf.get(), count+1, valueMask, false), IoError);

            ss.str("");
            io::writeCompressedValues(ss, srcBuf.get(), count, valueMask, childMask, false);
            CPPUNIT_ASSERT_THROW(io::readCompressedValues(ss, destBuf.get(), 1, valueMask, false), IoError);
        }
#endif

        { // seek
            ss.str("");

            io::writeCompressedValues(ss, srcBuf.get(), count, valueMask, childMask, false);

            int test(10772832);
            ss.write(reinterpret_cast<const char*>(&test), sizeof(int));

            PointDataIndex32* buf = nullptr;

            io::readCompressedValues(ss, buf, count, valueMask, false);
            int test2;
            ss.read(reinterpret_cast<char*>(&test2), sizeof(int));

            CPPUNIT_ASSERT_EQUAL(test, test2);
        }
    }

    { // two values for non-compressible example
        std::stringstream ss;

        Index count = 2;
        std::unique_ptr<PointDataIndex32[]> srcBuf(new PointDataIndex32[count]);

        for (Index i = 0; i < count; i++)  srcBuf[i] = i;

        io::writeCompressedValues(ss, srcBuf.get(), count, valueMask, childMask, false);

        std::unique_ptr<PointDataIndex32[]> destBuf(new PointDataIndex32[count]);

        io::readCompressedValues(ss, destBuf.get(), count, valueMask, false);

        for (Index i = 0; i < count; i++) {
            CPPUNIT_ASSERT_EQUAL(srcBuf.get()[i], destBuf.get()[i]);
        }
    }

    { // throw at limit of 16-bit
        std::stringstream ss;
        PointDataIndex32* buf = nullptr;
        Index count = std::numeric_limits<uint16_t>::max();

        CPPUNIT_ASSERT_THROW(io::writeCompressedValues(ss, buf, count, valueMask, childMask, false), IoError);
        CPPUNIT_ASSERT_THROW(io::readCompressedValues(ss, buf, count, valueMask, false), IoError);
    }
}


void
TestPointDataLeaf::testIO()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types

    using AttributeVec3s    = TypedAttributeArray<openvdb::Vec3s>;
    using AttributeF        = TypedAttributeArray<float>;

    AttributeVec3s::registerType();
    AttributeF::registerType();

    // create a descriptor

    using Descriptor = AttributeSet::Descriptor;

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3s::attributeType());

    // create a leaf and initialize attributes using this descriptor

    const size_t size = LeafType::NUM_VOXELS;

    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/size/2);

    descrA = descrA->duplicateAppend("density", AttributeF::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("density"));

    // manually activate some voxels

    leaf.setOffsetOn(1, 10);
    leaf.setOffsetOn(4, 20);
    leaf.setOffsetOn(7, 5);

    // manually change some values in the density array

    TypedAttributeArray<float>& attr = TypedAttributeArray<float>::cast(leaf.attributeArray("density"));

    attr.set(0, 5.0f);
    attr.set(50, 2.0f);
    attr.set(51, 8.1f);

    // read and write topology to disk

    {
        LeafType leaf2(openvdb::Coord(0, 0, 0));

        std::ostringstream ostr(std::ios_base::binary);
        leaf.writeTopology(ostr);

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        leaf2.readTopology(istr);

        // check topology matches

        CPPUNIT_ASSERT_EQUAL(leaf.onVoxelCount(), leaf2.onVoxelCount());
        CPPUNIT_ASSERT(leaf2.isValueOn(4));
        CPPUNIT_ASSERT(!leaf2.isValueOn(5));

        // check only topology (values and attributes still empty)

        CPPUNIT_ASSERT_EQUAL(leaf2.getValue(4), ValueType(0));
        CPPUNIT_ASSERT_EQUAL(leaf2.attributeSet().size(), size_t(0));
    }

    // read and write buffers to disk

    {
        LeafType leaf2(openvdb::Coord(0, 0, 0));

        std::ostringstream ostr(std::ios_base::binary);
        leaf.writeTopology(ostr);
        leaf.writeBuffers(ostr);

        std::istringstream istr(ostr.str(), std::ios_base::binary);

        // Since the input stream doesn't include a VDB header with file format version info,
        // tag the input stream explicitly with the current version number.
        openvdb::io::setCurrentVersion(istr);

        leaf2.readTopology(istr);
        leaf2.readBuffers(istr);

        // check topology matches

        CPPUNIT_ASSERT_EQUAL(leaf.onVoxelCount(), leaf2.onVoxelCount());
        CPPUNIT_ASSERT(leaf2.isValueOn(4));
        CPPUNIT_ASSERT(!leaf2.isValueOn(5));

        // check only topology (values and attributes still empty)

        CPPUNIT_ASSERT_EQUAL(leaf2.getValue(4), ValueType(20));
        CPPUNIT_ASSERT_EQUAL(leaf2.attributeSet().size(), size_t(2));
    }
}


void
TestPointDataLeaf::testSwap()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types

    using AttributeVec3s    = TypedAttributeArray<openvdb::Vec3s>;
    using AttributeF        = TypedAttributeArray<float>;
    using AttributeI        = TypedAttributeArray<int>;

    AttributeVec3s::registerType();
    AttributeF::registerType();

    // create a descriptor

    using Descriptor = AttributeSet::Descriptor;

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3s::attributeType());

    // create a leaf and initialize attributes using this descriptor

    const size_t initialArrayLength = 100;
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/initialArrayLength);

    descrA = descrA->duplicateAppend("density", AttributeF::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("density"));

    descrA = descrA->duplicateAppend("id", AttributeI::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("id"));

    // swap out the underlying attribute set with a new attribute set with a matching
    // descriptor

    CPPUNIT_ASSERT_EQUAL(initialArrayLength, leaf.attributeSet().get("density")->size());
    CPPUNIT_ASSERT_EQUAL(initialArrayLength, leaf.attributeSet().get("id")->size());

    descrA = Descriptor::create(AttributeVec3s::attributeType());

    const size_t newArrayLength = initialArrayLength / 2;

    AttributeSet* newAttributeSet(new AttributeSet(descrA, /*arrayLength*/newArrayLength));

    newAttributeSet->appendAttribute("density", AttributeF::attributeType());
    newAttributeSet->appendAttribute("id", AttributeI::attributeType());

    leaf.swap(newAttributeSet);

    CPPUNIT_ASSERT_EQUAL(newArrayLength, leaf.attributeSet().get("density")->size());
    CPPUNIT_ASSERT_EQUAL(newArrayLength, leaf.attributeSet().get("id")->size());

    // ensure we refuse to swap when the attribute set is null

    CPPUNIT_ASSERT_THROW(leaf.swap(0), openvdb::ValueError);

    // ensure we refuse to swap when the descriptors do not match

    Descriptor::Ptr descrB = Descriptor::create(AttributeVec3s::attributeType());
    AttributeSet* attributeSet = new AttributeSet(descrB, newArrayLength);

    attributeSet->appendAttribute("extra", AttributeF::attributeType());

    CPPUNIT_ASSERT_THROW(leaf.swap(attributeSet), openvdb::ValueError);
    delete attributeSet;
}

void
TestPointDataLeaf::testCopyOnWrite()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types

    using AttributeVec3s    = TypedAttributeArray<openvdb::Vec3s>;
    using AttributeF        = TypedAttributeArray<float>;

    AttributeVec3s::registerType();
    AttributeF::registerType();

    // create a descriptor

    using Descriptor = AttributeSet::Descriptor;

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3s::attributeType());

    // create a leaf and initialize attributes using this descriptor

    const size_t initialArrayLength = 100;
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/initialArrayLength);

    descrA = descrA->duplicateAppend("density", AttributeF::attributeType());
    leaf.appendAttribute(leaf.attributeSet().descriptor(), descrA, descrA->find("density"));

    const AttributeSet& attributeSet = leaf.attributeSet();

    CPPUNIT_ASSERT_EQUAL(attributeSet.size(), size_t(2));

    // ensure attribute arrays are shared between leaf nodes until write

    const LeafType leafCopy(leaf);

    const AttributeSet& attributeSetCopy = leafCopy.attributeSet();

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/1));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/1));

    // test that from a const leaf, accesses to the attribute arrays do not
    // make then unique

    const AttributeArray* constArray = attributeSetCopy.getConst(/*pos=*/1);
    CPPUNIT_ASSERT(constArray);

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/1));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/1));

    constArray = attributeSetCopy.get(/*pos=*/1);

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/1));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/1));

    constArray = &(leafCopy.attributeArray(/*pos=*/1));

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/1));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/1));

    constArray = &(leafCopy.attributeArray("density"));

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/1));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/1));

    // test makeUnique is called from non const getters

    AttributeArray* attributeArray = &(leaf.attributeArray(/*pos=*/1));
    CPPUNIT_ASSERT(attributeArray);

    CPPUNIT_ASSERT(!attributeSet.isShared(/*pos=*/1));
    CPPUNIT_ASSERT(!attributeSetCopy.isShared(/*pos=*/1));
}


void
TestPointDataLeaf::testCopyDescriptor()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types
    using AttributeVec3s    = openvdb::tools::TypedAttributeArray<openvdb::Vec3s>;
    using AttributeS        = openvdb::tools::TypedAttributeArray<float>;

    AttributeVec3s::registerType();
    AttributeS::registerType();

    using LeafNode = PointDataTree::LeafNodeType;

    PointDataTree tree;

    LeafNode* leaf = tree.touchLeaf(openvdb::Coord(0, 0, 0));
    LeafNode* leaf2 = tree.touchLeaf(openvdb::Coord(0, 8, 0));

    // create a descriptor

    using Descriptor = openvdb::tools::AttributeSet::Descriptor;

    Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3s::attributeType());

    // initialize attributes using this descriptor

    leaf->initializeAttributes(descrA, /*arrayLength=*/100);
    leaf2->initializeAttributes(descrA, /*arrayLength=*/50);

    // copy the PointDataTree and ensure that descriptors are shared

    PointDataTree tree2(tree);

    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), openvdb::Index32(2));

    descrA->setGroup("test", size_t(1));

    PointDataTree::LeafCIter iter2 = tree2.cbeginLeaf();
    CPPUNIT_ASSERT(iter2->attributeSet().descriptor().hasGroup("test"));
    ++iter2;
    CPPUNIT_ASSERT(iter2->attributeSet().descriptor().hasGroup("test"));

    // call makeDescriptorUnique and ensure that descriptors are no longer shared

    makeDescriptorUnique(tree2);

    descrA->setGroup("test2", size_t(2));

    iter2 = tree2.cbeginLeaf();
    CPPUNIT_ASSERT(!iter2->attributeSet().descriptor().hasGroup("test2"));
    ++iter2;
    CPPUNIT_ASSERT(!iter2->attributeSet().descriptor().hasGroup("test2"));
}


CPPUNIT_TEST_SUITE_REGISTRATION(TestPointDataLeaf);


////////////////////////////////////////


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
