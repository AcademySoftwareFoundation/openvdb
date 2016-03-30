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
    CPPUNIT_TEST(testIO);
    CPPUNIT_TEST(testSwap);
    CPPUNIT_TEST(testCopyOnWrite);
    CPPUNIT_TEST_SUITE_END();

    void testEmptyLeaf();
    void testOffsets();
    void testSetValue();
    void testMonotonicity();
    void testAttributes();
    void testTopologyCopy();
    void testEquivalence();
    void testIO();
    void testSwap();
    void testCopyOnWrite();

private:
    // Generate random points by uniformly distributing points
    // on a unit-sphere.
    // (borrowed from PointIndexGrid unit test)
    void genPoints(const int numPoints, std::vector<openvdb::Vec3R>& points) const
    {
        // init
        openvdb::math::Random01 randNumber(0);
        const int n = int(std::sqrt(double(numPoints)));
        const double xScale = (2.0 * M_PI) / double(n);
        const double yScale = M_PI / double(n);

        double x, y, theta, phi;
        openvdb::Vec3R pos;

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
                pos[0] = std::sin(theta)*std::cos(phi);
                pos[1] = std::sin(theta)*std::sin(phi);
                pos[2] = std::cos(theta);

                points.push_back(pos);
            }
        }
    }
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

    for (LeafType::ValueOnCIter iter = leafNode.cbeginValueOn(); iter; ++iter) {
        if (previous > int(*iter))  return false;
        previous = int(*iter);
    }

    return true;
}

// (borrowed from PointIndexGrid unit test)

class PointList
{
public:
    typedef openvdb::Vec3R PosType;
    typedef openvdb::Vec3R value_type;

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

        std::vector<LeafType::ValueType> newOffsets;
        newOffsets.resize(LeafType::SIZE);

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

        std::vector<LeafType::ValueType> newOffsets;
        newOffsets.resize(LeafType::SIZE);

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
        typedef openvdb::tools::TypedAttributeArray<float>    AttributeS;
        typedef openvdb::tools::TypedAttributeArray<int32_t>  AttributeI;
        typedef openvdb::tools::AttributeSet::Descriptor      Descriptor;

        Descriptor::Inserter names;
        names.add("density", AttributeS::attributeType());
        names.add("id", AttributeI::attributeType());
        Descriptor::Ptr descriptor = Descriptor::create(names.vec);

        // ensure validateOffsets succeeds for monotonically increasing offsets that fully
        // utilise the underlying attribute arrays

        {
            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            std::vector<LeafType::ValueType> offsets;
            offsets.resize(LeafType::SIZE);
            offsets.back() = numAttributes;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_NO_THROW(leafNode->validateOffsets());
            delete leafNode;
        }

        // ensure validateOffsets detects non-monotonic offset values

        {
            LeafType* leafNode = new LeafType();

            std::vector<LeafType::ValueType> offsets;
            offsets.resize(LeafType::SIZE);
            *offsets.begin() = 1;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_THROW(leafNode->validateOffsets(), openvdb::ValueError);
            delete leafNode;
        }

        // ensure validateOffsets detects inconsistent attribute array sizes

        {
            using openvdb::tools::AttributeSet;

            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            AttributeSet* newSet = new AttributeSet(descriptor, numAttributes);
            newSet->replace("density", AttributeS::create(numAttributes+1));
            leafNode->swap(newSet);

            std::vector<LeafType::ValueType> offsets;
            offsets.resize(LeafType::SIZE);
            offsets.back() = numAttributes;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_THROW(leafNode->validateOffsets(), openvdb::ValueError);
            delete leafNode;
        }

        // ensure validateOffsets detects unused attributes (e.g. final voxel offset not
        // equal to size of attribute arrays)

        {
            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            std::vector<LeafType::ValueType> offsets;
            offsets.resize(LeafType::SIZE);
            offsets.back() = numAttributes - 1;
            leafNode->setOffsets(offsets);

            CPPUNIT_ASSERT_THROW(leafNode->validateOffsets(), openvdb::ValueError);
            delete leafNode;
        }

        // ensure validateOffsets detects out-of-bounds offset values

        {
            const size_t numAttributes = 1;
            LeafType* leafNode = new LeafType();
            leafNode->initializeAttributes(descriptor, numAttributes);

            std::vector<LeafType::ValueType> offsets;
            offsets.resize(LeafType::SIZE);
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
    typedef openvdb::tools::TypedAttributeArray<float>    AttributeS;
    typedef openvdb::tools::TypedAttributeArray<int32_t>  AttributeI;

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
    CPPUNIT_ASSERT(leaf.hasAttribute("density"));

    CPPUNIT_ASSERT(leaf.hasAttribute(/*pos*/1));
    CPPUNIT_ASSERT(leaf.hasAttribute("id"));

    CPPUNIT_ASSERT(!leaf.hasAttribute(/*pos*/2));
    CPPUNIT_ASSERT(!leaf.hasAttribute("test"));

    // test underlying attributeArray can be accessed by name and index, and that their types are as expected.

    const LeafType* constLeaf = &leaf;

    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos*/0).type(), AttributeS::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("density").type(), AttributeS::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos*/1).type(), AttributeI::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("id").type(), AttributeI::attributeType()));

    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray(/*pos*/0).type(), AttributeS::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray("density").type(), AttributeS::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray(/*pos*/1).type(), AttributeI::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray("id").type(), AttributeI::attributeType()));

    // check invalid pos or name throws

    CPPUNIT_ASSERT_THROW(leaf.attributeArray(/*pos=*/3), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(leaf.attributeArray("not_there"), openvdb::LookupError);

    CPPUNIT_ASSERT_THROW(constLeaf->attributeArray(/*pos=*/3), openvdb::LookupError);
    CPPUNIT_ASSERT_THROW(constLeaf->attributeArray("not_there"), openvdb::LookupError);

    // test leaf can be successfully cast to TypedAttributeArray and check types

    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos=*/0).type(),
                         AttributeS::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("density").type(),
                         AttributeS::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray(/*pos=*/1).type(),
                         AttributeI::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(leaf.attributeArray("id").type(),
                         AttributeI::attributeType()));

    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray(/*pos=*/0).type(),
                         AttributeS::attributeType()));
    CPPUNIT_ASSERT(matchingNamePairs(constLeaf->attributeArray("density").type(),
                         AttributeS::attributeType()));
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

    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

    const LeafType::BaseLeaf& baseLeaf = static_cast<LeafType::BaseLeaf&>(leaf);

    const Index64 memUsage = baseLeaf.memUsage() + leaf.attributeSet().memUsage();

    CPPUNIT_ASSERT_EQUAL(memUsage, leaf.memUsage());
}


void
TestPointDataLeaf::testTopologyCopy()
{
    // test topology copy from a float Leaf

    {
        typedef openvdb::FloatTree::LeafNodeType FloatLeaf;

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

        std::vector<openvdb::Vec3R> points;
        genPoints(40000, points);

        PointList pointList(points);

        // construct point index grid

        typedef openvdb::tools::PointIndexGrid PointIndexGrid;

        PointIndexGrid::Ptr pointGridPtr =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);

        PointIndexGrid::TreeType::LeafCIter iter = pointGridPtr->tree().cbeginLeaf();

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

    typedef TypedAttributeArray<float>    AttributeS;
    typedef TypedAttributeArray<int32_t>  AttributeI;

    AttributeS::registerType();
    AttributeI::registerType();

    // create a descriptor

    typedef AttributeSet::Descriptor Descriptor;

    Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());
    names.add("id", AttributeI::attributeType());

    Descriptor::Ptr descrA = Descriptor::create(names.vec);

    // create a leaf and initialize attributes using this descriptor

    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

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
TestPointDataLeaf::testIO()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types

    typedef TypedAttributeArray<float>    AttributeS;
    typedef TypedAttributeArray<int32_t>  AttributeI;

    AttributeS::registerType();
    AttributeI::registerType();

    // create a descriptor

    typedef AttributeSet::Descriptor Descriptor;

    Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());
    names.add("id", AttributeI::attributeType());

    Descriptor::Ptr descrA = Descriptor::create(names.vec);

    // create a leaf and initialize attributes using this descriptor

    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

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

    typedef TypedAttributeArray<float>    AttributeS;
    typedef TypedAttributeArray<int32_t>  AttributeI;

    AttributeS::registerType();
    AttributeI::registerType();

    // create a descriptor

    typedef AttributeSet::Descriptor Descriptor;

    Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());
    names.add("id", AttributeI::attributeType());

    Descriptor::Ptr descrA = Descriptor::create(names.vec);

    // create a leaf and initialize attributes using this descriptor

    const size_t initialArrayLength = 100;
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.initializeAttributes(descrA, /*arrayLength=*/initialArrayLength);

    // swap out the underlying attribute set with a new attribute set with a matching
    // descriptor

    CPPUNIT_ASSERT_EQUAL(initialArrayLength, leaf.attributeSet().get("density")->size());
    CPPUNIT_ASSERT_EQUAL(initialArrayLength, leaf.attributeSet().get("id")->size());

    const size_t newArrayLength = initialArrayLength / 2;
    leaf.swap(new AttributeSet(descrA, /*arrayLength*/newArrayLength));

    CPPUNIT_ASSERT_EQUAL(newArrayLength, leaf.attributeSet().get("density")->size());
    CPPUNIT_ASSERT_EQUAL(newArrayLength, leaf.attributeSet().get("id")->size());

    // ensure we refuse to swap when the attribute set is null

    CPPUNIT_ASSERT_THROW(leaf.swap(0), openvdb::ValueError);

    // ensure we refuse to swap when the descriptors do not match

    names.add("extra", AttributeS::attributeType());
    Descriptor::Ptr descrB = Descriptor::create(names.vec);
    AttributeSet* attributeSet = new AttributeSet(descrB, newArrayLength);

    CPPUNIT_ASSERT_THROW(leaf.swap(attributeSet), openvdb::ValueError);
    delete attributeSet;
}

void
TestPointDataLeaf::testCopyOnWrite()
{
    using namespace openvdb::tools;

    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>    AttributeS;

    AttributeS::registerType();

    // create a descriptor

    typedef openvdb::tools::AttributeSet::Descriptor Descriptor;

    Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());

    Descriptor::Ptr descrA = Descriptor::create(names.vec);

    // create a leaf and initialize attributes using this descriptor

    LeafType leaf(openvdb::Coord(0, 0, 0));

    leaf.initializeAttributes(descrA, /*arrayLength=*/100);

    const AttributeSet& attributeSet = leaf.attributeSet();

    CPPUNIT_ASSERT_EQUAL(attributeSet.size(), size_t(1));

    // ensure attribute arrays are shared between leaf nodes until write

    const LeafType leafCopy(leaf);

    const AttributeSet& attributeSetCopy = leafCopy.attributeSet();

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/0));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/0));

    // test that from a const leaf, accesses to the attribute arrays do not
    // make then unique

    const AttributeArray* constArray = attributeSetCopy.getConst(/*pos=*/0);
    CPPUNIT_ASSERT(constArray);

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/0));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/0));

    constArray = attributeSetCopy.get(/*pos=*/0);

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/0));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/0));

    constArray = &(leafCopy.attributeArray(/*pos=*/0));

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/0));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/0));

    constArray = &(leafCopy.attributeArray("density"));

    CPPUNIT_ASSERT(attributeSet.isShared(/*pos=*/0));
    CPPUNIT_ASSERT(attributeSetCopy.isShared(/*pos=*/0));

    // test makeUnique is called from non const getters

    AttributeArray* attributeArray = &(leaf.attributeArray(/*pos=*/0));
    CPPUNIT_ASSERT(attributeArray);

    CPPUNIT_ASSERT(!attributeSet.isShared(/*pos=*/0));
    CPPUNIT_ASSERT(!attributeSetCopy.isShared(/*pos=*/0));
}


CPPUNIT_TEST_SUITE_REGISTRATION(TestPointDataLeaf);


////////////////////////////////////////


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
