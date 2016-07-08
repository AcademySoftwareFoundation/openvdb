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
#include <openvdb_points/tools/IndexIterator.h>
#include <openvdb_points/tools/IndexFilter.h>
#include <openvdb_points/tools/PointAttribute.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/tools/PointGroup.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/assign/std/vector.hpp> // until C++11

#include <sstream>
#include <iostream>

using namespace openvdb;
using namespace openvdb::tools;

class TestIndexFilter: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestIndexFilter);
    CPPUNIT_TEST(testMultiGroupFilter);
    CPPUNIT_TEST(testRandomLeafFilter);
    CPPUNIT_TEST(testAttributeHashFilter);
    CPPUNIT_TEST(testLevelSetFilter);
    CPPUNIT_TEST(testBBoxFilter);
    CPPUNIT_TEST(testBinaryFilter);
    CPPUNIT_TEST_SUITE_END();

    void testMultiGroupFilter();
    void testRandomLeafFilter();
    void testAttributeHashFilter();
    void testLevelSetFilter();
    void testBBoxFilter();
    void testBinaryFilter();
}; // class TestIndexFilter

CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexFilter);


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
    struct Data
    {
        Data(const int _threshold) : threshold(_threshold) { }
        const int threshold;
    };

    ThresholdFilter(const int threshold) : mThreshold(threshold) { }

    template <typename LeafT>
    static ThresholdFilter create(const LeafT&, const Data& data) {
        return ThresholdFilter(data.threshold);
    }

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
    typedef typename GridType::ValueType ValueT;
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
    typedef FilterIndexIter<IndexIter, MultiGroupFilter> IndexGroupIter;
    IndexIter indexIter(0, size);
    MultiGroupFilter::Data data(include, exclude);
    MultiGroupFilter filter = MultiGroupFilter::create(leaf, data);
    IndexGroupIter iter(indexIter, filter);
    for (unsigned i = 0; i < indices.size(); ++i, ++iter) {
        if (!iter)                                  return false;
        if (*iter != Index32(indices[i]))           return false;
    }
    return !iter;
}


void
TestIndexFilter::testMultiGroupFilter()
{
    using namespace boost::assign; // bring 'operator+=()' into scope

    using namespace openvdb;
    using namespace openvdb::tools;

    typedef PointDataTree::LeafNodeType LeafNode;
    typedef openvdb::tools::TypedAttributeArray<float>    AttributeS;

    AttributeS::registerType();
    GroupAttributeArray::registerType();

    PointDataTree tree;
    LeafNode* leaf = tree.touchLeaf(openvdb::Coord(0, 0, 0));

    AttributeSet::Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());
    AttributeSet::Descriptor::Ptr descriptor = AttributeSet::Descriptor::create(names.vec);

    const size_t size = 5;

    leaf->initializeAttributes(descriptor, size);

    appendGroup(tree, "even");
    appendGroup(tree, "odd");
    appendGroup(tree, "all");
    appendGroup(tree, "first");

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

    // test multi group iteration

    { // all (implicit, no include or exclude)
        std::vector<Name> include;
        std::vector<Name> exclude;
        std::vector<int> indices; indices += 0, 1, 2, 3, 4;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all include
        std::vector<Name> include; include += "all";
        std::vector<Name> exclude;
        std::vector<int> indices; indices += 0, 1, 2, 3, 4;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all exclude
        std::vector<Name> include;
        std::vector<Name> exclude; exclude += "all";
        std::vector<int> indices;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all include and exclude
        std::vector<Name> include; include += "all";
        std::vector<Name> exclude; exclude += "all";
        std::vector<int> indices;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // even include
        std::vector<Name> include; include += "even";
        std::vector<Name> exclude;
        std::vector<int> indices; indices += 0, 2, 4;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd include
        std::vector<Name> include; include += "odd";
        std::vector<Name> exclude;
        std::vector<int> indices; indices += 1, 3;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd include and exclude
        std::vector<Name> include; include += "odd";
        std::vector<Name> exclude; exclude += "odd";
        std::vector<int> indices;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd and first include
        std::vector<Name> include; include += "odd", "first";
        std::vector<Name> exclude;
        std::vector<int> indices; indices += 0, 1, 3;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // even include, first exclude
        std::vector<Name> include; include += "even";
        std::vector<Name> exclude; exclude += "first";
        std::vector<int> indices; indices += 2, 4;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // all include, first and odd exclude
        std::vector<Name> include; include += "all";
        std::vector<Name> exclude; exclude += "first", "odd";
        std::vector<int> indices; indices += 2, 4;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }

    { // odd and first include, even exclude
        std::vector<Name> include; include += "odd", "first";
        std::vector<Name> exclude; exclude += "even";
        std::vector<int> indices; indices += 1, 3;
        CPPUNIT_ASSERT(multiGroupMatches(*leaf, size, include, exclude, indices));
    }
}


void
TestIndexFilter::testRandomLeafFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    { // generateRandomSubset
        std::vector<int> values;
        std::vector<int> values2;

        index_filter_internal::generateRandomSubset<boost::mt19937, int>(values, /*seed*/(unsigned) 0, 1, 20);

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(1));

        // different seed

        index_filter_internal::generateRandomSubset<boost::mt19937, int>(values2, /*seed*/(unsigned) 1, 1, 20);

        CPPUNIT_ASSERT_EQUAL(values2.size(), size_t(1));
        CPPUNIT_ASSERT(values[0] != values2[0]);

        // different integer type

        std::vector<long> values3;

        index_filter_internal::generateRandomSubset<boost::mt19937, long>(values3, /*seed*/(unsigned) 0, 1, 20);

        CPPUNIT_ASSERT_EQUAL(values3.size(), size_t(1));
        CPPUNIT_ASSERT(values[0] == values3[0]);

        // different random number generator

        values.clear();

        index_filter_internal::generateRandomSubset<boost::mt11213b, int>(values, /*seed*/(unsigned) 1, 1, 20);

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(1));
        CPPUNIT_ASSERT(values[0] != values2[0]);

        // no values

        values.clear();

        index_filter_internal::generateRandomSubset<boost::mt19937, int>(values, /*seed*/(unsigned) 0, 0, 20);

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(0));

        // all values

        index_filter_internal::generateRandomSubset<boost::mt19937, int>(values, /*seed*/(unsigned) 0, 1000, 1000);

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(1000));

        // ensure all numbers are represented

        std::sort(values.begin(), values.end());

        for (int i = 0; i < 1000; i++) {
            CPPUNIT_ASSERT_EQUAL(values[i], i);
        }
    }

    { // RandomLeafFilter
        typedef RandomLeafFilter<boost::mt11213b> RandFilter;

        RandFilter::Data data;

        data.leafMap[Coord(0, 0, 0)] = std::pair<Index, Index>(0, 10);
        data.leafMap[Coord(0, 0, 8)] = std::pair<Index, Index>(1, 1);
        data.leafMap[Coord(0, 8, 0)] = std::pair<Index, Index>(2, 50);

        { // all 10 values
            RandFilter filter = RandFilter::create(OriginLeaf(Coord(0, 0, 0), 10), data);
            std::vector<int> values;

            for (SimpleIter iter; *iter < 100; ++iter) {
                if (filter.valid(iter))     values.push_back(*iter);
            }

            CPPUNIT_ASSERT_EQUAL(values.size(), size_t(10));

            for (int i = 0; i < 10; i++) {
                CPPUNIT_ASSERT_EQUAL(values[i], i);
            }
        }

        { // 50 of 100
            RandFilter filter = RandFilter::create(OriginLeaf(Coord(0, 8, 0), 100), data);
            std::vector<int> values;

            for (SimpleIter iter; *iter < 100; ++iter) {
                if (filter.valid(iter))     values.push_back(*iter);
            }

            CPPUNIT_ASSERT_EQUAL(values.size(), size_t(50));

            // ensure no duplicates

            std::sort(values.begin(), values.end());
            std::vector<int>::const_iterator it = std::adjacent_find(values.begin(), values.end());

            CPPUNIT_ASSERT(it == values.end());
        }
    }
}


void setId(PointDataTree& tree, const size_t index, const std::vector<int>& ids)
{
    int offset = 0;
    for (PointDataTree::LeafIter leafIter = tree.beginLeaf(); leafIter; ++leafIter) {
        AttributeWriteHandle<int>::Ptr id = AttributeWriteHandle<int>::create(leafIter->attributeArray(index));

        for (PointDataTree::LeafNodeType::IndexIter iter = leafIter->beginIndex(); iter; ++iter) {
            if (offset >= int(ids.size()))   throw std::runtime_error("Out of range");

            id->set(*iter, ids[offset++]);
        }
    }
}


void
TestIndexFilter::testAttributeHashFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;
    typedef TypedAttributeArray<int>     AttributeI;

    AttributeVec3s::registerType();
    AttributeI::registerType();

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(2, 2, 2));
    positions.push_back(Vec3s(11, 11, 11));
    positions.push_back(Vec3s(12, 12, 12));

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // four points, two leafs

    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(2));

    appendAttribute<AttributeI>(tree, "id");

    const size_t index = tree.cbeginLeaf()->attributeSet().descriptor().find("id");

    // ascending integers, block one
    std::vector<int> ids;
    ids.push_back(1);
    ids.push_back(2);
    ids.push_back(3);
    ids.push_back(4);
    setId(tree, index, ids);

    typedef AttributeHashFilter<boost::mt11213b, int> HashFilter;

    { // zero percent
        HashFilter::Data data(index, 0.0f);

        PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();

        PointDataTree::LeafNodeType::IndexIter indexIter = leafIter->beginIndex();
        HashFilter filter = HashFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(!filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndex();
        HashFilter filter2 = HashFilter::create(*leafIter, data);
        CPPUNIT_ASSERT(!filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
    }

    { // one hundred percent
        HashFilter::Data data(index, 100.0f);

        PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();

        PointDataTree::LeafNodeType::IndexIter indexIter = leafIter->beginIndex();
        HashFilter filter = HashFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndex();
        HashFilter filter2 = HashFilter::create(*leafIter, data);
        CPPUNIT_ASSERT(filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
    }

    { // fifty percent
        HashFilter::Data data(index, 50.0f);

        PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();

        PointDataTree::LeafNodeType::IndexIter indexIter = leafIter->beginIndex();
        HashFilter filter = HashFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(!filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndex();
        HashFilter filter2 = HashFilter::create(*leafIter, data);
        CPPUNIT_ASSERT(filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
    }

    { // fifty percent, new seed
        HashFilter::Data data(index, 50.0f, /*seed=*/100);

        PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();

        PointDataTree::LeafNodeType::IndexIter indexIter = leafIter->beginIndex();
        HashFilter filter = HashFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(filter.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
        ++leafIter;

        indexIter = leafIter->beginIndex();
        HashFilter filter2 = HashFilter::create(*leafIter, data);
        CPPUNIT_ASSERT(!filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(filter2.valid(indexIter));
        ++indexIter;
        CPPUNIT_ASSERT(!indexIter);
    }
}


void
TestIndexFilter::testLevelSetFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;

    AttributeVec3s::registerType();

    // create a point grid

    PointDataGrid::Ptr points;

    {
        std::vector<Vec3s> positions;
        positions.push_back(Vec3s(1, 1, 1));
        positions.push_back(Vec3s(1, 2, 1));
        positions.push_back(Vec3s(10.1, 10, 1));

        const double voxelSize(1.0);
        math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

        points = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
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

    typedef LevelSetFilter<FloatGrid> LSFilter;

    { // capture both points near origin
        LSFilter::Data data(*sphere, points->transform(), -4.0f, 4.0f);
        PointDataGrid::TreeType::LeafCIter leafIter = points->tree().cbeginLeaf();
        PointDataGrid::TreeType::LeafNodeType::IndexOnIter iter = leafIter->beginIndexOn();
        LSFilter filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT(!filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);
    }

    { // capture just the inner-most point
        LSFilter::Data data(*sphere, points->transform(), -0.3f, -0.25f);
        PointDataGrid::TreeType::LeafCIter leafIter = points->tree().cbeginLeaf();
        PointDataGrid::TreeType::LeafNodeType::IndexOnIter iter = leafIter->beginIndexOn();
        LSFilter filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT(!filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);
    }

    { // capture everything but the second point (min > max)
        LSFilter::Data data(*sphere, points->transform(), -0.25f, -0.3f);
        PointDataGrid::TreeType::LeafCIter leafIter = points->tree().cbeginLeaf();
        PointDataGrid::TreeType::LeafNodeType::IndexOnIter iter = leafIter->beginIndexOn();
        LSFilter filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(!filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT(filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);
    }

    {
        std::vector<Vec3s> positions;
        positions.push_back(Vec3s(1, 1, 1));
        positions.push_back(Vec3s(1, 2, 1));
        positions.push_back(Vec3s(10.1, 10, 1));

        const double voxelSize(0.25);
        math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

        points = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    }

    {
        double voxelSize = 1.0;
        sphere = FloatGrid::create(/*backgroundValue=*/5.0);
        sphere->setTransform(math::Transform::createLinearTransform(voxelSize));

        const openvdb::Coord dim(40, 40, 40);
        const openvdb::Vec3f center(10.0f, 10.0f, 0.1f);
        const float radius = 0.2;
        makeSphere<FloatGrid>(dim, center, radius, *sphere);
    }

    { // capture only the last point using a different transform and a new sphere
        LSFilter::Data data(*sphere, points->transform(), 0.5f, 1.0f);
        PointDataGrid::TreeType::LeafCIter leafIter = points->tree().cbeginLeaf();
        PointDataGrid::TreeType::LeafNodeType::IndexOnIter iter = leafIter->beginIndexOn();
        LSFilter filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(!filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(!filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);

        ++leafIter;
        iter = leafIter->beginIndexOn();
        filter = LSFilter::create(*leafIter, data);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT(filter.valid(iter));
        ++iter;
        CPPUNIT_ASSERT(!iter);
    }
}


void
TestIndexFilter::testBBoxFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<Vec3s>   AttributeVec3s;
    typedef PointDataTree::LeafNodeType::ValueOnCIter ValueOnCIter;

    AttributeVec3s::registerType();

    std::vector<Vec3s> positions;
    positions.push_back(Vec3s(1, 1, 1));
    positions.push_back(Vec3s(1, 2, 1));
    positions.push_back(Vec3s(10.1, 10, 1));

    const float voxelSize(0.5);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<PointDataGrid>(positions, AttributeVec3s::attributeType(), *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(2));

    // build some bounding box filters to test

    BBoxFilter::Data data1 = BBoxFilter::Data(*transform, BBoxd(Vec3d(0.5, 0.5, 0.5), Vec3d(1.5, 1.5, 1.5)));
    BBoxFilter::Data data2 = BBoxFilter::Data(*transform, BBoxd(Vec3d(0.5, 0.5, 0.5), Vec3d(1.5, 2.01, 1.5)));
    BBoxFilter::Data data3 = BBoxFilter::Data(*transform, BBoxd(Vec3d(0.5, 0.5, 0.5), Vec3d(11, 11, 1.5)));
    BBoxFilter::Data data4 = BBoxFilter::Data(*transform, BBoxd(Vec3d(-10, 0, 0), Vec3d(11, 1.2, 1.2)));

    // leaf 1

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();

    {
        ValueOnCIter valueIter(leafIter->beginValueOn());
        ValueIndexIter<ValueOnCIter> iter(valueIter);

        // point 1

        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data1).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data2).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data3).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data4).valid(iter));

        ++iter;

        // point 2

        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data1).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data2).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data3).valid(iter));
        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data4).valid(iter));

        ++iter;
        CPPUNIT_ASSERT(!iter);
    }

    ++leafIter;

    // leaf 2

    {
        ValueOnCIter valueIter(leafIter->beginValueOn());
        ValueIndexIter<ValueOnCIter> iter(valueIter);

        // point 3

        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data1).valid(iter));
        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data2).valid(iter));
        CPPUNIT_ASSERT(BBoxFilter::create(*leafIter, data3).valid(iter));
        CPPUNIT_ASSERT(!BBoxFilter::create(*leafIter, data4).valid(iter));

        ++iter;
        CPPUNIT_ASSERT(!iter);
    }
}


void
TestIndexFilter::testBinaryFilter()
{
    typedef ThresholdFilter<true> LessThanFilter;
    typedef ThresholdFilter<false> GreaterThanFilter;

    { // less than
        LessThanFilter filter = LessThanFilter::create(OriginLeaf(Coord(0, 0, 0)), LessThanFilter::Data(5));
        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(5));

        for (int i = 0; i < 5; i++) {
            CPPUNIT_ASSERT_EQUAL(values[i], i);
        }
    }

    { // greater than
        GreaterThanFilter filter = GreaterThanFilter::create(OriginLeaf(Coord(0, 0, 0)), GreaterThanFilter::Data(94));
        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(5));

        int offset = 0;
        for (int i = 95; i < 100; i++) {
            CPPUNIT_ASSERT_EQUAL(values[offset++], i);
        }
    }

    { // binary and
        typedef BinaryFilter<LessThanFilter, GreaterThanFilter, /*And=*/true> RangeFilter;

        RangeFilter::Data data(LessThanFilter::Data(55), GreaterThanFilter::Data(45));
        RangeFilter filter = RangeFilter::create(OriginLeaf(Coord(0, 0, 0)), data);

        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(9));

        int offset = 0;
        for (int i = 46; i < 55; i++) {
            CPPUNIT_ASSERT_EQUAL(values[offset++], i);
        }
    }

    { // binary or
        typedef BinaryFilter<LessThanFilter, GreaterThanFilter, /*And=*/false> HeadTailFilter;

        HeadTailFilter::Data data(LessThanFilter::Data(5), GreaterThanFilter::Data(95));
        HeadTailFilter filter = HeadTailFilter::create(OriginLeaf(Coord(0, 0, 0)), data);

        std::vector<int> values;

        for (SimpleIter iter; *iter < 100; ++iter) {
            if (filter.valid(iter))     values.push_back(*iter);
        }

        CPPUNIT_ASSERT_EQUAL(values.size(), size_t(9));

        int offset = 0;
        for (int i = 0; i < 5; i++) {
            CPPUNIT_ASSERT_EQUAL(values[offset++], i);
        }
        for (int i = 96; i < 100; i++) {
            CPPUNIT_ASSERT_EQUAL(values[offset++], i);
        }
    }
}


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
