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

#include <openvdb/Types.h>
#include <openvdb/tree/LeafNode.h>

#include "ProfileTimer.h"

#include <sstream>
#include <iostream>

using namespace openvdb;
using namespace openvdb::tools;

class TestIndexIterator: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestIndexIterator);
    CPPUNIT_TEST(testValueIndexIterator);
    CPPUNIT_TEST(testFilterIndexIterator);
    CPPUNIT_TEST(testProfile);

    CPPUNIT_TEST_SUITE_END();

    void testValueIndexIterator();
    void testFilterIndexIterator();
    void testProfile();
}; // class TestIndexIterator

CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIterator);


////////////////////////////////////////


void
TestIndexIterator::testValueIndexIterator()
{
    using namespace openvdb;
    using namespace openvdb::tree;

    using LeafNode      = LeafNode<unsigned, 1>;
    using ValueOnIter   = LeafNode::ValueOnIter;

    const int size = LeafNode::SIZE;

    { // one per voxel offset, all active
        LeafNode leafNode;

        for (int i = 0; i < size; i++) {
            leafNode.setValueOn(i, i+1);
        }

        ValueOnIter valueIter = leafNode.beginValueOn();

        IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

        CPPUNIT_ASSERT(iter);

        CPPUNIT_ASSERT_EQUAL(iterCount(iter), Index64(size));

        // check assignment operator
        auto iter2 = iter;
        CPPUNIT_ASSERT_EQUAL(iterCount(iter2), Index64(size));

        ++iter;

        // check coord value
        Coord xyz;
        iter.getCoord(xyz);
        CPPUNIT_ASSERT_EQUAL(xyz, openvdb::Coord(0, 0, 1));
        CPPUNIT_ASSERT_EQUAL(iter.getCoord(), openvdb::Coord(0, 0, 1));

        // check iterators retrieval
        CPPUNIT_ASSERT_EQUAL(iter.valueIter().getCoord(), openvdb::Coord(0, 0, 1));
        CPPUNIT_ASSERT_EQUAL(iter.end(), Index32(2));

        ++iter;

        // check coord value
        iter.getCoord(xyz);
        CPPUNIT_ASSERT_EQUAL(xyz, openvdb::Coord(0, 1, 0));
        CPPUNIT_ASSERT_EQUAL(iter.getCoord(), openvdb::Coord(0, 1, 0));

        // check iterators retrieval
        CPPUNIT_ASSERT_EQUAL(iter.valueIter().getCoord(), openvdb::Coord(0, 1, 0));
        CPPUNIT_ASSERT_EQUAL(iter.end(), Index32(3));
    }

    { // one per even voxel offsets, only these active
        LeafNode leafNode;

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            if ((i % 2) == 0) {
                leafNode.setValueOn(i, ++offset);
            }
            else {
                leafNode.setValueOff(i, offset);
            }
        }

        {
            ValueOnIter valueIter = leafNode.beginValueOn();

            IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

            CPPUNIT_ASSERT(iter);

            CPPUNIT_ASSERT_EQUAL(iterCount(iter), Index64(size/2));
        }
    }

    { // one per odd voxel offsets, all active
        LeafNode leafNode;

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            if ((i % 2) == 1) {
                leafNode.setValueOn(i, offset++);
            }
            else {
                leafNode.setValueOn(i, offset);
            }
        }

        {
            ValueOnIter valueIter = leafNode.beginValueOn();

            IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

            CPPUNIT_ASSERT(iter);

            CPPUNIT_ASSERT_EQUAL(iterCount(iter), Index64(3));
        }
    }

    { // one per even voxel offsets, all active
        LeafNode leafNode;

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            if ((i % 2) == 0) {
                leafNode.setValueOn(i, offset++);
            }
            else {
                leafNode.setValueOn(i, offset);
            }
        }

        {
            ValueOnIter valueIter = leafNode.beginValueOn();

            IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

            CPPUNIT_ASSERT(iter);

            CPPUNIT_ASSERT_EQUAL(iterCount(iter), Index64(size/2));
        }
    }

    { // one per voxel offset, none active
        LeafNode leafNode;

        for (int i = 0; i < size; i++) {
            leafNode.setValueOff(i, i);
        }

        ValueOnIter valueIter = leafNode.beginValueOn();

        IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

        CPPUNIT_ASSERT(!iter);

        CPPUNIT_ASSERT_EQUAL(iterCount(iter), Index64(0));
    }
}


struct EvenIndexFilter
{
    static bool initialized() { return true; }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return ((*iter) % 2) == 0;
    }
};


struct OddIndexFilter
{
    static bool initialized() { return true; }
    OddIndexFilter() : mFilter() { }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return !mFilter.valid(iter);
    }
private:
    EvenIndexFilter mFilter;
};


struct ConstantIter
{
    ConstantIter(const int _value) : value(_value) { }
    int operator*() const { return value; }
    const int value;
};


void
TestIndexIterator::testFilterIndexIterator()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    { // index iterator with even filter
        EvenIndexFilter filter;
        ValueVoxelCIter indexIter(0, 5);
        IndexIter<ValueVoxelCIter, EvenIndexFilter> iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(4));

        CPPUNIT_ASSERT(!iter.next());

        CPPUNIT_ASSERT_EQUAL(iter.end(), Index32(5));
        CPPUNIT_ASSERT_EQUAL(filter.valid(ConstantIter(1)), iter.filter().valid(ConstantIter(1)));
        CPPUNIT_ASSERT_EQUAL(filter.valid(ConstantIter(2)), iter.filter().valid(ConstantIter(2)));
    }

    { // index iterator with odd filter
        OddIndexFilter filter;
        ValueVoxelCIter indexIter(0, 5);
        IndexIter<ValueVoxelCIter, OddIndexFilter> iter(indexIter, filter);

        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }
}

void
TestIndexIterator::testProfile()
{
    using namespace openvdb;
    using namespace openvdb::util;
    using namespace openvdb::math;
    using namespace openvdb::tools;
    using namespace openvdb::tree;

#ifdef PROFILE
    const int elements(1000 * 1000 * 1000);

    std::cerr << std::endl;
#else
    const int elements(10 * 1000 * 1000);
#endif

    { // for loop
        ProfileTimer timer("ForLoop: sum");
        volatile int sum = 0;
        for (int i = 0; i < elements; i++) {
            sum += i;
        }
        CPPUNIT_ASSERT(sum);
    }

    { // index iterator
        ProfileTimer timer("IndexIter: sum");
        volatile int sum = 0;
        ValueVoxelCIter iter(0, elements);
        for (; iter; ++iter) {
            sum += *iter;
        }
        CPPUNIT_ASSERT(sum);
    }

    using LeafNode = LeafNode<unsigned, 3>;
    LeafNode leafNode;

    const int size = LeafNode::SIZE;

    for (int i = 0; i < size - 1; i++) {
        leafNode.setValueOn(i, (elements / size) * i);
    }
    leafNode.setValueOn(size - 1, elements);

    { // manual value iteration
        ProfileTimer timer("ValueIteratorManual: sum");
        volatile int sum = 0;
        auto indexIter(leafNode.cbeginValueOn());
        int offset = 0;
        for (; indexIter; ++indexIter) {
            int start = offset > 0 ? leafNode.getValue(offset - 1) : 0;
            int end = leafNode.getValue(offset);
            for (int i = start; i < end; i++) {
                sum += i;
            }
            offset++;
        }
        CPPUNIT_ASSERT(sum);
    }

    { // value on iterator (all on)
        ProfileTimer timer("ValueIndexIter: sum");
        volatile int sum = 0;
        auto indexIter(leafNode.cbeginValueAll());
        IndexIter<LeafNode::ValueAllCIter, NullFilter>::ValueIndexIter iter(indexIter);
        for (; iter; ++iter) {
            sum += *iter;
        }
        CPPUNIT_ASSERT(sum);
    }
}

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
