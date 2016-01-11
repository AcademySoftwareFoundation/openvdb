///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
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

#include <sstream>
#include <iostream>

using namespace openvdb;
using namespace openvdb::tools;

class TestIndexIterator: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestIndexIterator);
    CPPUNIT_TEST(testIndexIterator);
    CPPUNIT_TEST(testValueIndexIterator);
    CPPUNIT_TEST(testFilterIndexIterator);

    CPPUNIT_TEST_SUITE_END();

    void testIndexIterator();
    void testValueIndexIterator();
    void testFilterIndexIterator();
}; // class TestIndexIterator

CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIterator);


////////////////////////////////////////


void
TestIndexIterator::testIndexIterator()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    { // empty iterator
        IndexIter iter;

        CPPUNIT_ASSERT(!iter);
        CPPUNIT_ASSERT(!iter.next());
    }

    { // index iterator next
        IndexIter iter(0, 3);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index iterator prefix ++
        IndexIter iter(0, 3);
        IndexIter old;

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        old = ++iter;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(1));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        old = ++iter;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(2));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index iterator postfix ++
        IndexIter iter(0, 3);
        IndexIter old;

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        old = iter++;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(0));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        old = iter++;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(1));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // overflow iterator ++
        IndexIter iter;

        iter++;
        iter++;

        CPPUNIT_ASSERT(!iter);
    }

    { // index iterator equality
        IndexIter iter(0, 5);

        CPPUNIT_ASSERT(iter == IndexIter(0, 3));
        CPPUNIT_ASSERT(iter != IndexIter(1, 3));
        CPPUNIT_ASSERT(iter == IndexIter(0, 4));
    }
}

template <typename IteratorT>
int count(IteratorT& iter)
{
    int total = 0;
    for (; iter; ++iter) {
        total++;
    }
    return total;
}

void
TestIndexIterator::testValueIndexIterator()
{
    using namespace openvdb;
    using namespace openvdb::tree;

    typedef LeafNode<unsigned, 1> LeafNode;
    typedef LeafNode::ValueOnIter ValueOnIter;

    const int size = LeafNode::SIZE;

    { // one per voxel offset, all active
        LeafNode leafNode;

        for (int i = 0; i < size; i++) {
            leafNode.setValueOn(i, i+1);
        }

        ValueOnIter valueIter = leafNode.beginValueOn();

        IndexValueIter<ValueOnIter> iter(valueIter);

        CPPUNIT_ASSERT(iter);

        CPPUNIT_ASSERT_EQUAL(count(iter), size);
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

            IndexValueIter<ValueOnIter> iter(valueIter);

            CPPUNIT_ASSERT(iter);

            CPPUNIT_ASSERT_EQUAL(count(iter), size/2);
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

            IndexValueIter<ValueOnIter> iter(valueIter);

            CPPUNIT_ASSERT(iter);

            CPPUNIT_ASSERT_EQUAL(count(iter), 3);
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

            IndexValueIter<ValueOnIter> iter(valueIter);

            CPPUNIT_ASSERT(iter);

            CPPUNIT_ASSERT_EQUAL(count(iter), size/2);
        }
    }

    { // one per voxel offset, none active
        LeafNode leafNode;

        for (int i = 0; i < size; i++) {
            leafNode.setValueOff(i, i);
        }

        ValueOnIter valueIter = leafNode.beginValueOn();

        IndexValueIter<ValueOnIter> iter(valueIter);

        CPPUNIT_ASSERT(!iter);

        CPPUNIT_ASSERT_EQUAL(count(iter), 0);
    }
}


struct EvenIndexFilter
{
    bool valid(const Index32 offset) const {
        return (offset % 2) == 0;
    }
};


struct OddIndexFilter
{
    OddIndexFilter() : mFilter() { }
    bool valid(const Index32 offset) const {
        return !mFilter.valid(offset);
    }
private:
    EvenIndexFilter mFilter;
};


void
TestIndexIterator::testFilterIndexIterator()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    { // index iterator with even filter
        EvenIndexFilter filter;
        IndexIter indexIter(0, 5);
        FilterIndexIter<IndexIter, EvenIndexFilter> iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(4));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index iterator with odd filter
        OddIndexFilter filter;
        IndexIter indexIter(0, 5);
        FilterIndexIter<IndexIter, OddIndexFilter> iter(indexIter, filter);

        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index iterator where the beginning and end are adjusted
        EvenIndexFilter filter;
        IndexIter indexIter(0, 5);
        FilterIndexIter<IndexIter, EvenIndexFilter> iter(indexIter, filter);

        iter++;
        iter++;

        CPPUNIT_ASSERT_EQUAL(*iter, Index32(4));

        iter.reset(1, 3);

        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }
}

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
