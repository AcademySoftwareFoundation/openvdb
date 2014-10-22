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
#include <openvdb/Exceptions.h>
#include <openvdb/tree/LeafNode.h>

class TestLeaf: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestLeaf);
    CPPUNIT_TEST(testBuffer);
    CPPUNIT_TEST(testGetValue);
    CPPUNIT_TEST(testSetValue);
    CPPUNIT_TEST(testIsValueSet);
    CPPUNIT_TEST(testProbeValue);
    CPPUNIT_TEST(testIterators);
    CPPUNIT_TEST(testEquivalence);
    CPPUNIT_TEST(testGetOrigin);
    CPPUNIT_TEST(testIteratorGetCoord);
    CPPUNIT_TEST(testNegativeIndexing);
    CPPUNIT_TEST_SUITE_END();

    void testBuffer();
    void testGetValue();
    void testSetValue();
    void testIsValueSet();
    void testProbeValue();
    void testIterators();
    void testEquivalence();
    void testGetOrigin();
    void testIteratorGetCoord();
    void testNegativeIndexing();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLeaf);

typedef openvdb::tree::LeafNode<int, 3> LeafType;
typedef LeafType::Buffer                BufferType;
using openvdb::Index;

void
TestLeaf::testBuffer()
{
    {// access
        BufferType buf;

        for (Index i = 0; i < BufferType::size(); ++i) {
            buf.mData[i] = i;
            CPPUNIT_ASSERT(buf[i] == buf.mData[i]);
        }
        for (Index i = 0; i < BufferType::size(); ++i) {
            buf[i] = i;
            CPPUNIT_ASSERT_EQUAL(int(i), buf[i]);
        }
    }

    {// swap
        BufferType buf0, buf1, buf2;

        int *buf0Data = buf0.mData;
        int *buf1Data = buf1.mData;

        for (Index i = 0; i < BufferType::size(); ++i) {
            buf0[i] = i;
            buf1[i] = i * 2;
        }

        buf0.swap(buf1);

        CPPUNIT_ASSERT(buf0.mData == buf1Data);
        CPPUNIT_ASSERT(buf1.mData == buf0Data);

        buf1.swap(buf0);

        CPPUNIT_ASSERT(buf0.mData == buf0Data);
        CPPUNIT_ASSERT(buf1.mData == buf1Data);

        buf0.swap(buf2);

        CPPUNIT_ASSERT(buf2.mData == buf0Data);

        buf2.swap(buf0);

        CPPUNIT_ASSERT(buf0.mData == buf0Data);
    }

}

void
TestLeaf::testGetValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));

    leaf.mBuffer[0] = 2;
    leaf.mBuffer[1] = 3;
    leaf.mBuffer[2] = 4;
    leaf.mBuffer[65] = 10;

    CPPUNIT_ASSERT_EQUAL(2, leaf.getValue(openvdb::Coord(0, 0, 0)));
    CPPUNIT_ASSERT_EQUAL(3, leaf.getValue(openvdb::Coord(0, 0, 1)));
    CPPUNIT_ASSERT_EQUAL(4, leaf.getValue(openvdb::Coord(0, 0, 2)));

    CPPUNIT_ASSERT_EQUAL(10, leaf.getValue(openvdb::Coord(1, 0, 1)));
}

void
TestLeaf::testSetValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0), 3);

    openvdb::Coord xyz(0, 0, 0);
    leaf.setValueOn(xyz, 10);
    CPPUNIT_ASSERT_EQUAL(10, leaf.getValue(xyz));

    xyz.reset(7, 7, 7);
    leaf.setValueOn(xyz, 7);
    CPPUNIT_ASSERT_EQUAL(7, leaf.getValue(xyz));
    leaf.setValueOnly(xyz, 10);
    CPPUNIT_ASSERT_EQUAL(10, leaf.getValue(xyz));

    xyz.reset(2, 3, 6);
    leaf.setValueOn(xyz, 236);
    CPPUNIT_ASSERT_EQUAL(236, leaf.getValue(xyz));

    leaf.setValueOff(xyz, 1);
    CPPUNIT_ASSERT_EQUAL(1, leaf.getValue(xyz));
    CPPUNIT_ASSERT(!leaf.isValueOn(xyz));
}

void
TestLeaf::testIsValueSet()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 5, 7), 10);

    CPPUNIT_ASSERT(leaf.isValueOn(openvdb::Coord(1, 5, 7)));

    CPPUNIT_ASSERT(!leaf.isValueOn(openvdb::Coord(0, 5, 7)));
    CPPUNIT_ASSERT(!leaf.isValueOn(openvdb::Coord(1, 6, 7)));
    CPPUNIT_ASSERT(!leaf.isValueOn(openvdb::Coord(0, 5, 6)));
}

void
TestLeaf::testProbeValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 6, 5), 10);

    LeafType::ValueType val;
    CPPUNIT_ASSERT(leaf.probeValue(openvdb::Coord(1, 6, 5), val));
    CPPUNIT_ASSERT(!leaf.probeValue(openvdb::Coord(1, 6, 4), val));
}

void
TestLeaf::testIterators()
{
    LeafType leaf(openvdb::Coord(0, 0, 0), 2);
    leaf.setValueOn(openvdb::Coord(1, 2, 3), -3);
    leaf.setValueOn(openvdb::Coord(5, 2, 3),  4);
    LeafType::ValueType sum = 0;
    for (LeafType::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) sum += *iter;
    CPPUNIT_ASSERT_EQUAL((-3 + 4), sum);
}

void
TestLeaf::testEquivalence()
{
    LeafType leaf( openvdb::Coord(0, 0, 0), 2);
    LeafType leaf2(openvdb::Coord(0, 0, 0), 3);

    CPPUNIT_ASSERT(leaf != leaf2);

    for(openvdb::Index32 i = 0; i < LeafType::size(); ++i) {
        leaf.setValueOnly(i, i);
        leaf2.setValueOnly(i, i);
    }
    CPPUNIT_ASSERT(leaf == leaf2);

    // set some values.
    leaf.setValueOn(openvdb::Coord(0, 0, 0), 1);
    leaf.setValueOn(openvdb::Coord(0, 1, 0), 1);
    leaf.setValueOn(openvdb::Coord(1, 1, 0), 1);
    leaf.setValueOn(openvdb::Coord(1, 1, 2), 1);

    leaf2.setValueOn(openvdb::Coord(0, 0, 0), 1);
    leaf2.setValueOn(openvdb::Coord(0, 1, 0), 1);
    leaf2.setValueOn(openvdb::Coord(1, 1, 0), 1);
    leaf2.setValueOn(openvdb::Coord(1, 1, 2), 1);

    CPPUNIT_ASSERT(leaf == leaf2);

    leaf2.setValueOn(openvdb::Coord(0, 0, 1), 1);

    CPPUNIT_ASSERT(leaf != leaf2);

    leaf2.setValueOff(openvdb::Coord(0, 0, 1), 1);

    CPPUNIT_ASSERT(leaf == leaf2);
}

void
TestLeaf::testGetOrigin()
{
    {
        LeafType leaf(openvdb::Coord(1, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 1, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1024, 1, 3), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(128*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1023, 1, 3), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(127*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(512, 512, 512), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(512, 512, 512), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(2, 52, 515), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 48, 512), leaf.origin());
    }
}

void
TestLeaf::testIteratorGetCoord()
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(8, 8, 0), 2);

    CPPUNIT_ASSERT_EQUAL(Coord(8, 8, 0), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(9, 10, 3), xyz);

    ++iter;
    xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(13, 10, 3), xyz);
}

void
TestLeaf::testNegativeIndexing()
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(-9, -2, -8), 1);

    CPPUNIT_ASSERT_EQUAL(Coord(-16, -8, -8), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    CPPUNIT_ASSERT_EQUAL(-3, leaf.getValue(Coord(1, 2, 3)));
    CPPUNIT_ASSERT_EQUAL(4, leaf.getValue(Coord(5, 2, 3)));

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(-15, -6, -5), xyz);

    ++iter;
    xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(-11, -6, -5), xyz);
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
