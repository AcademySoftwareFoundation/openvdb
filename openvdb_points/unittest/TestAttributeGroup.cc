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
#include <openvdb_points/tools/AttributeArray.h>
#include <openvdb_points/tools/AttributeGroup.h>

#include <openvdb_points/openvdb.h>
#include <openvdb/openvdb.h>

#include <iostream>
#include <sstream>

using namespace openvdb;
using namespace openvdb::tools;

class TestAttributeGroup: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestAttributeGroup);
    CPPUNIT_TEST(testAttributeGroupHandle);
    CPPUNIT_TEST(testAttributeGroupFilter);

    CPPUNIT_TEST_SUITE_END();

    void testAttributeGroupHandle();
    void testAttributeGroupFilter();
}; // class TestAttributeGroup

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeGroup);

////////////////////////////////////////


void
TestAttributeGroup::testAttributeGroupHandle()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    GroupAttributeArray attr(4);
    attr.setGroup(true);
    GroupHandle handle(attr, 3);

    CPPUNIT_ASSERT_EQUAL(handle.size(), (unsigned long) 4);
    CPPUNIT_ASSERT_EQUAL(handle.size(), attr.size());

    // construct bitmasks

    const GroupType bitmask3 = GroupType(1) << 3;
    const GroupType bitmask6 = GroupType(1) << 6;
    const GroupType bitmask36 = GroupType(1) << 3 | GroupType(1) << 6;

    // enable attribute 1,2,3 for group permutations of 3 and 6
    attr.set(0, 0);
    attr.set(1, bitmask3);
    attr.set(2, bitmask6);
    attr.set(3, bitmask36);

    CPPUNIT_ASSERT(attr.get(2) != bitmask36);
    CPPUNIT_ASSERT_EQUAL(attr.get(3), bitmask36);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        CPPUNIT_ASSERT(!handle3.get(0));
        CPPUNIT_ASSERT(handle3.get(1));
        CPPUNIT_ASSERT(!handle3.get(2));
        CPPUNIT_ASSERT(handle3.get(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        CPPUNIT_ASSERT(!handle6.get(0));
        CPPUNIT_ASSERT(!handle6.get(1));
        CPPUNIT_ASSERT(handle6.get(2));
        CPPUNIT_ASSERT(handle6.get(3));
    }

    { // groups 3 and 6 only valid for attribute 3 (using bitmask)
        GroupHandle handle36(attr, bitmask36, GroupHandle::BitMask());

        CPPUNIT_ASSERT(!handle36.get(0));
        CPPUNIT_ASSERT(!handle36.get(1));
        CPPUNIT_ASSERT(!handle36.get(2));
        CPPUNIT_ASSERT(handle36.get(3));
    }

    // clear the array

    attr.fill(0);

    CPPUNIT_ASSERT_EQUAL(attr.get(1), GroupType(0));

    // write handles

    GroupWriteHandle writeHandle3(attr, 3);
    GroupWriteHandle writeHandle6(attr, 6);

    writeHandle3.set(1, true);
    writeHandle6.set(2, true);
    writeHandle3.set(3, true);
    writeHandle6.set(3, true);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        CPPUNIT_ASSERT(!handle3.get(0));
        CPPUNIT_ASSERT(handle3.get(1));
        CPPUNIT_ASSERT(!handle3.get(2));
        CPPUNIT_ASSERT(handle3.get(3));

        CPPUNIT_ASSERT(!writeHandle3.get(0));
        CPPUNIT_ASSERT(writeHandle3.get(1));
        CPPUNIT_ASSERT(!writeHandle3.get(2));
        CPPUNIT_ASSERT(writeHandle3.get(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        CPPUNIT_ASSERT(!handle6.get(0));
        CPPUNIT_ASSERT(!handle6.get(1));
        CPPUNIT_ASSERT(handle6.get(2));
        CPPUNIT_ASSERT(handle6.get(3));

        CPPUNIT_ASSERT(!writeHandle6.get(0));
        CPPUNIT_ASSERT(!writeHandle6.get(1));
        CPPUNIT_ASSERT(writeHandle6.get(2));
        CPPUNIT_ASSERT(writeHandle6.get(3));
    }

    writeHandle3.set(3, false);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        CPPUNIT_ASSERT(!handle3.get(0));
        CPPUNIT_ASSERT(handle3.get(1));
        CPPUNIT_ASSERT(!handle3.get(2));
        CPPUNIT_ASSERT(!handle3.get(3));

        CPPUNIT_ASSERT(!writeHandle3.get(0));
        CPPUNIT_ASSERT(writeHandle3.get(1));
        CPPUNIT_ASSERT(!writeHandle3.get(2));
        CPPUNIT_ASSERT(!writeHandle3.get(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        CPPUNIT_ASSERT(!handle6.get(0));
        CPPUNIT_ASSERT(!handle6.get(1));
        CPPUNIT_ASSERT(handle6.get(2));
        CPPUNIT_ASSERT(handle6.get(3));

        CPPUNIT_ASSERT(!writeHandle6.get(0));
        CPPUNIT_ASSERT(!writeHandle6.get(1));
        CPPUNIT_ASSERT(writeHandle6.get(2));
        CPPUNIT_ASSERT(writeHandle6.get(3));
    }
}


class GroupNotFilter
{
public:
    GroupNotFilter(const GroupHandle& handle)
        : mFilter(handle) { }

    bool valid(const Index32 offset) const {
        return !mFilter.valid(offset);
    }
private:
    const GroupFilter mFilter;
}; // class GroupNotFilter


void
TestAttributeGroup::testAttributeGroupFilter()
{

}


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
