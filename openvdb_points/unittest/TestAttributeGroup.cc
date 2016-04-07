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
    CPPUNIT_TEST(testAttributeGroup);
    CPPUNIT_TEST(testAttributeGroupHandle);
    CPPUNIT_TEST(testAttributeGroupFilter);

    CPPUNIT_TEST_SUITE_END();

    void testAttributeGroup();
    void testAttributeGroupHandle();
    void testAttributeGroupFilter();
}; // class TestAttributeGroup

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeGroup);


////////////////////////////////////////


namespace {

bool
matchingNamePairs(const openvdb::NamePair& lhs,
                  const openvdb::NamePair& rhs)
{
    if (lhs.first != rhs.first)     return false;
    if (lhs.second != rhs.second)     return false;

    return true;
}

} // namespace


////////////////////////////////////////


void
TestAttributeGroup::testAttributeGroup()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    { // Typed class API

        const size_t count = 50;
        GroupAttributeArray attr(count);

        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());
        CPPUNIT_ASSERT(attr.isGroup());

        attr.setTransient(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());
        CPPUNIT_ASSERT(attr.isGroup());

        attr.setHidden(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());
        CPPUNIT_ASSERT(attr.isGroup());

        attr.setTransient(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());
        CPPUNIT_ASSERT(attr.isGroup());

        attr.setGroup(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());
        CPPUNIT_ASSERT(!attr.isGroup());

        GroupAttributeArray attrB(attr);

        attr.setGroup(true);

        CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
        CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
        CPPUNIT_ASSERT_EQUAL(attr.isGroup(), attrB.isGroup());
    }

    { // IO
        const size_t count = 50;
        GroupAttributeArray attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        attrA.setHidden(true);
        attrA.setGroup(true);

        std::ostringstream ostr(std::ios_base::binary);
        attrA.write(ostr);

        GroupAttributeArray attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        CPPUNIT_ASSERT(matchingNamePairs(attrA.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attrA.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());
        CPPUNIT_ASSERT_EQUAL(attrA.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attrA.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attrA.isHidden(), attrB.isHidden());
        CPPUNIT_ASSERT_EQUAL(attrA.isGroup(), attrB.isGroup());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
        }
    }
}


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

    // test collapse

    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), false);
    CPPUNIT_ASSERT_EQUAL(writeHandle6.get(1), false);

    CPPUNIT_ASSERT(writeHandle3.collapse(true));

    CPPUNIT_ASSERT(attr.isUniform());
    CPPUNIT_ASSERT(writeHandle3.isUniform());
    CPPUNIT_ASSERT(writeHandle6.isUniform());

    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), true);
    CPPUNIT_ASSERT_EQUAL(writeHandle6.get(1), false);

    CPPUNIT_ASSERT(writeHandle3.collapse(false));

    CPPUNIT_ASSERT(writeHandle3.isUniform());
    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), false);

    attr.fill(0);

    writeHandle3.set(1, true);

    CPPUNIT_ASSERT(!attr.isUniform());
    CPPUNIT_ASSERT(!writeHandle3.isUniform());
    CPPUNIT_ASSERT(!writeHandle6.isUniform());

    CPPUNIT_ASSERT(!writeHandle3.collapse(true));

    CPPUNIT_ASSERT(!attr.isUniform());
    CPPUNIT_ASSERT(!writeHandle3.isUniform());
    CPPUNIT_ASSERT(!writeHandle6.isUniform());

    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), true);
    CPPUNIT_ASSERT_EQUAL(writeHandle6.get(1), false);

    writeHandle6.set(2, true);

    CPPUNIT_ASSERT(!writeHandle3.collapse(false));

    CPPUNIT_ASSERT(!writeHandle3.isUniform());

    attr.fill(0);

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

    template <typename IterT>
    bool valid(const IterT& iter) const {
        return !mFilter.valid(iter);
    }
private:
    const GroupFilter mFilter;
}; // class GroupNotFilter


void
TestAttributeGroup::testAttributeGroupFilter()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef FilterIndexIter<IndexIter, GroupFilter> IndexGroupAllIter;

    GroupAttributeArray attrGroup(4);
    attrGroup.setGroup(true);
    const Index32 size = attrGroup.size();

    { // group values all zero
        IndexIter indexIter(0, size);
        GroupFilter filter(GroupHandle(attrGroup, 0));
        IndexGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(!iter);
    }

    // enable attributes 0 and 2 for groups 3 and 6

    const GroupType bitmask = GroupType(1) << 3 | GroupType(1) << 6;

    attrGroup.set(0, bitmask);
    attrGroup.set(2, bitmask);

    // index iterator only valid in groups 3 and 6
    {
        IndexIter indexIter(0, size);

        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 0))));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 1))));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 2))));
        CPPUNIT_ASSERT(IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 3))));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 4))));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 5))));
        CPPUNIT_ASSERT(IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 6))));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, GroupFilter(GroupHandle(attrGroup, 7))));
    }

    attrGroup.set(1, bitmask);
    attrGroup.set(3, bitmask);

    typedef FilterIndexIter<IndexIter, GroupNotFilter> IndexNotGroupAllIter;

    // index iterator only not valid in groups 3 and 6
    {
        IndexIter indexIter(0, size);

        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 0))));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 1))));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 2))));
        CPPUNIT_ASSERT(!IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 3))));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 4))));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 5))));
        CPPUNIT_ASSERT(!IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 6))));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 7))));
    }

    // clear group membership for attributes 1 and 3
    attrGroup.set(1, GroupType(0));
    attrGroup.set(3, GroupType(0));

    { // index in group next
        IndexIter indexIter(0, size);
        IndexGroupAllIter iter(indexIter, GroupFilter(GroupHandle(attrGroup, 3)));

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index in group prefix ++
        IndexIter indexIter(0, size);
        IndexGroupAllIter iter(indexIter, GroupFilter(GroupHandle(attrGroup, 3)));

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        IndexGroupAllIter old = ++iter;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(2));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index in group postfix ++/--
        IndexIter indexIter(0, size);
        IndexGroupAllIter iter(indexIter, GroupFilter(GroupHandle(attrGroup, 3)));

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        IndexGroupAllIter old = iter++;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(0));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index not in group next
        IndexIter indexIter(0, size);
        IndexNotGroupAllIter iter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 3)));

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index not in group prefix ++
        IndexIter indexIter(0, size);
        IndexNotGroupAllIter iter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 3)));

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        IndexNotGroupAllIter old = ++iter;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(3));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index not in group postfix ++
        IndexIter indexIter(0, size);
        IndexNotGroupAllIter iter(indexIter, GroupNotFilter(GroupHandle(attrGroup, 3)));

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        IndexNotGroupAllIter old = iter++;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(1));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }
}


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
