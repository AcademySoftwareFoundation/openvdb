// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/AttributeGroup.h>
#include <openvdb/points/IndexIterator.h>
#include <openvdb/points/IndexFilter.h>

#include <openvdb/openvdb.h>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

using namespace openvdb;
using namespace openvdb::points;

class TestAttributeGroup: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestAttributeGroup


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


TEST_F(TestAttributeGroup, testAttributeGroup)
{
    { // Typed class API

        const size_t count = 50;
        GroupAttributeArray attr(count);

        EXPECT_TRUE(!attr.isTransient());
        EXPECT_TRUE(!attr.isHidden());
        EXPECT_TRUE(isGroup(attr));

        attr.setTransient(true);
        EXPECT_TRUE(attr.isTransient());
        EXPECT_TRUE(!attr.isHidden());
        EXPECT_TRUE(isGroup(attr));

        attr.setHidden(true);
        EXPECT_TRUE(attr.isTransient());
        EXPECT_TRUE(attr.isHidden());
        EXPECT_TRUE(isGroup(attr));

        attr.setTransient(false);
        EXPECT_TRUE(!attr.isTransient());
        EXPECT_TRUE(attr.isHidden());
        EXPECT_TRUE(isGroup(attr));

        GroupAttributeArray attrB(attr);

        EXPECT_TRUE(matchingNamePairs(attr.type(), attrB.type()));
        EXPECT_EQ(attr.size(), attrB.size());
        EXPECT_EQ(attr.memUsage(), attrB.memUsage());
        EXPECT_EQ(attr.isUniform(), attrB.isUniform());
        EXPECT_EQ(attr.isTransient(), attrB.isTransient());
        EXPECT_EQ(attr.isHidden(), attrB.isHidden());
        EXPECT_EQ(isGroup(attr), isGroup(attrB));

        AttributeArray& baseAttr(attr);
        EXPECT_EQ(Name(typeNameAsString<GroupType>()), baseAttr.valueType());
        EXPECT_EQ(Name("grp"), baseAttr.codecType());
        EXPECT_EQ(Index(1), baseAttr.valueTypeSize());
        EXPECT_EQ(Index(1), baseAttr.storageTypeSize());
        EXPECT_TRUE(!baseAttr.valueTypeIsFloatingPoint());
    }

    { // casting
        TypedAttributeArray<float> floatAttr(4);
        AttributeArray& floatArray = floatAttr;
        const AttributeArray& constFloatArray = floatAttr;

        EXPECT_THROW(GroupAttributeArray::cast(floatArray), TypeError);
        EXPECT_THROW(GroupAttributeArray::cast(constFloatArray), TypeError);

        GroupAttributeArray groupAttr(4);
        AttributeArray& groupArray = groupAttr;
        const AttributeArray& constGroupArray = groupAttr;

        EXPECT_NO_THROW(GroupAttributeArray::cast(groupArray));
        EXPECT_NO_THROW(GroupAttributeArray::cast(constGroupArray));
    }

    { // IO
        const size_t count = 50;
        GroupAttributeArray attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        attrA.setHidden(true);

        std::ostringstream ostr(std::ios_base::binary);
        attrA.write(ostr);

        GroupAttributeArray attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        EXPECT_TRUE(matchingNamePairs(attrA.type(), attrB.type()));
        EXPECT_EQ(attrA.size(), attrB.size());
        EXPECT_EQ(attrA.memUsage(), attrB.memUsage());
        EXPECT_EQ(attrA.isUniform(), attrB.isUniform());
        EXPECT_EQ(attrA.isTransient(), attrB.isTransient());
        EXPECT_EQ(attrA.isHidden(), attrB.isHidden());
        EXPECT_EQ(isGroup(attrA), isGroup(attrB));

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(attrA.get(i), attrB.get(i));
        }
    }
}


TEST_F(TestAttributeGroup, testAttributeGroupHandle)
{
    GroupAttributeArray attr(4);
    GroupHandle handle(attr, 3);

    EXPECT_EQ(handle.size(), Index(4));
    EXPECT_EQ(handle.size(), attr.size());

    // construct bitmasks

    const GroupType bitmask3 = GroupType(1) << 3;
    const GroupType bitmask6 = GroupType(1) << 6;
    const GroupType bitmask36 = GroupType(1) << 3 | GroupType(1) << 6;

    // enable attribute 1,2,3 for group permutations of 3 and 6
    attr.set(0, 0);
    attr.set(1, bitmask3);
    attr.set(2, bitmask6);
    attr.set(3, bitmask36);

    EXPECT_TRUE(attr.get(2) != bitmask36);
    EXPECT_EQ(attr.get(3), bitmask36);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        EXPECT_TRUE(!handle3.get(0));
        EXPECT_TRUE(handle3.get(1));
        EXPECT_TRUE(!handle3.get(2));
        EXPECT_TRUE(handle3.get(3));
    }

    { // test group 3 valid for attributes 1 and 3 (unsafe access)
        GroupHandle handle3(attr, 3);

        EXPECT_TRUE(!handle3.getUnsafe(0));
        EXPECT_TRUE(handle3.getUnsafe(1));
        EXPECT_TRUE(!handle3.getUnsafe(2));
        EXPECT_TRUE(handle3.getUnsafe(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        EXPECT_TRUE(!handle6.get(0));
        EXPECT_TRUE(!handle6.get(1));
        EXPECT_TRUE(handle6.get(2));
        EXPECT_TRUE(handle6.get(3));
    }

    { // groups 3 and 6 only valid for attribute 3 (using bitmask)
        GroupHandle handle36(attr, bitmask36, GroupHandle::BitMask());

        EXPECT_TRUE(!handle36.get(0));
        EXPECT_TRUE(!handle36.get(1));
        EXPECT_TRUE(!handle36.get(2));
        EXPECT_TRUE(handle36.get(3));
    }

    // clear the array

    attr.fill(0);

    EXPECT_EQ(attr.get(1), GroupType(0));

    // write handles

    GroupWriteHandle writeHandle3(attr, 3);
    GroupWriteHandle writeHandle6(attr, 6);

    // test collapse

    EXPECT_EQ(writeHandle3.get(1), false);
    EXPECT_EQ(writeHandle6.get(1), false);

    EXPECT_TRUE(writeHandle6.compact());
    EXPECT_TRUE(writeHandle6.isUniform());

    attr.expand();

    EXPECT_TRUE(!writeHandle6.isUniform());

    EXPECT_TRUE(writeHandle3.collapse(true));

    EXPECT_TRUE(attr.isUniform());
    EXPECT_TRUE(writeHandle3.isUniform());
    EXPECT_TRUE(writeHandle6.isUniform());

    EXPECT_EQ(writeHandle3.get(1), true);
    EXPECT_EQ(writeHandle6.get(1), false);

    EXPECT_TRUE(writeHandle3.collapse(false));

    EXPECT_TRUE(writeHandle3.isUniform());
    EXPECT_EQ(writeHandle3.get(1), false);

    attr.fill(0);

    writeHandle3.set(1, true);

    EXPECT_TRUE(!attr.isUniform());
    EXPECT_TRUE(!writeHandle3.isUniform());
    EXPECT_TRUE(!writeHandle6.isUniform());

    EXPECT_TRUE(!writeHandle3.collapse(true));

    EXPECT_TRUE(!attr.isUniform());
    EXPECT_TRUE(!writeHandle3.isUniform());
    EXPECT_TRUE(!writeHandle6.isUniform());

    EXPECT_EQ(writeHandle3.get(1), true);
    EXPECT_EQ(writeHandle6.get(1), false);

    writeHandle6.set(2, true);

    EXPECT_TRUE(!writeHandle3.collapse(false));

    EXPECT_TRUE(!writeHandle3.isUniform());

    attr.fill(0);

    writeHandle3.set(1, true);
    writeHandle6.set(2, true);
    writeHandle3.setUnsafe(3, true);
    writeHandle6.setUnsafe(3, true);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        EXPECT_TRUE(!handle3.get(0));
        EXPECT_TRUE(handle3.get(1));
        EXPECT_TRUE(!handle3.get(2));
        EXPECT_TRUE(handle3.get(3));

        EXPECT_TRUE(!writeHandle3.get(0));
        EXPECT_TRUE(writeHandle3.get(1));
        EXPECT_TRUE(!writeHandle3.get(2));
        EXPECT_TRUE(writeHandle3.get(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        EXPECT_TRUE(!handle6.get(0));
        EXPECT_TRUE(!handle6.get(1));
        EXPECT_TRUE(handle6.get(2));
        EXPECT_TRUE(handle6.get(3));

        EXPECT_TRUE(!writeHandle6.get(0));
        EXPECT_TRUE(!writeHandle6.get(1));
        EXPECT_TRUE(writeHandle6.get(2));
        EXPECT_TRUE(writeHandle6.get(3));
    }

    writeHandle3.set(3, false);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        EXPECT_TRUE(!handle3.get(0));
        EXPECT_TRUE(handle3.get(1));
        EXPECT_TRUE(!handle3.get(2));
        EXPECT_TRUE(!handle3.get(3));

        EXPECT_TRUE(!writeHandle3.get(0));
        EXPECT_TRUE(writeHandle3.get(1));
        EXPECT_TRUE(!writeHandle3.get(2));
        EXPECT_TRUE(!writeHandle3.get(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        EXPECT_TRUE(!handle6.get(0));
        EXPECT_TRUE(!handle6.get(1));
        EXPECT_TRUE(handle6.get(2));
        EXPECT_TRUE(handle6.get(3));

        EXPECT_TRUE(!writeHandle6.get(0));
        EXPECT_TRUE(!writeHandle6.get(1));
        EXPECT_TRUE(writeHandle6.get(2));
        EXPECT_TRUE(writeHandle6.get(3));
    }
}


class GroupNotFilter
{
public:
    explicit GroupNotFilter(const AttributeSet::Descriptor::GroupIndex& index)
        : mFilter(index) { }

    inline bool initialized() const { return mFilter.initialized(); }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        mFilter.reset(leaf);
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        return !mFilter.valid(iter);
    }

private:
    GroupFilter mFilter;
}; // class GroupNotFilter


struct HandleWrapper
{
    HandleWrapper(const GroupHandle& handle)
        : mHandle(handle) { }

    GroupHandle groupHandle(const AttributeSet::Descriptor::GroupIndex& /*index*/) const {
        return mHandle;
    }

private:
    const GroupHandle mHandle;
}; // struct HandleWrapper


TEST_F(TestAttributeGroup, testAttributeGroupFilter)
{
    using GroupIndex = AttributeSet::Descriptor::GroupIndex;

    GroupIndex zeroIndex;

    typedef IndexIter<ValueVoxelCIter, GroupFilter> IndexGroupAllIter;

    GroupAttributeArray attrGroup(4);
    const Index32 size = attrGroup.size();

    { // group values all zero
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        EXPECT_TRUE(filter.state() == index::PARTIAL);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 0)));
        IndexGroupAllIter iter(indexIter, filter);

        EXPECT_TRUE(!iter);
    }

    // enable attributes 0 and 2 for groups 3 and 6

    const GroupType bitmask = GroupType(1) << 3 | GroupType(1) << 6;

    attrGroup.set(0, bitmask);
    attrGroup.set(2, bitmask);

    // index iterator only valid in groups 3 and 6
    {
        ValueVoxelCIter indexIter(0, size);

        GroupFilter filter(zeroIndex);

        filter.reset(HandleWrapper(GroupHandle(attrGroup, 0)));
        EXPECT_TRUE(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 1)));
        EXPECT_TRUE(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 2)));
        EXPECT_TRUE(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        EXPECT_TRUE(IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 4)));
        EXPECT_TRUE(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 5)));
        EXPECT_TRUE(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 6)));
        EXPECT_TRUE(IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 7)));
        EXPECT_TRUE(!IndexGroupAllIter(indexIter, filter));
    }

    attrGroup.set(1, bitmask);
    attrGroup.set(3, bitmask);

    using IndexNotGroupAllIter = IndexIter<ValueVoxelCIter, GroupNotFilter>;

    // index iterator only not valid in groups 3 and 6
    {
        ValueVoxelCIter indexIter(0, size);

        GroupNotFilter filter(zeroIndex);

        filter.reset(HandleWrapper(GroupHandle(attrGroup, 0)));
        EXPECT_TRUE(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 1)));
        EXPECT_TRUE(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 2)));
        EXPECT_TRUE(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        EXPECT_TRUE(!IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 4)));
        EXPECT_TRUE(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 5)));
        EXPECT_TRUE(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 6)));
        EXPECT_TRUE(!IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 7)));
        EXPECT_TRUE(IndexNotGroupAllIter(indexIter, filter));
    }

    // clear group membership for attributes 1 and 3

    attrGroup.set(1, GroupType(0));
    attrGroup.set(3, GroupType(0));

    { // index in group next
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexGroupAllIter iter(indexIter, filter);

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, Index32(0));

        EXPECT_TRUE(iter.next());
        EXPECT_EQ(*iter, Index32(2));

        EXPECT_TRUE(!iter.next());
    }

    { // index in group prefix ++
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexGroupAllIter iter(indexIter, filter);

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, Index32(0));

        IndexGroupAllIter old = ++iter;
        EXPECT_EQ(*old, Index32(2));
        EXPECT_EQ(*iter, Index32(2));

        EXPECT_TRUE(!iter.next());
    }

    { // index in group postfix ++/--
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexGroupAllIter iter(indexIter, filter);

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, Index32(0));

        IndexGroupAllIter old = iter++;
        EXPECT_EQ(*old, Index32(0));
        EXPECT_EQ(*iter, Index32(2));

        EXPECT_TRUE(!iter.next());
    }

    { // index not in group next
        ValueVoxelCIter indexIter(0, size);
        GroupNotFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexNotGroupAllIter iter(indexIter, filter);

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, Index32(1));

        EXPECT_TRUE(iter.next());
        EXPECT_EQ(*iter, Index32(3));

        EXPECT_TRUE(!iter.next());
    }

    { // index not in group prefix ++
        ValueVoxelCIter indexIter(0, size);
        GroupNotFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexNotGroupAllIter iter(indexIter, filter);

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, Index32(1));

        IndexNotGroupAllIter old = ++iter;
        EXPECT_EQ(*old, Index32(3));
        EXPECT_EQ(*iter, Index32(3));

        EXPECT_TRUE(!iter.next());
    }

    { // index not in group postfix ++
        ValueVoxelCIter indexIter(0, size);
        GroupNotFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexNotGroupAllIter iter(indexIter, filter);

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, Index32(1));

        IndexNotGroupAllIter old = iter++;
        EXPECT_EQ(*old, Index32(1));
        EXPECT_EQ(*iter, Index32(3));

        EXPECT_TRUE(!iter.next());
    }
}
