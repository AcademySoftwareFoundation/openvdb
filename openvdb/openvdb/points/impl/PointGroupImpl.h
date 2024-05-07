// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey
///
/// @file PointGroupImpl.h
///

#ifndef OPENVDB_POINTS_POINT_GROUP_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_GROUP_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace point_group_internal {


/// Copy a group attribute value from one group offset to another
template<typename PointDataTreeType>
struct CopyGroupOp {

    using LeafManagerT  = typename tree::LeafManager<PointDataTreeType>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;
    using GroupIndex    = AttributeSet::Descriptor::GroupIndex;

    CopyGroupOp(const GroupIndex& targetIndex,
                const GroupIndex& sourceIndex)
        : mTargetIndex(targetIndex)
        , mSourceIndex(sourceIndex) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {

            GroupHandle sourceGroup = leaf->groupHandle(mSourceIndex);
            GroupWriteHandle targetGroup = leaf->groupWriteHandle(mTargetIndex);

            for (auto iter = leaf->beginIndexAll(); iter; ++iter) {
                const bool groupOn = sourceGroup.get(*iter);
                targetGroup.set(*iter, groupOn);
            }
        }
    }

    //////////

    const GroupIndex        mTargetIndex;
    const GroupIndex        mSourceIndex;
};


/// Set membership on or off for the specified group
template <typename PointDataTreeT, bool Member>
struct SetGroupOp
{
    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using GroupIndex    = AttributeSet::Descriptor::GroupIndex;

    SetGroupOp(const AttributeSet::Descriptor::GroupIndex& index)
        : mIndex(index) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the group attribute array

            GroupWriteHandle group(leaf->groupWriteHandle(mIndex));

            // set the group value

            group.collapse(Member);
        }
    }

    //////////

    const GroupIndex&       mIndex;
}; // struct SetGroupOp


template <typename PointDataTreeT, typename PointIndexTreeT, bool Remove>
struct SetGroupFromIndexOp
{
    using LeafManagerT          = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;
    using PointIndexLeafNode    = typename PointIndexTreeT::LeafNodeType;
    using IndexArray            = typename PointIndexLeafNode::IndexArray;
    using GroupIndex            = AttributeSet::Descriptor::GroupIndex;
    using MembershipArray       = std::vector<short>;

    SetGroupFromIndexOp(const PointIndexTreeT& indexTree,
                        const MembershipArray& membership,
                        const GroupIndex& index)
        : mIndexTree(indexTree)
        , mMembership(membership)
        , mIndex(index) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            // obtain the group attribute array

            GroupWriteHandle group(leaf->groupWriteHandle(mIndex));

            // initialise the attribute storage

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (const Index64 i: indices) {
                if (Remove) {
                    group.set(static_cast<Index>(index), mMembership[i]);
                } else if (mMembership[i] == short(1)) {
                    group.set(static_cast<Index>(index), short(1));
                }
                index++;
            }

            // attempt to compact the array

            group.compact();
        }
    }

    //////////

    const PointIndexTreeT& mIndexTree;
    const MembershipArray& mMembership;
    const GroupIndex& mIndex;
}; // struct SetGroupFromIndexOp


template <typename PointDataTreeT, typename FilterT, typename IterT = typename PointDataTreeT::LeafNodeType::ValueAllCIter>
struct SetGroupByFilterOp
{
    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;
    using LeafNodeT     = typename PointDataTreeT::LeafNodeType;
    using GroupIndex    = AttributeSet::Descriptor::GroupIndex;

    SetGroupByFilterOp( const GroupIndex& index, const FilterT& filter)
        : mIndex(index)
        , mFilter(filter) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the group attribute array

            GroupWriteHandle group(leaf->groupWriteHandle(mIndex));

            auto iter = leaf->template beginIndex<IterT, FilterT>(mFilter);

            for (; iter; ++iter) {
                group.set(*iter, true);
            }

            // attempt to compact the array

            group.compact();
        }
    }

    //////////

    const GroupIndex& mIndex;
    const FilterT& mFilter; // beginIndex takes a copy of mFilter
}; // struct SetGroupByFilterOp


////////////////////////////////////////


} // namespace point_group_internal

/// @endcond

////////////////////////////////////////


inline void deleteMissingPointGroups(   std::vector<std::string>& groups,
                                        const AttributeSet::Descriptor& descriptor)
{
    for (auto it = groups.begin(); it != groups.end();) {
        if (!descriptor.hasGroup(*it))  it = groups.erase(it);
        else                            ++it;
    }
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void appendGroup(PointDataTreeT& tree, const Name& group)
{
    if (group.empty()) {
        OPENVDB_THROW(KeyError, "Cannot use an empty group name as a key.");
    }

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    auto descriptor = attributeSet.descriptorPtr();

    // don't add if group already exists

    if (descriptor->hasGroup(group))    return;

    const bool hasUnusedGroup = descriptor->unusedGroups() > 0;

    // add a new group attribute if there are no unused groups

    if (!hasUnusedGroup) {

        // find a new internal group name

        const Name groupName = descriptor->uniqueName("__group");

        descriptor = descriptor->duplicateAppend(groupName, GroupAttributeArray::attributeType());
        const size_t pos = descriptor->find(groupName);

        // insert new group attribute

        tree::LeafManager<PointDataTreeT> leafManager(tree);
        leafManager.foreach(
            [&](typename PointDataTreeT::LeafNodeType& leaf, size_t /*idx*/) {
                auto expected = leaf.attributeSet().descriptorPtr();
                leaf.appendAttribute(*expected, descriptor, pos);
            }, /*threaded=*/true
        );
    }
    else {
        // make the descriptor unique before we modify the group map

        makeDescriptorUnique(tree);
        descriptor = attributeSet.descriptorPtr();
    }

    // ensure that there are now available groups

    OPENVDB_ASSERT(descriptor->unusedGroups() > 0);

    // find next unused offset

    const size_t offset = descriptor->unusedGroupOffset();

    // add the group mapping to the descriptor

    descriptor->setGroup(group, offset);

    // if there was an unused group then we did not need to append a new attribute, so
    // we must manually clear membership in the new group as its bits may have been
    // previously set

    if (hasUnusedGroup)    setGroup(tree, group, false);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void appendGroups(PointDataTreeT& tree,
                         const std::vector<Name>& groups)
{
    // TODO: could be more efficient by appending multiple groups at once
    // instead of one-by-one, however this is likely not that common a use case

    for (const Name& name : groups) {
        appendGroup(tree, name);
    }
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropGroup(PointDataTreeT& tree, const Name& group, const bool compact)
{
    using Descriptor = AttributeSet::Descriptor;

    if (group.empty()) {
        OPENVDB_THROW(KeyError, "Cannot use an empty group name as a key.");
    }

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();

    // make the descriptor unique before we modify the group map

    makeDescriptorUnique(tree);
    Descriptor::Ptr descriptor = attributeSet.descriptorPtr();

    // now drop the group

    descriptor->dropGroup(group);

    if (compact) {
        compactGroups(tree);
    }
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropGroups( PointDataTreeT& tree,
                        const std::vector<Name>& groups)
{
    for (const Name& name : groups) {
        dropGroup(tree, name, /*compact=*/false);
    }

    // compaction done once for efficiency

    compactGroups(tree);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropGroups( PointDataTreeT& tree)
{
    using Descriptor = AttributeSet::Descriptor;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();

    // make the descriptor unique before we modify the group map

    makeDescriptorUnique(tree);
    Descriptor::Ptr descriptor = attributeSet.descriptorPtr();

    descriptor->clearGroups();

    // find all indices for group attribute arrays

    std::vector<size_t> indices = attributeSet.groupAttributeIndices();

    // drop these attributes arrays

    dropAttributes(tree, indices);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void compactGroups(PointDataTreeT& tree)
{
    using Descriptor = AttributeSet::Descriptor;
    using GroupIndex = Descriptor::GroupIndex;
    using LeafManagerT = typename tree::template LeafManager<PointDataTreeT>;

    using point_group_internal::CopyGroupOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();

    // early exit if not possible to compact

    if (!attributeSet.descriptor().canCompactGroups())    return;

    // make the descriptor unique before we modify the group map

    makeDescriptorUnique(tree);
    Descriptor::Ptr descriptor = attributeSet.descriptorPtr();

    // generate a list of group offsets and move them (one-by-one)
    // TODO: improve this algorithm to move multiple groups per array at once
    // though this is likely not that common a use case

    Name sourceName;
    size_t sourceOffset, targetOffset;

    while (descriptor->requiresGroupMove(sourceName, sourceOffset, targetOffset)) {

        const GroupIndex sourceIndex = attributeSet.groupIndex(sourceOffset);
        const GroupIndex targetIndex = attributeSet.groupIndex(targetOffset);

        CopyGroupOp<PointDataTreeT> copy(targetIndex, sourceIndex);
        LeafManagerT leafManager(tree);
        tbb::parallel_for(leafManager.leafRange(), copy);

        descriptor->setGroup(sourceName, targetOffset);
    }

    // drop unused attribute arrays

    const std::vector<size_t> indices = attributeSet.groupAttributeIndices();

    const size_t totalAttributesToDrop = descriptor->unusedGroups() / descriptor->groupBits();

    OPENVDB_ASSERT(totalAttributesToDrop <= indices.size());

    const std::vector<size_t> indicesToDrop(indices.end() - totalAttributesToDrop,
        indices.end());

    dropAttributes(tree, indicesToDrop);
}


////////////////////////////////////////


template <typename PointDataTreeT, typename PointIndexTreeT>
inline void setGroup(   PointDataTreeT& tree,
                        const PointIndexTreeT& indexTree,
                        const std::vector<short>& membership,
                        const Name& group,
                        const bool remove)
{
    using Descriptor    = AttributeSet::Descriptor;
    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using point_group_internal::SetGroupFromIndexOp;

    auto iter = tree.cbeginLeaf();
    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const Descriptor& descriptor = attributeSet.descriptor();

    if (!descriptor.hasGroup(group)) {
        OPENVDB_THROW(LookupError, "Group must exist on Tree before defining membership.");
    }

    {
        // Check that that the largest index in the PointIndexTree is smaller than the size
        // of the membership vector. The index tree will be used to lookup membership
        // values. If the index tree was constructed with nan positions, this index will
        // differ from the PointDataTree count

        using IndexTreeManager = tree::LeafManager<const PointIndexTreeT>;
        IndexTreeManager leafManager(indexTree);

        const int64_t max = tbb::parallel_reduce(leafManager.leafRange(), -1,
            [](const typename IndexTreeManager::LeafRange& range, int64_t value) -> int64_t {
                for (auto leaf = range.begin(); leaf; ++leaf) {
                    auto it = std::max_element(leaf->indices().begin(), leaf->indices().end());
                    value = std::max(value, static_cast<int64_t>(*it));
                }
                return value;
            },
            [](const int64_t a, const int64_t b) {
                return std::max(a, b);
            }
        );

        if (max != -1 && membership.size() <= static_cast<size_t>(max)) {
            OPENVDB_THROW(IndexError, "Group membership vector size must be larger than "
                " the maximum index within the provided index tree.");
        }
    }

    const Descriptor::GroupIndex index = attributeSet.groupIndex(group);
    LeafManagerT leafManager(tree);

    // set membership

    if (remove) {
        SetGroupFromIndexOp<PointDataTreeT, PointIndexTreeT, true>
            set(indexTree, membership, index);
        tbb::parallel_for(leafManager.leafRange(), set);
    }
    else {
        SetGroupFromIndexOp<PointDataTreeT, PointIndexTreeT, false>
            set(indexTree, membership, index);
        tbb::parallel_for(leafManager.leafRange(), set);
    }
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void setGroup(   PointDataTreeT& tree,
                        const Name& group,
                        const bool member)
{
    using Descriptor    = AttributeSet::Descriptor;
    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;

    using point_group_internal::SetGroupOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const Descriptor& descriptor = attributeSet.descriptor();

    if (!descriptor.hasGroup(group)) {
        OPENVDB_THROW(LookupError, "Group must exist on Tree before defining membership.");
    }

    const Descriptor::GroupIndex index = attributeSet.groupIndex(group);
    LeafManagerT leafManager(tree);

    // set membership based on member variable

    if (member)     tbb::parallel_for(leafManager.leafRange(), SetGroupOp<PointDataTreeT, true>(index));
    else            tbb::parallel_for(leafManager.leafRange(), SetGroupOp<PointDataTreeT, false>(index));
}


////////////////////////////////////////


template <typename PointDataTreeT, typename FilterT>
inline void setGroupByFilter(   PointDataTreeT& tree,
                                const Name& group,
                                const FilterT& filter)
{
    using Descriptor    = AttributeSet::Descriptor;
    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;

    using point_group_internal::SetGroupByFilterOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const Descriptor& descriptor = attributeSet.descriptor();

    if (!descriptor.hasGroup(group)) {
        OPENVDB_THROW(LookupError, "Group must exist on Tree before defining membership.");
    }

    const Descriptor::GroupIndex index = attributeSet.groupIndex(group);

    // set membership using filter

    SetGroupByFilterOp<PointDataTreeT, FilterT> set(index, filter);
    LeafManagerT leafManager(tree);

    tbb::parallel_for(leafManager.leafRange(), set);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void setGroupByRandomTarget( PointDataTreeT& tree,
                                    const Name& group,
                                    const Index64 targetPoints,
                                    const unsigned int seed = 0)
{
    using RandomFilter = RandomLeafFilter<PointDataTreeT, std::mt19937>;

    RandomFilter filter(tree, targetPoints, seed);

    setGroupByFilter<PointDataTreeT, RandomFilter>(tree, group, filter);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void setGroupByRandomPercentage( PointDataTreeT& tree,
                                        const Name& group,
                                        const float percentage = 10.0f,
                                        const unsigned int seed = 0)
{
    using RandomFilter =  RandomLeafFilter<PointDataTreeT, std::mt19937>;

    const int currentPoints = static_cast<int>(pointCount(tree));
    const int targetPoints = int(math::Round((percentage * float(currentPoints))/100.0f));

    RandomFilter filter(tree, targetPoints, seed);

    setGroupByFilter<PointDataTreeT, RandomFilter>(tree, group, filter);
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_POINTS_POINT_GROUP_IMPL_HAS_BEEN_INCLUDED
