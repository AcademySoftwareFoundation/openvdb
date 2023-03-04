// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Nick Avramoussis
///
/// @file PointReplicateImpl.h
///

#ifndef OPENVDB_POINTS_POINT_REPLICATE_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_REPLICATE_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

template <typename PointDataGridT>
typename PointDataGridT::Ptr
replicate(const PointDataGridT& source,
          const Index multiplier,
          const std::vector<std::string>& attributes,
          const std::string& scaleAttribute,
          const std::string& replicationIndex)
{
    // The copy iterator, used to transfer array values from the source grid
    // to the target (replicated grid).
    struct CopyIter
    {
#ifdef __clang__
        // Silence incorrect clang warning
        _Pragma("clang diagnostic push")
        _Pragma("clang diagnostic ignored \"-Wunused-local-typedef\"")
        using GetIncrementCB = std::function<Index(const Index)>;
        _Pragma("clang diagnostic pop")
#else
        using GetIncrementCB = std::function<Index(const Index)>;
#endif

        CopyIter(const Index end, const GetIncrementCB& cb)
            : mIt(0), mEnd(0), mSource(0), mSourceEnd(end), mCallback(cb) {
            mEnd = mCallback(mSource);
         }

        operator bool() const { return mSource < mSourceEnd; }

        CopyIter& operator++()
        {
            ++mIt;
            // If we have hit the end for current source idx, increment the source idx
            // moving end to the new position for the next source with a non zero
            // number of new values
            while (mIt >= mEnd) {
                ++mSource;
                if (*this) mEnd += mCallback(mSource);
                else return *this;
            }

            return *this;
        }

        Index sourceIndex() const { assert(*this); return mSource; }
        Index targetIndex() const { assert(*this); return mIt; }

    private:
        Index mIt, mEnd, mSource;
        const Index mSourceEnd;
        const GetIncrementCB mCallback;
    }; // struct CopyIter


    // We want the topology and index values of the leaf nodes, but we
    // DON'T want to copy the arrays. This should only shallow copy the
    // descriptor and arrays
    PointDataGrid::Ptr points = source.deepCopy();

    auto iter = source.tree().cbeginLeaf();
    if (!iter) return points;

    const AttributeSet::Descriptor& sourceDescriptor =
        iter->attributeSet().descriptor();

    // verify position

    const size_t ppos = sourceDescriptor.find("P");
    assert(ppos != AttributeSet::INVALID_POS);

    // build new dummy attribute set

    AttributeSet::Ptr set;
    std::vector<size_t> attribsToDrop;
    if (!attributes.empty()) {
        for (const auto& nameIdxPair : sourceDescriptor.map()) {
            if (std::find(attributes.begin(), attributes.end(), nameIdxPair.first) != attributes.end()) continue;
            if (nameIdxPair.first == "P") continue;
            attribsToDrop.emplace_back(nameIdxPair.second);
        }
        set.reset(new AttributeSet(iter->attributeSet(), 1));
    }
    else {
        set.reset(new AttributeSet(AttributeSet::Descriptor::create(sourceDescriptor.type(ppos))));
    }

    if (!replicationIndex.empty()) {
        // don't copy replicationIndex attribute if it exists
        // as it'll be overwritten and we create it after
        const size_t replIdxIdx = sourceDescriptor.find(replicationIndex);
        if (replIdxIdx != AttributeSet::INVALID_POS) attribsToDrop.emplace_back(replIdxIdx);
    }
    set->dropAttributes(attribsToDrop);

    // validate replication attributes

    size_t replicationIdx = AttributeSet::INVALID_POS;
    if (!replicationIndex.empty()) {
        set->appendAttribute(replicationIndex, TypedAttributeArray<int32_t>::attributeType());
        replicationIdx = set->descriptor().find(replicationIndex);
    }

    AttributeSet::DescriptorPtr descriptor = set->descriptorPtr();

    const size_t scaleIdx = !scaleAttribute.empty() ?
        sourceDescriptor.find(scaleAttribute) : AttributeSet::INVALID_POS;

    openvdb::tree::LeafManager<const PointDataTree> sourceManager(source.tree());
    openvdb::tree::LeafManager<openvdb::points::PointDataTree> manager(points->tree());

    manager.foreach(
        [&](PointDataTree::LeafNodeType& leaf, size_t pos)
    {
        using ValueType = PointDataTree::ValueType;

        const auto& sourceLeaf = sourceManager.leaf(pos);
        // @note  This really shoudn't return uint64_t as AttributeArray's size is
        //  limited to the max of a uint32_t...
        assert(sourceLeaf.pointCount() < Index64(std::numeric_limits<Index>::max()));
        const Index sourceCount = static_cast<Index>(sourceLeaf.pointCount());

        Index uniformMultiplier = multiplier;
        AttributeHandle<float>::UniquePtr scaleHandle(nullptr);
        bool useScale = scaleIdx != AttributeSet::INVALID_POS;
        if (useScale) {
            scaleHandle = std::make_unique<AttributeHandle<float>>
                (sourceLeaf.constAttributeArray(scaleIdx));
        }
        // small lambda that returns the amount of points to generate
        // based on a scale. Should only be called if useScale is true,
        // otherwise the scaleHandle will be reset or null
        auto getPointsToGenerate = [&](const Index index) -> Index {
            const float scale = std::max(0.0f, scaleHandle->get(index));
            return static_cast<Index>
                (math::Round(static_cast<float>(multiplier) * scale));
        };

        // if uniform, update the multiplier and don't bother using the scale attribute

        if (useScale && scaleHandle->isUniform()) {
            uniformMultiplier = getPointsToGenerate(0);
            scaleHandle.reset();
            useScale = false;
        }

        // get the new count and build the new offsets - do this in this loop so we
        // don't have to cache the offset vector. Note that the leaf offsets become
        // invalid until leaf.replaceAttributeSet is called and should not be used

        Index total = 0;

        if (useScale) {
            for (auto iter = sourceLeaf.cbeginValueAll(); iter; ++iter) {
                for (auto piter = sourceLeaf.beginIndexVoxel(iter.getCoord());
                     piter; ++piter) { total += getPointsToGenerate(*piter); }
                leaf.setOffsetOnly(iter.pos(), total);
            }
        }
        else {
            total = uniformMultiplier * sourceCount;

            // with a uniform multiplier, just multiply each voxel value
            auto* data = leaf.buffer().data();
            for (size_t i = 0; i < leaf.buffer().size(); ++i) {
                const ValueType::IntType value = data[i];
                data[i] = value * uniformMultiplier;
            }
        }

        // turn voxels off if no points
        leaf.updateValueMask();
        const AttributeSet& sourceSet = sourceLeaf.attributeSet();

        std::unique_ptr<openvdb::points::AttributeSet> newSet
            (new AttributeSet(*set, total));

        auto copy = [&](const std::string& name)
        {
            const auto* sourceArray = sourceSet.getConst(name);
            assert(sourceArray);

            // manually expand so that copyValues() doesn't expand and fill the array -
            // we don't want to unnecessarily zero initialize the target values as we know
            // we're going to write to all of them.
            auto* array = newSet->get(name);
            assert(array);
            array->expand(/*fill*/false);

            if (useScale) {
                const CopyIter iter(sourceCount, [&](const Index i) { return getPointsToGenerate(i); });
                array->copyValues(*sourceArray, iter);
            }
            else {
                const CopyIter iter(sourceCount, [&](const Index) { return uniformMultiplier; });
                array->copyValues(*sourceArray, iter);
            }
        };

        copy("P");
        for (const auto& iter : descriptor->map()) {
            if (iter.first == "P")              continue;
            if (iter.first == replicationIndex) continue;
            copy(iter.first);
        }

        // assign the replication idx if requested

        if (replicationIdx != AttributeSet::INVALID_POS && total > 0) {
            AttributeWriteHandle<int32_t>
                idxHandle(*newSet->get(replicationIdx), /*expand*/false);
            idxHandle.expand(/*fill*/false);
            assert(idxHandle.size() == total);


            Index offset = 0;

            if (useScale) {
                for (Index i = 0; i < sourceCount; ++i) {
                    const Index pointRepCount = getPointsToGenerate(i);
                    for (Index j = 0; j < pointRepCount; ++j) {
                        idxHandle.set(offset++, j);
                    }
                }
            }
            else {
                while (offset < total) {
                    for (Index j = 0; j < uniformMultiplier; ++j) {
                        idxHandle.set(offset++, j);
                    }
                }
            }
        }

        leaf.replaceAttributeSet(newSet.release(), /*mismatch*/true);
    });

    if (!scaleAttribute.empty()) {
        tools::pruneInactive(points->tree());
    }

    return points;
}

template <typename PointDataGridT>
typename PointDataGridT::Ptr
replicate(const PointDataGridT& source,
          const Index multiplier,
          const std::string& scaleAttribute,
          const std::string& replicationIndex)
{
    auto iter = source.tree().cbeginLeaf();
    if (!iter) return source.deepCopy();

    const openvdb::points::AttributeSet::Descriptor& sourceDescriptor =
        iter->attributeSet().descriptor();

    std::vector<std::string> attribs;
    attribs.reserve(sourceDescriptor.map().size());
    for (const auto& namepos : sourceDescriptor.map()) {
        attribs.emplace_back(namepos.first);
    }

    return replicate(source, multiplier, attribs, scaleAttribute, replicationIndex);
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_REPLICATE_IMPL_HAS_BEEN_INCLUDED
