// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey
///
/// @file PointMove.h
///
/// @brief Ability to move VDB Points using a custom deformer.
///
/// Deformers used when moving points are in world space by default and must adhere
/// to the interface described in the example below:
/// @code
/// struct MyDeformer
/// {
///     // A reset is performed on each leaf in turn before the points in that leaf are
///     // deformed. A leaf and leaf index (standard leaf traversal order) are supplied as
///     // the arguments, which matches the functor interface for LeafManager::foreach().
///     template <typename LeafNoteType>
///     void reset(LeafNoteType& leaf, size_t idx);
///
///     // Evaluate the deformer and modify the given position to generate the deformed
///     // position. An index iterator is supplied as the argument to allow querying the
///     // point offset or containing voxel coordinate.
///     template <typename IndexIterT>
///     void apply(Vec3d& position, const IndexIterT& iter) const;
/// };
/// @endcode
///
/// @note The DeformerTraits struct (defined in PointMask.h) can be used to configure
/// a deformer to evaluate in index space.

#ifndef OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointDataPartitioner.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


/// @brief Move points in a PointDataGrid using a custom deformer
/// @param points           the PointDataGrid containing the points to be moved.
/// @param deformer         a custom deformer that defines how to move the points.
/// @param filter           an optional index filter
/// @param threaded         enable or disable threading  (threading is enabled by default)
template <typename PointDataGridT, typename DeformerT, typename FilterT = NullFilter, typename OtherFilterT = NullFilter>
inline void movePoints(PointDataGridT& points,
                       DeformerT& deformer,
                       const FilterT& filter = NullFilter(),
                       const std::vector<typename PointDataGridT::Ptr>& pointsToSteal = std::vector<typename PointDataGridT::Ptr>(),
                       const std::vector<typename PointDataGridT::ConstPtr>& pointsToCopy = std::vector<typename PointDataGridT::ConstPtr>(),
                       const OtherFilterT& mergeFilter = NullFilter(),
                       bool deformMergedPoints = false,
                       bool threaded = true);


/// @brief Move points in a PointDataGrid using a custom deformer and a new transform
/// @param points           the PointDataGrid containing the points to be moved.
/// @param transform        target transform to use for the resulting points.
/// @param deformer         a custom deformer that defines how to move the points.
/// @param filter           an optional index filter
/// @param threaded         enable or disable threading  (threading is enabled by default)
template <typename PointDataGridT, typename DeformerT, typename FilterT = NullFilter, typename OtherFilterT = NullFilter>
inline void movePoints(PointDataGridT& points,
                       const math::Transform& transform,
                       DeformerT& deformer,
                       const FilterT& filter = NullFilter(),
                       const std::vector<typename PointDataGridT::Ptr>& pointsToSteal = std::vector<typename PointDataGridT::Ptr>(),
                       const std::vector<typename PointDataGridT::ConstPtr>& pointsToCopy = std::vector<typename PointDataGridT::ConstPtr>(),
                       const OtherFilterT& mergeFilter = NullFilter(),
                       bool deformMergedPoints = false,
                       bool threaded = true);


////////////////////////////////////////


namespace point_move_internal {


// define leaf index in use as 32-bit
using LeafIndex = Index32;


////////////////////////////////////////


struct AttributeArrayCopyHelper
{
    using CopyPtr = AttributeArray::BufferAccessor::CopyPtr;

    template <typename ValueT>
    static bool isEqual(const Name& valueType) { return valueType == typeNameAsString<ValueT>(); }
    static bool isNullCodec(const Name& codec) { return codec == NullCodec::name(); }

    static void copyPtrInvalid(AttributeArray&, Index, const AttributeArray&, Index)
    {
        OPENVDB_THROW(TypeError, "Unsupported Attribute Array Copy");
    }

    template <typename TargetValueT, typename TargetCodecT, typename SourceValueT, typename SourceCodecT>
    static void copyPtrArray(AttributeArray& target, Index targetIndex, const AttributeArray& source, Index sourceIndex)
    {
        const auto& sourceArray = TypedAttributeArray<SourceValueT, SourceCodecT>::cast(source);
        auto& targetArray = TypedAttributeArray<TargetValueT, TargetCodecT>::cast(target);
        targetArray.setUnsafe(targetIndex, static_cast<TargetValueT>(sourceArray.getUnsafe(sourceIndex)));
    }

    template <typename TargetValueT, typename SourceValueT>
    static void copyPtrHandle(AttributeArray& target, Index targetIndex, const AttributeArray& source, Index sourceIndex)
    {
        AttributeWriteHandle<TargetValueT> targetHandle(target);
        AttributeHandle<SourceValueT> sourceHandle(source);
        targetHandle.set(targetIndex, static_cast<TargetValueT>(sourceHandle.get(sourceIndex)));
    }

    template <typename TargetValueT, typename SourceValueT>
    static CopyPtr getCopyPtrNullCodec()
    {
        return copyPtrArray<TargetValueT, NullCodec, SourceValueT, NullCodec>;
    }

    template <typename ValueT>
    static CopyPtr getCopyPtrSameValueType(const Name& targetCodec, const Name& sourceCodec)
    {
        if (isNullCodec(targetCodec) && isNullCodec(sourceCodec)) {
            return copyPtrArray<ValueT, NullCodec, ValueT, NullCodec>;
        }

        return copyPtrHandle<ValueT, ValueT>;
    }

    static CopyPtr getCopyPtrVec3f(const Name& targetCodec, const Name& sourceCodec)
    {
        if (isNullCodec(targetCodec)) {
            if (sourceCodec == FixedPointCodec<true>::name()) {
                return copyPtrArray<Vec3f, NullCodec, Vec3f, FixedPointCodec<true>>;
            } else if (sourceCodec == FixedPointCodec<false>::name()) {
                return copyPtrArray<Vec3f, NullCodec, Vec3f, FixedPointCodec<false>>;
            } else if (sourceCodec == UnitVecCodec::name()) {
                return copyPtrArray<Vec3f, NullCodec, Vec3f, UnitVecCodec>;
            } else if (sourceCodec == TruncateCodec::name()) {
                return copyPtrArray<Vec3f, NullCodec, Vec3f, TruncateCodec>;
            }
        } else if (targetCodec == FixedPointCodec<true>::name()) {
            if (isNullCodec(sourceCodec)) {
                return copyPtrArray<Vec3f, FixedPointCodec<true>, Vec3f, NullCodec>;
            } else if (sourceCodec == FixedPointCodec<false>::name()) {
                return copyPtrArray<Vec3f, FixedPointCodec<true>, Vec3f, FixedPointCodec<false>>;
            } else if (sourceCodec == UnitVecCodec::name()) {
                return copyPtrArray<Vec3f, FixedPointCodec<true>, Vec3f, UnitVecCodec>;
            } else if (sourceCodec == TruncateCodec::name()) {
                return copyPtrArray<Vec3f, FixedPointCodec<true>, Vec3f, TruncateCodec>;
            }
        } else if (targetCodec == FixedPointCodec<false>::name()) {
            if (isNullCodec(sourceCodec)) {
                return copyPtrArray<Vec3f, FixedPointCodec<false>, Vec3f, NullCodec>;
            } else if (sourceCodec == FixedPointCodec<true>::name()) {
                return copyPtrArray<Vec3f, FixedPointCodec<false>, Vec3f, FixedPointCodec<true>>;
            } else if (sourceCodec == UnitVecCodec::name()) {
                return copyPtrArray<Vec3f, FixedPointCodec<false>, Vec3f, UnitVecCodec>;
            } else if (sourceCodec == TruncateCodec::name()) {
                return copyPtrArray<Vec3f, FixedPointCodec<false>, Vec3f, TruncateCodec>;
            }
        } else if (targetCodec == UnitVecCodec::name()) {
            if (isNullCodec(sourceCodec)) {
                return copyPtrArray<Vec3f, UnitVecCodec, Vec3f, NullCodec>;
            } else if (sourceCodec == FixedPointCodec<true>::name()) {
                return copyPtrArray<Vec3f, UnitVecCodec, Vec3f, FixedPointCodec<true>>;
            } else if (sourceCodec == FixedPointCodec<false>::name()) {
                return copyPtrArray<Vec3f, UnitVecCodec, Vec3f, FixedPointCodec<false>>;
            } else if (sourceCodec == TruncateCodec::name()) {
                return copyPtrArray<Vec3f, UnitVecCodec, Vec3f, TruncateCodec>;
            }
        } else if (targetCodec == TruncateCodec::name()) {
            if (isNullCodec(sourceCodec)) {
                return copyPtrArray<Vec3f, TruncateCodec, Vec3f, NullCodec>;
            } else if (sourceCodec == FixedPointCodec<true>::name()) {
                return copyPtrArray<Vec3f, TruncateCodec, Vec3f, FixedPointCodec<true>>;
            } else if (sourceCodec == FixedPointCodec<false>::name()) {
                return copyPtrArray<Vec3f, TruncateCodec, Vec3f, FixedPointCodec<false>>;
            } else if (sourceCodec == UnitVecCodec::name()) {
                return copyPtrArray<Vec3f, TruncateCodec, Vec3f, UnitVecCodec>;
            }
        }

        return copyPtrHandle<Vec3f, Vec3f>;
    }

    static CopyPtr getCopyPtrFloat(const Name& targetCodec, const Name& sourceCodec)
    {
        if (isNullCodec(targetCodec) && sourceCodec == TruncateCodec::name()) {
            return copyPtrArray<float, NullCodec, float, TruncateCodec>;
        } else if (targetCodec == TruncateCodec::name() && isNullCodec(sourceCodec)) {
            return copyPtrArray<float, TruncateCodec, float, NullCodec>;
        }

        return copyPtrHandle<Vec3f, Vec3f>;
    }

    static CopyPtr getCopyPtr(const NamePair& targetType, const NamePair& sourceType)
    {
        using namespace openvdb::math;

        const Name& targetValue = targetType.first;
        const Name& targetCodec = targetType.second;
        const Name& sourceValue = sourceType.first;
        const Name& sourceCodec = sourceType.second;

        // instantiate some common conversions

        if (targetValue == sourceValue) {
            assert(targetCodec != sourceCodec);

            if (isEqual<bool>(targetValue))                 return getCopyPtrSameValueType<bool>(targetCodec, sourceCodec);
            else if (isEqual<int8_t>(targetValue))          return getCopyPtrSameValueType<int8_t>(targetCodec, sourceCodec);
            else if (isEqual<int16_t>(targetValue))         return getCopyPtrSameValueType<int16_t>(targetCodec, sourceCodec);
            else if (isEqual<int32_t>(targetValue))         return getCopyPtrSameValueType<int32_t>(targetCodec, sourceCodec);
            else if (isEqual<int64_t>(targetValue))         return getCopyPtrSameValueType<int64_t>(targetCodec, sourceCodec);
            else if (isEqual<float>(targetValue))           return getCopyPtrFloat(targetCodec, sourceCodec);
            else if (isEqual<double>(targetValue))          return getCopyPtrSameValueType<double>(targetCodec, sourceCodec);
            else if (isEqual<Vec3<int32_t>>(targetValue))   return getCopyPtrSameValueType<Vec3<int32_t>>(targetCodec, sourceCodec);
            else if (isEqual<Vec3<float>>(targetValue))     return getCopyPtrVec3f(targetCodec, sourceCodec);
            else if (isEqual<Vec3<double>>(targetValue))    return getCopyPtrSameValueType<Vec3<double>>(targetCodec, sourceCodec);
            else if (isEqual<Mat3<float>>(targetValue))     return getCopyPtrSameValueType<Mat3<float>>(targetCodec, sourceCodec);
            else if (isEqual<Mat3<double>>(targetValue))    return getCopyPtrSameValueType<Mat3<double>>(targetCodec, sourceCodec);
            else if (isEqual<Mat4<float>>(targetValue))     return getCopyPtrSameValueType<Mat4<float>>(targetCodec, sourceCodec);
            else if (isEqual<Mat4<double>>(targetValue))    return getCopyPtrSameValueType<Mat4<double>>(targetCodec, sourceCodec);
            else if (isEqual<Quat<float>>(targetValue))     return getCopyPtrSameValueType<Quat<float>>(targetCodec, sourceCodec);
            else if (isEqual<Quat<double>>(targetValue))    return getCopyPtrSameValueType<Quat<double>>(targetCodec, sourceCodec);
        }

        if (isNullCodec(targetCodec) && isNullCodec(sourceCodec)) {
            assert(targetValue != sourceValue);

            if (isEqual<int8_t>(targetValue)) {
                if (isEqual<int16_t>(sourceValue))          return getCopyPtrNullCodec<int8_t, int16_t>();
                if (isEqual<int32_t>(sourceValue))          return getCopyPtrNullCodec<int8_t, int32_t>();
                if (isEqual<int64_t>(sourceValue))          return getCopyPtrNullCodec<int8_t, int64_t>();
            } else if (isEqual<int16_t>(targetValue)) {
                if (isEqual<int8_t>(sourceValue))           return getCopyPtrNullCodec<int16_t, int8_t>();
                if (isEqual<int32_t>(sourceValue))          return getCopyPtrNullCodec<int16_t, int32_t>();
                if (isEqual<int64_t>(sourceValue))          return getCopyPtrNullCodec<int16_t, int64_t>();
            } else if (isEqual<int32_t>(targetValue)) {
                if (isEqual<int8_t>(sourceValue))           return getCopyPtrNullCodec<int32_t, int8_t>();
                if (isEqual<int16_t>(sourceValue))          return getCopyPtrNullCodec<int32_t, int16_t>();
                if (isEqual<int64_t>(sourceValue))          return getCopyPtrNullCodec<int32_t, int64_t>();
            } else if (isEqual<int64_t>(targetValue)) {
                if (isEqual<int8_t>(sourceValue))           return getCopyPtrNullCodec<int64_t, int8_t>();
                if (isEqual<int16_t>(sourceValue))          return getCopyPtrNullCodec<int64_t, int16_t>();
                if (isEqual<int32_t>(sourceValue))          return getCopyPtrNullCodec<int64_t, int32_t>();
            } else if (isEqual<float>(targetValue)) {
                if (isEqual<double>(sourceValue))           return getCopyPtrNullCodec<float, double>();
            } else if (isEqual<double>(targetValue)) {
                if (isEqual<float>(sourceValue))            return getCopyPtrNullCodec<double, float>();
            } else if (isEqual<Vec3<float>>(targetValue)) {
                if (isEqual<Vec3<double>>(sourceValue))     return getCopyPtrNullCodec<Vec3<float>, Vec3<double>>();
            } else if (isEqual<Vec3<double>>(targetValue)) {
                if (isEqual<Vec3<float>>(sourceValue))      return getCopyPtrNullCodec<Vec3<double>, Vec3<float>>();
            } else if (isEqual<Mat3<float>>(targetValue)) {
                if (isEqual<Mat3<double>>(sourceValue))     return getCopyPtrNullCodec<Mat3<float>, Mat3<double>>();
            } else if (isEqual<Mat3<double>>(targetValue)) {
                if (isEqual<Mat3<float>>(sourceValue))      return getCopyPtrNullCodec<Mat3<double>, Mat3<float>>();
            } else if (isEqual<Mat4<float>>(targetValue)) {
                if (isEqual<Mat4<double>>(sourceValue))     return getCopyPtrNullCodec<Mat4<float>, Mat4<double>>();
            } else if (isEqual<Mat4<double>>(targetValue)) {
                if (isEqual<Mat4<float>>(sourceValue))      return getCopyPtrNullCodec<Mat4<double>, Mat4<float>>();
            }

            // note: there is currently no conversion constructor for Quaternions
        }

        return copyPtrInvalid;
    }
}; // struct AttributeArrayCopyHelper


template <typename LeafT>
struct LeafBuffers
{
    using ValueT = typename LeafT::ValueType;

    struct Accessor
    {
        Accessor(LeafT& leafNode, bool stolen)
            : mLeafNode(leafNode)
            , mStolen(stolen) { }
        ~Accessor() { mLeafNode.updateValueMask(); }

        bool isStolen() const { return mStolen; }
        void set(Index n, const ValueT& value)
        {
            assert(!mStolen);
            assert(n < LeafT::NUM_VOXELS);
            mLeafNode.setOffsetOnly(n, value);
        }

    private:
        LeafT& mLeafNode;
        const bool mStolen;
    }; // struct Accessor

    LeafBuffers(LeafT** leafNodes, const size_t leafCount,
        const std::unordered_set<Coord>* stolenCoords)
        : mLeafNodes(leafNodes)
        , mLeafCount(leafCount)
        , mStolenCoords(stolenCoords) { }

    size_t size() const { return mLeafCount; }

    Accessor accessor(size_t n)
    {
        assert(mLeafNodes);
        assert(n < mLeafCount);
        assert(mLeafNodes[n]);
        LeafT& leaf = *mLeafNodes[n];
        const bool isStolen = mStolenCoords && mStolenCoords->find(leaf.origin()) != mStolenCoords->end();
        return Accessor(leaf, isStolen);
    }

    private:
        LeafT** mLeafNodes;
        const size_t mLeafCount;
        const std::unordered_set<Coord>* mStolenCoords;
}; // struct LeafBuffers


} // namespace point_move_internal


////////////////////////////////////////


template <typename PointDataGridT, typename DeformerT, typename FilterT, typename OtherFilterT>
inline void movePoints( PointDataGridT& points,
                        const math::Transform& transform,
                        DeformerT& deformer,
                        const FilterT& filter,
                        const std::vector<typename PointDataGridT::Ptr>& pointsToSteal,
                        const std::vector<typename PointDataGridT::ConstPtr>& pointsToCopy,
                        const OtherFilterT& mergeFilter,
                        bool deformMergedPoints,
                        bool threaded)
{
    using namespace openvdb;
    using namespace point_move_internal;

    (void) threaded;

    // TODO: handle case where points grid has no leaf nodes

    // build a thread-local bin for PointPartitioner (single-threaded)

    using PointDataTreeT = typename PointDataGridT::TreeType;
    using PointDataLeafT = typename PointDataTreeT::LeafNodeType;
    using LeafManagerT = typename tree::LeafManager<PointDataTreeT>;
    using IndexIterator = typename PointDataPartitioner<PointDataGridT>::IndexIterator;

    PointDataLeafArray<PointDataGridT> leafArray(points, transform,
        pointsToSteal, pointsToCopy, deformMergedPoints);

    const Index64 mergeCount = leafArray.gridCount()-1;

    PointDataPartitioner<PointDataGridT> pointDataPartitioner;
    pointDataPartitioner.construct(leafArray, deformer, filter,
        /*updatePosition=*/true, mergeFilter);

    const size_t leafCount = pointDataPartitioner.size();

    tbb::blocked_range<size_t> leafRange(0, leafCount);

    // extract the first attribute set

    const AttributeSet* existingAttributeSet = leafArray.attributeSet();
    assert(existingAttributeSet);

    // build and merge attribute sets into an AttributeInfo class

    AttributeSet::Info attributeInfo(*existingAttributeSet);

    for (size_t i = 1; i < leafArray.gridCount(); i++) {
        const AttributeSet* attributeSet = leafArray.attributeSet(static_cast<Index>(i));
        if (!attributeSet)  continue;
        AttributeSet::Info otherAttributeInfo(*attributeSet);
        attributeInfo.merge(otherAttributeInfo/*, merge_policy=*/);
    }

    // if there are no group index collisions across all the attribute set descriptors,
    // groups can be copied as normal attributes otherwise they need to be copied by value

    bool copyGroupsByValue = false;
    for (size_t i = 1; i < leafArray.gridCount(); i++) {
        const AttributeSet* attributeSet = leafArray.attributeSet(static_cast<Index>(i));
        if (!attributeSet)  continue;
        if (attributeInfo.descriptor().groupIndexCollision(attributeSet->descriptor())) {
            copyGroupsByValue = true;
            break;
        }
    }

    // compute string meta caches for each metamap

    std::vector<StringMetaCache> metaCaches(mergeCount);
    if (mergeCount > 0) {
        auto metaCacheOp =
            [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t n = range.begin(); n < range.end(); n++) {
                    const AttributeSet* attributeSet = leafArray.attributeSet(static_cast<Index>(n+1));
                    if (!attributeSet)  continue;
                    const auto& metadata = attributeSet->descriptor().getMetadata();
                    metaCaches[n].reset(metadata);
                }
            };

        tbb::blocked_range<size_t> metaCacheRange(0, mergeCount);
        if (threaded && mergeCount > 1) {
            tbb::parallel_for(metaCacheRange, metaCacheOp);
        } else {
            metaCacheOp(metaCacheRange);
        }
    }

    // allocate the leaf nodes in parallel

    std::unique_ptr<PointDataLeafT*[]> leafNodes(new PointDataLeafT*[leafCount]);

    {
        // acquire registry lock to avoid locking when appending attributes in parallel

        AttributeArray::ScopedRegistryLock lock;

        auto allocateAttributesOp =
            [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t n = range.begin(); n < range.end(); n++) {

                    PointDataLeafT* stolenLeaf = nullptr;

                    Index stealableLeafIndex = pointDataPartitioner.stealableLeafIndex(n);

                    bool canBeStolen = stealableLeafIndex != std::numeric_limits<Index>::max();

                    std::unique_ptr<AttributeSet> existingAttributeSet;
                    std::unique_ptr<AttributeSet> newAttributeSet;

                    if (canBeStolen) {
                        stolenLeaf = leafArray.stealLeaf(stealableLeafIndex);
                        assert(stolenLeaf);

                        // partially create the new leaf (no voxel buffer is allocated)
                        leafNodes[n] = new PointDataLeafT(PartialCreate(), pointDataPartitioner.origin(n));
                        // steal the voxel buffer of the stolen leafs
                        leafNodes[n]->swap(stolenLeaf->buffer());
                        // steal the attribute set of the stolen leaf
                        AttributeSet::UniquePtr stolenAttrSet = stolenLeaf->stealAttributeSet();
                        existingAttributeSet.reset(stolenAttrSet.release());
                    } else {
                        // allocate the new leaf
                        leafNodes[n] = new PointDataLeafT(pointDataPartitioner.origin(n));
                    }

                    // initialize the attribute set, stealing attribute arrays from an existing
                    // attribute set if one is supplied
                    const Index pointCount = static_cast<Index>(pointDataPartitioner.size(n));
                    newAttributeSet.reset(
                        new AttributeSet(attributeInfo, existingAttributeSet.get(), pointCount, &lock));
                    leafNodes[n]->replaceAttributeSet(newAttributeSet.release(),
                        /*allowMismatchingDescriptors=*/true);
                }
            };

        if (threaded) {
            tbb::parallel_for(leafRange, allocateAttributesOp);
        } else {
            allocateAttributesOp(leafRange);
        }
    }

    // track local-only moves and collapse source attributes where possible...

    std::unordered_set<Coord> stolenCoords;
    std::vector<uint8_t> localOnlyMoves(leafCount, uint8_t(0));

    for (size_t n = 0; n < leafCount; ++n) {
        Index stealableLeafIndex = pointDataPartitioner.stealableLeafIndex(n);
        if (stealableLeafIndex == std::numeric_limits<Index>::max())    continue;
        if (!leafArray.isLeafValid(stealableLeafIndex)) {
            stolenCoords.insert(pointDataPartitioner.origin(n));
        } else {
            const auto& info = leafArray.info(stealableLeafIndex);
            if (info.localOnly) {
                localOnlyMoves[n] = uint8_t(1);
            }
        }
    }

    bool hasLocalOnlyMoves = false;
    auto localOnlyOp =
        [&] (tbb::blocked_range<size_t>& range, bool result) -> bool {
            if (result)     return true;
            for (size_t n = range.begin(); n < range.end(); n++) {
                if (localOnlyMoves[n] == uint8_t(1))    return true;
            }
            return false;
        };
    if (threaded) {
        hasLocalOnlyMoves = tbb::parallel_reduce(leafRange, hasLocalOnlyMoves, localOnlyOp,
            [] (bool n, bool m) -> bool { return n || m; });
    }
    else {
        hasLocalOnlyMoves = localOnlyOp(leafRange, hasLocalOnlyMoves);
    }

    auto leafWasStolen = [&stolenCoords](const Coord& ijk) {
        return stolenCoords.find(ijk) != stolenCoords.end();
    };

    // assign the voxel values if a new leaf was allocated

    LeafBuffers<PointDataLeafT> leafBuffers(leafNodes.get(), leafCount,
        stolenCoords.empty() ? nullptr : &stolenCoords);

    pointDataPartitioner.assignVoxelValues(leafBuffers);

    const std::unique_ptr<Index[]>& sourceGridIndices = pointDataPartitioner.sourceGridIndices();

    // create a new points tree

    typename PointDataTreeT::Ptr newTree(new PointDataTreeT);

    // insert the leaf nodes in serial

    for (size_t n = 0; n < leafCount; n++) {
        newTree->addLeaf(leafNodes[n]);
    }
    leafNodes.reset();

    // compute the merged string metadata and remappings

    StringMetaCache mainMetaCache(attributeInfo.descriptor().getMetadata());

    std::vector<std::unordered_map<Index, Index>> metaRemaps(mergeCount);
    bool copyStringsByValue = false;

    if (mergeCount > 0) {
        // merge meta caches values into meta inserter and track any required remappings

        for (size_t i = 0; i < mergeCount; i++) {
            std::unordered_map<Index, Index>& remaps = metaRemaps[i];

            for (const auto& metaElement : metaCaches[i].map()) {
                const Name& key = metaElement.first;
                const Index desiredIndex = metaElement.second;

                auto it = mainMetaCache.map().find(key);
                assert(it != mainMetaCache.map().end());
                const Index remappedIndex = it->second;

                if (remappedIndex != desiredIndex) {
                    remaps.insert({desiredIndex, remappedIndex});
                }
            }
        }

        metaCaches.clear();
    }

    // if any remappings required, strings must be copied by value

    for (const auto& remaps : metaRemaps) {
        if (!remaps.empty()) {
            copyStringsByValue = true;
            break;
        }
    }

    if (copyGroupsByValue || copyStringsByValue)    hasLocalOnlyMoves = false;

    // steal all source leaf nodes

    size_t sourceLeafCount = leafArray.leafCount();

    // create a leaf manager for the target tree

    LeafManagerT targetLeafManager(*newTree);

    const AttributeSet::Descriptor::NameToPosMap& attributeMap =
        attributeInfo.descriptor().map();

    auto copyAttributesOp =
        [&](tbb::blocked_range<size_t>& r)
        {
            std::vector<AttributeArray::BufferAccessor> bufferAccessors(sourceLeafCount);

            // advance an iterator to the nth element of the attribute map
            // corresponding with the beginning of the range

            auto attributeIter = attributeMap.cbegin();
            for (int i = 0; i < int(r.begin()); i++)    ++attributeIter;
            assert(attributeIter != attributeMap.cend());

            for (size_t i = r.begin(); i < r.end(); i++, ++attributeIter) {

                const std::string& name = attributeIter->first;

                // ignore any group attributes if copying groups by value or
                // string attributes if copying strings by value
                if ((copyGroupsByValue && attributeInfo.arrayInfo(name).group) ||
                    (copyStringsByValue && attributeInfo.arrayInfo(name).string))   continue;

                const auto& attributeType = attributeInfo.descriptor().type(attributeIter->second);

                auto initializeBuffersOp =
                    [&](tbb::blocked_range<size_t>& range)
                    {
                        for (size_t n = range.begin(); n < range.end(); n++) {
                            // if (!leafArray.isLeafValid(n))  continue;
                            const PointDataLeafT* leaf = leafArray.leaf(n);
                            if (!leaf) {
                                bufferAccessors[n].reset();
                                continue;
                            }
                            size_t attributeIndex = leaf->attributeSet().find(name);
                            // attribute not found in this leaf
                            if (attributeIndex == AttributeSet::INVALID_POS) {
                                bufferAccessors[n].reset();
                                continue;
                            }

                            // position can be copied if leaf is const
                            const bool hasPositionCopy = attributeIndex == 0 &&
                                leafArray.isLeafConst(n) && bool(leafArray.info(n).positionArray);

                            const AttributeArray& array = hasPositionCopy ?
                                *leafArray.info(n).positionArray :
                                leaf->constAttributeArray(attributeIndex);
                            array.loadData();
                            bufferAccessors[n].reset(array);
                            // set a copy functor that handles many type and codec conversions
                            // note that this is more expensive than when attributes are of the same type
                            if (attributeType != array.type()) {
                                bufferAccessors[n].copy =
                                    AttributeArrayCopyHelper::getCopyPtr(attributeType, array.type());
                            }
                        }
                    };

                tbb::blocked_range<size_t> sourceLeafRange(0, sourceLeafCount);
                if (threaded) {
                    tbb::parallel_for(sourceLeafRange, initializeBuffersOp);
                } else {
                    initializeBuffersOp(sourceLeafRange);
                }

                // if no custom copy or conversion in use, disable this feature
                // in the copy routine for faster performance

                bool hasCopyConversion = false;
                auto copyCheckOp =
                    [&] (tbb::blocked_range<size_t>& range, bool result) -> bool {
                        if (result)     return true;
                        for (size_t n = range.begin(); n < range.end(); n++) {
                            if (bufferAccessors[n].copy)    return true;
                        }
                        return false;
                    };
                if (threaded) {
                    hasCopyConversion = tbb::parallel_reduce(sourceLeafRange, hasCopyConversion, copyCheckOp,
                        [] (bool n, bool m) -> bool { return n || m; });
                }
                else {
                    hasCopyConversion = copyCheckOp(sourceLeafRange, hasCopyConversion);
                }

                // iterate over the leaf nodes in the point tree.

                targetLeafManager.foreach(
                    [&](PointDataLeafT& leaf, size_t idx)
                    {
                        if (leafWasStolen(leaf.origin()))     return;

                        AttributeArray& array = leaf.attributeArray(name);
                        IndexIterator indexIterator = pointDataPartitioner.indices(idx);

                        if (hasCopyConversion) {
                            array.copyValuesUnsafe</*AllowConversion=*/true>(bufferAccessors, indexIterator);
                        } else {
                            array.copyValuesUnsafe</*AllowConversion=*/false>(bufferAccessors, indexIterator);
                        }

                        // if leaf has only local moves, collapse the attribute

                        if (hasLocalOnlyMoves && localOnlyMoves[idx] == uint8_t(1)) {
                            Index stealableLeafIndex = pointDataPartitioner.stealableLeafIndex(idx);
                            leafArray.collapseArray(stealableLeafIndex, name);
                        }
                    }
                );
            }
        };

    tbb::blocked_range<size_t> attributeRange(0, attributeMap.size());

    if (threaded && attributeMap.size() > 1) {
        tbb::parallel_for(attributeRange, copyAttributesOp);
    } else {
        copyAttributesOp(attributeRange);
    }

    // if copying groups by value, process buckets of groups in parallel with the bucket
    // size equal to the number of groups that can belong to one group attribute

#if 0
    if (copyGroupsByValue) {

        // load and expand all group attributes

        auto expandGroupAttributesOp =
            [&](tbb::blocked_range<size_t>& range)
            {
                // advance an iterator to the nth element of the attribute map
                // corresponding with the beginning of the range

                auto attributeIter = attributeMap.cbegin();
                for (int i = 0; i < int(range.begin()); i++)    ++attributeIter;
                assert(attributeIter != attributeMap.cend());

                for (size_t i = range.begin(); i < range.end(); i++, ++attributeIter) {

                    const std::string& name = attributeIter->first;

                    targetLeafManager.foreach(
                        [&](PointDataLeafT& leaf, size_t)
                        {
                            if (leafWasStolen(leaf.origin()))     return;

                            AttributeArray& array = leaf.attributeArray(name);
                            if (isGroup(array)) {
                                array.loadData();
                                array.expand();
                            }
                        }, threaded
                    );
                }
            };

        if (threaded && attributeMap.size() > 1) {
            tbb::parallel_for(attributeRange, expandGroupAttributesOp);
        } else {
            expandGroupAttributesOp(attributeRange);
        }

        // divide groups up into bins as writing to groups that belong to the same
        // attribute concurrently is not thread-safe

        GroupIndexBins groupIndexBins(attributeInfo.descriptor().groupMap());

        // perform group value copying

        auto copyGroupsOp =
            [&](tbb::blocked_range<size_t>& r)
            {
                std::vector<GroupHandle::UniquePtr> groupHandles(sourceLeafCount);

                for (size_t i = r.begin(); i < r.end(); i++) {
                    for (size_t j = 0; j < groupIndexBins.groupsPerBin(); j++) {
                        const std::string& groupName = groupIndexBins.group(i, j);
                        if (groupName.empty())  continue;

                        auto initializeHandlesOp =
                            [&](tbb::blocked_range<size_t>& range)
                            {
                                for (size_t n = range.begin(); n < range.end(); n++) {
                                    const PointDataLeafT* leaf = leafArray.leaf(n);
                                    if (leaf && leaf->attributeSet().descriptor().hasGroup(groupName)) {
                                        groupHandles[n].reset(new GroupHandle(leaf->groupHandle(groupName)));
                                    } else {
                                        groupHandles[n].reset();
                                    }
                                }
                            };

                        tbb::blocked_range<size_t> sourceLeafRange(0, sourceLeafCount);
                        if (threaded) {
                            tbb::parallel_for(sourceLeafRange, initializeHandlesOp);
                        } else {
                            initializeHandlesOp(sourceLeafRange);
                        }

                        // iterate over the leaf nodes in the point tree.

                        targetLeafManager.foreach(
                            [&](PointDataLeafT& leaf, size_t idx)
                            {
                                if (leafWasStolen(leaf.origin()))     return;

                                IndexIterator indexIterator = pointDataPartitioner.indices(idx);

                                GroupWriteHandle groupWriteHandle = leaf.groupWriteHandle(groupName);
                                groupWriteHandle.copyGroups(groupHandles, indexIterator);
                            }, threaded
                        );
                    }
                }
            };

        tbb::blocked_range<size_t> groupRange(0, groupIndexBins.bins());

        if (threaded && groupIndexBins.bins() > 1) {
            tbb::parallel_for(groupRange, copyGroupsOp);
        } else {
            copyGroupsOp(groupRange);
        }
    }
#endif

    if (copyStringsByValue) {
        // build a name-to-pos map for string attributes only

        AttributeSet::Descriptor::NameToPosMap stringAttributeMap;
        for (const auto& it : attributeInfo.descriptor().map()) {
            const auto& array = attributeInfo.arrayInfo(it.second);
            if (array.string) {
                stringAttributeMap.emplace(it.first, it.second);
            }
        }

        const auto& meta = attributeInfo.descriptor().getMetadata();

        auto copyStringsOp =
            [&](tbb::blocked_range<size_t>& r)
            {
                std::vector<StringAttributeHandle::UniquePtr> stringHandles(sourceLeafCount);

                for (size_t i = r.begin(); i < r.end(); i++) {

                    // advance an iterator to the nth element of the attribute map
                    // corresponding with the beginning of the range

                    auto attributeIter = stringAttributeMap.cbegin();
                    for (int i = 0; i < int(r.begin()); i++)    ++attributeIter;
                    assert(attributeIter != stringAttributeMap.cend());

                    const Name& stringName = attributeIter->first;
                    const size_t stringIndex = attributeIter->second;

                    // cache the StringAttributeHandles for all source leaf nodes

                    auto initializeHandlesOp =
                        [&](tbb::blocked_range<size_t>& range)
                        {
                            for (size_t n = range.begin(); n < range.end(); n++) {
                                const PointDataLeafT* leaf = leafArray.leaf(n);
                                if (!leaf) {
                                    stringHandles[n].reset();
                                    continue;
                                }
                                size_t idx = leaf->attributeSet().descriptor().find(stringName);
                                if (idx != AttributeSet::INVALID_POS) {
                                    const auto& sourceArray = leaf->constAttributeArray(idx);
                                    stringHandles[n].reset(new StringAttributeHandle(sourceArray, meta));
                                } else {
                                    stringHandles[n].reset();
                                }
                            }
                        };

                    tbb::blocked_range<size_t> sourceLeafRange(0, sourceLeafCount);
                    if (threaded) {
                        tbb::parallel_for(sourceLeafRange, initializeHandlesOp);
                    } else {
                        initializeHandlesOp(sourceLeafRange);
                    }

                    // iterate over the leaf nodes in the point tree.

                    targetLeafManager.foreach(
                        [&](PointDataLeafT& leaf, size_t idx)
                        {
                            if (leafWasStolen(leaf.origin())) {
                                Index stealableLeafIndex = pointDataPartitioner.stealableLeafIndex(idx);
                                assert(stealableLeafIndex != std::numeric_limits<Index>::max());
                                Index gridIndex = sourceGridIndices[stealableLeafIndex];

                                const std::unordered_map<Index, Index>& metaRemap = metaRemaps[gridIndex-1];

                                if (!metaRemap.empty()) {
                                    auto& attributeArray = leaf.attributeArray(stringIndex);
                                    StringAttributeWriteHandle stringWriteHandle(attributeArray, meta);
                                    stringWriteHandle.remapStrings(metaRemap);
                                }
                            } else {
                                IndexIterator indexIterator = pointDataPartitioner.indices(idx);

                                auto& attributeArray = leaf.attributeArray(stringIndex);
                                StringAttributeWriteHandle stringWriteHandle(attributeArray, meta);
                                stringWriteHandle.copyStrings(stringHandles, indexIterator, metaRemaps, sourceGridIndices);
                            }
                        }
                    );
                }
            };

        tbb::blocked_range<size_t> stringRange(0, stringAttributeMap.size());

        if (threaded && stringAttributeMap.size() > 1) {
            tbb::parallel_for(stringRange, copyStringsOp);
        } else {
            copyStringsOp(stringRange);
        }
    }

    pointDataPartitioner.clear();

    points.setTree(newTree);
    points.setTransform(transform.copy());

    // delete everything in the merge grids

    for (auto& mergeGridPtr : pointsToSteal) {
        mergeGridPtr->tree().clear();
    }
}


template <typename PointDataGridT, typename DeformerT, typename FilterT, typename OtherFilterT>
inline void movePoints( PointDataGridT& points,
                        DeformerT& deformer,
                        const FilterT& filter,
                        const std::vector<typename PointDataGridT::Ptr>& pointsToSteal,
                        const std::vector<typename PointDataGridT::ConstPtr>& pointsToCopy,
                        const OtherFilterT& mergeFilter,
                        bool deformMergedPoints,
                        bool threaded)
{
    movePoints(points, points.transform(), deformer, filter,
        pointsToSteal, pointsToCopy, mergeFilter, deformMergedPoints, threaded);
}


////////////////////////////////////////


/// @brief A Deformer that caches the resulting positions from evaluating another Deformer
/// @deprecated No longer needed as moving points now uses the point partitioner so can
/// be performed in a single pass
template <typename T>
class CachedDeformer
{
public:
    using LeafIndex = point_move_internal::LeafIndex;
    using Vec3T = typename math::Vec3<T>;
    using LeafVecT = std::vector<Vec3T>;
    using LeafMapT = std::unordered_map<LeafIndex, Vec3T>;

    // Internal data cache to allow the deformer to offer light-weight copying
    struct Cache
    {
        OPENVDB_DEPRECATED Cache() = default;

        struct Leaf
        {
            /// @brief clear data buffers and reset counter
            void clear() {
                vecData.clear();
                mapData.clear();
                totalSize = 0;
            }

            LeafVecT vecData;
            LeafMapT mapData;
            Index totalSize = 0;
        }; // struct Leaf

        std::vector<Leaf> leafs;
    }; // struct Cache

    /// Cache is expected to be persistent for the lifetime of the CachedDeformer
    OPENVDB_DEPRECATED explicit CachedDeformer(Cache& cache)
        : mCache(cache) { }

    /// Caches the result of evaluating the supplied point grid using the deformer and filter
    /// @param grid         the points to be moved
    /// @param deformer     the deformer to apply to the points
    /// @param filter       the point filter to use when evaluating the points
    /// @param threaded     enable or disable threading  (threading is enabled by default)
    template <typename PointDataGridT, typename DeformerT, typename FilterT>
    void evaluate(PointDataGridT& grid, DeformerT& deformer, const FilterT& filter,
        bool threaded = true)
    {
        using TreeT = typename PointDataGridT::TreeType;
        using LeafT = typename TreeT::LeafNodeType;
        using LeafManagerT = typename tree::LeafManager<TreeT>;
        LeafManagerT leafManager(grid.tree());

        // initialize cache
        auto& leafs = mCache.leafs;
        leafs.resize(leafManager.leafCount());

        const auto& transform = grid.transform();

        // insert deformed positions into the cache

        auto cachePositionsOp = [&](const LeafT& leaf, size_t idx) {

            const Index64 totalPointCount = leaf.pointCount();
            if (totalPointCount == 0)   return;

            // deformer is copied to ensure that it is unique per-thread

            DeformerT newDeformer(deformer);

            newDeformer.reset(leaf, idx);

            auto handle = AttributeHandle<Vec3f>::create(leaf.constAttributeArray("P"));

            auto& cache = leafs[idx];
            cache.clear();

            // only insert into a vector directly if the filter evaluates all points
            // and all points are stored in active voxels
            const bool useVector = filter.state() == index::ALL &&
                (leaf.isDense() || (leaf.onPointCount() == leaf.pointCount()));
            if (useVector) {
                cache.vecData.resize(totalPointCount);
            }

            for (auto iter = leaf.beginIndexOn(filter); iter; iter++) {

                // extract index-space position and apply index-space deformation (if defined)

                Vec3d position = handle->get(*iter) + iter.getCoord().asVec3d();

                // if deformer is designed to be used in index-space, perform deformation prior
                // to transforming position to world-space, otherwise perform deformation afterwards

                if (DeformerTraits<DeformerT>::IndexSpace) {
                    newDeformer.apply(position, iter);
                    position = transform.indexToWorld(position);
                }
                else {
                    position = transform.indexToWorld(position);
                    newDeformer.apply(position, iter);
                }

                // insert new position into the cache

                if (useVector) {
                    cache.vecData[*iter] = static_cast<Vec3T>(position);
                }
                else {
                    cache.mapData.insert({*iter, static_cast<Vec3T>(position)});
                }
            }

            // store the total number of points to allow use of an expanded vector on access

            if (!cache.mapData.empty()) {
                cache.totalSize = static_cast<Index>(totalPointCount);
            }
        };

        leafManager.foreach(cachePositionsOp, threaded);
    }

    /// Stores pointers to the vector or map and optionally expands the map into a vector
    /// @throw IndexError if idx is out-of-range of the leafs in the cache
    template <typename LeafT>
    void reset(const LeafT&, size_t idx)
    {
        if (idx >= mCache.leafs.size()) {
            if (mCache.leafs.empty()) {
                throw IndexError("No leafs in cache, perhaps CachedDeformer has not been evaluated?");
            } else {
                throw IndexError("Leaf index is out-of-range of cache leafs.");
            }
        }
        auto& cache = mCache.leafs[idx];
        if (!cache.mapData.empty()) {
            mLeafMap = &cache.mapData;
            mLeafVec = nullptr;
        }
        else {
            mLeafVec = &cache.vecData;
            mLeafMap = nullptr;
        }
    }

    /// Retrieve the new position from the cache
    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT& iter) const
    {
        assert(*iter >= 0);

        if (mLeafMap) {
            auto it = mLeafMap->find(*iter);
            if (it == mLeafMap->end())      return;
            position = static_cast<openvdb::Vec3d>(it->second);
        }
        else {
            assert(mLeafVec);

            if (mLeafVec->empty())          return;
            assert(*iter < mLeafVec->size());
            position = static_cast<openvdb::Vec3d>((*mLeafVec)[*iter]);
        }
    }

private:
    Cache& mCache;
    const LeafVecT* mLeafVec = nullptr;
    const LeafMapT* mLeafMap = nullptr;
}; // class CachedDeformer


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED
