// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Nick Avramoussis
///
/// @file PointScatterImpl.h
///

#ifndef OPENVDB_POINTS_POINT_SCATTER_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_SCATTER_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace point_scatter_internal
{

/// @brief initialise the topology of a PointDataGrid and ensure
/// everything is voxelized
/// @param grid   The source grid from which to base the topology generation
template<typename PointDataGridT, typename GridT>
inline typename PointDataGridT::Ptr
initialisePointTopology(const GridT& grid)
{
    typename PointDataGridT::Ptr points(new PointDataGridT);
    points->setTransform(grid.transform().copy());
    points->topologyUnion(grid);
    if (points->tree().hasActiveTiles()) {
        points->tree().voxelizeActiveTiles();
    }

    return points;
}

/// @brief Generate random point positions for a leaf node
/// @param leaf       The leaf node to initialize
/// @param descriptor The descriptor containing the position type
/// @param count      The number of points to generate
/// @param spread     The spread of points from the voxel center
/// @param rand01     The random number generator, expected to produce floating point
///                   values between 0 and 1.
template<typename PositionType,
         typename CodecT,
         typename RandGenT,
         typename LeafNodeT>
inline void
generatePositions(LeafNodeT& leaf,
                  const AttributeSet::Descriptor::Ptr& descriptor,
                  const Index64& count,
                  const float spread,
                  RandGenT& rand01)
{
    using PositionTraits = VecTraits<PositionType>;
    using ValueType = typename PositionTraits::ElementType;
    using PositionWriteHandle = AttributeWriteHandle<PositionType, CodecT>;

    leaf.initializeAttributes(descriptor, static_cast<Index>(count));

    // directly expand to avoid needlessly setting uniform values in the
    // write handle
    auto& array = leaf.attributeArray(0);
    array.expand(/*fill*/false);

    PositionWriteHandle pHandle(array, /*expand*/false);
    PositionType P;
    for (Index64 index = 0; index < count; ++index) {
        P[0] = (spread * (rand01() - ValueType(0.5)));
        P[1] = (spread * (rand01() - ValueType(0.5)));
        P[2] = (spread * (rand01() - ValueType(0.5)));
        pHandle.set(static_cast<Index>(index), P);
    }
}

} // namespace point_scatter_internal

/// @endcond

////////////////////////////////////////


template<
    typename GridT,
    typename RandGenT,
    typename PositionArrayT,
    typename PointDataGridT,
    typename InterrupterT>
inline typename PointDataGridT::Ptr
uniformPointScatter(const GridT& grid,
                    const Index64 count,
                    const unsigned int seed,
                    const float spread,
                    InterrupterT* interrupter)
{
    using PositionType = typename PositionArrayT::ValueType;
    using PositionTraits = VecTraits<PositionType>;
    using ValueType = typename PositionTraits::ElementType;
    using CodecType = typename PositionArrayT::Codec;

    using RandomGenerator = math::Rand01<ValueType, RandGenT>;

    using TreeType = typename PointDataGridT::TreeType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using LeafManagerT = tree::LeafManager<TreeType>;

    struct Local
    {
        /// @brief Get the prefixed voxel counts for each leaf node with an
        /// additional value to represent the end voxel count.
        /// See also LeafManager::getPrefixSum()
        static void getPrefixSum(LeafManagerT& leafManager,
                                 std::vector<Index64>& offsets)
        {
            Index64 offset = 0;
            offsets.reserve(leafManager.leafCount() + 1);
            offsets.push_back(0);
            const auto leafRange = leafManager.leafRange();
            for (auto leaf = leafRange.begin(); leaf; ++leaf) {
                offset += leaf->onVoxelCount();
                offsets.push_back(offset);
            }
        }
    };

    static_assert(PositionTraits::IsVec && PositionTraits::Size == 3,
                  "Invalid Position Array type.");

    if (spread < 0.0f || spread > 1.0f) {
        OPENVDB_THROW(ValueError, "Spread must be between 0 and 1.");
    }

    if (interrupter) interrupter->start("Uniform scattering with fixed point count");

    typename PointDataGridT::Ptr points =
        point_scatter_internal::initialisePointTopology<PointDataGridT>(grid);
    TreeType& tree = points->tree();
    if (!tree.cbeginLeaf()) return points;

    LeafManagerT leafManager(tree);
    const Index64 voxelCount = leafManager.activeLeafVoxelCount();
    assert(voxelCount != 0);

    const double pointsPerVolume = double(count) / double(voxelCount);
    const Index32 pointsPerVoxel = static_cast<Index32>(math::RoundDown(pointsPerVolume));
    const Index64 remainder = count - (pointsPerVoxel * voxelCount);

    if (remainder == 0) {
        return denseUniformPointScatter<
            GridT, RandGenT, PositionArrayT, PointDataGridT, InterrupterT>(
                grid, float(pointsPerVoxel), seed, spread, interrupter);
    }

    std::vector<Index64> voxelOffsets, values;
    std::thread worker(&Local::getPrefixSum, std::ref(leafManager), std::ref(voxelOffsets));

    {
        math::RandInt<Index64, RandGenT> gen(seed, 0, voxelCount-1);
        values.reserve(remainder);
        for (Index64 i = 0; i < remainder; ++i) values.emplace_back(gen());
    }

    worker.join();

    if (util::wasInterrupted<InterrupterT>(interrupter)) {
        tree.clear();
        return points;
    }

    tbb::parallel_sort(values.begin(), values.end());
    const bool fractionalOnly(pointsPerVoxel == 0);

    leafManager.foreach([&voxelOffsets, &values, fractionalOnly]
                        (LeafNodeType& leaf, const size_t idx)
    {
        const Index64 lowerOffset = voxelOffsets[idx]; // inclusive
        const Index64 upperOffset = voxelOffsets[idx + 1]; // exclusive
        assert(upperOffset > lowerOffset);

        const auto valuesEnd = values.end();
        auto lower = std::lower_bound(values.begin(), valuesEnd, lowerOffset);

        auto* const data = leaf.buffer().data();
        auto iter = leaf.beginValueOn();

        Index32 currentOffset(0);
        bool addedPoints(!fractionalOnly);
        while (lower != valuesEnd) {
            const Index64 vId = *lower;
            if (vId >= upperOffset) break;

            const Index32 nextOffset = Index32(vId - lowerOffset);
            iter.increment(nextOffset - currentOffset);
            currentOffset = nextOffset;
            assert(iter);

            auto& value = data[iter.pos()];
            value = value + 1; // no += operator support
            addedPoints = true;
            ++lower;
        }

        // deactivate this leaf if no points were added. This will speed up
        // the unthreaded rng
        if (!addedPoints) leaf.setValuesOff();
    });

    voxelOffsets.clear();
    values.clear();

    if (fractionalOnly) {
        tools::pruneInactive(tree);
        leafManager.rebuild();
    }

    const AttributeSet::Descriptor::Ptr descriptor =
        AttributeSet::Descriptor::create(PositionArrayT::attributeType());
    RandomGenerator rand01(seed);

    const auto leafRange = leafManager.leafRange();
    auto leaf = leafRange.begin();
    for (; leaf; ++leaf) {
        if (util::wasInterrupted<InterrupterT>(interrupter)) break;
        Index32 offset(0);
        for (auto iter = leaf->beginValueAll(); iter; ++iter) {
            if (iter.isValueOn()) {
                const Index32 value = Index32(pointsPerVolume + Index32(*iter));
                if (value == 0) leaf->setValueOff(iter.pos());
                else            offset += value;
            }
            // @note can't use iter.setValue(offset) on point grids
            leaf->setOffsetOnly(iter.pos(), offset);
        }

        // offset should always be non zero
        assert(offset != 0);
        point_scatter_internal::generatePositions<PositionType, CodecType>
            (*leaf, descriptor, offset, spread, rand01);
    }

    // if interrupted, remove remaining leaf nodes
    if (leaf) {
        for (; leaf; ++leaf) leaf->setValuesOff();
        tools::pruneInactive(tree);
    }

    if (interrupter) interrupter->end();
    return points;
}


////////////////////////////////////////


template<
    typename GridT,
    typename RandGenT,
    typename PositionArrayT,
    typename PointDataGridT,
    typename InterrupterT>
inline typename PointDataGridT::Ptr
denseUniformPointScatter(const GridT& grid,
                         const float pointsPerVoxel,
                         const unsigned int seed,
                         const float spread,
                         InterrupterT* interrupter)
{
    using PositionType = typename PositionArrayT::ValueType;
    using PositionTraits = VecTraits<PositionType>;
    using ValueType = typename PositionTraits::ElementType;
    using CodecType = typename PositionArrayT::Codec;

    using RandomGenerator = math::Rand01<ValueType, RandGenT>;

    using TreeType = typename PointDataGridT::TreeType;

    static_assert(PositionTraits::IsVec && PositionTraits::Size == 3,
                  "Invalid Position Array type.");

    if (pointsPerVoxel < 0.0f) {
        OPENVDB_THROW(ValueError, "Points per voxel must not be less than zero.");
    }

    if (spread < 0.0f || spread > 1.0f) {
        OPENVDB_THROW(ValueError, "Spread must be between 0 and 1.");
    }

    if (interrupter) interrupter->start("Dense uniform scattering with fixed point count");

    typename PointDataGridT::Ptr points =
        point_scatter_internal::initialisePointTopology<PointDataGridT>(grid);
    TreeType& tree = points->tree();
    auto leafIter = tree.beginLeaf();
    if (!leafIter) return points;

    const Index32 pointsPerVoxelInt = math::Floor(pointsPerVoxel);
    const double delta = pointsPerVoxel - float(pointsPerVoxelInt);
    const bool fractional = !math::isApproxZero(delta, 1.0e-6);
    const bool fractionalOnly = pointsPerVoxelInt == 0;

    const AttributeSet::Descriptor::Ptr descriptor =
        AttributeSet::Descriptor::create(PositionArrayT::attributeType());
    RandomGenerator rand01(seed);

    for (; leafIter; ++leafIter) {
        if (util::wasInterrupted<InterrupterT>(interrupter)) break;
        Index32 offset(0);
        for (auto iter = leafIter->beginValueAll(); iter; ++iter) {
            if (iter.isValueOn()) {
                offset += pointsPerVoxelInt;
                if (fractional && rand01() < delta) ++offset;
                else if (fractionalOnly) leafIter->setValueOff(iter.pos());
            }
            // @note can't use iter.setValue(offset) on point grids
            leafIter->setOffsetOnly(iter.pos(), offset);
        }

        if (offset != 0) {
            point_scatter_internal::generatePositions<PositionType, CodecType>
                (*leafIter, descriptor, offset, spread, rand01);
        }
    }

    // if interrupted, remove remaining leaf nodes
    const bool prune(leafIter || fractionalOnly);
    for (; leafIter; ++leafIter) leafIter->setValuesOff();

    if (prune) tools::pruneInactive(tree);
    if (interrupter) interrupter->end();
    return points;
}


////////////////////////////////////////


template<
    typename GridT,
    typename RandGenT,
    typename PositionArrayT,
    typename PointDataGridT,
    typename InterrupterT>
inline typename PointDataGridT::Ptr
nonUniformPointScatter(const GridT& grid,
                       const float pointsPerVoxel,
                       const unsigned int seed,
                       const float spread,
                       InterrupterT* interrupter)
{
    using PositionType = typename PositionArrayT::ValueType;
    using PositionTraits = VecTraits<PositionType>;
    using ValueType = typename PositionTraits::ElementType;
    using CodecType = typename PositionArrayT::Codec;

    using RandomGenerator = math::Rand01<ValueType, RandGenT>;

    using TreeType = typename PointDataGridT::TreeType;

    static_assert(PositionTraits::IsVec && PositionTraits::Size == 3,
                  "Invalid Position Array type.");
    static_assert(std::is_arithmetic<typename GridT::ValueType>::value,
                  "Scalar grid type required for weighted voxel scattering.");

    if (pointsPerVoxel < 0.0f) {
        OPENVDB_THROW(ValueError, "Points per voxel must not be less than zero.");
    }

    if (spread < 0.0f || spread > 1.0f) {
        OPENVDB_THROW(ValueError, "Spread must be between 0 and 1.");
    }

    if (interrupter) interrupter->start("Non-uniform scattering with local point density");

    typename PointDataGridT::Ptr points =
        point_scatter_internal::initialisePointTopology<PointDataGridT>(grid);
    TreeType& tree = points->tree();
    auto leafIter = tree.beginLeaf();
    if (!leafIter) return points;

    const AttributeSet::Descriptor::Ptr descriptor =
        AttributeSet::Descriptor::create(PositionArrayT::attributeType());
    RandomGenerator rand01(seed);
    const auto accessor = grid.getConstAccessor();

    for (; leafIter; ++leafIter) {
        if (util::wasInterrupted<InterrupterT>(interrupter)) break;
        Index32 offset(0);
        for (auto iter = leafIter->beginValueAll(); iter; ++iter) {
            if (iter.isValueOn()) {
                double fractional =
                    double(accessor.getValue(iter.getCoord())) * pointsPerVoxel;
                fractional = std::max(0.0, fractional);
                int count = int(fractional);
                if (rand01() < (fractional - double(count))) ++count;
                else if (count == 0) leafIter->setValueOff(iter.pos());
                offset += count;
            }
            // @note can't use iter.setValue(offset) on point grids
            leafIter->setOffsetOnly(iter.pos(), offset);
        }

        if (offset != 0) {
            point_scatter_internal::generatePositions<PositionType, CodecType>
                (*leafIter, descriptor, offset, spread, rand01);
        }
    }

    // if interrupted, remove remaining leaf nodes
    for (; leafIter; ++leafIter) leafIter->setValuesOff();

    tools::pruneInactive(points->tree());
    if (interrupter) interrupter->end();
    return points;
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_POINTS_POINT_SCATTER_IMPL_HAS_BEEN_INCLUDED
