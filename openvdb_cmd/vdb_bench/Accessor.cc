// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Accessor.cc
///
/// @brief Group accessor: reads of the neighbors of every active voxel, with a
/// ValueAccessor and with a tree::NeighborAccessor, one read pattern per tool.
/// The ValueAccessor rows exist in every build. The NeighborAccessor rows exist
/// only when the library has tree/NeighborAccessor.h.
///
/// Compare the rows of one tool to see what a read pattern costs. Compare the
/// same row between two builds of the library to see what a change to the
/// accessor costs. Every row sums the values it reads in the order of the
/// leaves, so the hash is the same for every implementation of a pattern.

#include "Harness.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#if defined(__has_include)
#if __has_include(<openvdb/tree/NeighborAccessor.h>)
#include <openvdb/tree/NeighborAccessor.h>
#define BENCH_HAS_NEIGHBOR_ACCESSOR 1
#endif
#endif

namespace bench {

namespace {

using LeafType = FloatTree::LeafNodeType;
using Leaves = std::vector<const LeafType*>;

/// The leaves are split in ranges. Each range has its own accessor, made by makeAccessor().
/// The sum of a leaf is written at the index of the leaf, so the total does not depend on
/// which thread ran which range.
template<typename MakeAccessorFunction, typename LeafFunction>
void
forEachLeaf(const Leaves& leaves, std::vector<double>& sums, MakeAccessorFunction makeAccessor,
    LeafFunction leafFunction)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size(), 4),
        [&](const tbb::blocked_range<size_t>& range) {
            auto accessor = makeAccessor();
            for (size_t i = range.begin(); i != range.end(); ++i) {
                sums[i] = leafFunction(*leaves[i], accessor);
            }
        });
}

/// Time a pattern and record it. @a makeAccessor and @a leafFunction are as for forEachLeaf.
template<typename MakeAccessorFunction, typename LeafFunction>
void
patternRow(Context& context, const Leaves& leaves, const std::string& tool, const std::string& name,
    MakeAccessorFunction makeAccessor, LeafFunction leafFunction)
{
    if (!context.wanted(tool, name)) return;
    context.progress(tool, name);
    std::vector<double> sums(leaves.size(), 0.0);
    const Timing timing = context.time(
        []() { return 0; },
        [&](int) { forEachLeaf(leaves, sums, makeAccessor, leafFunction); });
    Digest digest;
    digest.addArray(sums);
    context.record(tool, name, timing, digest);
}

constexpr int kDim = int(LeafType::DIM);

////////////////////////////////////////
// ValueAccessor

using ConstAccessor = FloatGrid::ConstAccessor;

OPENVDB_NO_INLINE double
valueFaces(const LeafType& leaf, ConstAccessor& accessor)
{
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        const Coord xyz = iter.getCoord();
        sum += accessor.getValue(xyz.offsetBy(1, 0, 0)) + accessor.getValue(xyz.offsetBy(-1, 0, 0)) +
            accessor.getValue(xyz.offsetBy(0, 1, 0)) + accessor.getValue(xyz.offsetBy(0, -1, 0)) +
            accessor.getValue(xyz.offsetBy(0, 0, 1)) + accessor.getValue(xyz.offsetBy(0, 0, -1));
    }
    return sum;
}

OPENVDB_NO_INLINE double
valueCorners(const LeafType& leaf, ConstAccessor& accessor)
{
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        const Coord xyz = iter.getCoord();
        for (int i = 0; i < 8; ++i) {
            sum += accessor.getValue(xyz.offsetBy((i & 1) ? 1 : -1, (i & 2) ? 1 : -1, (i & 4) ? 1 : -1));
        }
    }
    return sum;
}

OPENVDB_NO_INLINE double
valueCenter(const LeafType& leaf, ConstAccessor& accessor)
{
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) sum += accessor.getValue(iter.getCoord());
    return sum;
}

OPENVDB_NO_INLINE double
valueFar(const LeafType& leaf, ConstAccessor& accessor)
{
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        sum += accessor.getValue(iter.getCoord().offsetBy(2 * kDim, 0, 0));
    }
    return sum;
}

OPENVDB_NO_INLINE double
valueFarAlternating(const LeafType& leaf, ConstAccessor& accessor)
{
    double sum = 0.0;
    int sign = 1;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        sum += accessor.getValue(iter.getCoord().offsetBy(sign * 2 * kDim, 0, 0));
        sign = -sign;
    }
    return sum;
}

////////////////////////////////////////
// NeighborAccessor

#ifdef BENCH_HAS_NEIGHBOR_ACCESSOR

using Neighbors = tree::NeighborAccessor<const FloatTree>;

OPENVDB_NO_INLINE double
neighborFacesIterator(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveTo(leaf.origin());
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        sum += accessor.getValue<1, 0, 0>(iter) + accessor.getValue<-1, 0, 0>(iter) +
            accessor.getValue<0, 1, 0>(iter) + accessor.getValue<0, -1, 0>(iter) +
            accessor.getValue<0, 0, 1>(iter) + accessor.getValue<0, 0, -1>(iter);
    }
    return sum;
}

OPENVDB_NO_INLINE double
neighborFacesOffset(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveTo(leaf.origin());
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        const Index offset = iter.pos();
        sum += accessor.getValue<1, 0, 0>(offset) + accessor.getValue<-1, 0, 0>(offset) +
            accessor.getValue<0, 1, 0>(offset) + accessor.getValue<0, -1, 0>(offset) +
            accessor.getValue<0, 0, 1>(offset) + accessor.getValue<0, 0, -1>(offset);
    }
    return sum;
}

/// The offsets are known at run time, as in a filter with a width option
OPENVDB_NO_INLINE double
neighborFacesRuntime(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveTo(leaf.origin());
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        sum += accessor.getValue(iter, 1, 0, 0) + accessor.getValue(iter, -1, 0, 0) +
            accessor.getValue(iter, 0, 1, 0) + accessor.getValue(iter, 0, -1, 0) +
            accessor.getValue(iter, 0, 0, 1) + accessor.getValue(iter, 0, 0, -1);
    }
    return sum;
}

/// A new accessor for every leaf, which pays for the construction every time
OPENVDB_NO_INLINE double
neighborFacesConstruct(const LeafType& leaf, const FloatTree& tree)
{
    Neighbors accessor(tree, leaf.origin());
    return neighborFacesIterator(leaf, accessor);
}

OPENVDB_NO_INLINE double
neighborCorners(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveTo(leaf.origin());
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        sum += accessor.getValue<1, 1, 1>(iter) + accessor.getValue<-1, 1, 1>(iter) +
            accessor.getValue<1, -1, 1>(iter) + accessor.getValue<-1, -1, 1>(iter) +
            accessor.getValue<1, 1, -1>(iter) + accessor.getValue<-1, 1, -1>(iter) +
            accessor.getValue<1, -1, -1>(iter) + accessor.getValue<-1, -1, -1>(iter);
    }
    return sum;
}

OPENVDB_NO_INLINE double
neighborCenter(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveTo(leaf.origin());
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) sum += accessor.getValue(iter);
    return sum;
}

OPENVDB_NO_INLINE double
neighborLeaf(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveToLeaf(leaf.origin());
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) sum += accessor.getLeafValueUnsafe(iter);
    return sum;
}

/// Two leaf widths away in x, the same region for every voxel of the leaf
OPENVDB_NO_INLINE double
neighborFar(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveTo(leaf.origin());
    double sum = 0.0;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) sum += accessor.getValue(iter, 2 * kDim, 0, 0);
    return sum;
}

/// Two leaf widths away in x and then in -x, so that the cache of far reads is never reused
OPENVDB_NO_INLINE double
neighborFarAlternating(const LeafType& leaf, Neighbors& accessor)
{
    accessor.moveTo(leaf.origin());
    double sum = 0.0;
    int sign = 1;
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        sum += accessor.getValue(iter, sign * 2 * kDim, 0, 0);
        sign = -sign;
    }
    return sum;
}

#endif // BENCH_HAS_NEIGHBOR_ACCESSOR

} // anonymous namespace


void
runAccessor(Context& context)
{
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(context.options().voxelSize);
    const FloatTree& tree = sphere->tree();
    Leaves leaves;
    tree.getNodes(leaves);
    const std::string label = voxelLabel(context.options().voxelSize);

    auto valueAccessor = [&]() { return sphere->getConstAccessor(); };
    const std::string value = "ValueAccessor";

    patternRow(context, leaves, "read center", value + " " + label, valueAccessor, valueCenter);
    patternRow(context, leaves, "read 6 faces", value + " " + label, valueAccessor, valueFaces);
    patternRow(context, leaves, "read 8 corners", value + " " + label, valueAccessor, valueCorners);
    patternRow(context, leaves, "read far, one region", value + " " + label, valueAccessor, valueFar);
    patternRow(context, leaves, "read far, alternating", value + " " + label, valueAccessor,
        valueFarAlternating);

#ifdef BENCH_HAS_NEIGHBOR_ACCESSOR
    auto neighbors = [&]() { return Neighbors(tree); };
    const std::string neighbor = "NeighborAccessor";

    patternRow(context, leaves, "read center", neighbor + " getValue " + label, neighbors, neighborCenter);
    patternRow(context, leaves, "read center", neighbor + " leaf only " + label, neighbors, neighborLeaf);
    patternRow(context, leaves, "read 6 faces", neighbor + " iterator " + label, neighbors,
        neighborFacesIterator);
    patternRow(context, leaves, "read 6 faces", neighbor + " offset " + label, neighbors,
        neighborFacesOffset);
    patternRow(context, leaves, "read 6 faces", neighbor + " run time " + label, neighbors,
        neighborFacesRuntime);
    patternRow(context, leaves, "read 6 faces", neighbor + " new per leaf " + label,
        [&]() { return &tree; },
        [](const LeafType& leaf, const FloatTree* treePointer) {
            return neighborFacesConstruct(leaf, *treePointer);
        });
    patternRow(context, leaves, "read 8 corners", neighbor + " " + label, neighbors, neighborCorners);
    patternRow(context, leaves, "read far, one region", neighbor + " " + label, neighbors, neighborFar);
    patternRow(context, leaves, "read far, alternating", neighbor + " " + label, neighbors,
        neighborFarAlternating);
#else
    std::cerr << "accessor: this library has no tree/NeighborAccessor.h, ValueAccessor rows only\n";
#endif
}

} // namespace bench
