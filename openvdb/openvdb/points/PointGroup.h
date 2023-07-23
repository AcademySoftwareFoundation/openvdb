// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey
///
/// @file points/PointGroup.h
///
/// @brief  Point group manipulation in a VDB Point Grid.

#ifndef OPENVDB_POINTS_POINT_GROUP_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_GROUP_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include "IndexIterator.h" // FilterTraits
#include "IndexFilter.h" // FilterTraits
#include "AttributeSet.h"
#include "PointDataGrid.h"
#include "PointAttribute.h"
#include "PointCount.h"

#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Delete any group that is not present in the Descriptor.
///
/// @param groups        the vector of group names.
/// @param descriptor    the descriptor that holds the group map.
inline void deleteMissingPointGroups(   std::vector<std::string>& groups,
                                        const AttributeSet::Descriptor& descriptor);

/// @brief Appends a new empty group to the VDB tree.
///
/// @param tree          the PointDataTree to be appended to.
/// @param group         name of the new group.
template <typename PointDataTreeT>
inline void appendGroup(PointDataTreeT& tree,
                        const Name& group);

/// @brief Appends new empty groups to the VDB tree.
///
/// @param tree          the PointDataTree to be appended to.
/// @param groups        names of the new groups.
template <typename PointDataTreeT>
inline void appendGroups(PointDataTreeT& tree,
                         const std::vector<Name>& groups);

/// @brief Drops an existing group from the VDB tree.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param group         name of the group.
/// @param compact       compact attributes if possible to reduce memory - if dropping
///                      more than one group, compacting once at the end will be faster
template <typename PointDataTreeT>
inline void dropGroup(  PointDataTreeT& tree,
                        const Name& group,
                        const bool compact = true);

/// @brief Drops existing groups from the VDB tree, the tree is compacted after dropping.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param groups        names of the groups.
template <typename PointDataTreeT>
inline void dropGroups( PointDataTreeT& tree,
                        const std::vector<Name>& groups);

/// @brief Drops all existing groups from the VDB tree, the tree is compacted after dropping.
///
/// @param tree          the PointDataTree to be dropped from.
template <typename PointDataTreeT>
inline void dropGroups( PointDataTreeT& tree);

/// @brief Compacts existing groups of a VDB Tree to use less memory if possible.
///
/// @param tree          the PointDataTree to be compacted.
template <typename PointDataTreeT>
inline void compactGroups(PointDataTreeT& tree);

/// @brief Sets group membership from a PointIndexTree-ordered vector.
///
/// @param tree          the PointDataTree.
/// @param indexTree     the PointIndexTree.
/// @param membership    @c 1 if the point is in the group, 0 otherwise.
/// @param group         the name of the group.
/// @param remove        if @c true also perform removal of points from the group.
///
/// @note vector<bool> is not thread-safe on concurrent write, so use vector<short> instead
template <typename PointDataTreeT, typename PointIndexTreeT>
inline void setGroup(   PointDataTreeT& tree,
                        const PointIndexTreeT& indexTree,
                        const std::vector<short>& membership,
                        const Name& group,
                        const bool remove = false);

/// @brief Sets membership for the specified group for all points (on/off).
///
/// @param tree         the PointDataTree.
/// @param group        the name of the group.
/// @param member       true / false for membership of the group.
template <typename PointDataTreeT>
inline void setGroup(   PointDataTreeT& tree,
                        const Name& group,
                        const bool member = true);

/// @brief Sets group membership based on a provided filter.
///
/// @param tree     the PointDataTree.
/// @param group    the name of the group.
/// @param filter   filter data that is used to create a per-leaf filter
template <typename PointDataTreeT, typename FilterT>
inline void setGroupByFilter(   PointDataTreeT& tree,
                                const Name& group,
                                const FilterT& filter);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointGroupImpl.h"

#endif // OPENVDB_POINTS_POINT_GROUP_HAS_BEEN_INCLUDED
