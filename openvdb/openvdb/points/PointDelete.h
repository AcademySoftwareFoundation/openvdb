// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Nick Avramoussis, Francisco Gochez, Dan Bailey
///
/// @file PointDelete.h
///
/// @brief Methods for deleting points based on group membership

#ifndef OPENVDB_POINTS_POINT_DELETE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_DELETE_HAS_BEEN_INCLUDED

#include "PointDataGrid.h"
#include "PointGroup.h"
#include "IndexIterator.h"
#include "IndexFilter.h"

#include <openvdb/tools/Prune.h>
#include <openvdb/tree/LeafManager.h>

#include <memory>
#include <string>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief   Delete points that are members of specific groups
///
/// @details This method will delete points which are members of any of the supplied groups and
///          will optionally drop the groups from the tree. An invert flag can be used to
///          delete points that belong to none of the groups.
///
/// @param   pointTree    the point tree
/// @param   groups       the groups from which to delete points
/// @param   invert       if enabled, points not belonging to any of the groups will be deleted
/// @param   drop         if enabled and invert is disabled, the groups will be dropped from the tree
///
/// @note    If the invert flag is true, none of the groups will be dropped after deleting points
///          regardless of the value of the drop parameter.

template <typename PointDataTreeT>
inline void deleteFromGroups(PointDataTreeT& pointTree,
                             const std::vector<std::string>& groups,
                             bool invert = false,
                             bool drop = true);

/// @brief   Delete points that are members of a group
///
/// @details This method will delete points which are members of the supplied group and will
///          optionally drop the group from the tree. An invert flag can be used to
///          delete points that belong to none of the groups.
///
/// @param   pointTree    the point tree with the group to delete
/// @param   group        the name of the group to delete
/// @param   invert       if enabled, points not belonging to any of the groups will be deleted
/// @param   drop         if enabled and invert is disabled, the group will be dropped from the tree
///
/// @note    If the invert flag is true, the group will not be dropped after deleting points
///          regardless of the value of the drop parameter.

template <typename PointDataTreeT>
inline void deleteFromGroup(PointDataTreeT& pointTree,
                            const std::string& group,
                            bool invert = false,
                            bool drop = true);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointDeleteImpl.h"

#endif // OPENVDB_POINTS_POINT_DELETE_HAS_BEEN_INCLUDED
