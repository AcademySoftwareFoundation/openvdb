// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Nick Avramoussis
///
/// @file PointReplicate.h
///
/// @brief  Algorithms to duplicate points in PointDataGrids.

#ifndef OPENVDB_POINTS_POINT_REPLICATE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_REPLICATE_HAS_BEEN_INCLUDED

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/util/Assert.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Replicates points provided in a source grid into a new grid,
///        transfering and creating attributes found in a provided
///        attribute vector. If an attribute doesn't exist, it is ignored.
///        Position is always replicated, leaving the new points exactly
///        over the top of the source points.
/// @note  The position attribute must exist
/// @param source      The source grid to replicate points from
/// @param multiplier  The base number of points to replicate per point
/// @param attributes  Attributes to transfer to the new grid
/// @param scaleAttribute  A scale float attribute which multiplies the base
///                        multiplier to vary the point count per point.
/// @param replicationIndex  When provided, creates a replication attribute
///                          of the given name which holds the replication
///                          index. This can be subsequently used to modify
///                          the replicated points as a post process.
template <typename PointDataGridT>
typename PointDataGridT::Ptr
replicate(const PointDataGridT& source,
          const Index multiplier,
          const std::vector<std::string>& attributes,
          const std::string& scaleAttribute = "",
          const std::string& replicationIndex = "");

/// @brief Replicates points provided in a source grid into a new grid,
///        transfering and creating all attributes from the source grid.
///        Position is always replicated, leaving the new points exactly
///        over the top of the source points.
/// @note  The position attribute must exist
/// @param source      The source grid to replicate points from
/// @param multiplier  The base number of points to replicate per point
/// @param scaleAttribute  A scale float attribute which multiplies the base
///                        multiplier to vary the point count per point.
/// @param replicationIndex  When provided, creates a replication attribute
///                          of the given name which holds the replication
///                          index. This can be subsequently used to modify
///                          the replicated points as a post process.
template <typename PointDataGridT>
typename PointDataGridT::Ptr
replicate(const PointDataGridT& source,
          const Index multiplier,
          const std::string& scaleAttribute = "",
          const std::string& replicationIndex = "");

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointReplicateImpl.h"

#endif // OPENVDB_POINTS_POINT_REPLICATE_HAS_BEEN_INCLUDED
