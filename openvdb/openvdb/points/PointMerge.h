// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey
///
/// @file PointMerge.h
///
/// @brief Ability to merge VDB Points.

#ifndef OPENVDB_POINTS_POINT_MERGE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_MERGE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointMove.h>

#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


template <typename GridT>
inline typename GridT::Ptr
mergePoints(const std::vector<typename GridT::Ptr>& gridsToSteal,
            const std::vector<typename GridT::ConstPtr>& gridsToCopy = std::vector<typename GridT::ConstPtr>(),
            bool threaded = true)
{
    using GridPtr = typename GridT::Ptr;

    const std::vector<GridPtr> adjustedGridsToSteal(gridsToSteal.begin()+1, gridsToSteal.end());

    GridPtr result;
    if (!gridsToSteal.empty()) {
        result = gridsToSteal.front();
    } else if (!gridsToCopy.empty()) {
        GridBase::Ptr gridBase = gridsToCopy.front()->copyGridWithNewTree();
        result = gridPtrCast<GridT>(gridBase);
    } else {
        return result;
    }

    NullDeformer nullDeformer;
    NullFilter nullFilter;
    movePoints(*result, nullDeformer, nullFilter, adjustedGridsToSteal, gridsToCopy,
        nullFilter, /*deformMergedPoints=*/false, threaded);

    return result;
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_MERGE_HAS_BEEN_INCLUDED
