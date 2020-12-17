// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file test/integration/CompareGrids.h
///
/// @authors Francisco Gochez, Nick Avramoussis
///
/// @brief  Functions for comparing entire VDB grids and generating
///   reports on their differences
///

#ifndef OPENVDB_POINTS_UNITTEST_COMPARE_GRIDS_INCLUDED
#define OPENVDB_POINTS_UNITTEST_COMPARE_GRIDS_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/Prune.h>

namespace unittest_util
{


struct ComparisonSettings
{
    bool mCheckTransforms = true;         // Check grid transforms
    bool mCheckTopologyStructure = true;  // Checks node (voxel/leaf/tile) layout
    bool mCheckActiveStates = true;       // Checks voxel active states match
    bool mCheckBufferValues = true;       // Checks voxel buffer values match

    bool mCheckDescriptors = true;        // Check points leaf descriptors
    bool mCheckArrayValues = true;        // Checks attribute array sizes and values
    bool mCheckArrayFlags = true;         // Checks attribute array flags
};

/// @brief The results collected from compareGrids()
///
struct ComparisonResult
{
    ComparisonResult(std::ostream& os = std::cout)
        : mOs(os)
        , mDifferingTopology(openvdb::MaskGrid::create())
        , mDifferingValues(openvdb::MaskGrid::create()) {}

    std::ostream& mOs;
    openvdb::MaskGrid::Ptr mDifferingTopology; // Always empty if mCheckActiveStates is false
    openvdb::MaskGrid::Ptr mDifferingValues;   // Always empty if mCheckBufferValues is false
                                               // or if mCheckBufferValues and mCheckArrayValues
                                               // is false for point data grids
};

template <typename GridType>
bool compareGrids(ComparisonResult& resultData,
                  const GridType& firstGrid,
                  const GridType& secondGrid,
                  const ComparisonSettings& settings,
                  const openvdb::MaskGrid::ConstPtr maskGrid,
                  const typename GridType::ValueType tolerance =
                    openvdb::zeroVal<typename GridType::ValueType>());

bool compareUntypedGrids(ComparisonResult& resultData,
                         const openvdb::GridBase& firstGrid,
                         const openvdb::GridBase& secondGrid,
                         const ComparisonSettings& settings,
                         const openvdb::MaskGrid::ConstPtr maskGrid);

} // namespace unittest_util

#endif // OPENVDB_POINTS_UNITTEST_COMPARE_GRIDS_INCLUDED

