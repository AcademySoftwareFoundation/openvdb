// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/points.cc

#include <openvdb/openvdb.h>
#include "PointDataGrid.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

template <typename ArrayT> struct RegisterArray { inline void operator()() { ArrayT::registerType(); } };

void
internal::initialize()
{
    AttributeTypes::foreach<RegisterArray>();

    // Register types associated with point data grids.
    Metadata::registerType(typeNameAsString<PointDataIndex32>(), Int32Metadata::createMetadata);
    Metadata::registerType(typeNameAsString<PointDataIndex64>(), Int64Metadata::createMetadata);
    if (!PointDataGrid::isRegistered()) PointDataGrid::registerGrid();
}


void
internal::uninitialize()
{
    AttributeArray::clearRegistry();
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
