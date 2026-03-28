// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED

#include <memory>

#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {


enum class ReadMode {
    Original,      // Read data as-is (default)
    TopologyOnly   // Read topology only, no buffer data
};

/// Global read configuration that applies to all grids
/// Contains common options shared across all codec types
struct OPENVDB_API ReadOptions
{
    // BBox Clip Data
    BBoxd clipBBox = BBoxd();

    // Grid Conversion Options
    ReadMode readMode = ReadMode::Original;
}; // struct ReadOptions

/// Global write configuration that applies to all grids
/// Contains common options shared across all codec types
struct OPENVDB_API WriteOptions
{
}; // struct WriteOptions


} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED
