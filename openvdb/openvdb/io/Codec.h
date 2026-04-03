// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED

#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>

#include <openvdb/Types.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

// Forward declaration
class GridBase;

namespace io {

namespace internal {

void initialize();
void uninitialize();

} // namespace internal

////////////////////////////////////////

enum class ReadMode {
    Original,      // Read data as-is (default)
    TopologyOnly   // Read topology only, no buffer data
};

/// Base class for per-grid-type codec-specific options
/// Derive from this to add codec-specific read configuration
struct OPENVDB_API ReadTypedOptions
{
    using Ptr = std::shared_ptr<ReadTypedOptions>;

    virtual ~ReadTypedOptions() = default;

    /// Type-safe casting helper for derived ReadTypedOptions types
    /// Uses OPENVDB_ASSERT to verify type safety when assertions are enabled
    template<typename T>
    static T& cast(const Ptr& data) {
        OPENVDB_ASSERT(dynamic_cast<T*>(data.get()) != nullptr);
        return *static_cast<T*>(data.get());
    }
}; // struct ReadTypedOptions

/// Global read configuration that applies to all grids
/// Contains common options shared across all codec types
struct OPENVDB_API ReadOptions
{
    using Ptr = std::shared_ptr<ReadOptions>;

    virtual ~ReadOptions() = default;

    // BBox Clip Data
    BBoxd clipBBox = BBoxd();

    // Grid Conversion Options
    ReadMode readMode = ReadMode::Original;

    // Per-grid-type codec-specific options, keyed by grid type string
    std::unordered_map<std::string, ReadTypedOptions::Ptr> typeData;
}; // struct ReadOptions

/// Global write configuration that applies to all grids
/// Contains common options shared across all codec types
struct OPENVDB_API WriteOptions
{
    using Ptr = std::shared_ptr<WriteOptions>;

    virtual ~WriteOptions() = default;
}; // struct WriteOptions

/// Per-codec mutable data created by Codec::createData()
/// Contains the grid and any codec-specific state
struct OPENVDB_API CodecData
{
    using Ptr = std::unique_ptr<CodecData>;

    virtual ~CodecData() = default;

    SharedPtr<GridBase> grid;
}; // struct CodecData

struct OPENVDB_API Codec
{
    using Ptr = std::unique_ptr<Codec>;

    virtual ~Codec() = default;

    /// Create codec data including a new grid of the appropriate type
    /// @return A new CodecData object containing the grid and any codec-specific state
    virtual CodecData::Ptr createData() = 0;

    /// Read the grid topology
    virtual void readTopology(std::istream&, CodecData&, const ReadOptions&) { }

    /// Read all data buffers for this grid
    virtual void readBuffers(std::istream&, CodecData&, const ReadOptions&) { }

    /// Write the grid topology
    virtual void writeTopology(std::ostream&, const GridBase&, const WriteOptions&) { }

    /// Write all data buffers for this grid
    virtual void writeBuffers(std::ostream&, const GridBase&, const WriteOptions&) { }
}; // struct Codec

struct OPENVDB_API CodecRegistry
{
    /// Return @c true if the given codec name is registered.
    static bool isRegistered(const std::string& name);

    /// Register a codec by name with a factory function.
    static void registerCodecByName(const std::string& name, Codec::Ptr&& codec);

    /// Register a codec by type.
    template <typename CodecT>
    static void registerCodec()
    {
        registerCodecByName(CodecT::name(), std::make_unique<CodecT>());
    }

    /// Return a pointer to the codec with the given name, or nullptr if not found.
    static Codec* get(const std::string& name);

    /// Clear the codec registry.
    static void clear();
}; // struct CodecRegistry

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED
