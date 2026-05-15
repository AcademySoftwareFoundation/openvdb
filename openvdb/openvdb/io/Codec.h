// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED

#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <vector>
#include <mutex>

#include <openvdb/Types.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

// Forward declaration
class GridBase;

namespace io {

namespace internal {

// Global registration of codecs. These initialization functions are NOT intended
// to be called directly by the user.
// While they are thread-safe, they lack the early-exit logic of openvdb::initialize() -
// this can result in exceptions being thrown when attempting to register a codec that is
// already registered.
// They are also used extensively in unit tests.

OPENVDB_API void initialize();
OPENVDB_API void uninitialize();

} // namespace internal

////////////////////////////////////////

/// @brief Controls which operations a codec exposes.
///
/// Passed to codec registration to restrict how the codec may be used at
/// runtime. A @c ReadOnly codec can still be used to read existing files but
/// will not be selected as the write codec for new grids.
enum class CodecMode {
    /// Both @c readTopology()/@c readBuffers() and
    /// @c writeTopology()/@c writeBuffers() are enabled. This is the default.
    ReadWrite,
    /// Only @c readTopology() and @c readBuffers() are enabled; the codec
    /// will not be offered as a write target.
    ReadOnly
};

/// @brief Controls how grid data is read.
///
/// Passed via @c ReadOptions::readMode to influence which portions of a grid
/// are deserialized and whether any on-the-fly conversion is performed during read.
/// Codecs are expected to honour this setting in their @c readTopology() and
/// @c readBuffers() implementations.
enum class ReadMode {
    /// Deserialize both topology and all value buffers. This is the default
    /// behaviour and produces a fully populated grid identical to the one
    /// that was written.
    Original,
    /// Deserialize topology and value buffers, converting each voxel value
    /// to half-precision floating point on the fly. Codecs that do not
    /// support native half conversion should fall back to @c Original and
    /// record a @c ReadDiagnostic warning.
    Half,
    /// Deserialize topology and value buffers, converting each voxel value
    /// to @c bool (non-zero to @c true, zero to @c false). Produces a
    /// @c BoolGrid whose active set matches the source grid's active set.
    Bool,
    /// Deserialize topology only and promote the result to a mask grid,
    /// discarding all value data. Equivalent to reading @c TopologyOnly and
    /// then constructing a @c MaskGrid from the active-voxel set, but may
    /// be performed more efficiently inside the codec.
    Mask,
    /// Deserialize topology only; value buffers are skipped. The resulting
    /// grid has a valid tree structure (active/inactive state, node
    /// hierarchy) but leaf buffer data is left at its default (background)
    /// value. Useful when only the active-voxel mask is needed and
    /// avoiding the cost of reading large value buffers is desirable.
    TopologyOnly
};

/// @brief Base class for per-grid-type, codec-specific read options.
///
/// Codecs that require type-specific read configuration should derive from
/// this class and store an instance in @c ReadOptions::typeData, keyed by
/// the grid type string (e.g. @c "Vec3SGrid").
///
/// @par Example
/// @code
/// struct MyCodecReadOptions : public ReadTypedOptions {
///     bool someFlag = false;
/// };
///
/// ReadOptions options;
/// auto data = std::make_shared<MyCodecReadOptions>();
/// data->someFlag = true;
/// options.typeData["Vec3SGrid"] = data;
/// @endcode
struct OPENVDB_API ReadTypedOptions
{
    using Ptr = std::shared_ptr<ReadTypedOptions>;

    virtual ~ReadTypedOptions() = default;

    /// @brief Downcast a @c ReadTypedOptions::Ptr to a concrete derived type.
    ///
    /// Performs a @c static_cast to @c T after verifying via @c dynamic_cast
    /// (when @c OPENVDB_ASSERT is enabled) that @p data actually holds an
    /// instance of @c T. Calling this with a mismatched type is undefined
    /// behaviour in release builds.
    ///
    /// @tparam T  The concrete @c ReadTypedOptions subclass to cast to.
    /// @param  data  A shared pointer to the base @c ReadTypedOptions object.
    /// @return A reference to the underlying @c T instance.
    template<typename T>
    static T& cast(const Ptr& data) {
        OPENVDB_ASSERT(dynamic_cast<T*>(data.get()) != nullptr);
        return *static_cast<T*>(data.get());
    }
}; // struct ReadTypedOptions

/// @brief Global read configuration passed to every codec during deserialization.
///
/// An instance of this struct is threaded through all codec @c readTopology() and
/// @c readBuffers() calls so that site-wide policy (clipping, read mode) and any
/// type-specific overrides are available in a single place.
///
/// @par Clipping
/// When @c clipBBox is non-empty (i.e. @c !clipBBox.empty()), codecs should
/// restrict the voxel data they load to the region that intersects that world-space
/// bounding box. An empty bbox (the default) means no clipping is applied.
///
/// @par Read mode
/// @c readMode controls the granularity and on-the-fly conversion applied
/// during deserialization. The default, @c ReadMode::Original, loads both
/// topology and value buffers without modification.
/// @c ReadMode::Half, @c ReadMode::Bool, and @c ReadMode::Mask request
/// in-place type conversion as data is read.
/// @c ReadMode::TopologyOnly skips value buffers entirely, which can be
/// significantly faster when only the active-voxel mask is needed.
///
/// @par Per-type options
/// @c typeData allows callers to attach codec-specific configuration for
/// individual grid types. Entries are keyed by the grid type string
/// (e.g. @c "Vec3SGrid") and hold a @c ReadTypedOptions-derived object.
/// Codecs retrieve their entry via @c ReadTypedOptions::cast<T>().
struct OPENVDB_API ReadOptions
{
    /// World-space bounding box used to spatially clip the read.
    /// An empty bbox (the default) disables clipping.
    BBoxd clipBBox = BBoxd();

    /// Controls which portions of each grid are deserialized.
    /// Defaults to @c ReadMode::Original (full topology + value buffers).
    ReadMode readMode = ReadMode::Original;

    /// Optional per-grid-type codec configuration, keyed by grid type string.
    /// Values are @c ReadTypedOptions subclass instances; use
    /// @c ReadTypedOptions::cast<T>() to retrieve the concrete type.
    std::unordered_map<std::string, ReadTypedOptions::Ptr> typeData;
}; // struct ReadOptions

/// @brief Global write configuration passed to every codec during serialization.
///
/// Currently carries no fields, but is provided for forward compatibility:
/// future write-time options (e.g. compression hints, metadata policies) can
/// be added here without changing the codec interface.
struct OPENVDB_API WriteOptions
{
}; // struct WriteOptions

/// @brief Severity level for a read diagnostic.
/// @note Currently only @c Warning is defined.
enum class DiagnosticSeverity { Warning };

/// @brief A single read diagnostic message.
/// @note @c context is typically the grid name or another identifier that
///   locates the source of the message.
struct ReadDiagnostic {
    DiagnosticSeverity severity;
    std::string context;
    std::string message;
};

/// @brief Thread-safe collection of read diagnostics accumulated during a codec read.
///
/// @details Diagnostics report situations where a requested read option could
/// not be honoured by the codec — for example, clipping that was not applied
/// natively and had to fall back to a post-process, or an option that the codec
/// ignores entirely. The object must be explicitly enabled before any messages
/// are recorded; it is disabled by default.
struct OPENVDB_API ReadDiagnostics {
    ReadDiagnostics() = default;
    ReadDiagnostics(const ReadDiagnostics& other) {
        std::lock_guard<std::mutex> lock(other.mMutex);
        mEnabled = other.mEnabled;
        mDiagnostics = other.mDiagnostics;
    }
    ReadDiagnostics& operator=(const ReadDiagnostics& other) {
        if (this != &other) {
            std::scoped_lock lock(mMutex, other.mMutex);
            mEnabled = other.mEnabled;
            mDiagnostics = other.mDiagnostics;
        }
        return *this;
    }
    /// @brief Append a warning diagnostic. No-op if the object is disabled.
    /// @note This method is thread-safe.
    void addWarning(const std::string& context, const std::string& message) {
        if (!mEnabled) return;
        std::lock_guard<std::mutex> lock(mMutex);
        mDiagnostics.push_back({DiagnosticSeverity::Warning, context, message});
    }
    /// @brief Remove all recorded diagnostics.
    /// @note This method is thread-safe.
    void clear() {
        std::lock_guard<std::mutex> lock(mMutex);
        mDiagnostics.clear();
    }
    /// @brief Enable diagnostic recording.
    void enable() { mEnabled = true; }
    /// @brief Disable diagnostic recording.
    void disable() { mEnabled = false; }
    /// @brief Return @c true if diagnostic recording is enabled.
    bool enabled() const { return mEnabled; }
    /// @brief Return the list of recorded diagnostics.
    const std::vector<ReadDiagnostic>& diagnostics() const { return mDiagnostics; }
    /// @brief Return @c true if no diagnostics have been recorded.
    bool empty() const { return mDiagnostics.empty(); }
private:
    bool mEnabled = false;
    std::vector<ReadDiagnostic> mDiagnostics;
    mutable std::mutex mMutex;
}; // struct ReadDiagnostics

/// @brief Mutable per-operation state created by @c Codec::createData() and
///   passed into every read and write call on a given codec.
///
/// @details The main purpose of this class is to provide a mechanism for passing
/// additional state between the different methods in the codec. @c ReadOptions is
/// immutable and cannot carry per-operation state; the codec itself is stateless
/// by design (to allow sharing across threads) and therefore also cannot be used
/// for this purpose.
///
/// The base class provides the single field that every codec requires:
/// a shared pointer to the @c GridBase being populated (on read) or inspected
/// (on write). Derived codecs that need additional transient state should
/// subclass @c CodecData and add those fields.
///
/// A @c CodecData instance is created once per grid per read or write operation
/// by @c Codec::createData(), which also allocates the concrete @c GridBase
/// subclass and stores it in @c grid.
///
/// @par Deriving from CodecData
/// @code
/// struct MyCodecData : public io::CodecData {
///     // Additional per-operation state goes here
///     SomeCache intermediateBuffer;
/// };
/// @endcode
///
/// Inside @c Codec::createData(), the derived type is allocated and returned:
/// @code
/// io::CodecData::Ptr createData() override {
///     auto data = std::make_unique<MyCodecData>();
///     data->grid = MyGridType::create();
///     return data;
/// }
/// @endcode
///
/// In the read and write methods, downcast the reference back to the concrete
/// type to access the extra fields:
/// @code
/// void readBuffers(std::istream& is, io::CodecData& data, ...) override {
///     auto& myData = static_cast<MyCodecData&>(data);
///     MyGridType& grid = static_cast<MyGridType&>(*myData.grid);
///     // Use myData.intermediateBuffer, grid, etc.
/// }
/// @endcode
struct OPENVDB_API CodecData
{
    using Ptr = std::unique_ptr<CodecData>;

    virtual ~CodecData() = default;

    /// The grid being populated on read, or being serialized on write.
    SharedPtr<GridBase> grid;
}; // struct CodecData

/// @brief Abstract base class for grid I/O codecs.
///
/// @details A codec encapsulates the serialization and deserialization logic
/// for a specific grid type (or family of grid types). The I/O subsystem
/// selects a codec by name at runtime, calls @c createData() to allocate
/// per-operation state, and then dispatches to the appropriate read or write
/// methods.
///
/// Codecs must be stateless: all mutable per-operation state is stored in a
/// @c CodecData object (see above) that is created fresh for each grid read
/// or write. This ensures that a single registered codec instance can safely
/// handle concurrent I/O operations.
///
/// @par Implementing a new codec
///
/// 1. **Declare the codec struct**, deriving publicly from @c Codec (or from a
///    convenience intermediate that already handles topology, if one exists for
///    the grid family in question):
///    @code
///    struct MyCodec : public io::Codec { ... };
///    @endcode
///
/// 2. **Provide a static @c name() method** that returns the unique string
///    identifier under which the codec will be registered. Any globally
///    unique ASCII string is acceptable:
///    @code
///    static std::string name() { return "mycodec"; }
///    @endcode
///
/// 3. **Override @c createData()** (the only pure-virtual method). Allocate a
///    @c CodecData (or a derived subclass if extra per-operation state is
///    needed), create the appropriate @c GridBase subclass, assign it to
///    @c CodecData::grid, and return the object:
///    @code
///    io::CodecData::Ptr createData() override {
///        auto data = std::make_unique<io::CodecData>();
///        data->grid = MyGridType::create();
///        return data;
///    }
///    @endcode
///
/// 4. **Override the read and/or write methods** as required. All four
///    methods have default no-op implementations, so override only those that
///    the codec actually uses. The I/O layer separates topology (the tree
///    structure and active/inactive voxel mask) from value buffers (the actual
///    voxel data), allowing callers to request topology-only reads via
///    @c ReadOptions::readMode. Implementations should honour that setting:
///    @code
///    void readTopology(std::istream& is, io::CodecData& data,
///                      const io::ReadOptions& options,
///                      io::ReadDiagnostics& diagnostics) override
///    {
///        MyGridType& grid = static_cast<MyGridType&>(*data.grid);
///        // Deserialize the tree structure into grid...
///    }
///
///    void readBuffers(std::istream& is, io::CodecData& data,
///                     const io::ReadOptions& options,
///                     io::ReadDiagnostics& diagnostics) override
///    {
///        if (options.readMode == io::ReadMode::TopologyOnly) return;
///        MyGridType& grid = static_cast<MyGridType&>(*data.grid);
///        // Deserialize voxel values into grid...
///    }
///    @endcode
///    If a requested option (e.g. spatial clipping) cannot be honoured natively,
///    record a warning via @c diagnostics.addWarning() rather than silently ignoring it.
///
/// 5. **Register the codec** once at start-up, typically from the library's
///    @c initialize() function, using @c CodecRegistry::registerCodec<MyCodec>():
///    @code
///    io::CodecRegistry::registerCodec<MyCodec>();
///    @endcode
///
/// @note The codec struct itself is never copied; the registry takes ownership
///   of a single heap-allocated instance via @c Codec::Ptr
///   (@c std::unique_ptr<Codec>).
struct OPENVDB_API Codec
{
    using Ptr = std::unique_ptr<Codec>;

    virtual ~Codec() = default;

    /// @brief Allocate per-operation codec state, including the target grid.
    ///
    /// This is the only pure-virtual method. Implementations must create a
    /// concrete @c GridBase subclass appropriate for the codec, store it in
    /// @c CodecData::grid, and return the @c CodecData (or a derived subclass
    /// carrying additional state). The returned object is passed by reference
    /// to every subsequent read or write call for the same grid.
    ///
    /// @return A fully initialized @c CodecData whose @c grid field is non-null.
    virtual CodecData::Ptr createData() = 0;

    /// @brief Deserialize the grid topology (tree structure and active-voxel
    ///   mask) from @a is into the grid held by @a data.
    virtual void readTopology(std::istream& /*is*/, CodecData& /*data*/,
        const ReadOptions& /*options*/, ReadDiagnostics& /*diagnostics*/) { }

    /// @brief Deserialize all voxel-value buffers from @a is into the grid
    ///   held by @a data.
    ///
    /// The default implementation is a no-op. Override to populate leaf-node
    /// value buffers after topology has been established. When
    /// @c options.readMode is @c ReadMode::TopologyOnly this method will not
    /// be called by the I/O layer, so implementations may also guard against
    /// that mode internally for safety. If @c options.clipBBox is non-empty,
    /// restrict the loaded data to the region that intersects it; if the codec
    /// cannot honour clipping natively, fall back to a post-process and record
    /// a warning via @a diagnostics.
    virtual void readBuffers(std::istream& /*is*/, CodecData& /*data*/,
        const ReadOptions& /*options*/, ReadDiagnostics& /*diagnostics*/) { }

    /// @brief Serialize the grid topology (tree structure and active-voxel
    ///   mask) from @a grid to @a os.
    ///
    /// The default implementation is a no-op. Override when the codec stores
    /// topology as a distinct section that precedes the value buffers in the
    /// stream.
    virtual void writeTopology(std::ostream& /*os*/, const GridBase& /*grid*/,
        const WriteOptions& /*options*/) { }

    /// @brief Serialize all voxel-value buffers from @a grid to @a os.
    ///
    /// The default implementation is a no-op. Override to write the leaf-node
    /// value buffers that follow the topology section.
    virtual void writeBuffers(std::ostream& /*os*/, const GridBase& /*grid*/,
        const WriteOptions& /*options*/) { }
}; // struct Codec

/// @brief A thread-safe, process-global registry that maps codec names to
///   @c Codec instances.
///
/// Codecs are identified by a unique string name (typically provided by @c Codec::name())
/// and stored as @c Codec::Ptr (i.e. @c std::unique_ptr<Codec>).
///
/// @par Typical usage
/// @code
/// // Register a custom codec once at start-up (e.g. from initialize()):
/// CodecRegistry::registerCodec<MyCodec>();
/// @endcode
///
/// @note Attempting to register a codec whose name is already present throws
///   @c openvdb::KeyError. Call @c isRegistered() first when the caller
///   cannot guarantee uniqueness.
struct OPENVDB_API CodecRegistry
{
    /// Return @c true if a codec with the given @a name has been registered.
    static bool isRegistered(const std::string& name);

    /// Register a codec under the explicit string @a name, transferring
    /// ownership of @a codec to the registry.
    /// @throw KeyError if @a name is already registered.
    static void registerCodecByName(const std::string& name, Codec::Ptr&& codec);

    /// Convenience wrapper that registers @c CodecT using the name returned
    /// by @c CodecT::name(), constructing the instance internally.
    /// @throw KeyError if the codec is already registered.
    template <typename CodecT>
    static void registerCodec()
    {
        registerCodecByName(CodecT::name(), std::make_unique<CodecT>());
    }

    /// Return a raw (non-owning) pointer to the codec registered under
    /// @a name, or @c nullptr if no such codec exists.
    /// The returned pointer remains valid for the lifetime of the registry.
    static Codec* get(const std::string& name);

    /// Deregister all codecs and reset the registry to an empty state.
    static void clear();
}; // struct CodecRegistry

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODEC_HAS_BEEN_INCLUDED
