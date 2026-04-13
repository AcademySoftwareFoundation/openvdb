// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODECS_POINTINDEXCODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODECS_POINTINDEXCODEC_HAS_BEEN_INCLUDED

#include <openvdb/io/Codec.h>

#include <openvdb/tools/PointIndexGrid.h>

#include "ScalarLeafCodec.h"
#include "TopologyCodec.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace codecs {
namespace internal {

template <typename GridT>
struct ReadPointIndexBuffersOp
{
    using TreeT = typename GridT::TreeType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    ReadPointIndexBuffersOp(std::istream& _is, bool _saveFloatAsHalf,
        const ValueT& _background)
        : is(_is)
        , saveFloatAsHalf(_saveFloatAsHalf)
        , background(_background) { }

    template <typename NodeT>
    void operator()(NodeT&, size_t) { }

    void operator()(LeafT& leaf, size_t)
    {
        using BaseLeaf = typename LeafT::BaseLeaf;

        // Read the value mask and voxel data via base class
        BaseLeaf& baseLeaf = static_cast<BaseLeaf&>(leaf);
        readScalarLeafBuffers(baseLeaf, is, saveFloatAsHalf, background, /*skip=*/false, /*clipBBox=*/nullptr);

        // Read the number of indices.
        Index64 numIndices = Index64(0);
        is.read(reinterpret_cast<char*>(&numIndices), sizeof(Index64));

        // Read the indices data.
        leaf.indices().resize(size_t(numIndices));
        is.read(reinterpret_cast<char*>(leaf.indices().data()), numIndices * sizeof(ValueT));

        // Reserved for future use.
        Index64 auxDataBytes = Index64(0);
        is.read(reinterpret_cast<char*>(&auxDataBytes), sizeof(Index64));
        if (auxDataBytes > 0) {
            // For now, read and discard any auxiliary data.
            std::unique_ptr<char[]> auxData{new char[auxDataBytes]};
            is.read(auxData.get(), auxDataBytes);
        }
    }

    std::istream& is;
    const bool saveFloatAsHalf;
    const ValueT& background;
}; // struct ReadPointIndexBuffersOp

template <typename GridT>
struct WritePointIndexBuffersOp
{
    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    WritePointIndexBuffersOp(std::ostream& _os, bool _saveFloatAsHalf)
        : os(_os)
        , saveFloatAsHalf(_saveFloatAsHalf) { }

    template <typename NodeT>
    void operator()(const NodeT&, size_t) { }

    void operator()(const LeafT& leaf, size_t)
    {
        using BaseLeaf = typename LeafT::BaseLeaf;

        // Write out the value mask and voxel values via base class
        const BaseLeaf& baseLeaf = static_cast<const BaseLeaf&>(leaf);
        writeScalarLeafBuffers(baseLeaf, os, saveFloatAsHalf);

        // Write the number of indices.
        Index64 numIndices = Index64(leaf.indices().size());
        os.write(reinterpret_cast<const char*>(&numIndices), sizeof(Index64));

        // Write the indices data.
        os.write(reinterpret_cast<const char*>(leaf.indices().data()), numIndices * sizeof(ValueT));

        // Reserved for future use.
        const Index64 auxDataBytes = Index64(0);
        os.write(reinterpret_cast<const char*>(&auxDataBytes), sizeof(Index64));
    }

    std::ostream& os;
    const bool saveFloatAsHalf;
}; // struct WritePointIndexBuffersOp

} // namespace internal

template <typename GridT>
struct OPENVDB_API PointIndexCodec : public TopologyCodec<GridT>
{
    using Ptr = std::unique_ptr<PointIndexCodec<GridT>>;

    ~PointIndexCodec() noexcept = default;

    static inline std::string name() { return GridT::gridType(); }

    void readBuffers(std::istream& is, io::CodecData& data, const io::ReadOptions& options, io::ReadDiagnostics& diagnostics) final
    {
        GridT& grid = static_cast<GridT&>(*data.grid);

        if (grid.hasMultiPassIO()) {
            OPENVDB_THROW(IoError, "Multi-pass IO is not supported in PointIndexCodec");
        }

        io::checkFormatVersion(is);

        bool saveFloatAsHalf = grid.saveFloatAsHalf();

        auto& tree = grid.tree();
        tree.clearAllAccessors();

        if (options.clipBBox.isSorted()) {
            diagnostics.addWarning(grid.getName(), "bounding box clipping is not supported for PointIndexGrids");
        }

        internal::ReadPointIndexBuffersOp<GridT> readBuffersOp(is, saveFloatAsHalf, tree.background());
        tools::visitNodesDepthFirst(grid.tree(), readBuffersOp, /*idx=*/0, /*topDown=*/false);
    }

    void writeBuffers(std::ostream& os, const GridBase& gridBase, const io::WriteOptions&) final
    {
        const GridT& grid = static_cast<const GridT&>(gridBase);

        if (grid.hasMultiPassIO()) {
            OPENVDB_THROW(IoError, "Multi-pass IO is not supported in PointIndexCodec");
        }

        internal::WritePointIndexBuffersOp<GridT> writeBuffersOp(os, grid.saveFloatAsHalf());
        tools::visitNodesDepthFirst(grid.tree(), writeBuffersOp);
    }
}; // struct PointIndexCodec

} // namespace codecs
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODECS_POINTINDEXCODEC_HAS_BEEN_INCLUDED
