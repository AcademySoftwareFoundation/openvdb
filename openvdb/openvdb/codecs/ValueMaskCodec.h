// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODECS_VALUEMASKCODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODECS_VALUEMASKCODEC_HAS_BEEN_INCLUDED

#include <openvdb/io/Codec.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Clip.h>

#include "TopologyCodec.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace codecs {
namespace internal {

template <typename GridT>
struct ReadValueMaskBuffersOp
{
    using TreeT = typename GridT::TreeType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    ReadValueMaskBuffersOp(std::istream& _is, bool _saveFloatAsHalf,
        const ValueT& _background, const CoordBBox* _clipBBox)
        : is(_is)
        , saveFloatAsHalf(_saveFloatAsHalf)
        , background(_background)
        , clipBBox(_clipBBox) { }

    void operator()(RootT& root, size_t)
    {
        // Clip root-level tiles and prune children that were clipped.
        if (clipBBox) {
            root.clip(*clipBBox);
        }
    }

    template <typename NodeT>
    void operator()(NodeT& node, size_t)
    {
        // Clip internal node tiles and prune children that were clipped.
        if (clipBBox) {
            node.clip(*clipBBox, background);
        }
    }

    void operator()(LeafT& leaf, size_t)
    {
        // Read in the value mask.
        leaf.getValueMask().load(is);
        // Read in the origin.
        Coord origin;
        is.read(reinterpret_cast<char*>(&origin), sizeof(Coord::ValueType) * 3);
        leaf.setOrigin(origin);
    }

    std::istream& is;
    const bool saveFloatAsHalf;
    const ValueT& background;
    const CoordBBox* clipBBox = nullptr;
}; // struct ReadValueMaskBuffersOp

template <typename GridT>
struct WriteValueMaskBuffersOp
{
    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;

    WriteValueMaskBuffersOp(std::ostream& _os, bool _saveFloatAsHalf)
        : os(_os)
        , saveFloatAsHalf(_saveFloatAsHalf) { }

    template <typename NodeT>
    void operator()(const NodeT&, size_t) { }

    void operator()(const LeafT& leaf, size_t)
    {
        // Write out the value mask.
        leaf.getValueMask().save(os);

        // Write out the origin.
        os.write(reinterpret_cast<const char*>(&leaf.origin()), sizeof(Coord::ValueType) * 3);
    }

    std::ostream& os;
    const bool saveFloatAsHalf;
}; // struct WriteValueMaskBuffersOp

} // namespace internal

template <typename GridT>
struct ValueMaskCodec : public TopologyCodec<GridT>
{
    using Ptr = std::unique_ptr<ValueMaskCodec<GridT>>;

    ~ValueMaskCodec() noexcept = default;

    static inline std::string name() { return GridT::gridType(); }

    void readBuffers(std::istream& is, io::CodecData& data, const io::ReadOptions& options, io::ReadDiagnostics&) final
    {
        GridT& grid = static_cast<GridT&>(*data.grid);

        if (grid.hasMultiPassIO()) {
            OPENVDB_THROW(IoError, "Multi-pass IO is not supported in ValueMaskCodec");
        }

        io::checkFormatVersion(is);

        bool saveFloatAsHalf = grid.saveFloatAsHalf();

        auto& tree = grid.tree();
        tree.clearAllAccessors();

        std::unique_ptr<CoordBBox> clipIndexBBox;
        if (options.clipBBox.isSorted()) {
            clipIndexBBox = std::make_unique<CoordBBox>(grid.constTransform().worldToIndexNodeCentered(options.clipBBox));
        }

        internal::ReadValueMaskBuffersOp<GridT> readBuffersOp(is, saveFloatAsHalf, tree.background(), clipIndexBBox.get());
        tools::visitNodesDepthFirst(grid.tree(), readBuffersOp, /*idx=*/0, /*topDown=*/false);
    }

    void writeBuffers(std::ostream& os, const GridBase& gridBase, const io::WriteOptions&) final
    {
        const GridT& grid = static_cast<const GridT&>(gridBase);

        if (grid.hasMultiPassIO()) {
            OPENVDB_THROW(IoError, "Multi-pass IO is not supported in ValueMaskCodec");
        }

        internal::WriteValueMaskBuffersOp<GridT> writeBuffersOp(os, grid.saveFloatAsHalf());
        tools::visitNodesDepthFirst(grid.tree(), writeBuffersOp);
    }
}; // struct ValueMaskCodec

} // namespace codecs
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODECS_VALUEMASKCODEC_HAS_BEEN_INCLUDED
