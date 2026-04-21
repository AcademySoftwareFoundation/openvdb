// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODECS_SCALARCODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODECS_SCALARCODEC_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tools/Clip.h>
#include <openvdb/tools/NodeVisitor.h>
#include <openvdb/io/Codec.h>

#include "ScalarLeafCodec.h"
#include "TopologyCodec.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace codecs {
namespace internal {

template <typename TreeT>
struct WriteBuffersOp
{
    using LeafT = typename TreeT::LeafNodeType;

    WriteBuffersOp(std::ostream& _os, bool _saveFloatAsHalf)
        : os(_os)
        , saveFloatAsHalf(_saveFloatAsHalf) { }

    template <typename NodeT>
    void operator()(const NodeT&, size_t) { }

    void operator()(const LeafT& leaf, size_t)
    {
        writeScalarLeafBuffers(leaf, os, saveFloatAsHalf);
    }

    std::ostream& os;
    const bool saveFloatAsHalf;
}; // struct WriteBuffersOp


template <typename TreeT, typename StorageTreeT>
struct ReadBuffersOp
{
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;
    using StorageLeafT = typename StorageTreeT::LeafNodeType;

    ReadBuffersOp(std::istream& _is, bool _saveFloatAsHalf, const ValueT& _background,
        const CoordBBox* _clipBBox)
        : is(_is)
        , saveFloatAsHalf(_saveFloatAsHalf)
        , background(_background)
        , clipBBox(_clipBBox) { }

    void operator()(RootT& root, size_t)
    {
        if (clipBBox) {
            root.clip(*clipBBox);
        }
    }

    template <typename NodeT>
    void operator()(NodeT& node, size_t)
    {
        if (clipBBox) {
            node.clip(*clipBBox, background);
        }
    }

    void operator()(LeafT& leaf, size_t)
    {
        readScalarLeafBuffers<LeafT, StorageLeafT>(leaf, is, saveFloatAsHalf, background, /*skip=*/false, clipBBox);
    }

    std::istream& is;
    const bool saveFloatAsHalf;
    const ValueT& background;
    const CoordBBox* clipBBox = nullptr;
}; // struct ReadBuffersOp


// Free-standing function for both standard and conversion codec cases
// Uses StorageGridT = GridT by default, but allows different storage type for conversions
template<typename GridT, typename StorageGridT = GridT>
void scalarCodecReadBuffers(GridT& grid, std::istream& is, const io::ReadOptions& options)
{
    if (grid.hasMultiPassIO()) {
        OPENVDB_THROW(IoError, "Multi-pass IO is not supported in ScalarCodec");
    }

    using TreeT = typename GridT::TreeType;
    using StorageTreeT = typename StorageGridT::TreeType;

    io::checkFormatVersion(is);

    bool saveFloatAsHalf = grid.saveFloatAsHalf();

    auto& tree = grid.tree();
    tree.clearAllAccessors();

    std::unique_ptr<CoordBBox> clipIndexBBox;
    if (options.clipBBox.isSorted()) {
        clipIndexBBox = std::make_unique<CoordBBox>(grid.constTransform().worldToIndexNodeCentered(options.clipBBox));
    }

    // Works for both standard (TreeT == StorageTreeT) and conversion cases
    ReadBuffersOp<TreeT, StorageTreeT> readBuffersOp(is, saveFloatAsHalf, tree.background(),
        clipIndexBBox.get());
    tools::visitNodesDepthFirst(grid.tree(), readBuffersOp, /*idx=*/0, /*topDown=*/false);
}

// Free-standing function for write case (no StorageGridT needed)
template<typename GridT>
void scalarCodecWriteBuffers(const GridT& grid, std::ostream& os)
{
    using TreeType = typename GridT::TreeType;

    if (grid.hasMultiPassIO()) {
        OPENVDB_THROW(IoError, "Multi-pass IO is not supported in ScalarCodec");
    }

    WriteBuffersOp<TreeType> writeBuffersOp(os, grid.saveFloatAsHalf());
    tools::visitNodesDepthFirst(grid.tree(), writeBuffersOp);
}

} // namespace internal

template <typename GridT, typename StorageGridT = GridT, io::CodecMode Mode = io::CodecMode::ReadWrite>
struct OPENVDB_API ScalarCodec : public TopologyCodec<GridT, StorageGridT, Mode>
{
    static_assert(GridT::TreeType::RootNodeType::template SameConfiguration<
        typename StorageGridT::TreeType::RootNodeType>::value,
        "GridT and StorageGridT must have the same configuration");

    using Ptr = std::unique_ptr<ScalarCodec<GridT, StorageGridT, Mode>>;

    struct ScalarCodecData : public io::CodecData
    {
        std::vector<int> leafCoords;
    };

    ~ScalarCodec() noexcept = default;

    io::CodecData::Ptr createData() final
    {
        auto data = std::make_unique<ScalarCodecData>();
        data->grid = GridT::create();
        return data;
    }

    static inline std::string name()
    {
        if constexpr (std::is_same_v<GridT, StorageGridT>) {
            return GridT::gridType();
        } else {
            std::string buildType = typeNameAsString<typename GridT::BuildType>();
            return StorageGridT::gridType() + "_to_" + buildType;
        }
    }

    void readBuffers(std::istream& is, io::CodecData& data, const io::ReadOptions& options, io::ReadDiagnostics&) final
    {
        GridT& grid = static_cast<GridT&>(*data.grid);
        internal::scalarCodecReadBuffers<GridT, StorageGridT>(grid, is, options);
    }

    void writeBuffers(std::ostream& os, const GridBase& gridBase, const io::WriteOptions&) final
    {
        if constexpr (Mode == io::CodecMode::ReadOnly) return;

        const GridT& grid = static_cast<const GridT&>(gridBase);
        internal::scalarCodecWriteBuffers(grid, os);
    }
}; // struct ScalarCodec


} // namespace codecs
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODECS_SCALARCODEC_HAS_BEEN_INCLUDED
