// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODECS_TOPOLOGYCODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODECS_TOPOLOGYCODEC_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/NodeManager.h>
#include <openvdb/tools/NodeVisitor.h>
#include <openvdb/io/Codec.h>
#include <openvdb/Grid.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace codecs {
namespace internal {

template <typename TreeT>
struct WriteTopologyOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    WriteTopologyOp(std::ostream& _os, bool _saveFloatAsHalf)
        : os(_os)
        , saveFloatAsHalf(_saveFloatAsHalf) { }

    void operator()(const RootT& root, size_t)
    {
        int32_t bufferCount = 1;
        os.write(reinterpret_cast<char*>(&bufferCount), sizeof(int32_t));

        background = &root.background();
        if (!saveFloatAsHalf) {
            os.write(reinterpret_cast<const char*>(background), sizeof(ValueT));
        } else {
            ValueT truncatedVal = io::truncateRealToHalf(*background);
            os.write(reinterpret_cast<const char*>(&truncatedVal), sizeof(ValueT));
        }
        io::setGridBackgroundValuePtr(os, background);

        const Index numTiles = root.tileCount(), numChildren = root.childCount();
        os.write(reinterpret_cast<const char*>(&numTiles), sizeof(Index));
        os.write(reinterpret_cast<const char*>(&numChildren), sizeof(Index));

        // Write tiles.
        for (auto iter = root.cbeginValueAll(); iter; ++iter) {
            const auto& ijk = iter.getCoord();
            os.write(reinterpret_cast<const char*>(ijk.asPointer()), 3 * sizeof(Int32));
            ValueT value;
            bool active = root.getTileValueUnsafe(ijk, value);
            os.write(reinterpret_cast<const char*>(&value), sizeof(ValueT));
            os.write(reinterpret_cast<const char*>(&active), sizeof(bool));
        }

        rootChildLevel = RootT::LEVEL-1;
    }

    template <typename NodeT>
    void operator()(const NodeT& node, size_t)
    {
        // Write origin for RootNode children only
        if (rootChildLevel == NodeT::LEVEL) {
            const auto& ijk = node.origin();
            os.write(reinterpret_cast<const char*>(ijk.asPointer()), 3 * sizeof(Int32));
        }

        const auto& childMask = node.getChildMask();
        const auto& valueMask = node.getValueMask();

        childMask.save(os);
        valueMask.save(os);

        {
            // Copy all of this node's values into an array.
            std::unique_ptr<ValueT[]> valuePtr(new ValueT[NodeT::NUM_VALUES]);
            ValueT* values = valuePtr.get();
            const ValueT zero = zeroVal<ValueT>();
            for (Index i = 0; i < NodeT::NUM_VALUES; ++i) {
                values[i] = (node.isChildMaskOff(i) ? node.getValueUnsafe(i) : zero);
            }
            // Compress (optionally) and write out the contents of the array.
            io::writeCompressedValues(os, values, NodeT::NUM_VALUES, valueMask, childMask, saveFloatAsHalf);
        }
    }

    void operator()(const LeafT& leaf, size_t)
    {
        leaf.getValueMask().save(os);
    }

    std::ostream& os;
    const bool saveFloatAsHalf;
    const ValueT* background = nullptr;
    Index rootChildLevel = std::numeric_limits<Index>::max();
}; // struct WriteTopologyOp


template <typename TreeT, typename StorageTreeT>
struct ReadTopologyOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using StorageValueT = typename StorageTreeT::ValueType;

    ReadTopologyOp(std::istream& _is, bool _saveFloatAsHalf)
        : is(_is)
        , saveFloatAsHalf(_saveFloatAsHalf) { }

    void operator()(RootT& root)
    {
        using ChildT = typename RootT::ChildNodeType;

        int32_t bufferCount;
        is.read(reinterpret_cast<char*>(&bufferCount), sizeof(int32_t));
        if (bufferCount != 1) OPENVDB_LOG_WARN("multi-buffer trees are no longer supported");

        // Delete the existing tree.
        root.clear();

        // Read a RootNode that was stored in the current format.

        if constexpr (std::is_same_v<ValueT, StorageValueT>) {
            is.read(reinterpret_cast<char*>(&background), sizeof(ValueT));
        } else {
            StorageValueT _background;
            is.read(reinterpret_cast<char*>(&_background), sizeof(StorageValueT));
            background = static_cast<ValueT>(_background);
        }

        root.setBackground(background, false);
        io::setGridBackgroundValuePtr(is, &root.background());

        Index numTiles = 0, numChildren = 0;
        is.read(reinterpret_cast<char*>(&numTiles), sizeof(Index));
        is.read(reinterpret_cast<char*>(&numChildren), sizeof(Index));

        Int32 vec[3];
        StorageValueT value;
        bool active;

        // Read tiles.
        for (Index n = 0; n < numTiles; ++n) {
            is.read(reinterpret_cast<char*>(vec), 3 * sizeof(Int32));
            is.read(reinterpret_cast<char*>(&value), sizeof(StorageValueT));
            is.read(reinterpret_cast<char*>(&active), sizeof(bool));
            Coord origin(vec);
            if constexpr (std::is_same_v<ValueT, StorageValueT>) {
                root.addTile(origin, value, active);
            } else {
                root.addTile(origin, static_cast<ValueT>(value), active);
            }
        }

        // Read child nodes.
        for (Index n = 0; n < numChildren; ++n) {
            is.read(reinterpret_cast<char*>(vec), 3 * sizeof(Int32));
            Coord origin(vec);
            ChildT* child = new ChildT(PartialCreate(), origin, background);
            (*this)(*child);
            root.addChild(child);
        }
    }

    template <typename NodeT>
    void operator()(NodeT& node)
    {
        using ChildT = typename NodeT::ChildNodeType;
        using NodeMaskT = typename NodeT::NodeMaskType;

        NodeMaskT childMask, valueMask;
        childMask.load(is);
        valueMask.load(is);
        node.setValueMaskUnsafe(valueMask);

        const Index numValues = NodeT::NUM_VALUES;
        {
            // Read in (and uncompress, if necessary) all of this node's values
            // into a contiguous array.
            std::unique_ptr<StorageValueT[]> valuePtr(new StorageValueT[numValues]);
            StorageValueT* values = valuePtr.get();
            io::readCompressedValues(is, values, numValues, valueMask, saveFloatAsHalf);

            // Copy values from the array into this node's table.
            if constexpr (std::is_same_v<ValueT, StorageValueT>) {
                for (auto iter = node.beginValueAll(); iter; ++iter) {
                    node.setValueOnlyUnsafe(iter.pos(), values[iter.pos()]);
                }
            } else {
                for (auto iter = node.beginValueAll(); iter; ++iter) {
                    node.setValueOnlyUnsafe(iter.pos(), static_cast<ValueT>(values[iter.pos()]));
                }
            }
        }

        // Read in all child nodes and insert them into the table at their proper locations.
        for (auto iter = childMask.beginOn(); iter; ++iter) {
            Coord origin = node.offsetToGlobalCoord(iter.pos());
            auto* child = new ChildT(PartialCreate(), origin, background);
            (*this)(*child);
            node.setChildUnsafe(iter.pos(), child);
        }
    }

    void operator()(LeafT& leaf)
    {
        typename LeafT::NodeMaskType valueMask;
        valueMask.load(is);
        leaf.setValueMask(valueMask);
    }

    std::istream& is;
    bool saveFloatAsHalf;
    ValueT background;
}; // struct ReadTopologyOp

template <typename TreeT>
struct SetTilesToBackgroundOp
{
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    explicit SetTilesToBackgroundOp(const ValueT& background)
        : mBackground(background) { }

    bool operator()(RootT& root, size_t) const
    {
        for (auto it = root.beginValueAll(); it; ++it) {
            it.setValue(mBackground);
        }
        return true;
    }

    template<typename NodeT>
    bool operator()(NodeT& node, size_t) const
    {
        for (Index i = 0; i < NodeT::NUM_VALUES; ++i) {
            if (node.isChildMaskOff(i)) {
                node.setValueOnlyUnsafe(i, mBackground);
            }
        }
        return true;
    }

    bool operator()(LeafT&, size_t) const { return true; }

private:
    const ValueT mBackground;
}; // struct SetTilesToBackgroundOp

template<typename TreeT>
void setTilesToBackground(TreeT& tree)
{
    const typename TreeT::ValueType& background = tree.root().background();
    SetTilesToBackgroundOp<TreeT> op(background);
    tree::DynamicNodeManager<TreeT> nodeManager(tree);
    nodeManager.foreachTopDown(op);
}

// Free-standing function for read case (supports type conversion via StorageGridT)
template<typename GridT, typename StorageGridT = GridT>
void topologyCodecReadTopology(GridBase& gridBase, std::istream& is, const io::ReadOptions& options)
{
    io::checkFormatVersion(is);

    GridT& grid = static_cast<GridT&>(gridBase);
    grid.tree().clearAllAccessors();

    internal::ReadTopologyOp<typename GridT::TreeType, typename StorageGridT::TreeType> readTopologyOp(is, grid.saveFloatAsHalf());
    readTopologyOp(grid.tree().root());

    if (options.readMode == io::ReadMode::TopologyOnly) {
        internal::setTilesToBackground(grid.tree());
        return;
    }
}

// Free-standing function for write case (no StorageGridT needed)
template<typename GridT>
void topologyCodecWriteTopology(const GridBase& gridBase, std::ostream& os)
{
    const GridT& grid = static_cast<const GridT&>(gridBase);
    using TreeType = typename GridT::TreeType;

    internal::WriteTopologyOp<TreeType> writeTopologyOp(os, grid.saveFloatAsHalf());
    tools::visitNodesDepthFirst(grid.tree(), writeTopologyOp);
}

} // namespace internal

template <typename GridT, typename StorageGridT = GridT>
struct TopologyCodec : public io::Codec
{
    using Ptr = std::unique_ptr<TopologyCodec<GridT, StorageGridT>>;

    ~TopologyCodec() noexcept = default;

    io::CodecData::Ptr createData() override
    {
        auto data = std::make_unique<io::CodecData>();
        data->grid = GridT::create();
        return data;
    }

    void readTopology(std::istream& is, io::CodecData& data, const io::ReadOptions& options) final
    {
        internal::topologyCodecReadTopology<GridT, StorageGridT>(*data.grid, is, options);
    }

    void writeTopology(std::ostream& os, const GridBase& gridBase, const io::WriteOptions&) final
    {
        internal::topologyCodecWriteTopology<GridT>(gridBase, os);
    }
}; // struct TopologyCodec


} // namespace codecs
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODECS_TOPOLOGYCODEC_HAS_BEEN_INCLUDED
