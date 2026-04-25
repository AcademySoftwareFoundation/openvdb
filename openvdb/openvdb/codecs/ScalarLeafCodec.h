// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODECS_SCALARLEAFCODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODECS_SCALARLEAFCODEC_HAS_BEEN_INCLUDED

#include <openvdb/io/Compression.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace codecs {
namespace internal {

template <typename LeafT>
void writeScalarLeafBuffers(const LeafT& leaf, std::ostream& os, bool saveFloatAsHalf)
{
    using NodeMaskT = typename LeafT::NodeMaskType;

    // Write out the value mask.
    leaf.getValueMask().save(os);

    leaf.buffer().data(); // load values

    io::writeCompressedValues(os, leaf.buffer().data(), LeafT::SIZE,
        leaf.getValueMask(), /*childMask=*/NodeMaskT(), saveFloatAsHalf);
}


template <typename LeafT, typename StorageLeafT = LeafT>
void readScalarLeafBuffers(LeafT& leaf, std::istream& is, bool saveFloatAsHalf,
    const typename LeafT::ValueType& background, bool skip = false,
    const math::CoordBBox* clipBBox = nullptr)
{
    using ValueT = typename LeafT::ValueType;
    using NodeMaskT = typename LeafT::NodeMaskType;
    using StorageBufferT = typename StorageLeafT::Buffer;
    using StorageValueT = typename StorageLeafT::ValueType;

    constexpr Index SIZE = LeafT::SIZE;

    SharedPtr<io::StreamMetadata> meta = io::getStreamMetadataPtr(is);
    const bool seekable = meta && meta->seekable();

    auto& valueMask = leaf.getValueMask();

    // Load or seek the value mask
    if (seekable)   valueMask.seek(is);
    else            valueMask.load(is);

    if (skip) {
        if (seekable) {
            io::readCompressedValues<StorageValueT, NodeMaskT>(is, nullptr, SIZE, valueMask, saveFloatAsHalf);
        } else {
            StorageBufferT storageTemp;
            io::readCompressedValues(is, storageTemp.data(), SIZE, valueMask, saveFloatAsHalf);
        }
        // Clear the value mask so that the skipped leaf has no active
        // voxels. Without this, the leaf retains its on-disk active
        // topology even though no data was read into its buffer.
        valueMask.setOff();
        return;
    }

    if constexpr (std::is_same_v<typename LeafT::BuildType, ValueMask>) {
        // ValueMask leaf: value == active state, already captured in the value mask above.
        // Seek/consume past the storage buffer without populating any separate leaf buffer.
        if (seekable) {
            io::readCompressedValues<StorageValueT, NodeMaskT>(is, nullptr, SIZE, valueMask, saveFloatAsHalf);
        } else {
            StorageBufferT storageTemp;
            io::readCompressedValues(is, storageTemp.data(), SIZE, valueMask, saveFloatAsHalf);
        }
    } else if constexpr (std::is_same_v<ValueT, bool>) {
        // Bool leaf: must read storage values regardless of seekability, then convert.
        StorageBufferT storageTemp;
        io::readCompressedValues(is, storageTemp.data(), SIZE, valueMask, saveFloatAsHalf);
        for (Index i = 0; i < SIZE; ++i) {
            leaf.buffer().setValue(i, static_cast<bool>(storageTemp.getValue(i)));
        }
    } else {
        leaf.buffer().allocate();
        if constexpr (std::is_same_v<StorageValueT, ValueT>) {
            io::readCompressedValues(is, leaf.buffer().data(), SIZE, valueMask, saveFloatAsHalf);
        } else {
            StorageBufferT storageTemp;
            io::readCompressedValues(is, storageTemp.data(), SIZE, valueMask, saveFloatAsHalf);
            for (Index i = 0; i < SIZE; ++i) {
                leaf.buffer().setValue(i, static_cast<ValueT>(storageTemp.getValue(i)));
            }
        }
    }

    if (clipBBox) {
        leaf.clip(*clipBBox, background);
    }
}

} // namespace internal
} // namespace codecs
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODECS_SCALARLEAFCODEC_HAS_BEEN_INCLUDED
