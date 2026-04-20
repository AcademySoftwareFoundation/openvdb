// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_CODECS_POINTDATACODEC_HAS_BEEN_INCLUDED
#define OPENVDB_IO_CODECS_POINTDATACODEC_HAS_BEEN_INCLUDED

#include <openvdb/io/Codec.h>

#include <openvdb/tools/Clip.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/AttributeSet.h>
#include <openvdb/points/StreamCompression.h>

#include "ScalarLeafCodec.h"
#include "TopologyCodec.h"

#include <set>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace codecs {
namespace internal {

/// Look up an existing paged stream or create a new one for the given attribute index
template <typename PagedStreamPtrT>
inline typename PagedStreamPtrT::element_type* getOrCreatePagedStream(
    std::map<Index, PagedStreamPtrT>& pagedStreams, Index attributeIndex)
{
    auto it = pagedStreams.find(attributeIndex);
    if (it != pagedStreams.end()) return it->second.get();
    using PagedStreamT = typename PagedStreamPtrT::element_type;
    pagedStreams[attributeIndex] = std::make_shared<PagedStreamT>();
    return pagedStreams[attributeIndex].get();
}

////////////////////////////////////////
// Read-side functions

template <typename LeafT>
inline void readPointDataVoxelSizes(const std::vector<LeafT*>& leaves,
    std::istream& is, std::map<Coord, uint16_t>& voxelBufferSizes)
{
    for (auto* leaf : leaves) {
        uint16_t voxelBufferSize;
        is.read(reinterpret_cast<char*>(&voxelBufferSize), sizeof(uint16_t));
        voxelBufferSizes[leaf->origin()] = voxelBufferSize;
    }
}

template <typename LeafT>
inline void readPointDataDescriptors(const std::vector<LeafT*>& leaves,
    std::istream& is)
{
    points::AttributeSet::Descriptor::Ptr sharedDescriptor;
    for (auto* leaf : leaves) {
        points::AttributeSet::UniquePtr attrSet = leaf->stealAttributeSet();
        if (sharedDescriptor) {
            // Reuse shared descriptor from first leaf
            attrSet->resetDescriptor(sharedDescriptor, /*allowMismatchingDescriptors=*/true);
        }
        else {
            uint8_t header;
            is.read(reinterpret_cast<char*>(&header), sizeof(uint8_t));
            attrSet->readDescriptor(is);
            if (header & uint8_t(1)) {
                // Store descriptor for subsequent leaves
                sharedDescriptor = attrSet->descriptorPtr();
            }
            // a forwards-compatibility mechanism for future use,
            // if a 0x2 bit is set, read and skip over a specific number of bytes
            if (header & uint8_t(2)) {
                uint64_t bytesToSkip;
                is.read(reinterpret_cast<char*>(&bytesToSkip), sizeof(uint64_t));
                if (bytesToSkip > uint64_t(0)) {
                    std::vector<uint8_t> tempData(bytesToSkip);
                    is.read(reinterpret_cast<char*>(&tempData[0]), bytesToSkip);
                }
            }
            // this reader is only able to read headers with 0x1 and 0x2 bits set
            if (header > uint8_t(3)) {
                OPENVDB_THROW(IoError, "Unrecognised header flags in PointDataLeafNode");
            }
        }
        attrSet->readMetadata(is);
        leaf->replaceAttributeSet(attrSet.release(), /*allowMismatchingDescriptors=*/true);
    }
}

template <typename LeafT>
inline void readPointDataAttributeSizes(const std::vector<LeafT*>& leaves,
    std::istream& is, Index attributeIndex,
    std::map<Index, compression::PagedInputStream::Ptr>& pagedStreams)
{
    for (auto* leaf : leaves) {
        points::AttributeArray* array = attributeIndex < leaf->attributeSet().size() ?
            &leaf->attributeArray(attributeIndex) : nullptr;
        if (array) {
            auto* pagedStream = getOrCreatePagedStream(pagedStreams, attributeIndex);
            pagedStream->setInputStream(is);
            pagedStream->setSizeOnly(true);
            array->readPagedBuffers(*pagedStream);
        }
    }
}

template <typename LeafT>
inline void readPointDataVoxelData(const std::vector<LeafT*>& leaves,
    std::istream& is, bool saveFloatAsHalf,
    const typename LeafT::ValueType& background,
    const std::map<Coord, uint16_t>& voxelBufferSizes)
{
    using BaseLeaf = typename LeafT::BaseLeaf;
    for (auto* leaf : leaves) {
        auto it = voxelBufferSizes.find(leaf->origin());
        OPENVDB_ASSERT(it != voxelBufferSizes.end());
        BaseLeaf& baseLeaf = static_cast<BaseLeaf&>(*leaf);
        readScalarLeafBuffers(baseLeaf, is, saveFloatAsHalf, background);
    }
}

template <typename LeafT>
inline void readPointDataAttributeData(const std::vector<LeafT*>& leaves,
    std::istream& is, Index attributeIndex,
    std::map<Index, compression::PagedInputStream::Ptr>& pagedStreams)
{
    for (auto* leaf : leaves) {
        points::AttributeArray* array = attributeIndex < leaf->attributeSet().size() ?
            &leaf->attributeArray(attributeIndex) : nullptr;
        if (array) {
            auto* pagedStream = getOrCreatePagedStream(pagedStreams, attributeIndex);
            pagedStream->setInputStream(is);
            pagedStream->setSizeOnly(false);
            array->readPagedBuffers(*pagedStream);
        }
    }
}

template <typename LeafT>
inline void skipPointDataAttributeSizes(const std::vector<LeafT*>& leaves,
    std::istream& is, Index attributeIndex,
    std::map<Index, compression::PagedInputStream::Ptr>& pagedStreams)
{
    for (auto* leaf : leaves) {
        points::AttributeArray* array = attributeIndex < leaf->attributeSet().size() ?
            &leaf->attributeArray(attributeIndex) : nullptr;
        if (array) {
            auto* pagedStream = getOrCreatePagedStream(pagedStreams, attributeIndex);
            pagedStream->setInputStream(is);
            pagedStream->setSizeOnly(true);
            array->skipPagedBuffers(*pagedStream);
        }
    }
}

template <typename LeafT>
inline void skipPointDataAttributeData(const std::vector<LeafT*>& leaves,
    std::istream& is, Index attributeIndex,
    std::map<Index, compression::PagedInputStream::Ptr>& pagedStreams)
{
    for (auto* leaf : leaves) {
        points::AttributeArray* array = attributeIndex < leaf->attributeSet().size() ?
            &leaf->attributeArray(attributeIndex) : nullptr;
        if (array) {
            auto* pagedStream = getOrCreatePagedStream(pagedStreams, attributeIndex);
            pagedStream->setInputStream(is);
            pagedStream->setSizeOnly(false);
            array->skipPagedBuffers(*pagedStream);
        }
    }
}

////////////////////////////////////////
// Write-side functions

template <typename LeafT>
inline Index countPointDataPasses(const std::vector<const LeafT*>& leaves)
{
    Index maxRequiredPasses = 0;
    for (const auto* leaf : leaves) {
        const Index requiredPasses = leaf->buffers();
        if (requiredPasses > maxRequiredPasses) {
            maxRequiredPasses = requiredPasses;
        }
    }
    return maxRequiredPasses;
}

template <typename LeafT>
inline void writePointDataVoxelSizes(const std::vector<const LeafT*>& leaves,
    std::ostream& os, bool& matching,
    points::AttributeSet::Descriptor::Ptr& sharedDescriptor)
{
    bool descriptorChecked = false;
    matching = true;
    for (const auto* leaf : leaves) {
        io::writeCompressedValuesSize(os, leaf->buffer().data(), LeafT::SIZE);

        // Track descriptor matching
        const auto& descriptor = leaf->attributeSet().descriptorPtr();
        if (!descriptorChecked) {
            // First leaf - store descriptor
            descriptorChecked = true;
            sharedDescriptor = descriptor;
        }
        else if (matching && *sharedDescriptor != *descriptor) {
            matching = false;
        }
    }
}

template <typename LeafT>
inline void writePointDataDescriptors(const std::vector<const LeafT*>& leaves,
    std::ostream& os, bool matching,
    const points::AttributeSet::Descriptor::Ptr&)
{
    bool firstWrite = true;
    for (const auto* leaf : leaves) {
        const points::AttributeSet& attributeSet = leaf->attributeSet();
        if (matching) {
            // Shared descriptor - only write on first leaf
            if (firstWrite) {
                firstWrite = false;
                uint8_t header(1);
                os.write(reinterpret_cast<const char*>(&header), sizeof(uint8_t));
                attributeSet.writeDescriptor(os, /*transient=*/false);
            }
        }
        else {
            // Non-shared descriptor - write on every leaf
            uint8_t header(0);
            os.write(reinterpret_cast<const char*>(&header), sizeof(uint8_t));
            attributeSet.writeDescriptor(os, /*transient=*/false);
        }
        attributeSet.writeMetadata(os, /*transient=*/false, /*paged=*/true);
    }
}

template <typename LeafT>
inline void writePointDataAttributeSizes(const std::vector<const LeafT*>& leaves,
    std::ostream& os, Index attributeIndex)
{
    std::map<Index, compression::PagedOutputStream::Ptr> pagedStreams;
    for (const auto* leaf : leaves) {
        const points::AttributeArray* array = attributeIndex < leaf->attributeSet().size() ?
            leaf->attributeSet().getConst(attributeIndex) : nullptr;
        if (array) {
            auto* pagedStream = getOrCreatePagedStream(pagedStreams, attributeIndex);
            pagedStream->setOutputStream(os);
            pagedStream->setSizeOnly(true);
            array->writePagedBuffers(*pagedStream, /*outputTransient*/false);
        }
    }
    // Flush paged streams to write any remaining buffered page headers
    for (auto& pair : pagedStreams) {
        pair.second->flush();
    }
}

template <typename LeafT>
inline void writePointDataVoxelData(const std::vector<const LeafT*>& leaves,
    std::ostream& os, bool saveFloatAsHalf)
{
    using BaseLeaf = typename LeafT::BaseLeaf;
    for (const auto* leaf : leaves) {
        const BaseLeaf& baseLeaf = static_cast<const BaseLeaf&>(*leaf);
        writeScalarLeafBuffers(baseLeaf, os, saveFloatAsHalf);
    }
}

template <typename LeafT>
inline void writePointDataAttributeData(const std::vector<const LeafT*>& leaves,
    std::ostream& os, Index attributeIndex)
{
    std::map<Index, compression::PagedOutputStream::Ptr> pagedStreams;
    for (const auto* leaf : leaves) {
        const points::AttributeArray* array = attributeIndex < leaf->attributeSet().size() ?
            leaf->attributeSet().getConst(attributeIndex) : nullptr;
        if (array) {
            auto* pagedStream = getOrCreatePagedStream(pagedStreams, attributeIndex);
            pagedStream->setOutputStream(os);
            pagedStream->setSizeOnly(false);
            array->writePagedBuffers(*pagedStream, /*outputTransient*/false);
        }
    }
    // Flush paged streams to write any remaining buffered page data
    for (auto& pair : pagedStreams) {
        pair.second->flush();
    }
}

} // namespace internal

/// Per-grid-type codec-specific options for PointDataCodec
/// Contains point attribute filtering options
struct OPENVDB_API PointDataCodecTypeData : public io::ReadTypedOptions
{
    // Point Attribute Options - which attributes to read
    std::vector<std::string> pointAttributeNames;
}; // struct PointDataCodecTypeData

template <typename GridT>
struct OPENVDB_API PointDataCodec : public TopologyCodec<GridT>
{
    using Ptr = std::unique_ptr<PointDataCodec<GridT>>;

    ~PointDataCodec() noexcept = default;

    static inline std::string name() { return GridT::gridType(); }

    void readBuffers(std::istream& is, io::CodecData& data, const io::ReadOptions& options, io::ReadDiagnostics&) final
    {
        GridT& grid = static_cast<GridT&>(*data.grid);

        std::vector<std::string> pointAttributeNames;

        // Look up point-specific options if provided
        auto it = options.typeData.find(name());
        if (it != options.typeData.end()) {
            auto& pointTypeData = io::ReadTypedOptions::cast<PointDataCodecTypeData>(it->second);
            pointAttributeNames = pointTypeData.pointAttributeNames;
        }

        io::checkFormatVersion(is);

        bool saveFloatAsHalf = grid.saveFloatAsHalf();

        auto& tree = grid.tree();
        tree.clearAllAccessors();

        uint16_t numPasses = 1;
        is.read(reinterpret_cast<char*>(&numPasses), sizeof(uint16_t));
        const Index attributes = (numPasses - 4) / 2;

        using LeafT = typename GridT::TreeType::LeafNodeType;
        std::vector<LeafT*> leaves;
        tree.getNodes(leaves);

        // Pass 0: read voxel data sizes
        std::map<Coord, uint16_t> voxelBufferSizes;
        internal::readPointDataVoxelSizes(leaves, is, voxelBufferSizes);

        // Pass 1: read descriptor and attribute metadata
        internal::readPointDataDescriptors(leaves, is);

        // Build set of attribute indices to skip based on pointAttributeNames.
        // An empty pointAttributeNames means no filtering (read all attributes).
        std::set<Index> skipIndices;
        if (!pointAttributeNames.empty() && !leaves.empty()) {
            const auto& nameMap = leaves[0]->attributeSet().descriptor().map();
            const std::set<std::string> wantedNames(
                pointAttributeNames.begin(),
                pointAttributeNames.end());
            for (const auto& namePos : nameMap) {
                if (wantedNames.find(namePos.first) == wantedNames.end()) {
                    skipIndices.insert(static_cast<Index>(namePos.second));
                }
            }
        }

        // Passes 2..N+1: read attribute buffer sizes
        std::map<Index, compression::PagedInputStream::Ptr> pagedStreams;
        for (Index i = 0; i < attributes; ++i) {
            if (skipIndices.count(i)) {
                internal::skipPointDataAttributeSizes(leaves, is, i, pagedStreams);
            } else {
                internal::readPointDataAttributeSizes(leaves, is, i, pagedStreams);
            }
        }

        // Pass N+2: read voxel data
        internal::readPointDataVoxelData(leaves, is, saveFloatAsHalf,
            tree.background(), voxelBufferSizes);

        // Passes N+3..2N+2: read attribute data buffers
        for (Index i = 0; i < attributes; ++i) {
            if (skipIndices.count(i)) {
                internal::skipPointDataAttributeData(leaves, is, i, pagedStreams);
            } else {
                internal::readPointDataAttributeData(leaves, is, i, pagedStreams);
            }
        }

        // Drop skipped attributes from each leaf's AttributeSet
        if (!skipIndices.empty() && !leaves.empty()) {
            const std::vector<size_t> dropPositions(skipIndices.begin(), skipIndices.end());
            auto filteredDescriptor =
                leaves[0]->attributeSet().descriptorPtr()->duplicateDrop(dropPositions);
            for (auto* leaf : leaves) {
                leaf->dropAttributes(dropPositions,
                    leaf->attributeSet().descriptor(), filteredDescriptor);
            }
        }

        // PointDataGrid uses multiple passes, so clip after reading
        // the buffers if bbox is not infinite.
        if (options.clipBBox.isSorted()) {
            CoordBBox indexBBox =
                grid.constTransform().worldToIndexNodeCentered(options.clipBBox);
            grid.tree().root().clip(indexBBox);
        }
    }

    void writeBuffers(std::ostream& os, const GridBase& gridBase, const io::WriteOptions&) final
    {
        const GridT& grid = static_cast<const GridT&>(gridBase);
        bool saveFloatAsHalf = grid.saveFloatAsHalf();

        using LeafT = typename GridT::TreeType::LeafNodeType;
        std::vector<const LeafT*> leaves;
        grid.tree().getNodes(leaves);

        // Determine how many leaf buffer passes are required for this grid
        uint16_t numPasses =
            static_cast<uint16_t>(internal::countPointDataPasses(leaves));
        os.write(reinterpret_cast<const char*>(&numPasses), sizeof(uint16_t));

        const Index attributes = (numPasses - 4) / 2;

        // Pass 0: write voxel data sizes + descriptor tracking
        bool matching = true;
        points::AttributeSet::Descriptor::Ptr sharedDescriptor;
        internal::writePointDataVoxelSizes(leaves, os, matching, sharedDescriptor);

        // Pass 1: write descriptor and attribute metadata
        internal::writePointDataDescriptors(leaves, os, matching, sharedDescriptor);

        // Passes 2..N+1: write attribute buffer sizes (page headers)
        for (Index i = 0; i < attributes; ++i) {
            internal::writePointDataAttributeSizes(leaves, os, i);
        }

        // Pass N+2: write voxel data
        internal::writePointDataVoxelData(leaves, os, saveFloatAsHalf);

        // Passes N+3..2N+2: write attribute data buffers (page data)
        for (Index i = 0; i < attributes; ++i) {
            internal::writePointDataAttributeData(leaves, os, i);
        }
    }
}; // struct PointDataCodec

} // namespace codecs
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_CODECS_POINTDATACODEC_HAS_BEEN_INCLUDED
