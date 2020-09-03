// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "PointDataGrid.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

////////////////////////////////////////

// PointDataLeafNode implementation


namespace points {

template<typename T, Index Log2Dim>
inline AttributeSet::UniquePtr
PointDataLeafNode<T, Log2Dim>::stealAttributeSet()
{
    AttributeSet::UniquePtr ptr = std::make_unique<AttributeSet>();
    std::swap(ptr, mAttributeSet);
    return ptr;
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::initializeAttributes(const Descriptor::Ptr& descriptor, const Index arrayLength,
    const AttributeArray::ScopedRegistryLock* lock)
{
    if (descriptor->size() != 1 ||
        descriptor->find("P") == AttributeSet::INVALID_POS ||
        descriptor->valueType(0) != typeNameAsString<Vec3f>())
    {
        OPENVDB_THROW(IndexError, "Initializing attributes only allowed with one Vec3f position attribute.");
    }

    mAttributeSet.reset(new AttributeSet(descriptor, arrayLength, lock));
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::clearAttributes(const bool updateValueMask,
    const AttributeArray::ScopedRegistryLock* lock)
{
    mAttributeSet.reset(new AttributeSet(*mAttributeSet, 0, lock));

    // zero voxel values

    this->buffer().fill(ValueType(0));

    // if updateValueMask, also de-activate all voxels

    if (updateValueMask)    this->setValuesOff();
}

template<typename T, Index Log2Dim>
inline AttributeArray::Ptr
PointDataLeafNode<T, Log2Dim>::appendAttribute( const Descriptor& expected, Descriptor::Ptr& replacement,
                                                const size_t pos, const Index strideOrTotalSize,
                                                const bool constantStride,
                                                const Metadata* metadata,
                                                const AttributeArray::ScopedRegistryLock* lock)
{
    return mAttributeSet->appendAttribute(
        expected, replacement, pos, strideOrTotalSize, constantStride, metadata, lock);
}

// deprecated
template<typename T, Index Log2Dim>
inline AttributeArray::Ptr
PointDataLeafNode<T, Log2Dim>::appendAttribute( const Descriptor& expected, Descriptor::Ptr& replacement,
                                                const size_t pos, const Index strideOrTotalSize,
                                                const bool constantStride,
                                                const AttributeArray::ScopedRegistryLock* lock)
{
    return this->appendAttribute(expected, replacement, pos,
        strideOrTotalSize, constantStride, nullptr, lock);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::dropAttributes(const std::vector<size_t>& pos,
                    const Descriptor& expected, Descriptor::Ptr& replacement)
{
    mAttributeSet->dropAttributes(pos, expected, replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::reorderAttributes(const Descriptor::Ptr& replacement)
{
    mAttributeSet->reorderAttributes(replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::renameAttributes(const Descriptor& expected, Descriptor::Ptr& replacement)
{
    mAttributeSet->renameAttributes(expected, replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::replaceAttributeSet(AttributeSet* attributeSet, bool allowMismatchingDescriptors)
{
    if (!attributeSet) {
        OPENVDB_THROW(ValueError, "Cannot replace with a null attribute set");
    }

    if (!allowMismatchingDescriptors && mAttributeSet->descriptor() != attributeSet->descriptor()) {
        OPENVDB_THROW(ValueError, "Attribute set descriptors are not equal.");
    }

    mAttributeSet.reset(attributeSet);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::resetDescriptor(const Descriptor::Ptr& replacement)
{
    mAttributeSet->resetDescriptor(replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::compactAttributes()
{
    for (size_t i = 0; i < mAttributeSet->size(); i++) {
        AttributeArray* array = mAttributeSet->get(i);
        array->compact();
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::setOffsets(const std::vector<ValueType>& offsets, const bool updateValueMask)
{
    if (offsets.size() != LeafNodeType::NUM_VALUES) {
        OPENVDB_THROW(ValueError, "Offset vector size doesn't match number of voxels.")
    }

    for (Index index = 0; index < offsets.size(); ++index) {
        setOffsetOnly(index, offsets[index]);
    }

    if (updateValueMask) this->updateValueMask();
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::validateOffsets() const
{
    // Ensure all of the offset values are monotonically increasing
    for (Index index = 1; index < BaseLeaf::SIZE; ++index) {
        if (this->getValue(index-1) > this->getValue(index)) {
            OPENVDB_THROW(ValueError, "Voxel offset values are not monotonically increasing");
        }
    }

    // Ensure all attribute arrays are of equal length
    for (size_t attributeIndex = 1; attributeIndex < mAttributeSet->size(); ++attributeIndex ) {
        if (mAttributeSet->getConst(attributeIndex-1)->size() != mAttributeSet->getConst(attributeIndex)->size()) {
            OPENVDB_THROW(ValueError, "Attribute arrays have inconsistent length");
        }
    }

    // Ensure the last voxel's offset value matches the size of each attribute array
    if (mAttributeSet->size() > 0 && this->getValue(BaseLeaf::SIZE-1) != mAttributeSet->getConst(0)->size()) {
        OPENVDB_THROW(ValueError, "Last voxel offset value does not match attribute array length");
    }
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::groupPointCount(const Name& groupName) const
{
    if (!this->attributeSet().descriptor().hasGroup(groupName)) {
        return Index64(0);
    }
    GroupFilter filter(groupName, this->attributeSet());
    if (filter.state() == index::ALL) {
        return this->pointCount();
    } else {
        return iterCount(this->beginIndexAll(filter));
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::updateValueMask()
{
    ValueType start = 0, end = 0;
    for (Index n = 0; n < LeafNodeType::NUM_VALUES; n++) {
        end = this->getValue(n);
        this->setValueMask(n, (end - start) > 0);
        start = end;
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readTopology(std::istream& is, bool fromHalf)
{
    BaseLeaf::readTopology(is, fromHalf);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::writeTopology(std::ostream& os, bool toHalf) const
{
    BaseLeaf::writeTopology(os, toHalf);
}

template<typename T, Index Log2Dim>
inline Index
PointDataLeafNode<T, Log2Dim>::buffers() const
{
    return Index(   /*voxel buffer sizes*/          1 +
                    /*voxel buffers*/               1 +
                    /*attribute metadata*/          1 +
                    /*attribute uniform values*/    mAttributeSet->size() +
                    /*attribute buffers*/           mAttributeSet->size() +
                    /*cleanup*/                     1);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    this->readBuffers(is, CoordBBox::inf(), fromHalf);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& /*bbox*/, bool fromHalf)
{
    struct Local
    {
        static void destroyPagedStream(const io::StreamMetadata::AuxDataMap& auxData, const Index index)
        {
            // if paged stream exists, delete it
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(it);
            }
        }

        static compression::PagedInputStream& getOrInsertPagedStream(   const io::StreamMetadata::AuxDataMap& auxData,
                                                                        const Index index)
        {
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                return *(boost::any_cast<compression::PagedInputStream::Ptr>(it->second));
            }
            else {
                compression::PagedInputStream::Ptr pagedStream = std::make_shared<compression::PagedInputStream>();
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[key] = pagedStream;
                return *pagedStream;
            }
        }

        static bool hasMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            return itMatching != auxData.end();
        }

        static void clearMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            std::string descriptorKey("descriptorPtr");
            auto itMatching = auxData.find(matchingKey);
            auto itDescriptor = auxData.find(descriptorKey);
            if (itMatching != auxData.end())    (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itMatching);
            if (itDescriptor != auxData.end())  (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itDescriptor);
        }

        static void insertDescriptor(   const io::StreamMetadata::AuxDataMap& auxData,
                                        const Descriptor::Ptr descriptor)
        {
            std::string descriptorKey("descriptorPtr");
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            if (itMatching == auxData.end()) {
                // if matching bool is not found, insert "true" and the descriptor
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[matchingKey] = true;
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[descriptorKey] = descriptor;
            }
        }

        static AttributeSet::Descriptor::Ptr retrieveMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string descriptorKey("descriptorPtr");
            auto itDescriptor = auxData.find(descriptorKey);
            assert(itDescriptor != auxData.end());
            const Descriptor::Ptr descriptor = boost::any_cast<AttributeSet::Descriptor::Ptr>(itDescriptor->second);
            return descriptor;
        }
    };

    const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(is);

    if (!meta) {
        OPENVDB_THROW(IoError, "Cannot read in a PointDataLeaf without StreamMetadata.");
    }

    const Index pass(static_cast<uint16_t>(meta->pass()));
    const Index maximumPass(static_cast<uint16_t>(meta->pass() >> 16));

    const Index attributes = (maximumPass - 4) / 2;

    if (pass == 0) {
        // pass 0 - voxel data sizes
        is.read(reinterpret_cast<char*>(&mVoxelBufferSize), sizeof(uint16_t));
        Local::clearMatchingDescriptor(meta->auxData());
    }
    else if (pass == 1) {
        // pass 1 - descriptor and attribute metadata
        if (Local::hasMatchingDescriptor(meta->auxData())) {
            AttributeSet::Descriptor::Ptr descriptor = Local::retrieveMatchingDescriptor(meta->auxData());
            mAttributeSet->resetDescriptor(descriptor, /*allowMismatchingDescriptors=*/true);
        }
        else {
            uint8_t header;
            is.read(reinterpret_cast<char*>(&header), sizeof(uint8_t));
            mAttributeSet->readDescriptor(is);
            if (header & uint8_t(1)) {
                AttributeSet::DescriptorPtr descriptor = mAttributeSet->descriptorPtr();
                Local::insertDescriptor(meta->auxData(), descriptor);
            }
            // a forwards-compatibility mechanism for future use,
            // if a 0x2 bit is set, read and skip over a specific number of bytes
            if (header & uint8_t(2)) {
                uint64_t bytesToSkip;
                is.read(reinterpret_cast<char*>(&bytesToSkip), sizeof(uint64_t));
                if (bytesToSkip > uint64_t(0)) {
                    auto metadata = io::getStreamMetadataPtr(is);
                    if (metadata && metadata->seekable()) {
                        is.seekg(bytesToSkip, std::ios_base::cur);
                    }
                    else {
                        std::vector<uint8_t> tempData(bytesToSkip);
                        is.read(reinterpret_cast<char*>(&tempData[0]), bytesToSkip);
                    }
                }
            }
            // this reader is only able to read headers with 0x1 and 0x2 bits set
            if (header > uint8_t(3)) {
                OPENVDB_THROW(IoError, "Unrecognised header flags in PointDataLeafNode");
            }
        }
        mAttributeSet->readMetadata(is);
    }
    else if (pass < (attributes + 2)) {
        // pass 2...n+2 - attribute uniform values
        const size_t attributeIndex = pass - 2;
        AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->get(attributeIndex) : nullptr;
        if (array) {
            compression::PagedInputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), static_cast<Index>(attributeIndex));
            pagedStream.setInputStream(is);
            pagedStream.setSizeOnly(true);
            array->readPagedBuffers(pagedStream);
        }
    }
    else if (pass == attributes + 2) {
        // pass n+2 - voxel data

        const Index passValue(meta->pass());

        // StreamMetadata pass variable used to temporarily store voxel buffer size
        io::StreamMetadata& nonConstMeta = const_cast<io::StreamMetadata&>(*meta);
        nonConstMeta.setPass(mVoxelBufferSize);

        // readBuffers() calls readCompressedValues specialization above
        BaseLeaf::readBuffers(is, fromHalf);

        // pass now reset to original value
        nonConstMeta.setPass(passValue);
    }
    else if (pass < (attributes*2 + 3)) {
        // pass n+2..2n+2 - attribute buffers
        const Index attributeIndex = pass - attributes - 3;
        AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->get(attributeIndex) : nullptr;
        if (array) {
            compression::PagedInputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), attributeIndex);
            pagedStream.setInputStream(is);
            pagedStream.setSizeOnly(false);
            array->readPagedBuffers(pagedStream);
        }
        // cleanup paged stream reference in auxiliary metadata
        if (pass > attributes + 3) {
            Local::destroyPagedStream(meta->auxData(), attributeIndex-1);
        }
    }
    else if (pass < buffers()) {
        // pass 2n+3 - cleanup last paged stream
        const Index attributeIndex = pass - attributes - 4;
        Local::destroyPagedStream(meta->auxData(), attributeIndex);
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    struct Local
    {
        static void destroyPagedStream(const io::StreamMetadata::AuxDataMap& auxData, const Index index)
        {
            // if paged stream exists, flush and delete it
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                compression::PagedOutputStream& stream = *(boost::any_cast<compression::PagedOutputStream::Ptr>(it->second));
                stream.flush();
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(it);
            }
        }

        static compression::PagedOutputStream& getOrInsertPagedStream(  const io::StreamMetadata::AuxDataMap& auxData,
                                                                        const Index index)
        {
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                return *(boost::any_cast<compression::PagedOutputStream::Ptr>(it->second));
            }
            else {
                compression::PagedOutputStream::Ptr pagedStream = std::make_shared<compression::PagedOutputStream>();
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[key] = pagedStream;
                return *pagedStream;
            }
        }

        static void insertDescriptor(   const io::StreamMetadata::AuxDataMap& auxData,
                                        const Descriptor::Ptr descriptor)
        {
            std::string descriptorKey("descriptorPtr");
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            auto itDescriptor = auxData.find(descriptorKey);
            if (itMatching == auxData.end()) {
                // if matching bool is not found, insert "true" and the descriptor
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[matchingKey] = true;
                assert(itDescriptor == auxData.end());
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[descriptorKey] = descriptor;
            }
            else {
                // if matching bool is found and is false, early exit (a previous descriptor did not match)
                bool matching = boost::any_cast<bool>(itMatching->second);
                if (!matching)    return;
                assert(itDescriptor != auxData.end());
                // if matching bool is true, check whether the existing descriptor matches the current one and set
                // matching bool to false if not
                const Descriptor::Ptr existingDescriptor = boost::any_cast<AttributeSet::Descriptor::Ptr>(itDescriptor->second);
                if (*existingDescriptor != *descriptor) {
                    (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[matchingKey] = false;
                }
            }
        }

        static bool hasMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            // if matching key is not found, no matching descriptor
            if (itMatching == auxData.end())                return false;
            // if matching key is found and is false, no matching descriptor
            if (!boost::any_cast<bool>(itMatching->second)) return false;
            return true;
        }

        static AttributeSet::Descriptor::Ptr retrieveMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string descriptorKey("descriptorPtr");
            auto itDescriptor = auxData.find(descriptorKey);
            // if matching key is true, however descriptor is not found, it has already been retrieved
            if (itDescriptor == auxData.end())              return nullptr;
            // otherwise remove it and return it
            const Descriptor::Ptr descriptor = boost::any_cast<AttributeSet::Descriptor::Ptr>(itDescriptor->second);
            (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itDescriptor);
            return descriptor;
        }

        static void clearMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            std::string descriptorKey("descriptorPtr");
            auto itMatching = auxData.find(matchingKey);
            auto itDescriptor = auxData.find(descriptorKey);
            if (itMatching != auxData.end())    (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itMatching);
            if (itDescriptor != auxData.end())  (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itDescriptor);
        }
    };

    const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(os);

    if (!meta) {
        OPENVDB_THROW(IoError, "Cannot write out a PointDataLeaf without StreamMetadata.");
    }

    const Index pass(static_cast<uint16_t>(meta->pass()));

    // leaf traversal analysis deduces the number of passes to perform for this leaf
    // then updates the leaf traversal value to ensure all passes will be written

    if (meta->countingPasses()) {
        const Index requiredPasses = this->buffers();
        if (requiredPasses > pass) {
            meta->setPass(requiredPasses);
        }
        return;
    }

    const Index maximumPass(static_cast<uint16_t>(meta->pass() >> 16));
    const Index attributes = (maximumPass - 4) / 2;

    if (pass == 0) {
        // pass 0 - voxel data sizes
        io::writeCompressedValuesSize(os, this->buffer().data(), SIZE);
        // track if descriptor is shared or not
        Local::insertDescriptor(meta->auxData(), mAttributeSet->descriptorPtr());
    }
    else if (pass == 1) {
        // pass 1 - descriptor and attribute metadata
        bool matchingDescriptor = Local::hasMatchingDescriptor(meta->auxData());
        if (matchingDescriptor) {
            AttributeSet::Descriptor::Ptr descriptor = Local::retrieveMatchingDescriptor(meta->auxData());
            if (descriptor) {
                // write a header to indicate a shared descriptor
                uint8_t header(1);
                os.write(reinterpret_cast<const char*>(&header), sizeof(uint8_t));
                mAttributeSet->writeDescriptor(os, /*transient=*/false);
            }
        }
        else {
            // write a header to indicate a non-shared descriptor
            uint8_t header(0);
            os.write(reinterpret_cast<const char*>(&header), sizeof(uint8_t));
            mAttributeSet->writeDescriptor(os, /*transient=*/false);
        }
        mAttributeSet->writeMetadata(os, /*transient=*/false, /*paged=*/true);
    }
    else if (pass < attributes + 2) {
        // pass 2...n+2 - attribute buffer sizes
        const Index attributeIndex = pass - 2;
        // destroy previous paged stream
        if (pass > 2) {
            Local::destroyPagedStream(meta->auxData(), attributeIndex-1);
        }
        const AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->getConst(attributeIndex) : nullptr;
        if (array) {
            compression::PagedOutputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), attributeIndex);
            pagedStream.setOutputStream(os);
            pagedStream.setSizeOnly(true);
            array->writePagedBuffers(pagedStream, /*outputTransient*/false);
        }
    }
    else if (pass == attributes + 2) {
        const Index attributeIndex = pass - 3;
        Local::destroyPagedStream(meta->auxData(), attributeIndex);
        // pass n+2 - voxel data
        BaseLeaf::writeBuffers(os, toHalf);
    }
    else if (pass < (attributes*2 + 3)) {
        // pass n+3...2n+3 - attribute buffers
        const Index attributeIndex = pass - attributes - 3;
        // destroy previous paged stream
        if (pass > attributes + 2) {
            Local::destroyPagedStream(meta->auxData(), attributeIndex-1);
        }
        const AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->getConst(attributeIndex) : nullptr;
        if (array) {
            compression::PagedOutputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), attributeIndex);
            pagedStream.setOutputStream(os);
            pagedStream.setSizeOnly(false);
            array->writePagedBuffers(pagedStream, /*outputTransient*/false);
        }
    }
    else if (pass < buffers()) {
        Local::clearMatchingDescriptor(meta->auxData());
        // pass 2n+3 - cleanup last paged stream
        const Index attributeIndex = pass - attributes - 4;
        Local::destroyPagedStream(meta->auxData(), attributeIndex);
    }
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::memUsage() const
{
    return BaseLeaf::memUsage() + mAttributeSet->memUsage();
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    BaseLeaf::evalActiveBoundingBox(bbox, visitVoxels);
}

template<typename T, Index Log2Dim>
inline CoordBBox
PointDataLeafNode<T, Log2Dim>::getNodeBoundingBox() const
{
    return BaseLeaf::getNodeBoundingBox();
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    if (!this->allocate()) return;

    this->assertNonModifiableUnlessZero(value);

    // active state is permitted to be updated

    for (Int32 x = bbox.min().x(); x <= bbox.max().x(); ++x) {
        const Index offsetX = (x & (DIM-1u)) << 2*Log2Dim;
        for (Int32 y = bbox.min().y(); y <= bbox.max().y(); ++y) {
            const Index offsetXY = offsetX + ((y & (DIM-1u)) << Log2Dim);
            for (Int32 z = bbox.min().z(); z <= bbox.max().z(); ++z) {
                const Index offset = offsetXY + (z & (DIM-1u));
                this->setValueMask(offset, active);
            }
        }
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::fill(const ValueType& value, bool active)
{
    this->assertNonModifiableUnlessZero(value);

    // active state is permitted to be updated

    if (active)                 this->setValuesOn();
    else                        this->setValuesOff();
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
