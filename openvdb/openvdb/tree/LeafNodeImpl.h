// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/io/Compression.h> // for io::readCompressedValues(), etc.

#include "LeafNode.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {


template<typename T, Index Log2Dim>
inline std::string
LeafNode<T, Log2Dim>::str() const
{
    OPENVDB_THROW(NotImplementedError, "LeafNode::str() is no longer implemented.");
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::clip(const CoordBBox& clipBBox, const T& background)
{
    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.  Fill it with the background.
        this->fill(background, /*active=*/false);
    } else if (clipBBox.isInside(nodeBBox)) {
        // This node lies completely inside the clipping region.  Leave it intact.
        return;
    }

    // This node isn't completely contained inside the clipping region.
    // Set any voxels that lie outside the region to the background value.

    // Construct a boolean mask that is on inside the clipping region and off outside it.
    NodeMaskType mask;
    nodeBBox.intersect(clipBBox);
    Coord xyz;
    int &x = xyz.x(), &y = xyz.y(), &z = xyz.z();
    for (x = nodeBBox.min().x(); x <= nodeBBox.max().x(); ++x) {
        for (y = nodeBBox.min().y(); y <= nodeBBox.max().y(); ++y) {
            for (z = nodeBBox.min().z(); z <= nodeBBox.max().z(); ++z) {
                mask.setOn(static_cast<Index32>(this->coordToOffset(xyz)));
            }
        }
    }

    // Set voxels that lie in the inactive region of the mask (i.e., outside
    // the clipping region) to the background value.
    for (MaskOffIterator maskIter = mask.beginOff(); maskIter; ++maskIter) {
        this->setValueOff(maskIter.pos(), background);
    }
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    if (!this->allocate()) return;

    auto clippedBBox = this->getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    if (!clippedBBox) return;

    for (Int32 x = clippedBBox.min().x(); x <= clippedBBox.max().x(); ++x) {
        const Index offsetX = (x & (DIM-1u)) << 2*Log2Dim;
        for (Int32 y = clippedBBox.min().y(); y <= clippedBBox.max().y(); ++y) {
            const Index offsetXY = offsetX + ((y & (DIM-1u)) << Log2Dim);
            for (Int32 z = clippedBBox.min().z(); z <= clippedBBox.max().z(); ++z) {
                const Index offset = offsetXY + (z & (DIM-1u));
                mBuffer[offset] = value;
                mValueMask.set(offset, active);
            }
        }
    }
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::fill(const ValueType& value)
{
    mBuffer.fill(value);
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::fill(const ValueType& value, bool active)
{
    mBuffer.fill(value);
    mValueMask.set(active);
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::readTopology(std::istream& is, bool /*fromHalf*/)
{
    mValueMask.load(is);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::writeTopology(std::ostream& os, bool /*toHalf*/) const
{
    mValueMask.save(os);
}


////////////////////////////////////////



template<typename T, Index Log2Dim>
inline void
LeafNode<T,Log2Dim>::skipCompressedValues(bool seekable, std::istream& is, bool fromHalf)
{
    if (seekable) {
        // Seek over voxel values.
        io::readCompressedValues<ValueType, NodeMaskType>(
            is, nullptr, SIZE, mValueMask, fromHalf);
    } else {
        // Read and discard voxel values.
        Buffer temp;
        io::readCompressedValues(is, temp.mData, SIZE, mValueMask, fromHalf);
    }
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T,Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    this->readBuffers(is, CoordBBox::inf(), fromHalf);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T,Log2Dim>::readBuffers(std::istream& is, const CoordBBox& clipBBox, bool fromHalf)
{
    SharedPtr<io::StreamMetadata> meta = io::getStreamMetadataPtr(is);
    const bool seekable = meta && meta->seekable();

    std::streamoff maskpos = is.tellg();

    if (seekable) {
        // Seek over the value mask.
        mValueMask.seek(is);
    } else {
        // Read in the value mask.
        mValueMask.load(is);
    }

    int8_t numBuffers = 1;
    if (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
        // Read in the origin.
        is.read(reinterpret_cast<char*>(&mOrigin), sizeof(Coord::ValueType) * 3);

        // Read in the number of buffers, which should now always be one.
        is.read(reinterpret_cast<char*>(&numBuffers), sizeof(int8_t));
    }

    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.
        skipCompressedValues(seekable, is, fromHalf);
        mValueMask.setOff();
        mBuffer.setOutOfCore(false);
    } else {
        // If this node lies completely inside the clipping region and it is being read
        // from a memory-mapped file, delay loading of its buffer until the buffer
        // is actually accessed.  (If this node requires clipping, its buffer
        // must be accessed and therefore must be loaded.)
        io::MappedFile::Ptr mappedFile = io::getMappedFilePtr(is);
        const bool delayLoad = ((mappedFile.get() != nullptr) && clipBBox.isInside(nodeBBox));

        if (delayLoad) {
            mBuffer.setOutOfCore(true);
            mBuffer.mFileInfo = new typename Buffer::FileInfo;
            mBuffer.mFileInfo->meta = meta;
            mBuffer.mFileInfo->bufpos = is.tellg();
            mBuffer.mFileInfo->mapping = mappedFile;
            // Save the offset to the value mask, because the in-memory copy
            // might change before the value buffer gets read.
            mBuffer.mFileInfo->maskpos = maskpos;
            // Skip over voxel values.
            skipCompressedValues(seekable, is, fromHalf);
        } else {
            mBuffer.allocate();
            io::readCompressedValues(is, mBuffer.mData, SIZE, mValueMask, fromHalf);
            mBuffer.setOutOfCore(false);

            // Get this tree's background value.
            T background = zeroVal<T>();
            if (const void* bgPtr = io::getGridBackgroundValuePtr(is)) {
                background = *static_cast<const T*>(bgPtr);
            }
            this->clip(clipBBox, background);
        }
    }

    if (numBuffers > 1) {
        // Read in and discard auxiliary buffers that were created with earlier
        // versions of the library.  (Auxiliary buffers are not mask compressed.)
        const bool zipped = io::getDataCompression(is) & io::COMPRESS_ZIP;
        Buffer temp;
        for (int i = 1; i < numBuffers; ++i) {
            if (fromHalf) {
                io::HalfReader<io::RealToHalf<T>::isReal, T>::read(is, temp.mData, SIZE, zipped);
            } else {
                io::readData<T>(is, temp.mData, SIZE, zipped);
            }
        }
    }

    // increment the leaf number
    if (meta)   meta->setLeaf(meta->leaf() + 1);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    // Write out the value mask.
    mValueMask.save(os);

    mBuffer.loadValues();

    io::writeCompressedValues(os, mBuffer.mData, SIZE,
        mValueMask, /*childMask=*/NodeMaskType(), toHalf);
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline Index64
LeafNode<T, Log2Dim>::memUsage() const
{
    // Use sizeof(*this) to capture alignment-related padding
    // (but note that sizeof(*this) includes sizeof(mBuffer)).
    return sizeof(*this) + mBuffer.memUsage() - sizeof(mBuffer);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    CoordBBox this_bbox = this->getNodeBoundingBox();
    if (bbox.isInside(this_bbox)) return;//this LeafNode is already enclosed in the bbox
    if (ValueOnCIter iter = this->cbeginValueOn()) {//any active values?
        if (visitVoxels) {//use voxel granularity?
            this_bbox.reset();
            for(; iter; ++iter) this_bbox.expand(this->offsetToLocalCoord(iter.pos()));
            this_bbox.translate(this->origin());
        }
        bbox.expand(this_bbox);
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline bool
LeafNode<T, Log2Dim>::isConstant(ValueType& firstValue,
                                 bool& state,
                                 const ValueType& tolerance) const
{
    if (!mValueMask.isConstant(state)) return false;// early termination
    firstValue = mBuffer[0];
    for (Index i = 1; i < SIZE; ++i) {
        if ( !math::isApproxEqual(mBuffer[i], firstValue, tolerance) ) return false;// early termination
    }
    return true;
}

template<typename T, Index Log2Dim>
inline bool
LeafNode<T, Log2Dim>::isConstant(ValueType& minValue,
                                 ValueType& maxValue,
                                 bool& state,
                                 const ValueType& tolerance) const
{
    if (!mValueMask.isConstant(state)) return false;// early termination
    minValue = maxValue = mBuffer[0];
    for (Index i = 1; i < SIZE; ++i) {
        const T& v = mBuffer[i];
        if (v < minValue) {
            if ((maxValue + math::negative(v)) > tolerance) return false;// early termination
            minValue = v;
        } else if (v > maxValue) {
            if ((v + math::negative(minValue)) > tolerance) return false;// early termination
            maxValue = v;
        }
    }
    return true;
}

template<typename T, Index Log2Dim>
inline T
LeafNode<T, Log2Dim>::medianAll(T *tmp) const
{
    std::unique_ptr<T[]> data(nullptr);
    if (tmp == nullptr) {//allocate temporary storage
        data.reset(new T[NUM_VALUES]);
        tmp = data.get();
    }
    if (tmp != mBuffer.data()) {
        const T* src = mBuffer.data();
        for (T* dst = tmp; dst-tmp < NUM_VALUES;) *dst++ = *src++;
    }
    static const size_t midpoint = (NUM_VALUES - 1) >> 1;
    std::nth_element(tmp, tmp + midpoint, tmp + NUM_VALUES);
    return tmp[midpoint];
}

template<typename T, Index Log2Dim>
inline Index
LeafNode<T, Log2Dim>::medianOn(T &value, T *tmp) const
{
    const Index count = mValueMask.countOn();
    if (count == NUM_VALUES) {//special case: all voxels are active
        value = this->medianAll(tmp);
        return NUM_VALUES;
    } else if (count == 0) {
        return 0;
    }
    std::unique_ptr<T[]> data(nullptr);
    if (tmp == nullptr) {//allocate temporary storage
        data.reset(new T[count]);// 0 < count < NUM_VALUES
        tmp = data.get();
    }
    for (auto iter=this->cbeginValueOn(); iter; ++iter) *tmp++ = *iter;
    T *begin = tmp - count;
    const size_t midpoint = (count - 1) >> 1;
    std::nth_element(begin, begin + midpoint, tmp);
    value = begin[midpoint];
    return count;
}

template<typename T, Index Log2Dim>
inline Index
LeafNode<T, Log2Dim>::medianOff(T &value, T *tmp) const
{
    const Index count = mValueMask.countOff();
    if (count == NUM_VALUES) {//special case: all voxels are inactive
        value = this->medianAll(tmp);
        return NUM_VALUES;
    } else if (count == 0) {
        return 0;
    }
    std::unique_ptr<T[]> data(nullptr);
    if (tmp == nullptr) {//allocate temporary storage
        data.reset(new T[count]);// 0 < count < NUM_VALUES
        tmp = data.get();
    }
    for (auto iter=this->cbeginValueOff(); iter; ++iter) *tmp++ = *iter;
    T *begin = tmp - count;
    const size_t midpoint = (count - 1) >> 1;
    std::nth_element(begin, begin + midpoint, tmp);
    value = begin[midpoint];
    return count;
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::resetBackground(const ValueType& oldBackground,
                                      const ValueType& newBackground)
{
    if (!this->allocate()) return;

    typename NodeMaskType::OffIterator iter;
    // For all inactive values...
    for (iter = this->mValueMask.beginOff(); iter; ++iter) {
        ValueType &inactiveValue = mBuffer[iter.pos()];
        if (math::isApproxEqual(inactiveValue, oldBackground)) {
            inactiveValue = newBackground;
        } else if (math::isApproxEqual(inactiveValue, math::negative(oldBackground))) {
            inactiveValue = math::negative(newBackground);
        }
    }
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::negate()
{
    if (!this->allocate()) return;

    for (Index i = 0; i < SIZE; ++i) {
        mBuffer[i] = math::negative(mBuffer[i]);
    }
}


////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

// Specialization for LeafNodes of type bool


template<Index Log2Dim>
inline Index64
LeafNode<bool, Log2Dim>::memUsage() const
{
    // Use sizeof(*this) to capture alignment-related padding
    return sizeof(*this);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    CoordBBox this_bbox = this->getNodeBoundingBox();
    if (bbox.isInside(this_bbox)) return;//this LeafNode is already enclosed in the bbox
    if (ValueOnCIter iter = this->cbeginValueOn()) {//any active values?
        if (visitVoxels) {//use voxel granularity?
            this_bbox.reset();
            for(; iter; ++iter) this_bbox.expand(this->offsetToLocalCoord(iter.pos()));
            this_bbox.translate(this->origin());
        }
        bbox.expand(this_bbox);
    }
}


////////////////////////////////////////


template<Index Log2Dim>
inline std::string
LeafNode<bool, Log2Dim>::str() const
{
    std::ostringstream ostr;
    ostr << "LeafNode @" << mOrigin << ": ";
    for (Index32 n = 0; n < SIZE; ++n) ostr << (mValueMask.isOn(n) ? '#' : '.');
    return ostr.str();
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::readTopology(std::istream& is, bool /*fromHalf*/)
{
    mValueMask.load(is);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::writeTopology(std::ostream& os, bool /*toHalf*/) const
{
    mValueMask.save(os);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& clipBBox, bool fromHalf)
{
    // Boolean LeafNodes don't currently implement lazy loading.
    // Instead, load the full buffer, then clip it.

    this->readBuffers(is, fromHalf);

    // Get this tree's background value.
    bool background = false;
    if (const void* bgPtr = io::getGridBackgroundValuePtr(is)) {
        background = *static_cast<const bool*>(bgPtr);
    }
    this->clip(clipBBox, background);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::readBuffers(std::istream& is, bool /*fromHalf*/)
{
    // Read in the value mask.
    mValueMask.load(is);
    // Read in the origin.
    is.read(reinterpret_cast<char*>(&mOrigin), sizeof(Coord::ValueType) * 3);

    if (io::getFormatVersion(is) >= OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION) {
        // Read in the mask for the voxel values.
        mBuffer.mData.load(is);
    } else {
        // Older files stored one or more bool arrays.

        // Read in the number of buffers, which should now always be one.
        int8_t numBuffers = 0;
        is.read(reinterpret_cast<char*>(&numBuffers), sizeof(int8_t));

        // Read in the buffer.
        // (Note: prior to the bool leaf optimization, buffers were always compressed.)
        std::unique_ptr<bool[]> buf{new bool[SIZE]};
        io::readData<bool>(is, buf.get(), SIZE, /*isCompressed=*/true);

        // Transfer values to mBuffer.
        mBuffer.mData.setOff();
        for (Index i = 0; i < SIZE; ++i) {
            if (buf[i]) mBuffer.mData.setOn(i);
        }

        if (numBuffers > 1) {
            // Read in and discard auxiliary buffers that were created with
            // earlier versions of the library.
            for (int i = 1; i < numBuffers; ++i) {
                io::readData<bool>(is, buf.get(), SIZE, /*isCompressed=*/true);
            }
        }
    }
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::writeBuffers(std::ostream& os, bool /*toHalf*/) const
{
    // Write out the value mask.
    mValueMask.save(os);
    // Write out the origin.
    os.write(reinterpret_cast<const char*>(&mOrigin), sizeof(Coord::ValueType) * 3);
    // Write out the voxel values.
    mBuffer.mData.save(os);
}


////////////////////////////////////////


template<Index Log2Dim>
inline bool
LeafNode<bool, Log2Dim>::isConstant(bool& constValue, bool& state, bool tolerance) const
{
    if (!mValueMask.isConstant(state)) return false;

    // Note: if tolerance is true (i.e., 1), then all boolean values compare equal.
    if (!tolerance && !(mBuffer.mData.isOn() || mBuffer.mData.isOff())) return false;

    constValue = mBuffer.mData.isOn();
    return true;
}


////////////////////////////////////////


template<Index Log2Dim>
inline bool
LeafNode<bool, Log2Dim>::medianAll() const
{
    const Index countTrue = mBuffer.mData.countOn();
    return countTrue > (NUM_VALUES >> 1);
}


template<Index Log2Dim>
inline Index
LeafNode<bool, Log2Dim>::medianOn(bool& state) const
{
    const NodeMaskType tmp = mBuffer.mData & mValueMask;//both true and active
    const Index countTrueOn = tmp.countOn(), countOn = mValueMask.countOn();
    state = countTrueOn > (NUM_VALUES >> 1);
    return countOn;
}


template<Index Log2Dim>
inline Index
LeafNode<bool, Log2Dim>::medianOff(bool& state) const
{
    const NodeMaskType tmp = mBuffer.mData & (!mValueMask);//both true and inactive
    const Index countTrueOff = tmp.countOn(), countOff = mValueMask.countOff();
    state = countTrueOff > (NUM_VALUES >> 1);
    return countOff;
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::resetBackground(bool oldBackground, bool newBackground)
{
    if (newBackground != oldBackground) {
        // Flip mBuffer's background bits and zero its foreground bits.
        NodeMaskType bgMask = !(mBuffer.mData | mValueMask);
        // Overwrite mBuffer's background bits, leaving its foreground bits intact.
        mBuffer.mData = (mBuffer.mData & mValueMask) | bgMask;
    }
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::clip(const CoordBBox& clipBBox, bool background)
{
    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.  Fill it with background tiles.
        this->fill(nodeBBox, background, /*active=*/false);
    } else if (clipBBox.isInside(nodeBBox)) {
        // This node lies completely inside the clipping region.  Leave it intact.
        return;
    }

    // This node isn't completely contained inside the clipping region.
    // Set any voxels that lie outside the region to the background value.

    // Construct a boolean mask that is on inside the clipping region and off outside it.
    NodeMaskType mask;
    nodeBBox.intersect(clipBBox);
    Coord xyz;
    int &x = xyz.x(), &y = xyz.y(), &z = xyz.z();
    for (x = nodeBBox.min().x(); x <= nodeBBox.max().x(); ++x) {
        for (y = nodeBBox.min().y(); y <= nodeBBox.max().y(); ++y) {
            for (z = nodeBBox.min().z(); z <= nodeBBox.max().z(); ++z) {
                mask.setOn(static_cast<Index32>(this->coordToOffset(xyz)));
            }
        }
    }

    // Set voxels that lie in the inactive region of the mask (i.e., outside
    // the clipping region) to the background value.
    for (MaskOffIter maskIter = mask.beginOff(); maskIter; ++maskIter) {
        this->setValueOff(maskIter.pos(), background);
    }
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::fill(const CoordBBox& bbox, bool value, bool active)
{
    auto clippedBBox = this->getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    if (!clippedBBox) return;

    for (Int32 x = clippedBBox.min().x(); x <= clippedBBox.max().x(); ++x) {
        const Index offsetX = (x & (DIM-1u))<<2*Log2Dim;
        for (Int32 y = clippedBBox.min().y(); y <= clippedBBox.max().y(); ++y) {
            const Index offsetXY = offsetX + ((y & (DIM-1u))<<  Log2Dim);
            for (Int32 z = clippedBBox.min().z(); z <= clippedBBox.max().z(); ++z) {
                const Index offset = offsetXY + (z & (DIM-1u));
                mValueMask.set(offset, active);
                mBuffer.mData.set(offset, value);
            }
        }
    }
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::fill(const bool& value)
{
    mBuffer.fill(value);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::fill(const bool& value, bool active)
{
    mBuffer.fill(value);
    mValueMask.set(active);
}


////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

// Specialization for LeafNodes with mask information only


template<Index Log2Dim>
inline Index64
LeafNode<ValueMask, Log2Dim>::memUsage() const
{
    // Use sizeof(*this) to capture alignment-related padding
    return sizeof(*this);
}


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    CoordBBox this_bbox = this->getNodeBoundingBox();
    if (bbox.isInside(this_bbox)) return;//this LeafNode is already enclosed in the bbox
    if (ValueOnCIter iter = this->cbeginValueOn()) {//any active values?
        if (visitVoxels) {//use voxel granularity?
            this_bbox.reset();
            for(; iter; ++iter) this_bbox.expand(this->offsetToLocalCoord(iter.pos()));
            this_bbox.translate(this->origin());
        }
        bbox.expand(this_bbox);
    }
}


template<Index Log2Dim>
inline std::string
LeafNode<ValueMask, Log2Dim>::str() const
{
    std::ostringstream ostr;
    ostr << "LeafNode @" << mOrigin << ": ";
    for (Index32 n = 0; n < SIZE; ++n) ostr << (mBuffer.mData.isOn(n) ? '#' : '.');
    return ostr.str();
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::readTopology(std::istream& is, bool /*fromHalf*/)
{
    mBuffer.mData.load(is);
}


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::writeTopology(std::ostream& os, bool /*toHalf*/) const
{
    mBuffer.mData.save(os);
}


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& clipBBox, bool fromHalf)
{
    // Boolean LeafNodes don't currently implement lazy loading.
    // Instead, load the full buffer, then clip it.

    this->readBuffers(is, fromHalf);

    // Get this tree's background value.
    bool background = false;
    if (const void* bgPtr = io::getGridBackgroundValuePtr(is)) {
        background = *static_cast<const bool*>(bgPtr);
    }
    this->clip(clipBBox, background);
}


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::readBuffers(std::istream& is, bool /*fromHalf*/)
{
    // Read in the value mask = buffer.
    mBuffer.mData.load(is);
    // Read in the origin.
    is.read(reinterpret_cast<char*>(&mOrigin), sizeof(Coord::ValueType) * 3);
}


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::writeBuffers(std::ostream& os, bool /*toHalf*/) const
{
    // Write out the value mask = buffer.
    mBuffer.mData.save(os);
    // Write out the origin.
    os.write(reinterpret_cast<const char*>(&mOrigin), sizeof(Coord::ValueType) * 3);
}


////////////////////////////////////////


template<Index Log2Dim>
inline bool
LeafNode<ValueMask, Log2Dim>::isConstant(bool& constValue, bool& state, bool) const
{
    if (!mBuffer.mData.isConstant(state)) return false;

    constValue = state;
    return true;
}


////////////////////////////////////////


template<Index Log2Dim>
inline bool
LeafNode<ValueMask, Log2Dim>::medianAll() const
{
    const Index countTrue = mBuffer.mData.countOn();
    return countTrue > (NUM_VALUES >> 1);
}

template<Index Log2Dim>
inline Index
LeafNode<ValueMask, Log2Dim>::medianOn(bool& state) const
{
    const Index countTrueOn = mBuffer.mData.countOn();
    state = true;//since value and state are the same for this specialization of the leaf node
    return countTrueOn;
}

template<Index Log2Dim>
inline Index
LeafNode<ValueMask, Log2Dim>::medianOff(bool& state) const
{
    const Index countFalseOff = mBuffer.mData.countOff();
    state = false;//since value and state are the same for this specialization of the leaf node
    return countFalseOff;
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::clip(const CoordBBox& clipBBox, bool background)
{
    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.  Fill it with background tiles.
        this->fill(nodeBBox, background, /*active=*/false);
    } else if (clipBBox.isInside(nodeBBox)) {
        // This node lies completely inside the clipping region.  Leave it intact.
        return;
    }

    // This node isn't completely contained inside the clipping region.
    // Set any voxels that lie outside the region to the background value.

    // Construct a boolean mask that is on inside the clipping region and off outside it.
    NodeMaskType mask;
    nodeBBox.intersect(clipBBox);
    Coord xyz;
    int &x = xyz.x(), &y = xyz.y(), &z = xyz.z();
    for (x = nodeBBox.min().x(); x <= nodeBBox.max().x(); ++x) {
        for (y = nodeBBox.min().y(); y <= nodeBBox.max().y(); ++y) {
            for (z = nodeBBox.min().z(); z <= nodeBBox.max().z(); ++z) {
                mask.setOn(static_cast<Index32>(this->coordToOffset(xyz)));
            }
        }
    }

    // Set voxels that lie in the inactive region of the mask (i.e., outside
    // the clipping region) to the background value.
    for (MaskOffIter maskIter = mask.beginOff(); maskIter; ++maskIter) {
        this->setValueOff(maskIter.pos(), background);
    }
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::fill(const CoordBBox& bbox, bool value, bool)
{
    auto clippedBBox = this->getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    if (!clippedBBox) return;

    for (Int32 x = clippedBBox.min().x(); x <= clippedBBox.max().x(); ++x) {
        const Index offsetX = (x & (DIM-1u))<<2*Log2Dim;
        for (Int32 y = clippedBBox.min().y(); y <= clippedBBox.max().y(); ++y) {
            const Index offsetXY = offsetX + ((y & (DIM-1u))<<  Log2Dim);
            for (Int32 z = clippedBBox.min().z(); z <= clippedBBox.max().z(); ++z) {
                const Index offset = offsetXY + (z & (DIM-1u));
                mBuffer.mData.set(offset, value);
            }
        }
    }
}

template<Index Log2Dim>
inline void
LeafNode<ValueMask, Log2Dim>::fill(const bool& value, bool)
{
    mBuffer.fill(value);
}


} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
