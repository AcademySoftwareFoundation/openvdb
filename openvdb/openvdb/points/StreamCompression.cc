// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file points/StreamCompression.cc

#include "StreamCompression.h"
#include <openvdb/util/logging.h>
#include <openvdb/util/Assert.h>
#include <limits>
#include <map>
#include <vector>
#include <openvdb_blosc.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace compression {


bool
bloscCanCompress()
{
    return true;
}


size_t
bloscUncompressedSize(const char* buffer)
{
    size_t bytes, _1, _2;
    openvdb_blosc_cbuffer_sizes(buffer, &bytes, &_1, &_2);
    return bytes;
}


void
bloscCompress(char* compressedBuffer, size_t& compressedBytes, const size_t bufferBytes,
    const char* uncompressedBuffer, const size_t uncompressedBytes)
{
    if (bufferBytes > OPENVDB_BLOSC_MAX_BUFFERSIZE) {
        OPENVDB_LOG_DEBUG("Blosc compress failed due to exceeding maximum buffer size.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }
    if (bufferBytes < uncompressedBytes + OPENVDB_BLOSC_MAX_OVERHEAD) {
        OPENVDB_LOG_DEBUG("Blosc compress failed due to insufficient space in compressed buffer.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    if (uncompressedBytes <= OPENVDB_BLOSC_MINIMUM_BYTES) {
        // no Blosc compression performed below this limit
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    if (uncompressedBytes < OPENVDB_BLOSC_PAD_BYTES && bufferBytes < OPENVDB_BLOSC_PAD_BYTES + OPENVDB_BLOSC_MAX_OVERHEAD) {
        OPENVDB_LOG_DEBUG(
            "Blosc compress failed due to insufficient space in compressed buffer for padding.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    size_t inputBytes = uncompressedBytes;

    const char* buffer = uncompressedBuffer;

    std::unique_ptr<char[]> paddedBuffer;
    if (uncompressedBytes < OPENVDB_BLOSC_PAD_BYTES) {
        // input array padded with zeros below this limit to improve compression
        paddedBuffer.reset(new char[OPENVDB_BLOSC_PAD_BYTES]);
        std::memcpy(paddedBuffer.get(), buffer, uncompressedBytes);
        for (int i = static_cast<int>(uncompressedBytes); i < OPENVDB_BLOSC_PAD_BYTES; i++) {
            paddedBuffer.get()[i] = 0;
        }
        buffer = paddedBuffer.get();
        inputBytes = OPENVDB_BLOSC_PAD_BYTES;
    }

    int _compressedBytes = openvdb_blosc_compress_ctx(
        /*clevel=*/9, // 0 (no compression) to 9 (maximum compression)
        /*doshuffle=*/true,
        /*typesize=*/sizeof(float), // hard-coded to 4-bytes for better compression
        /*srcsize=*/inputBytes,
        /*src=*/buffer,
        /*dest=*/compressedBuffer,
        /*destsize=*/bufferBytes,
        OPENVDB_BLOSC_LZ4_COMPNAME,
        /*blocksize=*/inputBytes,
        /*numthreads=*/1);

    if (_compressedBytes <= 0) {
        std::ostringstream ostr;
        ostr << "Blosc failed to compress " << uncompressedBytes << " byte"
            << (uncompressedBytes == 1 ? "" : "s");
        if (_compressedBytes < 0) ostr << " (internal error " << _compressedBytes << ")";
        OPENVDB_LOG_DEBUG(ostr.str());
        compressedBytes = 0;
        return;
    }

    compressedBytes = _compressedBytes;

    // fail if compression does not result in a smaller buffer
    if (compressedBytes >= uncompressedBytes) {
        compressedBytes = 0;
    }
}


std::unique_ptr<char[]>
bloscCompress(const char* buffer, const size_t uncompressedBytes, size_t& compressedBytes,
    const bool resize)
{
    size_t tempBytes = uncompressedBytes;
    // increase temporary buffer for padding if necessary
    if (tempBytes >= OPENVDB_BLOSC_MINIMUM_BYTES && tempBytes < OPENVDB_BLOSC_PAD_BYTES) {
        tempBytes += OPENVDB_BLOSC_PAD_BYTES;
    }
    // increase by Blosc max overhead
    tempBytes += OPENVDB_BLOSC_MAX_OVERHEAD;
    const bool outOfRange = tempBytes > OPENVDB_BLOSC_MAX_BUFFERSIZE;
    std::unique_ptr<char[]> outBuffer(outOfRange ? new char[1] : new char[tempBytes]);

    bloscCompress(outBuffer.get(), compressedBytes, tempBytes, buffer, uncompressedBytes);

    if (compressedBytes == 0) {
        return nullptr;
    }

    // buffer size is larger due to Blosc overhead so resize
    // (resize can be skipped if the buffer is only temporary)

    if (resize) {
        std::unique_ptr<char[]> newBuffer(new char[compressedBytes]);
        std::memcpy(newBuffer.get(), outBuffer.get(), compressedBytes);
        outBuffer.reset(newBuffer.release());
    }

    return outBuffer;
}


size_t
bloscCompressedSize( const char* buffer, const size_t uncompressedBytes)
{
    size_t compressedBytes;
    bloscCompress(buffer, uncompressedBytes, compressedBytes, /*resize=*/false);
    return compressedBytes;
}


void
bloscDecompress(char* uncompressedBuffer, const size_t expectedBytes,
    const size_t bufferBytes, const char* compressedBuffer)
{
    size_t uncompressedBytes = bloscUncompressedSize(compressedBuffer);

    if (bufferBytes > OPENVDB_BLOSC_MAX_BUFFERSIZE) {
        OPENVDB_THROW(RuntimeError,
            "Blosc decompress failed due to exceeding maximum buffer size.");
    }
    if (bufferBytes < uncompressedBytes + OPENVDB_BLOSC_MAX_OVERHEAD) {
        OPENVDB_THROW(RuntimeError,
            "Blosc decompress failed due to insufficient space in uncompressed buffer.");
    }

    uncompressedBytes = openvdb_blosc_decompress_ctx(   /*src=*/compressedBuffer,
                                                /*dest=*/uncompressedBuffer,
                                                bufferBytes,
                                                /*numthreads=*/1);

    if (uncompressedBytes < 1) {
        OPENVDB_THROW(RuntimeError, "Blosc decompress returned error code " << uncompressedBytes);
    }

    if (uncompressedBytes == OPENVDB_BLOSC_PAD_BYTES && expectedBytes <= OPENVDB_BLOSC_PAD_BYTES) {
        // padded array to improve compression
    }
    else if (uncompressedBytes != expectedBytes) {
        OPENVDB_THROW(RuntimeError, "Expected to decompress " << expectedBytes
            << " byte" << (expectedBytes == 1 ? "" : "s") << ", got "
            << uncompressedBytes << " byte" << (uncompressedBytes == 1 ? "" : "s"));
    }
}


std::unique_ptr<char[]>
bloscDecompress(const char* buffer, const size_t expectedBytes, const bool resize)
{
    size_t uncompressedBytes = bloscUncompressedSize(buffer);
    size_t tempBytes = uncompressedBytes + OPENVDB_BLOSC_MAX_OVERHEAD;
    const bool outOfRange = tempBytes > OPENVDB_BLOSC_MAX_BUFFERSIZE;
    if (outOfRange)     tempBytes = 1;
    std::unique_ptr<char[]> outBuffer(new char[tempBytes]);

    bloscDecompress(outBuffer.get(), expectedBytes, tempBytes, buffer);

    // buffer size is larger due to Blosc overhead so resize
    // (resize can be skipped if the buffer is only temporary)

    if (resize) {
        std::unique_ptr<char[]> newBuffer(new char[expectedBytes]);
        std::memcpy(newBuffer.get(), outBuffer.get(), expectedBytes);
        outBuffer.reset(newBuffer.release());
    }

    return outBuffer;
}


void
bloscValidateCompressedBytes(const char* buffer, const size_t bufferBytes)
{
    if (bufferBytes < OPENVDB_BLOSC_MIN_HEADER_LENGTH) {
        OPENVDB_THROW(IoError, "Compressed buffer is smaller than the Blosc header.");
    }
    size_t uncompressedBytes, compressedBytes, blockSize;
    openvdb_blosc_cbuffer_sizes(buffer, &uncompressedBytes, &compressedBytes, &blockSize);
    if (compressedBytes > bufferBytes) {
        OPENVDB_THROW(IoError, "Compressed buffer is smaller than its Blosc header reports.");
    }
}


////////////////////////////////////////


long
Page::uncompressedBytes() const
{
    OPENVDB_ASSERT(mInfo);
    return mInfo->uncompressedBytes;
}


const char*
Page::buffer(const int index) const
{
    // page sizes that do not match the attribute sizes in a file can leave a page unread
    if (!mData) {
        OPENVDB_THROW(IoError, "Paged attribute data has not been read.");
    }
    return mData.get() + index;
}


void
Page::readHeader(std::istream& is)
{
    OPENVDB_ASSERT(mInfo);

    // read the (compressed) size of the page
    int compressedSize;
    is.read(reinterpret_cast<char*>(&compressedSize), sizeof(int));

    if (compressedSize == 0 || compressedSize == std::numeric_limits<int>::min()) {
        OPENVDB_THROW(IoError, "Invalid compressed page size for VDB file format.");
    }

    int uncompressedSize;
    // if uncompressed, read the (compressed) size of the page
    if (compressedSize > 0)     is.read(reinterpret_cast<char*>(&uncompressedSize), sizeof(int));
    else                        uncompressedSize = -compressedSize;

    if (uncompressedSize <= 0) {
        OPENVDB_THROW(IoError, "Invalid uncompressed page size for VDB file format.");
    }

    mInfo->compressedBytes = compressedSize;
    mInfo->uncompressedBytes = uncompressedSize;
}


void
Page::readBuffers(std::istream&is, bool delayed)
{
    (void) delayed;

    OPENVDB_ASSERT(mInfo);

    bool isCompressed = mInfo->compressedBytes > 0;

    std::unique_ptr<char[]> buffer(new char[
        (isCompressed ? mInfo->compressedBytes : -mInfo->compressedBytes)]);
    is.read(buffer.get(), (isCompressed ? mInfo->compressedBytes : -mInfo->compressedBytes));

    if (mInfo->compressedBytes > 0) {
        this->decompress(buffer);
    } else {
        this->copy(buffer, -static_cast<int>(mInfo->compressedBytes));
    }
    mInfo.reset();
}


void
Page::skipBuffers(std::istream& is)
{
    OPENVDB_ASSERT(mInfo);

    bool isCompressed = mInfo->compressedBytes > 0;
    std::streamsize bytes = isCompressed ?
        mInfo->compressedBytes : -mInfo->compressedBytes;

    auto meta = io::getStreamMetadataPtr(is);
    if (meta && meta->seekable()) {
        is.seekg(bytes, std::ios_base::cur);
    } else {
        std::vector<char> tempData(bytes);
        is.read(tempData.data(), bytes);
    }

    mInfo.reset();
}


void
Page::copy(const std::unique_ptr<char[]>& temp, int pageSize)
{
    mData.reset(new char[pageSize]);
    std::memcpy(mData.get(), temp.get(), pageSize);
}


void
Page::decompress(const std::unique_ptr<char[]>& temp)
{
    OPENVDB_ASSERT(mInfo);

    bloscValidateCompressedBytes(temp.get(), static_cast<size_t>(mInfo->compressedBytes));

    size_t uncompressedBytes = bloscUncompressedSize(temp.get());
    size_t tempBytes = uncompressedBytes;
    tempBytes += uncompressedBytes;
    mData.reset(new char[tempBytes]);

    // page handles index the data using the uncompressed size from the page header,
    // so the decompressed size must match it
    const size_t expectedBytes = static_cast<size_t>(mInfo->uncompressedBytes);
    bloscDecompress(mData.get(), expectedBytes, tempBytes, temp.get());
}


////////////////////////////////////////


PageHandle::PageHandle( const Page::Ptr& page, const int index, const int size)
    : mPage(page)
    , mIndex(index)
    , mSize(size)
{
}


Page&
PageHandle::page()
{
    OPENVDB_ASSERT(mPage);
    return *mPage;
}


std::unique_ptr<char[]>
PageHandle::read()
{
    OPENVDB_ASSERT(mIndex >= 0);
    OPENVDB_ASSERT(mSize > 0);
    std::unique_ptr<char[]> buffer(new char[mSize]);
    std::memcpy(buffer.get(), mPage->buffer(mIndex), mSize);
    return buffer;
}


////////////////////////////////////////


PagedInputStream::PagedInputStream(std::istream& is)
    : mIs(&is)
{
}


PageHandle::Ptr
PagedInputStream::createHandle(std::streamsize n)
{
    OPENVDB_ASSERT(mByteIndex <= mUncompressedBytes);

    if (mByteIndex == mUncompressedBytes) {

        mPage = std::make_shared<Page>();
        mPage->readHeader(*mIs);
        mUncompressedBytes = static_cast<int>(mPage->uncompressedBytes());
        mByteIndex = 0;
    }

    // a handle must lie within a single page, as PageHandle::read() copies from the page
    // buffer, which holds at least the uncompressed bytes from the page header
    if (n <= 0 || n > std::streamsize(mUncompressedBytes - mByteIndex)) {
        OPENVDB_THROW(IoError, "Paged attribute data does not fit in the page.");
    }

    // TODO: C++14 introduces std::make_unique
    PageHandle::Ptr pageHandle(new PageHandle(mPage, mByteIndex, int(n)));

    mByteIndex += int(n);

    return pageHandle;
}


void
PagedInputStream::read(PageHandle::Ptr& pageHandle, std::streamsize n, bool delayed)
{
    OPENVDB_ASSERT(mByteIndex <= mUncompressedBytes);

    Page& page = pageHandle->page();

    if (mByteIndex == mUncompressedBytes) {
        mUncompressedBytes = static_cast<int>(page.uncompressedBytes());
        page.readBuffers(*mIs, delayed);
        mByteIndex = 0;
    }

    if (n <= 0 || n > std::streamsize(mUncompressedBytes - mByteIndex)) {
        OPENVDB_THROW(IoError, "Paged attribute data does not fit in the page.");
    }

    mByteIndex += int(n);
}


void
PagedInputStream::skip(PageHandle::Ptr& pageHandle, std::streamsize n)
{
    OPENVDB_ASSERT(mByteIndex <= mUncompressedBytes);

    Page& page = pageHandle->page();

    if (mByteIndex == mUncompressedBytes) {
        mUncompressedBytes = static_cast<int>(page.uncompressedBytes());
        page.skipBuffers(*mIs);
        mByteIndex = 0;
    }

    if (n <= 0 || n > std::streamsize(mUncompressedBytes - mByteIndex)) {
        OPENVDB_THROW(IoError, "Paged attribute data does not fit in the page.");
    }

    mByteIndex += int(n);
}


////////////////////////////////////////


PagedOutputStream::PagedOutputStream()
{
    mCompressedData.reset(new char[PageSize + OPENVDB_BLOSC_MAX_OVERHEAD]);
}


PagedOutputStream::PagedOutputStream(std::ostream& os)
    : mOs(&os)
{
    mCompressedData.reset(new char[PageSize + OPENVDB_BLOSC_MAX_OVERHEAD]);
}


PagedOutputStream&
PagedOutputStream::write(const char* str, std::streamsize n)
{
    if (n > PageSize) {
        this->flush();
        // write out the block as if a whole page
        this->compressAndWrite(str, size_t(n));
    }
    else {
        // if the size of this block will overflow the page, flush to disk
        if ((int(n) + mBytes) > PageSize) {
            this->flush();
        }

        // store and increment the data in the current page
        std::memcpy(mData.get() + mBytes, str, n);
        mBytes += int(n);
    }

    return *this;
}


void
PagedOutputStream::flush()
{
    this->compressAndWrite(mData.get(), mBytes);
    mBytes = 0;
}


void
PagedOutputStream::compressAndWrite(const char* buffer, size_t size)
{
    if (size == 0)  return;

    OPENVDB_ASSERT(size < std::numeric_limits<int>::max());

    this->resize(size);

    size_t compressedBytes(0);
    if (mSizeOnly) {
        compressedBytes = bloscCompressedSize(buffer, size);
    }
    else {
        bloscCompress(mCompressedData.get(), compressedBytes, mCapacity + OPENVDB_BLOSC_MAX_OVERHEAD, buffer, size);
    }

    if (compressedBytes == 0) {
        int uncompressedBytes = -static_cast<int>(size);
        if (mSizeOnly) {
            mOs->write(reinterpret_cast<const char*>(&uncompressedBytes), sizeof(int));
        }
        else {
            mOs->write(buffer, size);
        }
    }
    else {
        if (mSizeOnly) {
            mOs->write(reinterpret_cast<const char*>(&compressedBytes), sizeof(int));
            mOs->write(reinterpret_cast<const char*>(&size), sizeof(int));
        }
        else {
            mOs->write(mCompressedData.get(), compressedBytes);
        }
    }
}


void
PagedOutputStream::resize(size_t size)
{
    // grow the capacity if not sufficient space
    size_t requiredSize = size;
    if (size < OPENVDB_BLOSC_PAD_BYTES && size >= OPENVDB_BLOSC_MINIMUM_BYTES) {
        requiredSize = OPENVDB_BLOSC_PAD_BYTES;
    }
    if (requiredSize > mCapacity) {
        mCapacity = requiredSize;
        mData.reset(new char[mCapacity]);
        mCompressedData.reset(new char[mCapacity + OPENVDB_BLOSC_MAX_OVERHEAD]);
    }
}

} // namespace compression
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
