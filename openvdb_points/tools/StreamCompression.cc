///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file StreamCompression.cc
///
/// @authors Dan Bailey

#include <map>

#include <openvdb_points/tools/StreamCompression.h>

#include <openvdb/util/logging.h>

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace compression {


#ifdef OPENVDB_USE_BLOSC


bool bloscCanCompress()
{
    return true;
}


size_t bloscUncompressedSize(const char* buffer)
{
    size_t bytes, _1, _2;
    blosc_cbuffer_sizes(buffer, &bytes, &_1, &_2);
    return bytes;
}


void bloscCompress( char* compressedBuffer, size_t& compressedBytes, const size_t bufferBytes,
                    const char* uncompressedBuffer, const size_t uncompressedBytes)
{
    if (bufferBytes > BLOSC_MAX_BUFFERSIZE) {
        OPENVDB_LOG_DEBUG("Blosc compress failed due to exceeding maximum buffer size.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }
    if (bufferBytes < uncompressedBytes + BLOSC_MAX_OVERHEAD) {
        OPENVDB_LOG_DEBUG("Blosc compress failed due to insufficient space in compressed buffer.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    if (uncompressedBytes <= BLOSC_MINIMUM_BYTES) {
        // no Blosc compression performed below this limit
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    if (uncompressedBytes < BLOSC_PAD_BYTES && bufferBytes < BLOSC_PAD_BYTES + BLOSC_MAX_OVERHEAD) {
        OPENVDB_LOG_DEBUG("Blosc compress failed due to insufficient space in compressed buffer for padding.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    size_t inputBytes = uncompressedBytes;

    const char* buffer = uncompressedBuffer;

    std::unique_ptr<char[]> paddedBuffer;
    if (uncompressedBytes < BLOSC_PAD_BYTES) {
        // input array padded with zeros below this limit to improve compression
        paddedBuffer.reset(new char[BLOSC_PAD_BYTES]);
        std::memcpy(paddedBuffer.get(), buffer, uncompressedBytes);
        for (int i = uncompressedBytes; i < BLOSC_PAD_BYTES; i++) {
            paddedBuffer.get()[i] = 0;
        }
        buffer = paddedBuffer.get();
        inputBytes = BLOSC_PAD_BYTES;
    }

    int _compressedBytes = blosc_compress_ctx(
        /*clevel=*/9, // 0 (no compression) to 9 (maximum compression)
        /*doshuffle=*/true,
        /*typesize=*/sizeof(float), // hard-coded to 4-bytes for better compression
        /*srcsize=*/inputBytes,
        /*src=*/buffer,
        /*dest=*/compressedBuffer,
        /*destsize=*/bufferBytes,
        BLOSC_LZ4_COMPNAME,
        /*blocksize=*/inputBytes,
        /*numthreads=*/1);

    if (_compressedBytes <= 0) {
        std::ostringstream ostr;
        ostr << "Blosc failed to compress " << uncompressedBytes << " byte" << (uncompressedBytes == 1 ? "" : "s");
        if (_compressedBytes < 0) ostr << " (internal error " << _compressedBytes << ")";
        OPENVDB_LOG_DEBUG(ostr.str());
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    compressedBytes = _compressedBytes;

    // fail if compression does not result in a smaller buffer

    if (compressedBytes >= uncompressedBytes) {
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }
}


std::unique_ptr<char[]> bloscCompress(const char* buffer, const size_t uncompressedBytes, size_t& compressedBytes, const bool resize)
{
    size_t tempBytes = uncompressedBytes + BLOSC_MAX_OVERHEAD;
    // increase temporary buffer for padding if necessary
    if (tempBytes >= BLOSC_MINIMUM_BYTES && tempBytes < BLOSC_PAD_BYTES) {
        tempBytes += BLOSC_PAD_BYTES + BLOSC_MAX_OVERHEAD;
    }
    const bool outOfRange = tempBytes > BLOSC_MAX_BUFFERSIZE;
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


size_t bloscCompressedSize( const char* buffer, const size_t uncompressedBytes)
{
    size_t compressedBytes;
    bloscCompress(buffer, uncompressedBytes, compressedBytes, /*resize=*/false);
    return compressedBytes;
}


void bloscDecompress(char* uncompressedBuffer, const size_t expectedBytes, const size_t bufferBytes, const char* compressedBuffer)
{
    size_t uncompressedBytes = bloscUncompressedSize(compressedBuffer);

    if (bufferBytes > BLOSC_MAX_BUFFERSIZE) {
        OPENVDB_THROW(RuntimeError, "Blosc decompress failed due to exceeding maximum buffer size.");
    }
    if (bufferBytes < uncompressedBytes + BLOSC_MAX_OVERHEAD) {
        OPENVDB_THROW(RuntimeError, "Blosc decompress failed due to insufficient space in uncompressed buffer.");
    }

    uncompressedBytes = blosc_decompress_ctx(   /*src=*/compressedBuffer,
                                                /*dest=*/uncompressedBuffer,
                                                bufferBytes,
                                                /*numthreads=*/1);

    if (uncompressedBytes < 1) {
        OPENVDB_THROW(RuntimeError, "Blosc decompress returned error code " << uncompressedBytes);
    }

    if (uncompressedBytes == BLOSC_PAD_BYTES && expectedBytes <= BLOSC_PAD_BYTES) {
        // padded array to improve compression
    }
    else if (uncompressedBytes != expectedBytes) {
        OPENVDB_THROW(RuntimeError, "Expected to decompress " << expectedBytes
            << " byte" << (expectedBytes == 1 ? "" : "s") << ", got "
            << uncompressedBytes << " byte" << (uncompressedBytes == 1 ? "" : "s"));
    }
}


std::unique_ptr<char[]> bloscDecompress(const char* buffer, const size_t expectedBytes, const bool resize)
{
    size_t uncompressedBytes = bloscUncompressedSize(buffer);
    size_t tempBytes = uncompressedBytes + BLOSC_MAX_OVERHEAD;
    const bool outOfRange = tempBytes > BLOSC_MAX_BUFFERSIZE;
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


#else


bool bloscCanCompress()
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    return false;
}


size_t bloscUncompressedSize(const char*)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


void bloscCompress( char*, size_t& compressedBytes, const size_t,
                    const char*, const size_t)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    compressedBytes = 0;
}


std::unique_ptr<char[]> bloscCompress(const char*, const size_t, size_t& compressedBytes, const bool)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    compressedBytes = 0;
    return nullptr;
}


size_t bloscCompressedSize(const char*, const size_t)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    return 0;
}


void bloscDecompress(char*, const size_t, const size_t, const char*)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


std::unique_ptr<char[]> bloscDecompress(const char*, const size_t, const bool)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


#endif // OPENVDB_USE_BLOSC


////////////////////////////////////////


} // namespace compression
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
