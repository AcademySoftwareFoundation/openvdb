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
/// @file StreamCompression.h
///
/// @authors Dan Bailey
///
/// @brief Lossless block compression schemes such as Blosc.
///


#ifndef OPENVDB_TOOLS_STREAM_COMPRESSION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_STREAM_COMPRESSION_HAS_BEEN_INCLUDED

#include <openvdb/io/Compression.h> // COMPRESS_BLOSC

#include <memory>
#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace compression {


// This is the minimum number of bytes below which Blosc compression is not used to
// avoid unecessary computation, as Blosc offers minimal compression until this limit
static const int BLOSC_MINIMUM_BYTES = 48;

// This is the minimum number of bytes below which the array is padded with zeros up
// to this number of bytes to allow Blosc to perform compression with small arrays
static const int BLOSC_PAD_BYTES = 128;


/// @brief Returns true if compression is available
bool bloscCanCompress();

/// @brief Retrieves the uncompressed size of buffer when uncompressed
///
/// @param buffer the compressed buffer
size_t bloscUncompressedSize(const char* buffer);

/// @brief Compress into the supplied buffer.
///
/// @param compressedBuffer     the buffer to compress
/// @param compressedBytes      number of compressed bytes
/// @param bufferBytes          the number of bytes in compressedBuffer available to be filled
/// @param compressedBuffer     the uncompressed buffer to compress
/// @param uncompressedBytes    number of uncompressed bytes
void bloscCompress( char* compressedBuffer, size_t& compressedBytes, const size_t bufferBytes,
                    const char* uncompressedBuffer, const size_t uncompressedBytes);

/// @brief Compress and return the heap-allocated compressed buffer.
///
/// @param buffer               the buffer to compress
/// @param uncompressedBytes    number of uncompressed bytes
/// @param compressedBytes      number of compressed bytes (written to this variable)
/// @param resize               the compressed buffer will be exactly resized to remove the
///                             portion used for Blosc overhead, for efficiency this can be
///                             skipped if it is known that the resulting buffer is temporary
std::unique_ptr<char[]> bloscCompress(  const char* buffer, const size_t uncompressedBytes,
                                        size_t& compressedBytes, const bool resize = true);

/// @brief Convenience wrapper to retrieve the compressed size of buffer when compressed
///
/// @param buffer the uncompressed buffer
/// @param uncompressedBytes number of uncompressed bytes
size_t bloscCompressedSize(const char* buffer, const size_t uncompressedBytes);

/// @brief Decompress into the supplied buffer. Will throw if decompression fails or
///        uncompressed buffer has insufficient space in which to decompress.
///
/// @param uncompressedBuffer the uncompressed buffer to decompress into
/// @param expectedBytes the number of bytes expected once the buffer is decompressed
/// @param bufferBytes the number of bytes in uncompressedBuffer available to be filled
/// @param compressedBuffer the compressed buffer to decompress
void bloscDecompress(   char* uncompressedBuffer, const size_t expectedBytes,
                        const size_t bufferBytes, const char* compressedBuffer);

/// @brief Decompress and return the the heap-allocated uncompressed buffer.
///
/// @param buffer the buffer to decompress
/// @param expectedBytes the number of bytes expected once the buffer is decompressed
/// @param resize               the compressed buffer will be exactly resized to remove the
///                             portion used for Blosc overhead, for efficiency this can be
///                             skipped if it is known that the resulting buffer is temporary
std::unique_ptr<char[]> bloscDecompress(const char* buffer, const size_t expectedBytes,
                                        const bool resize = true);


} // namespace compression
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_STREAM_COMPRESSION_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
