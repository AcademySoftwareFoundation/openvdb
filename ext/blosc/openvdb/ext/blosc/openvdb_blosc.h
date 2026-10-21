// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_BLOSC_H
#define OPENVDB_BLOSC_H

// Private copy of the vendored Blosc API for OpenVDB. Every name is prefixed so
// this header can be included alongside the other copy and any external
// blosc.h in any order.

#include <limits.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Decorates symbols exported by a shared library build
#if defined(OPENVDB_BLOSC_SHARED_LIBRARY)
  #if defined(_MSC_VER)
    #define OPENVDB_BLOSC_EXPORT __declspec(dllexport)
  #elif (defined(__GNUC__) && __GNUC__ >= 4) || defined(__clang__)
    #if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
      #define OPENVDB_BLOSC_EXPORT __attribute__((dllexport))
    #else
      #define OPENVDB_BLOSC_EXPORT __attribute__((visibility("default")))
    #endif
  #else
    #error Cannot determine how to define OPENVDB_BLOSC_EXPORT for this compiler.
  #endif
#else
  #define OPENVDB_BLOSC_EXPORT
#endif

/* Minimum header length */
#define OPENVDB_BLOSC_MIN_HEADER_LENGTH 16

/* The maximum overhead during compression in bytes. */
#define OPENVDB_BLOSC_MAX_OVERHEAD OPENVDB_BLOSC_MIN_HEADER_LENGTH

/* Maximum source buffer size to be compressed */
#define OPENVDB_BLOSC_MAX_BUFFERSIZE (INT_MAX - OPENVDB_BLOSC_MAX_OVERHEAD)

/* Name for the only compressor shipped with Blosc */
#define OPENVDB_BLOSC_LZ4_COMPNAME "lz4"

/**
  Compress `nbytes` from `src` into `dest`.

  `clevel`: compression level, 0 to 9.

  `doshuffle`: non-zero to apply byte-wise shuffle.

  `compressor`: name of the compressor, only "lz4" is supported.

  `blocksize`: requested size of the compressed blocks. If 0, an automatic
   blocksize is used.

  `numinternalthreads`: number of threads to use internally.

  Returns the compressed size, 0 if the data could not be compressed into
  `destsize` bytes, or a negative value on internal error.
*/
OPENVDB_BLOSC_EXPORT int openvdb_blosc_compress_ctx(int clevel, int doshuffle, size_t typesize,
                                         size_t nbytes, const void* src, void* dest,
                                         size_t destsize, const char* compressor,
                                         size_t blocksize, int numinternalthreads);

/**
  Decompress `src` into `dest`. Use openvdb_blosc_cbuffer_sizes to determine the
  size of the destination buffer. Never writes more than `destsize` bytes.

  Returns the decompressed size, or 0 or a negative value on error.
*/
OPENVDB_BLOSC_EXPORT int openvdb_blosc_decompress_ctx(const void* src, void* dest,
                                           size_t destsize, int numinternalthreads);

/**
  Return the uncompressed size (`nbytes`), compressed size (`cbytes`) and
  `blocksize` of a compressed buffer. Only the first OPENVDB_BLOSC_MIN_HEADER_LENGTH
  bytes of the buffer are read. All outputs are zero if the format is not
  supported.
*/
OPENVDB_BLOSC_EXPORT void openvdb_blosc_cbuffer_sizes(const void* cbuffer, size_t* nbytes,
                                           size_t* cbytes, size_t* blocksize);

#ifdef __cplusplus
}
#endif

#endif
