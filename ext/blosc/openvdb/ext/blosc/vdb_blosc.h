/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Author: Francesc Alted <francesc@blosc.org>

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/
#ifndef BLOSC_H
#define BLOSC_H

#include <limits.h>
#include <stdlib.h>

#ifndef VDB_BLOSC_SYMBOL_PREFIX
#error VDB_BLOSC_SYMBOL_PREFIX must be defined to avoid symbol clashes with other blosc builds
#endif

#define VDB_BLOSC_CONCAT_(a, b) a##_##b
#define VDB_BLOSC_CONCAT(a, b) VDB_BLOSC_CONCAT_(a, b)
#define blosc_init            VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_init)
#define blosc_compress_ctx    VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_compress_ctx)
#define blosc_decompress_ctx  VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_decompress_ctx)
#define blosc_cbuffer_sizes   VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_cbuffer_sizes)

#ifdef __cplusplus
extern "C" {
#endif

/* BLOSC_EXPORT decorates symbols exported by the blosc shared library. */
#if defined(BLOSC_SHARED_LIBRARY)
  #if defined(_MSC_VER)
    #define BLOSC_EXPORT __declspec(dllexport)
  #elif (defined(__GNUC__) && __GNUC__ >= 4) || defined(__clang__)
    #if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
      #define BLOSC_EXPORT __attribute__((dllexport))
    #else
      #define BLOSC_EXPORT __attribute__((visibility("default")))
    #endif  /* defined(_WIN32) || defined(__CYGWIN__) */
  #else
    #error Cannot determine how to define BLOSC_EXPORT for this compiler.
  #endif
#else
  #define BLOSC_EXPORT
#endif  /* defined(BLOSC_SHARED_LIBRARY) */

/* Minimum header length */
#define BLOSC_MIN_HEADER_LENGTH 16

/* The maximum overhead during compression in bytes.  This equals to
   BLOSC_MIN_HEADER_LENGTH now, but can be higher in future
   implementations */
#define BLOSC_MAX_OVERHEAD BLOSC_MIN_HEADER_LENGTH

/* Maximum source buffer size to be compressed */
#define BLOSC_MAX_BUFFERSIZE (INT_MAX - BLOSC_MAX_OVERHEAD)

/* Name for the only compressor shipped with Blosc */
#define BLOSC_LZ4_COMPNAME       "lz4"


/**
  Initialize the Blosc library environment.

  You must call this previous to any other Blosc call, unless you want
  Blosc to be used simultaneously in a multi-threaded environment, in
  which case you should *exclusively* use the
  blosc_compress_ctx()/blosc_decompress_ctx() pair (see below).
  */
BLOSC_EXPORT void blosc_init(void);

/**
  Context interface to blosc compression. This does not require a call
  to blosc_init() and can be called from multithreaded applications
  without the global lock being used, so allowing Blosc be executed
  simultaneously in those scenarios.

  It uses the same parameters than the blosc_compress() function plus:

  `compressor`: the string representing the type of compressor to use.

  `blocksize`: the requested size of the compressed blocks.  If 0, an
   automatic blocksize will be used.

  `numinternalthreads`: the number of threads to use internally.

  A negative return value means that an internal error happened.  This
  should never happen.  If you see this, please report it back
  together with the buffer data causing this and compression settings.
*/
BLOSC_EXPORT int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize,
                                    size_t nbytes, const void* src, void* dest,
                                    size_t destsize, const char* compressor,
                                    size_t blocksize, int numinternalthreads);

/**
  Context interface to blosc decompression. This does not require a
  call to blosc_init() and can be called from multithreaded
  applications without the global lock being used, so allowing Blosc
  be executed simultaneously in those scenarios.

  Call `blosc_cbuffer_sizes` to determine the size of the destination buffer.

  It uses the same parameters than the blosc_decompress() function plus:

  `numinternalthreads`: number of threads to use internally.

  Decompression is memory safe and guaranteed not to write the `dest`
  buffer more than what is specified in `destsize`.

  If an error occurs, e.g. the compressed data is corrupted or the
  output buffer is not large enough, then 0 (zero) or a negative value
  will be returned instead.
*/
BLOSC_EXPORT int blosc_decompress_ctx(const void *src, void *dest,
                                      size_t destsize, int numinternalthreads);

/**
  Return information about a compressed buffer, namely the number of
  uncompressed bytes (`nbytes`) and compressed (`cbytes`).  It also
  returns the `blocksize` (which is used internally for doing the
  compression by blocks).

  You only need to pass the first BLOSC_MIN_HEADER_LENGTH bytes of a
  compressed buffer for this call to work.

  If the format is not supported by the library, all output arguments will be
  filled with zeros.
  */
BLOSC_EXPORT void blosc_cbuffer_sizes(const void *cbuffer, size_t *nbytes,
				      size_t *cbytes, size_t *blocksize);

#ifdef __cplusplus
}
#endif


#endif
