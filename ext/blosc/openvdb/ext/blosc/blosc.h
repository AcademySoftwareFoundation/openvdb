/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Author: Francesc Alted <francesc@blosc.org>

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/
#ifndef BLOSC_H
#define BLOSC_H

#include <limits.h>
#include <stdlib.h>

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

/* Version numbers */
#define BLOSC_VERSION_MAJOR    1    /* for major interface/format changes  */
#define BLOSC_VERSION_MINOR    21   /* for minor interface/format changes  */
#define BLOSC_VERSION_RELEASE  6    /* for tweaks, bug-fixes, or development */

#define BLOSC_VERSION_STRING   "1.21.6"  /* string version.  Sync with above! */
#define BLOSC_VERSION_REVISION "$Rev$"   /* revision version */
#define BLOSC_VERSION_DATE     "$Date:: 2024-06-24 #$"    /* date version */

/* The *_FORMAT symbols should be just 1-byte long */
#define BLOSC_VERSION_FORMAT    2   /* Blosc format version, starting at 1 */

/* Minimum header length */
#define BLOSC_MIN_HEADER_LENGTH 16

/* The maximum overhead during compression in bytes.  This equals to
   BLOSC_MIN_HEADER_LENGTH now, but can be higher in future
   implementations */
#define BLOSC_MAX_OVERHEAD BLOSC_MIN_HEADER_LENGTH

/* Maximum source buffer size to be compressed */
#define BLOSC_MAX_BUFFERSIZE (INT_MAX - BLOSC_MAX_OVERHEAD)

/* Maximum typesize before considering source buffer as a stream of bytes */
#define BLOSC_MAX_TYPESIZE 255         /* Cannot be larger than 255 */

/* Codes for shuffling (see blosc_compress) */
#define BLOSC_NOSHUFFLE   0  /* no shuffle */
#define BLOSC_SHUFFLE     1  /* byte-wise shuffle */

/* Codes for internal flags */
#define BLOSC_DOSHUFFLE    0x1	/* byte-wise shuffle */
#define BLOSC_MEMCPYED     0x2	/* plain copy */

/* Code for the only compressor shipped with Blosc */
#define BLOSC_LZ4       1

/* Name for the only compressor shipped with Blosc */
#define BLOSC_LZ4_COMPNAME       "lz4"

/* Code for the compression library shipped with Blosc (code must be < 8) */
#define BLOSC_LZ4_LIB       1

/* Name for the compression library shipped with Blosc */
#define BLOSC_LZ4_LIBNAME       "LZ4"

/* The code for the compressor format shipped with Blosc */
#define BLOSC_LZ4_FORMAT      BLOSC_LZ4_LIB

/* The version format for the compressor shipped with Blosc (starts at 1) */
#define BLOSC_LZ4_VERSION_FORMAT      1

/* Split mode used for blocks.  Kept internal; there is only one mode. */
#define BLOSC_FORWARD_COMPAT_SPLIT 4

/**
  Initialize the Blosc library environment.

  You must call this previous to any other Blosc call, unless you want
  Blosc to be used simultaneously in a multi-threaded environment, in
  which case you should *exclusively* use the
  blosc_compress_ctx()/blosc_decompress_ctx() pair (see below).
  */
BLOSC_EXPORT void blosc_init(void);


/**
  Compress a block of data in the `src` buffer and returns the size of
  the compressed block.  The size of `src` buffer is specified by
  `nbytes`.  There is not a minimum for `src` buffer size (`nbytes`).

  `clevel` is the desired compression level and must be a number
  between 0 (no compression) and 9 (maximum compression).

  `doshuffle` specifies whether the byte-wise shuffle filter should be
  applied or not.  BLOSC_NOSHUFFLE means not applying it, BLOSC_SHUFFLE
  means applying it.

  `typesize` is the number of bytes for the atomic type in binary
  `src` buffer.  This is mainly useful for the shuffle filter.
  For implementation reasons, only a 1 < `typesize` < 256 will allow the
  shuffle filter to work.  When `typesize` is not in this range, shuffle
  will be silently disabled.

  The `dest` buffer must have at least the size of `destsize`.  Blosc
  guarantees that if you set `destsize` to, at least,
  (`nbytes` + BLOSC_MAX_OVERHEAD), the compression will always succeed.
  The `src` buffer and the `dest` buffer can not overlap.

  Compression is memory safe and guaranteed not to write the `dest`
  buffer beyond what is specified in `destsize`.

  If `src` buffer cannot be compressed into `destsize`, the return
  value is zero and you should discard the contents of the `dest`
  buffer.

  A negative return value means that an internal error happened.  This
  should never happen.  If you see this, please report it back
  together with the buffer data causing this and compression settings.

  Environment variables
  ---------------------

  blosc_compress() honors different environment variables to control
  internal parameters without the need of doing that programmatically.
  Here are the ones supported:

  BLOSC_CLEVEL=(INTEGER): This will overwrite the `clevel` parameter
  before the compression process starts.

  BLOSC_SHUFFLE=[NOSHUFFLE | SHUFFLE]: This will overwrite the
  `doshuffle` parameter before the compression process starts.

  BLOSC_TYPESIZE=(INTEGER): This will overwrite the `typesize`
  parameter before the compression process starts.

  BLOSC_COMPRESSOR=[LZ4]: This will call
  blosc_set_compressor(BLOSC_COMPRESSOR) before the compression
  process starts.

  BLOSC_WARN=(INTEGER): This will print some warning message on stderr
  showing more info in situations where data inputs cannot be compressed.
  The values can range from 1 (less verbose) to 10 (full verbose).  0 is
  the same as if the BLOSC_WARN envvar was not defined.
  */
BLOSC_EXPORT int blosc_compress(int clevel, int doshuffle, size_t typesize,
				size_t nbytes, const void *src, void *dest,
				size_t destsize);


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
  Decompress a block of compressed data in `src`, put the result in
  `dest` and returns the size of the decompressed block.

  Call `blosc_cbuffer_sizes` to determine the size of the destination buffer.

  The `src` buffer and the `dest` buffer can not overlap.

  Decompression is memory safe and guaranteed not to write the `dest`
  buffer beyond what is specified in `destsize`.

  If an error occurs, e.g. the compressed data is corrupted or the
  output buffer is not large enough, then 0 (zero) or a negative value
  will be returned instead.
*/
BLOSC_EXPORT int blosc_decompress(const void *src, void *dest, size_t destsize);

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
  Select the compressor to be used.  The only one supported is "lz4".
  If this function is not called, then "lz4" will be used by default.

  In case the compressor is not recognized, or there is not support
  for it in this build, it returns a -1.  Else it returns the code for
  the compressor (>=0).
  */
BLOSC_EXPORT int blosc_set_compressor(const char* compname);


/**
  Get the `compname` associated with the `compcode`.

  If the compressor code is not recognized, or there is not support
  for it in this build, -1 is returned.  Else, the compressor code is
  returned.
 */
BLOSC_EXPORT int blosc_compcode_to_compname(int compcode, const char **compname);


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
