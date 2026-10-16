/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Author: Francesc Alted <francesc@blosc.org>

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

/* Private header, only for vdb_blosc.c. Do not include it from any other
   file. It maps the upstream Blosc names used by vdb_blosc.c onto the
   prefixed public API in openvdb_blosc.h or nanovdb_blosc.h, which must be
   included first. */
#ifndef VDB_BLOSC_PRIVATE_H
#define VDB_BLOSC_PRIVATE_H

#ifndef VDB_BLOSC_SYMBOL_PREFIX
#error VDB_BLOSC_SYMBOL_PREFIX must be defined to avoid symbol clashes with other blosc builds
#endif

#define VDB_BLOSC_CONCAT_(a, b) a##_##b
#define VDB_BLOSC_CONCAT(a, b) VDB_BLOSC_CONCAT_(a, b)
#define blosc_init            VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_init)
#define blosc_compress_ctx    VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_compress_ctx)
#define blosc_decompress_ctx  VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_decompress_ctx)
#define blosc_cbuffer_sizes   VDB_BLOSC_CONCAT(VDB_BLOSC_SYMBOL_PREFIX, blosc_cbuffer_sizes)

#if defined(OPENVDB_BLOSC_H)
#define BLOSC_MAX_OVERHEAD OPENVDB_BLOSC_MAX_OVERHEAD
#define BLOSC_MAX_BUFFERSIZE OPENVDB_BLOSC_MAX_BUFFERSIZE
#elif defined(NANOVDB_BLOSC_H)
#define BLOSC_MAX_OVERHEAD NANOVDB_BLOSC_MAX_OVERHEAD
#define BLOSC_MAX_BUFFERSIZE NANOVDB_BLOSC_MAX_BUFFERSIZE
#else
#error Include openvdb_blosc.h or nanovdb_blosc.h before vdb_blosc.h
#endif

#endif
