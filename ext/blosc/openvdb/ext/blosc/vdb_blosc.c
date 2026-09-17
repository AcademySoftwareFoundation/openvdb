/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Author: Francesc Alted <francesc@blosc.org>
  Creation date: 2009-05-20

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Version numbers */
#define BLOSC_VERSION_MAJOR    1    /* for major interface/format changes  */
#define BLOSC_VERSION_MINOR    21   /* for minor interface/format changes  */
#define BLOSC_VERSION_RELEASE  6    /* for tweaks, bug-fixes, or development */

#define BLOSC_VERSION_STRING   "1.21.6"  /* string version.  Sync with above! */
#define BLOSC_VERSION_REVISION "$Rev$"   /* revision version */
#define BLOSC_VERSION_DATE     "$Date:: 2024-06-24 #$"    /* date version */

/* The *_FORMAT symbols should be just 1-byte long */
#define BLOSC_VERSION_FORMAT    2   /* Blosc format version, starting at 1 */



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

#include "vdb_blosc.h"

/* lz4 is compiled directly into this translation unit so that its symbols
   stay hidden rather than appearing in the shared library's export table -
   override the visibility macro before lz4.h defines it. */
#define LZ4LIB_VISIBILITY __attribute__((visibility("hidden")))
#include "lz4/lz4.h"
#include "lz4/lz4.c"

#include <stdint.h>

/* Some useful units */
#define _KB 1024

/* Minimum buffer size to be compressed */
#define MIN_BUFFERSIZE 128       /* Cannot be smaller than 66 */

/* The maximum number of splits in a block for compression */
#define MAX_SPLITS 16            /* Cannot be larger than 128 */

/* The size of L1 cache.  32 KB is quite common nowadays. */
#define L1 (32 * (_KB))

struct blosc_context {
  int32_t compress;               /* 1 if we are doing compression 0 if decompress */

  const uint8_t* src;
  uint8_t* dest;                  /* The current pos in the destination buffer */
  uint8_t* header_flags;          /* Flags for header */
  int compversion;                /* Compressor version byte, only used during decompression */
  int32_t sourcesize;             /* Number of bytes in source buffer (or uncompressed bytes in compressed file) */
  int32_t compressedsize;         /* Number of bytes of compressed data (only used when decompressing) */
  int32_t nblocks;                /* Number of total blocks in buffer */
  int32_t leftover;               /* Extra bytes at end of buffer */
  int32_t blocksize;              /* Length of the block in bytes */
  int32_t typesize;               /* Type size */
  int32_t num_output_bytes;       /* Counter for the number of output bytes */
  int32_t destsize;               /* Maximum size for destination buffer */
  uint8_t* bstarts;               /* Start of the buffer past header info */
  int32_t compcode;               /* Compressor code to use */
  int clevel;                     /* Compression level (1-9) */
};

static int32_t g_initlib = 0;


/* A portable malloc wrapper */
static uint8_t *my_malloc(size_t size)
{
  void *block = malloc(size);

  if (block == NULL) {
    printf("Error allocating memory!");
    return NULL;
  }

  return (uint8_t *)block;
}


/* Release memory booked by my_malloc */
static void my_free(void *block)
{
  free(block);
}


/* Copy 4 bytes from `*pa` to int32_t, changing endianness if necessary. */
static int32_t sw32_(const uint8_t *pa)
{
  int32_t idest;
  uint8_t *dest = (uint8_t *)&idest;
  int i = 1;                    /* for big/little endian detection */
  char *p = (char *)&i;

  if (p[0] != 1) {
    /* big endian */
    dest[0] = pa[3];
    dest[1] = pa[2];
    dest[2] = pa[1];
    dest[3] = pa[0];
  }
  else {
    /* little endian */
    dest[0] = pa[0];
    dest[1] = pa[1];
    dest[2] = pa[2];
    dest[3] = pa[3];
  }
  return idest;
}


/* Copy 4 bytes from `*pa` to `*dest`, changing endianness if necessary. */
static void _sw32(uint8_t* dest, int32_t a)
{
  uint8_t *pa = (uint8_t *)&a;
  int i = 1;                    /* for big/little endian detection */
  char *p = (char *)&i;

  if (p[0] != 1) {
    /* big endian */
    dest[0] = pa[3];
    dest[1] = pa[2];
    dest[2] = pa[1];
    dest[3] = pa[0];
  }
  else {
    /* little endian */
    dest[0] = pa[0];
    dest[1] = pa[1];
    dest[2] = pa[2];
    dest[3] = pa[3];
  }
}

/*
 * Conversion routines between compressor and compression libraries
 */

/* Get the compressor code for the compressor name. -1 if it is not available */
static int blosc_compname_to_compcode(const char *compname)
{
  int code = -1;  /* -1 means non-existent compressor code */

  if (strcmp(compname, BLOSC_LZ4_COMPNAME) == 0) {
    code = BLOSC_LZ4;
  }

  return code;
}


static int lz4_wrap_compress(const char* input, size_t input_length,
                             char* output, size_t maxout, int accel)
{
  int cbytes;
  cbytes = LZ4_compress_fast(input, output, (int)input_length, (int)maxout,
                             accel);
  return cbytes;
}

static int lz4_wrap_decompress(const void* input, int compressed_length,
                               void* output, int maxout)
{
  return LZ4_decompress_safe(input, output, compressed_length, maxout);
}

static int validate_decompress_format(const struct blosc_context* context) {
  int8_t header_flags = *(context->header_flags);
  int32_t compformat = (header_flags & 0xe0) >> 5;
  int compversion = context->compversion;

  if (compformat != BLOSC_LZ4_FORMAT) {
    return -5; /* signals no decompression support */
  }
  if (compversion != BLOSC_LZ4_VERSION_FORMAT) {
    return -9;
  }
  return 0;
}

/* Compute acceleration for lz4.  Based on discussions held in:
 * https://groups.google.com/forum/#!topic/lz4c/zosy90P8MQw */
static int get_accel(const struct blosc_context* context) {
  return 10 - context->clevel;
}


/* Shuffle a block of data by type size. */
static void shuffle_block(const size_t typesize, const size_t blocksize,
                          const uint8_t* const source, uint8_t* const destination)
{
  const size_t elements = blocksize / typesize;
  const size_t remainder = blocksize % typesize;
  size_t i, j;

  for (j = 0; j < typesize; j++) {
    for (i = 0; i < elements; i++) {
      destination[j * elements + i] = source[i * typesize + j];
    }
  }

  /* Copy any leftover bytes that don't fill a whole element. */
  memcpy(destination + (blocksize - remainder), source + (blocksize - remainder), remainder);
}

/* Reverse shuffle_block(). */
static void unshuffle_block(const size_t typesize, const size_t blocksize,
                            const uint8_t* const source, uint8_t* const destination)
{
  const size_t elements = blocksize / typesize;
  const size_t remainder = blocksize % typesize;
  size_t i, j;

  for (i = 0; i < elements; i++) {
    for (j = 0; j < typesize; j++) {
      destination[i * typesize + j] = source[j * elements + i];
    }
  }

  /* Copy any leftover bytes that don't fill a whole element. */
  memcpy(destination + (blocksize - remainder), source + (blocksize - remainder), remainder);
}

/* Shuffle & compress a single block */
static int blosc_c(const struct blosc_context* context, int32_t blocksize,
                   int32_t leftoverblock, int32_t ntbytes, int32_t maxbytes,
                   const uint8_t *src, uint8_t *dest, uint8_t *tmp)
{
  int8_t header_flags = *(context->header_flags);
  int dont_split = (header_flags & 0x10) >> 4;
  int32_t j, neblock, nsplits;
  int32_t cbytes;                   /* number of compressed bytes in split */
  int32_t ctbytes = 0;              /* number of compressed bytes in block */
  int32_t maxout;
  int32_t typesize = context->typesize;
  const uint8_t *_tmp = src;
  int accel;
  int doshuffle = (header_flags & BLOSC_DOSHUFFLE) && (typesize > 1);

  if (doshuffle) {
    /* Byte shuffling only makes sense if typesize > 1 */
    shuffle_block(typesize, blocksize, src, tmp);
    _tmp = tmp;
  }

  accel = get_accel(context);

  /* The number of splits for this block */
  if (!dont_split && !leftoverblock) {
    nsplits = typesize;
  }
  else {
    nsplits = 1;
  }
  neblock = blocksize / nsplits;
  for (j = 0; j < nsplits; j++) {
    dest += sizeof(int32_t);
    ntbytes += (int32_t)sizeof(int32_t);
    ctbytes += (int32_t)sizeof(int32_t);
    maxout = neblock;
    if (ntbytes+maxout > maxbytes) {
      maxout = maxbytes - ntbytes;   /* avoid buffer overrun */
      if (maxout <= 0) {
        return 0;                  /* non-compressible block */
      }
    }
    /* write_compression_header() has already validated that compcode is
       BLOSC_LZ4 before this point is reached. */
    cbytes = lz4_wrap_compress((char *)_tmp+j*neblock, (size_t)neblock,
                               (char *)dest, (size_t)maxout, accel);

    if (cbytes > maxout) {
      /* Buffer overrun caused by compression (should never happen) */
      return -1;
    }
    else if (cbytes < 0) {
      /* cbytes should never be negative */
      return -2;
    }
    else if (cbytes == 0 || cbytes == neblock) {
      /* The compressor has been unable to compress data at all. */
      /* Before doing the copy, check that we are not running into a
         buffer overflow. */
      if ((ntbytes+neblock) > maxbytes) {
        return 0;    /* Non-compressible data */
      }
      memcpy(dest, _tmp + j * neblock, neblock);
      cbytes = neblock;
    }
    _sw32(dest - 4, cbytes);
    dest += cbytes;
    ntbytes += cbytes;
    ctbytes += cbytes;
  }  /* Closes j < nsplits */

  return ctbytes;
}

/* Decompress & unshuffle a single block */
static int blosc_d(struct blosc_context* context, int32_t blocksize,
                   int32_t leftoverblock, const uint8_t* base_src,
                   int32_t src_offset, uint8_t* dest, uint8_t* tmp) {
  int8_t header_flags = *(context->header_flags);
  int dont_split = (header_flags & 0x10) >> 4;
  int32_t j, neblock, nsplits;
  int32_t nbytes;                /* number of decompressed bytes in split */
  const int32_t compressedsize = context->compressedsize;
  int32_t cbytes;                /* number of compressed bytes in split */
  int32_t ctbytes = 0;           /* number of compressed bytes in block */
  int32_t ntbytes = 0;           /* number of uncompressed bytes in block */
  uint8_t *_tmp = dest;
  int32_t typesize = context->typesize;
  int doshuffle = (header_flags & BLOSC_DOSHUFFLE) && (typesize > 1);
  const uint8_t* src;

  if (doshuffle) {
    _tmp = tmp;
  }

  /* The number of splits for this block */
  if (!dont_split &&
      /* For compatibility with before the introduction of the split flag */
      ((typesize <= MAX_SPLITS) && (blocksize/typesize) >= MIN_BUFFERSIZE) &&
      !leftoverblock) {
    nsplits = typesize;
  }
  else {
    nsplits = 1;
  }

  neblock = blocksize / nsplits;
  for (j = 0; j < nsplits; j++) {
    /* Validate src_offset */
    if (src_offset < 0 || src_offset > compressedsize - sizeof(int32_t)) {
      return -1;
    }
    cbytes = sw32_(base_src + src_offset); /* amount of compressed bytes */
    src_offset += sizeof(int32_t);
    /* Validate cbytes */
    if (cbytes < 0 || cbytes > context->compressedsize - src_offset) {
      return -1;
    }
    ctbytes += (int32_t)sizeof(int32_t);
    src = base_src + src_offset;
    /* Uncompress */
    if (cbytes == neblock) {
      memcpy(_tmp, src, neblock);
      nbytes = neblock;
    }
    else {
      nbytes = lz4_wrap_decompress(src, cbytes, _tmp, neblock);
      /* Check that decompressed bytes number is correct */
      if (nbytes != neblock) {
        return -2;
      }
    }
    src_offset += cbytes;
    ctbytes += cbytes;
    _tmp += nbytes;
    ntbytes += nbytes;
  } /* Closes j < nsplits */

  if (doshuffle) {
    unshuffle_block(typesize, blocksize, tmp, dest);
  }

  /* Return the number of uncompressed bytes */
  return ntbytes;
}

/* Serial version for compression/decompression */
static int serial_blosc(struct blosc_context* context)
{
  int32_t j, bsize, leftoverblock;
  int32_t cbytes;

  int32_t ntbytes = context->num_output_bytes;

  uint8_t *tmp = my_malloc(context->blocksize);

  for (j = 0; j < context->nblocks; j++) {
    if (context->compress && !(*(context->header_flags) & BLOSC_MEMCPYED)) {
      _sw32(context->bstarts + j * 4, ntbytes);
    }
    bsize = context->blocksize;
    leftoverblock = 0;
    if ((j == context->nblocks - 1) && (context->leftover > 0)) {
      bsize = context->leftover;
      leftoverblock = 1;
    }
    if (context->compress) {
      if (*(context->header_flags) & BLOSC_MEMCPYED) {
        /* We want to memcpy only */
        memcpy(context->dest + BLOSC_MAX_OVERHEAD + j * context->blocksize,
               context->src + j * context->blocksize, bsize);
        cbytes = bsize;
      }
      else {
        /* Regular compression */
        cbytes = blosc_c(context, bsize, leftoverblock, ntbytes,
                         context->destsize, context->src+j*context->blocksize,
                         context->dest+ntbytes, tmp);
        if (cbytes == 0) {
          ntbytes = 0;              /* incompressible data */
          break;
        }
      }
    }
    else {
      if (*(context->header_flags) & BLOSC_MEMCPYED) {
        /* We want to memcpy only */
        memcpy(context->dest + j * context->blocksize,
               context->src + BLOSC_MAX_OVERHEAD + j * context->blocksize, bsize);
        cbytes = bsize;
      }
      else {
        /* Regular decompression */
        cbytes = blosc_d(context, bsize, leftoverblock, context->src,
                         sw32_(context->bstarts + j * 4),
                         context->dest + j * context->blocksize, tmp);
      }
    }
    if (cbytes < 0) {
      ntbytes = cbytes;         /* error in blosc_c or blosc_d */
      break;
    }
    ntbytes += cbytes;
  }

  /* Free temporaries */
  my_free(tmp);

  return ntbytes;
}


/* Do the compression or decompression of the buffer. */
static int do_job(struct blosc_context* context)
{
  return serial_blosc(context);
}


/* Conditions for splitting a block before compressing with a codec. */
static int split_block(int typesize, int blocksize) {
  return (typesize <= MAX_SPLITS) && (blocksize / typesize) >= MIN_BUFFERSIZE;
}


static int32_t compute_blocksize(struct blosc_context* context, int32_t clevel,
                                 int32_t typesize, int32_t nbytes,
                                 int32_t forced_blocksize)
{
  int32_t blocksize;

  /* Protection against very small buffers */
  if (nbytes < (int32_t)typesize) {
    return 1;
  }

  blocksize = nbytes;           /* Start by a whole buffer as blocksize */

  if (forced_blocksize) {
    blocksize = forced_blocksize;
    /* Check that forced blocksize is not too small */
    if (blocksize < MIN_BUFFERSIZE) {
      blocksize = MIN_BUFFERSIZE;
    }
  }
  else if (nbytes >= L1) {
    blocksize = L1;

    switch (clevel) {
      case 0:
        /* Case of plain copy */
        blocksize /= 4;
        break;
      case 1:
        blocksize /= 2;
        break;
      case 2:
        blocksize *= 1;
        break;
      case 3:
        blocksize *= 2;
        break;
      case 4:
      case 5:
        blocksize *= 4;
        break;
      case 6:
      case 7:
      case 8:
        blocksize *= 8;
        break;
      case 9:
        blocksize *= 8;
        break;
      default:
        assert(0);
        break;
    }
  }

  /* Enlarge the blocksize for splittable codecs */
  if (clevel > 0 && split_block(typesize, blocksize)) {
    if (blocksize > (1 << 18)) {
      /* Do not use a too large split buffer (> 256 KB) for splitting codecs */
      blocksize = (1 << 18);
    }
    blocksize *= typesize;
    if (blocksize < (1 << 16)) {
      /* Do not use a too small blocksize (< 64 KB) when typesize is small */
      blocksize = (1 << 16);
    }
    if (blocksize > 1024 * 1024) {
      /* But do not exceed 1 MB per thread (having this capacity in L3 is normal in modern CPUs) */
      blocksize = 1024 * 1024;
    }

  }

  /* Check that blocksize is not too large */
  if (blocksize > (int32_t)nbytes) {
    blocksize = nbytes;
  }

  /* blocksize *must absolutely* be a multiple of the typesize */
  if (blocksize > typesize) {
    blocksize = blocksize / typesize * typesize;
  }

  return blocksize;
}

static int initialize_context_compression(struct blosc_context* context,
                          int clevel,
                          int doshuffle,
                          size_t typesize,
                          size_t sourcesize,
                          const void* src,
                          void* dest,
                          size_t destsize,
                          int32_t compressor,
                          int32_t blocksize)
{
  char *envvar = NULL;
  int warnlvl = 0;
  /* Set parameters */
  context->compress = 1;
  context->src = (const uint8_t*)src;
  context->dest = (uint8_t *)(dest);
  context->num_output_bytes = 0;
  context->destsize = (int32_t)destsize;
  context->sourcesize = sourcesize;
  context->typesize = typesize;
  context->compcode = compressor;
  context->clevel = clevel;

  envvar = getenv("BLOSC_WARN");
  if (envvar != NULL) {
    warnlvl = strtol(envvar, NULL, 10);
  }

  /* Check buffer size limits */
  if (sourcesize > BLOSC_MAX_BUFFERSIZE) {
    if (warnlvl > 0) {
      fprintf(stderr, "Input buffer size cannot exceed %d bytes\n",
              BLOSC_MAX_BUFFERSIZE);
    }
    return 0;
  }
  if (destsize < BLOSC_MAX_OVERHEAD) {
    if (warnlvl > 0) {
      fprintf(stderr, "Output buffer size should be larger than %d bytes\n",
              BLOSC_MAX_OVERHEAD);
    }
    return 0;
  }

  /* Compression level */
  if (clevel < 0 || clevel > 9) {
    fprintf(stderr, "`clevel` parameter must be between 0 and 9!\n");
    return -10;
  }

  /* Shuffle */
  if (doshuffle != 0 && doshuffle != 1) {
    fprintf(stderr, "`shuffle` parameter must be either 0 or 1!\n");
    return -10;
  }

  /* Check typesize limits */
  if (context->typesize > BLOSC_MAX_TYPESIZE) {
    /* If typesize is too large, treat buffer as an 1-byte stream. */
    context->typesize = 1;
  }

  /* Get the blocksize */
  context->blocksize = compute_blocksize(context, clevel, (int32_t)context->typesize, context->sourcesize, blocksize);

  /* Compute number of blocks in buffer */
  context->nblocks = context->sourcesize / context->blocksize;
  context->leftover = context->sourcesize % context->blocksize;
  context->nblocks = (context->leftover > 0) ? (context->nblocks + 1) : context->nblocks;

  return 1;
}


static int write_compression_header(struct blosc_context* context, int clevel, int doshuffle)
{
  int32_t compformat;
  int dont_split;

  /* Write version header for this block */
  context->dest[0] = BLOSC_VERSION_FORMAT;           /* blosc format version */

  /* Write compressor format */
  compformat = -1;
  if (context->compcode == BLOSC_LZ4) {
    compformat = BLOSC_LZ4_FORMAT;
    context->dest[1] = BLOSC_LZ4_VERSION_FORMAT;  /* lz4 format version */
  }
  else {
    fprintf(stderr, "Blosc has not been compiled with the requested ");
    fprintf(stderr, "compression support.  Please use one having it.");
    return -5;    /* signals no compression support */
  }

  context->header_flags = context->dest+2;  /* flags */
  context->dest[2] = 0;  /* zeroes flags */
  context->dest[3] = (uint8_t)context->typesize;  /* type size */
  _sw32(context->dest + 4, context->sourcesize);  /* size of the buffer */
  _sw32(context->dest + 8, context->blocksize);  /* block size */
  context->bstarts = context->dest + 16;  /* starts for every block */
  context->num_output_bytes = 16 + sizeof(int32_t)*context->nblocks;  /* space for header and pointers */

  if (context->clevel == 0) {
    /* Compression level 0 means buffer to be memcpy'ed */
    *(context->header_flags) |= BLOSC_MEMCPYED;
    context->num_output_bytes = 16;      /* space just for header */
  }

  if (context->sourcesize < MIN_BUFFERSIZE) {
    /* Buffer is too small.  Try memcpy'ing. */
    *(context->header_flags) |= BLOSC_MEMCPYED;
    context->num_output_bytes = 16;      /* space just for header */
  }

  if (doshuffle == BLOSC_SHUFFLE) {
    /* Byte-shuffle is active */
    *(context->header_flags) |= BLOSC_DOSHUFFLE;     /* bit 0 set to one in flags */
  }

  dont_split = !split_block(context->typesize, context->blocksize);
  *(context->header_flags) |= dont_split << 4;  /* dont_split is in bit 4 */
  *(context->header_flags) |= compformat << 5;  /* compressor format starts at bit 5 */

  return 1;
}


static int blosc_compress_context(struct blosc_context* context)
{
  int32_t ntbytes = 0;

  if ((*(context->header_flags) & BLOSC_MEMCPYED) &&
      (context->sourcesize + BLOSC_MAX_OVERHEAD > context->destsize)) {
    return 0;   /* data cannot be copied without overrun destination */
  }

  /* Do the actual compression */
  ntbytes = do_job(context);
  if (ntbytes < 0) {
    return -1;
  }
  if ((ntbytes == 0) && (context->sourcesize + BLOSC_MAX_OVERHEAD <= context->destsize)) {
    /* Last chance for fitting `src` buffer in `dest`.  Update flags and force a copy. */
    *(context->header_flags) |= BLOSC_MEMCPYED;
    context->num_output_bytes = BLOSC_MAX_OVERHEAD;  /* reset the output bytes in previous step */
    ntbytes = do_job(context);
    if (ntbytes < 0) {
      return -1;
    }
  }

  /* Set the number of compressed bytes in header */
  _sw32(context->dest + 12, ntbytes);

  assert(ntbytes <= context->destsize);
  return ntbytes;
}

/* The public routine for compression with context. */
int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize,
                       size_t nbytes, const void* src, void* dest,
                       size_t destsize, const char* compressor,
                       size_t blocksize, int numinternalthreads)
{
  int error, result;
  struct blosc_context context;
  (void)numinternalthreads;  /* no thread pool; kept for API compatibility */

  error = initialize_context_compression(&context, clevel, doshuffle, typesize,
					 nbytes, src, dest, destsize,
					 blosc_compname_to_compcode(compressor),
					 blocksize);
  if (error <= 0) { return error; }

  error = write_compression_header(&context, clevel, doshuffle);
  if (error <= 0) { return error; }

  result = blosc_compress_context(&context);

  return result;
}

static int blosc_run_decompression_with_context(struct blosc_context* context,
                                                const void* src,
                                                void* dest,
                                                size_t destsize)
{
  uint8_t version;
  int32_t ntbytes;

  context->compress = 0;
  context->src = (const uint8_t*)src;
  context->dest = (uint8_t*)dest;
  context->destsize = destsize;
  context->num_output_bytes = 0;

  /* Read the header block */
  version = context->src[0];                        /* blosc format version */
  context->compversion = context->src[1];

  context->header_flags = (uint8_t*)(context->src + 2);           /* flags */
  context->typesize = (int32_t)context->src[3];      /* typesize */
  context->sourcesize = sw32_(context->src + 4);     /* buffer size */
  context->blocksize = sw32_(context->src + 8);      /* block size */
  context->compressedsize = sw32_(context->src + 12); /* compressed buffer size */
  context->bstarts = (uint8_t*)(context->src + 16);

  if (context->sourcesize == 0) {
    /* Source buffer was empty, so we are done */
    return 0;
  }

  if (context->blocksize <= 0 || context->blocksize > destsize ||
      context->typesize <= 0 || context->typesize > BLOSC_MAX_TYPESIZE) {
    return -1;
  }

  if (version != BLOSC_VERSION_FORMAT) {
    /* Version from future */
    return -1;
  }
  if (*context->header_flags & 0x08) {
    /* compressor flags from the future */
    return -1;
  }

  /* Compute some params */
  /* Total blocks */
  context->nblocks = context->sourcesize / context->blocksize;
  context->leftover = context->sourcesize % context->blocksize;
  context->nblocks = (context->leftover>0)? context->nblocks+1: context->nblocks;

  /* Check that we have enough space to decompress */
  if (context->sourcesize > (int32_t)destsize) {
    return -1;
  }

  if (*(context->header_flags) & BLOSC_MEMCPYED) {
    /* Validate that compressed size is equal to decompressed size + header
       size. */
    if (context->sourcesize + BLOSC_MAX_OVERHEAD != context->compressedsize) {
      return -1;
    }
  } else {
    ntbytes = validate_decompress_format(context);
    if (ntbytes != 0) return ntbytes;

    /* Validate that compressed size is large enough to hold the bstarts array */
    if (context->nblocks > (context->compressedsize - 16) / 4) {
      return -1;
    }
  }

  /* Do the actual decompression */
  ntbytes = do_job(context);
  if (ntbytes < 0) {
    return -1;
  }

  assert(ntbytes <= (int32_t)destsize);
  return ntbytes;
}

int blosc_decompress_ctx(const void* src, void* dest, size_t destsize,
                         int numinternalthreads) {
  int result;
  struct blosc_context context;
  (void)numinternalthreads;  /* no thread pool; kept for API compatibility */

  result = blosc_run_decompression_with_context(&context, src, dest, destsize);

  return result;
}

/* Return `nbytes`, `cbytes` and `blocksize` from a compressed buffer. */
void blosc_cbuffer_sizes(const void *cbuffer, size_t *nbytes,
                         size_t *cbytes, size_t *blocksize)
{
  uint8_t *_src = (uint8_t *)(cbuffer);    /* current pos for source buffer */
  uint8_t version = _src[0];               /* version of header */

  if (version != BLOSC_VERSION_FORMAT) {
    *nbytes = *blocksize = *cbytes = 0;
    return;
  }

  /* Read the interesting values */
  *nbytes = (size_t)sw32_(_src + 4);       /* uncompressed buffer size */
  *blocksize = (size_t)sw32_(_src + 8);    /* block size */
  *cbytes = (size_t)sw32_(_src + 12);      /* compressed buffer size */
}

void blosc_init(void)
{
  /* Return if we are already initialized */
  if (g_initlib) return;

  g_initlib = 1;
}
