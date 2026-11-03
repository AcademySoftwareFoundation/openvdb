/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2021  The Blosc Developers <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

/*********************************************************************
  The code in this file is heavily based on FastLZ, a lightning-fast
  lossless compression library.  See LICENSES/FASTLZ.txt for details.
**********************************************************************/


/* Decoder extracted from c-blosc v1.21.6 (BloscLZ 2.5.1).
 * The encoder and SIMD match-copy helpers are omitted. Overlapping matches
 * use a byte loop instead of the separate fastcopy implementation.
 * Bounds are checked before forming pointers outside the input/output.
 * Included by vdb_blosc.c; all functions have internal linkage.
 */
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__) && (__GNUC__ > 2)
#define BLOSCLZ_UNLIKELY(c) (__builtin_expect((c), 0))
#else
#define BLOSCLZ_UNLIKELY(c) (c)
#endif
#define MAX_DISTANCE 8191

// LZ4 wildCopy which can reach excellent copy bandwidth (even if insecure)
static void blosclz_wild_copy(uint8_t *out, const uint8_t* from, uint8_t* end) {
  uint8_t* d = out;
  const uint8_t* s = from;
  uint8_t* const e = end;

  do { memcpy(d,s,8); d+=8; s+=8; } while (d<e);
}

static int blosclz_decompress(const void* input, int length, void* output, int maxout) {
  const uint8_t* ip = (const uint8_t*)input;
  const uint8_t* ip_limit;
  uint8_t* op = (uint8_t*)output;
  uint32_t ctrl;
  uint8_t* op_limit;
  if (BLOSCLZ_UNLIKELY(length <= 0 || maxout <= 0)) {
    return 0;
  }
  ip_limit = ip + length;
  op_limit = op + maxout;
  ctrl = (*ip++) & 31U;

  while (1) {
    if (ctrl >= 32) {
      // match
      int32_t len = (int32_t)(ctrl >> 5U) - 1 ;
      int32_t ofs = (int32_t)(ctrl & 31U) << 8U;
      uint8_t code;
      const uint8_t* ref;

      if (len == 7 - 1) {
        do {
          if (BLOSCLZ_UNLIKELY(ip_limit - ip < 2)) {
            return 0;
          }
          code = *ip++;
          if (BLOSCLZ_UNLIKELY((size_t)(op_limit - op) < (size_t)len + code + 3)) {
            return 0;
          }
          len += code;
        } while (code == 255);
      }
      else {
        if (BLOSCLZ_UNLIKELY(ip_limit - ip < 2)) {
          return 0;
        }
      }
      code = *ip++;
      len += 3;
      ofs += code;

      /* match from 16-bit distance */
      if (BLOSCLZ_UNLIKELY(code == 255)) {
        if (ofs == (31U << 8U) + 255) {
          if (ip_limit - ip < 2) {
            return 0;
          }
          ofs = (*ip++) << 8U;
          ofs += *ip++;
          ofs += MAX_DISTANCE;
        }
      }

      if (BLOSCLZ_UNLIKELY(len > op_limit - op)) {
        return 0;
      }

      if (BLOSCLZ_UNLIKELY(ofs >= op - (uint8_t*)output)) {
        return 0;
      }

      if (BLOSCLZ_UNLIKELY(ip >= ip_limit)) break;
      ctrl = *ip++;

      ref = op - ofs - 1;
      if (ref == op - 1) {
        /* optimized copy for a run */
        memset(op, *ref, len);
        op += len;
      }
      else if ((op - ref >= 8) && (op_limit - op - len >= 8)) {
        // copy with an overlap not larger than 8
        blosclz_wild_copy(op, ref, op + len);
        op += len;
      }
      else {
        // general copy with any overlap
        for (; len > 0; --len) {
          *op++ = *ref++;
        }
      }
    }
    else {
      // literal
      ctrl++;
      if (BLOSCLZ_UNLIKELY(ctrl > (size_t)(op_limit - op))) {
        return 0;
      }
      if (BLOSCLZ_UNLIKELY(ctrl > (size_t)(ip_limit - ip))) {
        return 0;
      }

      memcpy(op, ip, ctrl); op += ctrl; ip += ctrl;

      if (BLOSCLZ_UNLIKELY(ip >= ip_limit)) break;
      ctrl = *ip++;
    }
  }

  return (int)(op - (uint8_t*)output);
}

#undef MAX_DISTANCE
#undef BLOSCLZ_UNLIKELY
