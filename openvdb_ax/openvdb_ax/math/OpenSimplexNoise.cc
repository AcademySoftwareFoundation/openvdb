// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file math/OpenSimplexNoise.cc

#include "OpenSimplexNoise.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

// see OpenSimplexNoise.h for details about the origin on this code

namespace OSN {

namespace {

template <typename T>
inline T pow4 (T x)
{
    x *= x;
    return x*x;
}

template <typename T>
inline T pow2 (T x)
{
    return x*x;
}

template <typename T>
inline OSNoise::inttype fastFloori (T x)
{
    OSNoise::inttype ip = (OSNoise::inttype)x;

    if (x < 0.0) --ip;

    return ip;
}

inline void LCG_STEP (int64_t & x)
{
    // Magic constants are attributed to Donald Knuth's MMIX implementation.
    static const int64_t MULTIPLIER = 6364136223846793005LL;
    static const int64_t INCREMENT  = 1442695040888963407LL;
    x = ((x * MULTIPLIER) + INCREMENT);
}

} // anonymous namespace

// Array of gradient values for 3D. They approximate the directions to the
// vertices of a rhombicuboctahedron from its center, skewed so that the
// triangular and square facets can be inscribed in circles of the same radius.
// New gradient set 2014-10-06.
const int OSNoise::sGradients [] = {
    -11, 4, 4,  -4, 11, 4,  -4, 4, 11,   11, 4, 4,   4, 11, 4,   4, 4, 11,
    -11,-4, 4,  -4,-11, 4,  -4,-4, 11,   11,-4, 4,   4,-11, 4,   4,-4, 11,
    -11, 4,-4,  -4, 11,-4,  -4, 4,-11,   11, 4,-4,   4, 11,-4,   4, 4,-11,
    -11,-4,-4,  -4,-11,-4,  -4,-4,-11,   11,-4,-4,   4,-11,-4,   4,-4,-11
};

template <typename T>
inline T OSNoise::extrapolate(const OSNoise::inttype xsb,
                              const OSNoise::inttype ysb,
                              const OSNoise::inttype zsb,
                              const T dx,
                              const T dy,
                              const T dz) const
{
    unsigned int index = mPermGradIndex[(mPerm[(mPerm[xsb & 0xFF] + ysb) & 0xFF] + zsb) & 0xFF];
    return sGradients[index] * dx +
           sGradients[index + 1] * dy +
           sGradients[index + 2] * dz;

}

template <typename T>
inline T OSNoise::extrapolate(const OSNoise::inttype xsb,
                              const OSNoise::inttype ysb,
                              const OSNoise::inttype zsb,
                              const T dx,
                              const T dy,
                              const T dz,
                              T (&de) [3]) const
{
  unsigned int index = mPermGradIndex[(mPerm[(mPerm[xsb & 0xFF] + ysb) & 0xFF] + zsb) & 0xFF];
  return (de[0] = sGradients[index]) * dx +
         (de[1] = sGradients[index + 1]) * dy +
         (de[2] = sGradients[index + 2]) * dz;
}

OSNoise::OSNoise(OSNoise::inttype seed)
{
  int source [256];
  for (int i = 0; i < 256; ++i) { source[i] = i; }
  LCG_STEP(seed);
  LCG_STEP(seed);
  LCG_STEP(seed);
  for (int i = 255; i >= 0; --i) {
    LCG_STEP(seed);
    int r = (int)((seed + 31) % (i + 1));
    if (r < 0) { r += (i + 1); }
    mPerm[i] = source[r];
    mPermGradIndex[i] = (int)((mPerm[i] % (72 / 3)) * 3);
    source[r] = source[i];
  }
}

OSNoise::OSNoise(const int * p)
{
  // Copy the supplied permutation array into this instance.
  for (int i = 0; i < 256; ++i) {
    mPerm[i] = p[i];
    mPermGradIndex[i] = (int)((mPerm[i] % (72 / 3)) * 3);
  }
}

template <typename T>
T OSNoise::eval(const T x, const T y, const T z) const
{
  static_assert(std::is_floating_point<T>::value, "OpenSimplexNoise can only be used with floating-point types");

  static const T STRETCH_CONSTANT = (T)(-1.0 / 6.0); // (1 / sqrt(3 + 1) - 1) / 3
  static const T SQUISH_CONSTANT  = (T)(1.0 / 3.0);  // (sqrt(3 + 1) - 1) / 3
  static const T NORM_CONSTANT    = (T)(1.0 / 103.0);

  OSNoise::inttype xsb, ysb, zsb;
  T dx0, dy0, dz0;
  T xins, yins, zins;

  // Parameters for the individual contributions
  T contr_m [9], contr_ext [9];

  {
    // Place input coordinates on simplectic lattice.
    T stretchOffset = (x + y + z) * STRETCH_CONSTANT;
    T xs = x + stretchOffset;
    T ys = y + stretchOffset;
    T zs = z + stretchOffset;

    // Floor to get simplectic lattice coordinates of rhombohedron
    // (stretched cube) super-cell.
#ifdef __FAST_MATH__
    T xsbd = std::floor(xs);
    T ysbd = std::floor(ys);
    T zsbd = std::floor(zs);
    xsb = (OSNoise::inttype)xsbd;
    ysb = (OSNoise::inttype)ysbd;
    zsb = (OSNoise::inttype)zsbd;
#else
    xsb = fastFloori(xs);
    ysb = fastFloori(ys);
    zsb = fastFloori(zs);
    T xsbd = (T)xsb;
    T ysbd = (T)ysb;
    T zsbd = (T)zsb;
#endif

    // Skew out to get actual coordinates of rhombohedron origin.
    T squishOffset = (xsbd + ysbd + zsbd) * SQUISH_CONSTANT;
    T xb = xsbd + squishOffset;
    T yb = ysbd + squishOffset;
    T zb = zsbd + squishOffset;

    // Positions relative to origin point.
    dx0 = x - xb;
    dy0 = y - yb;
    dz0 = z - zb;

    // Compute simplectic lattice coordinates relative to rhombohedral origin.
    xins = xs - xsbd;
    yins = ys - ysbd;
    zins = zs - zsbd;
  }

  // These are given values inside the next block, and used afterwards.
  OSNoise::inttype xsv_ext0, ysv_ext0, zsv_ext0;
  OSNoise::inttype xsv_ext1, ysv_ext1, zsv_ext1;
  T dx_ext0, dy_ext0, dz_ext0;
  T dx_ext1, dy_ext1, dz_ext1;

  // Sum together to get a value that determines which cell we are in.
  T inSum = xins + yins + zins;

  if (inSum > (T)1.0 && inSum < (T)2.0) {
    // The point is inside the octahedron (rectified 3-Simplex) inbetween.

    T aScore;
    uint_fast8_t aPoint;
    bool aIsFurtherSide;
    T bScore;
    uint_fast8_t bPoint;
    bool bIsFurtherSide;

    // Decide between point (1,0,0) and (0,1,1) as closest.
    T p1 = xins + yins;
    if (p1 <= (T)1.0) {
      aScore = (T)1.0 - p1;
      aPoint = 4;
      aIsFurtherSide = false;
    } else {
      aScore = p1 - (T)1.0;
      aPoint = 3;
      aIsFurtherSide = true;
    }

    // Decide between point (0,1,0) and (1,0,1) as closest.
    T p2 = xins + zins;
    if (p2 <= (T)1.0) {
      bScore = (T)1.0 - p2;
      bPoint = 2;
      bIsFurtherSide = false;
    } else {
      bScore = p2 - (T)1.0;
      bPoint = 5;
      bIsFurtherSide = true;
    }

    // The closest out of the two (0,0,1) and (1,1,0) will replace the
    // furthest out of the two decided above if closer.
    T p3 = yins + zins;
    if (p3 > (T)1.0) {
      T score = p3 - (T)1.0;
      if (aScore > bScore && bScore < score) {
        bScore = score;
        bPoint = 6;
        bIsFurtherSide = true;
      } else if (aScore <= bScore && aScore < score) {
        aScore = score;
        aPoint = 6;
        aIsFurtherSide = true;
      }
    } else {
      T score = (T)1.0 - p3;
      if (aScore > bScore && bScore < score) {
        bScore = score;
        bPoint = 1;
        bIsFurtherSide = false;
      } else if (aScore <= bScore && aScore < score) {
        aScore = score;
        aPoint = 1;
        aIsFurtherSide = false;
      }
    }

    // Where each of the two closest points are determines how the
    // extra two vertices are calculated.
    if (aIsFurtherSide == bIsFurtherSide) {
      if (aIsFurtherSide) {
        // Both closest points on (1,1,1) side.

        // One of the two extra points is (1,1,1)
        xsv_ext0 = xsb + 1;
        ysv_ext0 = ysb + 1;
        zsv_ext0 = zsb + 1;
        dx_ext0 = dx0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
        dy_ext0 = dy0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
        dz_ext0 = dz0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);

        // Other extra point is based on the shared axis.
        uint_fast8_t c = aPoint & bPoint;
        if (c & 0x01) {
          xsv_ext1 = xsb + 2;
          ysv_ext1 = ysb;
          zsv_ext1 = zsb;
          dx_ext1 = dx0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
          dy_ext1 = dy0 - (SQUISH_CONSTANT * (T)2.0);
          dz_ext1 = dz0 - (SQUISH_CONSTANT * (T)2.0);
        } else if (c & 0x02) {
          xsv_ext1 = xsb;
          ysv_ext1 = ysb + 2;
          zsv_ext1 = zsb;
          dx_ext1 = dx0 - (SQUISH_CONSTANT * (T)2.0);
          dy_ext1 = dy0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
          dz_ext1 = dz0 - (SQUISH_CONSTANT * (T)2.0);
        } else {
          xsv_ext1 = xsb;
          ysv_ext1 = ysb;
          zsv_ext1 = zsb + 2;
          dx_ext1 = dx0 - (SQUISH_CONSTANT * (T)2.0);
          dy_ext1 = dy0 - (SQUISH_CONSTANT * (T)2.0);
          dz_ext1 = dz0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
        }
      } else {
        // Both closest points are on the (0,0,0) side.

        // One of the two extra points is (0,0,0).
        xsv_ext0 = xsb;
        ysv_ext0 = ysb;
        zsv_ext0 = zsb;
        dx_ext0 = dx0;
        dy_ext0 = dy0;
        dz_ext0 = dz0;

        // The other extra point is based on the omitted axis.
        uint_fast8_t c = aPoint | bPoint;
        if (!(c & 0x01)) {
          xsv_ext1 = xsb - 1;
          ysv_ext1 = ysb + 1;
          zsv_ext1 = zsb + 1;
          dx_ext1 = dx0 + (T)1.0 - SQUISH_CONSTANT;
          dy_ext1 = dy0 - (T)1.0 - SQUISH_CONSTANT;
          dz_ext1 = dz0 - (T)1.0 - SQUISH_CONSTANT;
        } else if (!(c & 0x02)) {
          xsv_ext1 = xsb + 1;
          ysv_ext1 = ysb - 1;
          zsv_ext1 = zsb + 1;
          dx_ext1 = dx0 - (T)1.0 - SQUISH_CONSTANT;
          dy_ext1 = dy0 + (T)1.0 - SQUISH_CONSTANT;
          dz_ext1 = dz0 - (T)1.0 - SQUISH_CONSTANT;
        } else {
          xsv_ext1 = xsb + 1;
          ysv_ext1 = ysb + 1;
          zsv_ext1 = zsb - 1;
          dx_ext1 = dx0 - (T)1.0 - SQUISH_CONSTANT;
          dy_ext1 = dy0 - (T)1.0 - SQUISH_CONSTANT;
          dz_ext1 = dz0 + (T)1.0 - SQUISH_CONSTANT;
        }
      }
    } else {
      // One point is on the (0,0,0) side, one point is on the (1,1,1) side.

      uint_fast8_t c1, c2;
      if (aIsFurtherSide) {
        c1 = aPoint;
        c2 = bPoint;
      } else {
        c1 = bPoint;
        c2 = aPoint;
      }

      // One contribution is a permutation of (1,1,-1).
      if (!(c1 & 0x01)) {
        xsv_ext0 = xsb - 1;
        ysv_ext0 = ysb + 1;
        zsv_ext0 = zsb + 1;
        dx_ext0 = dx0 + (T)1.0 - SQUISH_CONSTANT;
        dy_ext0 = dy0 - (T)1.0 - SQUISH_CONSTANT;
        dz_ext0 = dz0 - (T)1.0 - SQUISH_CONSTANT;
      } else if (!(c1 & 0x02)) {
        xsv_ext0 = xsb + 1;
        ysv_ext0 = ysb - 1;
        zsv_ext0 = zsb + 1;
        dx_ext0 = dx0 - (T)1.0 - SQUISH_CONSTANT;
        dy_ext0 = dy0 + (T)1.0 - SQUISH_CONSTANT;
        dz_ext0 = dz0 - (T)1.0 - SQUISH_CONSTANT;
      } else {
        xsv_ext0 = xsb + 1;
        ysv_ext0 = ysb + 1;
        zsv_ext0 = zsb - 1;
        dx_ext0 = dx0 - (T)1.0 - SQUISH_CONSTANT;
        dy_ext0 = dy0 - (T)1.0 - SQUISH_CONSTANT;
        dz_ext0 = dz0 + (T)1.0 - SQUISH_CONSTANT;
      }

      // One contribution is a permutation of (0,0,2).
      if (c2 & 0x01) {
        xsv_ext1 = xsb + 2;
        ysv_ext1 = ysb;
        zsv_ext1 = zsb;
        dx_ext1 = dx0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
        dy_ext1 = dy0 - (SQUISH_CONSTANT * (T)2.0);
        dz_ext1 = dz0 - (SQUISH_CONSTANT * (T)2.0);
      } else if (c2 & 0x02) {
        xsv_ext1 = xsb;
        ysv_ext1 = ysb + 2;
        zsv_ext1 = zsb;
        dx_ext1 = dx0 - (SQUISH_CONSTANT * (T)2.0);
        dy_ext1 = dy0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
        dz_ext1 = dz0 - (SQUISH_CONSTANT * (T)2.0);
      } else {
        xsv_ext1 = xsb;
        ysv_ext1 = ysb;
        zsv_ext1 = zsb + 2;
        dx_ext1 = dx0 - (SQUISH_CONSTANT * (T)2.0);
        dy_ext1 = dy0 - (SQUISH_CONSTANT * (T)2.0);
        dz_ext1 = dz0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
      }
    }

    contr_m[0] = contr_ext[0] = 0.0;

    // Contribution (0,0,1).
    T dx1 = dx0 - (T)1.0 - SQUISH_CONSTANT;
    T dy1 = dy0 - SQUISH_CONSTANT;
    T dz1 = dz0 - SQUISH_CONSTANT;
    contr_m[1] = pow2(dx1) + pow2(dy1) + pow2(dz1);
    contr_ext[1] = extrapolate(xsb + 1, ysb, zsb, dx1, dy1, dz1);

    // Contribution (0,1,0).
    T dx2 = dx0 - SQUISH_CONSTANT;
    T dy2 = dy0 - (T)1.0 - SQUISH_CONSTANT;
    T dz2 = dz1;
    contr_m[2] = pow2(dx2) + pow2(dy2) + pow2(dz2);
    contr_ext[2] = extrapolate(xsb, ysb + 1, zsb, dx2, dy2, dz2);

    // Contribution (1,0,0).
    T dx3 = dx2;
    T dy3 = dy1;
    T dz3 = dz0 - (T)1.0 - SQUISH_CONSTANT;
    contr_m[3] = pow2(dx3) + pow2(dy3) + pow2(dz3);
    contr_ext[3] = extrapolate(xsb, ysb, zsb + 1, dx3, dy3, dz3);

    // Contribution (1,1,0).
    T dx4 = dx0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
    T dy4 = dy0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
    T dz4 = dz0 - (SQUISH_CONSTANT * (T)2.0);
    contr_m[4] = pow2(dx4) + pow2(dy4) + pow2(dz4);
    contr_ext[4] = extrapolate(xsb + 1, ysb + 1, zsb, dx4, dy4, dz4);

    // Contribution (1,0,1).
    T dx5 = dx4;
    T dy5 = dy0 - (SQUISH_CONSTANT * (T)2.0);
    T dz5 = dz0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
    contr_m[5] = pow2(dx5) + pow2(dy5) + pow2(dz5);
    contr_ext[5] = extrapolate(xsb + 1, ysb, zsb + 1, dx5, dy5, dz5);

    // Contribution (0,1,1).
    T dx6 = dx0 - (SQUISH_CONSTANT * (T)2.0);
    T dy6 = dy4;
    T dz6 = dz5;
    contr_m[6] = pow2(dx6) + pow2(dy6) + pow2(dz6);
    contr_ext[6] = extrapolate(xsb, ysb + 1, zsb + 1, dx6, dy6, dz6);

  } else if (inSum <= (T)1.0) {
    // The point is inside the tetrahedron (3-Simplex) at (0,0,0)

    // Determine which of (0,0,1), (0,1,0), (1,0,0) are closest.
    uint_fast8_t aPoint = 1;
    T aScore = xins;
    uint_fast8_t bPoint = 2;
    T bScore = yins;
    if (aScore < bScore && zins > aScore) {
      aScore = zins;
      aPoint = 4;
    } else if (aScore >= bScore && zins > bScore) {
      bScore = zins;
      bPoint = 4;
    }

    // Determine the two lattice points not part of the tetrahedron that may contribute.
    // This depends on the closest two tetrahedral vertices, including (0,0,0).
    T wins = (T)1.0 - inSum;
    if (wins > aScore || wins > bScore) {
      // (0,0,0) is one of the closest two tetrahedral vertices.

      // The other closest vertex is the closer of a and b.
      uint_fast8_t c = ((bScore > aScore) ? bPoint : aPoint);

      if (c != 1) {
        xsv_ext0 = xsb - 1;
        xsv_ext1 = xsb;
        dx_ext0 = dx0 + (T)1.0;
        dx_ext1 = dx0;
      } else {
        xsv_ext0 = xsv_ext1 = xsb + 1;
        dx_ext0 = dx_ext1 = dx0 - (T)1.0;
      }

      if (c != 2) {
        ysv_ext0 = ysv_ext1 = ysb;
        dy_ext0 = dy_ext1 = dy0;
        if (c == 1) {
          ysv_ext0 -= 1;
          dy_ext0 += (T)1.0;
        } else {
          ysv_ext1 -= 1;
          dy_ext1 += (T)1.0;
        }
      } else {
        ysv_ext0 = ysv_ext1 = ysb + 1;
        dy_ext0 = dy_ext1 = dy0 - (T)1.0;
      }

      if (c != 4) {
        zsv_ext0 = zsb;
        zsv_ext1 = zsb - 1;
        dz_ext0 = dz0;
        dz_ext1 = dz0 + (T)1.0;
      } else {
        zsv_ext0 = zsv_ext1 = zsb + 1;
        dz_ext0 = dz_ext1 = dz0 - (T)1.0;
      }
    } else {
      // (0,0,0) is not one of the closest two tetrahedral vertices.

      // The two extra vertices are determined by the closest two.
      uint_fast8_t c = (aPoint | bPoint);

      if (c & 0x01) {
        xsv_ext0 = xsv_ext1 = xsb + 1;
        dx_ext0 = dx0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
        dx_ext1 = dx0 - (T)1.0 - SQUISH_CONSTANT;
      } else {
        xsv_ext0 = xsb;
        xsv_ext1 = xsb - 1;
        dx_ext0 = dx0 - (SQUISH_CONSTANT * (T)2.0);
        dx_ext1 = dx0 + (T)1.0 - SQUISH_CONSTANT;
      }

      if (c & 0x02) {
        ysv_ext0 = ysv_ext1 = ysb + 1;
        dy_ext0 = dy0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
        dy_ext1 = dy0 - (T)1.0 - SQUISH_CONSTANT;
      } else {
        ysv_ext0 = ysb;
        ysv_ext1 = ysb - 1;
        dy_ext0 = dy0 - (SQUISH_CONSTANT * (T)2.0);
        dy_ext1 = dy0 + (T)1.0 - SQUISH_CONSTANT;
      }

      if (c & 0x04) {
        zsv_ext0 = zsv_ext1 = zsb + 1;
        dz_ext0 = dz0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
        dz_ext1 = dz0 - (T)1.0 - SQUISH_CONSTANT;
      } else {
        zsv_ext0 = zsb;
        zsv_ext1 = zsb - 1;
        dz_ext0 = dz0 - (SQUISH_CONSTANT * (T)2.0);
        dz_ext1 = dz0 + (T)1.0 - SQUISH_CONSTANT;
      }
    }

    // Contribution (0,0,0)
    {
      contr_m[0] = pow2(dx0) + pow2(dy0) + pow2(dz0);
      contr_ext[0] = extrapolate(xsb, ysb, zsb, dx0, dy0, dz0);
    }

    // Contribution (0,0,1)
    T dx1 = dx0 - (T)1.0 - SQUISH_CONSTANT;
    T dy1 = dy0 - SQUISH_CONSTANT;
    T dz1 = dz0 - SQUISH_CONSTANT;
    contr_m[1] = pow2(dx1) + pow2(dy1) + pow2(dz1);
    contr_ext[1] = extrapolate(xsb + 1, ysb, zsb, dx1, dy1, dz1);

    // Contribution (0,1,0)
    T dx2 = dx0 - SQUISH_CONSTANT;
    T dy2 = dy0 - (T)1.0 - SQUISH_CONSTANT;
    T dz2 = dz1;
    contr_m[2] = pow2(dx2) + pow2(dy2) + pow2(dz2);
    contr_ext[2] = extrapolate(xsb, ysb + 1, zsb, dx2, dy2, dz2);

    // Contribution (1,0,0)
    T dx3 = dx2;
    T dy3 = dy1;
    T dz3 = dz0 - (T)1.0 - SQUISH_CONSTANT;
    contr_m[3] = pow2(dx3) + pow2(dy3) + pow2(dz3);
    contr_ext[3] = extrapolate(xsb, ysb, zsb + 1, dx3, dy3, dz3);

    contr_m[4] = contr_m[5] = contr_m[6] = 0.0;
    contr_ext[4] = contr_ext[5] = contr_ext[6] = 0.0;

  } else {
    // The point is inside the tetrahedron (3-Simplex) at (1,1,1)

    // Determine which two tetrahedral vertices are the closest
    // out of (1,1,0), (1,0,1), and (0,1,1), but not (1,1,1).
    uint_fast8_t aPoint = 6;
    T aScore = xins;
    uint_fast8_t bPoint = 5;
    T bScore = yins;
    if (aScore <= bScore && zins < bScore) {
      bScore = zins;
      bPoint = 3;
    } else if (aScore > bScore && zins < aScore) {
      aScore = zins;
      aPoint = 3;
    }

    // Determine the two lattice points not part of the tetrahedron that may contribute.
    // This depends on the closest two tetrahedral vertices, including (1,1,1).
    T wins = 3.0 - inSum;
    if (wins < aScore || wins < bScore) {
      // (1,1,1) is one of the closest two tetrahedral vertices.

      // The other closest vertex is the closest of a and b.
      uint_fast8_t c = ((bScore < aScore) ? bPoint : aPoint);

      if (c & 0x01) {
        xsv_ext0 = xsb + 2;
        xsv_ext1 = xsb + 1;
        dx_ext0 = dx0 - (T)2.0 - (SQUISH_CONSTANT * (T)3.0);
        dx_ext1 = dx0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
      } else {
        xsv_ext0 = xsv_ext1 = xsb;
        dx_ext0 = dx_ext1 = dx0 - (SQUISH_CONSTANT * (T)3.0);
      }

      if (c & 0x02) {
        ysv_ext0 = ysv_ext1 = ysb + 1;
        dy_ext0 = dy_ext1 = dy0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
        if (c & 0x01) {
          ysv_ext1 += 1;
          dy_ext1 -= (T)1.0;
        } else {
          ysv_ext0 += 1;
          dy_ext0 -= (T)1.0;
        }
      } else {
        ysv_ext0 = ysv_ext1 = ysb;
        dy_ext0 = dy_ext1 = dy0 - (SQUISH_CONSTANT * (T)3.0);
      }

      if (c & 0x04) {
        zsv_ext0 = zsb + 1;
        zsv_ext1 = zsb + 2;
        dz_ext0 = dz0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
        dz_ext1 = dz0 - (T)2.0 - (SQUISH_CONSTANT * (T)3.0);
      } else {
        zsv_ext0 = zsv_ext1 = zsb;
        dz_ext0 = dz_ext1 = dz0 - (SQUISH_CONSTANT * (T)3.0);
      }
    } else {
      // (1,1,1) is not one of the closest two tetrahedral vertices.

      // The two extra vertices are determined by the closest two.
      uint_fast8_t c = aPoint & bPoint;

      if (c & 0x01) {
        xsv_ext0 = xsb + 1;
        xsv_ext1 = xsb + 2;
        dx_ext0 = dx0 - (T)1.0 - SQUISH_CONSTANT;
        dx_ext1 = dx0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
      } else {
        xsv_ext0 = xsv_ext1 = xsb;
        dx_ext0 = dx0 - SQUISH_CONSTANT;
        dx_ext1 = dx0 - (SQUISH_CONSTANT * (T)2.0);
      }

      if (c & 0x02) {
        ysv_ext0 = ysb + 1;
        ysv_ext1 = ysb + 2;
        dy_ext0 = dy0 - (T)1.0 - SQUISH_CONSTANT;
        dy_ext1 = dy0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
      } else {
        ysv_ext0 = ysv_ext1 = ysb;
        dy_ext0 = dy0 - SQUISH_CONSTANT;
        dy_ext1 = dy0 - (SQUISH_CONSTANT * (T)2.0);
      }

      if (c & 0x04) {
        zsv_ext0 = zsb + 1;
        zsv_ext1 = zsb + 2;
        dz_ext0 = dz0 - (T)1.0 - SQUISH_CONSTANT;
        dz_ext1 = dz0 - (T)2.0 - (SQUISH_CONSTANT * (T)2.0);
      } else {
        zsv_ext0 = zsv_ext1 = zsb;
        dz_ext0 = dz0 - SQUISH_CONSTANT;
        dz_ext1 = dz0 - (SQUISH_CONSTANT * (T)2.0);
      }
    }

    // Contribution (1,1,0)
    T dx3 = dx0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
    T dy3 = dy0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
    T dz3 = dz0 - (SQUISH_CONSTANT * (T)2.0);
    contr_m[3] = pow2(dx3) + pow2(dy3) + pow2(dz3);
    contr_ext[3] = extrapolate(xsb + 1, ysb + 1, zsb, dx3, dy3, dz3);

    // Contribution (1,0,1)
    T dx2 = dx3;
    T dy2 = dy0 - (SQUISH_CONSTANT * (T)2.0);
    T dz2 = dz0 - (T)1.0 - (SQUISH_CONSTANT * (T)2.0);
    contr_m[2] = pow2(dx2) + pow2(dy2) + pow2(dz2);
    contr_ext[2] = extrapolate(xsb + 1, ysb, zsb + 1, dx2, dy2, dz2);

    // Contribution (0,1,1)
    {
      T dx1 = dx0 - (SQUISH_CONSTANT * (T)2.0);
      T dy1 = dy3;
      T dz1 = dz2;
      contr_m[1] = pow2(dx1) + pow2(dy1) + pow2(dz1);
      contr_ext[1] = extrapolate(xsb, ysb + 1, zsb + 1, dx1, dy1, dz1);
    }

    // Contribution (1,1,1)
    {
      dx0 = dx0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
      dy0 = dy0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
      dz0 = dz0 - (T)1.0 - (SQUISH_CONSTANT * (T)3.0);
      contr_m[0] = pow2(dx0) + pow2(dy0) + pow2(dz0);
      contr_ext[0] = extrapolate(xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0);
    }

    contr_m[4] = contr_m[5] = contr_m[6] = 0.0;
    contr_ext[4] = contr_ext[5] = contr_ext[6] = 0.0;

  }

  // First extra vertex.
  contr_m[7] = pow2(dx_ext0) + pow2(dy_ext0) + pow2(dz_ext0);
  contr_ext[7] = extrapolate(xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0);

  // Second extra vertex.
  contr_m[8] = pow2(dx_ext1) + pow2(dy_ext1) + pow2(dz_ext1);
  contr_ext[8] = extrapolate(xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1);

  T value = 0.0;
  for (int i=0; i<9; ++i) {
    value += pow4(std::max((T)2.0 - contr_m[i], (T)0.0)) * contr_ext[i];
  }

  return (value * NORM_CONSTANT);
}

template OPENVDB_AX_API double OSNoise::extrapolate(const OSNoise::inttype xsb, const OSNoise::inttype ysb, const OSNoise::inttype zsb,
                                     const double dx, const double dy, const double dz) const;
template OPENVDB_AX_API double OSNoise::extrapolate(const OSNoise::inttype xsb, const OSNoise::inttype ysb, const OSNoise::inttype zsb,
                                     const double dx, const double dy, const double dz,
                                     double (&de) [3]) const;

template OPENVDB_AX_API double OSNoise::eval(const double x, const double y, const double z) const;

} // namespace OSN
