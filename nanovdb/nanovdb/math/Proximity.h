// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   Proximity.h

    \author Efty Sifakis

    \brief  Closest-point queries on geometric primitives,
            suitable for use in both host and device code.
*/

#ifndef NANOVDB_MATH_PROXIMITY_H_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_PROXIMITY_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Util.h>  // for __hostdev__
#include <nanovdb/math/Math.h>  // for nanovdb::math::Sqrt

namespace nanovdb {

namespace math {

/// @brief Returns the closest point on the line segment [v0,v1] to @a p.
///
///        The result lies on the closed segment (t clamped to [0,1]).
///
/// @tparam Vec3T  Any Vec3 type supporting arithmetic and dot().
template<typename Vec3T>
__hostdev__ inline Vec3T
closestPointOnSegmentToPoint(const Vec3T &v0, const Vec3T &v1, const Vec3T &p)
{
    const Vec3T seg = v1 - v0;
    const Vec3T w   = p  - v0;

    const auto c1 = seg.dot(w);
    if (c1 <= 0) return v0;

    const auto c2 = seg.dot(seg);
    if (c2 <= c1) return v1;

    return v0 + (c1 / c2) * seg;
}

/// @brief Returns the closest point on triangle [v0,v1,v2] to @a p,
///        and (via @a t0, @a t1) the barycentric coordinates of that
///        point: closest = v0 + t0*(v1-v0) + t1*(v2-v0).
///
///        Uses Voronoi-region decomposition (Ericson, "Real-Time Collision
///        Detection", Sec. 5.1.5). Degenerate triangles (zero area, collinear or
///        coincident vertices) are handled implicitly: the face-interior test
///        naturally fails and the code falls through to the nearest edge or
///        vertex result.
///
/// @tparam Vec3T  Any Vec3 type supporting arithmetic and dot().
template<typename Vec3T>
__hostdev__ inline Vec3T
closestPointOnTriangleToPoint(
    const Vec3T &v0,
    const Vec3T &v1,
    const Vec3T &v2,
    const Vec3T &p,
    typename Vec3T::ValueType &t0,
    typename Vec3T::ValueType &t1)
{
    using RealT = typename Vec3T::ValueType;

    const Vec3T ab = v1 - v0;
    const Vec3T ac = v2 - v0;
    const Vec3T ap = p  - v0;

    // --- Region A (vertex v0) ---
    const RealT d1 = ab.dot(ap);
    const RealT d2 = ac.dot(ap);
    if (d1 <= RealT(0) && d2 <= RealT(0)) {
        t0 = RealT(0);
        t1 = RealT(0);
        return v0;
    }

    // --- Region B (vertex v1) ---
    const Vec3T bp = p - v1;
    const RealT d3 = ab.dot(bp);
    const RealT d4 = ac.dot(bp);
    if (d3 >= RealT(0) && d4 <= d3) {
        t0 = RealT(1);
        t1 = RealT(0);
        return v1;
    }

    // --- Region AB (edge v0-v1) ---
    const RealT vc = d1 * d4 - d3 * d2;
    if (vc <= RealT(0) && d1 >= RealT(0) && d3 <= RealT(0)) {
        t0 = d1 / (d1 - d3);
        t1 = RealT(0);
        return v0 + t0 * ab;
    }

    // --- Region C (vertex v2) ---
    const Vec3T cp = p - v2;
    const RealT d5 = ab.dot(cp);
    const RealT d6 = ac.dot(cp);
    if (d6 >= RealT(0) && d5 <= d6) {
        t0 = RealT(0);
        t1 = RealT(1);
        return v2;
    }

    // --- Region AC (edge v0-v2) ---
    const RealT vb = d5 * d2 - d1 * d6;
    if (vb <= RealT(0) && d2 >= RealT(0) && d6 <= RealT(0)) {
        t1 = d2 / (d2 - d6);
        t0 = RealT(0);
        return v0 + t1 * ac;
    }

    // --- Region BC (edge v1-v2) ---
    const RealT va = d3 * d6 - d5 * d4;
    if (va <= RealT(0) && (d4 - d3) >= RealT(0) && (d5 - d6) >= RealT(0)) {
        t1 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        t0 = RealT(1) - t1;
        return v1 + t1 * (v2 - v1);
    }

    // --- Interior of triangle ---
    const RealT denom = RealT(1) / (va + vb + vc);
    t0 = vb * denom;
    t1 = vc * denom;
    return v0 + t0 * ab + t1 * ac;
}

/// @brief Returns the squared distance from @a p to the closest point
///        on triangle [v0,v1,v2].
template<typename Vec3T>
__hostdev__ inline typename Vec3T::ValueType
pointToTriangleDistSqr(
    const Vec3T &v0,
    const Vec3T &v1,
    const Vec3T &v2,
    const Vec3T &p)
{
    typename Vec3T::ValueType t0, t1;
    const Vec3T closest = closestPointOnTriangleToPoint(v0, v1, v2, p, t0, t1);
    return (p - closest).lengthSqr();
}

} // namespace math

} // namespace nanovdb

#endif // NANOVDB_MATH_PROXIMITY_H_HAS_BEEN_INCLUDED
