///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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

#ifndef OPENVDB_MATH_PROXIMITY_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_PROXIMITY_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
//#include <openvdb/openvdb.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


/// @brief  Closest Point on Triangle to Point. Given a triangle @c abc and a
///         point @c p, returns the point on @c abc closest to @c p and the 
///         corresponding barycentric coordinates.
///
/// @note   Algorithms from "Real-Time Collision Detection" pg 136 to 142 by Christer Ericson.
///         The closest point is obtained by first determining which of the triangles
///         Voronoi feature regions @c p is in and then computing the orthogonal projection
///         of @c p onto the corresponding feature.
///
/// @param a    The triangle's first vertex point.
/// @param b    The triangle's second vertex point.
/// @param c    The triangle's third vertex point.
/// @param p    Point to compute the closest point on @c abc for. 
/// @param uvw  Barycentric coordinates, computed and returned.
OPENVDB_API Vec3d
closestPointOnTriangleToPoint(
    const Vec3d& a, const Vec3d& b, const Vec3d& c, const Vec3d& p, Vec3d& uvw);


/// @brief  Closest Point on Line Segment to Point. Given segment @c ab and
///         point @c p, returns the point on @c ab closest to @c p and @c t the 
///         parametric distance to @c b.
///
/// @param a    The segments's first vertex point.
/// @param b    The segments's second vertex point.
/// @param p    Point to compute the closest point on @c ab for.
/// @param t    Parametric distance to @c b.
OPENVDB_API Vec3d
closestPointOnSegmentToPoint(
    const Vec3d& a, const Vec3d& b, const Vec3d& p, double& t);


////////////////////////////////////////


// DEPRECATED METHODS


/// @brief Squared distance of a line segment p(t) = (1-t)*p0 + t*p1 to point. 
/// @return the closest point on the line segment as a function of t
OPENVDB_API OPENVDB_DEPRECATED double 
sLineSeg3ToPointDistSqr(const Vec3d &p0, 
                        const Vec3d &p1, 
                        const Vec3d &point, 
                        double &t, 
                        double  epsilon = 1e-10);


/// @brief Slightly modified version of the algorithm described in "Geometric Tools for
/// Computer Graphics" pg 376 to 382 by Schneider and Eberly. Extended to handle
/// the case of a degenerate triangle. Also returns barycentric rather than
/// (s,t) coordinates. 
/// 
/// Basic Idea (See book for details): 
///
/// Write the equation of the line as 
///
///     T(s,t) = v0 + s*(v1-v0) + t*(v2-v0)
///
/// Minimize the quadratic function 
///
///     || T(s,t) - point || ^2 
///
/// by solving for when the gradient is 0. This can be done without any 
/// square roots. 
///
/// If the resulting solution satisfies 0 <= s + t <= 1, then the solution lies
/// on the interior of the triangle, and we are done (region 0). If it does 
/// not then the closest solution lies on a boundary and we have to solve for 
/// it by solving a 1D problem where we use one variable as free say "s" and
/// set the other variable t = (1-s) 
///
/// @return the closest point on the triangle and barycentric coordinates.
OPENVDB_API OPENVDB_DEPRECATED double
sTri3ToPointDistSqr(const Vec3d &v0,
                    const Vec3d &v1,
                    const Vec3d &v2,
                    const Vec3d &point,
                          Vec2d &uv,
                          double epsilon);


/// @return the closest point on the triangle.
static inline OPENVDB_DEPRECATED double
triToPtnDistSqr(const Vec3d &v0,
                const Vec3d &v1,
                const Vec3d &v2,
                const Vec3d &point)
{
    Vec3d cpt, uvw;
    cpt = closestPointOnTriangleToPoint(v0, v1, v2, point, uvw);
    return (cpt - point).lengthSqr();
}


} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MESH_TO_VOLUME_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
