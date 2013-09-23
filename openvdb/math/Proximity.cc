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

#include "Proximity.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


OPENVDB_API Vec3d
closestPointOnTriangleToPoint(
    const Vec3d& a, const Vec3d& b, const Vec3d& c, const Vec3d& p, Vec3d& uvw)
{
    uvw.setZero();

    // degenerate triangle, singular
    if ((isApproxEqual(a, b) && isApproxEqual(a, c))) {
        uvw[0] = 1.0;
        return a; 
    }

    Vec3d ab = b - a, ac = c - a, ap = p - a;
    double d1 = ab.dot(ap), d2 = ac.dot(ap);

    // degenerate triangle edges
    if (isApproxEqual(a, b)) {

        double t = 0.0;
        Vec3d cp = closestPointOnSegmentToPoint(a, c, p, t);
        
        uvw[0] = 1.0 - t;
        uvw[2] = t;

        return cp;

    } else if (isApproxEqual(a, c) || isApproxEqual(b, c)) {

        double t = 0.0;
        Vec3d cp = closestPointOnSegmentToPoint(a, b, p, t);
        uvw[0] = 1.0 - t;
        uvw[1] = t;
        return cp;
    }

    if (d1 <= 0.0 && d2 <= 0.0) {
        uvw[0] = 1.0;
        return a; // barycentric coordinates (1,0,0)
    }

    // Check if P in vertex region outside B
    Vec3d bp = p - b;
    double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) {
        uvw[1] = 1.0;
        return b; // barycentric coordinates (0,1,0)
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        uvw[1] = d1 / (d1 - d3);
        uvw[0] = 1.0 - uvw[1];
        return a + uvw[1] * ab; // barycentric coordinates (1-v,v,0) 
    }

    // Check if P in vertex region outside C
    Vec3d cp = p - c;
    double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) {
        uvw[2] = 1.0;
        return c; // barycentric coordinates (0,0,1)
    }

    // Check if P in edge region of AC, if so return projection of P onto AC
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        uvw[2] = d2 / (d2 - d6);
        uvw[0] = 1.0 - uvw[2];
        return a + uvw[2] * ac; // barycentric coordinates (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    double va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        uvw[2] = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        uvw[1] = 1.0 - uvw[2];
        return b + uvw[2] * (c - b); // barycentric coordinates (0,1-w,w)    
    }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    double denom = 1.0 / (va + vb + vc);
    uvw[2] = vc * denom;
    uvw[1] = vb * denom;
    uvw[0] = 1.0 - uvw[1] - uvw[2];

    return a + ab*uvw[1] + ac*uvw[2]; // = u*a + v*b + w*c , u= va*denom = 1.0-v-w 
}


OPENVDB_API Vec3d
closestPointOnSegmentToPoint(const Vec3d& a, const Vec3d& b, const Vec3d& p, double& t)
{
    Vec3d ab = b - a;
    t = (p - a).dot(ab);

    if (t <= 0.0) {
        // c projects outside the [a,b] interval, on the a side.
        t = 0.0;
        return a;
    } else {

        // always nonnegative since denom = ||ab||^2
        double denom = ab.dot(ab);

        if (t >= denom) {
            // c projects outside the [a,b] interval, on the b side.
            t = 1.0;
            return b;
        } else {
            // c projects inside the [a,b] interval.
            t = t / denom;
            return a + (ab * t); 
        }
    }
}

////////////////////////////////////////


// DEPRECATED METHODS

double 
sLineSeg3ToPointDistSqr(const Vec3d &p0, 
                        const Vec3d &p1, 
                        const Vec3d &point, 
                        double &t, 
                        double  epsilon)
{
    Vec3d  pDelta;
    Vec3d  tDelta;
    double pDeltaDot;
    
    pDelta.sub(p1, p0); 
    tDelta.sub(point, p0); 

    //
    // Line is nearly a point check end points
    //
    pDeltaDot = pDelta.dot(pDelta);
    if (pDeltaDot < epsilon) {
        pDelta.sub(p1, point);
        if (pDelta.dot(pDelta) < tDelta.dot(tDelta)) {
            t = 1;
            return pDelta.dot(pDelta);
        } else {
            t = 0;
            return tDelta.dot(tDelta);
        }
    } 
    t = tDelta.dot(pDelta) / pDeltaDot;
    if (t < 0) {
        t = 0;
    } else if (t > 1) {
        t = 1;
        tDelta.sub(point, p1); 
    } else {
        tDelta -= t * pDelta;
    }
    return tDelta.dot(tDelta);    
}


////////////////////////////////////////


double
sTri3ToPointDistSqr(const Vec3d &v0,
                    const Vec3d &v1,
                    const Vec3d &v2,
                    const Vec3d &point,
                          Vec2d &uv,
                          double)
{
    Vec3d e0, e1;
    double distSqr;
    
    e0.sub(v1, v0);
    e1.sub(v2, v0);
    
    Vec3d  delta = v0 - point;
    double a00   = e0.dot(e0);
    double a01   = e0.dot(e1);
    double a11   = e1.dot(e1);
    double b0    = delta.dot(e0);
    double b1    = delta.dot(e1);
    double c     = delta.dot(delta);
    double det   = fabs(a00*a11-a01*a01);
    /* DEPRECATED
    double aMax  = (a00 > a11) ? a00 : a11;
    double epsilon2 = epsilon * epsilon;

    //
    // Triangle is degenerate. Use an absolute test for the length
    // of the edges and a relative test for area squared
    //
    if ((a00 <= epsilon2 && a11 <= epsilon2) || det <= epsilon * aMax * aMax) {

        double t;
        double minDistSqr;
        
        minDistSqr = sLineSeg3ToPointDistSqr(v0, v1, point, t, epsilon);
        uv[0] = 1.0 - t;
        uv[1] = t;

        distSqr = sLineSeg3ToPointDistSqr(v0, v2, point, t, epsilon);
        if (distSqr < minDistSqr) {
            minDistSqr = distSqr;
            uv[0] = 1.0 - t;
            uv[1] = 0;
        }

        distSqr = sLineSeg3ToPointDistSqr(v1, v2, point, t, epsilon);
        if (distSqr < minDistSqr) {
            minDistSqr = distSqr;
            uv[0] = 0;
            uv[1] = 1.0 - t;
        }

        return minDistSqr;
    }*/

    double s = a01*b1-a11*b0;
    double t = a01*b0-a00*b1;

    if (s + t <= det ) {
        if (s < 0.0) {
            if (t < 0.0) { 
                // region 4
                if (b0 < 0.0) {
                    t = 0.0;
                    if (-b0 >= a00) {
                        s = 1.0;
                        distSqr = a00+2.0*b0+c;
                    } else {
                        s = -b0/a00;
                        distSqr = b0*s+c;
                    }
                } else {
                    s = 0.0;
                    if (b1 >= 0.0) {
                        t = 0.0;
                        distSqr = c;
                    } else if (-b1 >= a11) {
                        t = 1.0;
                        distSqr = a11+2.0*b1+c;
                    } else {
                        t = -b1/a11;
                        distSqr = b1*t+c;
                    }
                }
            } else  { 
                // region 3  
                s = 0.0;
                if (b1 >= 0.0) {
                    t = 0.0;
                    distSqr = c;
                }
                else if (-b1 >= a11) {
                    t = 1.0;
                    distSqr = a11+2.0*b1+c;
                }
                else {
                    t = -b1/a11;
                    distSqr = b1*t+c;
                }
            }
        } else if (t < 0.0)  {
            // region 5        

            t = 0.0;
            if (b0 >= 0.0) {
                s = 0.0;
                distSqr = c;
            } else if (-b0 >= a00) {
                s = 1.0;
                distSqr = a00+2.0*b0+c;
            } else {
                s = -b0/a00;
                distSqr = b0*s+c;
            }
        } else { 
            // region 0

            // minimum at interior point
            double fInvDet = 1.0/det;
            s *= fInvDet;
            t *= fInvDet;
            distSqr = s*(a00*s+a01*t+2.0*b0) +
                      t*(a01*s+a11*t+2.0*b1)+c;
        }
    } else {
        double tmp0, tmp1, numer, denom;

        if (s < 0.0)  { 
            // region 2 

            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0) {
                numer = tmp1 - tmp0;
                denom = a00-2.0*a01+a11;
                if (numer >= denom) {
                    s = 1.0;
                    t = 0.0;
                    distSqr = a00+2.0*b0+c;
                } else {
                    s = numer/denom;
                    t = 1.0 - s;
                    distSqr = s*(a00*s+a01*t+2.0*b0) +
                              t*(a01*s+a11*t+2.0*b1)+c;
                }
            } else {
                s = 0.0;
                if (tmp1 <= 0.0) {
                    t = 1.0;
                    distSqr = a11+2.0*b1+c;
                } else if (b1 >= 0.0) {
                    t = 0.0;
                    distSqr = c;
                } else {
                    t = -b1/a11;
                    distSqr = b1*t+c;
                }
            }
        } else if (t < 0.0)  { 
            // region 6

            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0 ) {
                numer = tmp1 - tmp0;
                denom = a00-2.0*a01+a11;
                if (numer >= denom ) {
                    t = 1.0;
                    s = 0.0;
                    distSqr = a11+2.0*b1+c;
                } else {
                    t = numer/denom;
                    s = 1.0 - t;
                    distSqr = s*(a00*s+a01*t+2.0*b0) +
                              t*(a01*s+a11*t+2.0*b1)+c;
                }
            } else {
                t = 0.0;
                if (tmp1 <= 0.0) {
                    s = 1.0;
                    distSqr = a00+2.0*b0+c;
                } else if (b0 >= 0.0) {
                    s = 0.0;
                    distSqr = c;
                } else {
                    s = -b0/a00;
                    distSqr = b0*s+c;
                }
            }
        } else { 
            // region 1
            numer = a11 + b1 - a01 - b0;
            if (numer <= 0.0) {
                s = 0.0;
                t = 1.0;
                distSqr = a11+2.0*b1+c;
            } else {
                denom = a00-2.0*a01+a11;
                if (numer >= denom ) {
                    s = 1.0;
                    t = 0.0;
                    distSqr = a00+2.0*b0+c;
                } else {
                    s = numer/denom;
                    t = 1.0 - s;
                    distSqr = s*(a00*s+a01*t+2.0*b0) +
                              t*(a01*s+a11*t+2.0*b1)+c;
                }
            }
        }
    }

    // Convert s,t into barycentric coordinates
    uv[0] = 1.0 - s - t;
    uv[1] = s;

    return (distSqr < 0) ? 0.0 : distSqr;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
