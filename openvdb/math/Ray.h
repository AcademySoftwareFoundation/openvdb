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
//
/// @file Ray.h
///
/// @author Ken Museth
///
/// @brief A Ray class and a Digital Differential Analyzer specialized for VDB.

#ifndef OPENVDB_MATH_RAY_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_RAY_HAS_BEEN_INCLUDED

#include "Math.h"
#include "Vec3.h"
#include "Transform.h"
#include <iostream> // for std::ostream


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename RealT>
class Ray
{
public:
    typedef RealT      RealType;
    typedef Vec3<Real> Vec3Type;
    typedef Vec3Type   Vec3T;

    Ray(const Vec3Type& eye, const Vec3Type& direction,
        RealT t0 = 0, RealT t1 = std::numeric_limits<RealT>::max())
        : mEye(eye), mDir(direction), mInvDir(1/mDir), mT0(t0), mT1(t1)
    {
    }

    void setEye(const Vec3Type& eye) { mEye = eye; }

    void setDir(const Vec3Type& direction) { mDir = direction; mInvDir = 1/mDir; }

    void setTime(RealT t0, RealT t1) { mT0 = t0; mT1 = t1; }
    
    void reset(const Vec3Type& eye, const Vec3Type& direction,
               RealT t0 = 0, RealT t1 = std::numeric_limits<RealT>::max())
    {
        this->setEye(eye);
        this->setDir(direction);
        this->setTime(t0, t1);
    }

    const Vec3T& eye() const {return mEye;}
    const Vec3T& dir() const {return mDir;}
    const Vec3T& invDir() const {return mInvDir;}
    RealT t0() const {return mT0;}
    RealT t1() const {return mT1;}

    /// @brief Return the position along the ray at the specified time.
    Vec3R operator()(RealT time) const { return mEye + mDir * time; }
    /// @brief Return the starting point of the ray.
    Vec3R start() const { return (*this)(mT0); }
    /// @brief Return the endpoint of the ray.
    Vec3R end() const { return (*this)(mT1); }
    /// @brief Return @c true if @a time is within t0 and t1, both inclusive.
    bool test(RealT time) const { return (time>=mT0 && time<=mT1); }
    /// @brief Return a new Ray that is transformed with the specified map.
    /// @param map  the map from which to construct the new Ray.
    /// @note This assumes a linear map.
    template<typename MapType>
    Ray applyMap(const MapType& map) const
    {
        assert(map.isLinear());
        const Vec3T eye = map.applyMap(mEye);
        const Vec3T dir = map.applyMap(mDir) - map.applyMap(Vec3T(0,0,0));
        const RealT tmp = Sqrt(mDir.lengthSqr()/dir.lengthSqr());
        return Ray(eye, dir, mT0*tmp, mT1*tmp);
    }

    /// @brief Return a new Ray that is transformed with the inverse of the specified map.
    /// @param map  the map from which to construct the new Ray by inverse mapping.
    /// @note This assumes a linear map.
    template<typename MapType>
    Ray applyInverseMap(const MapType& map) const
    {
        assert(map.isLinear());
        const Vec3T eye = map.applyInverseMap(mEye);
        const Vec3T dir = map.applyInverseMap(mDir) - map.applyInverseMap(Vec3T(0,0,0));
        const RealT tmp = Sqrt(mDir.lengthSqr()/dir.lengthSqr());
        return Ray(eye, dir, mT0*tmp, mT1*tmp);
    }

    /// @brief Return true if this ray intersects the specified sphere.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    /// @param t0     The first intersection point if an intersection exists.
    /// @param t1     The second intersection point if an intersection exists.
    bool intersects(const Vec3T& center, RealT radius, RealT& t0, RealT& t1) const
    {
        const Vec3T origin = mEye - center;
        const RealT A = mDir.lengthSqr();
        const RealT B = 2 * mDir.dot(origin);
        const RealT C = origin.lengthSqr() - radius * radius;
        const RealT D = B * B - 4 * A * C;   

        if (D < 0) return false;

        const RealT Q = RealT(-0.5)*(B<0 ? (B + Sqrt(D)) : (B - Sqrt(D)));

        t0 = Q / A;
        t1 = C / Q;
        
        if (t0 > t1) std::swap(t0, t1);
        if (t0 < mT0) t0 = mT0;
        if (t1 > mT1) t1 = mT1;
        return t0 <= t1;
    }
    
    /// @brief Return true if this ray intersects the specified sphere.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    bool intersects(const Vec3T& center, RealT radius) const
    {
        RealT t0, t1;
        return this->intersects(center, radius, t0, t1);
    }

    /// @brief Return true if this ray intersects the specified sphere. 
    /// @note For intersection this ray is clipped to the two intersection points.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    bool clip(const Vec3T& center, RealT radius)
    {
        RealT t0, t1;
        const bool hit = this->intersects(center, radius, t0, t1);
        if (hit) {
            mT0 = t0;
            mT1 = t1;
        }
        return hit;
    }

    /// @brief Return true if the Ray intersects the specified
    /// axisaligned bounding box.
    /// @param bbox Axis-aligned bounding box in the same space as the Ray.
    /// @param t0   If an intersection is detected this is assigned
    ///             the time for the first intersection point.
    /// @param t1   If an intersection is detected this is assigned
    ///             the time for the second intersection point.
    template<typename BBoxT>
    bool intersects(const BBoxT& bbox, RealT& t0, RealT& t1) const
    {
        t0 = mT0;
        t1 = mT1;
        for (size_t i = 0; i < 3; ++i) {
            RealT a = (bbox.min()[i] - mEye[i]) * mInvDir[i];
            RealT b = (bbox.max()[i] - mEye[i]) * mInvDir[i];
            if (a > b) std::swap(a, b);
            if (a > t0) t0 = a;
            if (b < t1) t1 = b;
            if (t0 > t1) return false;
        }
        return true;
    }

    /// @brief Return true if this ray intersects the specified bounding box.
    /// @param bbox Axis-aligned bounding box in the same space as this ray.
    template<typename BBoxT>
    bool intersects(const BBoxT& bbox) const
    {
        RealT t0, t1;
        return this->intersects(bbox, t0, t1);
    }

    /// @brief Return true if this ray intersects the specified bounding box. 
    /// @note For intersection this ray is clipped to the two intersection points.
    /// @param bbox Axis-aligned bounding box in the same space as this ray.
    template<typename BBoxT>
    bool clip(const BBoxT& bbox)
    {
        RealT t0, t1;
        const bool hit = this->intersects(bbox, t0, t1);
        if (hit) {
            mT0 = t0;
            mT1 = t1;
        }
        return hit;
    }

private:
    Vec3T mEye, mDir, mInvDir;
    RealT mT0, mT1;
}; // end of Ray class


/// @brief Output streaming of the Ray class.
/// @note Primarily intended for debugging.
template<typename RealT>
inline std::ostream& operator<<(std::ostream& os, const Ray<RealT>& r)
{
    os << "eye=" << r.eye() << " dir=" << r.dir()
       << " t0=" << r.t0()  << " t0="  << r.t1();
    return os;
}


////////////////////////////////////////


/// @brief A Digital Differential Analyzer specialized for OpenVDB grids
/// @note Conceptually similar to Bresenham's line algorithm applied
/// to a 3D Ray intersecting OpenVDB nodes or voxels. Log2Dim = 0
/// corresponds to a voxel and Log2Dim a tree node of size 2^Log2Dim.     
///
/// @note The Ray template class is expected to have the following
/// four methods: test(time), t0(), invDir(), and  operator()(time).
/// See the example Ray class above for their definition.
template<typename RayT, Index Log2Dim = 0>
class DDA
{
public:
    typedef typename RayT::RealType RealType;
    typedef RealType                RealT;
    typedef typename RayT::Vec3Type Vec3Type;
    typedef Vec3Type                Vec3T;

    DDA(const RayT& ray) { this->init(ray, ray.t0()); }

    DDA(const RayT& ray, RealT time) { this->init(ray, time); }

    void init(const RayT& ray, RealT time)
    {
        static const int DIM = 1 << Log2Dim;
        mTime = time;
        assert(ray.test(time));
        const Vec3T &pos = ray(time), &inv = ray.invDir();
        mVoxel = Coord::floor(pos) & (~(DIM-1));
        for (size_t axis = 0; axis < 3; ++axis) {
            mStep[axis]  = inv[axis] > 0 ? DIM : -DIM;
            mNext[axis]  = time + (mVoxel[axis] + mStep[axis] - pos[axis]) * inv[axis];
            mDelta[axis] = mStep[axis] * inv[axis];
        }
    }

    /// Increment the voxel index to next intersected voxel or node and return the
    /// corresponding intersection point paramerterized in time along the initializing ray.
    RealType step()
    {
        const size_t stepAxis = math::MinIndex(mNext);
        mTime = mNext[stepAxis];
        mNext[stepAxis]  += mDelta[stepAxis];
        mVoxel[stepAxis] += mStep[stepAxis];
        return mTime;
    }

    /// @brief Return the index coordinates of the next node or voxel
    /// intersected by the ray. If Log2Dim = 0 the return value is the
    /// actual signed coordinate of the voxel, else it is the origin
    /// of the corresponding VDB tree node.
    /// @note Incurs no computational overhead.
    const Coord& voxel() const { return mVoxel; }

    /// @brief Return the time parameterization along the initial Ray.
    /// @note Incurs no computational overhead.
    RealType time() const { return mTime; }

private:
    RealT mTime;
    Coord mVoxel, mStep;
    Vec3T mDelta, mNext;
}; // class DDA

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_RAY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
