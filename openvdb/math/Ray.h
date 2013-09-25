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

template<typename RealT = double>
class Ray
{
public:
    typedef RealT      RealType;
    typedef Vec3<Real> Vec3Type;
    typedef Vec3Type   Vec3T;

    Ray(const Vec3Type& eye = Vec3Type(0,0,0),
        const Vec3Type& direction = Vec3Type(1,0,0),
        RealT t0 = 1e-3,
        RealT t1 = std::numeric_limits<RealT>::max())
        : mEye(eye), mDir(direction), mInvDir(1/mDir), mT0(t0), mT1(t1)
    {
    }

    inline void setEye(const Vec3Type& eye) { mEye = eye; }

    inline void setDir(const Vec3Type& dir)
      {
          mDir = dir;
          mInvDir = 1/mDir;
      }

    inline void setMinTime(RealT t0) { assert(t0>0); mT0 = t0; }

    inline void setMaxTime(RealT t1) { assert(t1>0); mT1 = t1; }

    inline void setTimes(RealT t0, RealT t1) { assert(t0>0 && t1>0);mT0 = t0; mT1 = t1; }

    inline void scaleTimes(RealT scale) {  assert(scale>0); mT0 *= scale; mT1 *= scale; }
    
    inline void reset(const Vec3Type& eye, const Vec3Type& direction,
                      RealT t0 = 0, RealT t1 = std::numeric_limits<RealT>::max())
    {
        this->setEye(eye);
        this->setDir(direction);
        this->setTimes(t0, t1);
    }

    inline const Vec3T& eye() const {return mEye;}

    inline const Vec3T& dir() const {return mDir;}

    inline const Vec3T& invDir() const {return mInvDir;}

    inline RealT t0() const {return mT0;}

    inline RealT t1() const {return mT1;}

    /// @brief Return the position along the ray at the specified time.
    inline Vec3R operator()(RealT time) const { return mEye + mDir * time; }

    /// @brief Return the starting point of the ray.
    inline Vec3R start() const { return (*this)(mT0); }

    /// @brief Return the endpoint of the ray.
    inline Vec3R end() const { return (*this)(mT1); }

    /// @brief Return the midpoint of the ray.
    inline Vec3R mid() const { return (*this)(0.5*(mT0+mT1)); }

     /// @brief Return @c true if t0 is strictly less then t1.
    inline bool test() const { return (mT0 < mT1); }
    
    /// @brief Return @c true if @a time is within t0 and t1, both inclusive.
    inline bool test(RealT time) const { return (time>=mT0 && time<=mT1); }

    /// @brief Return a new Ray that is transformed with the specified map.
    /// @param map  the map from which to construct the new Ray.
    /// @warning Assumes a linear map and a normalize direction.
    /// @details The requirement that the direction is normalized
    /// follows from the transformation of t0 and t1 - and that fact that
    /// we want applyMap and applyInverseMap to be inverse operations.
    template<typename MapType>
    inline Ray applyMap(const MapType& map) const
    {
        assert(map.isLinear());
        assert(math::isApproxEqual(mDir.length(), RealT(1)));
        const Vec3T eye = map.applyMap(mEye);
        const Vec3T dir = map.applyJacobian(mDir);
        const RealT length = dir.length();
        return Ray(eye, dir/length, length*mT0, length*mT1);
    }

    /// @brief Return a new Ray that is transformed with the inverse of the specified map.
    /// @param map  the map from which to construct the new Ray by inverse mapping.
    /// @warning Assumes a linear map and a normalize direction.
    /// @details The requirement that the direction is normalized
    /// follows from the transformation of t0 and t1 - and that fact that
    /// we want applyMap and applyInverseMap to be inverse operations.
    template<typename MapType>
    inline Ray applyInverseMap(const MapType& map) const
    {
        assert(map.isLinear());
        assert(math::isApproxEqual(mDir.length(), RealT(1)));
        const Vec3T eye = map.applyInverseMap(mEye);
        const Vec3T dir = map.applyInverseJacobian(mDir);
        const RealT length = dir.length();
        return Ray(eye, dir/length, length*mT0, length*mT1);
    }

    /// @brief Return a new ray in world space, assuming the existing
    /// ray is represented in the index space of the specified grid.
    template<typename GridType>
    inline Ray indexToWorld(const GridType& grid) const
    {
        return this->applyMap(*(grid.transform().baseMap()));
    }

    /// @brief Return a new ray in the index space of the specified
    /// grid, assuming the existing ray is represented in world space. 
    template<typename GridType>
    inline Ray worldToIndex(const GridType& grid) const
    {
        return this->applyInverseMap(*(grid.transform().baseMap()));
    }
    
    /// @brief Return true if this ray intersects the specified sphere.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    /// @param t0     The first intersection point if an intersection exists.
    /// @param t1     The second intersection point if an intersection exists.
    /// @note If the return value is true, i.e. a hit, and t0 =
    /// this->t0() or t1 == this->t1() only one true intersection exist.
    inline bool intersects(const Vec3T& center, RealT radius, RealT& t0, RealT& t1) const
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
    inline bool intersects(const Vec3T& center, RealT radius) const
    {
        RealT t0, t1;
        return this->intersects(center, radius, t0, t1)>0;
    }

    /// @brief Return true if this ray intersects the specified sphere. 
    /// @note For intersection this ray is clipped to the two intersection points.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    inline bool clip(const Vec3T& center, RealT radius)
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
    inline bool intersects(const BBoxT& bbox, RealT& t0, RealT& t1) const
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
    inline bool intersects(const BBoxT& bbox) const
    {
        RealT t0, t1;
        return this->intersects(bbox, t0, t1);
    }

    /// @brief Return true if this ray intersects the specified bounding box. 
    /// @note For intersection this ray is clipped to the two intersection points.
    /// @param bbox Axis-aligned bounding box in the same space as this ray.
    template<typename BBoxT>
    inline bool clip(const BBoxT& bbox)
    {
        RealT t0, t1;
        const bool hit = this->intersects(bbox, t0, t1);
        if (hit) {
            mT0 = t0;
            mT1 = t1;
        }
        return hit;
    }

    /// @brief Return true if the Ray intersects the plane specified
    /// by a normal and distance from the origin.
    /// @param normal   Normal of the plane.
    /// @param distance Distance of the plane to the origin.
    /// @param t        Time of intersection, if one exists.
    inline bool intersects(const Vec3T& normal, RealT distance, RealT& t) const
      {
          const RealT cosAngle = mDir.dot(normal);
          if (math::isApproxZero(cosAngle)) return false;//parallel
          t = (distance - mEye.dot(normal))/cosAngle;
          return this->test(t);
      }

    /// @brief Return true if the Ray intersects the plane specified
    /// by a normal and point.
    /// @param normal   Normal of the plane.
    /// @param point    Point in the plane.
    /// @param t        Time of intersection, if one exists.
    inline bool intersects(const Vec3T& normal, const Vec3T& point, RealT& t) const
      {
          return this->intersects(normal, point.dot(normal), t);
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
    os << "eye=" << r.eye() << " dir=" << r.dir() << " 1/dir="<<r.invDir()
       << " t0=" << r.t0()  << " t1="  << r.t1();
    return os;
}


////////////////////////////////////////


/// @brief A Digital Differential Analyzer specialized for OpenVDB grids
/// @note Conceptually similar to Bresenham's line algorithm applied
/// to a 3D Ray intersecting OpenVDB nodes or voxels. Log2Dim = 0
/// corresponds to a voxel and Log2Dim a tree node of size 2^Log2Dim.     
///
/// @note The Ray template class is expected to have the following
/// methods: test(time), t0(), t1(), invDir(), and  operator()(time).
/// See the example Ray class above for their definition.
template<typename RayT, Index Log2Dim = 0>
class DDA
{
public:
    typedef typename RayT::RealType RealType;
    typedef RealType                RealT;
    typedef typename RayT::Vec3Type Vec3Type;
    typedef Vec3Type                Vec3T;

    DDA(const RayT& ray) { this->init(ray, ray.t0(), ray.t1()); }

    DDA(const RayT& ray, RealT startTime) { this->init(ray, startTime, ray.t1()); }

    DDA(const RayT& ray, RealT startTime, RealT maxTime) { this->init(ray, startTime, maxTime); }
    
    inline void init(const RayT& ray, RealT startTime, RealT maxTime)
    {
        assert(startTime <= maxTime);
        static const int DIM = 1 << Log2Dim;
        mT0 = startTime;
        mT1 = maxTime;
        const Vec3T &pos = ray(mT0), &dir = ray.dir(), &inv = ray.invDir();
        mVoxel = Coord::floor(pos) & (~(DIM-1));
        for (size_t axis = 0; axis < 3; ++axis) {
            if (math::isZero(dir[axis])) {//handles dir = +/- 0
                mStep[axis]  = 0;//dummy value
                mNext[axis]  = std::numeric_limits<RealT>::max();//i.e. disabled!
                mDelta[axis] = std::numeric_limits<RealT>::max();//dummy value
            } else if (inv[axis] > 0) {
                mStep[axis]  = DIM;
                mNext[axis]  = mT0 + (mVoxel[axis] + DIM - pos[axis]) * inv[axis];
                mDelta[axis] = mStep[axis] * inv[axis];
            } else {
                mStep[axis]  = -DIM;
                mNext[axis]  = mT0 + (mVoxel[axis] - pos[axis]) * inv[axis];
                mDelta[axis] = mStep[axis] * inv[axis];
            }
        }
    }

    /// @brief Increment the voxel index to next intersected voxel or node
    /// and returns true if the step in time does not exceed maxTime.
    inline bool step()
    {
        const size_t stepAxis = math::MinIndex(mNext);
        mT0 = mNext[stepAxis];
        mNext[stepAxis]  += mDelta[stepAxis];
        mVoxel[stepAxis] += mStep[stepAxis];
        return mT0 <= mT1;
    }

    /// @brief Return the index coordinates of the next node or voxel
    /// intersected by the ray. If Log2Dim = 0 the return value is the
    /// actual signed coordinate of the voxel, else it is the origin
    /// of the corresponding VDB tree node or tile.
    /// @note Incurs no computational overhead.
    inline const Coord& voxel() const { return mVoxel; }

    /// @brief Return the time (parameterized along the Ray) of the
    /// first hit of a tree node of size 2^Log2Dim.
    /// @details This value is initialized to startTime or ray.t0()
    /// depending on the constructor used.
    /// @note Incurs no computational overhead.
    inline RealType time() const { return mT0; }

    /// @brief Return the time (parameterized along the Ray) of the
    /// second (i.e. next) hit of a tree node of size 2^Log2Dim.
    /// @note Incurs a (small) computational overhead.
    inline RealType next() const { return math::Min(mT1, mNext[0], mNext[1], mNext[2]); }

    /// @brief Print information about this DDA for debugging.
    /// @param os    a stream to which to write textual information.
    void print(std::ostream& os = std::cout) const
      {
          os << "Dim=" << (1<<Log2Dim) << " time=" << mT0 << " next()="
             << this->next() << " voxel=" << mVoxel << " next=" << mNext
             << " delta=" << mDelta << " step=" << mStep << std::endl;
      }

private:
    RealT mT0, mT1;
    Coord mVoxel, mStep;
    Vec3T mDelta, mNext;
}; // class DDA

/// @brief Output streaming of the Ray class.
/// @note Primarily intended for debugging.
template<typename RayT, Index Log2Dim>
inline std::ostream& operator<<(std::ostream& os, const DDA<RayT, Log2Dim>& dda)
{
    os << "Dim="     << (1<<Log2Dim) << " time="  << dda.time()
       << " next()=" << dda.next()   << " voxel=" << dda.voxel();
    return os;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_RAY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
