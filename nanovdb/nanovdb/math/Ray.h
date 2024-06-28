// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Ray.h
///
/// @author Ken Museth
///
/// @brief A Ray class.

#ifndef NANOVDB_MATH_RAY_H_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_RAY_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h> // for Vec3
namespace nanovdb {// ===================================================

namespace math {// ======================================================

template<typename RealT>
class Ray
{
public:
    using RealType = RealT;
    using Vec3Type = Vec3<RealT>;
    using Vec3T = Vec3Type;

    struct TimeSpan
    {
        RealT t0, t1;
        /// @brief Default constructor
        __hostdev__ TimeSpan() {}
        /// @brief Constructor
        __hostdev__ TimeSpan(RealT _t0, RealT _t1)
            : t0(_t0)
            , t1(_t1)
        {
        }
        /// @brief Set both times
        __hostdev__ void set(RealT _t0, RealT _t1)
        {
            t0 = _t0;
            t1 = _t1;
        }
        /// @brief Get both times
        __hostdev__ void get(RealT& _t0, RealT& _t1) const
        {
            _t0 = t0;
            _t1 = t1;
        }
        /// @brief Return @c true if t1 is larger than t0 by at least eps.
        __hostdev__ bool valid(RealT eps = Delta<RealT>::value()) const { return (t1 - t0) > eps; }
        /// @brief Return the midpoint of the ray.
        __hostdev__ RealT mid() const { return 0.5 * (t0 + t1); }
        /// @brief Multiplies both times
        __hostdev__ void scale(RealT s)
        {
            assert(s > 0);
            t0 *= s;
            t1 *= s;
        }
        /// @brief Return @c true if time is inclusive
        __hostdev__ bool test(RealT t) const { return (t >= t0 && t <= t1); }
    };

    __hostdev__ Ray(const Vec3Type& eye = Vec3Type(0, 0, 0),
                    const Vec3Type& direction = Vec3Type(1, 0, 0),
                    RealT           t0 = Delta<RealT>::value(),
                    RealT           t1 = Maximum<RealT>::value())
        : mEye(eye)
        , mDir(direction)
        , mInvDir(1 / mDir[0], 1 / mDir[1], 1 / mDir[2])
        , mTimeSpan(t0, t1)
        , mSign{mInvDir[0] < 0, mInvDir[1] < 0, mInvDir[2] < 0}
    {
    }

    __hostdev__ Ray& offsetEye(RealT offset)
    {
        mEye[0] += offset;
        mEye[1] += offset;
        mEye[2] += offset;
        return *this;
    }

    __hostdev__ Ray& setEye(const Vec3Type& eye)
    {
        mEye = eye;
        return *this;
    }

    __hostdev__ Ray& setDir(const Vec3Type& dir)
    {
        mDir = dir;
        mInvDir[0] = 1.0 / mDir[0];
        mInvDir[1] = 1.0 / mDir[1];
        mInvDir[2] = 1.0 / mDir[2];
        mSign[0] = mInvDir[0] < 0;
        mSign[1] = mInvDir[1] < 0;
        mSign[2] = mInvDir[2] < 0;
        return *this;
    }

    __hostdev__ Ray& setMinTime(RealT t0)
    {
        mTimeSpan.t0 = t0;
        return *this;
    }

    __hostdev__ Ray& setMaxTime(RealT t1)
    {
        mTimeSpan.t1 = t1;
        return *this;
    }

    __hostdev__ Ray& setTimes(
        RealT t0 = Delta<RealT>::value(),
        RealT t1 = Maximum<RealT>::value())
    {
        assert(t0 > 0 && t1 > 0);
        mTimeSpan.set(t0, t1);
        return *this;
    }

    __hostdev__ Ray& scaleTimes(RealT scale)
    {
        mTimeSpan.scale(scale);
        return *this;
    }

    __hostdev__ Ray& reset(
        const Vec3Type& eye,
        const Vec3Type& direction,
        RealT           t0 = Delta<RealT>::value(),
        RealT           t1 = Maximum<RealT>::value())
    {
        this->setEye(eye);
        this->setDir(direction);
        this->setTimes(t0, t1);
        return *this;
    }

    __hostdev__ const Vec3T& eye() const { return mEye; }

    __hostdev__ const Vec3T& dir() const { return mDir; }

    __hostdev__ const Vec3T& invDir() const { return mInvDir; }

    __hostdev__ RealT t0() const { return mTimeSpan.t0; }

    __hostdev__ RealT t1() const { return mTimeSpan.t1; }

    __hostdev__ int sign(int i) const { return mSign[i]; }

    /// @brief Return the position along the ray at the specified time.
    __hostdev__ Vec3T operator()(RealT time) const
    {
#if 1
        return Vec3T(fmaf(time, mDir[0], mEye[0]),
                     fmaf(time, mDir[1], mEye[1]),
                     fmaf(time, mDir[2], mEye[2]));
#else
        return mEye + mDir * time;
#endif
    }

    /// @brief Return the starting point of the ray.
    __hostdev__ Vec3T start() const { return (*this)(mTimeSpan.t0); }

    /// @brief Return the endpoint of the ray.
    __hostdev__ Vec3T end() const { return (*this)(mTimeSpan.t1); }

    /// @brief Return the midpoint of the ray.
    __hostdev__ Vec3T mid() const { return (*this)(mTimeSpan.mid()); }

    /// @brief Return @c true if t1 is larger than t0 by at least eps.
    __hostdev__ bool valid(RealT eps = Delta<float>::value()) const { return mTimeSpan.valid(eps); }

    /// @brief Return @c true if @a time is within t0 and t1, both inclusive.
    __hostdev__ bool test(RealT time) const { return mTimeSpan.test(time); }

    /// @brief Return a new Ray that is transformed with the specified map.
    ///
    /// @param map  the map from which to construct the new Ray.
    ///
    /// @warning Assumes a linear map and a normalized direction.
    ///
    /// @details The requirement that the direction is normalized
    ///          follows from the transformation of t0 and t1 - and that fact that
    ///          we want applyMap and applyInverseMap to be inverse operations.
    template<typename MapType>
    __hostdev__ Ray applyMap(const MapType& map) const
    {
        const Vec3T eye = map.applyMap(mEye);
        const Vec3T dir = map.applyJacobian(mDir);
        const RealT length = dir.length(), invLength = RealT(1) / length;
        RealT       t1 = mTimeSpan.t1;
        if (mTimeSpan.t1 < Maximum<RealT>::value()) {
            t1 *= length;
        }
        return Ray(eye, dir * invLength, length * mTimeSpan.t0, t1);
    }
    template<typename MapType>
    __hostdev__ Ray applyMapF(const MapType& map) const
    {
        const Vec3T eye = map.applyMapF(mEye);
        const Vec3T dir = map.applyJacobianF(mDir);
        const RealT length = dir.length(), invLength = RealT(1) / length;
        RealT       t1 = mTimeSpan.t1;
        if (mTimeSpan.t1 < Maximum<RealT>::value()) {
            t1 *= length;
        }
        return Ray(eye, dir * invLength, length * mTimeSpan.t0, t1);
    }

    /// @brief Return a new Ray that is transformed with the inverse of the specified map.
    ///
    /// @param map  the map from which to construct the new Ray by inverse mapping.
    ///
    /// @warning Assumes a linear map and a normalized direction.
    ///
    /// @details The requirement that the direction is normalized
    ///          follows from the transformation of t0 and t1 - and that fact that
    ///          we want applyMap and applyInverseMap to be inverse operations.
    template<typename MapType>
    __hostdev__ Ray applyInverseMap(const MapType& map) const
    {
        const Vec3T eye = map.applyInverseMap(mEye);
        const Vec3T dir = map.applyInverseJacobian(mDir);
        const RealT length = dir.length(), invLength = RealT(1) / length;
        return Ray(eye, dir * invLength, length * mTimeSpan.t0, length * mTimeSpan.t1);
    }
    template<typename MapType>
    __hostdev__ Ray applyInverseMapF(const MapType& map) const
    {
        const Vec3T eye = map.applyInverseMapF(mEye);
        const Vec3T dir = map.applyInverseJacobianF(mDir);
        const RealT length = dir.length(), invLength = RealT(1) / length;
        return Ray(eye, dir * invLength, length * mTimeSpan.t0, length * mTimeSpan.t1);
    }

    /// @brief Return a new ray in world space, assuming the existing
    ///        ray is represented in the index space of the specified grid.
    template<typename GridType>
    __hostdev__ Ray indexToWorldF(const GridType& grid) const
    {
        const Vec3T eye = grid.indexToWorldF(mEye);
        const Vec3T dir = grid.indexToWorldDirF(mDir);
        const RealT length = dir.length(), invLength = RealT(1) / length;
        RealT       t1 = mTimeSpan.t1;
        if (mTimeSpan.t1 < Maximum<RealT>::value()) {
            t1 *= length;
        }
        return Ray(eye, dir * invLength, length * mTimeSpan.t0, t1);
    }

    /// @brief Return a new ray in index space, assuming the existing
    ///        ray is represented in the world space of the specified grid.
    template<typename GridType>
    __hostdev__ Ray worldToIndexF(const GridType& grid) const
    {
        const Vec3T eye = grid.worldToIndexF(mEye);
        const Vec3T dir = grid.worldToIndexDirF(mDir);
        const RealT length = dir.length(), invLength = RealT(1) / length;
        RealT       t1 = mTimeSpan.t1;
        if (mTimeSpan.t1 < Maximum<RealT>::value()) {
            t1 *= length;
        }
        return Ray(eye, dir * invLength, length * mTimeSpan.t0, t1);
    }

    /// @brief Return true if this ray intersects the specified sphere.
    ///
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    /// @param t0     The first intersection point if an intersection exists.
    /// @param t1     The second intersection point if an intersection exists.
    ///
    /// @note If the return value is true, i.e. a hit, and t0 =
    ///       this->t0() or t1 == this->t1() only one true intersection exist.
    __hostdev__ bool intersects(const Vec3T& center, RealT radius, RealT& t0, RealT& t1) const
    {
        const Vec3T origin = mEye - center;
        const RealT A = mDir.lengthSqr();
        const RealT B = 2 * mDir.dot(origin);
        const RealT C = origin.lengthSqr() - radius * radius;
        const RealT D = B * B - 4 * A * C;

        if (D < 0) {
            return false;
        }
        const RealT Q = RealT(-0.5) * (B < 0 ? (B + Sqrt(D)) : (B - Sqrt(D)));

        t0 = Q / A;
        t1 = C / Q;

        if (t0 > t1) {
            RealT tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        if (t0 < mTimeSpan.t0) {
            t0 = mTimeSpan.t0;
        }
        if (t1 > mTimeSpan.t1) {
            t1 = mTimeSpan.t1;
        }
        return t0 <= t1;
    }

    /// @brief Return true if this ray intersects the specified sphere.
    ///
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    __hostdev__ bool intersects(const Vec3T& center, RealT radius) const
    {
        RealT t0, t1;
        return this->intersects(center, radius, t0, t1) > 0;
    }

    /// @brief Return true if this ray intersects the specified sphere.
    ///
    /// @note For intersection this ray is clipped to the two intersection points.
    ///
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    __hostdev__ bool clip(const Vec3T& center, RealT radius)
    {
        RealT      t0, t1;
        const bool hit = this->intersects(center, radius, t0, t1);
        if (hit) {
            mTimeSpan.set(t0, t1);
        }
        return hit;
    }
#if 0
    /// @brief Return true if the Ray intersects the specified
    ///        axisaligned bounding box.
    ///
    /// @param bbox Axis-aligned bounding box in the same space as the Ray.
    /// @param t0   If an intersection is detected this is assigned
    ///             the time for the first intersection point.
    /// @param t1   If an intersection is detected this is assigned
    ///             the time for the second intersection point.
    template<typename BBoxT>
    __hostdev__  bool intersects(const BBoxT& bbox, RealT& t0, RealT& t1) const
    {
        t0       = (bbox[  mSign[0]][0] - mEye[0]) * mInvDir[0];
        RealT t2 = (bbox[1-mSign[1]][1] - mEye[1]) * mInvDir[1];
        if (t0 > t2) return false;
        t1       = (bbox[1-mSign[0]][0] - mEye[0]) * mInvDir[0];
        RealT t3 = (bbox[  mSign[1]][1] - mEye[1]) * mInvDir[1];
        if (t3 > t1) return false;
        if (t3 > t0) t0 = t3;
        if (t2 < t1) t1 = t2;
        t3 = (bbox[  mSign[2]][2] - mEye[2]) * mInvDir[2];
        if (t3 > t1) return false;
        t2 = (bbox[1-mSign[2]][2] - mEye[2]) * mInvDir[2];
        if (t0 > t2) return false;
        if (t3 > t0) t0 = t3;
        if (mTimeSpan.t1 < t0) return false;
        if (t2 < t1) t1 = t2;
        if (mTimeSpan.t0 > t1) return false;
        if (mTimeSpan.t0 > t0) t0 = mTimeSpan.t0;
        if (mTimeSpan.t1 < t1) t1 = mTimeSpan.t1;
        return true;
        /*
        mTimeSpan.get(_t0, _t1);
        double t0 = _t0, t1 = _t1;
        for (int i = 0; i < 3; ++i) {
            //if (abs(mDir[i])<1e-3) continue;
            double a = (double(bbox.min()[i]) - mEye[i]) * mInvDir[i];
            double b = (double(bbox.max()[i]) - mEye[i]) * mInvDir[i];
            if (a > b) {
                double tmp = a;
                a = b;
                b = tmp;
            }
            if (a > t0) t0 = a;
            if (b < t1) t1 = b;
            if (t0 > t1) {
                //if (gVerbose) printf("Missed BBOX: (%i,%i,%i) -> (%i,%i,%i) t0=%f t1=%f\n",
                //                     bbox.min()[0], bbox.min()[1], bbox.min()[2],
                //                     bbox.max()[0], bbox.max()[1], bbox.max()[2], t0, t1);
                return false;
            }
        }
        _t0 = t0; _t1 = t1;
        return true;
        */
    }
#else
    /// @brief Returns true if this ray intersects an index bounding box.
    ///        If the return value is true t0 and t1 are set to the intersection
    ///        times along the ray.
    ///
    /// @warning Intersection with a CoordBBox internally converts to a floating-point bbox
    ///          which imples that the max is padded with one voxel, i.e. bbox.max += 1! This
    ///          avoids gaps between neighboring CoordBBox'es, say from neighboring tree nodes.
    __hostdev__ bool intersects(const CoordBBox& bbox, RealT& t0, RealT& t1) const
    {
        mTimeSpan.get(t0, t1);
        for (int i = 0; i < 3; ++i) {
            RealT a = RealT(bbox.min()[i]), b = RealT(bbox.max()[i] + 1);
            if (a >= b) { // empty bounding box
                return false;
            }
            a = (a - mEye[i]) * mInvDir[i];
            b = (b - mEye[i]) * mInvDir[i];
            if (a > b) {
                RealT tmp = a;
                a = b;
                b = tmp;
            }
            if (a > t0) {
                t0 = a;
            }
            if (b < t1) {
                t1 = b;
            }
            if (t0 > t1) {
                return false;
            }
        }
        return true;
    }
    /// @brief Returns true if this ray intersects a floating-point bounding box.
    ///        If the return value is true t0 and t1 are set to the intersection
    ///        times along the ray.
    template<typename OtherVec3T>
    __hostdev__ bool intersects(const BBox<OtherVec3T>& bbox, RealT& t0, RealT& t1) const
    {
        static_assert(util::is_floating_point<typename OtherVec3T::ValueType>::value, "Ray::intersects: Expected a floating point coordinate");
        mTimeSpan.get(t0, t1);
        for (int i = 0; i < 3; ++i) {
            RealT a = RealT(bbox.min()[i]), b = RealT(bbox.max()[i]);
            if (a >= b) { // empty bounding box
                return false;
            }
            a = (a - mEye[i]) * mInvDir[i];
            b = (b - mEye[i]) * mInvDir[i];
            if (a > b) {
                RealT tmp = a;
                a = b;
                b = tmp;
            }
            if (a > t0) {
                t0 = a;
            }
            if (b < t1) {
                t1 = b;
            }
            if (t0 > t1) {
                return false;
            }
        }
        return true;
    }
#endif

    /// @brief Return true if this ray intersects the specified bounding box.
    ///
    /// @param bbox Axis-aligned bounding box in the same space as this ray.
    ///
    /// @warning If @a bbox is of the type CoordBBox it is converted to a floating-point
    ///          bounding box, which imples that the max is padded with one voxel, i.e.
    ///          bbox.max += 1! This avoids gaps between neighboring CoordBBox'es, say
    ///          from neighboring tree nodes.
    template<typename BBoxT>
    __hostdev__ bool intersects(const BBoxT& bbox) const
    {
#if 1
        RealT t0, t1;
        return this->intersects(bbox, t0, t1);
#else
        //BBox<Vec3T> bbox(Vec3T(_bbox[0][0]-1e-4,_bbox[0][1]-1e-4,_bbox[0][2]-1e-4),
        //                 Vec3T(_bbox[1][0]+1e-4,_bbox[1][1]+1e-4,_bbox[1][2]+1e-4));
        RealT t0 = (bbox[mSign[0]][0] - mEye[0]) * mInvDir[0];
        RealT t2 = (bbox[1 - mSign[1]][1] - mEye[1]) * mInvDir[1];
        if (t0 > t2) return false;
        RealT t1 = (bbox[1 - mSign[0]][0] - mEye[0]) * mInvDir[0];
        RealT t3 = (bbox[mSign[1]][1] - mEye[1]) * mInvDir[1];
        if (t3 > t1) return false;
        if (t3 > t0) t0 = t3;
        if (t2 < t1) t1 = t2;
        t3 = (bbox[mSign[2]][2] - mEye[2]) * mInvDir[2];
        if (t3 > t1) return false;
        t2 = (bbox[1 - mSign[2]][2] - mEye[2]) * mInvDir[2];
        if (t0 > t2) return false;
        //if (t3 > t0) t0 = t3;
        //if (mTimeSpan.t1 < t0) return false;
        //if (t2 < t1) t1 = t2;
        //return mTimeSpan.t0 < t1;
        return true;
#endif
    }

    /// @brief Return true if this ray intersects the specified bounding box.
    ///
    /// @param bbox Axis-aligned bounding box in the same space as this ray.
    ///
    /// @warning If @a bbox is of the type CoordBBox it is converted to a floating-point
    ///          bounding box, which imples that the max is padded with one voxel, i.e.
    ///          bbox.max += 1! This avoids gaps between neighboring CoordBBox'es, say
    ///          from neighboring tree nodes.
    ///
    /// @note For intersection this ray is clipped to the two intersection points.
    template<typename BBoxT>
    __hostdev__ bool clip(const BBoxT& bbox)
    {
        RealT      t0, t1;
        const bool hit = this->intersects(bbox, t0, t1);
        if (hit) {
            mTimeSpan.set(t0, t1);
        }
        return hit;
    }

    /// @brief Return true if the Ray intersects the plane specified
    ///        by a normal and distance from the origin.
    ///
    /// @param normal   Normal of the plane.
    /// @param distance Distance of the plane to the origin.
    /// @param t        Time of intersection, if one exists.
    __hostdev__ bool intersects(const Vec3T& normal, RealT distance, RealT& t) const
    {
        const RealT cosAngle = mDir.dot(normal);
        if (isApproxZero(cosAngle)) {
            return false; // ray is parallel to plane
        }
        t = (distance - mEye.dot(normal)) / cosAngle;
        return this->test(t);
    }

    /// @brief Return true if the Ray intersects the plane specified
    ///        by a normal and point.
    ///
    /// @param normal   Normal of the plane.
    /// @param point    Point in the plane.
    /// @param t        Time of intersection, if one exists.
    __hostdev__ bool intersects(const Vec3T& normal, const Vec3T& point, RealT& t) const
    {
        return this->intersects(normal, point.dot(normal), t);
    }

private:
    Vec3T    mEye, mDir, mInvDir;
    TimeSpan mTimeSpan;
    int      mSign[3];
}; // end of Ray class

} // namespace math =========================================================

template<typename RealT>
using Ray [[deprecated("Use nanovdb::math::Ray instead")]] = math::Ray<RealT>;

} // namespace nanovdb =======================================================

#endif // NANOVDB_MATH_RAY_HAS_BEEN_INCLUDED
