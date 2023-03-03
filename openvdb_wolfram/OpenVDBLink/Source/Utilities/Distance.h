// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_DISTANCE_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_DISTANCE_HAS_BEEN_INCLUDED

#include <openvdb/tools/VolumeToSpheres.h>


/* openvdbmma::distance members

 class DistanceMeasurementsMma

 public members are

 gridMember, which can operate in index or world coordinate inputs
 gridNearest
 gridDistance
 gridSignedDistance

*/


namespace openvdbmma {
namespace distance {

//////////// distance measurements class

template<typename GridT>
class DistanceMeasurementsMma
{
public:

    using GridPtr = typename GridT::Ptr;

    DistanceMeasurementsMma(GridPtr grid, float isovalue)
    : mGrid(grid), mIsovalue(isovalue)
    {
    }

    ~DistanceMeasurementsMma() {}

    mma::IntVectorRef gridMember(mma::IntCoordinatesRef pts) const;
    mma::IntVectorRef gridMember(mma::RealCoordinatesRef pts) const;

    mma::RealCoordinatesRef gridNearest(mma::RealCoordinatesRef pts) const;

    mma::RealVectorRef gridDistance(mma::RealCoordinatesRef pts) const;

    mma::RealVectorRef gridSignedDistance(mma::RealCoordinatesRef pts) const;

private:

    void nearestAndDistance(mma::RealCoordinatesRef pts,
        std::vector<Vec3R>& vpts, std::vector<float>& dists) const
    {
        using CSP = typename openvdb::tools::ClosestSurfacePoint<GridT>::Ptr;

        mma::interrupt::LLInterrupter interrupt;

        CSP csp = openvdb::tools::ClosestSurfacePoint<GridT>::create(*mGrid, mIsovalue, &interrupt);

        const mint len = pts.size();
        vpts.resize(len);
        dists.resize(len);

        mma::check_abort();

        tbb::parallel_for(
            tbb::blocked_range<mint>(0, len),
            [&](tbb::blocked_range<mint> rng)
            {
                for (mint i = rng.begin(); i < rng.end(); ++i) {
                    vpts[i] = Vec3R(pts.x(i), pts.y(i), pts.z(i));
                }
            }
        );

        csp->searchAndReplace(vpts, dists);
    }

    //////////// private members

    float mIsovalue;

    GridPtr mGrid;

}; // end of DistanceMeasurementsMma class


//////////// DistanceMeasurementsMma public member function definitions

template<typename GridT>
inline mma::IntVectorRef
DistanceMeasurementsMma<GridT>::gridMember(mma::IntCoordinatesRef pts) const
{
    using AccT = typename GridT::Accessor;

    const AccT accessor = mGrid->getAccessor();

    const mint len = pts.size();

    mma::IntVectorRef mem = mma::makeVector<mint>(len);

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, len),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i) {
                const Coord xyz(pts.x(i), pts.y(i), pts.z(i));
                mem[i] = accessor.getValue(xyz) <= mIsovalue ? 1 : 0;
            }
        }
    );

    return mem;
}

template<typename GridT>
inline mma::IntVectorRef
DistanceMeasurementsMma<GridT>::gridMember(mma::RealCoordinatesRef pts) const
{
    const float dx = (mGrid->voxelSize())[0];
    const mint len = pts.size();

    mma::IntCoordinatesRef intpts = mma::makeCoordinatesList<mint>(len);

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, 3*len),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i) {
                intpts[i] = math::Round(pts[i]/dx);
            }
        }
    );

    mma::IntVectorRef mems = gridMember(intpts);

    intpts.free();

    return mems;
}

template<typename GridT>
inline mma::RealCoordinatesRef
DistanceMeasurementsMma<GridT>::gridNearest(mma::RealCoordinatesRef pts) const
{
    std::vector<Vec3R> vpts;
    std::vector<float> dists;

    nearestAndDistance(pts, vpts, dists);

    const mint len = pts.size();
    mma::RealCoordinatesRef npts = mma::makeCoordinatesList<double>(len);

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, len),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i) {
                npts[3*i] = vpts[i][0];
                npts[3*i+1] = vpts[i][1];
                npts[3*i+2] = vpts[i][2];
            }
        }
    );

    return npts;
}

template<typename GridT>
inline mma::RealVectorRef
DistanceMeasurementsMma<GridT>::gridDistance(mma::RealCoordinatesRef pts) const
{
    std::vector<Vec3R> vpts;
    std::vector<float> dists;

    nearestAndDistance(pts, vpts, dists);

    const mint len = pts.size();
    mma::RealVectorRef ndists = mma::makeVector<double>(len);

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, len),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i)
                ndists[i] = dists[i];
        }
    );

    return ndists;
}

template<typename GridT>
inline mma::RealVectorRef
DistanceMeasurementsMma<GridT>::gridSignedDistance(mma::RealCoordinatesRef pts) const
{
    const mma::IntVectorRef mem = gridMember(pts);
    mma::RealVectorRef dists = gridDistance(pts);

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, dists.size()),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i)
                if (mem[i] != 0)
                    dists[i] *= -1;
        }
    );

    mem.free();

    return dists;
}

} // namespace distance
} // namespace openvdbmma

#endif // OPENVDBLINK_UTILITIES_DISTANCE_HAS_BEEN_INCLUDED
