// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_DISTANCEMEASURE_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_DISTANCEMEASURE_HAS_BEEN_INCLUDED

#include "../Utilities/Distance.h"

#include <openvdb/tools/VolumeToSpheres.h>

#include <vector>


/* OpenVDBGrid public member function list

mma::IntVectorRef gridMember(mma::IntCoordinatesRef pts)

mma::RealCoordinatesRef gridNearest(mma::RealCoordinatesRef pts)

mma::RealVectorRef gridDistance(mma::RealCoordinatesRef pts)

mma::RealVectorRef gridSignedDistance(mma::RealCoordinatesRef pts)

mma::RealMatrixRef fillWithBalls(mint bmin, mint bmax, bool overlapping, float rmin, float rmax,
    float isovalue, mint instanceCount)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
mma::IntVectorRef
openvdbmma::OpenVDBGrid<V>::gridMember(mma::IntCoordinatesRef pts, double isovalue) const
{
    scalar_type_assert<V>();

    openvdbmma::distance::DistanceMeasurementsMma<wlGridType> dist_func(grid(), isovalue);

    return dist_func.gridMember(pts);
}

template<typename V>
mma::RealCoordinatesRef
openvdbmma::OpenVDBGrid<V>::gridNearest(mma::RealCoordinatesRef pts, double isovalue) const
{
    scalar_type_assert<V>();

    openvdbmma::distance::DistanceMeasurementsMma<wlGridType> dist_func(grid(), isovalue);

    return dist_func.gridNearest(pts);
}

template<typename V>
mma::RealVectorRef
openvdbmma::OpenVDBGrid<V>::gridDistance(mma::RealCoordinatesRef pts, double isovalue) const
{
    scalar_type_assert<V>();

    openvdbmma::distance::DistanceMeasurementsMma<wlGridType> dist_func(grid(), isovalue);

    return dist_func.gridDistance(pts);
}

template<typename V>
mma::RealVectorRef
openvdbmma::OpenVDBGrid<V>::gridSignedDistance(mma::RealCoordinatesRef pts, double isovalue) const
{
    scalar_type_assert<V>();

    openvdbmma::distance::DistanceMeasurementsMma<wlGridType> dist_func(grid(), isovalue);

    return dist_func.gridSignedDistance(pts);
}

template<typename V>
mma::RealMatrixRef
openvdbmma::OpenVDBGrid<V>::fillWithBalls(mint bmin, mint bmax, bool overlapping,
    float rmin, float rmax, float isovalue, mint instanceCount) const
{
    scalar_type_assert<V>();

    if (bmax <= 0 || bmax < bmin)
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    if (rmax < rmin)
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    if (instanceCount <= 0)
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    std::vector<openvdb::Vec4s> spheres;
    const Vec2i sphereCount(bmin, bmax);

    mma::interrupt::LLInterrupter interrupt;

    openvdb::tools::fillWithSpheres(*grid(), spheres, sphereCount, overlapping,
        rmin, rmax, isovalue, instanceCount, &interrupt);

    mma::RealMatrixRef balldata = mma::makeMatrix<double>(spheres.size(), 4);

    for (mint i = 0; i < spheres.size(); ++i) {
        balldata(i, 0) = spheres[i][0];
        balldata(i, 1) = spheres[i][1];
        balldata(i, 2) = spheres[i][2];
        balldata(i, 3) = spheres[i][3];
    }

    return balldata;
}

#endif // OPENVDBLINK_OPENVDBGRID_DISTANCEMEASURE_HAS_BEEN_INCLUDED
