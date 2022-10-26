// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_LEVELSETCREATION_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_LEVELSETCREATION_HAS_BEEN_INCLUDED

#include "../Utilities/LevelSet.h"

#include <openvdb/tools/LevelSetSphere.h>


/* OpenVDBGrid public member function list

void ballLevelSet(mma::RealVectorRef center, double radius,
    double spacing, double bandWidth, bool is_signed = true)

void cuboidLevelSet(mma::RealBounds3DRef bounds,
    double spacing, double bandWidth, bool is_signed = true)

void meshLevelSet(mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
    double spacing, double bandWidth, bool is_signed = true)

void offsetSurfaceLevelSet(mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
    double offset, double spacing, double width, bool is_signed = true)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::ballLevelSet(mma::RealVectorRef center, double radius,
    double spacing, double bandWidth, bool is_signed)
{
    scalar_type_assert<V>();

    using AbsF = openvdbmma::levelset::AbsOp<ValueT>;

    if (center.size() != 3)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    wlGridPtr grid = openvdb::tools::createLevelSetSphere<wlGridType>(
        radius, Vec3f(center[0], center[1], center[2]), spacing, bandWidth);

    if(!is_signed) {
        mma::check_abort();
        transformActiveLeafValues<wlTreeType, AbsF>(grid->tree(), AbsF());
    }

    setGrid(grid);
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::cuboidLevelSet(
    mma::RealBounds3DRef bounds, double spacing, double bandWidth, bool is_signed)
{
    scalar_type_assert<V>();

    using AbsF = openvdbmma::levelset::AbsOp<ValueT>;

    const math::BBox<Vec3f> bbox(
        Vec3f(bounds.xmin(), bounds.ymin(), bounds.zmin()),
        Vec3f(bounds.xmax(), bounds.ymax(), bounds.zmax())
    );

    const math::Transform xform(*(math::Transform::createLinearTransform(spacing)));

    wlGridPtr grid = openvdb::tools::createLevelSetBox<wlGridType>(bbox, xform, bandWidth);

    if (!is_signed) {
        mma::check_abort();
        transformActiveLeafValues<wlTreeType, AbsF>(grid->tree(), AbsF());
    }

    setGrid(grid);
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::meshLevelSet(
    mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
    double spacing, double bandWidth, bool is_signed)
{
    scalar_type_assert<V>();

    if (tri_cells.cols() != 3)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    const int conversionFlags = is_signed ? 0 : UNSIGNED_DISTANCE_FIELD;

    wlGridPtr grid = openvdbmma::levelset::meshToLevelSet<wlGridType>(pts, tri_cells,
        spacing, bandWidth, conversionFlags);

    setGrid(grid);
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::offsetSurfaceLevelSet(
    mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
    double offset, double spacing, double width, bool is_signed)
{
    scalar_type_assert<V>();

    if (tri_cells.cols() != 3)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    wlGridPtr grid = openvdbmma::levelset::offsetSurfaceLevelSet<wlGridType>(pts,
        tri_cells, offset, spacing, width, is_signed);

    setGrid(grid);
}

#endif // OPENVDBLINK_OPENVDBGRID_LEVELSETCREATION_HAS_BEEN_INCLUDED
