// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_TRANSFORM_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_TRANSFORM_HAS_BEEN_INCLUDED

#include "../Utilities/Transform.h"

#include <openvdb/tools/GridTransformer.h>


/* OpenVDBGrid public member function list

void transformGrid(OpenVDBScalarGrid &target_vdb, mma::RealMatrixRef mat, mint resampling = 1)

void scalarMultiply(double fac)

void gammaAdjustment(double gamma)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::transformGrid(OpenVDBGrid<V>& vdb,
    mma::RealMatrixRef mat, mint resampling)
{
    if (mat.rows() != 4 || mat.cols() != 4)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    wlGridPtr gridsource = vdb.grid();
    wlGridPtr gridtarget = grid();

    const Mat4R xform(mat.data());
    GridTransformer transformer(xform);

    mma::interrupt::LLInterrupter interrupt;
    transformer.setInterrupter(interrupt);

    switch(resampling) {

        case RS_NEAREST:
            transformer.transformGrid<PointSampler, wlGridType>(*gridsource, *gridtarget);
            break;

        case RS_LINEAR:
            transformer.transformGrid<BoxSampler, wlGridType>(*gridsource, *gridtarget);
            break;

        case RS_QUADRATIC:
            transformer.transformGrid<QuadraticSampler, wlGridType>(*gridsource, *gridtarget);
            break;

        default:
            throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);
            break;
    }

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::scalarMultiply(double fac)
{
    scalar_type_assert<V>();

    openvdbmma::transform::GridAdjustment<wlGridType, ValueT> adjuster(grid());

    adjuster.scalarMultiply(fac);

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gammaAdjustment(double gamma)
{
    scalar_type_assert<V>();

    openvdbmma::transform::GridAdjustment<wlGridType, ValueT> adjuster(grid());

    adjuster.gammaAdjust(gamma);

    setLastModified();
}

#endif // OPENVDBLINK_OPENVDBGRID_TRANSFORM_HAS_BEEN_INCLUDED
