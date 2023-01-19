// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_FILTER_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_FILTER_HAS_BEEN_INCLUDED

#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/Interpolation.h>

#include <algorithm>
#include <functional>
#include <type_traits>


/* OpenVDBGrid public member function list

void filterGrid(mint filter_type, mint width, mint iter)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::filterGrid(mint filter_type, mint width, mint iter)
{
    scalar_type_assert<V>();

    using MaskT = typename wlGridType::template ValueConverter<float>::Type;
    using InterrupterT = mma::interrupt::LLInterrupter;

    if (width < 1 || iter < 0)
        throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

    if (iter == 0)
        return;

    InterrupterT interrupt;

    openvdb::tools::LevelSetFilter<wlGridType, MaskT, InterrupterT> lsf(*grid(), &interrupt);

    for (int i = 0; i < iter; i++) {
        mma::check_abort();

        switch (filter_type) {

            case MEAN_FILTER:
                lsf.mean(width);
                break;

            case MEDIAN_FILTER:
                lsf.median(width);
                break;

            case GAUSSIAN_FILTER:
                lsf.gaussian(width);
                break;

            case LAPLACIAN_FILTER:
                lsf.laplacian();
                break;

            case MEAN_CURVATURE_FILTER:
                lsf.meanCurvature();
                break;

            default:
                throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);
                break;
        }
    }

    setLastModified();
}

#endif // OPENVDBLINK_OPENVDBGRID_FILTER_HAS_BEEN_INCLUDED
