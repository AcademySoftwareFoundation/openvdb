// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_MORPHOLOGY_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_MORPHOLOGY_HAS_BEEN_INCLUDED

#include <openvdb/tools/LevelSetFilter.h>


/* OpenVDBGrid public member function list

void resizeBandwidth(double width)

void offsetLevelSet(double r)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::resizeBandwidth(double width)
{
    scalar_type_assert<V>();

    using MaskT = typename wlGridType::template ValueConverter<float>::Type;
    using InterrupterT = mma::interrupt::LLInterrupter;

    InterrupterT interrupt;

    openvdb::tools::LevelSetFilter<wlGridType, MaskT, InterrupterT> filter(*grid(), &interrupt);
    filter.resize(width);

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::offsetLevelSet(double r)
{
    scalar_type_assert<V>();

    using MaskT = typename wlGridType::template ValueConverter<float>::Type;
    using InterrupterT = mma::interrupt::LLInterrupter;

    InterrupterT interrupt;

    openvdb::tools::LevelSetFilter<wlGridType, MaskT, InterrupterT> filter(*grid(), &interrupt);
    filter.offset(r);

    setLastModified();
}

#endif // OPENVDBLINK_OPENVDBGRID_MORPHOLOGY_HAS_BEEN_INCLUDED
