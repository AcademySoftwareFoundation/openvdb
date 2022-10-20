// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_MEASURE_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_MEASURE_HAS_BEEN_INCLUDED

#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/LevelSetMeasure.h>


/* OpenVDBGrid public member function list

 double levelSetGridArea()

 mint levelSetGridEulerCharacteristic()

 mint levelSetGridGenus()

 double levelSetGridVolume()

 */


//////////// OpenVDBGrid public member function definitions

template<typename V>
double
openvdbmma::OpenVDBGrid<V>::levelSetGridArea() const
{
    scalar_type_assert<V>();

    return levelSetArea(*grid(), true /* world space */);
}

template<typename V>
mint
openvdbmma::OpenVDBGrid<V>::levelSetGridEulerCharacteristic() const
{
    scalar_type_assert<V>();

    return levelSetEulerCharacteristic(*grid());
}

template<typename V>
mint
openvdbmma::OpenVDBGrid<V>::levelSetGridGenus() const
{
    scalar_type_assert<V>();

    return levelSetGenus(*grid());
}

template<typename V>
double
openvdbmma::OpenVDBGrid<V>::levelSetGridVolume() const
{
    scalar_type_assert<V>();

    return levelSetVolume(*grid(), true /* world space */);
}

#endif // OPENVDBLINK_OPENVDBGRID_MEASURE_HAS_BEEN_INCLUDED
