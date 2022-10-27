// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_MESH_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_MESH_HAS_BEEN_INCLUDED

#include "../Utilities/Mesh.h"


/* OpenVDBGrid public member function list

mma::IntVectorRef meshCellCount(double isovalue, double adaptivity, bool tri_only)

mma::RealVectorRef meshData(double isovalue, double adaptivity, bool tri_only)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
mma::IntVectorRef
openvdbmma::OpenVDBGrid<V>::meshCellCount(double isovalue, double adaptivity, bool tri_only) const
{
    scalar_type_assert<V>();

    openvdbmma::mesh::VolumeToMmaMesh mesher(isovalue, adaptivity);
    mesher.operator()<wlGridType>(*grid());

    mma::IntVectorRef meshcounts = mesher.meshCellCounts(tri_only);

    return meshcounts;
}

// Would be nice if mesh coordinates and cells were
// returned separately as class members in OpenVDBGrid.
// Instead we return the cells as floats, which WL will convert to ints...

template<typename V>
mma::RealVectorRef
openvdbmma::OpenVDBGrid<V>::meshData(double isovalue, double adaptivity, bool tri_only) const
{
    scalar_type_assert<V>();

    openvdbmma::mesh::VolumeToMmaMesh mesher(isovalue, adaptivity);
    mesher.operator()<wlGridType>(*grid());

    mma::check_abort();

    mma::RealVectorRef meshdata = mesher.meshData(tri_only);

    return meshdata;
}

#endif // OPENVDBLINK_OPENVDBGRID_MESH_HAS_BEEN_INCLUDED
