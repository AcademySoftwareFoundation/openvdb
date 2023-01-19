// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_FOGVOLUME_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_FOGVOLUME_HAS_BEEN_INCLUDED

#include <openvdb/tools/LevelSetUtil.h>

/* OpenVDBGrid public member function list

void levelSetToFogVolume(double cutoff)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::levelSetToFogVolume(double cutoff)
{
    scalar_type_assert<V>();

    if (cutoff > 0)
        openvdb::tools::sdfToFogVolume<wlGridType>(*grid(), cutoff);
    else
        openvdb::tools::sdfToFogVolume<wlGridType>(*grid());

    float mDx = OpenVDBGrid::getVoxelSize();

    grid()->insertMeta(META_CUTOFF_DISTANCE, openvdb::FloatMetadata(cutoff / mDx));
    grid()->insertMeta(META_GAMMA_ADJUSTMENT, openvdb::FloatMetadata(1.0f));
    grid()->insertMeta(META_SCALING_FACTOR, openvdb::FloatMetadata(1.0f));
}

#endif // OPENVDBLINK_OPENVDBGRID_FOGVOLUME_HAS_BEEN_INCLUDED
