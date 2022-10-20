// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_SETTERS_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_SETTERS_HAS_BEEN_INCLUDED

#include <openvdb/math/Coord.h>
#include <openvdb/math/Transform.h>


/* OpenVDBGrid public member function list

void setActiveStates(mma::IntCoordinatesRef coords, mma::IntVectorRef states)

void setBackgroundValue(GlueScalar bg)

void setGridClass(mint grid_class)

void setCreator(const char* creator)

void setName(const char* name)

void setValues(mma::IntCoordinatesRef coords, GlueVector vals)

void setVoxelSize(double spacing)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::setActiveStates(
    mma::IntCoordinatesRef coords, mma::IntVectorRef states)
{
    typename wlGridType::Accessor accessor = grid()->getAccessor();

    const int n = states.size();
    Coord xyz(0, 0, 0);

    if (coords.size() != n)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    for (int i = 0; i < n; i++) {
        xyz.reset(coords.x(i), coords.y(i), coords.z(i));
        accessor.setActiveState(xyz, states(i));
    }

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::setBackgroundValue(GlueScalar bg)
{
    openvdbmma::types::non_mask_type_assert<V>();

    if (!valid_glueScalar(bg))
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    openvdb::tools::changeBackground(grid()->tree(), mma::toVDB<mmaBaseValT, V>(bg));

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::setGridClass(mint grid_class)
{
    switch (grid_class) {

        case GC_LEVELSET:
            grid()->setGridClass(GRID_LEVEL_SET);
            break;

        case GC_FOGVOLUME:
            grid()->setGridClass(GRID_FOG_VOLUME);
            break;

        default:
            grid()->setGridClass(GRID_UNKNOWN);
            break;
    }

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::setCreator(const char* creator)
{
    grid()->setCreator(std::string(creator));
    mma::disownString(creator);

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::setName(const char* name)
{
    grid()->setName(std::string(name));
    mma::disownString(name);

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::setValues(mma::IntCoordinatesRef coords, GlueVector vals)
{
    openvdbmma::types::non_mask_type_assert<V>();

    const int n = coords.size();
    if (!valid_glueVector(vals, n))
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    typename wlGridType::Accessor accessor = grid()->getAccessor();
    Coord xyz(0, 0, 0);

    for (int i = 0; i < n; i++) {
        xyz.reset(coords.x(i), coords.y(i), coords.z(i));

        accessor.setValue(xyz, mma::toVDB<mmaBaseValT, V>(vals, i));
    }

    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::setVoxelSize(double spacing)
{
    math::Transform xform(*(math::Transform::createLinearTransform(spacing)));
    grid()->setTransform(xform.copy());

    setLastModified();
}

#endif // OPENVDBLINK_OPENVDBGRID_SETTERS_HAS_BEEN_INCLUDED
