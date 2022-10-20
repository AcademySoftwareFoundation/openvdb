#pragma once

#include <openvdb/tools/LevelSetUtil.h>

#include <openvdb/math/Coord.h>
#include <openvdb/math/Transform.h>


/* OpenVDBGrid public member function list

mma::IntVectorRef getActiveStates(mma::IntCoordinatesRef coords)

mint getActiveLeafVoxelCount()

mint getActiveTileCount()

mint getActiveVoxelCount()

GlueScalar getBackgroundValue()

mint getGridClass() const

const char* getCreator()

mma::IntMatrixRef getGridBoundingBox()

mma::IntVectorRef getGridDimensions()

const char* getGridType()

double getHalfwidth()

bool getHasUniformVoxels()

bool getIsEmpty()

mint getMemoryUsage()

GlueVector getMinMaxValues()

const char* getName()

GlueVector getValues(mma::IntCoordinatesRef coords)

double getVoxelSize()

*/

//////////// OpenVDBGrid public member function definitions

template<typename V>
mma::IntVectorRef
OpenVDBGrid<V>::getActiveStates(mma::IntCoordinatesRef coords) const
{
    typename wlGridType::Accessor accessor = grid()->getAccessor();

    int n = coords.size();
    Coord xyz(0, 0, 0);

    mma::IntVectorRef states = mma::makeVector<mint>(n);

    for (int i = 0; i < n; i++) {
        xyz.reset(coords.x(i), coords.y(i), coords.z(i));
        states[i] = accessor.isValueOn(xyz);
    }

    return states;
}

template<typename V>
mint
OpenVDBGrid<V>::getActiveLeafVoxelCount() const { return grid()->tree().activeLeafVoxelCount(); }

template<typename V>
inline mint
OpenVDBGrid<V>::getActiveTileCount() const { return grid()->tree().activeTileCount(); }

template<typename V>
inline mint
OpenVDBGrid<V>::getActiveVoxelCount() const { return grid()->activeVoxelCount(); }

template<typename V>
typename OpenVDBGrid<V>::GlueScalar
OpenVDBGrid<V>::getBackgroundValue() const
{
    openvdbmma::types::non_mask_type_assert<V>();

    return GScalar(grid()->background()).mmaData();
}


template<typename V>
mint
OpenVDBGrid<V>::getGridClass() const
{
    int grid_class_id;

    GridClass grid_class = grid()->getGridClass();

    if (grid_class == GRID_LEVEL_SET)
        grid_class_id = GC_LEVELSET;
    else if (grid_class == GRID_FOG_VOLUME)
        grid_class_id = GC_FOGVOLUME;
    else
        grid_class_id = GC_UNKNOWN;

    return grid_class_id;
}

template<typename V>
const char*
OpenVDBGrid<V>::getCreator()
{
    //Let the class handle memory management when passing a string to WL
    return WLString(grid()->getCreator());
}

template<typename V>
mma::IntMatrixRef
OpenVDBGrid<V>::getGridBoundingBox() const
{
    CoordBBox gbbox = grid()->evalActiveVoxelBoundingBox();
    Coord p1 = gbbox.min(), p2 = gbbox.max();

    mma::IntMatrixRef bbox = mma::makeMatrix<mint>(
        {{p1.x(), p2.x()}, {p1.y(), p2.y()}, {p1.z(), p2.z()}});

    return bbox;
}

template<typename V>
mma::IntVectorRef
OpenVDBGrid<V>::getGridDimensions() const
{
    const Coord gdims = grid()->evalActiveVoxelDim();

    mma::IntVectorRef dims = mma::makeVector<mint>({gdims.x(), gdims.y(), gdims.z()});

    return dims;
}

template<typename V>
const char*
OpenVDBGrid<V>::getGridType()
{
    //Let the class handle memory management when passing a string to WL
    return WLString(grid()->gridType());
}

template<typename V>
inline double
OpenVDBGrid<V>::getHalfwidth() const
{
    return this->getBackgroundValue()/this->getVoxelSize();
}

template<typename V>
inline bool
OpenVDBGrid<V>::getHasUniformVoxels() const { return grid()->hasUniformVoxels(); }

template<typename V>
inline bool
OpenVDBGrid<V>::getIsEmpty() const { return grid()->empty(); }

template<typename V>
inline mint
OpenVDBGrid<V>::getMemoryUsage() const { return grid()->memUsage(); }

template<typename V>
mma::TensorRef<typename OpenVDBGrid<V>::mmaBaseValT>
OpenVDBGrid<V>::getMinMaxValues() const
{
    openvdbmma::types::non_mask_type_assert<V>();

    openvdb::math::MinMax<ValueT> extrema = openvdb::tools::minMax(grid()->tree());

    GVector minmax(2);

    minmax.template setValue<V>(0, extrema.min());
    minmax.template setValue<V>(1, extrema.max());

    return minmax.mmaData();
}

template<typename V>
const char*
OpenVDBGrid<V>::getName()
{
    //Let the class handle memory management when passing a string to WL
    return WLString(grid()->getName());
}

template<typename V>
mma::TensorRef<typename OpenVDBGrid<V>::mmaBaseValT>
OpenVDBGrid<V>::getValues(mma::IntCoordinatesRef coords) const
{
    openvdbmma::types::non_mask_type_assert<V>();

    const typename wlGridType::Accessor accessor = grid()->getAccessor();
    const int n = coords.size();
    Coord xyz(0, 0, 0);

    GVector values(n);

    for (int i = 0; i < n; ++i) {
        xyz.reset(coords.x(i), coords.y(i), coords.z(i));
        values.template setValue<V>(i, accessor.getValue(xyz));
    }

    return values.mmaData();
}

template<typename V>
inline double
OpenVDBGrid<V>::getVoxelSize() const { return (grid()->voxelSize())[0]; }
