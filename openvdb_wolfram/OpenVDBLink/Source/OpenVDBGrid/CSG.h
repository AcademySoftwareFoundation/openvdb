// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_CSG_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_CSG_HAS_BEEN_INCLUDED

#include <openvdb/tools/Clip.h>
#include <openvdb/tools/Composite.h>


/* OpenVDBGrid vpublic member function list

void gridUnion(OpenVDBGrid& vdb)

void gridIntersection(OpenVDBGrid& vdb)

void gridDifference(OpenVDBGrid& vdb)

void gridUnionCopy(mma::IntTensorRef ids)

void gridIntersectionCopy(mma::IntTensorRef ids)

void gridDifferenceCopy(OpenVDBGrid& vdb1, OpenVDBGrid& vdb2)

void clipGrid(OpenVDBGrid& vdb, mma::RealBounds3DRef bds)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridUnion(OpenVDBGrid<V>& vdb)
{
    scalar_type_assert<V>();

    const wlGridPtr grid1 = grid();
    const wlGridPtr grid2 = vdb.grid();

    openvdb::tools::csgUnion(*grid1, *grid2);

    vdb.deleteGrid();
    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridIntersection(OpenVDBGrid<V>& vdb)
{
    scalar_type_assert<V>();

    const wlGridPtr grid1 = grid();
    const wlGridPtr grid2 = vdb.grid();

    openvdb::tools::csgIntersection(*grid1, *grid2);

    vdb.deleteGrid();
    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridDifference(OpenVDBGrid<V>& vdb)
{
    scalar_type_assert<V>();

    const wlGridPtr grid1 = grid();
    const wlGridPtr grid2 = vdb.grid();

    openvdb::tools::csgDifference(*grid1, *grid2);

    vdb.deleteGrid();
    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridUnionCopy(mma::IntTensorRef ids)
{
    scalar_type_assert<V>();

    const int idcnt = ids.size();

    if (idcnt == 0)
        return;

    if (idcnt == 1) {
        setGrid(instanceGrid(ids[0])->deepCopy(), false);
        return;
    }

    wlGridPtr grid = openvdb::tools::csgUnionCopy(*instanceGrid(ids[0]),
        *instanceGrid(ids[1]));

    for (int i = 2; i < idcnt; i++) {
        wlGridPtr gridcopy = instanceGrid(ids[i])->deepCopy();
        openvdb::tools::csgUnion(*grid, *gridcopy);
    }

    setGrid(grid, false);
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridIntersectionCopy(mma::IntTensorRef ids)
{
    scalar_type_assert<V>();

    const int idcnt = ids.size();

    if (idcnt == 0)
        return;

    if (idcnt == 1) {
        setGrid(instanceGrid(ids[0])->deepCopy(), false);
        return;
    }

    wlGridPtr grid = openvdb::tools::csgIntersectionCopy(*instanceGrid(ids[0]),
        *instanceGrid(ids[1]));

    for (int i = 2; i < idcnt; i++) {
        wlGridPtr gridcopy = instanceGrid(ids[i])->deepCopy();
        openvdb::tools::csgIntersection(*grid, *gridcopy);
    }

    setGrid(grid, false);
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridDifferenceCopy(OpenVDBGrid<V>& vdb1, OpenVDBGrid<V>& vdb2)
{
    scalar_type_assert<V>();

    const wlGridPtr griddiff = openvdb::tools::csgDifferenceCopy(
        *vdb1.grid(), *vdb2.grid());

    setGrid(griddiff, false);
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridMax(openvdbmma::OpenVDBGrid<V>& vdb)
{
    const wlGridPtr grid1 = grid();
    const wlGridPtr grid2 = vdb.grid();

    openvdb::tools::compMax(*grid1, *grid2);

    vdb.deleteGrid();
    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::gridMin(openvdbmma::OpenVDBGrid<V>& vdb)
{
    const wlGridPtr grid1 = grid();
    const wlGridPtr grid2 = vdb.grid();

    openvdb::tools::compMin(*grid1, *grid2);

    vdb.deleteGrid();
    setLastModified();
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::clipGrid(OpenVDBGrid<V>& vdb, mma::RealBounds3DRef bds)
{
    scalar_type_assert<V>();

    const Vec3d xyzMin(bds.xmin(), bds.ymin(), bds.zmin());
    const Vec3d xyzMax(bds.xmax(), bds.ymax(), bds.zmax());

    const BBoxd bbox(xyzMin, xyzMax);

    const wlGridPtr gridclipped = openvdb::tools::clip(*vdb.grid(), bbox);

    setGrid(gridclipped, false);
}

#endif // OPENVDBLINK_OPENVDBGRID_CSG_HAS_BEEN_INCLUDED
