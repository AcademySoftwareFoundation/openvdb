// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_MESH_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_MESH_HAS_BEEN_INCLUDED

#include <openvdb/tools/VolumeToMesh.h>

#include <openvdb/math/Transform.h>


/* openvdbmma::mesh members

 class VolumeToMmaMesh

 public members are

 meshCellCounts
 meshData

 It would be nice to have a meshCellCountsEstimate function.
 This would enable a target number of faces option in OpenVDBMesh.

*/


namespace openvdbmma {
namespace mesh {

//////////// mma mesh class

class VolumeToMmaMesh
{
public:

    VolumeToMmaMesh(double isovalue, double adaptivity)
    : mMesher(isovalue, adaptivity)
    {
    }

    ~VolumeToMmaMesh() {}

    template<typename GridT>
    void operator()(const GridT& grid)
    {
        mMesher.operator()<GridT>(grid);
    }

    mma::IntVectorRef meshCellCounts(const bool tri_only = false) const;

    mma::RealVectorRef meshData(const bool tri_only = false) const;

private:

    VolumeToMesh mMesher;

}; // end of VolumeToMmaMesh class


//////////// VolumeToMmaMesh public member function definitions

inline mma::IntVectorRef
VolumeToMmaMesh::meshCellCounts(const bool tri_only) const
{
    const PointList *vertlist = &mMesher.pointList();
    const PolygonPoolList *facelist = &mMesher.polygonPoolList();

    const mint coordcnt = mMesher.pointListSize();
    mint tricnt = 0, quadcnt = 0;
    for (mint i = 0; i < mMesher.polygonPoolListSize(); i++) {
        if (tri_only) {
            tricnt += (*facelist)[i].numTriangles() + 2*(*facelist)[i].numQuads();
        } else {
            tricnt += (*facelist)[i].numTriangles();
            quadcnt += (*facelist)[i].numQuads();
        }
    }

    mma::IntVectorRef meshcounts = mma::makeVector<mint>(3);
    meshcounts[0] = coordcnt;
    meshcounts[1] = tricnt;
    meshcounts[2] = quadcnt;

    return meshcounts;
}

inline mma::RealVectorRef
VolumeToMmaMesh::meshData(const bool tri_only) const
{
    const PointList *vertlist = &mMesher.pointList();
    const PolygonPoolList *facelist = &mMesher.polygonPoolList();

    const mint coordcnt = mMesher.pointListSize();
    mint tricnt = 0, quadcnt = 0;
    for (mint i = 0; i < mMesher.polygonPoolListSize(); i++) {
        if (tri_only) {
            tricnt += (*facelist)[i].numTriangles() + 2*(*facelist)[i].numQuads();
        } else {
            tricnt += (*facelist)[i].numTriangles();
            quadcnt += (*facelist)[i].numQuads();
        }
    }

    mma::check_abort();

    mma::RealVectorRef meshdata = mma::makeVector<double>(3 + 3*coordcnt + 3*tricnt + 4*quadcnt);
    meshdata[0] = coordcnt;
    meshdata[1] = tricnt;
    meshdata[2] = quadcnt;

    mint cnt = 3;
    for (mint i = 0; i < mMesher.pointListSize(); i++) {
        Vec3s *coord = &((*vertlist)[i]);
        meshdata[cnt++] = coord->x(); meshdata[cnt++] = coord->y(); meshdata[cnt++] = coord->z();
    }

    mma::check_abort();

    for (mint i = 0; i < mMesher.polygonPoolListSize(); i++) {
        for (mint j = 0; j < (*facelist)[i].numTriangles(); j++) {
            Vec3I *p = &((*facelist)[i].triangle(j));
            meshdata[cnt++] = p->x(); meshdata[cnt++] = p->y(); meshdata[cnt++] = p->z();
        }
    }

    mma::check_abort();

    if (tri_only) {
        for (mint i = 0; i < mMesher.polygonPoolListSize(); i++) {
            for (mint j = 0; j < (*facelist)[i].numQuads(); j++) {
                Vec4I *p = &((*facelist)[i].quad(j));
                meshdata[cnt++] = p->x(); meshdata[cnt++] = p->y(); meshdata[cnt++] = p->z();
                meshdata[cnt++] = p->x(); meshdata[cnt++] = p->z(); meshdata[cnt++] = p->w();
            }
        }
    } else {
        for (mint i = 0; i < mMesher.polygonPoolListSize(); i++) {
            for (mint j = 0; j < (*facelist)[i].numQuads(); j++) {
                Vec4I *p = &((*facelist)[i].quad(j));
                meshdata[cnt++] = p->x(); meshdata[cnt++] = p->y(); meshdata[cnt++] = p->z(); meshdata[cnt++] = p->w();
            }
        }
    }

    return meshdata;
}

} // namespace mesh
} // namespace openvdbmma

#endif // OPENVDBLINK_UTILITIES_MESH_HAS_BEEN_INCLUDED
