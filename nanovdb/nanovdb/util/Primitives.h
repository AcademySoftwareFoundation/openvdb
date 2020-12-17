// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file Primitives.h

    \author Ken Museth

    \date June 26, 2020

    \brief Generates volumetric primitives, e.g. sphere, torus etc, as NanoVDB grid.

    \note This has no dependency on openvdb.
*/

#ifndef NANOVDB_PRIMITIVES_H_HAS_BEEN_INCLUDED
#define NANOVDB_PRIMITIVES_H_HAS_BEEN_INCLUDED

#include "GridBuilder.h"

namespace nanovdb {

/// @brief Returns a handle to a narrow-band level set of a sphere
///
/// @param radius    Radius of sphere in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param sMode     Mode of computation for the statistics.
/// @param cMode     Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetSphere(ValueT             radius = 100,
                     const Vec3d&       center = Vec3d(0),
                     ValueT             voxelSize = 1.0,
                     ValueT             halfWidth = 3.0,
                     const Vec3d&       origin = Vec3d(0),
                     const std::string& name = "sphere_ls",
                     StatsMode          sMode = StatsMode::Default,
                     ChecksumMode       cMode = ChecksumMode::Default,
                     const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a sparse fog volume of a sphere such
///        that the exterior is 0 and inactive, the interior is active
///        with values varying smoothly from 0 at the surface of the
///        sphere to 1 at the halfWidth and interior of the sphere.
///
/// @param radius    Radius of sphere in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param sMode     Mode of computation for the statistics.
/// @param cMode     Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createFogVolumeSphere(ValueT             radius = 100,
                      const Vec3d&       center = Vec3d(0),
                      ValueT             voxelSize = 1.0,
                      ValueT             halfWidth = 3.0,
                      const Vec3d&       origin = Vec3d(0),
                      const std::string& name = "sphere_fog",
                      StatsMode          sMode = StatsMode::Default,
                      ChecksumMode       cMode = ChecksumMode::Default,
                      const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a PointDataGrid containing points scattered
///        on the surface of a sphere.
///
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param radius         Radius of sphere in world units
/// @param center         Center of sphere in world units
/// @param voxelSize      Size of a voxel in world units
/// @param origin         Origin of grid in world units
/// @param name           Name of the grid
/// @param mode           Mode of computation for the checksum.
/// @param buffer         Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointSphere(int                pointsPerVoxel = 1,
                  ValueT             radius = 100,
                  const Vec3d&       center = Vec3d(0),
                  ValueT             voxelSize = 1.0,
                  const Vec3d&       origin = Vec3d(0),
                  const std::string& name = "sphere_points",
                  ChecksumMode       mode = ChecksumMode::Default,
                  const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a narrow-band level set of a torus in the xz-plane
///
/// @param majorRadius Major radius of torus in world units
/// @param minorRadius Minor radius of torus in world units
/// @param center      Center of torus in world units
/// @param voxelSize   Size of a voxel in world units
/// @param halfWidth   Half-width of narrow band in voxel units
/// @param origin      Origin of grid in world units
/// @param name        Name of the grid
/// @param sMode       Mode of computation for the statistics.
/// @param cMode       Mode of computation for the checksum.
/// @param buffer      Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetTorus(ValueT             majorRadius = 100,
                    ValueT             minorRadius = 50,
                    const Vec3d&       center = Vec3d(0),
                    ValueT             voxelSize = 1.0,
                    ValueT             halfWidth = 3.0,
                    const Vec3d&       origin = Vec3d(0),
                    const std::string& name = "torus_ls",
                    StatsMode          sMode = StatsMode::Default,
                    ChecksumMode       cMode = ChecksumMode::Default,
                    const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a sparse fog volume of a torus in the xz-plane such
///        that the exterior is 0 and inactive, the interior is active
///        with values varying smoothly from 0 at the surface of the
///        torus to 1 at the halfWidth and interior of the torus.
///
/// @param majorRadius Major radius of torus in world units
/// @param minorRadius Minor radius of torus in world units
/// @param center      Center of torus in world units
/// @param voxelSize   Size of a voxel in world units
/// @param halfWidth   Half-width of narrow band in voxel units
/// @param origin      Origin of grid in world units
/// @param name        Name of the grid
/// @param sMode       Mode of computation for the statistics.
/// @param cMode       Mode of computation for the checksum.
/// @param buffer      Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createFogVolumeTorus(ValueT             majorRadius = 100,
                     ValueT             minorRadius = 50,
                     const Vec3d&       center = Vec3d(0),
                     ValueT             voxelSize = 1.0,
                     ValueT             halfWidth = 3.0,
                     const Vec3d&       origin = Vec3d(0),
                     const std::string& name = "torus_fog",
                     StatsMode          sMode = StatsMode::Default,
                     ChecksumMode       cMode = ChecksumMode::Default,
                     const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a PointDataGrid containing points scattered
///        on the surface of a torus.
///
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param majorRadius    Major radius of torus in world units
/// @param minorRadius    Minor radius of torus in world units
/// @param center         Center of torus in world units
/// @param voxelSize      Size of a voxel in world units
/// @param origin         Origin of grid in world units
/// @param name           Name of the grid
/// @param mode           Mode of computation for the checksum.
/// @param buffer         Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointTorus(int                pointsPerVoxel = 1, // half-width of narrow band in voxel units
                 ValueT             majorRadius = 100, // major radius of torus in world units
                 ValueT             minorRadius = 50, // minor radius of torus in world units
                 const Vec3d&       center = Vec3d(0), //center of torus in world units
                 ValueT             voxelSize = 1.0, // size of a voxel in world units
                 const Vec3d&       origin = Vec3d(0), // origin of grid in world units
                 const std::string& name = "torus_points", // name of grid
                 ChecksumMode       cMode = ChecksumMode::Default,
                 const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a narrow-band level set of a box
///
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param center    Center of box in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param sMode     Mode of computation for the statistics.
/// @param cMode     Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetBox(ValueT             width = 40,
                  ValueT             height = 60,
                  ValueT             depth = 100,
                  const Vec3d&       center = Vec3d(0),
                  ValueT             voxelSize = 1.0,
                  ValueT             halfWidth = 3.0,
                  const Vec3d&       origin = Vec3d(0),
                  const std::string& name = "box_ls",
                  StatsMode          sMode = StatsMode::Default,
                  ChecksumMode       cMode = ChecksumMode::Default,
                  const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a sparse fog volume of a box such
///        that the exterior is 0 and inactive, the interior is active
///        with values varying smoothly from 0 at the surface of the
///        box to 1 at the halfWidth and interior of the box.
///
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param center    Center of box in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param sMode     Mode of computation for the statistics.
/// @param cMode     Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createFogVolumeBox(ValueT             width = 40,
                   ValueT             height = 60,
                   ValueT             depth = 100,
                   const Vec3d&       center = Vec3d(0),
                   ValueT             voxelSize = 1.0,
                   ValueT             halfWidth = 3.0,
                   const Vec3d&       origin = Vec3d(0),
                   const std::string& name = "box_fog",
                   StatsMode          sMode = StatsMode::Default,
                   ChecksumMode       cMode = ChecksumMode::Default,
                   const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a narrow-band level set of a octahedron
///
/// @param scale     Scale of octahedron in world units
/// @param center    Center of octahedron in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param sMode     Mode of computation for the statistics.
/// @param cMode     Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetOctahedron(ValueT             scale = 100,
                         const Vec3d&       center = Vec3d(0),
                         ValueT             voxelSize = 1.0,
                         ValueT             halfWidth = 3.0,
                         const Vec3d&       origin = Vec3d(0),
                         const std::string& name = "octadedron_ls",
                         StatsMode          sMode = StatsMode::Default,
                         ChecksumMode       cMode = ChecksumMode::Default,
                         const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a sparse fog volume of an octahedron such
///        that the exterior is 0 and inactive, the interior is active
///        with values varying smoothly from 0 at the surface of the
///        octahedron to 1 at the halfWidth and interior of the octahedron.
///
/// @param scale     Scale of octahedron in world units
/// @param center    Center of box in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param sMode     Mode of computation for the statistics.
/// @param cMode     Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createFogVolumeOctahedron(ValueT             scale = 100,
                          const Vec3d&       center = Vec3d(0),
                          ValueT             voxelSize = 1.0,
                          ValueT             halfWidth = 3.0,
                          const Vec3d&       origin = Vec3d(0),
                          const std::string& name = "octadedron_fog",
                          StatsMode          sMode = StatsMode::Default,
                          ChecksumMode       cMode = ChecksumMode::Default,
                          const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a narrow-band level set of a bounding-box (= wireframe of a box)
///
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param thickness Thickness of the wire in world units
/// @param center    Center of bbox in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param sMode     Mode of computation for the statistics.
/// @param cMode     Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetBBox(ValueT             width = 40,
                   ValueT             height = 60,
                   ValueT             depth = 100,
                   ValueT             thickness = 10,
                   const Vec3d&       center = Vec3d(0),
                   ValueT             voxelSize = 1.0,
                   ValueT             halfWidth = 3.0,
                   const Vec3d&       origin = Vec3d(0),
                   const std::string& name = "bbox_ls",
                   StatsMode          sMode = StatsMode::Default,
                   ChecksumMode       cMode = ChecksumMode::Default,
                   const BufferT&     buffer = BufferT());



//================================================================================================

/// @brief Returns a handle to a PointDataGrid containing points scattered
///        on the surface of a box.
///
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param center    Center of box in world units
/// @param voxelSize Size of a voxel in world units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param mode      Mode of computation for the checksum.
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointBox(int                pointsPerVoxel = 1, // half-width of narrow band in voxel units
               ValueT             width = 40, // width of box in world units
               ValueT             height = 60, // height of box in world units
               ValueT             depth = 100, // depth of box in world units
               const Vec3d&       center = Vec3d(0), //center of box in world units
               ValueT             voxelSize = 1.0, // size of a voxel in world units
               const Vec3d&       origin = Vec3d(0), // origin of grid in world units
               const std::string& name = "box_points", // name of grid
               ChecksumMode       mode = ChecksumMode::Default,
               const BufferT&     buffer = BufferT());

//================================================================================================

/// @brief Given an input NanoVDB voxel grid this methods returns a GridHandle to another NanoVDB
///        PointDataGrid with points scattered in the active leaf voxels of in input grid.
///
/// @param srcGrid        Const input grid used to determine the active voxels to scatter point intp
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param name           Name of the grid
/// @param mode           Mode of computation for the checksum.
/// @param buffer         Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointScatter(const NanoGrid<ValueT>& srcGrid, // origin of grid in world units
                   int                     pointsPerVoxel = 1, // half-width of narrow band in voxel units
                   const std::string&      name = "point_scatter", // name of grid
                   ChecksumMode            mode = ChecksumMode::Default,
                   const BufferT&          buffer = BufferT());

//================================================================================================

namespace {

/// @brief Returns a shared pointer to a GridBuilder with narrow-band SDF values for a sphere
///
/// @brief Note, this is not (yet) a valid level set SDF field since values inside sphere (and outside
///        the narrow band) are still undefined. Call GridBuilder::sdfToLevelSet() to set those
///        values or alternatively call GridBuilder::sdfToFog to generate a FOG volume.
template<typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initSphere(ValueT       radius, // radius of sphere in world units
           const Vec3d& center, //center of sphere in world units
           ValueT       voxelSize, // size of a voxel in world units
           ValueT       halfWidth, // half-width of narrow band in voxel units
           const Vec3d& origin) // origin of grid in world units
{
    static_assert(is_floating_point<ValueT>::value, "Sphere: expect floating point");
    if (!(radius > 0))
        throw std::runtime_error("Sphere: radius must be positive!");
    if (!(voxelSize > 0))
        throw std::runtime_error("Sphere: voxelSize must be positive!");
    if (!(halfWidth > 0))
        throw std::runtime_error("Sphere: halfWidth must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth * voxelSize);
    auto acc = builder->getAccessor();

    // Define radius of sphere with narrow-band in voxel units
    const ValueT r0 = radius / voxelSize, rmax = r0 + halfWidth;

    // Radius below the Nyquist frequency
    if (r0 < ValueT(1.5))
        return builder;

    // Define center of sphere in voxel units
    const Vec3<ValueT> c(ValueT(center[0] - origin[0]) / voxelSize,
                         ValueT(center[1] - origin[1]) / voxelSize,
                         ValueT(center[2] - origin[2]) / voxelSize);

    // Define bounds of the voxel coordinates
    const int imin = Floor(c[0] - rmax), imax = Ceil(c[0] + rmax);
    const int jmin = Floor(c[1] - rmax), jmax = Ceil(c[1] + rmax);
    const int kmin = Floor(c[2] - rmax), kmax = Ceil(c[2] + rmax);

    Coord ijk;
    int & i = ijk[0], &j = ijk[1], &k = ijk[2], m = 1;
    // Compute signed distances to sphere using leapfrogging in k
    for (i = imin; i <= imax; ++i) {
        const auto x2 = Pow2(ValueT(i) - c[0]);
        for (j = jmin; j <= jmax; ++j) {
            const auto x2y2 = Pow2(ValueT(j) - c[1]) + x2;
            for (k = kmin; k <= kmax; k += m) {
                m = 1;
                const auto v = Sqrt(x2y2 + Pow2(ValueT(k) - c[2])) - r0; // Distance in voxel units
                const auto d = v < 0 ? -v : v;
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(ijk, voxelSize * v); // distance in world units
                } else { // outside narrow band
                    m += Floor(d - halfWidth); // leapfrog
                }
            } //end leapfrog over k
        } //end loop over j
    } //end loop over i

    return builder;
} // initSphere

template<typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initTorus(ValueT       radius1, // major radius of torus in world units
          ValueT       radius2, // minor radius of torus in world units
          const Vec3d& center, //center of torus in world units
          ValueT       voxelSize, // size of a voxel in world units
          ValueT       halfWidth, // half-width of narrow band in voxel units
          const Vec3d& origin) // origin of grid in world units
{
    static_assert(is_floating_point<ValueT>::value, "Torus: expect floating point");
    if (!(radius2 > 0))
        throw std::runtime_error("Torus: radius2 must be positive!");
    if (!(radius1 > radius2))
        throw std::runtime_error("Torus: radius1 must be larger than radius2!");
    if (!(voxelSize > 0))
        throw std::runtime_error("Torus: voxelSize must be positive!");
    if (!(halfWidth > 0))
        throw std::runtime_error("Torus: halfWidth must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth * voxelSize);
    auto acc = builder->getAccessor();

    // Define size of torus with narrow-band in voxel units
    const ValueT r1 = radius1 / voxelSize, r2 = radius2 / voxelSize, rmax1 = r1 + r2 + halfWidth, rmax2 = r2 + halfWidth;

    // Radius below the Nyquist frequency
    if (r2 < ValueT(1.5))
        return builder;

    // Define center of torus in voxel units
    const Vec3<ValueT> c(ValueT(center[0] - origin[0]) / voxelSize,
                         ValueT(center[1] - origin[1]) / voxelSize,
                         ValueT(center[2] - origin[2]) / voxelSize);

    // Define bounds of the voxel coordinates
    const int imin = Floor(c[0] - rmax1), imax = Ceil(c[0] + rmax1);
    const int jmin = Floor(c[1] - rmax2), jmax = Ceil(c[1] + rmax2);
    const int kmin = Floor(c[2] - rmax1), kmax = Ceil(c[2] + rmax1);

    Coord ijk;
    int & i = ijk[0], &j = ijk[1], &k = ijk[2], m = 1;
    // Compute signed distances to torus using leapfrogging in k
    for (i = imin; i <= imax; ++i) {
        const auto x2 = Pow2(ValueT(i) - c[0]);
        for (k = kmin; k <= kmax; ++k) {
            const auto x2z2 = Pow2(Sqrt(Pow2(ValueT(k) - c[2]) + x2) - r1);
            for (j = jmin; j <= jmax; j += m) {
                m = 1;
                const auto v = Sqrt(x2z2 + Pow2(ValueT(j) - c[1])) - r2; // Distance in voxel units
                const auto d = v < 0 ? -v : v;
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(ijk, voxelSize * v); // distance in world units
                } else { // outside narrow band
                    m += Floor(d - halfWidth); // leapfrog
                }
            } //end leapfrog over k
        } //end loop over j
    } //end loop over i

    return builder;
} // initTorus

template<typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initBox(ValueT       width, // major radius of torus in world units
        ValueT       height, // minor radius of torus in world units
        ValueT       depth,
        const Vec3d& center, //center of box in world units
        ValueT       voxelSize, // size of a voxel in world units
        ValueT       halfWidth, // half-width of narrow band in voxel units
        const Vec3d& origin) // origin of grid in world units
{
    using Vec3T = Vec3<ValueT>;
    static_assert(is_floating_point<ValueT>::value, "Box: expect floating point");
    if (!(width > 0))
        throw std::runtime_error("Box: width must be positive!");
    if (!(height > 0))
        throw std::runtime_error("Box: height must be positive!");
    if (!(depth > 0))
        throw std::runtime_error("Box: depth must be positive!");

    if (!(voxelSize > 0))
        throw std::runtime_error("Box: voxelSize must be positive!");
    if (!(halfWidth > 0))
        throw std::runtime_error("Box: halfWidth must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth * voxelSize);
    auto acc = builder->getAccessor();

    // Define size of box with narrow-band in voxel units
    const Vec3T r(width / (2 * voxelSize), height / (2 * voxelSize), depth / (2 * voxelSize));

    // Below the Nyquist frequency
    if (r.min() < ValueT(1.5))
        return builder;

    // Define center of box in voxel units
    const Vec3T c(ValueT(center[0] - origin[0]) / voxelSize,
                  ValueT(center[1] - origin[1]) / voxelSize,
                  ValueT(center[2] - origin[2]) / voxelSize);

    // Define utinity functions
    auto Pos = [](ValueT x) { return x > 0 ? x : 0; };
    auto Neg = [](ValueT x) { return x < 0 ? x : 0; };

    // Define bounds of the voxel coordinates
    const BBox<Vec3T> b(c - r - Vec3T(halfWidth), c + r + Vec3T(halfWidth));
    const CoordBBox   bbox(Coord(Floor(b[0][0]), Floor(b[0][1]), Floor(b[0][2])),
                         Coord(Ceil(b[1][0]), Ceil(b[1][1]), Ceil(b[1][2])));

    // Compute signed distances to box using leapfrogging in k
    int m = 1;
    for (Coord p = bbox[0]; p[0] <= bbox[1][0]; ++p[0]) {
        const auto q1 = Abs(ValueT(p[0]) - c[0]) - r[0];
        const auto x2 = Pow2(Pos(q1));
        for (p[1] = bbox[0][1]; p[1] <= bbox[1][1]; ++p[1]) {
            const auto q2 = Abs(ValueT(p[1]) - c[1]) - r[1];
            const auto q0 = Max(q1, q2);
            const auto x2y2 = x2 + Pow2(Pos(q2));
            for (p[2] = bbox[0][2]; p[2] <= bbox[1][2]; p[2] += m) {
                m = 1;
                const auto q3 = Abs(ValueT(p[2]) - c[2]) - r[2];
                const auto v = Sqrt(x2y2 + Pow2(Pos(q3))) + Neg(Max(q0, q3)); // Distance in voxel units
                const auto d = Abs(v);
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(p, voxelSize * v); // distance in world units
                } else { // outside narrow band
                    m += Floor(d - halfWidth); // leapfrog
                }
            } //end leapfrog over k
        } //end loop over j
    } //end loop over i

    return builder;
} // initBox

template<typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initBBox(ValueT       width, // width of the bbox in world units
         ValueT       height, // height of the bbox in world units
         ValueT       depth, // depth of the bbox in world units
         ValueT       thickness, // thickness of the wire in world units
         const Vec3d& center, //center of bbox in world units
         ValueT       voxelSize, // size of a voxel in world units
         ValueT       halfWidth, // half-width of narrow band in voxel units
         const Vec3d& origin) // origin of grid in world units
{
    using Vec3T = Vec3<ValueT>;
    static_assert(is_floating_point<ValueT>::value, "BBox: expect floating point");
    if (!(width > 0))
        throw std::runtime_error("BBox: width must be positive!");
    if (!(height > 0))
        throw std::runtime_error("BBox: height must be positive!");
    if (!(depth > 0))
        throw std::runtime_error("BBox: depth must be positive!");
    if (!(thickness > 0))
        throw std::runtime_error("BBox: thickness must be positive!");
    if (!(voxelSize > 0))
        throw std::runtime_error("BBox: voxelSize must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth * voxelSize);
    auto acc = builder->getAccessor();

    // Define size of bbox with narrow-band in voxel units
    const Vec3T  r(width / (2 * voxelSize), height / (2 * voxelSize), depth / (2 * voxelSize));
    const ValueT e = thickness / voxelSize;

    // Below the Nyquist frequency
    if (r.min() < ValueT(1.5) || e < ValueT(1.5))
        return builder;

    // Define center of bbox in voxel units
    const Vec3T c(ValueT(center[0] - origin[0]) / voxelSize,
                  ValueT(center[1] - origin[1]) / voxelSize,
                  ValueT(center[2] - origin[2]) / voxelSize);

    // Define utility functions
    auto Pos = [](ValueT x) { return x > 0 ? x : 0; };
    auto Neg = [](ValueT x) { return x < 0 ? x : 0; };

    // Define bounds of the voxel coordinates
    const BBox<Vec3T> b(c - r - Vec3T(e + halfWidth), c + r + Vec3T(e + halfWidth));
    const CoordBBox   bbox(Coord(Floor(b[0][0]), Floor(b[0][1]), Floor(b[0][2])),
                         Coord(Ceil(b[1][0]), Ceil(b[1][1]), Ceil(b[1][2])));

    // Compute signed distances to bbox using leapfrogging in k
    int m = 1;
    for (Coord p = bbox[0]; p[0] <= bbox[1][0]; ++p[0]) {
        const ValueT px = Abs(ValueT(p[0]) - c[0]) - r[0];
        const ValueT qx = Abs(ValueT(px) + e) - e;
        const ValueT px2 = Pow2(Pos(px));
        const ValueT qx2 = Pow2(Pos(qx));
        for (p[1] = bbox[0][1]; p[1] <= bbox[1][1]; ++p[1]) {
            const ValueT py = Abs(ValueT(p[1]) - c[1]) - r[1];
            const ValueT qy = Abs(ValueT(py) + e) - e;
            const ValueT qy2 = Pow2(Pos(qy));
            ;
            const ValueT px2qy2 = px2 + qy2;
            const ValueT qx2py2 = qx2 + Pow2(Pos(py));
            const ValueT qx2qy2 = qx2 + qy2;
            const ValueT a[3] = {Max(px, qy), Max(qx, py), Max(qx, qy)};
            for (p[2] = bbox[0][2]; p[2] <= bbox[1][2]; p[2] += m) {
                m = 1;
                const ValueT pz = Abs(ValueT(p[2]) - c[2]) - r[2];
                const ValueT qz = Abs(ValueT(pz) + e) - e;
                const ValueT qz2 = Pow2(Pos(qz));
                const ValueT s1 = Sqrt(px2qy2 + qz2) + Neg(Max(a[0], qz));
                const ValueT s2 = Sqrt(qx2py2 + qz2) + Neg(Max(a[1], qz));
                const ValueT s3 = Sqrt(qx2qy2 + Pow2(Pos(pz))) + Neg(Max(a[2], pz));
                const ValueT v = Min(s1, Min(s2, s3)); // Distance in voxel units
                const ValueT d = Abs(v);
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(p, voxelSize * v); // distance in world units
                } else { // outside narrow band
                    m += Floor(d - halfWidth); // leapfrog
                }
            } //end leapfrog over k
        } //end loop over j
    } //end loop over i

    return builder;
} // initBBox

template<typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initOctahedron(ValueT       scale, // scale of the octahedron in world units
               const Vec3d& center, //center of octahedron in world units
               ValueT       voxelSize, // size of a voxel in world units
               ValueT       halfWidth, // half-width of narrow band in voxel units
               const Vec3d& origin) // origin of grid in world units
{
    using Vec3T = Vec3<ValueT>;
    static_assert(is_floating_point<ValueT>::value, "Octahedron: expect floating point");
    if (!(scale > 0))
        throw std::runtime_error("Octahedron: width must be positive!");
    if (!(voxelSize > 0))
        throw std::runtime_error("Octahedron: voxelSize must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth * voxelSize);
    auto acc = builder->getAccessor();

    // Define size of octahedron with narrow-band in voxel units
    const ValueT s = scale / (2 * voxelSize);

    // Below the Nyquist frequency
    if ( s < ValueT(1.5) )
        return builder;

    // Define center of octahedron in voxel units
    const Vec3T c(ValueT(center[0] - origin[0]) / voxelSize,
                  ValueT(center[1] - origin[1]) / voxelSize,
                  ValueT(center[2] - origin[2]) / voxelSize);

    // Define utility functions
    auto sdf = [&s](ValueT x, ValueT y, ValueT z) {
        const ValueT d = ValueT(0.5)*(z - y + s);
        if (d < ValueT(0)) {
            return Vec3T(x, y - s, z).length();
        } else if (d > s) {
            return Vec3T(x, y, z - s).length(); 
        }
        return Vec3T(x, y - s + d, z - d).length();
    };

    // Define bounds of the voxel coordinates
    const BBox<Vec3T> b(c - Vec3T(s + halfWidth), c + Vec3T(s + halfWidth));
    const CoordBBox   bbox(Coord(Floor(b[0][0]), Floor(b[0][1]), Floor(b[0][2])),
                           Coord(Ceil(b[1][0]),  Ceil(b[1][1]),  Ceil(b[1][2])));

    /*
   float sdOctahedron( vec3 p, float s)
   {
       p = abs(p);
       float m = p.x+p.y+p.z-s;
       vec3 q;
       if( 3.0*p.x < m ) q = p.xyz;
       else if( 3.0*p.y < m ) q = p.yzx;
       else if( 3.0*p.z < m ) q = p.zxy;
       else return m*0.57735027;
    
       float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
       return length(vec3(q.x,q.y-s+k,q.z-k)); 
    }
    */

    // Compute signed distances to octahedron using leapfrogging in k
    int m = 1;
    static const ValueT a = Sqrt(ValueT(1)/ValueT(3));
    for (Coord p = bbox[0]; p[0] <= bbox[1][0]; ++p[0]) {
        const ValueT px = Abs(ValueT(p[0]) - c[0]);
        for (p[1] = bbox[0][1]; p[1] <= bbox[1][1]; ++p[1]) {
            const ValueT py = Abs(ValueT(p[1]) - c[1]);
            for (p[2] = bbox[0][2]; p[2] <= bbox[1][2]; p[2] += m) {
                m = 1;
                const ValueT pz = Abs(ValueT(p[2]) - c[2]);
                ValueT d =  px + py + pz - s;
                ValueT v;
                if (ValueT(3)*px < d) {
                    v = sdf(px, py, pz);
                } else if (ValueT(3)*py < d) {
                    v = sdf(py, pz, px);
                } else if (ValueT(3)*pz < d) {
                    v = sdf(pz, px, py);
                } else {
                    v = a * d; 
                }
                d = Abs(v);
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(p, voxelSize * v); // distance in world units
                } else { // outside narrow band
                    m += Floor(d - halfWidth); // leapfrog
                }
            } //end leapfrog over k
        } //end loop over j
    } //end loop over i

    return builder;
} // initOctahedron

} // unnamed namespace

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetSphere(ValueT             radius, // radius of sphere in world units
                     const Vec3d&       center, //center of sphere in world units
                     ValueT             voxelSize, // size of a voxel in world units
                     ValueT             halfWidth, // half-width of narrow band in voxel units
                     const Vec3d&       origin, // origin of grid in world units
                     const std::string& name, // name of grid
                     StatsMode          sMode, // mode of computation for the statistics
                     ChecksumMode       cMode, // mode of computation for the checksum
                     const BufferT&     buffer)
{
    auto builder = initSphere(radius, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createLevelSetSphere

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createFogVolumeSphere(ValueT             radius, // radius of sphere in world units
                      const Vec3d&       center, //center of sphere in world units
                      ValueT             voxelSize, // size of a voxel in world units
                      ValueT             halfWidth, // half-width of narrow band in voxel units
                      const Vec3d&       origin, // origin of grid in world units
                      const std::string& name, // name of grid
                      StatsMode          sMode, // mode of computation for the statistics
                      ChecksumMode       cMode, // mode of computation for the checksum
                      const BufferT&     buffer)
{
    auto builder = initSphere(radius, center, voxelSize, halfWidth, origin);
    builder->sdfToFog();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::FogVolume, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createFogVolumeSphere

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointSphere(int                pointsPerVoxel, // half-width of narrow band in voxel units
                  ValueT             radius, // radius of sphere in world units
                  const Vec3d&       center, //center of sphere in world units
                  ValueT             voxelSize, // size of a voxel in world units
                  const Vec3d&       origin, // origin of grid in world units
                  const std::string& name, // name of grid
                  ChecksumMode       cMode, // mode of computation for the checksum
                  const BufferT&     buffer)
{
    auto sphereHandle = createLevelSetSphere(radius, center, voxelSize, 0.5f, origin, "dummy", StatsMode::BBox, ChecksumMode::Disable, buffer);
    assert(sphereHandle);
    auto* sphereGrid = sphereHandle.template grid<ValueT>();
    assert(sphereGrid);
    auto pointHandle = createPointScatter(*sphereGrid, pointsPerVoxel, name, cMode, buffer);
    assert(pointHandle);
    return pointHandle;
} // createPointSphere

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetTorus(ValueT             majorRadius, // major radius of torus in world units
                    ValueT             minorRadius, // minor radius of torus in world units
                    const Vec3d&       center, //center of torus in world units
                    ValueT             voxelSize, // size of a voxel in world units
                    ValueT             halfWidth, // half-width of narrow band in voxel units
                    const Vec3d&       origin, // origin of grid in world units
                    const std::string& name, // name of grid
                    StatsMode          sMode, // mode of computation for the statistics
                    ChecksumMode       cMode, // mode of computation for the checksum
                    const BufferT&     buffer)
{
    auto builder = initTorus(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createLevelSetTorus

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createFogVolumeTorus(ValueT             majorRadius, // major radius of torus in world units
                     ValueT             minorRadius, // minor radius of torus in world units
                     const Vec3d&       center, //center of torus in world units
                     ValueT             voxelSize, // size of a voxel in world units
                     ValueT             halfWidth, // half-width of narrow band in voxel units
                     const Vec3d&       origin, // origin of grid in world units
                     const std::string& name, // name of grid
                     StatsMode          sMode, // mode of computation for the statistics
                     ChecksumMode       cMode, // mode of computation for the checksum
                     const BufferT&     buffer)
{
    auto builder = initTorus(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    builder->sdfToFog();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::FogVolume, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createFogVolumeTorus

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointTorus(int                pointsPerVoxel, // half-width of narrow band in voxel units
                 ValueT             majorRadius, // major radius of torus in world units
                 ValueT             minorRadius, // minor radius of torus in world units
                 const Vec3d&       center, //center of torus in world units
                 ValueT             voxelSize, // size of a voxel in world units
                 const Vec3d&       origin, // origin of grid in world units
                 const std::string& name, // name of grid
                 ChecksumMode       cMode, // mode of computation for the checksum
                 const BufferT&     buffer)
{
    auto torusHandle = createLevelSetTorus(majorRadius, minorRadius, center, voxelSize, 0.5f, origin, "dummy", StatsMode::BBox, ChecksumMode::Disable, buffer);
    assert(torusHandle);
    auto* torusGrid = torusHandle.template grid<ValueT>();
    assert(torusGrid);
    auto pointHandle = createPointScatter(*torusGrid, pointsPerVoxel, name, cMode, buffer);
    assert(pointHandle);
    return pointHandle;
} // createPointTorus

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetBox(ValueT             width, // width of box in world units
                  ValueT             height, // height of box in world units
                  ValueT             depth, // depth of box in world units
                  const Vec3d&       center, //center of box in world units
                  ValueT             voxelSize, // size of a voxel in world units
                  ValueT             halfWidth, // half-width of narrow band in voxel units
                  const Vec3d&       origin, // origin of grid in world units
                  const std::string& name, // name of grid
                  StatsMode          sMode, // mode of computation for the statistics
                  ChecksumMode       cMode, // mode of computation for the checksum
                  const BufferT&     buffer)
{
    auto builder = initBox(width, height, depth, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createLevelSetBox

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetOctahedron(ValueT             scale, // scale of the octahedron in world units
                         const Vec3d&       center, //center of box in world units
                         ValueT             voxelSize, // size of a voxel in world units
                         ValueT             halfWidth, // half-width of narrow band in voxel units
                         const Vec3d&       origin, // origin of grid in world units
                         const std::string& name, // name of grid
                         StatsMode          sMode, // mode of computation for the statistics
                         ChecksumMode       cMode, // mode of computation for the checksum
                         const BufferT&     buffer)
{
    auto builder = initOctahedron(scale, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createLevelSetOctahedron

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetBBox(ValueT             width, // width of bbox in world units
                   ValueT             height, // height of bbox in world units
                   ValueT             depth, // depth of bbox in world units
                   ValueT             thickness, // thickness of the wire in world units
                   const Vec3d&       center, //center of bbox in world units
                   ValueT             voxelSize, // size of a voxel in world units
                   ValueT             halfWidth, // half-width of narrow band in voxel units
                   const Vec3d&       origin, // origin of grid in world units
                   const std::string& name, // name of grid
                   StatsMode          sMode, // mode of computation for the statistics
                   ChecksumMode       cMode, // mode of computation for the checksum
                   const BufferT&     buffer)
{
    auto builder = initBBox(width, height, depth, thickness, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createLevelSetBBox

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createFogVolumeBox(ValueT             width, // width of box in world units
                   ValueT             height, // height of box in world units
                   ValueT             depth, // depth of box in world units
                   const Vec3d&       center, //center of box in world units
                   ValueT             voxelSize, // size of a voxel in world units
                   ValueT             halfWidth, // half-width of narrow band in voxel units
                   const Vec3d&       origin, // origin of grid in world units
                   const std::string& name, // name of grid
                   StatsMode          sMode, // mode of computation for the statistics
                   ChecksumMode       cMode, // mode of computation for the checksum
                   const BufferT&     buffer)
{
    auto builder = initBox(width, height, depth, center, voxelSize, halfWidth, origin);
    builder->sdfToFog();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::FogVolume, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createFogVolumeBox

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createFogVolumeOctahedron(ValueT             scale, // scale of octahedron in world units
                          const Vec3d&       center, //center of box in world units
                          ValueT             voxelSize, // size of a voxel in world units
                          ValueT             halfWidth, // half-width of narrow band in voxel units
                          const Vec3d&       origin, // origin of grid in world units
                          const std::string& name, // name of grid
                          StatsMode          sMode, // mode of computation for the statistics
                          ChecksumMode       cMode, // mode of computation for the checksum
                          const BufferT&     buffer)
{
    auto builder = initOctahedron(scale, center, voxelSize, halfWidth, origin);
    builder->sdfToFog();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::FogVolume, sMode, cMode, buffer);
    assert(handle);
    return handle;
} // createFogVolumeOctahedron

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointBox(int                pointsPerVoxel, // half-width of narrow band in voxel units
               ValueT             width, // width of box in world units
               ValueT             height, // height of box in world units
               ValueT             depth, // depth of box in world units
               const Vec3d&       center, //center of box in world units
               ValueT             voxelSize, // size of a voxel in world units
               const Vec3d&       origin, // origin of grid in world units
               const std::string& name, // name of grid
               ChecksumMode       cMode, // mode of computation for the checksum
               const BufferT&     buffer)
{
    auto boxHandle = createLevelSetBox(width, height, depth, center, voxelSize, 0.5f, origin, "dummy", StatsMode::BBox, ChecksumMode::Disable, buffer);
    assert(boxHandle);
    auto* boxGrid = boxHandle.template grid<ValueT>();
    assert(boxGrid);
    auto pointHandle = createPointScatter(*boxGrid, pointsPerVoxel, name, cMode, buffer);
    assert(pointHandle);
    return pointHandle;

} // createPointBox

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointScatter(const NanoGrid<ValueT>& srcGrid, // origin of grid in world units
                   int                     pointsPerVoxel, // half-width of narrow band in voxel units
                   const std::string&      name, // name of grid
                   ChecksumMode            cMode, // mode of computation for the checksum
                   const BufferT&          buffer)
{
    static_assert(is_floating_point<ValueT>::value, "createPointScatter: expect floating point");
    using Vec3T = Vec3<ValueT>;
    if (pointsPerVoxel < 1) {
        throw std::runtime_error("createPointScatter: Expected at least one point per voxel");
    }
    if (!srcGrid.isLevelSet()) {
        throw std::runtime_error("createPointScatter: Expected a level set grid");
    }
    if (!srcGrid.hasBBox()) {
        throw std::runtime_error("createPointScatter: ActiveVoxelCount is required");
    }
    const uint64_t pointCount = pointsPerVoxel * srcGrid.activeVoxelCount();
    const uint64_t pointSize = AlignUp<NANOVDB_DATA_ALIGNMENT>(pointCount * sizeof(Vec3T));
    if (pointCount == 0) {
        throw std::runtime_error("createPointScatter: No particles to scatter");
    }
    std::vector<Vec3T> xyz;
    xyz.reserve(pointCount);
    GridBuilder<uint32_t> builder(std::numeric_limits<uint32_t>::max(), pointSize);
    auto                  dstAcc = builder.getAccessor();
    std::srand(1234);
    const ValueT s = 1 / (1 + ValueT(RAND_MAX)); // scale so s*rand() is in ] 0, 1 [
    // return a point with random local voxel coordinates (-0.5 to +0.5)
    auto randomPoint = [&s]() {
        return s * Vec3T(rand(), rand(), rand()) - Vec3T(0.5);
    };
    const auto& srcTree = srcGrid.tree();
    for (uint32_t i = 0, end = srcTree.nodeCount(0); i < end; ++i) {
        auto* srcLeaf = srcTree.template getNode<0>(i);
        auto* dstLeaf = dstAcc.setValue(srcLeaf->origin(), pointsPerVoxel); // allocates leaf node
        dstLeaf->mValueMask = srcLeaf->valueMask();
        for (uint32_t j = 0, m = 0; j < 512; ++j) {
            if (dstLeaf->mValueMask.isOn(j)) {
                for (int n = 0; n < pointsPerVoxel; ++n, ++m) {
                    xyz.push_back(randomPoint());
                }
            }
            dstLeaf->mValues[j] = m;
        }
    }
    assert(pointCount == xyz.size());
    auto handle = builder.template getHandle<BufferT>(srcGrid.map(), name, GridClass::PointData, StatsMode::MinMax, ChecksumMode::Disable, buffer);
    assert(handle);
    auto* dstGrid = handle.template grid<uint32_t>();
    assert(dstGrid);
    auto& dstTree = dstGrid->tree();
    if (dstTree.nodeCount(0) == 0) {
        throw std::runtime_error("Expect leaf nodes!");
    }
    auto* leafData = const_cast<typename NanoLeaf<uint32_t>::DataType*>(dstTree.template getNode<0>(0u)->data());
    leafData[0].mMinimum = 0; // start of prefix sum
    for (uint32_t i = 1, n = dstTree.nodeCount(0); i < n; ++i) {
        leafData[i].mMinimum = leafData[i - 1].mMinimum + leafData[i - 1].mMaximum;
    }
    auto& meta = const_cast<GridBlindMetaData&>(dstGrid->blindMetaData(0u));
    meta.mByteOffset = handle.size() - pointSize; // offset from Grid to blind data
    meta.mElementCount = xyz.size();
    meta.mFlags = 0;
    meta.mDataClass = GridBlindDataClass::AttributeArray;
    meta.mSemantic = GridBlindDataSemantic::PointPosition;
    if (name.length() + 1 > GridBlindMetaData::MaxNameSize) {
        std::stringstream ss;
        ss << "Point attribute name \"" << name << "\" is more then "
           << nanovdb::GridBlindMetaData::MaxNameSize << " characters";
        throw std::runtime_error(ss.str());
    }
    memcpy(meta.mName, name.c_str(), name.size() + 1);
    if (std::is_same<ValueT, float>::value) { // resolved at compiletime
        meta.mDataType = GridType::Vec3f;
    } else if (std::is_same<ValueT, double>::value) {
        meta.mDataType = GridType::Vec3d;
    } else {
        throw std::runtime_error("Unsupported value type");
    }
    memcpy(handle.data() + meta.mByteOffset, xyz.data(), xyz.size() * sizeof(Vec3T));
    updateChecksum(*dstGrid, cMode);
    return handle;
} // createPointScatter

} // namespace nanovdb

#endif // NANOVDB_PRIMITIVES_H_HAS_BEEN_INCLUDED
