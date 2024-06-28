// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/tools/CreatePrimitives.h

    \author Ken Museth

    \date June 26, 2020

    \brief Generates volumetric primitives, e.g. sphere, torus etc, as NanoVDB grid.

    \note This has no dependency on openvdb.
*/

#ifndef NANOVDB_TOOLS_PRIMITIVES_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_PRIMITIVES_H_HAS_BEEN_INCLUDED

#define NANOVDB_PARALLEL_PRIMITIVES

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/util/ForEach.h>// for util::forEach and util::Range

namespace nanovdb {

namespace tools {// ===================================================

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
/// @param tolerance Global error tolerance use when VoxelT = FpN
/// @param ditherOn  If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer    Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<BuildT, float,  double>::value, GridHandle<BufferT>>::type
createLevelSetSphere(double              radius = 100.0,
                     const Vec3d&        center = Vec3d(0),
                     double              voxelSize = 1.0,
                     double              halfWidth = 3.0,
                     const Vec3d&        origin = Vec3d(0),
                     const std::string&  name = "sphere_ls",
                     StatsMode           sMode = StatsMode::Default,
                     CheckMode           cMode = CheckMode::Default,
                     const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<BuildT, Fp4, Fp8, Fp16>::value, GridHandle<BufferT>>::type
createLevelSetSphere(double              radius = 100.0,
                     const Vec3d&        center = Vec3d(0),
                     double              voxelSize = 1.0,
                     double              halfWidth = 3.0,
                     const Vec3d&        origin = Vec3d(0),
                     const std::string&  name = "sphere_ls",
                     StatsMode           sMode = StatsMode::Default,
                     CheckMode           cMode = CheckMode::Default,
                     bool                ditherOn = false,
                     const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetSphere(double              radius = 100.0,
                     const Vec3d&        center = Vec3d(0),
                     double              voxelSize = 1.0,
                     double              halfWidth = 3.0,
                     const Vec3d&        origin = Vec3d(0),
                     const std::string&  name = "sphere_ls_FpN",
                     StatsMode           sMode = StatsMode::Default,
                     CheckMode           cMode = CheckMode::Default,
                     float               tolerance = -1.0f,
                     bool                ditherOn = false,
                     const BufferT&      buffer = BufferT());

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
/// @param tolerance Global error tolerance use when VoxelT = FpN
/// @param ditherOn  If true dithering will be applied when BuildT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer    Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeSphere(double              radius = 100.0,
                      const Vec3d&        center = Vec3d(0.0),
                      double              voxelSize = 1.0,
                      double              halfWidth = 3.0,
                      const Vec3d&        origin = Vec3d(0.0),
                      const std::string&  name = "sphere_fog",
                      StatsMode           sMode = StatsMode::Default,
                      CheckMode           cMode = CheckMode::Default,
                      const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeSphere(double              radius = 100.0,
                      const Vec3d&        center = Vec3d(0.0),
                      double              voxelSize = 1.0,
                      double              halfWidth = 3.0,
                      const Vec3d&        origin = Vec3d(0.0),
                      const std::string&  name = "sphere_fog",
                      StatsMode           sMode = StatsMode::Default,
                      CheckMode           cMode = CheckMode::Default,
                      float               tolerance = -1.0f,
                      bool                ditherOn = false,
                      const BufferT&      buffer = BufferT());

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
///
/// @details The @c BuildT template parameter must be float (default) or double.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createPointSphere(int                 pointsPerVoxel = 1,
                  double              radius = 100.0,
                  const Vec3d&        center = Vec3d(0.0),
                  double              voxelSize = 1.0,
                  const Vec3d&        origin = Vec3d(0.0),
                  const std::string&  name = "sphere_points",
                  CheckMode           mode = CheckMode::Default,
                  const BufferT&      buffer = BufferT());

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
/// @param tolerance   Global error tolerance use when VoxelT = FpN
/// @param ditherOn    If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer      Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetTorus(double              majorRadius = 100.0,
                    double              minorRadius = 50.0,
                    const Vec3d&        center = Vec3d(0.0),
                    double              voxelSize = 1.0,
                    double              halfWidth = 3.0,
                    const Vec3d&        origin = Vec3d(0.0),
                    const std::string&  name = "torus_ls",
                    StatsMode           sMode = StatsMode::Default,
                    CheckMode           cMode = CheckMode::Default,
                    const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetTorus(double              majorRadius = 100.0,
                    double              minorRadius = 50.0,
                    const Vec3d&        center = Vec3d(0.0),
                    double              voxelSize = 1.0,
                    double              halfWidth = 3.0,
                    const Vec3d&        origin = Vec3d(0.0),
                    const std::string&  name = "torus_ls",
                    StatsMode           sMode = StatsMode::Default,
                    CheckMode           cMode = CheckMode::Default,
                    float               tolerance = -1.0f,
                    bool                ditherOn = false,
                    const BufferT&      buffer = BufferT());

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
/// @param tolerance   Global error tolerance use when VoxelT = FpN
/// @param ditherOn    If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer      Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeTorus(double              majorRadius = 100.0,
                     double              minorRadius = 50.0,
                     const Vec3d&        center = Vec3d(0.0),
                     double              voxelSize = 1.0,
                     double              halfWidth = 3.0,
                     const Vec3d&        origin = Vec3d(0.0),
                     const std::string&  name = "torus_fog",
                     StatsMode           sMode = StatsMode::Default,
                     CheckMode           cMode = CheckMode::Default,
                     const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeTorus(double              majorRadius = 100.0,
                     double              minorRadius = 50.0,
                     const Vec3d&        center = Vec3d(0.0),
                     double              voxelSize = 1.0,
                     double              halfWidth = 3.0,
                     const Vec3d&        origin = Vec3d(0.0),
                     const std::string&  name = "torus_fog_FpN",
                     StatsMode           sMode = StatsMode::Default,
                     CheckMode           cMode = CheckMode::Default,
                     float               tolerance = -1.0f,
                     bool                ditherOn = false,
                     const BufferT&      buffer = BufferT());

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
/// @param cMode          Mode of computation for the checksum.
/// @param buffer         Buffer used for memory allocation by the handle
//
/// @details The @c BuildT template parameter must be float (default) or double.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createPointTorus(int                 pointsPerVoxel = 1, // half-width of narrow band in voxel units
                 double              majorRadius = 100.0, // major radius of torus in world units
                 double              minorRadius = 50.0, // minor radius of torus in world units
                 const Vec3d&        center = Vec3d(0.0), // center of torus in world units
                 double              voxelSize = 1.0, // size of a voxel in world units
                 const Vec3d&        origin = Vec3d(0.0f), // origin of grid in world units
                 const std::string&  name = "torus_points", // name of grid
                 CheckMode           cMode = CheckMode::Default,
                 const BufferT&      buffer = BufferT());

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
/// @param tolerance Global error tolerance use when VoxelT = FpN
/// @param ditherOn  If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer    Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBox(double              width = 40.0,
                  double              height = 60.0,
                  double              depth = 100.0,
                  const Vec3d& center = Vec3d(0.0),
                  double              voxelSize = 1.0,
                  double              halfWidth = 3.0,
                  const Vec3d&        origin = Vec3d(0.0),
                  const std::string&  name = "box_ls",
                  StatsMode           sMode = StatsMode::Default,
                  CheckMode           cMode = CheckMode::Default,
                  const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBox(double              width = 40.0,
                  double              height = 60.0,
                  double              depth = 100.0,
                  const Vec3d& center = Vec3d(0.0),
                  double              voxelSize = 1.0,
                  double              halfWidth = 3.0,
                  const Vec3d&        origin = Vec3d(0.0),
                  const std::string&  name = "box_ls_FpN",
                  StatsMode           sMode = StatsMode::Default,
                  CheckMode           cMode = CheckMode::Default,
                  float               tolerance = -1.0f,
                  bool                ditherOn = false,
                  const BufferT&      buffer = BufferT());

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
/// @param tolerance Global error tolerance use when VoxelT = FpN
/// @param ditherOn  If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer    Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeBox(double              width = 40.0,
                   double              height = 60.0,
                   double              depth = 100.0,
                   const Vec3d& center = Vec3d(0.0),
                   double              voxelSize = 1.0,
                   double              halfWidth = 3.0,
                   const Vec3d&        origin = Vec3d(0.0),
                   const std::string&  name = "box_fog",
                   StatsMode           sMode = StatsMode::Default,
                   CheckMode           cMode = CheckMode::Default,
                   const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeBox(double              width = 40.0,
                   double              height = 60.0,
                   double              depth = 100.0,
                   const Vec3d& center = Vec3d(0.0),
                   double              voxelSize = 1.0,
                   double              halfWidth = 3.0,
                   const Vec3d&        origin = Vec3d(0.0),
                   const std::string&  name = "box_fog_FpN",
                   StatsMode           sMode = StatsMode::Default,
                   CheckMode           cMode = CheckMode::Default,
                   float               tolerance = -1.0f,
                   bool                ditherOn = false,
                   const BufferT&      buffer = BufferT());

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
/// @param tolerance Global error tolerance use when VoxelT = FpN
/// @param ditherOn  If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer    Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetOctahedron(double              scale = 100.0,
                         const Vec3d&        center = Vec3d(0.0),
                         double              voxelSize = 1.0,
                         double              halfWidth = 3.0,
                         const Vec3d&        origin = Vec3d(0.0),
                         const std::string&  name = "octadedron_ls",
                         StatsMode           sMode = StatsMode::Default,
                         CheckMode           cMode = CheckMode::Default,
                         const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetOctahedron(double              scale = 100.0,
                         const Vec3d&        center = Vec3d(0.0),
                         double              voxelSize = 1.0,
                         double              halfWidth = 3.0,
                         const Vec3d&        origin = Vec3d(0.0),
                         const std::string&  name = "octadedron_ls_FpN",
                         StatsMode           sMode = StatsMode::Default,
                         CheckMode           cMode = CheckMode::Default,
                         float               tolerance = -1.0f,
                         bool                ditherOn = false,
                         const BufferT&      buffer = BufferT());

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
/// @param tolerance Global error tolerance use when VoxelT = FpN
/// @param ditherOn  If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer    Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeOctahedron(double              scale = 100.0,
                          const Vec3d& center = Vec3d(0.0),
                          double              voxelSize = 1.0,
                          double              halfWidth = 3.0,
                          const Vec3d&        origin = Vec3d(0.0),
                          const std::string&  name = "octadedron_fog",
                          StatsMode           sMode = StatsMode::Default,
                          CheckMode           cMode = CheckMode::Default,
                          const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeOctahedron(double              scale = 100.0,
                          const Vec3d& center = Vec3d(0.0),
                          double              voxelSize = 1.0,
                          double              halfWidth = 3.0,
                          const Vec3d&        origin = Vec3d(0.0),
                          const std::string&  name = "octadedron_fog_FpN",
                          StatsMode           sMode = StatsMode::Default,
                          CheckMode           cMode = CheckMode::Default,
                          float               tolerance = -1.0f,
                          bool                ditherOn = false,
                          const BufferT&      buffer = BufferT());

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
/// @param tolerance Global error tolerance use when VoxelT = FpN
/// @param ditherOn  If true dithering will be applied when VoxelT = {Fp4,Fp8,Fp16,FpN}
/// @param buffer    Buffer used for memory allocation by the handle
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN. The @c tolerance
///          argument is only used when BuildT is set to FpN.
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBBox(double              width = 40.0,
                   double              height = 60.0,
                   double              depth = 100.0,
                   double              thickness = 10.0,
                   const Vec3d&        center = Vec3d(0.0),
                   double              voxelSize = 1.0,
                   double              halfWidth = 3.0,
                   const Vec3d&        origin = Vec3d(0.0),
                   const std::string&  name = "bbox_ls",
                   StatsMode           sMode = StatsMode::Default,
                   CheckMode           cMode = CheckMode::Default,
                   const BufferT&      buffer = BufferT());

template<typename BuildT, typename BufferT = HostBuffer>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBBox(double              width = 40.0,
                   double              height = 60.0,
                   double              depth = 100.0,
                   double              thickness = 10.0,
                   const Vec3d&        center = Vec3d(0.0),
                   double              voxelSize = 1.0,
                   double              halfWidth = 3.0,
                   const Vec3d&        origin = Vec3d(0.0),
                   const std::string&  name = "bbox_ls_FpN",
                   StatsMode           sMode = StatsMode::Default,
                   CheckMode           cMode = CheckMode::Default,
                   float               tolerance = -1.0f,
                   bool                ditherOn = false,
                   const BufferT&      buffer = BufferT());


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
template<typename BuildT = float, typename BufferT = HostBuffer>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createPointBox(int                 pointsPerVoxel = 1, // half-width of narrow band in voxel units
               double              width = 40.0, // width of box in world units
               double              height = 60.0, // height of box in world units
               double              depth = 100.0, // depth of box in world units
               const Vec3d& center = Vec3d(0.0), // center of box in world units
               double              voxelSize = 1.0, // size of a voxel in world units
               const Vec3d&        origin = Vec3d(0.0), // origin of grid in world units
               const std::string&  name = "box_points", // name of grid
               CheckMode           mode = CheckMode::Default,
               const BufferT&      buffer = BufferT());

//================================================================================================

/// @brief Given an input NanoVDB voxel grid this methods returns a GridHandle to another NanoVDB
///        PointDataGrid with points scattered in the active leaf voxels of in input grid. Note, the
///        coordinates of the points are encoded as blind data in world-space.
///
/// @param srcGrid        Const input grid used to determine the active voxels to scatter points into
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param name           Name of the grid
/// @param mode           Mode of computation for the checksum.
/// @param buffer         Buffer used for memory allocation by the handle
template<typename SrcBuildT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointScatter(const NanoGrid<SrcBuildT>& srcGrid, // source grid used to scatter points into
                   int                        pointsPerVoxel = 1, // half-width of narrow band in voxel units
                   const std::string&         name = "point_scatter", // name of grid
                   CheckMode                  mode = CheckMode::Default,
                   const BufferT&             buffer = BufferT());

//================================================================================================

namespace {

/// @brief Returns a shared pointer to a build::Grid containing a narrow-band SDF values for a sphere
///
/// @brief Note, this is not (yet) a valid level set SDF field since values inside sphere (and outside
///        the narrow band) are still undefined. Call builder::sdfToLevelSet() to set those
///        values or alternatively call builder::levelSetToFog to generate a FOG volume.
///
/// @details The @c BuildT template parameter must be one of the following:
///          float (default), double, Fp4, Fp8, Fp16 or FpN.
template<typename BuildT>
std::shared_ptr<build::Grid<BuildT>>
initSphere(double              radius, // radius of sphere in world units
           const Vec3d&        center, // center of sphere in world units
           double              voxelSize, // size of a voxel in world units
           double              halfWidth, // half-width of narrow band in voxel units
           const Vec3d&        origin) // origin of grid in world units
{
    using GridT = build::Grid<BuildT>;
    using ValueT = typename BuildToValueMap<BuildT>::type;
    static_assert(util::is_floating_point<ValueT>::value, "initSphere: expect floating point");
    if (!(radius > 0))
        throw std::runtime_error("Sphere: radius must be positive!");
    if (!(voxelSize > 0))
        throw std::runtime_error("Sphere: voxelSize must be positive!");
    if (!(halfWidth > 0))
        throw std::runtime_error("Sphere: halfWidth must be positive!");

    auto grid = std::make_shared<GridT>(ValueT(halfWidth * voxelSize));
    grid->setTransform(voxelSize, origin);

    // Define radius of sphere with narrow-band in voxel units
    const ValueT r0 = radius / ValueT(voxelSize), rmax = r0 + ValueT(halfWidth);

    // Radius below the Nyquist frequency
    if (r0 < ValueT(1.5f)) return grid;

    // Define center of sphere in voxel units
    const math::Vec3<ValueT> c(ValueT(center[0] - origin[0]) / ValueT(voxelSize),
                               ValueT(center[1] - origin[1]) / ValueT(voxelSize),
                               ValueT(center[2] - origin[2]) / ValueT(voxelSize));

    // Define bounds of the voxel coordinates
    const int imin = math::Floor(c[0] - rmax), imax = math::Ceil(c[0] + rmax);
    const int jmin = math::Floor(c[1] - rmax), jmax = math::Ceil(c[1] + rmax);
    const int kmin = math::Floor(c[2] - rmax), kmax = math::Ceil(c[2] + rmax);

    const util::Range<1,int> range(imin, imax+1, 32);

    auto kernel = [&](const util::Range<1,int> &r) {
        auto acc = grid->getWriteAccessor();
        Coord ijk;
        int &i = ijk[0], &j = ijk[1], &k = ijk[2], m = 1;
        // Compute signed distances to sphere using leapfrogging in k
        for (i = r.begin(); i < r.end(); ++i) {
            const auto x2 = math::Pow2(ValueT(i) - c[0]);
            for (j = jmin; j <= jmax; ++j) {
                const auto x2y2 = math::Pow2(ValueT(j) - c[1]) + x2;
                for (k = kmin; k <= kmax; k += m) {
                    m = 1;
                    const auto v = math::Sqrt(x2y2 + math::Pow2(ValueT(k) - c[2])) - r0; // Distance in voxel units
                    const auto d = v < 0 ? -v : v;
                    if (d < halfWidth) { // inside narrow band
                        acc.setValue(ijk, ValueT(voxelSize) * v); // distance in world units
                    } else { // outside narrow band
                        m += math::Floor(d - halfWidth); // leapfrog
                    }
                } //end leapfrog over k
            } //end loop over j
        } //end loop over i
    };// kernel
#ifdef NANOVDB_PARALLEL_PRIMITIVES
    util::forEach(range, kernel);
#else
    kernel(range);
#endif
    return grid;
} // initSphere

template<typename BuildT>
std::shared_ptr<build::Grid<BuildT>>
initTorus(double              radius1, // major radius of torus in world units
          double              radius2, // minor radius of torus in world units
          const Vec3d&        center, // center of torus in world units
          double              voxelSize, // size of a voxel in world units
          double              halfWidth, // half-width of narrow band in voxel units
          const Vec3d&        origin) // origin of grid in world units
{
    using GridT = build::Grid<BuildT>;
    using ValueT = typename BuildToValueMap<BuildT>::type;
    static_assert(util::is_floating_point<ValueT>::value, "initTorus: expect floating point");
    if (!(radius2 > 0))
        throw std::runtime_error("Torus: radius2 must be positive!");
    if (!(radius1 > radius2))
        throw std::runtime_error("Torus: radius1 must be larger than radius2!");
    if (!(voxelSize > 0))
        throw std::runtime_error("Torus: voxelSize must be positive!");
    if (!(halfWidth > 0))
        throw std::runtime_error("Torus: halfWidth must be positive!");

    auto grid = std::make_shared<GridT>(ValueT(halfWidth * voxelSize));
    grid->setTransform(voxelSize, origin);

    // Define size of torus with narrow-band in voxel units
    const ValueT r1 = radius1 / ValueT(voxelSize), r2 = radius2 / ValueT(voxelSize), rmax1 = r1 + r2 + ValueT(halfWidth), rmax2 = r2 + ValueT(halfWidth);

    // Radius below the Nyquist frequency
    if (r2 < ValueT(1.5)) return grid;

    // Define center of torus in voxel units
    const math::Vec3<ValueT> c(ValueT(center[0] - origin[0]) / ValueT(voxelSize),
                               ValueT(center[1] - origin[1]) / ValueT(voxelSize),
                               ValueT(center[2] - origin[2]) / ValueT(voxelSize));

    // Define bounds of the voxel coordinates
    const int imin = math::Floor(c[0] - rmax1), imax = math::Ceil(c[0] + rmax1);
    const int jmin = math::Floor(c[1] - rmax2), jmax = math::Ceil(c[1] + rmax2);
    const int kmin = math::Floor(c[2] - rmax1), kmax = math::Ceil(c[2] + rmax1);

    const util::Range<1,int> range(imin, imax+1, 32);
    auto kernel = [&](const util::Range<1,int> &r) {
        auto acc = grid->getWriteAccessor();
        Coord ijk;
        int &i = ijk[0], &j = ijk[1], &k = ijk[2], m = 1;
        // Compute signed distances to torus using leapfrogging in k
        for (i = r.begin(); i < r.end(); ++i) {
            const auto x2 = math::Pow2(ValueT(i) - c[0]);
            for (k = kmin; k <= kmax; ++k) {
                const auto x2z2 = math::Pow2(math::Sqrt(math::Pow2(ValueT(k) - c[2]) + x2) - r1);
                for (j = jmin; j <= jmax; j += m) {
                    m = 1;
                    const auto v = math::Sqrt(x2z2 + math::Pow2(ValueT(j) - c[1])) - r2; // Distance in voxel units
                    const auto d = v < 0 ? -v : v;
                    if (d < halfWidth) { // inside narrow band
                        acc.setValue(ijk, ValueT(voxelSize) * v); // distance in world units
                    } else { // outside narrow band
                        m += math::Floor(d - halfWidth); // leapfrog
                    }
                } //end leapfrog over k
            } //end loop over j
        } //end loop over i
     }; // kernel

#ifdef NANOVDB_PARALLEL_PRIMITIVES
    util::forEach(range, kernel);
#else
    kernel(range);
#endif

    return grid;
} // initTorus

template<typename BuildT>
std::shared_ptr<build::Grid<BuildT>>
initBox(double       width, // major radius of torus in world units
        double       height, // minor radius of torus in world units
        double       depth,
        const Vec3d& center, // center of box in world units
        double       voxelSize, // size of a voxel in world units
        double       halfWidth, // half-width of narrow band in voxel units
        const Vec3d& origin) // origin of grid in world units
{
    using GridT = build::Grid<BuildT>;
    using ValueT = typename BuildToValueMap<BuildT>::type;
    static_assert(util::is_floating_point<ValueT>::value, "initBox: expect floating point");
    using Vec3T = math::Vec3<ValueT>;
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

    auto grid = std::make_shared<GridT>(ValueT(halfWidth * voxelSize));
    grid->setTransform(voxelSize, origin);

    // Define size of box with narrow-band in voxel units
    const Vec3T r(width  / (2 * ValueT(voxelSize)),
                  height / (2 * ValueT(voxelSize)),
                  depth  / (2 * ValueT(voxelSize)));

    // Below the Nyquist frequency
    if (r.min() < ValueT(1.5)) return grid;

    // Define center of box in voxel units
    const Vec3T c(ValueT(center[0] - origin[0]) / ValueT(voxelSize),
                  ValueT(center[1] - origin[1]) / ValueT(voxelSize),
                  ValueT(center[2] - origin[2]) / ValueT(voxelSize));

    // Define utility functions
    auto Pos = [](ValueT x) { return x > 0 ? x : 0; };
    auto Neg = [](ValueT x) { return x < 0 ? x : 0; };

    // Define bounds of the voxel coordinates
    const math::BBox<Vec3T> b(c - r - Vec3T(ValueT(halfWidth)), c + r + Vec3T(ValueT(halfWidth)));
    const CoordBBox   bbox(Coord(math::Floor(b[0][0]), math::Floor(b[0][1]), math::Floor(b[0][2])),
                           Coord(math::Ceil(b[1][0]),  math::Ceil(b[1][1]),  math::Ceil(b[1][2])));
    const util::Range<1,int> range(bbox[0][0], bbox[1][0]+1, 32);

    // Compute signed distances to box using leapfrogging in k
    auto kernel = [&](const util::Range<1,int> &ra) {
        auto acc = grid->getWriteAccessor();
        int m = 1;
        for (Coord p(ra.begin(),bbox[0][1],bbox[0][2]); p[0] < ra.end(); ++p[0]) {
            const auto q1 = math::Abs(ValueT(p[0]) - c[0]) - r[0];
            const auto x2 = math::Pow2(Pos(q1));
            for (p[1] = bbox[0][1]; p[1] <= bbox[1][1]; ++p[1]) {
                const auto q2 = math::Abs(ValueT(p[1]) - c[1]) - r[1];
                const auto q0 = math::Max(q1, q2);
                const auto x2y2 = x2 + math::Pow2(Pos(q2));
                for (p[2] = bbox[0][2]; p[2] <= bbox[1][2]; p[2] += m) {
                    m = 1;
                    const auto q3 = math::Abs(ValueT(p[2]) - c[2]) - r[2];
                    const auto v = math::Sqrt(x2y2 + math::Pow2(Pos(q3))) + Neg(math::Max(q0, q3)); // Distance in voxel units
                    const auto d = math::Abs(v);
                    if (d < halfWidth) { // inside narrow band
                        acc.setValue(p, ValueT(voxelSize) * v); // distance in world units
                    } else { // outside narrow band
                        m += math::Floor(d - halfWidth); // leapfrog
                    }
                } //end leapfrog over k
            } //end loop over j
        } //end loop over i
    }; // kernel
#ifdef NANOVDB_PARALLEL_PRIMITIVES
    util::forEach(range, kernel);
#else
    kernel(range);
#endif
    return grid;
} // initBox

template<typename BuildT>
std::shared_ptr<build::Grid<BuildT>>
initBBox(double       width, // width of the bbox in world units
         double       height, // height of the bbox in world units
         double       depth, // depth of the bbox in world units
         double       thickness, // thickness of the wire in world units
         const Vec3d& center, // center of bbox in world units
         double       voxelSize, // size of a voxel in world units
         double       halfWidth, // half-width of narrow band in voxel units
         const Vec3d& origin) // origin of grid in world units
{
    using GridT = build::Grid<BuildT>;
    using ValueT = typename BuildToValueMap<BuildT>::type;
    static_assert(util::is_floating_point<ValueT>::value, "initBBox: expect floating point");
    using Vec3T = math::Vec3<ValueT>;
    if (!(width > 0))
        throw std::runtime_error("BBox: width must be positive!");
    if (!(height > 0))
        throw std::runtime_error("BBox: height must be positive!");
    if (!(depth > 0))
        throw std::runtime_error("BBox: depth must be positive!");
    if (!(thickness > 0))
        throw std::runtime_error("BBox: thickness must be positive!");
    if (!(voxelSize > 0.0))
        throw std::runtime_error("BBox: voxelSize must be positive!");


    auto grid = std::make_shared<GridT>(ValueT(halfWidth * voxelSize));
    grid->setTransform(voxelSize, origin);

    // Define size of bbox with narrow-band in voxel units
    const Vec3T  r(width / (2 * ValueT(voxelSize)),
                  height / (2 * ValueT(voxelSize)),
                  depth  / (2 * ValueT(voxelSize)));
    const ValueT e = thickness / ValueT(voxelSize);

    // Below the Nyquist frequency
    if (r.min() < ValueT(1.5) || e < ValueT(1.5)) return grid;

    // Define center of bbox in voxel units
    const Vec3T c(ValueT(center[0] - origin[0]) / ValueT(voxelSize),
                  ValueT(center[1] - origin[1]) / ValueT(voxelSize),
                  ValueT(center[2] - origin[2]) / ValueT(voxelSize));

    // Define utility functions
    auto Pos = [](ValueT x) { return x > 0 ? x : 0; };
    auto Neg = [](ValueT x) { return x < 0 ? x : 0; };

    // Define bounds of the voxel coordinates
    const math::BBox<Vec3T> b(c - r - Vec3T(e + ValueT(halfWidth)), c + r + Vec3T(e + ValueT(halfWidth)));
    const CoordBBox   bbox(Coord(math::Floor(b[0][0]), math::Floor(b[0][1]), math::Floor(b[0][2])),
                           Coord(math::Ceil(b[1][0]),  math::Ceil(b[1][1]),  math::Ceil(b[1][2])));
    const util::Range<1,int> range(bbox[0][0], bbox[1][0]+1, 32);

    // Compute signed distances to bbox using leapfrogging in k
    auto kernel = [&](const util::Range<1,int> &ra) {
        auto acc = grid->getWriteAccessor();
        int m = 1;
        for (Coord p(ra.begin(),bbox[0][1],bbox[0][2]); p[0] < ra.end(); ++p[0]) {
            const ValueT px = math::Abs(ValueT(p[0]) - c[0]) - r[0];
            const ValueT qx = math::Abs(ValueT(px) + e) - e;
            const ValueT px2 = math::Pow2(Pos(px));
            const ValueT qx2 = math::Pow2(Pos(qx));
            for (p[1] = bbox[0][1]; p[1] <= bbox[1][1]; ++p[1]) {
                const ValueT py = math::Abs(ValueT(p[1]) - c[1]) - r[1];
                const ValueT qy = math::Abs(ValueT(py) + e) - e;
                const ValueT qy2 = math::Pow2(Pos(qy));
                const ValueT px2qy2 = px2 + qy2;
                const ValueT qx2py2 = qx2 + math::Pow2(Pos(py));
                const ValueT qx2qy2 = qx2 + qy2;
                const ValueT a[3] = {math::Max(px, qy), math::Max(qx, py), math::Max(qx, qy)};
                for (p[2] = bbox[0][2]; p[2] <= bbox[1][2]; p[2] += m) {
                    m = 1;
                    const ValueT pz = math::Abs(ValueT(p[2]) - c[2]) - r[2];
                    const ValueT qz = math::Abs(ValueT(pz) + e) - e;
                    const ValueT qz2 = math::Pow2(Pos(qz));
                    const ValueT s1 = math::Sqrt(px2qy2 + qz2) + Neg(math::Max(a[0], qz));
                    const ValueT s2 = math::Sqrt(qx2py2 + qz2) + Neg(math::Max(a[1], qz));
                    const ValueT s3 = math::Sqrt(qx2qy2 + math::Pow2(Pos(pz))) + Neg(math::Max(a[2], pz));
                    const ValueT v = math::Min(s1, math::Min(s2, s3)); // Distance in voxel units
                    const ValueT d = math::Abs(v);
                    if (d < halfWidth) { // inside narrow band
                        acc.setValue(p, ValueT(voxelSize) * v); // distance in world units
                    } else { // outside narrow band
                        m += math::Floor(d - halfWidth); // leapfrog
                    }
                } //end leapfrog over k
            } //end loop over j
        } //end loop over i
    }; //kernel
#ifdef NANOVDB_PARALLEL_PRIMITIVES
    util::forEach(range, kernel);
#else
    kernel(range);
#endif

    return grid;
} // initBBox

template<typename BuildT>
std::shared_ptr<build::Grid<BuildT>>
initOctahedron(double       scale, // scale of the octahedron in world units
               const Vec3d& center, // center of octahedron in world units
               double       voxelSize, // size of a voxel in world units
               double       halfWidth, // half-width of narrow band in voxel units
               const Vec3d& origin) // origin of grid in world units
{
    using GridT = build::Grid<BuildT>;
    using ValueT = typename BuildToValueMap<BuildT>::type;
    using Vec3T = math::Vec3<ValueT>;
    static_assert(util::is_floating_point<ValueT>::value, "initOctahedron: expect floating point");

    if (!(scale > 0)) throw std::runtime_error("Octahedron: width must be positive!");
    if (!(voxelSize > 0)) throw std::runtime_error("Octahedron: voxelSize must be positive!");

    auto grid = std::make_shared<GridT>(ValueT(halfWidth * voxelSize));
    grid->setTransform(voxelSize, origin);

    // Define size of octahedron with narrow-band in voxel units
    const ValueT s = scale / (2 * ValueT(voxelSize));

    // Below the Nyquist frequency
    if ( s < ValueT(1.5) ) return grid;

    // Define center of octahedron in voxel units
    const Vec3T c(ValueT(center[0] - origin[0]) / ValueT(voxelSize),
                  ValueT(center[1] - origin[1]) / ValueT(voxelSize),
                  ValueT(center[2] - origin[2]) / ValueT(voxelSize));

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
    const math::BBox<Vec3T> b(c - Vec3T(s + ValueT(halfWidth)), c + Vec3T(s + ValueT(halfWidth)));
    const CoordBBox   bbox(Coord(math::Floor(b[0][0]), math::Floor(b[0][1]), math::Floor(b[0][2])),
                           Coord(math::Ceil(b[1][0]),  math::Ceil(b[1][1]),  math::Ceil(b[1][2])));
    const util::Range<1,int> range(bbox[0][0], bbox[1][0]+1, 32);

    // Compute signed distances to octahedron using leapfrogging in k
    auto kernel = [&](const util::Range<1,int> &ra) {
        auto acc = grid->getWriteAccessor();
        int m = 1;
        static const ValueT a = math::Sqrt(ValueT(1)/ValueT(3));
        for (Coord p(ra.begin(),bbox[0][1],bbox[0][2]); p[0] < ra.end(); ++p[0]) {
            const ValueT px = math::Abs(ValueT(p[0]) - c[0]);
            for (p[1] = bbox[0][1]; p[1] <= bbox[1][1]; ++p[1]) {
                const ValueT py = math::Abs(ValueT(p[1]) - c[1]);
                for (p[2] = bbox[0][2]; p[2] <= bbox[1][2]; p[2] += m) {
                    m = 1;
                    const ValueT pz = math::Abs(ValueT(p[2]) - c[2]);
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
                    d = math::Abs(v);
                    if (d < halfWidth) { // inside narrow band
                        acc.setValue(p, ValueT(voxelSize) * v); // distance in world units
                    } else { // outside narrow band
                        m += math::Floor(d - halfWidth); // leapfrog
                    }
                } //end leapfrog over k
            } //end loop over j
        } //end loop over i
     };// kernel
#ifdef NANOVDB_PARALLEL_PRIMITIVES
    util::forEach(range, kernel);
#else
    kernel(range);
#endif
    return grid;
} // initOctahedron

} // unnamed namespace

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<BuildT, float, double>::value, GridHandle<BufferT>>::type
createLevelSetSphere(double              radius, // radius of sphere in world units
                     const Vec3d&        center, // center of sphere in world units
                     double              voxelSize, // size of a voxel in world units
                     double              halfWidth, // half-width of narrow band in voxel units
                     const Vec3d&        origin, // origin of grid in world units
                     const std::string&  name, // name of grid
                     StatsMode           sMode, // mode of computation for the statistics
                     CheckMode           cMode, // mode of computation for the checksum
                     const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initSphere<BuildT>(radius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createLevelSetSphere<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<BuildT, Fp4, Fp8, Fp16>::value, GridHandle<BufferT>>::type
createLevelSetSphere(double              radius, // radius of sphere in world units
                     const Vec3d&        center, // center of sphere in world units
                     double              voxelSize, // size of a voxel in world units
                     double              halfWidth, // half-width of narrow band in voxel units
                     const Vec3d&        origin, // origin of grid in world units
                     const std::string&  name, // name of grid
                     StatsMode           sMode, // mode of computation for the statistics
                     CheckMode           cMode, // mode of computation for the checksum
                     bool                ditherOn,
                     const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initSphere<BuildT>(radius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createLevelSetSphere<Fp4 or Fp8 or Fp16>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetSphere(double              radius, // radius of sphere in world units
                     const Vec3d&        center, // center of sphere in world units
                     double              voxelSize, // size of a voxel in world units
                     double              halfWidth, // half-width of narrow band in voxel units
                     const Vec3d&        origin, // origin of grid in world units
                     const std::string&  name, // name of grid
                     StatsMode           sMode, // mode of computation for the statistics
                     CheckMode           cMode, // mode of computation for the checksum
                     float               tolerance,// only used if VoxelT = FpN
                     bool                ditherOn,
                     const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initSphere<BuildT>(radius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createLevelSetSphere<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeSphere(double              radius, // radius of sphere in world units
                      const Vec3d&        center, // center of sphere in world units
                      double              voxelSize, // size of a voxel in world units
                      double              halfWidth, // half-width of narrow band in voxel units
                      const Vec3d&        origin, // origin of grid in world units
                      const std::string&  name, // name of grid
                      StatsMode           sMode, // mode of computation for the statistics
                      CheckMode           cMode, // mode of computation for the checksum
                      const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initSphere<BuildT>(radius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createFogVolumeSphere<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeSphere(double              radius, // radius of sphere in world units
                      const Vec3d&        center, // center of sphere in world units
                      double              voxelSize, // size of a voxel in world units
                      double              halfWidth, // half-width of narrow band in voxel units
                      const Vec3d&        origin, // origin of grid in world units
                      const std::string&  name, // name of grid
                      StatsMode           sMode, // mode of computation for the statistics
                      CheckMode           cMode, // mode of computation for the checksum
                      float               tolerance,// only used if VoxelT = FpN
                      bool                ditherOn,
                      const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initSphere<BuildT>(radius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createFogVolumeSphere<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createPointSphere(int                 pointsPerVoxel, // number of points to be scattered in each active voxel
                  double              radius, // radius of sphere in world units
                  const Vec3d&        center, // center of sphere in world units
                  double              voxelSize, // size of a voxel in world units
                  const Vec3d&        origin, // origin of grid in world units
                  const std::string&  name, // name of grid
                  CheckMode           cMode, // mode of computation for the checksum
                  const BufferT&      buffer)
{
    auto sphereHandle = createLevelSetSphere(radius, center, voxelSize, 0.5, origin, "dummy",
                                             StatsMode::BBox, CheckMode::Disable, buffer);
    assert(sphereHandle);
    auto* sphereGrid = sphereHandle.template grid<BuildT>();
    assert(sphereGrid);
    auto pointHandle = createPointScatter(*sphereGrid, pointsPerVoxel, name, cMode, buffer);
    assert(pointHandle);
    return pointHandle;
} // createPointSphere

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetTorus(double              majorRadius, // major radius of torus in world units
                    double              minorRadius, // minor radius of torus in world units
                    const Vec3d&        center, // center of torus in world units
                    double              voxelSize, // size of a voxel in world units
                    double              halfWidth, // half-width of narrow band in voxel units
                    const Vec3d&        origin, // origin of grid in world units
                    const std::string&  name, // name of grid
                    StatsMode           sMode, // mode of computation for the statistics
                    CheckMode           cMode, // mode of computation for the checksum
                    const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initTorus<BuildT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createLevelSetTorus<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetTorus(double              majorRadius, // major radius of torus in world units
                    double              minorRadius, // minor radius of torus in world units
                    const Vec3d&        center, // center of torus in world units
                    double              voxelSize, // size of a voxel in world units
                    double              halfWidth, // half-width of narrow band in voxel units
                    const Vec3d&        origin, // origin of grid in world units
                    const std::string&  name, // name of grid
                    StatsMode           sMode, // mode of computation for the statistics
                    CheckMode           cMode, // mode of computation for the checksum
                    float               tolerance,
                    bool                ditherOn,
                    const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initTorus<BuildT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createLevelSetTorus<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeTorus(double              majorRadius, // major radius of torus in world units
                     double              minorRadius, // minor radius of torus in world units
                     const Vec3d&        center, // center of torus in world units
                     double              voxelSize, // size of a voxel in world units
                     double              halfWidth, // half-width of narrow band in voxel units
                     const Vec3d&        origin, // origin of grid in world units
                     const std::string&  name, // name of grid
                     StatsMode           sMode, // mode of computation for the statistics
                     CheckMode           cMode, // mode of computation for the checksum
                     const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initTorus<BuildT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createFogVolumeTorus<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeTorus(double              majorRadius, // major radius of torus in world units
                     double              minorRadius, // minor radius of torus in world units
                     const Vec3d&        center, // center of torus in world units
                     double              voxelSize, // size of a voxel in world units
                     double              halfWidth, // half-width of narrow band in voxel units
                     const Vec3d&        origin, // origin of grid in world units
                     const std::string&  name, // name of grid
                     StatsMode           sMode, // mode of computation for the statistics
                     CheckMode           cMode, // mode of computation for the checksum
                     float               tolerance,
                     bool                ditherOn,
                     const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initTorus<BuildT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createFogVolumeTorus<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createPointTorus(int                 pointsPerVoxel, // number of points to be scattered in each active voxel
                 double              majorRadius, // major radius of torus in world units
                 double              minorRadius, // minor radius of torus in world units
                 const Vec3d&        center, // center of torus in world units
                 double              voxelSize, // size of a voxel in world units
                 const Vec3d&        origin, // origin of grid in world units
                 const std::string&  name, // name of grid
                 CheckMode           cMode, // mode of computation for the checksum
                 const BufferT&      buffer)
{
    auto torusHandle = createLevelSetTorus(majorRadius, minorRadius, center, voxelSize, 0.5f, origin,
                                           "dummy", StatsMode::BBox, CheckMode::Disable, buffer);
    assert(torusHandle);
    auto* torusGrid = torusHandle.template grid<BuildT>();
    assert(torusGrid);
    auto pointHandle = createPointScatter(*torusGrid, pointsPerVoxel, name, cMode, buffer);
    assert(pointHandle);
    return pointHandle;
} // createPointTorus<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBox(double              width, // width of box in world units
                  double              height, // height of box in world units
                  double              depth, // depth of box in world units
                  const Vec3d&        center, // center of box in world units
                  double              voxelSize, // size of a voxel in world units
                  double              halfWidth, // half-width of narrow band in voxel units
                  const Vec3d&        origin, // origin of grid in world units
                  const std::string&  name, // name of grid
                  StatsMode           sMode, // mode of computation for the statistics
                  CheckMode           cMode, // mode of computation for the checksum
                  const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initBox<BuildT>(width, height, depth, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createLevelSetBox<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBox(double              width, // width of box in world units
                  double              height, // height of box in world units
                  double              depth, // depth of box in world units
                  const Vec3d&        center, // center of box in world units
                  double              voxelSize, // size of a voxel in world units
                  double              halfWidth, // half-width of narrow band in voxel units
                  const Vec3d&        origin, // origin of grid in world units
                  const std::string&  name, // name of grid
                  StatsMode           sMode, // mode of computation for the statistics
                  CheckMode           cMode, // mode of computation for the checksum
                  float               tolerance,
                  bool                ditherOn,
                  const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initBox<BuildT>(width, height, depth, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createLevelSetBox<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetOctahedron(double              scale, // scale of the octahedron in world units
                         const Vec3d&        center, // center of box in world units
                         double              voxelSize, // size of a voxel in world units
                         double              halfWidth, // half-width of narrow band in voxel units
                         const Vec3d&        origin, // origin of grid in world units
                         const std::string&  name, // name of grid
                         StatsMode           sMode, // mode of computation for the statistics
                         CheckMode           cMode, // mode of computation for the checksum
                         const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initOctahedron<BuildT>(scale, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createLevelSetOctahedron<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetOctahedron(double              scale, // scale of the octahedron in world units
                         const Vec3d&        center, // center of box in world units
                         double              voxelSize, // size of a voxel in world units
                         double              halfWidth, // half-width of narrow band in voxel units
                         const Vec3d&        origin, // origin of grid in world units
                         const std::string&  name, // name of grid
                         StatsMode           sMode, // mode of computation for the statistics
                         CheckMode           cMode, // mode of computation for the checksum
                         float               tolerance,
                         bool                ditherOn,
                         const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initOctahedron<BuildT>(scale, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createLevelSetOctahedron<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBBox(double              width, // width of bbox in world units
                   double              height, // height of bbox in world units
                   double              depth, // depth of bbox in world units
                   double              thickness, // thickness of the wire in world units
                   const Vec3d&        center, // center of bbox in world units
                   double              voxelSize, // size of a voxel in world units
                   double              halfWidth, // half-width of narrow band in voxel units
                   const Vec3d&        origin, // origin of grid in world units
                   const std::string&  name, // name of grid
                   StatsMode           sMode, // mode of computation for the statistics
                   CheckMode           cMode, // mode of computation for the checksum
                   const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initBBox<BuildT>(width, height, depth, thickness, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createLevelSetBBox<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createLevelSetBBox(double              width, // width of bbox in world units
                   double              height, // height of bbox in world units
                   double              depth, // depth of bbox in world units
                   double              thickness, // thickness of the wire in world units
                   const Vec3d&        center, // center of bbox in world units
                   double              voxelSize, // size of a voxel in world units
                   double              halfWidth, // half-width of narrow band in voxel units
                   const Vec3d&        origin, // origin of grid in world units
                   const std::string&  name, // name of grid
                   StatsMode           sMode, // mode of computation for the statistics
                   CheckMode           cMode, // mode of computation for the checksum
                   float               tolerance,
                   bool                ditherOn,
                   const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initBBox<BuildT>(width, height, depth, thickness, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createLevelSetBBox<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeBox(double              width, // width of box in world units
                   double              height, // height of box in world units
                   double              depth, // depth of box in world units
                   const Vec3d&        center, // center of box in world units
                   double              voxelSize, // size of a voxel in world units
                   double              halfWidth, // half-width of narrow band in voxel units
                   const Vec3d&        origin, // origin of grid in world units
                   const std::string&  name, // name of grid
                   StatsMode           sMode, // mode of computation for the statistics
                   CheckMode           cMode, // mode of computation for the checksum
                   const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initBox<BuildT>(width, height, depth, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createFogVolumeBox<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeBox(double              width, // width of box in world units
                   double              height, // height of box in world units
                   double              depth, // depth of box in world units
                   const Vec3d&        center, // center of box in world units
                   double              voxelSize, // size of a voxel in world units
                   double              halfWidth, // half-width of narrow band in voxel units
                   const Vec3d&        origin, // origin of grid in world units
                   const std::string&  name, // name of grid
                   StatsMode           sMode, // mode of computation for the statistics
                   CheckMode           cMode, // mode of computation for the checksum
                   float               tolerance,
                   bool                ditherOn,
                   const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initBox<BuildT>(width, height, depth, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createFogVolumeBox<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeOctahedron(double              scale, // scale of octahedron in world units
                          const Vec3d&        center, // center of box in world units
                          double              voxelSize, // size of a voxel in world units
                          double              halfWidth, // half-width of narrow band in voxel units
                          const Vec3d&        origin, // origin of grid in world units
                          const std::string&  name, // name of grid
                          StatsMode           sMode, // mode of computation for the statistics
                          CheckMode           cMode, // mode of computation for the checksum
                          const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initOctahedron<BuildT>(scale, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    auto handle = converter.template getHandle<BuildT, BufferT>(buffer);
    assert(handle);
    return handle;
} // createFogVolumeOctahedron<T>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::enable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createFogVolumeOctahedron(double              scale, // scale of octahedron in world units
                          const Vec3d&        center, // center of box in world units
                          double              voxelSize, // size of a voxel in world units
                          double              halfWidth, // half-width of narrow band in voxel units
                          const Vec3d&        origin, // origin of grid in world units
                          const std::string&  name, // name of grid
                          StatsMode           sMode, // mode of computation for the statistics
                          CheckMode           cMode, // mode of computation for the checksum
                          float               tolerance,
                          bool                ditherOn,
                          const BufferT&      buffer)
{
    using GridT = build::Grid<BuildT>;
    auto grid = initOctahedron<BuildT>(scale, center, voxelSize, halfWidth, origin);
    grid->mName = name;
    build::NodeManager<GridT> mgr(*grid);
    build::sdfToLevelSet(mgr);
    build::levelSetToFog(mgr, false);
    CreateNanoGrid<GridT> converter(*grid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    AbsDiff oracle(tolerance);
    auto handle = converter.template getHandle<BuildT, AbsDiff, BufferT>(oracle, buffer);
    assert(handle);
    return handle;
} // createFogVolumeOctahedron<FpN>

//================================================================================================

template<typename BuildT, typename BufferT>
typename util::disable_if<util::is_same<FpN, BuildT>::value, GridHandle<BufferT>>::type
createPointBox(int                 pointsPerVoxel, // number of points to be scattered in each active voxel
               double              width, // width of box in world units
               double              height, // height of box in world units
               double              depth, // depth of box in world units
               const Vec3d&        center, // center of box in world units
               double              voxelSize, // size of a voxel in world units
               const Vec3d&        origin, // origin of grid in world units
               const std::string&  name, // name of grid
               CheckMode           cMode, // mode of computation for the checksum
               const BufferT&      buffer)
{
    auto boxHandle = createLevelSetBox(width, height, depth, center, voxelSize, 0.5, origin, "dummy",
                                       StatsMode::BBox, CheckMode::Disable, buffer);
    assert(boxHandle);
    auto* boxGrid = boxHandle.template grid<BuildT>();
    assert(boxGrid);
    auto pointHandle = createPointScatter(*boxGrid, pointsPerVoxel, name, cMode, buffer);
    assert(pointHandle);
    return pointHandle;
} // createPointBox<T>

//================================================================================================

template<typename SrcBuildT, typename BufferT>
inline GridHandle<BufferT>
createPointScatter(const NanoGrid<SrcBuildT>& srcGrid, // origin of grid in world units
                   int                        pointsPerVoxel, // number of points to be scattered in each active voxel
                   const std::string&         name, // name of grid
                   CheckMode                  cMode, // mode of computation for the checksum
                   const BufferT&             buffer)
{
    using ValueT = typename BuildToValueMap<SrcBuildT>::type;
    static_assert(util::is_floating_point<ValueT>::value, "createPointScatter: expect floating point");
    using Vec3T = math::Vec3<ValueT>;
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
    if (pointCount == 0) {
        throw std::runtime_error("createPointScatter: No particles to scatter");
    }
    std::vector<Vec3T> xyz;
    xyz.reserve(pointCount);
    using DstGridT = build::Grid<uint32_t>;
    DstGridT dstGrid(std::numeric_limits<uint32_t>::max(), name, GridClass::PointData);
    dstGrid.mMap = srcGrid.map();
    auto dstAcc = dstGrid.getAccessor();
    std::srand(1234);
    const ValueT s = 1 / (1 + ValueT(RAND_MAX)); // scale so s*rand() is in ] 0, 1 [
    // return a point with random local voxel coordinates (-0.5 to +0.5)
    auto randomPoint = [&s](){return s * Vec3T(rand(), rand(), rand()) - Vec3T(0.5);};
    const auto& srcTree = srcGrid.tree();
    auto srcMgrHandle = createNodeManager(srcGrid);
    auto *srcMgr = srcMgrHandle.template mgr<SrcBuildT>();
    assert(srcMgr);
    for (uint32_t i = 0, end = srcTree.nodeCount(0); i < end; ++i) {
        auto& srcLeaf = srcMgr->leaf(i);
        auto* dstLeaf = dstAcc.setValue(srcLeaf.origin(), pointsPerVoxel); // allocates leaf node
        dstLeaf->mValueMask = srcLeaf.valueMask();
        for (uint32_t j = 0, m = 0; j < 512; ++j) {
            if (dstLeaf->mValueMask.isOn(j)) {
                const Vec3f ijk = dstLeaf->offsetToGlobalCoord(j).asVec3s();// floating-point representatrion of index coorindates
                for (int n = 0; n < pointsPerVoxel; ++n) xyz.push_back(srcGrid.indexToWorld(randomPoint() + ijk));
                m += pointsPerVoxel;
            }// active voxels
            dstLeaf->mValues[j] = m;
        }// loop over all voxels
    }// loop over leaf nodes
    assert(pointCount == xyz.size());
    CreateNanoGrid<DstGridT> converter(dstGrid);
    converter.setStats(StatsMode::MinMax);
    converter.setChecksum(CheckMode::Disable);

    converter.addBlindData(name,
                           GridBlindDataSemantic::WorldCoords,
                           GridBlindDataClass::AttributeArray,
                           toGridType<Vec3T>(),
                           pointCount,
                           sizeof(Vec3T));
    auto handle = converter.template getHandle<uint32_t>(buffer);
    assert(handle);

    auto* grid = handle.template grid<uint32_t>();
    assert(grid && grid->template isSequential<0>());
    auto &tree = grid->tree();
    if (tree.nodeCount(0) == 0) throw std::runtime_error("Expect leaf nodes!");
    auto *leafData = tree.getFirstLeaf()->data();
    leafData[0].mMinimum = 0; // start of prefix sum
    for (uint32_t i = 1, n = tree.nodeCount(0); i < n; ++i) {
        leafData[i].mMinimum = leafData[i - 1].mMinimum + leafData[i - 1].mMaximum;
    }
    if (Vec3T *blindData = grid->template getBlindData<Vec3T>(0)) {
        memcpy(blindData, xyz.data(), xyz.size() * sizeof(Vec3T));
    } else {
        throw std::runtime_error("Blind data pointer was NULL");
    }
    updateChecksum(grid, cMode);
    return handle;
} // createPointScatter

}// namespace tools

} // namespace nanovdb

#endif // NANOVDB_TOOLS_PRIMITIVES_H_HAS_BEEN_INCLUDED
