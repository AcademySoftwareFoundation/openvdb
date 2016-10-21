PROJECT ( OpenVDB_Houdini )

FIND_PACKAGE ( HDK REQUIRED )


# This is to work around DWA way of doing stuff - REPEATED WITH VARIATION
MESSAGE ( "PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}" )
MESSAGE ( "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}" )
FILE ( GLOB HOUDINI_UTILS_HEADER RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} houdini/*.h )

# This is to work around DWA way of doing stuff
SET  ( OPENVDB_HOUDINI_UTILS_DIR ${PROJECT_BINARY_DIR}/houdini_utils )
FILE ( MAKE_DIRECTORY ${OPENVDB_HOUDINI_UTILS_DIR} )
FILE ( COPY ${HOUDINI_UTILS_HEADER} DESTINATION ${OPENVDB_HOUDINI_UTILS_DIR} )

# This is to work around DWA way of doing stuff
SET  ( OPENVDB_HOUDINI_LOCAL_DIR ${PROJECT_BINARY_DIR}/openvdb_houdini )
FILE ( MAKE_DIRECTORY ${OPENVDB_HOUDINI_LOCAL_DIR} )
FILE ( COPY ${HOUDINI_UTILS_HEADER} DESTINATION ${OPENVDB_HOUDINI_LOCAL_DIR} )


INCLUDE_DIRECTORIES (
  # houdini
  ${OPENVDB_HOUDINI_UTILS_DIR}
  ${OPENVDB_HOUDINI_LOCAL_DIR}
  ${PROJECT_BINARY_DIR}
  ${HDK_INCLUDE_DIR}
  )

MESSAGE ( "HDK_DEFINITIONS = ${HDK_DEFINITIONS}" )

IF (NOT WIN32)
  ADD_DEFINITIONS ( -pthread -fPIC )
ENDIF ()

SET ( OPENVDB_HFS_INSTALL_BASE_DIR
  ${CMAKE_INSTALL_PREFIX}/houdini${HDK_VERSION_MAJOR}.${HDK_VERSION_MINOR}
  )

# RPath handling
IF ( OPENVDB_ENABLE_RPATH )

  # when building, don't use the install RPATH already
  # (but later on when installing)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

  # Use $ORIGIN for relative referencing
  # The second $ORIGIN is present because there is another shared library
  # specific to the SOPs that is not use elsewhere
  # SET(CMAKE_INSTALL_RPATH "$ORIGIN/../../lib:$ORIGIN")
  SET(CMAKE_INSTALL_RPATH "$ORIGIN")

ENDIF ( OPENVDB_ENABLE_RPATH )

ADD_LIBRARY ( openvdb_houdini SHARED
  houdini/GeometryUtil.cc
  houdini/GEO_PrimVDB.cc
  houdini/GT_GEOPrimCollectVDB.cc
  houdini/GU_PrimVDB.cc
  houdini/GU_VDBPointTools.cc
  houdini/geometry.cc
  houdini/ParmFactory.cc
  houdini/SOP_NodeVDB.cc
  houdini/UT_VDBUtils.cc
  houdini/Utils.cc
  )

# SOPs
HDK_ADD_LIBRARY ( SOP_OpenVDB_Advect
  houdini/SOP_OpenVDB_Advect.cc
  )

HDK_ADD_LIBRARY ( GEO_VDBTranslator
  houdini/GEO_VDBTranslator.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Advect_Points
  houdini/SOP_OpenVDB_Advect_Points.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Analysis
  houdini/SOP_OpenVDB_Analysis.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Clip
  houdini/SOP_OpenVDB_Clip.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Combine
  houdini/SOP_OpenVDB_Combine.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Convert
  houdini/SOP_OpenVDB_Convert.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Create
  houdini/SOP_OpenVDB_Create.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Diagnostics
  houdini/SOP_OpenVDB_Diagnostics.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Fill
  houdini/SOP_OpenVDB_Fill.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Filter
  houdini/SOP_OpenVDB_Filter.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Filter_Level_Set
  houdini/SOP_OpenVDB_Filter_Level_Set.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Fracture
  houdini/SOP_OpenVDB_Fracture.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_From_Particles
  houdini/SOP_OpenVDB_From_Particles.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_From_Polygons
  houdini/SOP_OpenVDB_From_Polygons.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_LOD
  houdini/SOP_OpenVDB_LOD.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Metadata
  houdini/SOP_OpenVDB_Metadata.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Morph_Level_Set
  houdini/SOP_OpenVDB_Morph_Level_Set.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Noise
  houdini/SOP_OpenVDB_Noise.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Occlusion_Mask
  houdini/SOP_OpenVDB_Occlusion_Mask.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Platonic
  houdini/SOP_OpenVDB_Platonic.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Prune
  houdini/SOP_OpenVDB_Prune.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Rasterize_Points
  houdini/SOP_OpenVDB_Rasterize_Points.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Ray
  houdini/SOP_OpenVDB_Ray.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Read
  houdini/SOP_OpenVDB_Read.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Rebuild_Level_Set
  houdini/SOP_OpenVDB_Rebuild_Level_Set.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Remap
  houdini/SOP_OpenVDB_Remap.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Remove_Divergence
  houdini/SOP_OpenVDB_Remove_Divergence.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Resample
  houdini/SOP_OpenVDB_Resample.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Sample_Points
  houdini/SOP_OpenVDB_Sample_Points.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Sort_Points
  houdini/SOP_OpenVDB_Sort_Points.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Scatter
  houdini/SOP_OpenVDB_Scatter.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Segment
  houdini/SOP_OpenVDB_Segment.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_To_Polygons
  houdini/SOP_OpenVDB_To_Polygons.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_To_Spheres
  houdini/SOP_OpenVDB_To_Spheres.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Topology_To_Level_Set
  houdini/SOP_OpenVDB_Topology_To_Level_Set.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Transform
  houdini/SOP_OpenVDB_Transform.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Vector_Merge
  houdini/SOP_OpenVDB_Vector_Merge.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Vector_Split
  houdini/SOP_OpenVDB_Vector_Split.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Visualize
  houdini/SOP_OpenVDB_Visualize.cc
  )

HDK_ADD_LIBRARY ( SOP_OpenVDB_Write
  houdini/SOP_OpenVDB_Write.cc
  )

# Linking
TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Advect
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( GEO_VDBTranslator
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Advect_Points
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Analysis
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Clip
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Combine
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Convert
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Create
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Diagnostics
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Fill
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Filter
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Filter_Level_Set
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Fracture
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_From_Particles
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_From_Polygons
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_LOD
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Metadata
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Morph_Level_Set
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Noise
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Occlusion_Mask
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Platonic
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Prune
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Rasterize_Points
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Ray
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Read
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Rebuild_Level_Set
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Remap
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Remove_Divergence
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Resample
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Sample_Points
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Sort_Points
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Scatter
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Segment
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_To_Polygons
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_To_Spheres
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Topology_To_Level_Set
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Transform
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Vector_Merge
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Vector_Split
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Visualize
  openvdb_houdini
  openvdb_static
  )

TARGET_LINK_LIBRARIES ( SOP_OpenVDB_Write
  openvdb_houdini
  openvdb_static
  )

# Installation
INSTALL ( FILES
  houdini/DW_OpenVDBRasterizePoints.cmd
  DESTINATION
  ${OPENVDB_HFS_INSTALL_BASE_DIR}/scripts/sop
  )

INSTALL ( TARGETS
  openvdb_houdini
  SOP_OpenVDB_Advect
  GEO_VDBTranslator
  SOP_OpenVDB_Advect_Points
  SOP_OpenVDB_Analysis
  SOP_OpenVDB_Clip
  SOP_OpenVDB_Combine
  SOP_OpenVDB_Convert
  SOP_OpenVDB_Create
  SOP_OpenVDB_Diagnostics
  SOP_OpenVDB_Fill
  SOP_OpenVDB_Filter
  SOP_OpenVDB_Filter_Level_Set
  SOP_OpenVDB_Fracture
  SOP_OpenVDB_From_Particles
  SOP_OpenVDB_From_Polygons
  SOP_OpenVDB_LOD
  SOP_OpenVDB_Metadata
  SOP_OpenVDB_Morph_Level_Set
  SOP_OpenVDB_Noise
  SOP_OpenVDB_Occlusion_Mask
  SOP_OpenVDB_Platonic
  SOP_OpenVDB_Prune
  SOP_OpenVDB_Rasterize_Points
  SOP_OpenVDB_Ray
  SOP_OpenVDB_Read
  SOP_OpenVDB_Rebuild_Level_Set
  SOP_OpenVDB_Remap
  SOP_OpenVDB_Remove_Divergence
  SOP_OpenVDB_Resample
  SOP_OpenVDB_Sample_Points
  SOP_OpenVDB_Sort_Points
  SOP_OpenVDB_Scatter
  SOP_OpenVDB_Segment
  SOP_OpenVDB_To_Polygons
  SOP_OpenVDB_To_Spheres
  SOP_OpenVDB_Topology_To_Level_Set
  SOP_OpenVDB_Transform
  SOP_OpenVDB_Vector_Merge
  SOP_OpenVDB_Vector_Split
  SOP_OpenVDB_Visualize
  SOP_OpenVDB_Write
  DESTINATION
  ${OPENVDB_HFS_INSTALL_BASE_DIR}/dso
  )
