# yue.nicholas@gmail.com
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# ====================================================================
# use clang instead of gcc? (this must come before declaring the project)
option(USE_CLANG "Build application with clang" OFF)
if(USE_CLANG)
    SET (CMAKE_C_COMPILER   "/usr/bin/clang")
    SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")
    SET (CMAKE_AR      "/usr/bin/llvm-ar")
    SET (CMAKE_LINKER  "/usr/bin/llvm-ld")
    SET (CMAKE_NM      "/usr/bin/llvm-nm")
    SET (CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
    SET (CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
else(USE_CLANG)
    SET (CMAKE_C_COMPILER   "/usr/bin/gcc")
    SET (CMAKE_CXX_COMPILER "/usr/bin/g++")
    SET (CMAKE_AR      "/usr/bin/ar")
    SET (CMAKE_LINKER  "/usr/bin/ld")
    SET (CMAKE_NM      "/usr/bin/nm")
    SET (CMAKE_OBJDUMP "/usr/bin/objdump")
    SET (CMAKE_RANLIB  "/usr/bin/ranlib")
endif(USE_CLANG)

PROJECT ( OpenVDB )

# -----------------------------------------------------
# default build type
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -g -O0")
    add_definitions(-DDEBUG)
ENDIF(CMAKE_BUILD_TYPE MATCHES "Debug")

IF(CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -O3")
    add_definitions(-DNDEBUG)
ENDIF(CMAKE_BUILD_TYPE MATCHES "Release")
# -----------------------------------------------------

set(DESTDIR "/usr/local")
set(OPENVDB_INCLUDE_DIR "${DESTDIR}/include/openvdb")
set(OPENVDB_LIB_DIR "${DESTDIR}/lib/openvdb")
set(OPENVDB_BIN_DIR "${DESTDIR}/bin")
set(OPENVDB_PYTHON_LIB_DIR "${DESTDIR}/lib/python2.7/dist-packages")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/")

find_package(PkgConfig REQUIRED)

pkg_search_module(GLFW REQUIRED glfw3)

FIND_PACKAGE ( Boost REQUIRED COMPONENTS python iostreams system thread )

FIND_PACKAGE ( IlmBase REQUIRED )
FIND_PACKAGE ( OpenEXR REQUIRED )

FIND_PACKAGE ( TBB REQUIRED )
FIND_PACKAGE ( ZLIB REQUIRED )
FIND_PACKAGE ( OpenGL REQUIRED )
FIND_PACKAGE ( PythonLibs 2.7 REQUIRED )

INCLUDE_DIRECTORIES ( ${Boost_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${ILMBASE_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${TBB_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${OPENGL_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${ZLIB_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${GLFW_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${PYTHON_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( .. ) # So they can simulate referencing installed headers
INCLUDE_DIRECTORIES ( . ) # So they can simulate referencing installed headers

LINK_DIRECTORIES( ${PYTHON_LIBRARY_DIRS} )

ADD_DEFINITIONS (
  -DGL_GLEXT_PROTOTYPES=1
  -DOPENVDB_USE_GLFW_3
  -DOPENVDB_PRIVATE
  )
ADD_DEFINITIONS ( -pthread )

# Working around hardcoding of path at DWA
FILE ( GLOB VIEWER_SOURCE_FILES RELATIVE ${CMAKE_SOURCE_DIR} viewer/* )
# MESSAGE ( "VIEWER_SOURCE_FILES = ${VIEWER_SOURCE_FILES}" )
FILE ( MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer )
FILE ( COPY ${VIEWER_SOURCE_FILES} DESTINATION ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer )
INCLUDE_DIRECTORIES ( ${CMAKE_BINARY_DIR}/dwa )

# ---
# libopenvdb
SET ( OPENVDB_LIBRARY_SOURCE_FILES
  Grid.cc
  io/Archive.cc
  io/Compression.cc
  io/File.cc
  io/GridDescriptor.cc
  io/Queue.cc
  io/Stream.cc
  io/TempFile.cc
  math/Maps.cc
  math/Proximity.cc
  math/QuantizedUnitVec.cc
  math/Transform.cc
  metadata/Metadata.cc
  metadata/MetaMap.cc
  openvdb.cc
  Platform.cc
  util/Formats.cc
  util/Util.cc
  )
ADD_LIBRARY ( openvdb SHARED
  ${OPENVDB_LIBRARY_SOURCE_FILES}
  )
TARGET_LINK_LIBRARIES ( openvdb
  ${Boost_LIBRARIES}
  ${TBB_LIBRARIES}
  ${ILMBASE_LIBRARIES}
  ${ZLIB_LIBRARY}
  )
install(TARGETS openvdb DESTINATION ${OPENVDB_LIB_DIR})
install(DIRECTORY . DESTINATION ${OPENVDB_INCLUDE_DIR} FILES_MATCHING PATTERN "*.h")

# ---
# vdb_print
ADD_EXECUTABLE ( vdb_print
  cmd/openvdb_print/main.cc
  )
TARGET_LINK_LIBRARIES ( vdb_print
  openvdb
  )
install(TARGETS vdb_print DESTINATION ${OPENVDB_BIN_DIR})

# ---
# vdb_render
ADD_EXECUTABLE ( vdb_render
  cmd/openvdb_render/main.cc
  )
TARGET_LINK_LIBRARIES ( vdb_render
  openvdb
  ${ILMBASE_LIBRARIES}
  ${OPENEXR_LIBRARIES}
  )
install(TARGETS vdb_render DESTINATION ${OPENVDB_BIN_DIR})

# ---
# vdb_view
ADD_EXECUTABLE ( vdb_view
  cmd/openvdb_view/main.cc
  viewer/Camera.cc
  viewer/ClipBox.cc
  viewer/Font.cc
  viewer/RenderModules.cc
  viewer/Viewer.cc
  )
#MESSAGE ( "OPENGL_LIBRARIES = ${OPENGL_LIBRARIES}")
#MESSAGE ( "GLFW_LIBRARIES = ${GLFW_STATIC_LIBRARIES}")
TARGET_LINK_LIBRARIES ( vdb_view
  openvdb
  ${OPENGL_LIBRARIES}
  ${GLFW_STATIC_LIBRARIES}
  )
install(TARGETS vdb_view DESTINATION ${OPENVDB_BIN_DIR})

# ---
# openvdb_unittest
ADD_EXECUTABLE ( openvdb_unittest
  unittest/main.cc
  unittest/TestBBox.cc
  unittest/TestConjGradient.cc
  unittest/TestCoord.cc
  unittest/TestCpt.cc
  unittest/TestCurl.cc
  unittest/TestDense.cc
  unittest/TestDenseSparseTools.cc
  unittest/TestDiagnostics.cc
  unittest/TestDivergence.cc
  unittest/TestDoubleMetadata.cc
  unittest/TestExceptions.cc
  unittest/TestFile.cc
  unittest/TestFloatMetadata.cc
  unittest/TestGradient.cc
  unittest/TestGridBbox.cc
  unittest/TestGrid.cc
  unittest/TestGridDescriptor.cc
  unittest/TestGridIO.cc
  unittest/TestGridTransformer.cc
  unittest/TestInit.cc
  unittest/TestInt32Metadata.cc
  unittest/TestInt64Metadata.cc
  unittest/TestInternalOrigin.cc
  unittest/TestLaplacian.cc
  unittest/TestLeafBool.cc
  unittest/TestLeaf.cc
  unittest/TestLeafIO.cc
  unittest/TestLeafMask.cc
  unittest/TestLeafOrigin.cc
  unittest/TestLevelSetRayIntersector.cc
  unittest/TestLevelSetUtil.cc
  unittest/TestLinearInterp.cc
  unittest/TestMaps.cc
  unittest/TestMat4Metadata.cc
  unittest/TestMath.cc
  unittest/TestMeanCurvature.cc
  unittest/TestMeshToVolume.cc
  unittest/TestMetadata.cc
  unittest/TestMetadataIO.cc
  unittest/TestMetaMap.cc
  unittest/TestName.cc
  unittest/TestNodeIterator.cc
  unittest/TestNodeMask.cc
  unittest/TestParticlesToLevelSet.cc
  unittest/TestPointIndexGrid.cc
  unittest/TestPointMaskGrid.cc
  unittest/TestPointPartitioner.cc
  unittest/TestPoissonSolver.cc
  unittest/TestPrePostAPI.cc
  unittest/TestQuadraticInterp.cc
  unittest/TestQuantizedUnitVec.cc
  unittest/TestQuat.cc
  unittest/TestRay.cc
  unittest/TestStats.cc
  unittest/TestStream.cc
  unittest/TestStringMetadata.cc
  unittest/TestTools.cc
  unittest/TestTopologyToLevelSet.cc
  unittest/TestTransform.cc
  unittest/TestTree.cc
  unittest/TestTreeCombine.cc
  unittest/TestTreeGetSetValues.cc
  unittest/TestTreeIterators.cc
  unittest/TestTreeVisitor.cc
  unittest/TestUtil.cc
  unittest/TestValueAccessor.cc
  unittest/TestVec2Metadata.cc
  unittest/TestVec3Metadata.cc
  unittest/TestVolumeRayIntersector.cc
  unittest/TestVolumeToMesh.cc
  )
TARGET_LINK_LIBRARIES ( openvdb_unittest
  openvdb
  cppunit
  )
install(TARGETS openvdb_unittest DESTINATION ${OPENVDB_BIN_DIR})

# ---
# pyopenvdb
ADD_LIBRARY ( pyopenvdb SHARED
  python/pyFloatGrid.cc
  python/pyIntGrid.cc
  python/pyMetadata.cc
  python/pyOpenVDBModule.cc
  python/pyTransform.cc
  python/pyVec3Grid.cc
  )
TARGET_LINK_LIBRARIES ( pyopenvdb
  openvdb
  ${Boost_LIBRARIES}
  )
set_target_properties(pyopenvdb PROPERTIES PREFIX "")
install(TARGETS pyopenvdb DESTINATION ${OPENVDB_PYTHON_LIB_DIR})

message(STATUS "CC flags: ${CMAKE_CC_FLAGS}")

