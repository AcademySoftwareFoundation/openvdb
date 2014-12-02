# yue.nicholas@gmail.com
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT ( OpenVDB )

set(DESTDIR "/usr/local")
set(OPENVDB_INCLUDE_DIR "${DESTDIR}/include/openvdb")
set(OPENVDB_LIB_DIR "${DESTDIR}/lib/openvdb")
set(OPENVDB_BIN_DIR "${DESTDIR}/bin")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/")

find_package(PkgConfig REQUIRED)

pkg_search_module(GLFW REQUIRED glfw3)

FIND_PACKAGE ( Boost REQUIRED COMPONENTS iostreams system thread )

FIND_PACKAGE ( IlmBase REQUIRED )
FIND_PACKAGE ( OpenEXR REQUIRED )

FIND_PACKAGE ( TBB REQUIRED )
FIND_PACKAGE ( ZLIB REQUIRED )
FIND_PACKAGE ( OpenGL REQUIRED )

INCLUDE_DIRECTORIES ( ${Boost_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${ILMBASE_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${TBB_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${OPENGL_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${ZLIB_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( ${GLFW_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( .. ) # So they can simulate referencing installed headers
INCLUDE_DIRECTORIES ( . ) # So they can simulate referencing installed headers

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
  math/Hermite.cc
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
ADD_LIBRARY ( openvdb STATIC
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
