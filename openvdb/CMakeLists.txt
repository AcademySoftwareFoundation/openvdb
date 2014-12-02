# yue.nicholas@gmail.com
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT ( OpenVDB )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

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

SET ( OPENVDB_LIBRARY_SOURCE_FILES
  ${OPENVDB_PATH}/Grid.cc
  ${OPENVDB_PATH}/io/Archive.cc
  ${OPENVDB_PATH}/io/Compression.cc
  ${OPENVDB_PATH}/io/File.cc
  ${OPENVDB_PATH}/io/GridDescriptor.cc
  ${OPENVDB_PATH}/io/Queue.cc
  ${OPENVDB_PATH}/io/Stream.cc
  ${OPENVDB_PATH}/io/TempFile.cc
  ${OPENVDB_PATH}/math/Hermite.cc
  ${OPENVDB_PATH}/math/Maps.cc
  ${OPENVDB_PATH}/math/Proximity.cc
  ${OPENVDB_PATH}/math/QuantizedUnitVec.cc
  ${OPENVDB_PATH}/math/Transform.cc
  ${OPENVDB_PATH}/metadata/Metadata.cc
  ${OPENVDB_PATH}/metadata/MetaMap.cc
  ${OPENVDB_PATH}/openvdb.cc
  ${OPENVDB_PATH}/Platform.cc
  ${OPENVDB_PATH}/util/Formats.cc
  ${OPENVDB_PATH}/util/Util.cc
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

ADD_EXECUTABLE ( vdb_print
  ${OPENVDB_PATH}/cmd/openvdb_print/main.cc
  )

TARGET_LINK_LIBRARIES ( vdb_print
  openvdb
  )

ADD_EXECUTABLE ( vdb_render
  ${OPENVDB_PATH}/cmd/openvdb_render/main.cc
  )

TARGET_LINK_LIBRARIES ( vdb_render
  openvdb
  ${ILMBASE_LIBRARIES}
  ${OPENEXR_LIBRARIES}
  )

ADD_EXECUTABLE ( vdb_view
  ${OPENVDB_PATH}/cmd/openvdb_view/main.cc
  ${OPENVDB_PATH}/viewer/Camera.cc
  ${OPENVDB_PATH}/viewer/ClipBox.cc
  ${OPENVDB_PATH}/viewer/Font.cc
  ${OPENVDB_PATH}/viewer/RenderModules.cc
  ${OPENVDB_PATH}/viewer/Viewer.cc
  )

#MESSAGE ( "OPENGL_LIBRARIES = ${OPENGL_LIBRARIES}")
#MESSAGE ( "GLFW_LIBRARIES = ${GLFW_STATIC_LIBRARIES}")
TARGET_LINK_LIBRARIES ( vdb_view
  openvdb
  ${OPENGL_LIBRARIES}
  ${GLFW_STATIC_LIBRARIES}
  )
