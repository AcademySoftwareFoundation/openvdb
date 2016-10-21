# yue.nicholas@gmail.com
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT ( OpenVDB )

OPTION ( OPENVDB_BUILD_PYTHON_MODULE "Build the pyopenvdb Python module" ON )
OPTION ( OPENVDB_BUILD_HOUDINI_SOPS "Build the Houdini SOPs" OFF )
OPTION ( OPENVDB_BUILD_MAYA_PLUGIN "Build the Maya plugin" OFF )
OPTION ( OPENVDB_ENABLE_RPATH "Build with RPATH information" ON )

IF ( APPLE )
  IF ( OPENVDB_BUILD_HOUDINI_SOPS AND OPENVDB_BUILD_MAYA_PLUGIN )
	MESSAGE ( FATAL_ERROR "On OSX, Houdini linked with libc++ whereas Maya is linked with libstdc++, the different runtimes are not mutually compatible, build them Houdini SOPs and Maya plugins separately with the requisite DCC's compiler" )
  ENDIF ()
ENDIF ()

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

SET ( OPENVDB_TOP_LEVEL_DIR ${PROJECT_SOURCE_DIR} CACHE PATH "OpenVDB Top Level Directory to share among sub projects" )

INCLUDE_DIRECTORIES ( ${OPENVDB_TOP_LEVEL_DIR} ) # To make sure we pick up headers from this version of OpenVDB we are building

ADD_SUBDIRECTORY ( openvdb )

IF ( OPENVDB_BUILD_HOUDINI_SOPS )
  ADD_SUBDIRECTORY ( openvdb_houdini )
ENDIF ()

IF ( OPENVDB_BUILD_MAYA_PLUGIN )
  ADD_SUBDIRECTORY ( openvdb_maya )
ENDIF ()
