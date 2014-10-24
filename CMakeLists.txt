PROJECT(OPENVDB)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

ADD_DEFINITIONS(-DUNICODE)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
SET(CMAKE_DEBUG_POSTFIX "d")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

if(WIN32)
	#enable M_PI constant in math.h and avoid redefinition of min max as macros and allow compile as a DLL
	ADD_DEFINITIONS(-D_USE_MATH_DEFINES -DNOMINMAX -DOPENVDB_DLL)

	# EIGEN SPEED UP, remove me for a real debug!!
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /DNDEBUG")
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /DNDEBUG")
endif(WIN32)

############################## Setup project version and set the link to version.h and version.rc
# Included project can access these information using OPENVDB_VERSION, VERSION_H_IN and VERSION_RC_IN
# Remember to call CONFIGURE_FILE( VERSION_RC_IN ${PROJECT_BINARY_DIR}/version.rc) to generate the resource file
INCLUDE(openvdbVersion)

############################## Setup output directories
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single directory for all libraries")
SET(EXECUTABLE_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single directory for all executables")
SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/installation)
MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)


####################### projects general dependencies
FIND_PACKAGE(Boost REQUIRED) 
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES( ${Boost_LIBRARY_DIRS} )
message(STATUS "BOOST_ROOT : $ENV{BOOST_ROOT}")

FIND_PACKAGE(ZLIB REQUIRED)
ADD_DEFINITIONS(-DZLIB_WINAPI)
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_PATH})

FIND_PACKAGE(OPENEXRHALF REQUIRED)
INCLUDE_DIRECTORIES(${OPENEXRHALF_INCLUDE_PATH})

FIND_PACKAGE(TBB REQUIRED)
INCLUDE_DIRECTORIES(${TBB_INCLUDE_PATH})


#SET(OPENVDB_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../include CACHE PATH "OpenVDB include folder")
#INCLUDE_DIRECTORIES(${OPENVDB_INCLUDE_PATH})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

##############################  Setup testing projects

SET(OPENVDB_TEST_DATA_PATH ${PROJECT_BINARY_DIR}/../test_data/ CACHE PATH "A folder containing the tests data files")
ENABLE_TESTING()

#######################################################################################################
#######################################################################################################
#######################################################################################################
############  Project specification, add files directly here...  ######################################
# io, math, metadata, tools, tree, util

OPTION(OPENVDB_ENABLE_HOUDINI "Build hudini plugin" OFF)
OPTION(OPENVDB_ENABLE_MAYA "Build maya plugin" OFF)


add_subdirectory(openvdb)

IF(${OPENVDB_ENABLE_HOUDINI})
	add_subdirectory(openvdb_houdini)
ENDIF(${OPENVDB_ENABLE_HOUDINI})

IF(${OPENVDB_ENABLE_MAYA})
	add_subdirectory(openvdb_maya)
ENDIF(${OPENVDB_ENABLE_MAYA})


add_subdirectory(tests)
add_subdirectory(samples)
