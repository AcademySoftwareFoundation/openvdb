# Copyright (c) 2012-2019 DreamWorks Animation LLC
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DreamWorks Animation nor the names of
# its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
# LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
#
#[=======================================================================[

  CMake Configuration for OpenVDB

  The OpenVDB CMake build system generates targets depending on the
  enabled components. It is designed for out of source CMake generation
  (a build location for CMake to write to will be required). For example,
  from the root of the repository:

    mkdir .build
    cd .build
    cmake ../

  Depending on the components you choose to build, a number of optional
  and required dependencies are expected. See the dependency documentation
  for more information:

    https://www.openvdb.org/documentation/doxygen/dependencies.html

  And the documentation on building OpenVDB for more in depth installation
  instructions:

    https://www.openvdb.org/documentation/doxygen/build.html

  This CMakeLists file provides most available options for configuring the
  build and installation of all OpenVDB components. By default the core
  library, binaries, python module and unit tests are enabled.

  Note that various packages have inbuilt CMake module support. See the
  CMake documentation for more ZLib, Doxygen, OpenGL, Boost and Python
  controls:

    https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html

  OpenVDB's CMake supports building the various components of against a
  prior installation of OpenVDB.

#]=======================================================================]

PROJECT ( OpenVDB )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
# Monitoring <PackageName>_ROOT variables
IF ( POLICY CMP0048 )
  CMAKE_POLICY ( SET CMP0048 NEW )
ENDIF ()

###### OpenVDB Build/Component Options

INCLUDE ( CMakeDependentOption )

# @todo SSE/AVX instruction options
# @todo static/shared library options
# @todo epydoc and pdflatex
OPTION ( OPENVDB_BUILD_CORE "Build the OpenVDB core" ON )
OPTION ( OPENVDB_BUILD_BINARIES "Build the vdb binaries" ON )
OPTION ( OPENVDB_BUILD_PYTHON_MODULE "Build the pyopenvdb Python module" ON )
OPTION ( OPENVDB_BUILD_UNITTESTS "Build the OpenVDB unit tests" ON )
OPTION ( OPENVDB_BUILD_DOCS "Build the OpenVDB documentation" OFF )
OPTION ( OPENVDB_BUILD_HOUDINI_PLUGIN "Build the Houdini plugin" OFF )
OPTION ( OPENVDB_BUILD_MAYA_PLUGIN "Build the Maya plugin" OFF )
OPTION ( OPENVDB_ENABLE_RPATH "Build with RPATH information" ON )
OPTION ( OPENVDB_CXX_STRICT "Enable or disable a set of pre-defined compiler warnings for clang and gcc" OFF )
CMAKE_DEPENDENT_OPTION ( OPENVDB_INSTALL_CMAKE_MODULES
  "Install the provided OpenVDB CMake modules when building the core library"
  ON "OPENVDB_BUILD_CORE" OFF )

SET ( OPENVDB_ABI_VERSION_NUMBER "" CACHE STRING [=[
Build for compatibility with version N of the OpenVDB Grid ABI, where N is 3, 4, 5 etc. (some newer features
will be disabled). If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the installed vdb_print binary to
determine the ABI number. You may set this to force a given ABI number.]=] )

###### Dependency options

# @todo scalable/concurrent malloc option(s)
# @todo non-required behavior for blosc, log4cplus (i.e. use if available)
OPTION ( USE_HOUDINI [=[
Build the library against a Houdini installation. Turns on automatically if OPENVDB_BUILD_HOUDINI_PLUGIN is enabled.
When enabled, you do not need to provide dependency locations for TBB, Blosc, IlmBase and OpenEXR. Boost must be
provided if Houdini Version >= 16.5. IlmBase/OpenEXR can optionally be provided if Houdini Version >= 17.5.]=] OFF )
OPTION ( USE_MAYA [=[
Build the library against a Maya installation. Turns on automatically if OPENVDB_BUILD_MAYA_PLUGIN is enabled.
When enabled, you do not need to provide dependency locations for TBB. All other dependencies must be provided.]=] OFF )
OPTION ( USE_BLOSC [=[
Use blosc while building openvdb components. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the located
openvdb build configuration to decide on blosc support. You may set this to on to force blosc to be used if you
know it to be required.]=] OFF )
OPTION ( USE_LOG4CPLUS [=[
Use log4cplus while building openvdb components. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the
located openvdb build configuration to decide on log4cplus support. You may set this to on to force log4cplus
to be used if you know it to be required.]=] OFF )
OPTION ( USE_EXR [=[
Use OpenEXR while building openvdb components. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the located
openvdb build configuration to decide on OpenEXR support. You may set this to on to force OpenEXR to be used if you
know it to be required.]=] OFF )
CMAKE_DEPENDENT_OPTION ( OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING
  "Disable the implicit linking of Boost libraries on Windows" ON "WIN32" OFF )
OPTION ( USE_SYSTEM_LIBRARY_PATHS "Build with system library paths" ON )

# Deprecated options

OPTION ( OPENVDB_BUILD_HOUDINI_SOPS "Build the Houdini plugin  (deprecated - see OPENVDB_BUILD_HOUDINI_PLUGIN)" OFF )
OPTION ( OPENVDB_ENABLE_3_ABI_COMPATIBLE "Build with OpenVDB ABI 3 (deprecated - see OPENVDB_ABI_VERSION_NUMBER)" OFF )

# Alias deprecated vars

IF ( OPENVDB_BUILD_HOUDINI_SOPS )
  # Support for legacy OPENVDB_BUILD_HOUDINI_SOPS variable
  MESSAGE ( DEPRECATION "The OPENVDB_BUILD_HOUDINI_SOPS option is deprecated and will be removed. "
    "Use OPENVDB_BUILD_HOUDINI_PLUGIN." )
  SET ( OPENVDB_BUILD_HOUDINI_PLUGIN ON )
ENDIF ()

IF ( OPENVDB_ENABLE_3_ABI_COMPATIBLE )
  MESSAGE ( DEPRECATION "OPENVDB_ENABLE_3_ABI_COMPATIBLE is deprecated and will be removed. Use "
    "-D OPENVDB_ABI_VERSION_NUMBER=N, where N is the abi version." )
  IF ( OPENVDB_ABI_VERSION_NUMBER )
    IF ( NOT ( ${OPENVDB_ABI_VERSION_NUMBER} EQUAL 3 ) )
      MESSAGE ( WARNING "OPENVDB_ABI_VERSION_NUMBER holds a different ABI value to "
        "OPENVDB_ENABLE_3_ABI_COMPATIBLE, which will be ignored." )
    ENDIF ()
  ELSE ()
    # Don't bother setting the docstring as we'll update it later
    SET ( OPENVDB_ABI_VERSION_NUMBER "3" CACHE STRING "" FORCE )
  ENDIF ()
ENDIF ()

# Various minimum version requirements

SET ( MINIMUM_BLOSC_VERSION 1.5 )
SET ( MINIMUM_BOOST_VERSION 1.53 )
SET ( MINIMUM_CPPUNIT_VERSION 1.10 )
SET ( MINIMUM_GLFW_VERSION 3.0 )
SET ( MINIMUM_HOUDINI_VERSION 16.5 )
SET ( MINIMUM_LOG4CPLUS_VERSION 1.1.2 )
SET ( MINIMUM_TBB_VERSION 3.0 )
# These promote warnings rather than errors
SET ( MINIMUM_MAYA_VERSION 2017 )
SET ( MINIMUM_OPENVDB_ABI_VERSION 3 )
SET ( MINIMUM_DOXYGEN_VERSION 1.8.8 )

#########################################################################

# General CMake and CXX settings

SET ( CMAKE_CXX_STANDARD 11 )
SET ( CMAKE_CXX_STANDARD_REQUIRED ON )
SET ( CMAKE_CXX_EXTENSIONS OFF )

SET ( CMAKE_DISABLE_SOURCE_CHANGES ON )
SET ( CMAKE_DISABLE_IN_SOURCE_BUILD ON )

IF ( OPENVDB_ENABLE_RPATH )
  # Configure rpath for installation base on the following:
  # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
  SET ( CMAKE_SKIP_BUILD_RPATH FALSE )
  SET ( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )
  SET ( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )
ENDIF ()

# For CMake's find Threads module which brings in pthread - This flag
# forces the compiler -pthread flag vs -lpthread
SET ( THREADS_PREFER_PTHREAD_FLAG TRUE )

ENABLE_TESTING ()

# Various root level CMake options which are marked as advanced
MARK_AS_ADVANCED (
  CCACHE_PATH
  OPENVDB_CXX_STRICT
  OPENVDB_ENABLE_3_ABI_COMPATIBLE
  OPENVDB_ENABLE_RPATH
  USE_HOUDINI
  USE_MAYA
  USE_LOG4CPLUS
  USE_SYSTEM_LIBRARY_PATHS
  OPENVDB_BUILD_HOUDINI_SOPS
  )

# ncrease the number of sections that an object file can contain
IF (MSVC)
  ADD_DEFINITIONS("/bigobj")
ENDIF()

# Add our cmake modules

LIST ( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

# Add backports to support older versions of CMake

IF ( ${CMAKE_VERSION} VERSION_LESS 3.8 )
  LIST ( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/backports" )
ENDIF ()

# Add cmake modules to installation command

IF ( OPENVDB_INSTALL_CMAKE_MODULES )
  SET ( OPENVDB_CMAKE_MODULES
    cmake/FindBlosc.cmake
    cmake/FindCppUnit.cmake
    #cmake/FindGLEW.cmake
    cmake/FindIlmBase.cmake
    cmake/FindLog4cplus.cmake
    cmake/FindOpenEXR.cmake
    cmake/FindOpenVDB.cmake
    cmake/FindTBB.cmake
    cmake/OpenVDBGLFW3Setup.cmake
    cmake/OpenVDBHoudiniSetup.cmake
    cmake/OpenVDBMayaSetup.cmake
  )
  INSTALL ( FILES ${OPENVDB_CMAKE_MODULES} DESTINATION lib/cmake/OpenVDB )
ENDIF ()

# Add the doxygen command if required - do this here so we guarantee not to error on
# unrelated build issues

IF ( OPENVDB_BUILD_DOCS )
  FIND_PACKAGE ( Doxygen REQUIRED )
  IF ( DOXYGEN_VERSION VERSION_LESS MINIMUM_DOXYGEN_VERSION )
    MESSAGE ( WARNING "The doxygen-config doxyfile has been generated with version "
      "\"${MINIMUM_DOXYGEN_VERSION}\". Found Doxygen version \"${DOXYGEN_VERSION}\". "
      "Documentation may contain errors."
      )
  ENDIF ()

  # @todo use cmake doxygen functions available from cmake 3.9
  FILE ( READ ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/doxygen-config DOXYGEN_CONFIG_CONTENT )
  FILE ( WRITE ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config ${DOXYGEN_CONFIG_CONTENT} )
  FILE ( APPEND ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config
    "QUIET=YES\nOUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/openvdb/doc\n"
    )

  ADD_CUSTOM_TARGET ( doc ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/openvdb
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )

  INSTALL ( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/openvdb/doc/html DESTINATION docs )
ENDIF ()

# Early exit if there's nothing to build

IF ( NOT (
    OPENVDB_BUILD_CORE OR
    OPENVDB_BUILD_BINARIES OR
    OPENVDB_BUILD_PYTHON_MODULE OR
    OPENVDB_BUILD_UNITTESTS OR
    OPENVDB_BUILD_HOUDINI_PLUGIN OR
    OPENVDB_BUILD_MAYA_PLUGIN )
  )
  RETURN ()
ENDIF ()

#########################################################################

# ccache setup

FIND_PROGRAM ( CCACHE_PATH ccache )
IF ( CCACHE_PATH )
  SET_PROPERTY ( GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache )
  SET_PROPERTY ( GLOBAL PROPERTY RULE_LAUNCH_LINK ccache )
  MESSAGE ( STATUS "Using ccache: ${CCACHE_PATH}" )
ENDIF ( CCACHE_PATH )

# Build type configuration - default to Release if none is set

IF ( NOT CMAKE_BUILD_TYPE )
  SET ( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE
    )
ENDIF ()
MESSAGE ( STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}" )

IF ( USE_SYSTEM_LIBRARY_PATHS )
  SET ( SYSTEM_LIBRARY_PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw  # Fink
    /opt/local  # DarwinPorts
    /opt/csw  # Blastwave
    /opt
    )
ENDIF ()

#########################################################################

# Compiler options

IF ( OPENVDB_CXX_STRICT )
  # Add definitions for a number of compiler warnings for GCC and Clang for all sub projects
  IF ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
    MESSAGE ( STATUS "Configuring Clang CXX warnings" )
    ADD_DEFINITIONS (
      -Wall
      -Wextra
      -Wconversion
      -Wno-sign-conversion
    )
  ELSEIF ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
    MESSAGE ( STATUS "Configuring GCC CXX warnings" )
    ADD_DEFINITIONS (
      -Wall
      -Wextra
      -pedantic
      -Wcast-align
      -Wcast-qual
      -Wconversion
      -Wdisabled-optimization
      -Woverloaded-virtual
    )
  ELSE ()
    MESSAGE ( WARNING "No available CXX warnings for compiler ${CMAKE_CXX_COMPILER_ID}" )
  ENDIF ()
ENDIF ()

##########################################################################

# Configure DCC installation if necessary

IF ( OPENVDB_BUILD_HOUDINI_PLUGIN )
  SET ( USE_HOUDINI ON )
ENDIF ()

IF ( OPENVDB_BUILD_MAYA_PLUGIN )
  SET ( USE_MAYA ON )
ENDIF ()

IF ( USE_MAYA AND USE_HOUDINI )
  # @todo technically this is possible so long as library versions match
  # exactly but it's difficult to validate and dangerous
  MESSAGE ( FATAL_ERROR "Cannot build both Houdini and Maya plugins against "
    "the same core dependencies. Plugins must be compiled separately to "
    "ensure the required DCC dependencies are met."
    )
ENDIF ()

# Configure component dependencies by loading the Houdini/Maya setup
# scripts. These also find the Houdini/Maya installations

IF ( USE_HOUDINI )
  INCLUDE ( OpenVDBHoudiniSetup )
ENDIF ()

IF ( USE_MAYA )
  INCLUDE ( OpenVDBMayaSetup )
ENDIF ()

#########################################################################

# Configure OpenVDB Library and ABI versions

IF ( NOT OPENVDB_BUILD_CORE )
  # Find VDB installation and determine lib/abi versions
  FIND_PACKAGE ( OpenVDB REQUIRED )
  # Check ABI version was found and explicitly error if attempting to build against
  # an incompatible Houdini version
  IF ( OpenVDB_ABI AND OPENVDB_HOUDINI_ABI )
    IF ( NOT ${OpenVDB_ABI} EQUAL ${OPENVDB_HOUDINI_ABI} )
      MESSAGE ( FATAL_ERROR "Located OpenVDB installation is not ABI compatible with "
        "Houdini Version ${Houdini_VERSION}. Requires ABI ${OPENVDB_HOUDINI_ABI}, found "
        "ABI ${OpenVDB_ABI}." )
    ENDIF ()
  ENDIF ()
ELSE ()
  # Only use find package to determine the VDB version in this directory
  SET ( OPENVDB_ROOT ${CMAKE_CURRENT_SOURCE_DIR} )
  FIND_PACKAGE ( OpenVDB QUIET )
  MESSAGE ( STATUS "Configuring for OpenVDB Version ${OpenVDB_VERSION}")
ENDIF ()

# Validate the OpenVDB ABI Version. If OpenVDB_ABI is not set, we're either building
# the core library OR the ABI hasn't been deduced from a VDB installation. Use the
# value from OPENVDB_ABI_VERSION_NUMBER, falling back to the lib major version number

IF ( NOT OpenVDB_ABI )
  IF ( OPENVDB_ABI_VERSION_NUMBER )
    SET ( OpenVDB_ABI ${OPENVDB_ABI_VERSION_NUMBER} )
  ELSE ()
    SET ( OpenVDB_ABI ${OpenVDB_MAJOR_VERSION} )
  ENDIF ()
ENDIF ()

# From the deduced ABI, check against the required ABI for Houdini (if set).
# Forcefully set the ABI to the required value if necessary - do this after to
# explicitly warn the user if their chosen value is different.

IF ( OPENVDB_HOUDINI_ABI AND ( NOT "${OpenVDB_ABI}" EQUAL "${OPENVDB_HOUDINI_ABI}" ) )
  MESSAGE ( WARNING "CMake will explicitly set the value of OPENVDB_ABI_VERSION_NUMBER to "
    "${OPENVDB_HOUDINI_ABI} to match the ABI of the target Houdini Version." )
  SET ( OpenVDB_ABI ${OPENVDB_HOUDINI_ABI} )
ENDIF ()

IF ( ${OpenVDB_ABI} LESS ${MINIMUM_OPENVDB_ABI_VERSION} )
  MESSAGE ( WARNING "OpenVDB ABI versions earlier than ${MINIMUM_OPENVDB_ABI_VERSION} are "
    "deprecated and will soon be removed." )
ENDIF ()

# The ABI is a global target definition, applicable to all components, so just add it
# via ADD_DEFINITIONS

IF ( OpenVDB_ABI EQUAL 3 )
  ADD_DEFINITIONS ( -DOPENVDB_3_ABI_COMPATIBLE )
ENDIF ()

ADD_DEFINITIONS ( -DOPENVDB_ABI_VERSION_NUMBER=${OpenVDB_ABI} )
MESSAGE ( STATUS "Configuring for OpenVDB ABI Version ${OpenVDB_ABI}")

# Always force set as we may need to change it if it's incompatible with Houdini
SET ( OPENVDB_ABI_VERSION_NUMBER ${OpenVDB_ABI} CACHE STRING [=[
Build for compatibility with version N of the OpenVDB Grid ABI, where N is 3, 4, 5 etc. (some newer features
will be disabled). If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the installed vdb_print binary to
determine the ABI number. You may set this to force a given ABI number.]=] FORCE )

##########################################################################

IF ( OPENVDB_BUILD_CORE )
  ADD_SUBDIRECTORY ( openvdb )
ENDIF ()

IF ( OPENVDB_BUILD_PYTHON_MODULE )
  ADD_SUBDIRECTORY ( openvdb/python )
ENDIF ()

IF ( OPENVDB_BUILD_BINARIES )
  ADD_SUBDIRECTORY ( openvdb/cmd )
ENDIF ()

IF ( OPENVDB_BUILD_UNITTESTS )
  ADD_SUBDIRECTORY ( openvdb/unittest )
ENDIF ()

IF ( OPENVDB_BUILD_HOUDINI_PLUGIN )
  ADD_SUBDIRECTORY ( openvdb_houdini )
ENDIF ()

IF ( OPENVDB_BUILD_MAYA_PLUGIN )
  ADD_SUBDIRECTORY ( openvdb_maya )
ENDIF ()
