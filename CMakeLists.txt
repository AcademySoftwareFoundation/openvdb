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

  This file provides most available options for configuring the build and
  installation of all OpenVDB components. By default the core library,
  python module and unit tests are built.

  The OpenVDB CMake build system generates targets depending on the
  enabled components. It is designed for out of source CMake generation
  (a build location for CMake to write to will be required). Depending on
  the components you choose to build, a number of optional and required
  dependencies are expected. Below is a full list of all dependencies,
  followed up by each OpenVDB components requirements.

  ########################################################################

  Dependencies:

    -  IlmBase
      - Used for the definition of half (half precision floating points)
        and for additional EXR serialization support. Usually available by
        default in most UNIX distributions.
      - Installation
        - UNIX apt-get:
          - apt-get install libilmbase-dev
        - Homebrew:
          - brew install ilmbase
        - From Source:
          - http://www.openexr.com
      - Usage
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export ILMBASE_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D ILMBASE_ROOT=<Installation Root Directory>
        - Provide Locations to CMake:
            -D ILMBASE_INCLUDEDIR=<Include path>
            -D ILMBASE_LIBRARYDIR=<Lib path>

    - OpenEXR
      - Used for OpenEXR image serialization support. Usually available by
        default in most UNIX distributions.
      - Installation
        - UNIX apt-get:
          - apt-get install libopenexr-dev
        - Homebrew:
          - brew install openexr
        - From Source:
          - http://www.openexr.com
      - Usage
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export OPENEXR_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D OPENEXR_ROOT=<Installation Root Directory>
        - Provide Locations to CMake:
            -D OPENEXR_INCLUDEDIR=<Include path>
            -D OPENEXR_LIBRARYDIR=<Lib path>

    - Threading Building Blocks (TBB)
      - A portable C++ template library for task parallelism.
      - Installation
        - UNIX apt-get:
          - apt-get install libtbb-dev
        - Homebrew:
          - brew install tbb
        - From Source:
          -  https://www.threadingbuildingblocks.org/
      - Usage
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export TBB_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D TBB_ROOT=<Installation Root Directory>
        - Provide Locations to CMake:
            -D TBB_INCLUDEDIR=<Include path>
            -D TBB_LIBRARYDIR=<Lib path>

    - ZLib
      - Compression library for disk serialization compression. Usually
        available by default in most UNIX distributions.
      - Installation
        - UNIX apt-get:
          - apt-get install zlibc
        - Homebrew:
          - brew install zlib
        - From Source:
          - https://www.zlib.net/
      - Usage
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export ZLIB_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D ZLIB_ROOT=<Installation Root Directory>

    - Boost Components: => 1.55
      - Various boost components are required for some OpenVDB components
        - Boost.System
        - Boost.IOStreams
        - Boost.Python
        - Boost.Thread
      - Installation
        - UNIX apt-get:
          - apt-get install libboost-system-dev
          - apt-get install libboost-iostreams-dev
          - apt-get install libboost-python-dev
          - apt-get install libboost-thread-dev
        - Homebrew:
          - brew install boost
          - brew install boost-python
        - From Source:
          - https://www.boost.org/
      - Usage
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export BOOST_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D BOOST_ROOT=<Installation Root Directory>
        - Provide Locations to CMake:
            -D BOOST_INCLUDEDIR=<Include path>
            -D BOOST_LIBRARYDIR=<Lib path>

    - Python
      - The python interpreter and libraries for C++/python bindings.
        Usually available by default in most UNIX distributions.
      - Installation
        - UNIX apt-get:
          - apt-get install python-dev
          - apt-get install python-numpy
          - apt-get install python-epydoc
        - Homebrew:
          - brew install python

    - CppUnit
      - A unit testing framework module for C++.
      - Installation
        - Homebrew:
          - brew install cppunit
        - From Source:
          - https://anongit.freedesktop.org/git/libreoffice/cppunit.git
      - Usage
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export CPPUNIT_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D CPPUNIT_ROOT=<Installation Root Directory>
        - Provide Locations to CMake:
            -D CPPUNIT_INCLUDEDIR=<Include path>
            -D CPPUNIT_LIBRARYDIR=<Lib path>

    - Blosc: => 1.5
      - An optional but recommended dependency for improved disk
        compression.
      - Installation
        - Homebrew:
          - brew install blosc
        - From Source:
          - https://github.com/Blosc/c-blosc/releases.
      - Usage
        Make sure that you set USE_BLOSC=ON
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export BLOSC_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D BLOSC_ROOT=<Installation Root Directory>
        - Provide Locations to CMake:
            -D BLOSC_INCLUDEDIR=<Include path>
            -D BLOSC_LIBRARYDIR=<Lib path>

    - Log4cplus
      - An optional dependency for improved OpenVDB Logging
      - Installation
        - UNIX apt-get:
          - apt-get install liblog4cplus-dev
        - From Source:
          - https://github.com/log4cplus/log4cplus
      - Usage
        Make sure that you set USE_LOG4CPLUS=ON
        If CMake cannot find your installation location automatically:
        - Provide Root from environment:
            export LOG4CPLUS_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D LOG4CPLUS_ROOT=<Installation Root Directory>
        - Provide Locations to CMake:
            -D LOG4CPLUS_INCLUDEDIR=<Include path>
            -D LOG4CPLUS_LIBRARYDIR=<Lib path>

    - GLFW: => 3
      - Simple API for OpenGL development
      - Installation
        - UNIX apt-get:
          - apt-get install libglfw3-dev
        - Homebrew:
          - brew install glfw
        - From Source:
          - https://www.glfw.org/
      - Usage
        - Provide Root from environment:
            export GLFW3_ROOT=<Installation Root Directory>
        - Provide Root to CMake:
            -D GLFW3_ROOT=<Installation Root Directory>
        - OpenVDB uses GLFW's CMake installation to find and configure
          the GLFW dependency. You should add GLFW's cmake location to
          CMAKE_PREFIX_PATH where possible, instead of the above:
            -D CMAKE_PREFIX_PATH=<GLFW CMake location>

    - Doxygen:
      - Documentation generation from C++
      - Installation
        - UNIX apt-get:
          - apt-get install doxygen
        - Homebrew:
          - brew install doxygen
        - From Source:
          - http://www.doxygen.nl/

  Note that various packages have inbuilt CMake module support. See the
  CMake documentation for more ZLib, Doxygen, OpenGL, Boost and Python
  controls:
  https://cmake.org/cmake/help/v3.14/manual/cmake-modules.7.html

  ########################################################################

  OpenVDB Components and their dependencies:

    OpenVDB Core
      Main CMake: openvdb/CMakeLists.txt
      Required: IlmBase::Half, TBB::tbb, ZLib, Boost::System,
        Boost::IOStream
      Optional: Blosc, Log4cplus, OpenEXR

    OpenVDB Binaries
      Main CMake: openvdb/cmd/CMakeLists.txt
      Required: (OpenVDB Core + Dependencies), GLFW3 (vdb_view),
        OpenGL (vdb_view), Boost::Thread (vdb_view), OpenEXR (vdb_render)
        IlmBase (vdb_render)

    OpenVDB Python
      Main CMake: openvdb/python/CMakeLists.txt
      Required: (OpenVDB Core + Dependencies), Python, Boost::Python

    OpenVDB Unit Tests
      Main CMake: openvdb/unittest/CMakeLists.txt
      Required: (OpenVDB Core + Dependencies), CppUnit

  ########################################################################

  The CMake supports building the various components of OpenVDB against
  a prior installation of OpenVDB.

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
OPTION ( OPENVDB_BUILD_CORE "Build the OpenVDB core" ON )
OPTION ( OPENVDB_BUILD_BINARIES "Build the vdb binaries" ON )
OPTION ( OPENVDB_BUILD_PYTHON_MODULE "Build the pyopenvdb Python module" ON )
OPTION ( OPENVDB_BUILD_UNITTESTS "Build the OpenVDB unit tests" ON )
OPTION ( OPENVDB_BUILD_DOCS "Build the OpenVDB documentation" OFF )
OPTION ( OPENVDB_BUILD_HOUDINI_SOPS "Build the Houdini SOPs" OFF )
OPTION ( OPENVDB_BUILD_MAYA_PLUGIN "Build the Maya plugin" OFF )
OPTION ( OPENVDB_ENABLE_RPATH "Build with RPATH information" ON )
OPTION ( OPENVDB_CXX_STRICT "Enable or disable a set of pre-defined compiler warnings for clang and gcc" OFF )
OPTION ( OPENVDB_ENABLE_3_ABI_COMPATIBLE "Build with OpenVDB ABI 3 (deprecated - see OPENVDB_ABI_VERSION_NUMBER)" OFF )

###### Dependency options

# @todo scalable/concurrent malloc option(s)
# @todo non-required behavior for blosc, log4cplus (i.e. use if available)
OPTION ( USE_HOUDINI [=[
Build the library against a Houdini installation. Turns on autmatically if OPENVDB_BUILD_HOUDINI_SOPS is enabled.
When enabled, you do not need to provide dependency locations for OpenEXR, IlmBase, TBB, Boost and Blosc. You can
optionally provide your own version of Boost when building against Houdini 16.5 and gerater.]=] OFF )
OPTION ( USE_BLOSC "Build the library with blosc support." ON )
OPTION ( USE_LOG4CPLUS "Build the library with log4cplus support." OFF )
OPTION ( OPENEXR_NAMESPACE_VERSIONING "Set the expected names of OpenEXR libraries to be version suffixed" OFF )
OPTION ( ILMBASE_NAMESPACE_VERSIONING "Set the expected names of ILM Base libraries to be version suffixed" OFF )
CMAKE_DEPENDENT_OPTION ( OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING
  "Disable the implicit linking of Boost libraries on Windows" ON "WIN32" OFF )
OPTION ( USE_SYSTEM_LIBRARY_PATHS "Build with system library paths" ON )
CMAKE_DEPENDENT_OPTION ( BOOST_PYTHON_NAMESPACE_VERSIONING
  "Set the expected names of Boost.Python libraries to be version suffixed"
  ON "OPENVDB_BUILD_PYTHON_MODULE" ON )

SET ( OPENVDB_ABI_VERSION_NUMBER "" CACHE STRING "Build for compatibility with version N of the
  OpenVDB Grid ABI, where N is 3, 4, 5 etc. (some newer features will be disabled)." )

# Various minimum version requirements - Cache options can be changed

SET ( MINIMUM_BOOST_VERSION 1.55 )
SET ( MINIMUM_BLOSC_VERSION 1.5 )
SET ( MINIMUM_GLFW_VERSION 3 )
SET ( MINIMUM_OPENVDB_ABI_VERSION 3 )

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
  ILMBASE_NAMESPACE_VERSIONING
  OPENEXR_NAMESPACE_VERSIONING
  OPENVDB_CXX_STRICT
  OPENVDB_ENABLE_3_ABI_COMPATIBLE
  OPENVDB_ENABLE_RPATH
  USE_HOUDINI
  USE_LOG4CPLUS
  USE_SYSTEM_LIBRARY_PATHS
  )

# Add our cmake modules

LIST ( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

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

# Configure Houdini installation if necessary

IF ( OPENVDB_BUILD_HOUDINI_SOPS )
  SET ( USE_HOUDINI ON )
ENDIF ()

IF ( APPLE )
  IF ( USE_HOUDINI AND OPENVDB_BUILD_MAYA_PLUGIN )
    MESSAGE ( FATAL_ERROR "On macOS, Houdini is linked with libc++ whereas Maya is linked with libstdc++. "
      "The different runtimes are not mutually compatible, so build the Houdini and Maya plugins separately "
      "with the requisite DCC's compiler." )
  ENDIF ()
ENDIF ()

# If we're building against a Houdini Install, load OpenVDBHoudiniSetup which
# configures all dependency locations and Houdini libraries for OpenVDB

IF ( USE_HOUDINI )
  INCLUDE ( OpenVDBHoudiniSetup )
ENDIF ()

#########################################################################

# Determine OpenVDB version by calling FIND_PACKAGE with OpenVDB - If we're
# building the core library, the find package call is set to quiet to
# ignore the missing libs

IF ( NOT OPENVDB_BUILD_CORE )
  FIND_PACKAGE ( OpenVDB REQUIRED )
ELSE ()
  SET ( OPENVDB_ROOT ${CMAKE_CURRENT_SOURCE_DIR} )
  FIND_PACKAGE ( OpenVDB QUIET )
  MESSAGE ( STATUS "Configuring for OpenVDB Version ${OpenVDB_VERSION}")
ENDIF ()

#########################################################################

# Determine the OpenVDB ABI Version. If OPENVDB_ABI_VERSION_NUMBER is defined, use that value,
# otherwise check the legacy OPENVDB_ENABLE_3_ABI_COMPATIBLE option, falling back to the major
# library version number

SET ( ABI "" )
IF ( OPENVDB_HOUDINI_ABI )
  IF ( OPENVDB_ABI_VERSION_NUMBER AND ( NOT "${OPENVDB_ABI_VERSION_NUMBER}" EQUAL "${OPENVDB_HOUDINI_ABI}" ) )
    MESSAGE ( WARNING "CMake will explicitly set the value of OPENVDB_ABI_VERSION_NUMBER to "
      "${OPENVDB_HOUDINI_ABI} to match the ABI of the target Houdini Version." )
  ENDIF ()
  SET ( ABI ${OPENVDB_HOUDINI_ABI} )
ELSEIF ( NOT OPENVDB_ABI_VERSION_NUMBER )
  IF ( OPENVDB_ENABLE_3_ABI_COMPATIBLE )
    SET ( ABI 3 )
    MESSAGE ( DEPRECATION "OPENVDB_ENABLE_3_ABI_COMPATIBLE is deprecated. Instead use "
      "-D OPENVDB_ABI_VERSION_NUMBER=N, where N is the abi version." )
  ELSE ()
    SET ( ABI ${OpenVDB_MAJOR_VERSION} )
  ENDIF ()
ELSE ()
  SET ( ABI ${OPENVDB_ABI_VERSION_NUMBER} )
  IF ( NOT OPENVDB_BUILD_CORE )
    # @todo  determine if the VDB version we're building against has the same ABI and error if not
  ENDIF ()
ENDIF ()

# Always force set as we may need to change it if it's incompatible with Houdini
SET ( OPENVDB_ABI_VERSION_NUMBER ${ABI} CACHE STRING "Build for compatibility with version N of the
  OpenVDB Grid ABI, where N is 3, 4, 5 etc. (some newer features will be disabled)."
  FORCE )

IF ( ${OPENVDB_ABI_VERSION_NUMBER} LESS ${MINIMUM_OPENVDB_ABI_VERSION} )
  MESSAGE ( WARNING "OpenVDB ABI versions earlier than ${MINIMUM_OPENVDB_ABI_VERSION} are "
      "deprecated and will soon be removed." )
ENDIF ()

IF ( OPENVDB_ABI_VERSION_NUMBER EQUAL 3 )
  ADD_DEFINITIONS ( -DOPENVDB_3_ABI_COMPATIBLE )
ENDIF ()

ADD_DEFINITIONS ( -DOPENVDB_ABI_VERSION_NUMBER=${OPENVDB_ABI_VERSION_NUMBER} )
MESSAGE ( STATUS "Configuring for OpenVDB ABI Version ${OPENVDB_ABI_VERSION_NUMBER}")

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

IF ( OPENVDB_BUILD_HOUDINI_SOPS )
  ADD_SUBDIRECTORY ( openvdb_houdini )
ENDIF ()

IF ( OPENVDB_BUILD_MAYA_PLUGIN )
  ADD_SUBDIRECTORY ( openvdb_maya )
ENDIF ()

IF ( OPENVDB_BUILD_DOCS )
  FIND_PACKAGE ( Doxygen REQUIRED )

  FILE ( READ ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/doxygen-config DOXYGEN_CONFIG_CONTENT )
  FILE ( WRITE ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config ${DOXYGEN_CONFIG_CONTENT} )
  FILE ( APPEND ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config "OUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/openvdb/doc\n" )

  ADD_CUSTOM_TARGET ( doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )

  INSTALL ( CODE "EXECUTE_PROCESS(COMMAND \"${CMAKE_COMMAND}\" --build  \"${PROJECT_BINARY_DIR}\" --target doc)")
  INSTALL ( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/openvdb/doc/html DESTINATION docs)
ENDIF ()
