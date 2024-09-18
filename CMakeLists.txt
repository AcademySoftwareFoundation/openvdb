# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB

  The OpenVDB CMake build system generates targets depending on the
  enabled components. It is designed for out of source CMake generation
  (a build location for CMake to write to will be required). For example,
  from the root of the repository:

    mkdir build
    cd build
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
  library and the vdb_print binary are enabled.

  Note that various packages have inbuilt CMake module support. See the
  CMake documentation for more ZLib, Doxygen, OpenGL, Boost and Python
  controls:

    https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html

  OpenVDB's CMake supports building the various components of against a
  prior installation of OpenVDB.

#]=======================================================================]

# note: cmake_minimum_required must be called before project commands to
#  ensure policy scope is set up correctly
cmake_minimum_required(VERSION 3.18)

# CMP0091 allows for MSVC ABI targetting via CMAKE_MSVC_RUNTIME_LIBRARY
# from CMake 3.15 and above. Must come before project().
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

###### Version

set(OpenVDB_MAJOR_VERSION 11)
set(OpenVDB_MINOR_VERSION 0)
set(OpenVDB_PATCH_VERSION 1)
set(OpenVDB_VERSION "${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}")

project(OpenVDB LANGUAGES CXX VERSION ${OpenVDB_VERSION})

###### OpenVDB Build/Component Options

include(CMakeDependentOption)
include(GNUInstallDirs)

# Components
option(OPENVDB_BUILD_CORE "Enable the core OpenVDB library. Both static and shared versions are enabled by default" ON)
option(OPENVDB_BUILD_BINARIES "Enable the vdb binaries. Only vdb_print is enabled by default" ON)
option(OPENVDB_BUILD_PYTHON_MODULE "Build the pyopenvdb Python module" OFF)
option(OPENVDB_BUILD_UNITTESTS "Build the OpenVDB unit tests" OFF)
option(OPENVDB_BUILD_DOCS "Build the OpenVDB documentation" OFF)
option(OPENVDB_BUILD_HOUDINI_PLUGIN "Build the Houdini plugin" OFF)
option(OPENVDB_BUILD_HOUDINI_ABITESTS "Build the Houdini ABI tests" OFF)
option(OPENVDB_BUILD_MAYA_PLUGIN "Build the Maya plugin" OFF)
option(OPENVDB_BUILD_AX "Build the OpenVDB AX library. Turns ON if USE_AX is ON." ${USE_AX})
option(OPENVDB_BUILD_AX_UNITTESTS "Build the OpenVDB AX unit tests" OFF)
option(OPENVDB_BUILD_NANOVDB "Build the NanoVDB library. Turns ON if USE_NANOVDB is ON." ${USE_NANOVDB})

# Global options
option(OPENVDB_ENABLE_RPATH "Build with RPATH information" ON)
option(OPENVDB_ENABLE_ASSERTS "Build with asserts in OpenVDB code enabled" OFF)
option(OPENVDB_USE_DELAYED_LOADING "Build the core OpenVDB library with delayed-loading." ON)
option(OPENVDB_CXX_STRICT "Enable or disable pre-defined compiler warnings" OFF)
cmake_dependent_option(OPENVDB_INSTALL_CMAKE_MODULES
  "Install the provided OpenVDB CMake modules when building the core library"
  ON "OPENVDB_BUILD_CORE" OFF)
option(USE_CCACHE "Build using Ccache if found on the path" ON)
# Disable this on Windows due to linker OOM issues:
#  LNK1248: image size (XXX) exceeds maximum allowable size (FFFFFFFF)
#    https://github.com/AcademySoftwareFoundation/openvdb/issues/1718
#    https://github.com/AcademySoftwareFoundation/openvdb/issues/1624
cmake_dependent_option(USE_EXPLICIT_INSTANTIATION "Use explicit instantiation for all supported classes
and methods against a pre-defined list of OpenVDB trees. This makes the core library larger and slower
to compile, but speeds up the compilation of all dependent code by bypassing the expensive template
instantation." ON "NOT WIN32" OFF)
option(OPENVDB_FUTURE_DEPRECATION "Generate messages for upcoming deprecation" ON)
option(OPENVDB_ENABLE_UNINSTALL "Adds a CMake uninstall target." ON)
option(USE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF)

# Component dependency options
cmake_dependent_option(USE_PKGCONFIG "Use pkg-config to search for dependent libraries." ON "NOT WIN32" OFF)
option(USE_HOUDINI [=[
Build the library against a Houdini installation. Turns on automatically if OPENVDB_BUILD_HOUDINI_PLUGIN is enabled.
When enabled, you do not need to provide dependency locations for TBB, Blosc, Imath and OpenEXR. Boost must be
provided. Imath/OpenEXR can optionally be provided.]=] OFF)
option(USE_MAYA [=[
Build the library against a Maya installation. Turns on automatically if OPENVDB_BUILD_MAYA_PLUGIN is enabled.
When enabled, you do not need to provide dependency locations for TBB. All other dependencies must be provided.]=] OFF)
option(USE_TBB [=[
Use TBB in building the NanoVDB library.]=] ON)
option(USE_BLOSC [=[
Use blosc while building openvdb components. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the located
openvdb build configuration to decide on blosc support. You may set this to on to force blosc to be used if you
know it to be required.]=] ON)
option(USE_ZLIB [=[
Use zlib while building openvdb components. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the located
openvdb build configuration to decide on zlib support. ZLib can only be disabled if Blosc is also disabled. ]=] ON)
option(USE_LOG4CPLUS [=[
Use log4cplus while building openvdb components. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the
located openvdb build configuration to decide on log4cplus support. You may set this to on to force log4cplus
to be used if you know it to be required.]=] OFF)
option(USE_EXR [=[
Use OpenEXR while building openvdb components. If enabled, this also switches USE_IMATH_HALF to ON unless
it too was provided and set to OFF.]=] OFF)
option(USE_IMATH_HALF [=[
Use the definiton of half floating point types from the Imath library. If OFF, the embedded definition provided
by OpenVDB is used. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the located openvdb build configuration
to select the correct half support. You may set this to on to force Imath half to be used if you know it to be
required.]=] ${USE_EXR})
option(USE_PNG "Use PNG while building openvdb components." OFF)
option(USE_AX "Use OpenVDB AX while building openvdb components." ${OPENVDB_BUILD_AX})
option(USE_NANOVDB "Use NanoVDB while building openvdb components." ${OPENVDB_BUILD_NANOVDB})

cmake_dependent_option(OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING
  "Disable the implicit linking of Boost libraries on Windows" ON "WIN32" OFF)
option(USE_STATIC_DEPENDENCIES [=[
Only search for and use static libraries. If OFF the shared versions of requried libraries are prioritised, falling
back to static libraries. Forcing individual static dependencies can be enabled by setting XXX_USE_STATIC_LIBS
to ON, where XXX is the package name. On Windows this behaviour is less strict, with any located libraries assumed
to be static. This has no effect for any packages found with Xxx_DIR (i.e. CMake CONFIG mode).]=] OFF)
option(DISABLE_DEPENDENCY_VERSION_CHECKS [=[
Disable minimum version checks for OpenVDB dependencies. It is strongly recommended that this remains disabled.
Consider updating your dependencies where possible if encountering minimum requirement CMake errors.]=] OFF)
option(DISABLE_CMAKE_SEARCH_PATHS [=[
Disable CMakes default system search paths when locating dependencies. When enabled, CMake will fall back to
its default system search routine if it cannot find a dependency with the provided settings. When disabled, only
paths provided through the Xxx_ROOT, supported XXX_INCLUDEDIR/XXX_LIBRARYDIR variables or the SYSTEM_LIBRARY_PATHS
list will be searched.]=] OFF)

set(SYSTEM_LIBRARY_PATHS "" CACHE STRING [=[
A global list of library paths to additionally use into when searching for dependencies.]=])
set(MSVC_MP_THREAD_COUNT "" CACHE STRING [=[
The number of threads to use when invoking MSVC builds with cmake --build. CMake defaults to 1. Note that
cmake --parallel and CMAKE_BUILD_PARALLEL_LEVEL only change the number of PROJECTS that are built in parallel,
NOT the number of TU's ]=])
option(MSVC_COMPRESS_PDB "Whether to attempt to compress PDB symbol data in MSVC debug builds." OFF)

set(_CONCURRENT_MALLOC_OPTIONS None Auto Jemalloc Tbbmalloc)
if(NOT CONCURRENT_MALLOC)
  set(CONCURRENT_MALLOC Auto CACHE STRING
    "Explicitly link the OpenVDB executables against a particular concurrent malloc library.
    Options are: None Auto Jemalloc Tbbmalloc. Although not required, it is strongly recommended
    to use a concurrent memory allocator. Has no effect if OPENVDB_BUILD_BINARIES and
    OPENVDB_BUILD_UNITTESTS are false. Auto is the default and implies Jemalloc, unless USE_MAYA
    is ON or Jemalloc is unavailable, in which case Tbbmalloc is used. Note that this is not
    linked into library builds and defers this choice to downstream applications via explicit
    CMake targets." FORCE
  )
elseif(NOT ${CONCURRENT_MALLOC} IN_LIST _CONCURRENT_MALLOC_OPTIONS)
  message(WARNING "Unrecognized value for CONCURRENT_MALLOC, using Auto instead.")
  set(CONCURRENT_MALLOC Auto CACHE STRING FORCE)
endif()

set(_OPENVDB_SIMD_OPTIONS None SSE42 AVX)
if(NOT OPENVDB_SIMD)
  set(OPENVDB_SIMD None CACHE STRING
    "Choose whether to enable SIMD compiler flags or not, options are: None SSE42 AVX.
    Although not required, it is strongly recommended to enable SIMD. AVX implies SSE42.
    None is the default." FORCE
  )
elseif(NOT ${OPENVDB_SIMD} IN_LIST _OPENVDB_SIMD_OPTIONS)
  message(WARNING "Unrecognized or unsupported value for OPENVDB_SIMD, "
    "using None instead.")
  set(OPENVDB_SIMD None CACHE STRING FORCE)
endif()

if(USE_BLOSC AND NOT USE_ZLIB)
  message(WARNING "ZLib can only be disabled if Blosc is also disabled. Enabling ZLib.")
endif()

if(USE_NANOVDB)
  if(OPENVDB_BUILD_CORE AND NOT OPENVDB_BUILD_NANOVDB)
    message(FATAL_ERROR "Invalid CMake build configuration:
      OPENVDB_BUILD_CORE:    [ON]
      OPENVDB_BUILD_NANOVDB: [OFF]
      USE_NANOVDB:           [ON]
    If you are rebuilding the core library and using sub components, building
    these sub components must also be enabled (i.e. -DOPENVDB_BUILD_NANOVDB=ON).")
  endif()
endif()

if(USE_AX)
  if(OPENVDB_BUILD_CORE AND NOT OPENVDB_BUILD_AX)
    message(FATAL_ERROR "Invalid CMake build configuration:
      OPENVDB_BUILD_CORE: [ON]
      OPENVDB_BUILD_AX:   [OFF]
      USE_AX:             [ON]
    If you are rebuilding the core library and using sub components, building
    these sub components must also be enabled (i.e. -DOPENVDB_BUILD_AX=ON).")
  endif()
endif()

if(OPENVDB_BUILD_HOUDINI_PLUGIN OR OPENVDB_BUILD_HOUDINI_ABITESTS)
  set(USE_HOUDINI ON)
endif()

if(OPENVDB_BUILD_MAYA_PLUGIN)
  set(USE_MAYA ON)
endif()

if(USE_MAYA AND USE_HOUDINI)
  # @todo technically this is possible so long as library versions match
  # exactly but it's difficult to validate and dangerous
  message(FATAL_ERROR "Cannot build both Houdini and Maya plugins against "
    "the same core dependencies. Plugins must be compiled separately to "
    "ensure the required DCC dependencies are met.")
endif()

###### Deprecated options

if(OPENVDB_CODE_COVERAGE)
  message(FATAL_ERROR "The OPENVDB_CODE_COVERAGE option  has been removed. Choose instead the unique"
      "build type -DCMAKE_BUILD_TYPE=coverage")
endif()
if(OPENVDB_BUILD_HOUDINI_SOPS)
  message(FATAL_ERROR "The OPENVDB_BUILD_HOUDINI_SOPS option has been removed. Use OPENVDB_BUILD_HOUDINI_PLUGIN.")
endif()
if(DEFINED USE_SYSTEM_LIBRARY_PATHS)
  message(FATAL_ERROR "The USE_SYSTEM_LIBRARY_PATHS option has been removed. Use DISABLE_CMAKE_SEARCH_PATHS.")
endif()

# Various root level CMake options which are marked as advanced

mark_as_advanced(
  CONCURRENT_MALLOC
  DISABLE_CMAKE_SEARCH_PATHS
  DISABLE_DEPENDENCY_VERSION_CHECKS
  OPENVDB_BUILD_HOUDINI_ABITESTS
  OPENVDB_CXX_STRICT
  OPENVDB_ENABLE_RPATH
  OPENVDB_USE_DELAYED_LOADING
  OPENVDB_FUTURE_DEPRECATION
  OPENVDB_SIMD
  SYSTEM_LIBRARY_PATHS
  USE_CCACHE
  USE_COLORED_OUTPUT
  USE_HOUDINI
  USE_IMATH_HALF
  USE_LOG4CPLUS
  USE_MAYA
  MSVC_COMPRESS_PDB
)

#########################################################################

# Configure MINIMUM and FUTURE_MINIMUM version variables

include(cmake/config/OpenVDBVersions.cmake)

foreach(DEPRECATED_ABI ${OPENVDB_DEPRECATED_ABI_LIST})
  option(OPENVDB_USE_DEPRECATED_ABI_${DEPRECATED_ABI} "Bypass minimum ABI requirement checks" OFF)
  mark_as_advanced(OPENVDB_USE_DEPRECATED_ABI_${DEPRECATED_ABI})
endforeach()
option(OPENVDB_USE_FUTURE_ABI_${FUTURE_OPENVDB_ABI_VERSION} "Bypass future ABI check" OFF)
mark_as_advanced(OPENVDB_USE_FUTURE_ABI_${FUTURE_OPENVDB_ABI_VERSION})

#########################################################################

enable_testing()

# Add our cmake modules

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Add cmake modules to installation command

if(OPENVDB_INSTALL_CMAKE_MODULES)
  set(OPENVDB_CMAKE_MODULES
    cmake/FindBlosc.cmake
    cmake/FindJemalloc.cmake
    cmake/FindLog4cplus.cmake
    cmake/FindOpenEXR.cmake
    cmake/FindOpenVDB.cmake
    cmake/FindTBB.cmake
    cmake/OpenVDBGLFW3Setup.cmake
    cmake/OpenVDBHoudiniSetup.cmake
    cmake/OpenVDBMayaSetup.cmake
    cmake/OpenVDBUtils.cmake
  )
  install(FILES ${OPENVDB_CMAKE_MODULES} DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenVDB)
endif()

# Configure component dependencies by loading the Houdini/Maya setup
# scripts. These also find the Houdini/Maya installations

if(USE_HOUDINI)
  include(OpenVDBHoudiniSetup)
endif()

if(USE_MAYA)
  include(OpenVDBMayaSetup)
endif()

if(OPENVDB_BUILD_DOCS)
  add_subdirectory(doc)
endif()

if(NOT OPENVDB_BUILD_AX AND OPENVDB_BUILD_AX_GRAMMAR)
  # only re-generate the AX grammar
  set(OPENVDB_AX_SHARED OFF CACHE BOOL "" FORCE)
  set(OPENVDB_AX_STATIC OFF CACHE BOOL "" FORCE)
  add_subdirectory(openvdb_ax/openvdb_ax)
endif()

# Early exit if there's nothing to build

if(NOT (
    OPENVDB_BUILD_CORE OR
    OPENVDB_BUILD_BINARIES OR
    OPENVDB_BUILD_AX OR
    OPENVDB_BUILD_AX_UNITTESTS OR
    OPENVDB_BUILD_PYTHON_MODULE OR
    OPENVDB_BUILD_UNITTESTS OR
    OPENVDB_BUILD_HOUDINI_PLUGIN OR
    OPENVDB_BUILD_HOUDINI_ABITESTS OR
    OPENVDB_BUILD_MAYA_PLUGIN OR
    OPENVDB_BUILD_NANOVDB)
  )
  return()
endif()

#########################################################################

# ccache setup

if(USE_CCACHE)
  find_program(CCACHE_PATH ccache)
  if(CCACHE_PATH)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PATH})
    message(STATUS "Using ccache: ${CCACHE_PATH}")
  endif()
endif()

#########################################################################

# Configure general CMake and CXX settings

include(cmake/config/OpenVDBCXX.cmake)

#########################################################################

# Configure malloc library. Use Jemalloc if available, tbbmalloc otherwise

if(CONCURRENT_MALLOC STREQUAL "Auto")
  find_package(Jemalloc QUIET)
  if(NOT TARGET Jemalloc::jemalloc)
    message(WARNING "Unable to find Jemalloc, attempting to fall back to TBB malloc.
      It is recommended to use Jemalloc for optimum performance.")
    set(CONCURRENT_MALLOC "Tbbmalloc")
  else()
    set(CONCURRENT_MALLOC "Jemalloc")
  endif()
endif()

# Configure SIMD. AVX implies SSE 4.2.

if(OPENVDB_SIMD STREQUAL "AVX")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mavx>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-msse4.2>")
  endif()
  add_compile_definitions("$<$<COMPILE_LANGUAGE:CXX>:OPENVDB_USE_AVX>")
  add_compile_definitions("$<$<COMPILE_LANGUAGE:CXX>:OPENVDB_USE_SSE42>")
elseif(OPENVDB_SIMD STREQUAL "SSE42")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-msse4.2>")
  endif()
  add_compile_definitions("$<$<COMPILE_LANGUAGE:CXX>:OPENVDB_USE_SSE42>")
endif()

#########################################################################

# Configure our cmake modules to only search for static libraries

if(USE_STATIC_DEPENDENCIES)
  set(BLOSC_USE_STATIC_LIBS ON)
  set(OPENEXR_USE_STATIC_LIBS ON)
  set(TBB_USE_STATIC_LIBS ON)
  set(LOG4CPLUS_USE_STATIC_LIBS ON)
  set(JEMALLOC_USE_STATIC_LIBS ON)
  set(GTEST_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC_LIBS ON)
  # @todo  glfw needs custom support.
  # set(GLFW_USE_STATIC_LIBS ON)
endif()

# Configure OpenVDB Library and ABI versions

set(NEEDS_OPENVDB OFF)

if(OPENVDB_BUILD_AX OR
   OPENVDB_BUILD_BINARIES OR
   OPENVDB_BUILD_UNITTESTS OR
   OPENVDB_BUILD_HOUDINI_PLUGIN OR
   OPENVDB_BUILD_MAYA_PLUGIN OR
   OPENVDB_BUILD_PYTHON_MODULE OR
   OPENVDB_BUILD_HOUDINI_ABITESTS)
  set(NEEDS_OPENVDB ON)
endif()

if(NOT OPENVDB_BUILD_CORE AND NEEDS_OPENVDB)
  # Find VDB installation and determine lib/abi versions. This resets the
  # version and ABI numbers
  find_package(OpenVDB REQUIRED)
  # Check ABI version was found and explicitly error if attempting to build against
  # an incompatible Houdini version
  if(OpenVDB_ABI AND OPENVDB_HOUDINI_ABI)
    if(NOT ${OpenVDB_ABI} EQUAL ${OPENVDB_HOUDINI_ABI})
      message(FATAL_ERROR "Located OpenVDB installation is not ABI compatible with "
        "Houdini Version ${Houdini_VERSION}. Requires ABI ${OPENVDB_HOUDINI_ABI}, found "
        "ABI ${OpenVDB_ABI}.")
    endif()
  endif()
endif()

message(STATUS "Configuring for OpenVDB Version ${OpenVDB_VERSION}")

# Locate openvdb_ax if necessary

if(NOT OPENVDB_BUILD_AX AND
    (OPENVDB_BUILD_AX_UNITTESTS OR
    USE_AX))
  find_package(OpenVDB REQUIRED COMPONENTS openvdb_ax)
endif()

# Locate nanovdb if necessary

if(NOT OPENVDB_BUILD_NANOVDB AND
   USE_NANOVDB)
  find_package(OpenVDB REQUIRED COMPONENTS nanovdb)
endif()


# Validate the OpenVDB ABI Version. If OpenVDB_ABI is not set, we're either building
# the core library OR the ABI hasn't been deduced from a VDB installation. Use the
# value from OPENVDB_ABI_VERSION_NUMBER, falling back to the lib major version number

if(NOT OpenVDB_ABI)
  if(OPENVDB_ABI_VERSION_NUMBER)
    set(OpenVDB_ABI ${OPENVDB_ABI_VERSION_NUMBER})
  else()
    set(OpenVDB_ABI ${OpenVDB_MAJOR_VERSION})
  endif()
endif()

# From the deduced ABI, check against the required ABI for Houdini (if set).
# Forcefully set the ABI to the required value if necessary - do this after to
# explicitly warn the user if their chosen value is different.

if(OPENVDB_HOUDINI_ABI AND (NOT "${OpenVDB_ABI}" EQUAL "${OPENVDB_HOUDINI_ABI}"))
  message(WARNING "CMake will explicitly set the value of OPENVDB_ABI_VERSION_NUMBER to "
    "${OPENVDB_HOUDINI_ABI} to match the ABI of the target Houdini Version.")
  set(OpenVDB_ABI ${OPENVDB_HOUDINI_ABI})
endif()

# Validate ABI value if outside supported range

if(OpenVDB_ABI LESS MINIMUM_OPENVDB_ABI_VERSION)
  message(FATAL_ERROR "OpenVDB ABI versions earlier than ${MINIMUM_OPENVDB_ABI_VERSION} are "
    "no longer supported.")
endif()

if(OpenVDB_ABI IN_LIST OPENVDB_DEPRECATED_ABI_LIST)
  if(NOT OPENVDB_USE_DEPRECATED_ABI_${OpenVDB_ABI})
    message(FATAL_ERROR "OpenVDB ABI versions earlier than ${OpenVDB_MAJOR_VERSION} "
      "are deprecated. Set CMake option OPENVDB_USE_DEPRECATED_ABI_${OpenVDB_ABI} to ON to "
      "suppress this error.")
  endif()

  message(DEPRECATION "OpenVDB ABI versions earlier than ${OpenVDB_MAJOR_VERSION} "
    "are deprecated and will soon be removed.")
  # global target definition
  add_definitions(-DOPENVDB_USE_DEPRECATED_ABI_${OpenVDB_ABI})
endif()

if(OpenVDB_ABI EQUAL FUTURE_OPENVDB_ABI_VERSION)
  if(NOT OPENVDB_USE_FUTURE_ABI_${OpenVDB_ABI})
    message(FATAL_ERROR "OpenVDB ABI version ${OpenVDB_ABI} contains changes to ABI that are still "
      "in active development and have not been finalized. Set CMake option "
      "OPENVDB_USE_FUTURE_ABI_${OpenVDB_ABI} to ON to suppress this error.")
  endif()

  message(WARNING "OpenVDB ABI version ${OpenVDB_ABI} contains changes to ABI that are still "
    "in active development and have not been finalized.")
  # global target definition
  add_definitions(-DOPENVDB_USE_FUTURE_ABI_${OpenVDB_ABI})
endif()

message(STATUS "Configuring for OpenVDB ABI Version ${OpenVDB_ABI}")

# Always force set as we may need to change it if it's incompatible with Houdini
set(OPENVDB_ABI_VERSION_NUMBER ${OpenVDB_ABI} CACHE STRING [=[
Build for compatibility with version N of the OpenVDB Grid ABI, where N is 6, 7, 8 etc. (some newer features
will be disabled). If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the installed vdb_print binary to
determine the ABI number. You may set this to force a given ABI number.]=] FORCE)

##########################################################################

if(OPENVDB_BUILD_CORE)
  add_subdirectory(openvdb/openvdb)
endif()

if(OPENVDB_BUILD_AX)
  add_subdirectory(openvdb_ax/openvdb_ax)
endif()

if(OPENVDB_BUILD_PYTHON_MODULE)
  add_subdirectory(openvdb/openvdb/python)
endif()

if(OPENVDB_BUILD_BINARIES)
  add_subdirectory(openvdb_cmd)
endif()

if(OPENVDB_BUILD_UNITTESTS)
  add_subdirectory(openvdb/openvdb/unittest)
endif()

# Tests the vdb_ax binary so needs to come after openvdb_cmd
if(OPENVDB_BUILD_AX_UNITTESTS)
  add_subdirectory(openvdb_ax/openvdb_ax/test)
endif()

if(OPENVDB_BUILD_HOUDINI_PLUGIN)
  add_subdirectory(openvdb_houdini/openvdb_houdini)
endif()

if(OPENVDB_BUILD_HOUDINI_ABITESTS)
  add_subdirectory(openvdb_houdini/openvdb_houdini/abitest)
endif()

if(OPENVDB_BUILD_MAYA_PLUGIN)
  add_subdirectory(openvdb_maya/openvdb_maya)
endif()

if(OPENVDB_BUILD_NANOVDB)
  add_subdirectory(nanovdb/nanovdb)
endif()

##########################################################################

if(OPENVDB_ENABLE_UNINSTALL)
  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/Uninstall.cmake
  )
endif()
