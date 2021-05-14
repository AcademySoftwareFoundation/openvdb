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

    https://academysoftwarefoundation.github.io/openvdb/dependencies.html

  And the documentation on building OpenVDB for more in depth installation
  instructions:

    https://academysoftwarefoundation.github.io/openvdb/build.html

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
cmake_minimum_required(VERSION 3.12)

# CMP0091 allows for MSVC ABI targetting via CMAKE_MSVC_RUNTIME_LIBRARY
# from CMake 3.15 and above. Must come before project().
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

###### Version

set(OpenVDB_MAJOR_VERSION 8)
set(OpenVDB_MINOR_VERSION 1)
set(OpenVDB_PATCH_VERSION 0)
set(OpenVDB_VERSION "${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}")

project(OpenVDB LANGUAGES CXX VERSION ${OpenVDB_VERSION})

###### OpenVDB Build/Component Options

include(CMakeDependentOption)
include(GNUInstallDirs)

# todo epydoc and pdflatex
option(OPENVDB_BUILD_CORE "Enable the core OpenVDB library. Both static and shared versions are enabled by default" ON)
option(OPENVDB_BUILD_BINARIES "Enable the vdb binaries. Only vdb_print is enabled by default" ON)
option(OPENVDB_BUILD_PYTHON_MODULE "Build the pyopenvdb Python module" OFF)
option(OPENVDB_BUILD_UNITTESTS "Build the OpenVDB unit tests" OFF)
option(OPENVDB_BUILD_DOCS "Build the OpenVDB documentation" OFF)
option(OPENVDB_BUILD_HOUDINI_PLUGIN "Build the Houdini plugin" OFF)
option(OPENVDB_BUILD_HOUDINI_ABITESTS "Build the Houdini ABI tests" OFF)

option(OPENVDB_BUILD_AX "Build the OpenVDB AX library" OFF)
option(OPENVDB_BUILD_AX_BINARIES "Build the OpenVDB AX command line binary" OFF)
option(OPENVDB_BUILD_AX_UNITTESTS "Build the OpenVDB AX unit tests" OFF)

option(OPENVDB_BUILD_MAYA_PLUGIN "Build the Maya plugin" OFF)
option(OPENVDB_ENABLE_RPATH "Build with RPATH information" ON)
option(OPENVDB_CXX_STRICT "Enable or disable pre-defined compiler warnings" OFF)
cmake_dependent_option(OPENVDB_INSTALL_CMAKE_MODULES
  "Install the provided OpenVDB CMake modules when building the core library"
  ON "OPENVDB_BUILD_CORE" OFF)
option(USE_HOUDINI [=[
Build the library against a Houdini installation. Turns on automatically if OPENVDB_BUILD_HOUDINI_PLUGIN is enabled.
When enabled, you do not need to provide dependency locations for TBB, Blosc, IlmBase and OpenEXR. Boost must be
provided. IlmBase/OpenEXR can optionally be provided if Houdini Version >= 17.5.]=] OFF)
option(USE_MAYA [=[
Build the library against a Maya installation. Turns on automatically if OPENVDB_BUILD_MAYA_PLUGIN is enabled.
When enabled, you do not need to provide dependency locations for TBB. All other dependencies must be provided.]=] OFF)
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
option(USE_IMATH_HALF [=[
Use the definiton of half floating point types from the Imath library. If OFF, the embedded definition provided
by OpenVDB is used. If OPENVDB_BUILD_CORE is OFF, CMake attempts to query the located openvdb build configuration
to select the correct half support. You may set this to on to force Imath half to be used if you know it to be
required.]=] OFF)
cmake_dependent_option(OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING
  "Disable the implicit linking of Boost libraries on Windows" ON "WIN32" OFF)
option(USE_CCACHE "Build using Ccache if found on the path" ON)
option(USE_STATIC_DEPENDENCIES [=[
Only search for and use static libraries. If OFF the shared versions of requried libraries are prioritised, falling
back to static libraries. Forcing individual static dependencies can be enabled by setting XXX_USE_STATIC_LIBS
to ON, where XXX is the package name. On Windows this behaviour is less strict, with any located libraries assumed
to be static. ]=] OFF)
option(DISABLE_DEPENDENCY_VERSION_CHECKS [=[
Disable minimum version checks for OpenVDB dependencies. It is strongly recommended that this remains disabled.
Consider updating your dependencies where possible if encountering minimum requirement CMake errors.]=] OFF)
option(DISABLE_CMAKE_SEARCH_PATHS [=[
Disable CMakes default system search paths when locating dependencies. When enabled, CMake will fall back to
its default system search routine if it cannot find a dependency with the provided settings. When disabled, only
paths provided through the Xxx_ROOT, supported XXX_INCLUDEDIR/XXX_LIBRARYDIR variables or the SYSTEM_LIBRARY_PATHS
list will be searched.]=] OFF)
option(OPENVDB_USE_DEPRECATED_ABI_6 "Bypass minimum ABI requirement checks" OFF)
option(OPENVDB_USE_FUTURE_ABI_9 "Bypass future ABI check" OFF)
option(OPENVDB_FUTURE_DEPRECATION "Generate messages for upcoming deprecation" OFF)
option(OPENVDB_ENABLE_UNINSTALL "Adds a CMake uninstall target." ON)
option(USE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF)
option(USE_PKGCONFIG "Use pkg-config to search for dependent libraries." ON)

set(SYSTEM_LIBRARY_PATHS "" CACHE STRING [=[
A global list of library paths to additionally use into when searching for dependencies.]=])
set(MSVC_MP_THREAD_COUNT "" CACHE STRING [=[
The number of threads to use when invoking MSVC builds with cmake --build. CMake defaults to 1. Note that
cmake --parallel and CMAKE_BUILD_PARALLEL_LEVEL only change the number of PROJECTS that are built in parallel,
NOT the number of TU's ]=])


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

# Top-level location for all openvdb headers
set(OPENVDB_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/openvdb")

###### Deprecated options

if(USE_EXR)
  message(FATAL_ERROR "The USE_EXR option has been removed. Use USE_IMATH_HALF to use the Half "
    "implementation from IMath, or leave OFF to use the new embedded version of half floating point types.")
endif()
if(OPENVDB_BUILD_HOUDINI_SOPS)
  message(FATAL_ERROR "The OPENVDB_BUILD_HOUDINI_SOPS option has been removed. Use OPENVDB_BUILD_HOUDINI_PLUGIN.")
endif()
if(DEFINED USE_SYSTEM_LIBRARY_PATHS)
  message(FATAL_ERROR "The USE_SYSTEM_LIBRARY_PATHS option has been removed. Use DISABLE_CMAKE_SEARCH_PATHS.")
endif()

# Various root level CMake options which are marked as advanced

mark_as_advanced(
  OPENVDB_CXX_STRICT
  OPENVDB_ENABLE_RPATH
  USE_HOUDINI
  USE_MAYA
  USE_LOG4CPLUS
  USE_IMATH_HALF
  USE_CCACHE
  OPENVDB_BUILD_HOUDINI_ABITESTS
  DISABLE_DEPENDENCY_VERSION_CHECKS
  DISABLE_CMAKE_SEARCH_PATHS
  OPENVDB_USE_DEPRECATED_ABI_6
  OPENVDB_USE_FUTURE_ABI_9
  OPENVDB_FUTURE_DEPRECATION
  CONCURRENT_MALLOC
  USE_COLORED_OUTPUT
  SYSTEM_LIBRARY_PATHS
  OPENVDB_SIMD
)

# Configure minimum version requirements - some are treated specially and fall
# outside of the DISABLE_DEPENDENCY_VERSION_CHECKS catch

set(MINIMUM_CXX_STANDARD 14)

# @note  Blosc version is currently treated as exception which must be adhered
# to. The minimum version must be at least 1.5.0. Previous versions are incompatible.
# Later versions (including 1.5.4), can be buggy on certain platforms.
set(MINIMUM_BLOSC_VERSION 1.5.0)
# @note  ABI always enforced so the correct deprecation messages are available.
# OPENVDB_USE_DEPRECATED_ABI_<VERSION> should be used to circumvent this
set(MINIMUM_OPENVDB_ABI_VERSION 6)
set(FUTURE_MINIMUM_OPENVDB_ABI_VERSION 7)
set(FUTURE_OPENVDB_ABI_VERSION 9)

if(NOT DISABLE_DEPENDENCY_VERSION_CHECKS)
  # @note  Currently tracking CY2019 of the VFX platform where available
  set(MINIMUM_GCC_VERSION 6.3.1)
  set(MINIMUM_CLANG_VERSION 3.8)
  set(MINIMUM_ICC_VERSION 17)
  set(MINIMUM_MSVC_VERSION 19.10)

  set(MINIMUM_BOOST_VERSION 1.61)  # @warning should be 1.66, but H18 ships with 1.61
  set(MINIMUM_ILMBASE_VERSION 2.2) # @warning should be 2.3, but H18 ships with 2.2
  set(MINIMUM_OPENEXR_VERSION 2.2) # @warning should be 2.3, but H18 ships with 2.2
  set(MINIMUM_ZLIB_VERSION 1.2.7)
  set(MINIMUM_TBB_VERSION 2018.0)
  set(MINIMUM_LLVM_VERSION 6.0.0)

  set(MINIMUM_PYTHON_VERSION 2.7)
  set(MINIMUM_NUMPY_VERSION 1.14.0)

  set(MINIMUM_GOOGLETEST_VERSION 1.8)
  set(MINIMUM_GLFW_VERSION 3.1)
  set(MINIMUM_LOG4CPLUS_VERSION 1.1.2)
  set(MINIMUM_HOUDINI_VERSION 18.0)

  # These always promote warnings rather than errors
  set(MINIMUM_MAYA_VERSION 2017)
  set(MINIMUM_DOXYGEN_VERSION 1.8.8)
endif()

# VFX 20 deprecations to transition to MINIMUM_* variables in OpenVDB 9.0.0

# Note: CMake 3.15 chosen as possible next version due to Windows support
# with CMAKE_MSVC_RUNTIME_LIBRARY, numpy modules with CMake 3.14 and simplified
# python linkage on macOS (Python::Module target in 3.15)
set(FUTURE_MINIMUM_CMAKE_VERSION 3.15)

# No compiler upgrades planned
#set(FUTURE_MINIMUM_GCC_VERSION 6.3.1)
#set(FUTURE_MINIMUM_ICC_VERSION 17)
#set(FUTURE_MINIMUM_MSVC_VERSION 19.10)

set(FUTURE_MINIMUM_ILMBASE_VERSION 2.4)
set(FUTURE_MINIMUM_OPENEXR_VERSION 2.4)
set(FUTURE_MINIMUM_BOOST_VERSION 1.70)
set(FUTURE_MINIMUM_TBB_VERSION 2019.0)
set(FUTURE_MINIMUM_PYTHON_VERSION 3.7)
set(FUTURE_MINIMUM_NUMPY_VERSION 1.17.0)
set(FUTURE_MINIMUM_HOUDINI_VERSION 18.5)
set(FUTURE_MINIMUM_LLVM_VERSION 8.0.0)

#########################################################################

# General CMake and CXX settings

if(FUTURE_MINIMUM_CMAKE_VERSION)
  if(${CMAKE_VERSION} VERSION_LESS ${FUTURE_MINIMUM_CMAKE_VERSION})
    message(DEPRECATION "Support for CMake versions < ${FUTURE_MINIMUM_CMAKE_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${MINIMUM_CXX_STANDARD} CACHE STRING
    "The C++ standard whose features are requested to build OpenVDB components." FORCE)
elseif(CMAKE_CXX_STANDARD LESS ${MINIMUM_CXX_STANDARD})
  message(FATAL_ERROR "Provided C++ Standard is less than the supported minimum."
    "Required is at least \"${MINIMUM_CXX_STANDARD}\" (found ${CMAKE_CXX_STANDARD})")
endif()

# Configure MS Runtime

if(CMAKE_MSVC_RUNTIME_LIBRARY AND CMAKE_VERSION VERSION_LESS 3.15)
  # CMAKE_MSVC_RUNTIME_LIBRARY selects the MSVC runtime library for use by compilers
  # targeting the MSVC ABI. Options are: MultiThreaded MultiThreadedDLL MultiThreadedDebug
  # MultiThreadedDebugDLL. If empty, CMake defaults to MultiThreaded$<$<CONFIG:Debug>:Debug>DLL
  message(FATAL_ERROR "CMAKE_MSVC_RUNTIME_LIBRARY support is only available from CMake 3.15")
endif()
if(WIN32 AND CMAKE_MSVC_RUNTIME_LIBRARY)
  message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY set to target ${CMAKE_MSVC_RUNTIME_LIBRARY}")

  # Configure Boost library varient on Windows
  if(NOT Boost_USE_STATIC_RUNTIME)
    set(Boost_USE_STATIC_RUNTIME OFF)
    if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreaded OR
       CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebug)
      set(Boost_USE_STATIC_RUNTIME ON)
    endif()
  endif()
  if(NOT Boost_USE_DEBUG_RUNTIME)
    set(Boost_USE_DEBUG_RUNTIME OFF)
    if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebugDLL OR
       CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebug)
      set(Boost_USE_DEBUG_RUNTIME ON)
    endif()
  endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(OPENVDB_ENABLE_RPATH)
  # Configure rpath for installation base on the following:
  # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# For CMake's find Threads module which brings in pthread - This flag
# forces the compiler -pthread flag vs -lpthread
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

enable_testing()

# Add our cmake modules

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Add backports to support older versions of CMake
#  FindNumPy.cmake is needed if CMake < 3.14

if(${CMAKE_VERSION} VERSION_LESS 3.14)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/backports")
endif()

# Add cmake modules to installation command
# @todo fix our glew cmake module

if(OPENVDB_INSTALL_CMAKE_MODULES)
  set(OPENVDB_CMAKE_MODULES
    cmake/FindBlosc.cmake
    cmake/FindJemalloc.cmake
    cmake/FindIlmBase.cmake
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

# Configure DCC installation if necessary

if(OPENVDB_BUILD_HOUDINI_PLUGIN OR
   OPENVDB_BUILD_HOUDINI_ABITESTS)
  set(USE_HOUDINI ON)
endif()

if(OPENVDB_BUILD_MAYA_PLUGIN)
  set(USE_MAYA ON)
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

# Early exit if there's nothing to build

if(NOT (
    OPENVDB_BUILD_CORE OR
    OPENVDB_BUILD_BINARIES OR
    OPENVDB_BUILD_AX OR
    OPENVDB_BUILD_AX_BINARIES OR
    OPENVDB_BUILD_AX_UNITTESTS OR
    OPENVDB_BUILD_PYTHON_MODULE OR
    OPENVDB_BUILD_UNITTESTS OR
    OPENVDB_BUILD_HOUDINI_PLUGIN OR
    OPENVDB_BUILD_HOUDINI_ABITESTS OR
    OPENVDB_BUILD_MAYA_PLUGIN)
  )
  return()
endif()

if(WIN32)
  if(OPENVDB_BUILD_AX OR
     OPENVDB_BUILD_AX_BINARIES OR
     OPENVDB_BUILD_AX_UNITTESTS)
    message(FATAL_ERROR "Currently no support for building OpenVDB AX on Windows.")
  endif()
endif()

if(USE_MAYA AND USE_HOUDINI)
  # @todo technically this is possible so long as library versions match
  # exactly but it's difficult to validate and dangerous
  message(FATAL_ERROR "Cannot build both Houdini and Maya plugins against "
    "the same core dependencies. Plugins must be compiled separately to "
    "ensure the required DCC dependencies are met."
  )
endif()

#########################################################################

# ccache setup

if(USE_CCACHE)
  find_program(CCACHE_PATH ccache)
  if(CCACHE_PATH)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(STATUS "Using ccache: ${CCACHE_PATH}")
  endif()
endif()

# Build type configuration

if(OPENVDB_CODE_COVERAGE)
  message(DEPRECATION "The OPENVDB_CODE_COVERAGE option is deprecated. Choose instead the unique"
    "build type -DCMAKE_BUILD_TYPE=coverage")
  set(CMAKE_BUILD_TYPE coverage)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/config/OpenVDBBuildTypes.cmake)

if(CMAKE_BUILD_TYPE EQUAL coverage)
  # use .gcno extension instead of .cc.gcno
  # @note This is an undocumented internal cmake var and does not work
  # with multi config generators
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
endif()

# CMAKE_BUILD_TYPE is ignored for multi config generators i.e. MSVS
get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _isMultiConfig)
  message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")
endif()

#########################################################################

# Compiler configuration. Add definitions for a number of compiler warnings
# for sub projects and verify version requirements
# @todo  add definitions for Intel.

set(HAS_AVAILABLE_WARNINGS FALSE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_CLANG_VERSION)
    message(FATAL_ERROR "Insufficient clang++ version. Minimum required is "
      "\"${MINIMUM_CLANG_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
    )
  endif()
  if(OPENVDB_CXX_STRICT)
    message(STATUS "Configuring Clang CXX warnings")
    set(HAS_AVAILABLE_WARNINGS TRUE)
    add_compile_options(
      -Werror
      -Wall
      -Wextra
      -Wconversion
      -Wno-sign-conversion
    )
  endif()
  if(USE_COLORED_OUTPUT)
    message(STATUS "Enabling colored compiler output")
    add_compile_options(-fcolor-diagnostics)
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_GCC_VERSION)
    message(FATAL_ERROR "Insufficient g++ version. Minimum required is "
      "\"${MINIMUM_GCC_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
    )
  endif()
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_GCC_VERSION)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS FUTURE_MINIMUM_GCC_VERSION)
      message(DEPRECATION "Support for GCC versions < ${FUTURE_MINIMUM_GCC_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
  if(OPENVDB_CXX_STRICT)
    message(STATUS "Configuring GCC CXX warnings")
    set(HAS_AVAILABLE_WARNINGS TRUE)
    add_compile_options(
      -Werror
      -Wall
      -Wextra
      -pedantic
      -Wcast-align
      -Wcast-qual
      -Wconversion
      -Wdisabled-optimization
      -Woverloaded-virtual
    )
  endif()
  if(USE_COLORED_OUTPUT)
    message(STATUS "Enabling colored compiler output")
    add_compile_options(-fdiagnostics-color=always)
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_ICC_VERSION)
    message(FATAL_ERROR "Insufficient ICC version. Minimum required is "
      "\"${MINIMUM_ICC_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
    )
  endif()
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_ICC_VERSION)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS FUTURE_MINIMUM_ICC_VERSION)
      message(DEPRECATION "Support for ICC versions < ${FUTURE_MINIMUM_ICC_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_MSVC_VERSION)
    message(FATAL_ERROR "Insufficient MSVC version. Minimum required is "
      "\"${MINIMUM_MSVC_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
  )
  endif()
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_MSVC_VERSION)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS FUTURE_MINIMUM_MSVC_VERSION)
      message(DEPRECATION "Support for MSVC versions < ${FUTURE_MINIMUM_MSVC_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
  # Increase the number of sections that an object file can contain
  add_compile_options(/bigobj)
  # Excludes APIs such as Cryptography, DDE, RPC, Shell, and Windows Sockets
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  # Disable non-secure CRT library function warnings
  # https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/
  #     compiler-warning-level-3-c4996?view=vs-2019#unsafe-crt-library-functions
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  # Disable POSIX function name warnings
  # https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/
  #     compiler-warning-level-3-c4996?view=vs-2019#posix-function-names
  add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
  if(NOT OPENVDB_CXX_STRICT)
    message(STATUS "Suppressing some noisy MSVC CXX warnings, "
      "set OPENVDB_CXX_STRICT=ON to re-enable them.")
    # Conversion from int64_t to long
    add_compile_options(/wd4244)
    # It's not possible to use STL types in DLL interfaces in a portable and
    # reliable way so disable this warning
    add_compile_options(/wd4251)
    # Conversion from size_t to uLong
    add_compile_options(/wd4267)
    # Non dll-interface class used as base for dll-interface class
    add_compile_options(/wd4275)
    # Truncation from 'int' to 'bool'
    add_compile_options(/wd4305)
  endif()
  # If a user has explicitly requested a parallel build, configure this during CMake
  if(MSVC_MP_THREAD_COUNT AND MSVC_MP_THREAD_COUNT GREATER 1)
    add_compile_options(/MP${MSVC_MP_THREAD_COUNT})
  endif()
endif()

if(OPENVDB_CXX_STRICT AND NOT HAS_AVAILABLE_WARNINGS)
  message(WARNING "No available CXX warnings for compiler ${CMAKE_CXX_COMPILER_ID}")
endif()
unset(HAS_AVAILABLE_WARNINGS)

# Configure malloc library. Use Jemalloc for Linux and non-Maya, otherwise Tbbmalloc.
# * On Mac OSX, linking against Jemalloc < 4.3.0 seg-faults with this error:
#     malloc: *** malloc_zone_unregister() failed for 0xaddress
#   Houdini 17.5 and older ships with Jemalloc 3.6.0, so we make Tbbmalloc the default
#   on Mac OSX (https://github.com/jemalloc/jemalloc/issues/420). Later versions of
#   Jemalloc are thought to work, but haven't been tested.
# * On Windows, we follow SideFX's example in using Tbbmalloc due to the challenges
#   of injecting into the Windows runtime to replace the system allocator.

if((OPENVDB_BUILD_BINARIES OR OPENVDB_BUILD_UNITTESTS) AND CONCURRENT_MALLOC STREQUAL "Auto")
  if(WIN32 OR APPLE OR USE_MAYA)
    set(CONCURRENT_MALLOC "Tbbmalloc")
  else()
    find_package(Jemalloc QUIET)
    if(NOT TARGET Jemalloc::jemalloc)
      message(WARNING "Unable to find Jemalloc, attempting to fall back to TBB malloc.
        It is recommended to use Jemalloc for optimum performance.")
      set(CONCURRENT_MALLOC "Tbbmalloc")
    else()
      set(CONCURRENT_MALLOC "Jemalloc")
    endif()
  endif()
endif()

# Configure SIMD. AVX implies SSE 4.2.

if(OPENVDB_SIMD STREQUAL "AVX")
  add_compile_options(-mavx -msse4.2)
  add_definitions(-DOPENVDB_USE_AVX)
  add_definitions(-DOPENVDB_USE_SSE42)
elseif(OPENVDB_SIMD STREQUAL "SSE42")
  add_compile_options(-msse4.2)
  add_definitions(-DOPENVDB_USE_SSE42)
endif()

#########################################################################

# Configure our cmake modules to only search for static libraries

if(USE_STATIC_DEPENDENCIES)
  set(BLOSC_USE_STATIC_LIBS ON)
  set(OPENEXR_USE_STATIC_LIBS ON)
  set(ILMBASE_USE_STATIC_LIBS ON)
  set(TBB_USE_STATIC_LIBS ON)
  set(LOG4CPLUS_USE_STATIC_LIBS ON)
  set(JEMALLOC_USE_STATIC_LIBS ON)
  set(GTEST_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC_LIBS ON)
  # @todo  glfw needs custom support.
  # set(GLFW_USE_STATIC_LIBS ON)
endif()

# Configure OpenVDB Library and ABI versions

if(NOT OPENVDB_BUILD_CORE)
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
   (OPENVDB_BUILD_AX_BINARIES OR
    OPENVDB_BUILD_AX_UNITTESTS))
  find_package(OpenVDB REQUIRED COMPONENTS openvdb_ax)
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

if(FUTURE_MINIMUM_OPENVDB_ABI_VERSION AND OpenVDB_ABI LESS FUTURE_MINIMUM_OPENVDB_ABI_VERSION)
  if(NOT OPENVDB_USE_DEPRECATED_ABI_${OpenVDB_ABI})
    message(FATAL_ERROR "OpenVDB ABI versions earlier than ${FUTURE_MINIMUM_OPENVDB_ABI_VERSION} "
      "are deprecated. Set CMake option OPENVDB_USE_DEPRECATED_ABI_${OpenVDB_ABI} to ON to "
      "suppress this error.")
  endif()

  message(DEPRECATION "OpenVDB ABI versions earlier than ${FUTURE_MINIMUM_OPENVDB_ABI_VERSION} "
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

if(OPENVDB_BUILD_AX_BINARIES)
  add_subdirectory(openvdb_ax/openvdb_ax/cmd)
endif()

if(OPENVDB_BUILD_AX_UNITTESTS)
  add_subdirectory(openvdb_ax/openvdb_ax/test)
endif()

if(OPENVDB_BUILD_PYTHON_MODULE)
  add_subdirectory(openvdb/openvdb/python)
endif()

if(OPENVDB_BUILD_BINARIES)
  add_subdirectory(openvdb/openvdb/cmd)
endif()

if(OPENVDB_BUILD_UNITTESTS)
  add_subdirectory(openvdb/openvdb/unittest)
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

##########################################################################

if(OPENVDB_ENABLE_UNINSTALL)
  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/Uninstall.cmake
  )
endif()
