# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Core

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(OpenVDBCore LANGUAGES CXX)

include(GNUInstallDirs)

###### OpenVDB Core Options

option(OPENVDB_CORE_SHARED "Build dynamically linked version of the core library." ON)
option(OPENVDB_CORE_STATIC "Build statically linked version of the core library." ON)

set(OPENVDB_SHARED_LIBRARY_NAME "openvdb" CACHE STRING
  "The base name of the built shared openvdb library. Prefixed by \"lib\" on UNIX platforms."
)
set(OPENVDB_STATIC_LIBRARY_NAME "openvdb" CACHE STRING
  "The base name of the built static openvdb library. Prefixed by \"lib\"."
)

mark_as_advanced(
  OPENVDB_STATIC_LIBRARY_NAME
  OPENVDB_SHARED_LIBRARY_NAME
)

if(NOT OPENVDB_CORE_SHARED AND NOT OPENVDB_CORE_STATIC)
  message(FATAL_ERROR "Both static and shared core OpenVDB libraries have been disabled. "
    "At least one must be enabled when building the core library.")
endif()

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "------------- Configuring OpenVDBCore --------------")
message(STATUS "----------------------------------------------------")

##########################################################################

# Win32 checks - we don't set any explicit settings so allow for all build
# types, but warn in cases where they probably have not been intended

if(WIN32)
  if(VCPKG_TOOLCHAIN AND VCPKG_TARGET_TRIPLET)
    string(FIND ${VCPKG_TARGET_TRIPLET} "static" _BUILDING_STATIC_TRIPLET)
    if(_BUILDING_STATIC_TRIPLET EQUAL -1)
      # x64-windows
      if(OPENVDB_CORE_STATIC)
        # building the static lib - check to see if we've explicitly set /MD
        if(NOT CMAKE_MSVC_RUNTIME_LIBRARY OR NOT
          (CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDLL OR
           CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebugDLL))
          # /MD not set, /MT will be used
          message(WARNING "Detected VCPKG toolchain is using a mismatching triplet for "
            "OpenVDB build artifacts. ${VCPKG_TARGET_TRIPLET} is dynamic, but OPENVDB_CORE_STATIC "
            "is ON. The static build of OpenVDB may not build correctly. It it recommended to only "
            "configure either the static or shared library on Windows in a single run of CMake.")
        endif()
      endif()
    else()
      # x64-windows-static
      if(OPENVDB_CORE_SHARED)
        # building the shared lib - check to see if we've explicitly set /MT
        if(NOT CMAKE_MSVC_RUNTIME_LIBRARY OR NOT
          (CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreaded OR
           CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebug))
          # /MT not set, /MD will be used
          message(WARNING "Detected VCPKG toolchain is using a mismatching triplet for "
            "OpenVDB build artifacts. ${VCPKG_TARGET_TRIPLET} is static, but OPENVDB_CORE_SHARED "
            "is ON. The shared build of OpenVDB may not build correctly. It it recommended to only "
            "configure either the static or shared library on Windows in a single run of CMake.")
        endif()
      endif()
    endif()
  endif()

  if(CMAKE_MSVC_RUNTIME_LIBRARY)
    if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded" OR
       CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDebug")
      if(OPENVDB_CORE_SHARED)
         message(WARNING "Unexpected value for the Windows CRT with target build artifacts."
          "You are attempting to use the static CRT (/MT) with a dynamic configuration of OpenVDB."
          "This is not recommended and may cause failures.")
      endif()
    endif()
    if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDLL" OR
       CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDebugDLL")
      if(OPENVDB_CORE_STATIC)
         message(WARNING "Unexpected value for the Windows CRT with target build artifacts."
          "You are attempting to use the dyanmic CRT (/MD) with a static configuration of OpenVDB."
          "This is not recommended and may cause failures.")
      endif()
    endif()
  endif()
endif()

##########################################################################

# Collect and configure lib dependencies

if(OPENVDB_CORE_SHARED AND NOT Boost_USE_STATIC_LIBS)
  # @note  Both of these must be set for Boost 1.70 (VFX2020) to link against
  #        boost shared libraries (more specifically libraries built with -fPIC).
  #        http://boost.2283326.n4.nabble.com/CMake-config-scripts-broken-in-1-70-td4708957.html
  #        https://github.com/boostorg/boost_install/commit/160c7cb2b2c720e74463865ef0454d4c4cd9ae7c
  set(BUILD_SHARED_LIBS ON)
  set(Boost_USE_STATIC_LIBS OFF)
endif()

find_package(Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams system)
if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_BOOST_VERSION)
  if(${Boost_LIB_VERSION} VERSION_LESS FUTURE_MINIMUM_BOOST_VERSION)
    message(DEPRECATION "Support for Boost versions < ${FUTURE_MINIMUM_BOOST_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbb)
if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_TBB_VERSION)
  if(${Tbb_VERSION} VERSION_LESS FUTURE_MINIMUM_TBB_VERSION)
    message(DEPRECATION "Support for TBB versions < ${FUTURE_MINIMUM_TBB_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

if(USE_IMATH_HALF)
  find_package(IlmBase ${MINIMUM_ILMBASE_VERSION} REQUIRED COMPONENTS Half)
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_ILMBASE_VERSION)
    if(${IlmBase_VERSION} VERSION_LESS FUTURE_MINIMUM_ILMBASE_VERSION)
      message(DEPRECATION "Support for IlmBase versions < ${FUTURE_MINIMUM_ILMBASE_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
endif()

if(USE_LOG4CPLUS)
  # Find Log4CPlus libraries
  find_package(Log4cplus ${MINIMUM_LOG4CPLUS_VERSION} REQUIRED)
endif()

if(USE_BLOSC)
  # Find Blosc libraries
  find_package(Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED)
  if(Blosc_FOUND AND Blosc_VERSION VERSION_GREATER MINIMUM_BLOSC_VERSION)
    message(WARNING "The version of Blosc located is greater than ${MINIMUM_BLOSC_VERSION}. "
      "There have been reported issues with using later versions of Blosc with OpenVDB. "
      "OpenVDB has been tested fully against Blosc ${MINIMUM_BLOSC_VERSION}, it is "
      "recommended that you use this version where possible."
    )
  endif()
else()
  message(WARNING "Blosc support is disabled. It is strongly recommended to "
    "enable blosc for optimal builds of OpenVDB and to support compatible "
    "serialization of other OpenVDB installations."
  )
endif()

if(USE_BLOSC OR USE_ZLIB)
  if(USE_STATIC_DEPENDENCIES)
    set(_ZLIB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32)
      set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    endif()
  endif()
  find_package(ZLIB ${MINIMUM_ZLIB_VERSION} REQUIRED)
  if(USE_STATIC_DEPENDENCIES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ZLIB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
    unset(_ZLIB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()
endif()

if(UNIX)
  find_package(Threads REQUIRED)
endif()

# Set deps. Note that the order here is important. If we're building against
# Houdini 17.5 we must include IlmBase deps first to ensure the users chosen
# namespaced headers are correctly prioritized. Otherwise other include paths
# from shared installs (including houdini) may pull in the wrong headers

set(OPENVDB_CORE_DEPENDENT_LIBS
  Boost::iostreams
  Boost::system
)

if(USE_IMATH_HALF)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS IlmBase::Half)
endif()

if(WIN32)
  # Boost headers contain #pragma commands on Windows which causes Boost
  # libraries to be linked in automatically. Custom boost installations
  # may find that these naming conventions don't always match and can
  # cause linker errors. This option disables this feature of Boost. Note
  # -DBOOST_ALL_NO_LIB can also be provided manually.
  if(OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING)
      list(APPEND OPENVDB_CORE_DEPENDENT_LIBS
        Boost::disable_autolinking  # add -DBOOST_ALL_NO_LIB
      )
  endif()
endif()

if(USE_LOG4CPLUS)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Log4cplus::log4cplus)
endif()

# @todo blosc and zlib should be hidden (privately linked in):
# See FindOpenVDB.cmake

if(USE_BLOSC)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Blosc::blosc)
endif()

if(USE_BLOSC OR USE_ZLIB)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS ZLIB::ZLIB)
endif()

list(APPEND OPENVDB_CORE_DEPENDENT_LIBS TBB::tbb)

if(UNIX)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Threads::Threads)
endif()

##########################################################################

##### Core library configuration

# Configure the version header

if(USE_IMATH_HALF)
  set(OPENVDB_USE_IMATH_HALF 1)
endif()
if(USE_ZLIB)
  set(OPENVDB_USE_ZLIB 1)
  if(USE_BLOSC)
    set(OPENVDB_USE_BLOSC 1)
  endif()
endif()

# Configure VDB version in the form: ("%02x%02x%04x", major, minor, patch)
math(EXPR OPENVDB_PACKED_VERSION "${OpenVDB_MAJOR_VERSION} << 24")
math(EXPR OPENVDB_PACKED_VERSION "${OPENVDB_PACKED_VERSION} | ((${OpenVDB_MINOR_VERSION} & 0xFF) << 16)")
math(EXPR OPENVDB_PACKED_VERSION "${OPENVDB_PACKED_VERSION} | (${OpenVDB_PATCH_VERSION} & 0xFFFF)")

configure_file(version.h.in openvdb/version.h)

##########################################################################

set(OPENVDB_LIBRARY_SOURCE_FILES
  Grid.cc
  io/Archive.cc
  io/Compression.cc
  io/DelayedLoadMetadata.cc
  io/File.cc
  io/GridDescriptor.cc
  io/Queue.cc
  io/Stream.cc
  io/TempFile.cc
  math/Half.cc
  math/Maps.cc
  math/Proximity.cc
  math/QuantizedUnitVec.cc
  math/Transform.cc
  Metadata.cc
  MetaMap.cc
  openvdb.cc
  Platform.cc
  points/AttributeArray.cc
  points/AttributeArrayString.cc
  points/AttributeGroup.cc
  points/AttributeSet.cc
  points/StreamCompression.cc
  points/points.cc
  util/Formats.cc
  util/Util.cc
)

set(OPENVDB_LIBRARY_INCLUDE_FILES
  Exceptions.h
  Grid.h
  Metadata.h
  MetaMap.h
  openvdb.h
  Platform.h
  PlatformConfig.h
  Types.h
  TypeList.h
)

set(OPENVDB_LIBRARY_IO_INCLUDE_FILES
  io/Archive.h
  io/Compression.h
  io/DelayedLoadMetadata.h
  io/File.h
  io/GridDescriptor.h
  io/io.h
  io/Queue.h
  io/Stream.h
  io/TempFile.h
)

set(OPENVDB_LIBRARY_MATH_INCLUDE_FILES
  math/BBox.h
  math/ConjGradient.h
  math/Coord.h
  math/DDA.h
  math/FiniteDifference.h
  math/Half.h
  math/HalfLimits.h
  math/LegacyFrustum.h
  math/Maps.h
  math/Mat.h
  math/Mat3.h
  math/Mat4.h
  math/Math.h
  math/Operators.h
  math/Proximity.h
  math/QuantizedUnitVec.h
  math/Quat.h
  math/Ray.h
  math/Stats.h
  math/Stencils.h
  math/Transform.h
  math/Tuple.h
  math/Vec2.h
  math/Vec3.h
  math/Vec4.h
)

set(OPENVDB_LIBRARY_POINTS_INCLUDE_FILES
  points/AttributeArray.h
  points/AttributeArrayString.h
  points/AttributeGroup.h
  points/AttributeSet.h
  points/IndexFilter.h
  points/IndexIterator.h
  points/PointAdvect.h
  points/PointAttribute.h
  points/PointConversion.h
  points/PointCount.h
  points/PointDataGrid.h
  points/PointDelete.h
  points/PointGroup.h
  points/PointMask.h
  points/PointMove.h
  points/PointSample.h
  points/PointScatter.h
  points/StreamCompression.h
)

set(OPENVDB_LIBRARY_TOOLS_INCLUDE_FILES
  tools/Activate.h
  tools/ChangeBackground.h
  tools/Clip.h
  tools/Composite.h
  tools/Count.h
  tools/Dense.h
  tools/DenseSparseTools.h
  tools/Diagnostics.h
  tools/FastSweeping.h
  tools/Filter.h
  tools/FindActiveValues.h
  tools/GridOperators.h
  tools/GridTransformer.h
  tools/Interpolation.h
  tools/LevelSetAdvect.h
  tools/LevelSetFilter.h
  tools/LevelSetFracture.h
  tools/LevelSetMeasure.h
  tools/LevelSetMorph.h
  tools/LevelSetPlatonic.h
  tools/LevelSetRebuild.h
  tools/LevelSetSphere.h
  tools/LevelSetTracker.h
  tools/LevelSetUtil.h
  tools/Mask.h
  tools/Merge.h
  tools/MeshToVolume.h
  tools/Morphology.h
  tools/MultiResGrid.h
  tools/NodeVisitor.h
  tools/ParticleAtlas.h
  tools/ParticlesToLevelSet.h
  tools/PointAdvect.h
  tools/PointIndexGrid.h
  tools/PointPartitioner.h
  tools/PointScatter.h
  tools/PointsToMask.h
  tools/PoissonSolver.h
  tools/PotentialFlow.h
  tools/Prune.h
  tools/RayIntersector.h
  tools/RayTracer.h
  tools/SignedFloodFill.h
  tools/Statistics.h
  tools/TopologyToLevelSet.h
  tools/ValueTransformer.h
  tools/VectorTransformer.h
  tools/VelocityFields.h
  tools/VolumeAdvect.h
  tools/VolumeToMesh.h
  tools/VolumeToSpheres.h
)

set(OPENVDB_LIBRARY_TREE_INCLUDE_FILES
  tree/InternalNode.h
  tree/Iterator.h
  tree/LeafBuffer.h
  tree/LeafManager.h
  tree/LeafNode.h
  tree/LeafNodeBool.h
  tree/LeafNodeMask.h
  tree/NodeManager.h
  tree/NodeUnion.h
  tree/RootNode.h
  tree/Tree.h
  tree/TreeIterator.h
  tree/ValueAccessor.h
)

set(OPENVDB_LIBRARY_UTIL_INCLUDE_FILES
  util/CpuTimer.h
  util/Formats.h
  util/logging.h
  util/MapsUtil.h
  util/Name.h
  util/NodeMasks.h
  util/NullInterrupter.h
  util/PagedArray.h
  util/Util.h
)

# @todo CMake >= 3.12, use an object library to consolidate shared/static
# builds. There are limitations with earlier versions of CMake when used with
# imported targets.

if(OPENVDB_CORE_SHARED)
  add_library(openvdb_shared SHARED ${OPENVDB_LIBRARY_SOURCE_FILES})
endif()

if(OPENVDB_CORE_STATIC)
  add_library(openvdb_static STATIC ${OPENVDB_LIBRARY_SOURCE_FILES})
endif()

# Alias either the shared or static library to the generic OpenVDB
# target. Dependent components should use this target to build against
# such that they are always able to find a valid build of OpenVDB

if(OPENVDB_CORE_SHARED)
  add_library(openvdb ALIAS openvdb_shared)
else()
  add_library(openvdb ALIAS openvdb_static)
endif()


##########################################################################

# Configure (static and shared lib) C flags

# Private defines (not inherited by dependent projects)
list(APPEND OPENVDB_CORE_PRIVATE_DEFINES -DOPENVDB_PRIVATE)

if(WIN32)
  if(USE_IMATH_HALF)
    # @note OPENVDB_OPENEXR_STATICLIB is old functionality and should be removed
    if(ILMBASE_USE_STATIC_LIBS OR (${ILMBASE_Half_LIB_TYPE} STREQUAL STATIC_LIBRARY))
      list(APPEND OPENVDB_CORE_PUBLIC_DEFINES -DOPENVDB_OPENEXR_STATICLIB)
    endif()
  endif()
endif()

if(USE_LOG4CPLUS)
  list(APPEND OPENVDB_CORE_PUBLIC_DEFINES -DOPENVDB_USE_LOG4CPLUS)
endif()

##########################################################################

# Configure static build

if(OPENVDB_CORE_STATIC)
  target_include_directories(openvdb_static
    PUBLIC ../ ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/openvdb
    PRIVATE .
  )

  target_compile_definitions(openvdb_static
    PUBLIC ${OPENVDB_CORE_PUBLIC_DEFINES} -DOPENVDB_STATICLIB
    PRIVATE ${OPENVDB_CORE_PRIVATE_DEFINES}
  )

  target_link_libraries(openvdb_static
    PUBLIC ${OPENVDB_CORE_DEPENDENT_LIBS}
  )

  set_target_properties(openvdb_static
    PROPERTIES OUTPUT_NAME ${OPENVDB_STATIC_LIBRARY_NAME})

  if(WIN32)
    if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
      # NOTE: MSVC_RUNTIME_LIBRARY does not propagate to targets, so
      # also add it explicitly as a compile option
      set_target_properties(openvdb_static PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
      target_compile_options(openvdb_static PUBLIC "/MT$<$<CONFIG:Debug>:d>")
    endif()
    set_target_properties(openvdb_static PROPERTIES PREFIX "lib")
  endif()
endif()

# Configure shared build

if(OPENVDB_CORE_SHARED)
  target_include_directories(openvdb_shared
    PUBLIC ../ ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/openvdb
    PRIVATE .
  )

  target_compile_definitions(openvdb_shared
    PUBLIC ${OPENVDB_CORE_PUBLIC_DEFINES} -DOPENVDB_DLL
    PRIVATE ${OPENVDB_CORE_PRIVATE_DEFINES}
  )

  target_link_libraries(openvdb_shared
    PUBLIC ${OPENVDB_CORE_DEPENDENT_LIBS}
  )

  set_target_properties(openvdb_shared
    PROPERTIES
      OUTPUT_NAME ${OPENVDB_SHARED_LIBRARY_NAME}
      SOVERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}
      VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}
  )

  if(OPENVDB_ENABLE_RPATH)
    # @todo There is probably a better way to do this for imported targets
    list(APPEND RPATHS
      ${Boost_LIBRARY_DIRS}
      ${IlmBase_LIBRARY_DIRS}
      ${Log4cplus_LIBRARY_DIRS}
      ${Blosc_LIBRARY_DIRS}
      ${Tbb_LIBRARY_DIRS}
    )
    list(REMOVE_DUPLICATES RPATHS)
    set_target_properties(openvdb_shared
      PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  endif()

  if(WIN32)
    if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
      # NOTE: MSVC_RUNTIME_LIBRARY does not propagate to targets, so
      # also add it explicitly as a compile option
      set_target_properties(openvdb_shared PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
      target_compile_options(openvdb_shared PUBLIC "/MD$<$<CONFIG:Debug>:d>")
    endif()
  endif()
endif()

##########################################################################

# Installation

if(OPENVDB_CORE_STATIC)
  install(TARGETS openvdb_static
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

if(OPENVDB_CORE_SHARED)
  install(TARGETS openvdb_shared
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

install(FILES ${OPENVDB_LIBRARY_INCLUDE_FILES} DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openvdb/version.h DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR})
install(FILES ${OPENVDB_LIBRARY_IO_INCLUDE_FILES} DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR}/io)
install(FILES ${OPENVDB_LIBRARY_MATH_INCLUDE_FILES} DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR}/math)
install(FILES ${OPENVDB_LIBRARY_POINTS_INCLUDE_FILES} DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR}/points)
install(FILES ${OPENVDB_LIBRARY_TOOLS_INCLUDE_FILES} DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR}/tools)
install(FILES ${OPENVDB_LIBRARY_TREE_INCLUDE_FILES} DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR}/tree)
install(FILES ${OPENVDB_LIBRARY_UTIL_INCLUDE_FILES} DESTINATION ${OPENVDB_INSTALL_INCLUDEDIR}/util)
