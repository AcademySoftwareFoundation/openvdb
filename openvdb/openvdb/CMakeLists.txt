# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Core

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(OpenVDBCore LANGUAGES CXX)

include(GNUInstallDirs)

###### OpenVDB Core Options

option(OPENVDB_CORE_SHARED "Build dynamically linked version of the core library." ON)
option(OPENVDB_CORE_STATIC "Build statically linked version of the core library." ON)

if(NOT OPENVDB_NAMESPACE_SUFFIX)
  set(OPENVDB_NAMESPACE_SUFFIX "" CACHE STRING
    "An optional custom string to append to the OpenVDB namespace. This is used to configure version.h.")
endif()

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

if(OPENVDB_USE_DELAYED_LOADING)
  find_package(Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams)
else()
  find_package(Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS headers)
endif()

if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_BOOST_VERSION)
  # The X.Y.Z boost version value isn't available until CMake 3.14
  set(FULL_BOOST_VERSION "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
  if(${FULL_BOOST_VERSION} VERSION_LESS FUTURE_MINIMUM_BOOST_VERSION)
    message(DEPRECATION "Support for Boost versions < ${FUTURE_MINIMUM_BOOST_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbb)
if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_TBB_VERSION)
  if(${TBB_VERSION} VERSION_LESS FUTURE_MINIMUM_TBB_VERSION)
    message(DEPRECATION "Support for TBB versions < ${FUTURE_MINIMUM_TBB_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

if(USE_IMATH_HALF)
  find_package(Imath ${MINIMUM_IMATH_VERSION} REQUIRED CONFIG)
  get_target_property(openvdb_imath_includes Imath::Imath INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Found Imath: ${openvdb_imath_includes} (found suitable version \"${Imath_VERSION}\", "
    "minimum required is \"${MINIMUM_IMATH_VERSION}\")")
  unset(openvdb_imath_includes)
endif()

if(USE_LOG4CPLUS)
  # Find Log4CPlus libraries
  find_package(Log4cplus ${MINIMUM_LOG4CPLUS_VERSION} REQUIRED)
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_LOG4CPLUS_VERSION)
    if(${Log4cplus_VERSION} VERSION_LESS FUTURE_MINIMUM_LOG4CPLUS_VERSION)
      message(DEPRECATION "Support for Log4cplus versions < ${FUTURE_MINIMUM_LOG4CPLUS_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
endif()

if(USE_BLOSC)
  # Find Blosc libraries
  find_package(Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED)
  if(Blosc_VERSION)
    if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_BLOSC_VERSION)
      if(${Blosc_VERSION} VERSION_LESS FUTURE_MINIMUM_BLOSC_VERSION)
        message(DEPRECATION "Support for Blosc versions < ${FUTURE_MINIMUM_BLOSC_VERSION} "
          "is deprecated and will be removed.")
      endif()
    endif()
    # Print a warning if using of the blosc versions with regression issues
    # with Blosc 1.5.0 caches.
    if((Blosc_VERSION VERSION_GREATER_EQUAL 1.11.0 AND
        Blosc_VERSION VERSION_LESS 1.14.0) OR
       (Blosc_VERSION VERSION_GREATER_EQUAL 1.16.0 AND
        Blosc_VERSION VERSION_LESS 1.16.2))
      message(WARNING "The following Blosc versions are incompatible with the "
        "recommended builds of OpenVDB: [1.11.0 -> 1.14.0) [1.16.0, 1.16.1]. "
        "Found Blosc version '${Blosc_VERSION}' which falls in this range. We "
        "strongly recommend using the new future minimum version '${FUTURE_MINIMUM_BLOSC_VERSION}'")
    endif()
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
    if(MSVC)
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
# Houdini 17.5 or later we must include IlmBase deps first to ensure the users
# chosen namespaced headers are correctly prioritized. Otherwise other include
# paths from shared installs (including houdini) may pull in the wrong headers

set(OPENVDB_CORE_DEPENDENT_LIBS "")

if(USE_IMATH_HALF)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Imath::Imath)
endif()

# We then choose to pull in TBB. If building aganst Houdini, TBB should
# always be pulled from there (we should always be using the version of TBB
# shipped with Houdini).

list(APPEND OPENVDB_CORE_DEPENDENT_LIBS TBB::tbb)

if(USE_LOG4CPLUS)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Log4cplus::log4cplus)
endif()

# @todo blosc and zlib should be hidden (privately linked in):
# See FindOpenVDB.cmake
# @warning If building against Houdini these should be picked up from the
# Houdini installation. If a custom version of EXR is in use, be careful
# that zlib/blosc headers/libs are not also in the same location e.g.
# /usr/local

if(USE_BLOSC)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Blosc::blosc)
endif()

if(USE_BLOSC OR USE_ZLIB)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS ZLIB::ZLIB)
endif()

if(UNIX)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Threads::Threads)
endif()

# Pull in Boost last as houdini's boost (hboost) is fully namespaced and libs
# are renamed too. Boost can be pulled in at any time, but do it last so that,
# if it's in a shared place (like /usr/local) it doesn't accidently pull in
# other headers.

if(OPENVDB_USE_DELAYED_LOADING)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Boost::iostreams)
else()
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Boost::headers)
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
if(USE_EXPLICIT_INSTANTIATION)
  set(OPENVDB_USE_EXPLICIT_INSTANTIATION 1)
endif()

# Configure VDB version in the form: ("%02x%02x%04x", major, minor, patch)
math(EXPR OPENVDB_PACKED_VERSION "${OpenVDB_MAJOR_VERSION} << 24")
math(EXPR OPENVDB_PACKED_VERSION "${OPENVDB_PACKED_VERSION} | ((${OpenVDB_MINOR_VERSION} & 0xFF) << 16)")
math(EXPR OPENVDB_PACKED_VERSION "${OPENVDB_PACKED_VERSION} | (${OpenVDB_PATCH_VERSION} & 0xFFFF)")

if(Imath_VERSION)
  set(OPENVDB_IMATH_VERSION ${Imath_VERSION})
endif()

# Configure explicit instantiation trees and extern template declarations
if(USE_EXPLICIT_INSTANTIATION)
  # Define the list of OpenVDB trees for which to use explicit instantiation
  # @note The tree definition is expected to be available when including openvdb/openvdb.h

  set(OPENVDB_LIBRARY_INSTANTIATION_REAL_TREES
    FloatTree
    DoubleTree
  )

  set(OPENVDB_LIBRARY_INSTANTIATION_NUMERIC_TREES
    Int32Tree
    Int64Tree
    ${OPENVDB_LIBRARY_INSTANTIATION_REAL_TREES}
  )

  set(OPENVDB_LIBRARY_INSTANTIATION_VEC3_TREES
    Vec3STree
    Vec3DTree
    Vec3ITree
  )

  set(OPENVDB_LIBRARY_INSTANTIATION_VOLUME_TREES
    BoolTree
    ${OPENVDB_LIBRARY_INSTANTIATION_NUMERIC_TREES}
    ${OPENVDB_LIBRARY_INSTANTIATION_VEC3_TREES}
  )

  set(OPENVDB_LIBRARY_INSTANTIATION_ALL_TREES
    MaskTree
    points::PointDataTree
    ${OPENVDB_LIBRARY_INSTANTIATION_VOLUME_TREES}
  )

  # Write all the template instantiations into strings to inject into version.h
  # The "Function" itself is an argument provided by the OPENVDB_*_TREE_INSTANTIATE macros

  set(TEMPLATE_STR " \\\n    OPENVDB_INSTANTIATE Function")
  foreach(TREE ${OPENVDB_LIBRARY_INSTANTIATION_ALL_TREES})
    if(${TREE} IN_LIST OPENVDB_LIBRARY_INSTANTIATION_REAL_TREES)
      string(APPEND OPENVDB_REAL_TREE_INSTANTIATIONS "${TEMPLATE_STR}(${TREE});")
    endif()
    if(${TREE} IN_LIST OPENVDB_LIBRARY_INSTANTIATION_NUMERIC_TREES)
      string(APPEND OPENVDB_NUMERIC_TREE_INSTANTIATIONS "${TEMPLATE_STR}(${TREE});")
    endif()
    if(${TREE} IN_LIST OPENVDB_LIBRARY_INSTANTIATION_VEC3_TREES)
      string(APPEND OPENVDB_VEC3_TREE_INSTANTIATIONS "${TEMPLATE_STR}(${TREE});")
    endif()
    if(${TREE} IN_LIST OPENVDB_LIBRARY_INSTANTIATION_VOLUME_TREES)
      string(APPEND OPENVDB_VOLUME_TREE_INSTANTIATIONS "${TEMPLATE_STR}(${TREE});")
    endif()
    string(APPEND OPENVDB_ALL_TREE_INSTANTIATIONS "${TEMPLATE_STR}(${TREE});")
  endforeach()
endif()

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
  util/Assert.cc
  util/Formats.cc
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
  points/PointRasterizeFrustum.h
  points/PointRasterizeSDF.h
  points/PointRasterizeTrilinear.h
  points/PointSample.h
  points/PointScatter.h
  points/PointStatistics.h
  points/PointTransfer.h
  points/StreamCompression.h
)

set(OPENVDB_LIBRARY_POINTS_IMPL_INCLUDE_FILES
  points/impl/PointAttributeImpl.h
  points/impl/PointConversionImpl.h
  points/impl/PointCountImpl.h
  points/impl/PointDeleteImpl.h
  points/impl/PointGroupImpl.h
  points/impl/PointMaskImpl.h
  points/impl/PointMoveImpl.h
  points/impl/PointRasterizeFrustumImpl.h
  points/impl/PointRasterizeSDFImpl.h
  points/impl/PointRasterizeTrilinearImpl.h
  points/impl/PointReplicateImpl.h
  points/impl/PointSampleImpl.h
  points/impl/PointScatterImpl.h
  points/impl/PointStatisticsImpl.h
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
  util/Assert.h
  util/CpuTimer.h
  util/ExplicitInstantiation.h
  util/Formats.h
  util/logging.h
  util/MapsUtil.h
  util/Name.h
  util/NodeMasks.h
  util/NullInterrupter.h
  util/PagedArray.h
  util/Util.h
)

set(OPENVDB_LIBRARY_THREAD_INCLUDE_FILES
  thread/Threading.h
)

##########################################################################

# Configure auto-generation of explicit template definition translation units

if(USE_EXPLICIT_INSTANTIATION)
  # Define the list of headers for which to generate new translation units.
  # Not all the tools headers have instantiations currently, but they are
  # inexpensive to generate in this case.
  set(OPENVDB_LIBRARY_INSTANTIATION_HEADERS
    ${OPENVDB_LIBRARY_TOOLS_INCLUDE_FILES}
  )

  # A new source file is generated by CMake for every header
  # into a new instantiations subdir.
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/instantiations)
  foreach(HEADER ${OPENVDB_LIBRARY_INSTANTIATION_HEADERS})
    # String replace to extract the name from the header filepath
    # "tools/Activate.h" => "Activate"
    string(REGEX REPLACE "^.*/(.*).h" "\\1" NAME ${HEADER})

    # Make name upper case "Activate" => "ACTIVATE"
    string(TOUPPER ${NAME} UPPER_NAME)

    # Define the source file location - "<binary_dir>/instantiations/Activate.cc"
    set(INSTANTIATE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/instantiations/${NAME}.cc")
    set(INSTANTIATE_INPUT "${CMAKE_CURRENT_BINARY_DIR}/instantiations.in/${NAME}.cc")

    # The auto-generated file holds the instantiate define and the header
    # which contains all the functions to be explicitly instantiated.
    file(WRITE ${INSTANTIATE_INPUT}
      "#define OPENVDB_INSTANTIATE_${UPPER_NAME}\n"
      "#include <openvdb/${HEADER}>\n"
    )

    # Update the actual source file only if the content has been modified.
    # Without this step, rerunning CMake would update the timestamp of the
    # auto-generated file systematically, retriggering compilation.
    configure_file(${INSTANTIATE_INPUT} ${INSTANTIATE_SOURCE} COPYONLY)

    # Add to the list of library source files
    # (prepend as instantiations tend to be slow to compile).
    list(PREPEND OPENVDB_LIBRARY_SOURCE_FILES ${INSTANTIATE_SOURCE})
  endforeach()
endif()

##########################################################################

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

if(MINGW)
  list(APPEND OPENVDB_CORE_PUBLIC_DEFINES -D_USE_MATH_DEFINES)
endif()

if(USE_LOG4CPLUS)
  list(APPEND OPENVDB_CORE_PUBLIC_DEFINES -DOPENVDB_USE_LOG4CPLUS)
endif()
if(OPENVDB_USE_DELAYED_LOADING)
  list(APPEND OPENVDB_CORE_PUBLIC_DEFINES -DOPENVDB_USE_DELAYED_LOADING)
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

  if(MSVC)
    if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
      # NOTE: MSVC_RUNTIME_LIBRARY does not propagate to targets, so
      # also add it explicitly as a compile option
      set_target_properties(openvdb_static PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
      target_compile_options(openvdb_static PUBLIC
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MT$<$<CONFIG:Debug>:d>>")
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
      VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION})

  if(MSVC)
    if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
      # NOTE: MSVC_RUNTIME_LIBRARY does not propagate to targets, so
      # also add it explicitly as a compile option
      set_target_properties(openvdb_shared PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
      target_compile_options(openvdb_shared PUBLIC
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MD$<$<CONFIG:Debug>:d>>")
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

install(FILES ${OPENVDB_LIBRARY_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openvdb/version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb)
install(FILES ${OPENVDB_LIBRARY_IO_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/io)
install(FILES ${OPENVDB_LIBRARY_MATH_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/math)
install(FILES ${OPENVDB_LIBRARY_POINTS_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/points)
install(FILES ${OPENVDB_LIBRARY_POINTS_IMPL_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/points/impl)
install(FILES ${OPENVDB_LIBRARY_TOOLS_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/tools)
install(FILES ${OPENVDB_LIBRARY_TREE_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/tree)
install(FILES ${OPENVDB_LIBRARY_UTIL_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/util)
install(FILES ${OPENVDB_LIBRARY_THREAD_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/thread)
