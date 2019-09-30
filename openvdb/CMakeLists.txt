# Copyright (c) DreamWorks Animation LLC
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

  CMake Configuration for OpenVDB Core

#]=======================================================================]

project(OpenVDBCore)
cmake_minimum_required(VERSION 3.3)
# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

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
    "At least one must be enabled when building the core library."
  )
endif()

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "------------- Configuring OpenVDBCore --------------")
message(STATUS "----------------------------------------------------")

##########################################################################

# Collect and configure lib dependencies

if(USE_EXR)
  find_package(IlmBase ${MINIMUM_ILMBASE_VERSION} REQUIRED)
  find_package(OpenEXR ${MINIMUM_OPENEXR_VERSION} REQUIRED)
else()
  find_package(IlmBase ${MINIMUM_ILMBASE_VERSION} REQUIRED COMPONENTS Half)
endif()

find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbb)
if(${Tbb_VERSION} VERSION_LESS FUTURE_MINIMUM_TBB_VERSION)
  message(DEPRECATION "Support for TBB versions < ${FUTURE_MINIMUM_TBB_VERSION} "
    "is deprecated and will be removed.")
endif()

find_package(ZLIB ${MINIMUM_ZLIB_VERSION} REQUIRED)

if(USE_LOG4CPLUS)
  find_package(Log4cplus ${MINIMUM_LOG4CPLUS_VERSION} REQUIRED)
endif()

if(USE_BLOSC)
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

find_package(Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams system)

if(UNIX)
  find_package(Threads REQUIRED)
endif()

if(CONCURRENT_MALLOC STREQUAL "Jemalloc")
  find_package(Jemalloc)
  if(NOT TARGET Jemalloc::jemalloc)
    message(WARNING "Unable to find Jemalloc, attempting to fall back to TBB malloc.
      It is recommended to use Jemalloc for optimum performance."
    )
    find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbbmalloc tbbmalloc_proxy)
  endif()
elseif(CONCURRENT_MALLOC STREQUAL "Tbbmalloc")
  find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbbmalloc tbbmalloc_proxy)
endif()

# Set deps. Note that the order here is important. If we're building against
# Houdini 17.5 we must include OpenEXR and IlmBase deps first to ensure the
# users chosen namespaced headers are correctly prioritized. Otherwise other
# include paths from shared installs (including houdini) may pull in the wrong
# headers

set(OPENVDB_CORE_DEPENDENT_LIBS
  Boost::iostreams
  Boost::system
  IlmBase::Half
)

if(USE_EXR)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS
    IlmBase::IlmThread
    IlmBase::Iex
    IlmBase::Imath
    OpenEXR::IlmImf
  )
endif()

if(USE_LOG4CPLUS)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Log4cplus::log4cplus)
endif()

# @todo blosc and zlib should be hidden (privately linked in):
# See FindOpenVDB.cmake

if(USE_BLOSC)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Blosc::blosc)
endif()

list(APPEND OPENVDB_CORE_DEPENDENT_LIBS
  TBB::tbb
  ZLIB::ZLIB
)

if(UNIX)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Threads::Threads)
endif()

if(TARGET Jemalloc::jemalloc)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS Jemalloc::jemalloc)
endif()

if(TARGET TBB::tbbmalloc AND TARGET TBB::tbbmalloc_proxy)
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
  )
endif()

##########################################################################

if(WIN32)
  # Because of implicit linking!
  link_directories(${Boost_LIBRARY_DIR})
  if(OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING)
    add_definitions(-DBOOST_ALL_NO_LIB)
  endif()
endif()

# @todo Should be target definitions
if(WIN32)
  add_definitions(-D_WIN32 -DNOMINMAX -DOPENVDB_DLL)
endif()

##### Core library configuration

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
  version.h
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
  tools/ChangeBackground.h
  tools/Clip.h
  tools/Composite.h
  tools/Dense.h
  tools/DenseSparseTools.h
  tools/Diagnostics.h
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
  tools/MeshToVolume.h
  tools/Morphology.h
  tools/MultiResGrid.h
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

# Configure c flags

set(OPENVDB_CORE_PRIVATE_DEFINES)
set(OPENVDB_CORE_PUBLIC_DEFINES)

# Private defines (not inherited by dependent projects)

list(APPEND OPENVDB_CORE_PRIVATE_DEFINES "-DOPENVDB_PRIVATE")
if(USE_BLOSC)
  list(APPEND OPENVDB_CORE_PRIVATE_DEFINES "-DOPENVDB_USE_BLOSC")
endif()

# Public defines

if(USE_EXR)
  list(APPEND OPENVDB_CORE_PUBLIC_DEFINES "-DOPENVDB_TOOLS_RAYTRACER_USE_EXR")
endif()
if(USE_LOG4CPLUS)
  list(APPEND OPENVDB_CORE_PUBLIC_DEFINES "-DOPENVDB_USE_LOG4CPLUS")
endif()

##########################################################################

# Configure static build

if(OPENVDB_CORE_STATIC)
  target_include_directories(openvdb_static
    PUBLIC ../
    PRIVATE .
  )

  target_compile_definitions(openvdb_static PRIVATE ${OPENVDB_CORE_PRIVATE_DEFINES})
  target_compile_definitions(openvdb_static PUBLIC ${OPENVDB_CORE_PUBLIC_DEFINES})

  set_target_properties(openvdb_static
    PROPERTIES OUTPUT_NAME ${OPENVDB_STATIC_LIBRARY_NAME}
  )

  if(WIN32)
    set_target_properties(openvdb_static
      PROPERTIES PREFIX "lib"
    )
  endif()

  target_link_libraries(openvdb_static
    ${OPENVDB_CORE_DEPENDENT_LIBS}
  )
endif()

# Configure shared build

if(OPENVDB_CORE_SHARED)
  target_include_directories(openvdb_shared
    PUBLIC ../
    PRIVATE .
  )

  target_compile_definitions(openvdb_shared PRIVATE ${OPENVDB_CORE_PRIVATE_DEFINES})
  target_compile_definitions(openvdb_shared PUBLIC ${OPENVDB_CORE_PUBLIC_DEFINES})

  set_target_properties(openvdb_shared
    PROPERTIES
      OUTPUT_NAME ${OPENVDB_SHARED_LIBRARY_NAME}
      SOVERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}
      VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}
  )

  target_link_libraries(openvdb_shared
    ${OPENVDB_CORE_DEPENDENT_LIBS}
  )

  if(OPENVDB_ENABLE_RPATH)
    # @todo There is probably a better way to do this for imported targets
    list(APPEND RPATHS
      ${Boost_LIBRARY_DIRS}
      ${IlmBase_LIBRARY_DIRS}
      ${OpenEXR_LIBRARY_DIRS}
      ${Log4cplus_LIBRARY_DIRS}
      ${Blosc_LIBRARY_DIRS}
      ${Tbb_LIBRARY_DIRS}
    )
    list(REMOVE_DUPLICATES RPATHS)
    set_target_properties(openvdb_shared
      PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  endif()
endif()

unset(OPENVDB_CORE_PRIVATE_DEFINES)
unset(OPENVDB_CORE_PUBLIC_DEFINES)

##########################################################################

# Installation

if(OPENVDB_CORE_STATIC)
  install(TARGETS
    openvdb_static
    DESTINATION
    lib
  )
endif()

if(OPENVDB_CORE_SHARED)
  if(WIN32)
    install(TARGETS
      openvdb_shared
      RUNTIME DESTINATION bin COMPONENT Runtime
      ARCHIVE DESTINATION lib COMPONENT Runtime
    )
  else()
    install(TARGETS
      openvdb_shared
      DESTINATION
      lib
    )
  endif()
endif()

install(FILES ${OPENVDB_LIBRARY_INCLUDE_FILES} DESTINATION include/openvdb)
install(FILES ${OPENVDB_LIBRARY_IO_INCLUDE_FILES} DESTINATION include/openvdb/io)
install(FILES ${OPENVDB_LIBRARY_MATH_INCLUDE_FILES} DESTINATION include/openvdb/math)
install(FILES ${OPENVDB_LIBRARY_POINTS_INCLUDE_FILES} DESTINATION include/openvdb/points)
install(FILES ${OPENVDB_LIBRARY_TOOLS_INCLUDE_FILES} DESTINATION include/openvdb/tools)
install(FILES ${OPENVDB_LIBRARY_TREE_INCLUDE_FILES} DESTINATION include/openvdb/tree)
install(FILES ${OPENVDB_LIBRARY_UTIL_INCLUDE_FILES} DESTINATION include/openvdb/util)
