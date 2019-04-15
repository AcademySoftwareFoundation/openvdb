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

  CMake Configuration for OpenVDB Core

#]=======================================================================]

PROJECT ( OpenVDBCore )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
# Monitoring <PackageName>_ROOT variables
IF ( POLICY CMP0074 )
  CMAKE_POLICY ( SET CMP0074 NEW )
ENDIF ()

###### OpenVDB Core Options

SET ( OPENVDB_LIBRARY_NAME "openvdb" CACHE STRING "The name of the openvdb library to build." )

MARK_AS_ADVANCED (
  OPENVDB_LIBRARY_NAME
  )

#########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "------------- Configuring OpenVDBCore --------------" )
MESSAGE ( STATUS "----------------------------------------------------" )

##########################################################################

# Collect and configure lib dependencies

IF ( USE_EXR )
  FIND_PACKAGE ( IlmBase REQUIRED )
  FIND_PACKAGE ( OpenEXR REQUIRED )
ELSE ()
  FIND_PACKAGE ( IlmBase REQUIRED COMPONENTS Half )
ENDIF ()

FIND_PACKAGE ( TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbb )
FIND_PACKAGE ( ZLIB REQUIRED )

IF ( USE_LOG4CPLUS )
  FIND_PACKAGE ( Log4cplus ${MINIMUM_LOG4CPLUS_VERSION} REQUIRED )
ENDIF ()

IF ( USE_BLOSC )
  FIND_PACKAGE ( Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED )
  IF ( Blosc_FOUND AND Blosc_VERSION VERSION_GREATER ${MINIMUM_BLOSC_VERSION} )
    MESSAGE ( WARNING "The version of Blosc located is greater than ${MINIMUM_BLOSC_VERSION}. "
      "There have been reported issues with using later versions of Blosc with OpenVDB. "
      "OpenVDB has been tested fully against Blosc ${MINIMUM_BLOSC_VERSION}, it is "
      "recommended that you use this version where possible."
      )
  ENDIF ()
ELSE ()
  MESSAGE ( STATUS "Blosc is currently disabled - it's recommended to install and "
    "enable blosc for optimal builds of OpenVDB."
    )
ENDIF ()

FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams system )

IF ( UNIX )
  FIND_PACKAGE ( Threads REQUIRED )
ENDIF ()

# Set deps. Note that the order here is important. If we're building against
# Houdini 17.5 we must include OpenEXR and IlmBase deps first to ensure the
# users chosen namespaced headers are correctly prioritized. Otherwise other
# include paths from shared installs (including houdini) may pull in the wrong
# headers

SET ( OPENVDB_CORE_DEPENDENT_LIBS
  Boost::iostreams
  Boost::system
  IlmBase::Half
  )

IF ( USE_EXR )
  LIST ( APPEND OPENVDB_CORE_DEPENDENT_LIBS
    IlmBase::IlmThread
    IlmBase::Iex
    IlmBase::Imath
    OpenEXR::IlmImf
    )
ENDIF ()

IF ( USE_LOG4CPLUS )
  LIST ( APPEND OPENVDB_CORE_DEPENDENT_LIBS Log4cplus::log4cplus )
ENDIF ()

IF ( USE_BLOSC )
  LIST ( APPEND OPENVDB_CORE_DEPENDENT_LIBS Blosc::blosc )
ENDIF ()

LIST ( APPEND OPENVDB_CORE_DEPENDENT_LIBS
  TBB::tbb
  ZLIB::ZLIB
  Threads::Threads
)

##########################################################################

IF ( WIN32 )
  # Because of implicit linking!
  LINK_DIRECTORIES ( ${Boost_LIBRARY_DIR} )
  IF ( OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING )
    ADD_DEFINITIONS ( -DBOOST_ALL_NO_LIB )
  ENDIF ()
ENDIF ()

# @todo Should be target definitions
IF ( WIN32 )
  ADD_DEFINITIONS ( -D_WIN32 -DNOMINMAX -DHALF_EXPORTS -DOPENVDB_DLL )
ENDIF ()

##### Core library configuration

SET ( OPENVDB_LIBRARY_SOURCE_FILES
  Grid.cc
  MetaMap.cc
  Metadata.cc
  Platform.cc
  io/Archive.cc
  io/Compression.cc
  io/File.cc
  io/GridDescriptor.cc
  io/Queue.cc
  io/Stream.cc
  io/TempFile.cc
  math/Maps.cc
  math/Proximity.cc
  math/QuantizedUnitVec.cc
  math/Transform.cc
  points/AttributeArray.cc
  points/AttributeArrayString.cc
  points/AttributeGroup.cc
  points/AttributeSet.cc
  points/StreamCompression.cc
  points/points.cc
  openvdb.cc
  util/Formats.cc
  util/Util.cc
  )

SET ( OPENVDB_LIBRARY_INCLUDE_FILES
  Exceptions.h
  Grid.h
  Types.h
  version.h
  Metadata.h
  MetaMap.h
  openvdb.h
  Platform.h
  PlatformConfig.h
)

SET ( OPENVDB_LIBRARY_IO_INCLUDE_FILES
  io/Archive.h
  io/Compression.h
  io/File.h
  io/GridDescriptor.h
  io/io.h
  io/Queue.h
  io/Stream.h
  io/TempFile.h
)

SET ( OPENVDB_LIBRARY_MATH_INCLUDE_FILES
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

SET ( OPENVDB_LIBRARY_POINTS_INCLUDE_FILES
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

SET ( OPENVDB_LIBRARY_TOOLS_INCLUDE_FILES
  tools/ChangeBackground.h
  tools/Clip.h
  tools/Composite.h
  tools/Dense.h
  tools/DenseSparseTools.h
  tools/Diagnostics.h
  tools/Filter.h
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

SET ( OPENVDB_LIBRARY_TREE_INCLUDE_FILES
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

SET ( OPENVDB_LIBRARY_UTIL_INCLUDE_FILES
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

ADD_LIBRARY ( openvdb_static STATIC
  ${OPENVDB_LIBRARY_SOURCE_FILES}
  )
ADD_LIBRARY ( openvdb_shared SHARED
  ${OPENVDB_LIBRARY_SOURCE_FILES}
  )

TARGET_INCLUDE_DIRECTORIES ( openvdb_static
  PUBLIC ../
  PRIVATE .
  )

TARGET_INCLUDE_DIRECTORIES ( openvdb_shared
  PUBLIC ../
  PRIVATE .
  )

##########################################################################

# Configure c flags

TARGET_COMPILE_DEFINITIONS ( openvdb_static PRIVATE "-DOPENVDB_PRIVATE" )
TARGET_COMPILE_DEFINITIONS ( openvdb_shared PRIVATE "-DOPENVDB_PRIVATE" )

IF ( USE_EXR )
  TARGET_COMPILE_DEFINITIONS ( openvdb_static PUBLIC "-DOPENVDB_TOOLS_RAYTRACER_USE_EXR" )
  TARGET_COMPILE_DEFINITIONS ( openvdb_shared PUBLIC "-DOPENVDB_TOOLS_RAYTRACER_USE_EXR" )
ENDIF ()

IF ( USE_BLOSC )
  TARGET_COMPILE_DEFINITIONS ( openvdb_static PRIVATE "-DOPENVDB_USE_BLOSC" )
  TARGET_COMPILE_DEFINITIONS ( openvdb_shared PRIVATE "-DOPENVDB_USE_BLOSC" )
ENDIF ()

IF ( USE_LOG4CPLUS )
  TARGET_COMPILE_DEFINITIONS ( openvdb_static PUBLIC "-DOPENVDB_USE_LOG4CPLUS" )
  TARGET_COMPILE_DEFINITIONS ( openvdb_shared PUBLIC "-DOPENVDB_USE_LOG4CPLUS" )
ENDIF ()

##########################################################################

TARGET_LINK_LIBRARIES ( openvdb_static
  ${OPENVDB_CORE_DEPENDENT_LIBS}
  )

TARGET_LINK_LIBRARIES ( openvdb_shared
  ${OPENVDB_CORE_DEPENDENT_LIBS}
  )

IF ( WIN32 )
  SET_TARGET_PROPERTIES ( openvdb_static
    PROPERTIES OUTPUT_NAME libopenvdb
    )
ELSE ()
  SET_TARGET_PROPERTIES ( openvdb_static
    PROPERTIES OUTPUT_NAME openvdb
    )
ENDIF ()

SET_TARGET_PROPERTIES ( openvdb_shared
  PROPERTIES
    OUTPUT_NAME ${OPENVDB_LIBRARY_NAME}
    SOVERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}
    VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}
  )

# Installation

INSTALL ( TARGETS
  openvdb_static
  DESTINATION
  lib
  )

IF ( WIN32 )
  INSTALL ( TARGETS
    openvdb_shared
    RUNTIME DESTINATION bin COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Runtime
    )
ELSE ()
  INSTALL ( TARGETS
    openvdb_shared
    DESTINATION
    lib
    )
ENDIF ()

IF ( OPENVDB_ENABLE_RPATH )
  # @todo There is probably a better way to do this for imported targets
  LIST ( APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${OpenEXR_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
  )
  LIST ( REMOVE_DUPLICATES RPATHS )
  SET_TARGET_PROPERTIES ( openvdb_shared
    PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
ENDIF ()


INSTALL ( FILES ${OPENVDB_LIBRARY_INCLUDE_FILES} DESTINATION include/openvdb )
INSTALL ( FILES ${OPENVDB_LIBRARY_IO_INCLUDE_FILES} DESTINATION include/openvdb/io )
INSTALL ( FILES ${OPENVDB_LIBRARY_MATH_INCLUDE_FILES} DESTINATION include/openvdb/math )
INSTALL ( FILES ${OPENVDB_LIBRARY_POINTS_INCLUDE_FILES} DESTINATION include/openvdb/points )
INSTALL ( FILES ${OPENVDB_LIBRARY_TOOLS_INCLUDE_FILES} DESTINATION include/openvdb/tools )
INSTALL ( FILES ${OPENVDB_LIBRARY_TREE_INCLUDE_FILES} DESTINATION include/openvdb/tree )
INSTALL ( FILES ${OPENVDB_LIBRARY_UTIL_INCLUDE_FILES} DESTINATION include/openvdb/util )
