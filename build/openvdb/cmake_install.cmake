# Install script for directory: /home/piyush/openvdb/openvdb

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/piyush/openvdb/build/openvdb/libopenvdb.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopenvdb.so.7.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopenvdb.so.7.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopenvdb.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/usr/lib/x86_64-linux-gnu:/usr/local/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/piyush/openvdb/build/openvdb/libopenvdb.so.7.1.0"
    "/home/piyush/openvdb/build/openvdb/libopenvdb.so.7.1"
    "/home/piyush/openvdb/build/openvdb/libopenvdb.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopenvdb.so.7.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopenvdb.so.7.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopenvdb.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/usr/local/lib::::::::::::::::::::::::::"
           NEW_RPATH "/usr/lib/x86_64-linux-gnu:/usr/local/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openvdb" TYPE FILE FILES
    "/home/piyush/openvdb/openvdb/Exceptions.h"
    "/home/piyush/openvdb/openvdb/Grid.h"
    "/home/piyush/openvdb/openvdb/Metadata.h"
    "/home/piyush/openvdb/openvdb/MetaMap.h"
    "/home/piyush/openvdb/openvdb/openvdb.h"
    "/home/piyush/openvdb/openvdb/Platform.h"
    "/home/piyush/openvdb/openvdb/PlatformConfig.h"
    "/home/piyush/openvdb/openvdb/Types.h"
    "/home/piyush/openvdb/openvdb/version.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openvdb/io" TYPE FILE FILES
    "/home/piyush/openvdb/openvdb/io/Archive.h"
    "/home/piyush/openvdb/openvdb/io/Compression.h"
    "/home/piyush/openvdb/openvdb/io/DelayedLoadMetadata.h"
    "/home/piyush/openvdb/openvdb/io/File.h"
    "/home/piyush/openvdb/openvdb/io/GridDescriptor.h"
    "/home/piyush/openvdb/openvdb/io/io.h"
    "/home/piyush/openvdb/openvdb/io/Queue.h"
    "/home/piyush/openvdb/openvdb/io/Stream.h"
    "/home/piyush/openvdb/openvdb/io/TempFile.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openvdb/math" TYPE FILE FILES
    "/home/piyush/openvdb/openvdb/math/BBox.h"
    "/home/piyush/openvdb/openvdb/math/ConjGradient.h"
    "/home/piyush/openvdb/openvdb/math/Coord.h"
    "/home/piyush/openvdb/openvdb/math/DDA.h"
    "/home/piyush/openvdb/openvdb/math/FiniteDifference.h"
    "/home/piyush/openvdb/openvdb/math/LegacyFrustum.h"
    "/home/piyush/openvdb/openvdb/math/Maps.h"
    "/home/piyush/openvdb/openvdb/math/Mat.h"
    "/home/piyush/openvdb/openvdb/math/Mat3.h"
    "/home/piyush/openvdb/openvdb/math/Mat4.h"
    "/home/piyush/openvdb/openvdb/math/Math.h"
    "/home/piyush/openvdb/openvdb/math/Operators.h"
    "/home/piyush/openvdb/openvdb/math/Proximity.h"
    "/home/piyush/openvdb/openvdb/math/QuantizedUnitVec.h"
    "/home/piyush/openvdb/openvdb/math/Quat.h"
    "/home/piyush/openvdb/openvdb/math/Ray.h"
    "/home/piyush/openvdb/openvdb/math/Stats.h"
    "/home/piyush/openvdb/openvdb/math/Stencils.h"
    "/home/piyush/openvdb/openvdb/math/Transform.h"
    "/home/piyush/openvdb/openvdb/math/Tuple.h"
    "/home/piyush/openvdb/openvdb/math/Vec2.h"
    "/home/piyush/openvdb/openvdb/math/Vec3.h"
    "/home/piyush/openvdb/openvdb/math/Vec4.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openvdb/points" TYPE FILE FILES
    "/home/piyush/openvdb/openvdb/points/AttributeArray.h"
    "/home/piyush/openvdb/openvdb/points/AttributeArrayString.h"
    "/home/piyush/openvdb/openvdb/points/AttributeGroup.h"
    "/home/piyush/openvdb/openvdb/points/AttributeSet.h"
    "/home/piyush/openvdb/openvdb/points/IndexFilter.h"
    "/home/piyush/openvdb/openvdb/points/IndexIterator.h"
    "/home/piyush/openvdb/openvdb/points/PointAdvect.h"
    "/home/piyush/openvdb/openvdb/points/PointAttribute.h"
    "/home/piyush/openvdb/openvdb/points/PointConversion.h"
    "/home/piyush/openvdb/openvdb/points/PointCount.h"
    "/home/piyush/openvdb/openvdb/points/PointDataGrid.h"
    "/home/piyush/openvdb/openvdb/points/PointDelete.h"
    "/home/piyush/openvdb/openvdb/points/PointGroup.h"
    "/home/piyush/openvdb/openvdb/points/PointMask.h"
    "/home/piyush/openvdb/openvdb/points/PointMove.h"
    "/home/piyush/openvdb/openvdb/points/PointSample.h"
    "/home/piyush/openvdb/openvdb/points/PointScatter.h"
    "/home/piyush/openvdb/openvdb/points/StreamCompression.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openvdb/tools" TYPE FILE FILES
    "/home/piyush/openvdb/openvdb/tools/ChangeBackground.h"
    "/home/piyush/openvdb/openvdb/tools/Clip.h"
    "/home/piyush/openvdb/openvdb/tools/Composite.h"
    "/home/piyush/openvdb/openvdb/tools/Dense.h"
    "/home/piyush/openvdb/openvdb/tools/DenseSparseTools.h"
    "/home/piyush/openvdb/openvdb/tools/Diagnostics.h"
    "/home/piyush/openvdb/openvdb/tools/Filter.h"
    "/home/piyush/openvdb/openvdb/tools/FindActiveValues.h"
    "/home/piyush/openvdb/openvdb/tools/GridOperators.h"
    "/home/piyush/openvdb/openvdb/tools/GridTransformer.h"
    "/home/piyush/openvdb/openvdb/tools/Interpolation.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetAdvect.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetFilter.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetFracture.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetMeasure.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetMorph.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetPlatonic.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetRebuild.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetSphere.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetTracker.h"
    "/home/piyush/openvdb/openvdb/tools/LevelSetUtil.h"
    "/home/piyush/openvdb/openvdb/tools/Mask.h"
    "/home/piyush/openvdb/openvdb/tools/MeshToVolume.h"
    "/home/piyush/openvdb/openvdb/tools/Morphology.h"
    "/home/piyush/openvdb/openvdb/tools/MultiResGrid.h"
    "/home/piyush/openvdb/openvdb/tools/ParticleAtlas.h"
    "/home/piyush/openvdb/openvdb/tools/ParticlesToLevelSet.h"
    "/home/piyush/openvdb/openvdb/tools/PointAdvect.h"
    "/home/piyush/openvdb/openvdb/tools/PointIndexGrid.h"
    "/home/piyush/openvdb/openvdb/tools/PointPartitioner.h"
    "/home/piyush/openvdb/openvdb/tools/PointScatter.h"
    "/home/piyush/openvdb/openvdb/tools/PointsToMask.h"
    "/home/piyush/openvdb/openvdb/tools/PoissonSolver.h"
    "/home/piyush/openvdb/openvdb/tools/PotentialFlow.h"
    "/home/piyush/openvdb/openvdb/tools/Prune.h"
    "/home/piyush/openvdb/openvdb/tools/RayIntersector.h"
    "/home/piyush/openvdb/openvdb/tools/RayTracer.h"
    "/home/piyush/openvdb/openvdb/tools/SignedFloodFill.h"
    "/home/piyush/openvdb/openvdb/tools/Statistics.h"
    "/home/piyush/openvdb/openvdb/tools/TopologyToLevelSet.h"
    "/home/piyush/openvdb/openvdb/tools/ValueTransformer.h"
    "/home/piyush/openvdb/openvdb/tools/VectorTransformer.h"
    "/home/piyush/openvdb/openvdb/tools/VelocityFields.h"
    "/home/piyush/openvdb/openvdb/tools/VolumeAdvect.h"
    "/home/piyush/openvdb/openvdb/tools/VolumeToMesh.h"
    "/home/piyush/openvdb/openvdb/tools/VolumeToSpheres.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openvdb/tree" TYPE FILE FILES
    "/home/piyush/openvdb/openvdb/tree/InternalNode.h"
    "/home/piyush/openvdb/openvdb/tree/Iterator.h"
    "/home/piyush/openvdb/openvdb/tree/LeafBuffer.h"
    "/home/piyush/openvdb/openvdb/tree/LeafManager.h"
    "/home/piyush/openvdb/openvdb/tree/LeafNode.h"
    "/home/piyush/openvdb/openvdb/tree/LeafNodeBool.h"
    "/home/piyush/openvdb/openvdb/tree/LeafNodeMask.h"
    "/home/piyush/openvdb/openvdb/tree/NodeManager.h"
    "/home/piyush/openvdb/openvdb/tree/NodeUnion.h"
    "/home/piyush/openvdb/openvdb/tree/RootNode.h"
    "/home/piyush/openvdb/openvdb/tree/Tree.h"
    "/home/piyush/openvdb/openvdb/tree/TreeIterator.h"
    "/home/piyush/openvdb/openvdb/tree/ValueAccessor.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openvdb/util" TYPE FILE FILES
    "/home/piyush/openvdb/openvdb/util/CpuTimer.h"
    "/home/piyush/openvdb/openvdb/util/Formats.h"
    "/home/piyush/openvdb/openvdb/util/logging.h"
    "/home/piyush/openvdb/openvdb/util/MapsUtil.h"
    "/home/piyush/openvdb/openvdb/util/Name.h"
    "/home/piyush/openvdb/openvdb/util/NodeMasks.h"
    "/home/piyush/openvdb/openvdb/util/NullInterrupter.h"
    "/home/piyush/openvdb/openvdb/util/PagedArray.h"
    "/home/piyush/openvdb/openvdb/util/Util.h"
    )
endif()

