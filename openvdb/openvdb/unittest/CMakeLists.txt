# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Unit Tests

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(OpenVDBUnitTests LANGUAGES CXX)

include(GNUInstallDirs)

set(OPENVDB_TESTS "" CACHE STRING [=[
  An optional list of unit tests to build. If not provided, all unit tests will be built.
  Tests should be listed without the Test prefix or the .cc suffix, for example the string
  \"Activate;NodeManager\" would build just the TestActivate.cc and TestNodeManager.cc
  unit tests.]=])

##########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "----------- Configuring OpenVDBUnitTests -----------")
message(STATUS "----------------------------------------------------")

##########################################################################

# Collect lib dependencies

if(NOT OPENVDB_BUILD_CORE)
  # @note  Could also use the openvdb_je target here, but we just opt to
  # handle the value of CONCURRENT_MALLOC outside of this branching for
  # both cases
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

find_package(GTest ${MINIMUM_GOOGLETEST_VERSION} REQUIRED)

set(OPENVDB_TEST_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  GTest::GTest
  GTest::Main
)

if(CONCURRENT_MALLOC STREQUAL "Jemalloc")
  find_package(Jemalloc REQUIRED)
  list(APPEND OPENVDB_TEST_DEPENDENT_LIBS Jemalloc::jemalloc)
elseif(CONCURRENT_MALLOC STREQUAL "Tbbmalloc")
  find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbbmalloc)
  list(APPEND OPENVDB_TEST_DEPENDENT_LIBS TBB::tbbmalloc)
endif()

##########################################################################

##### VDB unit tests

set(UNITTEST_SOURCE_FILES
  main.cc
)

if(OPENVDB_TESTS)
  foreach(test ${OPENVDB_TESTS})
    file(GLOB UNITTEST_GLOB CONFIGURE_DEPENDS Test${test}*.cc)
    list(APPEND UNITTEST_SOURCE_FILES
      ${UNITTEST_GLOB}
    )
  endforeach()
else()
  list(APPEND UNITTEST_SOURCE_FILES
    TestActivate.cc
    TestAttributeArray.cc
    TestAttributeArrayString.cc
    TestAttributeGroup.cc
    TestAttributeSet.cc
    TestBBox.cc
    TestClip.cc
    TestConjGradient.cc
    TestCoord.cc
    TestCount.cc
    TestCpt.cc
    TestCurl.cc
    TestDelayedLoadMetadata.cc
    TestDense.cc
    TestDenseSparseTools.cc
    TestDiagnostics.cc
    TestDivergence.cc
    TestDoubleMetadata.cc
    TestExceptions.cc
    TestFastSweeping.cc
    TestFile.cc
    TestFilter.cc
    TestFindActiveValues.cc
    TestFloatMetadata.cc
    TestGradient.cc
    TestGrid.cc
    TestGridBbox.cc
    TestGridDescriptor.cc
    TestGridIO.cc
    TestGridTransformer.cc
    TestIndexFilter.cc
    TestIndexIterator.cc
    TestInit.cc
    TestInt32Metadata.cc
    TestInt64Metadata.cc
    TestInternalOrigin.cc
    TestLaplacian.cc
    TestLeaf.cc
    TestLeafBool.cc
    TestLeafIO.cc
    TestLeafManager.cc
    TestLeafMask.cc
    TestLeafOrigin.cc
    TestLevelSetRayIntersector.cc
    TestLevelSetUtil.cc
    TestLinearInterp.cc
    TestMaps.cc
    TestMat4Metadata.cc
    TestMath.cc
    TestMeanCurvature.cc
    TestMerge.cc
    TestMeshToVolume.cc
    TestMetadata.cc
    TestMetadataIO.cc
    TestMetaMap.cc
    TestMorphology.cc
    TestMultiResGrid.cc
    TestName.cc
    TestNodeIterator.cc
    TestNodeManager.cc
    TestNodeMask.cc
    TestNodeVisitor.cc
    TestParticleAtlas.cc
    TestParticlesToLevelSet.cc
    TestPointAdvect.cc
    TestPointAttribute.cc
    TestPointConversion.cc
    TestPointCount.cc
    TestPointDataLeaf.cc
    TestPointDelete.cc
    TestPointGroup.cc
    TestPointIndexGrid.cc
    TestPointInstantiate.cc
    TestPointMask.cc
    TestPointMove.cc
    TestPointPartitioner.cc
    TestPointSample.cc
    TestPointScatter.cc
    TestPointsToMask.cc
    TestPoissonSolver.cc
    TestPotentialFlow.cc
    TestPrePostAPI.cc
    TestQuadraticInterp.cc
    TestQuantizedUnitVec.cc
    TestQuat.cc
    TestRay.cc
    TestStats.cc
    TestStream.cc
    TestStreamCompression.cc
    TestStringMetadata.cc
    TestTools.cc
    TestTopologyToLevelSet.cc
    TestTransform.cc
    TestTree.cc
    TestTreeCombine.cc
    TestTreeGetSetValues.cc
    TestTreeIterators.cc
    TestTreeVisitor.cc
    TestTypes.cc
    TestUtil.cc
    TestValueAccessor.cc
    TestVec2Metadata.cc
    TestVec3Metadata.cc
    TestVolumeRayIntersector.cc
    TestVolumeToMesh.cc
    TestVolumeToSpheres.cc
  )
endif()

add_executable(vdb_test ${UNITTEST_SOURCE_FILES})

# Blosc and ZLib are hidden dependencies for the core library
# (not exposed in headers), so we need to manually link them in
# here to provide header access for the relevant unit tests

if(USE_BLOSC OR OpenVDB_USES_BLOSC)
  find_package(Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED)
  list(APPEND OPENVDB_TEST_DEPENDENT_LIBS Blosc::blosc)
endif()

if(USE_BLOSC OR OpenVDB_USES_BLOSC OR USE_ZLIB OR OpenVDB_USES_ZLIB)
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
  list(APPEND OPENVDB_CORE_DEPENDENT_LIBS ZLIB::ZLIB)
endif()

target_link_libraries(vdb_test
  ${OPENVDB_TEST_DEPENDENT_LIBS}
)

if(OPENVDB_ENABLE_RPATH)
  set(RPATHS "")
  # @todo There is probably a better way to do this for imported targets
  list(APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
    ${Gtest_LIBRARY_DIRS}
  )
  if(OPENVDB_BUILD_CORE)
    list(APPEND RPATHS ${CMAKE_INSTALL_FULL_LIBDIR})
  else()
    list(APPEND RPATHS ${OpenVDB_LIBRARY_DIRS})
  endif()

  list(REMOVE_DUPLICATES RPATHS)

  set_target_properties(vdb_test
    PROPERTIES INSTALL_RPATH "${RPATHS}"
  )
endif()

add_test(vdb_unit_test vdb_test -v)
