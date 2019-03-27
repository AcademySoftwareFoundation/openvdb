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

  CMake Configuration for OpenVDB Unit Tests

#]=======================================================================]

PROJECT ( OpenVDBUnitTests )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
# Monitoring <PackageName>_ROOT variables
IF ( POLICY CMP0074 )
  CMAKE_POLICY ( SET CMP0074 NEW )
ENDIF ()

##########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "----------- Configuring OpenVDBUnitTests -----------" )
MESSAGE ( STATUS "----------------------------------------------------" )

##########################################################################

# Collect lib dependencies

IF ( NOT OPENVDB_BUILD_CORE )
  SET ( OPENVDB_LIB OpenVDB::openvdb )
ELSE ()
  SET ( OPENVDB_LIB openvdb_shared )
ENDIF ()

FIND_PACKAGE ( CppUnit REQUIRED )

SET ( OPENVDB_TEST_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  CppUnit::cppunit
)

##########################################################################

IF ( WIN32 AND OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING )
  ADD_DEFINITIONS ( -DBOOST_ALL_NO_LIB )
ENDIF ()

IF ( WIN32 )
  # Because of implicit linking!
  LINK_DIRECTORIES ( ${Boost_LIBRARY_DIR} )
ENDIF ()

IF ( WIN32 )
  ADD_DEFINITIONS ( -D_WIN32 -DNOMINMAX -DHALF_EXPORTS -DOPENVDB_DLL )
ENDIF ()

##### VDB unit tests

SET ( UNITTEST_SOURCE_FILES
  main.cc
  TestAttributeArray.cc
  TestAttributeArrayString.cc
  TestAttributeGroup.cc
  TestAttributeSet.cc
  TestBBox.cc
  TestConjGradient.cc
  TestCoord.cc
  TestCpt.cc
  TestCurl.cc
  TestDense.cc
  TestDenseSparseTools.cc
  TestDiagnostics.cc
  TestDivergence.cc
  TestDoubleMetadata.cc
  TestExceptions.cc
  TestFile.cc
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
  TestLeafManager.cc
  TestLeafMask.cc
  TestLeafIO.cc
  TestLeafOrigin.cc
  TestLevelSetRayIntersector.cc
  TestLevelSetUtil.cc
  TestLinearInterp.cc
  TestMaps.cc
  TestMat4Metadata.cc
  TestMath.cc
  TestMeanCurvature.cc
  TestMeshToVolume.cc
  TestMetadata.cc
  TestMetadataIO.cc
  TestMetaMap.cc
  TestMultiResGrid.cc
  TestName.cc
  TestNodeIterator.cc
  TestNodeManager.cc
  TestNodeMask.cc
  TestParticleAtlas.cc
  TestParticlesToLevelSet.cc
  TestPointAttribute.cc
  TestPointConversion.cc
  TestPointCount.cc
  TestPointDataLeaf.cc
  TestPointGroup.cc
  TestPointIndexGrid.cc
  TestPointPartitioner.cc
  TestPointsToMask.cc
  TestPoissonSolver.cc
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
  )

ADD_EXECUTABLE ( vdb_test ${UNITTEST_SOURCE_FILES} )

IF ( USE_BLOSC OR OpenVDB_USES_BLOSC )
  # Blosc is a hidden dependency for the core library (not exposed in headers)
  # so we need to manually link it in here to provide header access for the
  # blosc relevant unit tests
  FIND_PACKAGE ( Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED )
  LIST ( APPEND OPENVDB_TEST_DEPENDENT_LIBS Blosc::blosc )
  TARGET_COMPILE_DEFINITIONS ( vdb_test PRIVATE "-DOPENVDB_USE_BLOSC" )
ENDIF ()

TARGET_LINK_LIBRARIES ( vdb_test ${OPENVDB_TEST_DEPENDENT_LIBS} )

ADD_TEST ( vdb_unit_test vdb_test )
