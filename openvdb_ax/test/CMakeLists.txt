# Copyright (c) 2015-2020 DNEG
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DNEG nor the names
# of its contributors may be used to endorse or promote products derived
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

cmake_minimum_required(VERSION 3.3)
project(OpenVDBAXUnitTests LANGUAGES CXX)

option(OPENVDB_AX_TEST_PROFILE "Switch on profiling for some of the unit tests." OFF)

##########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "---------- Configuring OpenVDBAXUnitTests ----------")
message(STATUS "----------------------------------------------------")

##########################################################################

if(NOT OPENVDB_BUILD_AX)
  set(OPENVDBAX_LIB OpenVDB::openvdb_ax)
else()
  set(OPENVDBAX_LIB openvdb_ax)
endif()

find_package(CppUnit REQUIRED)

set(TEST_SOURCE_FILES
  ast/TestScanners.cc
  ast/TestPrinters.cc
  backend/TestComputeGeneratorFailures.cc
  backend/TestFunctionGroup.cc
  backend/TestFunctionRegistry.cc
  backend/TestFunctionTypes.cc
  backend/TestSymbolTable.cc
  backend/TestTypes.cc
  compiler/TestPointExecutable.cc
  compiler/TestVolumeExecutable.cc
  frontend/TestArrayPack.cc
  frontend/TestArrayUnpackNode.cc
  frontend/TestAssignExpressionNode.cc
  frontend/TestAttributeNode.cc
  frontend/TestBinaryOperatorNode.cc
  frontend/TestCastNode.cc
  frontend/TestCommaOperator.cc
  frontend/TestConditionalStatementNode.cc
  frontend/TestCrementNode.cc
  frontend/TestDeclareLocalNode.cc
  frontend/TestExternalVariableNode.cc
  frontend/TestFunctionCallNode.cc
  frontend/TestKeywordNode.cc
  frontend/TestLocalNode.cc
  frontend/TestLoopNode.cc
  frontend/TestStatementListNode.cc
  frontend/TestSyntaxFailures.cc
  frontend/TestTernaryOperatorNode.cc
  frontend/TestUnaryOperatorNode.cc
  frontend/TestValueNode.cc
  integration/CompareGrids.cc
  integration/TestArrayUnpack.cc
  integration/TestAssign.cc
  integration/TestBinary.cc
  integration/TestCast.cc
  integration/TestConditional.cc
  integration/TestCrement.cc
  integration/TestDeclare.cc
  integration/TestEmpty.cc
  integration/TestExternals.cc
  integration/TestHarness.cc
  integration/TestKeyword.cc
  integration/TestLoop.cc
  integration/TestStandardFunctions.cc
  integration/TestString.cc
  integration/TestTernary.cc
  integration/TestUnary.cc
  integration/TestVDBFunctions.cc
  integration/TestWorldSpaceAccessors.cc
  main.cc
  )

add_executable(vdb_ax_test
  ${TEST_SOURCE_FILES}
)

target_link_libraries(vdb_ax_test
  ${OPENVDBAX_LIB}
  CppUnit::cppunit
)
target_include_directories(vdb_ax_test
  PRIVATE ../ .
)

if(OPENVDB_AX_TEST_PROFILE)
  target_compile_definitions(vdb_ax_test PRIVATE "-DPROFILE")
endif()
if(NOT MATRIX_SUPPORT)
  target_compile_definitions(vdb_ax_test PRIVATE -DOPENVDB_AX_NO_MATRIX)
endif()

add_test(NAME vdb_ax_unit_test COMMAND vdb_ax_test -v WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)
