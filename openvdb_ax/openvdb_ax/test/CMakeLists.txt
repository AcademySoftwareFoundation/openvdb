# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Unit Tests

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(OpenVDBAXUnitTests LANGUAGES CXX)

option(OPENVDB_AX_TEST_PROFILE "Switch on profiling for some of the unit tests." OFF)
option(OPENVDB_AX_TEST_CMD "Enable/disable tests for the vdb_ax binary." ON)
option(OPENVDB_AX_TEST_CMD_DOWNLOADS "Enable/disable tests for the vdb_ax binary which require downloads." OFF)

##########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "---------- Configuring OpenVDBAXUnitTests ----------")
message(STATUS "----------------------------------------------------")

##########################################################################

if(TARGET vdb_ax AND OPENVDB_AX_TEST_CMD)
  add_test(NAME vdb_ax_cmd_test
    COMMAND ${CMAKE_COMMAND}
      -DVDB_AX_BINARY_PATH=$<TARGET_FILE:vdb_ax>
      -DDOWNLOAD_VDBS=${OPENVDB_AX_TEST_CMD_DOWNLOADS}
      -DCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}
      -P ${CMAKE_CURRENT_LIST_DIR}/TestAXCmd.cmake)
elseif(OPENVDB_BUILD_BINARIES)
  message(WARNING "Unable to test the vdb_ax binary as the vdb_ax target was not found!")
endif()

if(NOT OPENVDB_BUILD_AX)
  set(OPENVDBAX_LIB OpenVDB::openvdb_ax)
else()
  set(OPENVDBAX_LIB openvdb_ax)
endif()

set(OPENVDB_AX_TEST_DEPENDENT_LIBS ${OPENVDBAX_LIB})

if(CONCURRENT_MALLOC STREQUAL "Jemalloc")
  find_package(Jemalloc REQUIRED)
  list(APPEND OPENVDB_AX_TEST_DEPENDENT_LIBS Jemalloc::jemalloc)
elseif(CONCURRENT_MALLOC STREQUAL "Tbbmalloc")
  find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbbmalloc)
  list(APPEND OPENVDB_AX_TEST_DEPENDENT_LIBS TBB::tbbmalloc)
endif()

find_package(CppUnit ${MINIMUM_CPPUNIT_VERSION} REQUIRED)
list(APPEND OPENVDB_AX_TEST_DEPENDENT_LIBS CppUnit::cppunit)

set(TEST_SOURCE_FILES
  ast/TestScanners.cc
  ast/TestPrinters.cc
  backend/TestAttributeBindings.cc
  backend/TestCodecs.cc
  backend/TestComputeGeneratorFailures.cc
  backend/TestFunctionGroup.cc
  backend/TestFunctionRegistry.cc
  backend/TestFunctionTypes.cc
  backend/TestLogger.cc
  backend/TestStringIR.cc
  backend/TestSymbolTable.cc
  backend/TestTypes.cc
  compiler/TestAXRun.cc
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

add_executable(vdb_ax_test ${TEST_SOURCE_FILES})

# @note  From Clang 14, fp-contract has been switched to ON by default. Docs in
#  Clang versions before 14 state that fp-contract is ON, but it is in fact off.
#  This options effects the emmission on fmuladd instructions in the Clang
#  driver. Because this behaviour requires statement contexts (these instructions
#  are only emitted on intermediate floating point results, not across statements)
#  it cannot be an optimization pass in LLVM and must be emitted in the Clang
#  frontend. This causes a difference to how AX generated (a*b)+c. For now, turn
#  of fused instructions in tests where it matters.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set_source_files_properties(
      integration/TestBinary.cc
      integration/TestStandardFunctions.cc
    PROPERTIES
      COMPILE_OPTIONS "-ffp-contract=off")
endif()

target_link_libraries(vdb_ax_test ${OPENVDB_AX_TEST_DEPENDENT_LIBS})
target_include_directories(vdb_ax_test PRIVATE ../ .)

if(OPENVDB_AX_TEST_PROFILE)
  target_compile_definitions(vdb_ax_test PRIVATE "-DPROFILE")
endif()

add_test(NAME vdb_ax_unit_test COMMAND vdb_ax_test -v WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)

# For the sanitizers, add the suppression files and additional options
get_filename_component(PATH_TO_PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
get_filename_component(PATH_TO_PROJECT_ROOT ${PATH_TO_PROJECT_ROOT} DIRECTORY)
get_filename_component(PATH_TO_PROJECT_ROOT ${PATH_TO_PROJECT_ROOT} DIRECTORY)
set(LSAN_SUPRESSION_FILE ${PATH_TO_PROJECT_ROOT}/cmake/scripts/lsan.supp)
set(UBSAN_SUPRESSION_FILE ${PATH_TO_PROJECT_ROOT}/cmake/scripts/ubsan.supp)

set(UBSAN_OPTS "$<$<CONFIG:UBSAN>:UBSAN_OPTIONS=halt_on_error=1 report_error_type=1 suppressions=${UBSAN_SUPRESSION_FILE}>")
set(LSAN_OPTS  "$<$<CONFIG:LSAN>:LSAN_OPTIONS=suppressions=${LSAN_SUPRESSION_FILE}>")
set(ASAN_OPTS  "$<$<CONFIG:ASAN>:LSAN_OPTIONS=suppressions=${LSAN_SUPRESSION_FILE}>")

set_tests_properties(vdb_ax_unit_test PROPERTIES
    ENVIRONMENT "$<JOIN:${UBSAN_OPTS};${LSAN_OPTS};${ASAN_OPTS}, >")
