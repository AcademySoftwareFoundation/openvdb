# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB AX Binary Tests

  These test run the vdb_ax command line binary and check that is passes
  or fails. They also serve as regression tests to make sure the binary
  behaviour does not unexpectedly change. To update the baselines, run
  this script with UPDATE_BASELINES=ON.

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)

option(UPDATE_BASELINES "Replace the expected outputs whilst running tests" OFF)
option(DOWNLOAD_VDBS "Fetch .vdb files required for some tests" OFF)

# Add some basic tests to verify the binary commands

if(NOT VDB_AX_BINARY_PATH)
  message(FATAL_ERROR "VDB_AX_BINARY_PATH is not defined for vdb_ax binary tests.")
endif()

# Remove .exe extension on windows to produce compatible diff outputs
get_filename_component(BINARY_NAME ${VDB_AX_BINARY_PATH} NAME_WE)
get_filename_component(PATH_TO_BIN ${VDB_AX_BINARY_PATH} DIRECTORY)
set(VDB_AX_BINARY_PATH "${PATH_TO_BIN}/${BINARY_NAME}")

set(SPHERE_POINTS_VDB ${CMAKE_BINARY_DIR}/sphere_points.vdb)
set(SPHERE_VDB ${CMAKE_BINARY_DIR}/sphere.vdb)
set(TORUS_VDB ${CMAKE_BINARY_DIR}/torus.vdb)

if(DOWNLOAD_VDBS)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  if(NOT EXISTS ${SPHERE_VDB} OR
     NOT EXISTS ${TORUS_VDB} OR
     NOT EXISTS ${SPHERE_POINTS_VDB})
    set(DOWNLOAD_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/../../../ci/download_vdb_caches.py)
    execute_process(COMMAND ${Python_EXECUTABLE} ${DOWNLOAD_SCRIPT} -f sphere.vdb torus.vdb sphere_points.vdb
      OUTPUT_VARIABLE DOWNLOAD_OUTPUT
      ERROR_VARIABLE  DOWNLOAD_OUTPUT
      RESULT_VARIABLE RETURN_CODE)
    if(RETURN_CODE)
      message(FATAL_ERROR "Failed to download required .vdb files:\n${RETURN_CODE}\n${DOWNLOAD_OUTPUT}")
    endif()
  endif()
endif()

set(HAS_DOWNLOAD_VDBS FALSE)
if(EXISTS ${SPHERE_VDB} AND
   EXISTS ${TORUS_VDB} AND
   EXISTS ${SPHERE_POINTS_VDB})
  set(HAS_DOWNLOAD_VDBS TRUE)
endif()

set(OUTPUT_DIR ${CMAKE_BINARY_DIR})
if(UPDATE_BASELINES)
  set(OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/cmd)
endif()

# If diff is available, use for verbose error messages
find_program(DIFF_COMMAND diff)

# Tracking variables
set(FAILED_TESTS "")
set(TEST_PASS_NUM 0)
set(TEST_FAIL_NUM 0)

###############################################################################
## @brief Register and run a test. Takes various arguments. Note that if
##   running a vdb_ax command with -s, the AX string must be passed separately
##   to with the AX keyword (otherwise semicolons are impossible to handle).
## KEYWORDS:
##   PASS, FAIL: Whether the test is expected to pass/fail
##   FILE_SUFFIX: Suffix of the test file output. If not provided, it's set to
##     either pass/fail and the test number.
##   GENERATES_OUTPUT: Whether the test is expected to generate output. If so,
##     the outputs are diffed. Note that line endings are NOT compared.
##   IGNORE_OUTPUT: Whether any output should be ignored. Overrides GENERATES_OUTPUT.
## SINGLE:
##   COMMAND: The binary to run. Defaults to vdb_ax
## MULTI:
##   ARGS: Arguments to pass to the command.
##   AX: Literal AX string to pass to the command with -s
###############################################################################
function(RUN_TEST)
  set(KEYWORD_OPTIONS PASS FAIL GENERATES_OUTPUT IGNORE_OUTPUT)
  set(SINGLE_OPTIONS COMMAND FILE_SUFFIX)
  set(MULTI_OPTIONS AX ARGS)
  cmake_parse_arguments(RUN_TEST "${KEYWORD_OPTIONS}" "${SINGLE_OPTIONS}" "${MULTI_OPTIONS}" ${ARGN})

  if(RUN_TEST_IGNORE_OUTPUT)
    set(RUN_TEST_GENERATES_OUTPUT FALSE)
  endif()

  set(TEST_FILE "vdb_ax_test")
  if(RUN_TEST_GENERATES_OUTPUT)
    if(RUN_TEST_PASS)
      if(NOT RUN_TEST_FILE_SUFFIX)
        math(EXPR TEST_PASS_NUM ${TEST_PASS_NUM}+1)
        set(RUN_TEST_FILE_SUFFIX "pass_${TEST_PASS_NUM}")
      endif()

      set(TEST_FILE "${TEST_FILE}_${RUN_TEST_FILE_SUFFIX}")
      set(TEST_PASS_NUM ${TEST_PASS_NUM} PARENT_SCOPE)
    elseif(RUN_TEST_FAIL)
      if(NOT RUN_TEST_FILE_SUFFIX)
        math(EXPR TEST_FAIL_NUM ${TEST_FAIL_NUM}+1)
        set(RUN_TEST_FILE_SUFFIX "fail_${TEST_FAIL_NUM}")
      endif()

      set(TEST_FILE "${TEST_FILE}_${RUN_TEST_FILE_SUFFIX}")
      set(TEST_FAIL_NUM ${TEST_FAIL_NUM} PARENT_SCOPE)
    endif()
  endif()

  set(TEST_EXE_COMMAND ${VDB_AX_BINARY_PATH})
  if(RUN_TEST_COMMAND)
    set(TEST_EXE_COMMAND ${RUN_TEST_COMMAND})
  endif()

  unset(RETURN_CODE)
  unset(TEST_OUTPUT)

  if(RUN_TEST_AX)
    string(REPLACE ";" "\;" RUN_TEST_AX "${RUN_TEST_AX}")
    set(RUN_TEST_AX "${RUN_TEST_AX}\;")
    list(APPEND RUN_TEST_ARGS -s "${RUN_TEST_AX}")
  endif()

  get_filename_component(EXE_COMMAND ${TEST_EXE_COMMAND} NAME)
  message(STATUS "Running: ${EXE_COMMAND} ${RUN_TEST_ARGS}")

  if(RUN_TEST_GENERATES_OUTPUT)
    execute_process(COMMAND ${TEST_EXE_COMMAND} ${RUN_TEST_ARGS}
      OUTPUT_FILE ${OUTPUT_DIR}/${TEST_FILE}
      ERROR_FILE  ${OUTPUT_DIR}/${TEST_FILE}
      RESULT_VARIABLE RETURN_CODE)
  else()
    execute_process(COMMAND ${TEST_EXE_COMMAND} ${RUN_TEST_ARGS}
      OUTPUT_VARIABLE TEST_OUTPUT
      ERROR_VARIABLE  TEST_OUTPUT
      RESULT_VARIABLE RETURN_CODE)
    if(NOT RUN_TEST_IGNORE_OUTPUT AND TEST_OUTPUT)
      message(STATUS "Unexpected output: ${TEST_OUTPUT}")
    endif()
  endif()

  if((RUN_TEST_FAIL AND (RETURN_CODE EQUAL 0)) OR
     (RUN_TEST_PASS AND NOT (RETURN_CODE EQUAL 0)) OR
     (NOT RUN_TEST_IGNORE_OUTPUT AND TEST_OUTPUT))
    string(REPLACE ";" " " RUN_TEST_ARGS "${RUN_TEST_ARGS}")
    set(FAILED_TESTS "${FAILED_TESTS};${TEST_EXE_COMMAND} ${RUN_TEST_ARGS}" PARENT_SCOPE)
    return()
  endif()

  unset(RETURN_CODE)

  if(RUN_TEST_GENERATES_OUTPUT)
    execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files --ignore-eol
      ${OUTPUT_DIR}/${TEST_FILE}
      ${CMAKE_CURRENT_LIST_DIR}/cmd/${TEST_FILE}
    RESULT_VARIABLE RETURN_CODE)
  endif()

  if(RETURN_CODE)
    if(DIFF_COMMAND)
      unset(DIFF_OUTPUT)
      execute_process(COMMAND ${DIFF_COMMAND}
          ${OUTPUT_DIR}/${TEST_FILE}
          ${CMAKE_CURRENT_LIST_DIR}/cmd/${TEST_FILE}
        OUTPUT_VARIABLE DIFF_OUTPUT)
      message(STATUS "Diff outputs failed: (${TEST_FILE})\n${DIFF_OUTPUT}")
    endif()
    string(REPLACE ";" " " RUN_TEST_ARGS "${RUN_TEST_ARGS}")
    set(FAILED_TESTS "${FAILED_TESTS};${TEST_EXE_COMMAND} ${RUN_TEST_ARGS}" PARENT_SCOPE)
  endif()
endfunction()

###############################################################################

# These tests should pass and the output should be checked

run_test(PASS GENERATES_OUTPUT ARGS -h)
run_test(PASS GENERATES_OUTPUT ARGS analyze -h)
run_test(PASS GENERATES_OUTPUT ARGS analyze --ast-print AX "@a+=@b;")
run_test(PASS GENERATES_OUTPUT ARGS analyze --re-print  AX "@a+=@b;")
run_test(PASS GENERATES_OUTPUT ARGS analyze --reg-print --try-compile AX "@a+=@b;")
run_test(PASS GENERATES_OUTPUT ARGS analyze -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop --ast-print)
run_test(PASS GENERATES_OUTPUT ARGS analyze -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop --re-print)
run_test(PASS GENERATES_OUTPUT ARGS analyze -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop --reg-print)
run_test(PASS GENERATES_OUTPUT ARGS analyze -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop --try-compile)
run_test(PASS GENERATES_OUTPUT ARGS analyze -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop --try-compile points)
run_test(PASS GENERATES_OUTPUT ARGS analyze -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop --try-compile volumes)
run_test(PASS GENERATES_OUTPUT ARGS functions -h)
run_test(PASS GENERATES_OUTPUT ARGS functions --list log)
run_test(PASS GENERATES_OUTPUT ARGS functions --list-names)
run_test(PASS GENERATES_OUTPUT ARGS execute -h)

# These tests should pass and produce no output

run_test(PASS ARGS analyze AX "@a+=@b;")
run_test(PASS ARGS analyze --try-compile AX "@a+=@b;")
run_test(PASS ARGS analyze --try-compile points AX "@a+=@b;")
run_test(PASS ARGS analyze --try-compile volumes AX "@a+=@b;")
run_test(PASS ARGS functions --list non-existant-function)

# These tests should fail and the output should be checked

run_test(FAIL GENERATES_OUTPUT)
run_test(FAIL GENERATES_OUTPUT ARGS -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop)
run_test(FAIL GENERATES_OUTPUT ARGS --invalid-option)
run_test(FAIL GENERATES_OUTPUT ARGS analyze -f invalid_file)
run_test(FAIL GENERATES_OUTPUT ARGS analyze --werror --try-compile AX "int a = 1.0;")
run_test(FAIL GENERATES_OUTPUT ARGS analyze --werror --try-compile --max-errors 0 AX "int a = 1.0; int a = 1.0;")
run_test(FAIL GENERATES_OUTPUT ARGS analyze --werror --try-compile --max-errors 1 AX "int a = 1.0; int a = 1.0;")
run_test(FAIL GENERATES_OUTPUT ARGS analyze --werror --try-compile --max-errors 2 AX "int a = 1.0; int a = 1.0;")
run_test(FAIL GENERATES_OUTPUT ARGS analyze AX "invalid code")
run_test(FAIL GENERATES_OUTPUT ARGS analyze --try-compile volumes AX "string str; bool a = ingroup(str);")
run_test(FAIL GENERATES_OUTPUT ARGS analyze analyze AX "@a+=@b;")
run_test(FAIL GENERATES_OUTPUT ARGS -i file.vdb execute AX "@a+=@b;")
run_test(FAIL GENERATES_OUTPUT ARGS execute -i file.vdb execute AX "@a+=@b;")
run_test(FAIL GENERATES_OUTPUT ARGS analyze -i file.vdb AX "@a+=@b;")
run_test(FAIL GENERATES_OUTPUT ARGS functions)
run_test(FAIL GENERATES_OUTPUT ARGS functions -i file.vdb)
run_test(FAIL GENERATES_OUTPUT ARGS file.vdb -o tmp.vdb AX "@ls_sphere += 1;")
run_test(FAIL GENERATES_OUTPUT ARGS execute file.vdb -o tmp.vdb AX "@ls_sphere += 1;")
run_test(FAIL GENERATES_OUTPUT ARGS --points-grain nan)
run_test(FAIL GENERATES_OUTPUT ARGS --volume-grain nan)

if(HAS_DOWNLOAD_VDBS)
  run_test(PASS GENERATES_OUTPUT FILE_SUFFIX ls_sphere_1 ARGS ${SPHERE_VDB} --threads 1
    AX "
    vec3i c = getcoord();
    if(c.y > 60)
      if(c.x < 2 && c.x > -2)
        if(c.z < 2 && c.z > -2)
          print(@ls_sphere);
    ")
  run_test(PASS GENERATES_OUTPUT FILE_SUFFIX ls_sphere_2 ARGS execute ${SPHERE_VDB} --threads 1
    AX "
    vec3i c = getcoord();
    if(c.y > 60)
      if(c.x < 2 && c.x > -2)
        if(c.z < 2 && c.z > -2)
          print(@ls_sphere);
    ")
  run_test(PASS GENERATES_OUTPUT FILE_SUFFIX ls_sphere_3 ARGS execute ${TORUS_VDB} ${SPHERE_VDB} --threads 1
    AX "
    @ls_sphere = max(@ls_torus, @ls_sphere);
    if (abs(@ls_sphere) < 1e-4 && @ls_sphere != 0.0f) {
      print(@ls_sphere);
    }")
  run_test(PASS GENERATES_OUTPUT FILE_SUFFIX vol_bind
    ARGS ${TORUS_VDB} ${SPHERE_VDB} --bindings "a:ls_sphere,b:ls_torus" --create-missing "OFF"
    AX "@a = 1; @b = 1;")

  # @todo  diff these once we have attribute bindings (can't diff the files)
  #   due to UUID changing
  run_test(PASS ARGS -i ${SPHERE_VDB} -o ${CMAKE_BINARY_DIR}/tmp.vdb AX "@ls_sphere += 1;")
  run_test(PASS ARGS -i ${SPHERE_VDB} -i ${CMAKE_BINARY_DIR}/tmp.vdb -o ${CMAKE_BINARY_DIR}/tmp.vdb AX "@ls_sphere -= 1;")
  run_test(PASS GENERATES_OUTPUT FILE_SUFFIX uuid ARGS ${SPHERE_VDB} ${CMAKE_BINARY_DIR}/tmp.vdb AX "@ls_sphere -= 1;")
  run_test(PASS COMMAND ${CMAKE_COMMAND} ARGS -E remove -f ${CMAKE_BINARY_DIR}/tmp.vdb)

  # fail tests
  run_test(FAIL GENERATES_OUTPUT
      FILE_SUFFIX attr_create
      ARGS ${SPHERE_POINTS_VDB} --create-missing "OFF"
      AX "@nonexist = 1;")
endif()

# These test fail and output is not checked
# @todo tests here should be documented as to why the output is not being
#   checked and transition to a FAIL GENERATES_OUTPUT call when possible.

# @note Produces a different exception with MSVC.
# @todo Make openvdb::io::File::getSize() equal cross platform (i.e. C++17)
run_test(FAIL IGNORE_OUTPUT ARGS invalid_file -f ${CMAKE_CURRENT_LIST_DIR}/snippets/loop/forLoop)

###############################################################################

if(FAILED_TESTS)
  set(MSG "The following vdb_ax test command failed:")
  foreach(FAILED_TEST ${FAILED_TESTS})
    set(MSG " ${MSG}\n ${FAILED_TEST}")
  endforeach()
  message(FATAL_ERROR ${MSG})
endif()
