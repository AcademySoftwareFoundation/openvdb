# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Python bindings

#]=======================================================================]

cmake_minimum_required(VERSION 3.20)
project(OpenVDBPython LANGUAGES CXX)

include(GNUInstallDirs)

###### OpenVDB Python Options

option(USE_NUMPY "Build the python library with numpy support." OFF)
option(OPENVDB_PYTHON_WRAP_ALL_GRID_TYPES [=[
Expose (almost) all of the grid types in the python module. Otherwise, only FloatGrid, BoolGrid and
Vec3SGrid will be exposed (see, e.g., exportIntGrid() in python/pyIntGrid.cc). Compiling the Python
module with this ON can be very memory-intensive.]=] OFF)
option(OPENVDB_BUILD_PYTHON_UNITTESTS [=[
"Include the OpenVDB Python unit test. Requires a python interpreter]=]
${OPENVDB_BUILD_UNITTESTS})

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "------------ Configuring OpenVDBPython -------------")
message(STATUS "----------------------------------------------------")

##########################################################################

# Collect and configure lib dependencies

if(NOT OPENVDB_BUILD_CORE)
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

if(USE_AX)
  # Link the python module against openvdb_ax
  if(NOT OPENVDB_BUILD_AX)
    find_package(OpenVDB REQUIRED openvdb_ax)
    set(OPENVDB_AX_LIB OpenVDB::openvdb_ax)
  else()
    set(OPENVDB_AX_LIB openvdb_ax)
  endif()
endif()

##########################################################################

nanobind_add_module(openvdb_python NB_STATIC
  pyFloatGrid.cc
  pyGridBase.cc
  pyIntGrid.cc
  pyMetadata.cc
  pyOpenVDBModule.cc
  pyPointGrid.cc
  pyTransform.cc
  pyVec3Grid.cc
)

target_link_libraries(openvdb_python PUBLIC ${OPENVDB_LIB})
if(OPENVDB_PYTHON_WRAP_ALL_GRID_TYPES)
  target_compile_definitions(openvdb_python PRIVATE "-DPY_OPENVDB_WRAP_ALL_GRID_TYPES")
endif()
if(USE_NUMPY)
  target_compile_definitions(openvdb_python PUBLIC "-DPY_OPENVDB_USE_NUMPY")
endif()
if(USE_AX)
  target_link_libraries(openvdb_python PUBLIC ${OPENVDB_AX_LIB})
  target_compile_definitions(openvdb_python PUBLIC "-DPY_OPENVDB_USE_AX")
endif()
set_target_properties(openvdb_python PROPERTIES OUTPUT_NAME "openvdb")
if(SKBUILD)
  set_target_properties(openvdb_python PROPERTIES INSTALL_RPATH "$ORIGIN")
  install(TARGETS openvdb_python DESTINATION ${OPENVDB_INSTALL_LIBDIR})
  install(FILES __init__.py DESTINATION openvdb)
else()
  install(TARGETS openvdb_python DESTINATION ${VDB_PYTHON_INSTALL_DIRECTORY})
endif()

# pytest
if(OPENVDB_BUILD_PYTHON_UNITTESTS)

  set(OPENVDB_PYTHON_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  if(WIN32)
    set(OPENVDB_PYTHON_WORKING_DIR "${OPENVDB_PYTHON_WORKING_DIR}/$<CONFIG>")
  endif()

  add_test(NAME pytest_openvdb
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/TestOpenVDB.py -v
    WORKING_DIRECTORY "${OPENVDB_PYTHON_WORKING_DIR}")

  set(OPENVDB_PYTHON_TEST_ENV "")
  if(USE_AX)
    list(APPEND OPENVDB_PYTHON_TEST_ENV "OPENVDB_TEST_PYTHON_AX=1")
  endif()

  if(WIN32)
    set(PYTHONPATH "$ENV{PYTHONPATH};${OPENVDB_PYTHON_WORKING_DIR}")
    string(REPLACE "\\;" ";" PYTHONPATH "${PYTHONPATH}")
    string(REPLACE ";" "\\;" PYTHONPATH "${PYTHONPATH}")
    set_tests_properties(pytest_openvdb PROPERTIES
      ENVIRONMENT "PYTHONPATH=${PYTHONPATH};${OPENVDB_PYTHON_TEST_ENV}")
  else()
    set_tests_properties(pytest_openvdb PROPERTIES
      ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${OPENVDB_PYTHON_WORKING_DIR};${OPENVDB_PYTHON_TEST_ENV}")
  endif()
endif()
