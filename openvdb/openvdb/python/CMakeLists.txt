# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Python bindings

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
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

# Small function which mimics basic output (bar components) of
# FindPackageHandleStandardArgs. This is required as we want to ensure
# the minimum python version is MINIMUM_PYTHON_VERSION - however this cannot
# be provided to find_package(Python) with differing major versions. e.g.
# calls to find_package(Python 2.7) fails if python3 is found on the system.
function(OPENVDB_CHECK_PYTHON_VERSION)
  set(PY_TARGET ${ARGV0})
  set(PY_TARGET_VERSION ${ARGV1})
  set(PY_TARGET_INCLUDES ${ARGV2})
  set(MIN_VERSION ${ARGV3})
  set(FUTURE_MIN_VERSION ${ARGV4})

  if(NOT TARGET ${PY_TARGET})
    message(FATAL_ERROR "Could NOT find ${PY_TARGET} (Required is at least version "
      "\"${MIN_VERSION}\")"
    )
  endif()

  if(PY_TARGET_VERSION AND MIN_VERSION)
    if(PY_TARGET_VERSION VERSION_LESS MIN_VERSION)
      message(FATAL_ERROR "Could NOT find ${PY_TARGET}: Found unsuitable version "
        "\"${PY_TARGET_VERSION}\" but required is at least \"${MIN_VERSION}\" (found ${PY_TARGET_INCLUDES})"
      )
    endif()
  endif()

  message(STATUS "Found ${PY_TARGET}: ${PY_TARGET_INCLUDES}) (found suitable "
    "version \"${PY_TARGET_VERSION}\", minimum required is \"${MIN_VERSION}\")"
  )

  if(OPENVDB_FUTURE_DEPRECATION AND PY_TARGET_VERSION AND FUTURE_MIN_VERSION)
    if(PY_TARGET_VERSION VERSION_LESS FUTURE_MIN_VERSION)
      message(DEPRECATION "Support for ${PY_TARGET} versions < ${FUTURE_MIN_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
endfunction()

# Configure Python and Numpy.
# To ensure consistent versions between components Interpreter, Compiler,
# Development and NumPy, specify all components at the same time when using
# FindPython.
set(OPENVDB_PYTHON_DEPS)

# @note  explicitly only search for Development.Module from 3.18 as searching
#   Development.Embed can cause issues on linux systems where it doesn't exist
set(OPENVDB_PYTHON_REQUIRED_COMPONENTS Development.Module Interpreter)

if(NOT DEFINED PYOPENVDB_INSTALL_DIRECTORY)
  list(APPEND OPENVDB_PYTHON_REQUIRED_COMPONENTS Interpreter)
endif()

if(USE_NUMPY)
  list(APPEND OPENVDB_PYTHON_REQUIRED_COMPONENTS NumPy)
endif()

# Make sure find_package(Python) is only ever invoked once with all required components
find_package(Python COMPONENTS ${OPENVDB_PYTHON_REQUIRED_COMPONENTS})
find_package(pybind11 ${MINIMUM_PYBIND_VERSION} CONFIG REQUIRED)

openvdb_check_python_version(Python::Module
  "${Python_VERSION}"
  "${Python_INCLUDE_DIRS}"
  "${MINIMUM_PYTHON_VERSION}"
  "${FUTURE_MINIMUM_PYTHON_VERSION}")
list(APPEND OPENVDB_PYTHON_DEPS Python::Module)

if(USE_NUMPY)
  openvdb_check_python_version(Python::NumPy
    "${Python_NumPy_VERSION}"
    "${Python_NumPy_INCLUDE_DIRS}"
    "${MINIMUM_NUMPY_VERSION}"
    "${FUTURE_MINIMUM_NUMPY_VERSION}")
  list(APPEND OPENVDB_PYTHON_DEPS Python::NumPy)
endif()

##########################################################################

set(OPENVDB_PYTHON_MODULE_SOURCE_FILES
  pyFloatGrid.cc
  pyGridBase.cc
  pyIntGrid.cc
  pyMetadata.cc
  pyOpenVDBModule.cc
  pyPointGrid.cc
  pyTransform.cc
  pyVec3Grid.cc
)

if(NOT DEFINED PYOPENVDB_INSTALL_DIRECTORY)
  get_filename_component(Python_PACKAGES_DIR ${Python_SITELIB} NAME)
  set(PYOPENVDB_INSTALL_DIRECTORY
    ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/${Python_PACKAGES_DIR}
    CACHE STRING "The directory to install the pyopenvdb.so module."
  )
endif()

pybind11_add_module(pyopenvdb ${OPENVDB_PYTHON_MODULE_SOURCE_FILES})

target_link_libraries(pyopenvdb PUBLIC
  ${OPENVDB_LIB}
  ${OPENVDB_PYTHON_DEPS}
)

if(OPENVDB_PYTHON_WRAP_ALL_GRID_TYPES)
  target_compile_definitions(pyopenvdb PRIVATE "-DPY_OPENVDB_WRAP_ALL_GRID_TYPES")
endif()
if(USE_NUMPY)
  target_compile_definitions(pyopenvdb PUBLIC "-DPY_OPENVDB_USE_NUMPY")
endif()
if(USE_AX)
  target_link_libraries(pyopenvdb PUBLIC ${OPENVDB_AX_LIB})
  target_compile_definitions(pyopenvdb PUBLIC "-DPY_OPENVDB_USE_AX")
endif()

install(TARGETS
  pyopenvdb
  DESTINATION
  ${PYOPENVDB_INSTALL_DIRECTORY}
)

# pytest
if(OPENVDB_BUILD_PYTHON_UNITTESTS)

  set(PYVDB_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  if(WIN32)
    set(PYVDB_WORKING_DIR "${PYVDB_WORKING_DIR}/$<CONFIG>")
  endif()

  add_test(NAME pytest
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/TestOpenVDB.py -v
    WORKING_DIRECTORY "${PYVDB_WORKING_DIR}")

  set(PYOPENVDB_TEST_ENV "")
  if(USE_AX)
    list(APPEND PYOPENVDB_TEST_ENV "OPENVDB_TEST_PYTHON_AX=1")
  endif()

  if(WIN32)
    set(PYTHONPATH "$ENV{PYTHONPATH};${PYVDB_WORKING_DIR}")
    string(REPLACE "\\;" ";" PYTHONPATH "${PYTHONPATH}")
    string(REPLACE ";" "\\;" PYTHONPATH "${PYTHONPATH}")
    set_tests_properties(pytest PROPERTIES
      ENVIRONMENT "PYTHONPATH=${PYTHONPATH};${PYOPENVDB_TEST_ENV}")
  else()
    set_tests_properties(pytest PROPERTIES
      ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${PYVDB_WORKING_DIR};${PYOPENVDB_TEST_ENV}")
  endif()
endif()
