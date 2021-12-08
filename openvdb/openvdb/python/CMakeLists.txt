# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Python bindings

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
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

# Configure Python and Numpy. Note that:
#  - find_package(Python Development) target Python::Module requires CMake >= 3.15
#  - find_package(Python Development.Module) requires CMake >= 3.18
# To ensure consistent versions between components Interpreter, Compiler,
# Development and NumPy, specify all components at the same time when using
# FindPython.
# @note  the Python::Module target should be used over the Python::Python
#   target but this was only added in CMake 3.15. See:
#      https://github.com/AcademySoftwareFoundation/openvdb/issues/886
set(OPENVDB_PYTHON_DEPS)

# @note  explicitly only search for Development.Module from 3.18 as searching
#   Development.Embed can cause issues on linux systems where it doesn't exist
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  set(OPENVDB_PYTHON_REQUIRED_COMPONENTS Development)
else()
  set(OPENVDB_PYTHON_REQUIRED_COMPONENTS Development.Module)
endif()

if(NOT DEFINED PYOPENVDB_INSTALL_DIRECTORY)
  list(APPEND OPENVDB_PYTHON_REQUIRED_COMPONENTS Interpreter)
endif()

if(USE_NUMPY)
  list(APPEND OPENVDB_PYTHON_REQUIRED_COMPONENTS NumPy)
endif()

# Make sure find_package(Python) is only ever invoked once with all required components
find_package(Python COMPONENTS ${OPENVDB_PYTHON_REQUIRED_COMPONENTS})

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

if(TARGET openvdb_shared AND NOT Boost_USE_STATIC_LIBS)
  # @note  Both of these must be set for Boost 1.70 (VFX2020) to link against
  #        boost shared libraries (more specifically libraries built with -fPIC).
  #        http://boost.2283326.n4.nabble.com/CMake-config-scripts-broken-in-1-70-td4708957.html
  #        https://github.com/boostorg/boost_install/commit/160c7cb2b2c720e74463865ef0454d4c4cd9ae7c
  set(BUILD_SHARED_LIBS ON)
  set(Boost_USE_STATIC_LIBS OFF)
endif()

# Boost python handling. Try to find the following named boost python libraries:
# - boost_python{Python_VERSION_MAJOR}${Python_VERSION_MINOR}
# - boost_python{Python_VERSION_MAJOR}
# - boost_python
# Prioritize the version suffixed library, failing if none exist.

find_package(Boost ${MINIMUM_BOOST_VERSION}
  QUIET COMPONENTS python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}
)
find_package(Boost ${MINIMUM_BOOST_VERSION}
  QUIET COMPONENTS python${Python_VERSION_MAJOR}
)
find_package(Boost ${MINIMUM_BOOST_VERSION}
  QUIET COMPONENTS python
)

if(TARGET Boost::python${Python_VERSION_MAJOR}${Python_VERSION_MINOR})
  message(STATUS "Found boost_python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
  list(APPEND OPENVDB_PYTHON_DEPS Boost::python${Python_VERSION_MAJOR}${Python_VERSION_MINOR})
elseif(TARGET Boost::python${Python_VERSION_MAJOR})
  message(STATUS "Found boost_python${Python_VERSION_MAJOR}")
  list(APPEND OPENVDB_PYTHON_DEPS Boost::python${Python_VERSION_MAJOR})
elseif(TARGET Boost::python)
  message(STATUS "Found non-suffixed boost_python, assuming to be python version "
    "\"${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}\" compatible"
  )
  list(APPEND OPENVDB_PYTHON_DEPS Boost::python)
else()
  message(FATAL_ERROR "Unable to find boost_python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}, "
    "boost_python${Python_VERSION_MAJOR} or boost_python."
  )
endif()

# If boost >= 1.65, find the required numpy boost component and link that in.
# Use the same system as above, first trying without the suffix, then with.

if(USE_NUMPY)
  find_package(Boost ${MINIMUM_BOOST_VERSION}
    QUIET COMPONENTS numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR}
  )
  find_package(Boost ${MINIMUM_BOOST_VERSION}
    QUIET COMPONENTS numpy${Python_VERSION_MAJOR}
  )
  find_package(Boost ${MINIMUM_BOOST_VERSION}
    QUIET COMPONENTS numpy
  )

  if(TARGET Boost::numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR})
    message(STATUS "Found boost_numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
    list(APPEND OPENVDB_PYTHON_DEPS Boost::numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR})
  elseif(TARGET Boost::numpy${Python_VERSION_MAJOR})
    message(STATUS "Found boost_numpy${Python_VERSION_MAJOR}")
    list(APPEND OPENVDB_PYTHON_DEPS Boost::numpy${Python_VERSION_MAJOR})
  elseif(TARGET Boost::numpy)
    message(STATUS "Found non-suffixed boost_numpy, assuming to be python version "
      "\"${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}\" compatible"
    )
    list(APPEND OPENVDB_PYTHON_DEPS Boost::numpy)
  else()
    message(FATAL_ERROR "Unable to find boost_numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR}, "
      "boost_numpy${Python_VERSION_MAJOR} or boost_numpy."
    )
  endif()
endif()

##########################################################################

set(OPENVDB_PYTHON_MODULE_SOURCE_FILES
  pyFloatGrid.cc
  pyIntGrid.cc
  pyMetadata.cc
  pyPointGrid.cc
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

# @todo  pyopenvdb is currently advertised as "linkable" by extension libraries
#        so we have to mark it as SHARED. Really it should be MODULE, as the
#        amount of manipulation required for native python support nullifies
#        the ability for compilers to link against it (suffix/prefix renaming).
#        A proper shared lib should be built with the functions required for
#        extension (pyopenvdb.h) and a further module lib should be added
#        that can be imported.

add_library(pyopenvdb SHARED ${OPENVDB_PYTHON_MODULE_SOURCE_FILES})
set_target_properties(pyopenvdb PROPERTIES PREFIX "")  # no 'lib' prefix
if(UNIX)
  set_target_properties(pyopenvdb PROPERTIES SUFFIX ".so") # must be .so (not .dylib)
elseif(WIN32)
  set_target_properties(pyopenvdb PROPERTIES SUFFIX ".pyd") # .pyd on windows
endif()

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

set(PYTHON_PUBLIC_INCLUDE_NAMES
  pyopenvdb.h
)

install(TARGETS
  pyopenvdb
  DESTINATION
  ${PYOPENVDB_INSTALL_DIRECTORY}
)

install(FILES ${PYTHON_PUBLIC_INCLUDE_NAMES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb/python)

# pytest
if(OPENVDB_BUILD_PYTHON_UNITTESTS)
  add_test(pytest ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/TestOpenVDB.py -v)
  if(WIN32)
    set(PYTHONPATH "$ENV{PYTHONPATH};${CMAKE_CURRENT_BINARY_DIR}")
    string(REPLACE "\\;" ";" PYTHONPATH "${PYTHONPATH}")
    string(REPLACE ";" "\\;" PYTHONPATH "${PYTHONPATH}")
    set_tests_properties(pytest PROPERTIES
      ENVIRONMENT "PYTHONPATH=${PYTHONPATH}")
  else()
    set_tests_properties(pytest PROPERTIES
      ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}")
  endif()
endif()
