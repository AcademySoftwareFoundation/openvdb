# Copyright (c) DreamWorks Animation LLC
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

  CMake Configuration for OpenVDB Python bindings

#]=======================================================================]

project(OpenVDBPython)
cmake_minimum_required(VERSION 3.3)
# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

###### OpenVDB Python Options

option(USE_NUMPY [=[
Build the python library with numpy support. Currently requires CMake 3.14.]=] OFF)
option(OPENVDB_PYTHON_WRAP_ALL_GRID_TYPES [=[
Expose (almost) all of the grid types in the python module. Otherwise, only FloatGrid, BoolGrid and
Vec3SGrid will be exposed (see, e.g., exportIntGrid() in python/pyIntGrid.cc). Compiling the Python
module with this ON can be very memory-intensive.]=] OFF)

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

# NumPy requires CMake 3.14
if(USE_NUMPY AND (${CMAKE_VERSION} VERSION_LESS 3.14))
  message(FATAL_ERROR "No CMake support for FindPackage ( NumPy) currently "
    "available for OpenVDB. Requires CMake 3.14."
  )
endif()

set(OPENVDB_PYTHON_DEPS)
set(OPENVDB_PYTHON_INCLUDES)

# If CMake is < 3.12, use the old style python search and alias the variables
# Note that the Interpreter component is only required for the python test
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  find_package(PythonInterp ${MINIMUM_PYTHON_VERSION} REQUIRED)
  find_package(PythonLibs ${MINIMUM_PYTHON_VERSION} REQUIRED)

  set(Python_VERSION_MAJOR ${PYTHON_VERSION_MAJOR})
  set(Python_VERSION_MINOR ${PYTHON_VERSION_MINOR})
  set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
  set(OPENVDB_PYTHON_DEPS ${PYTHON_LIBRARIES})
  get_filename_component(Python_LIBRARY_DIRS ${OPENVDB_PYTHON_DEPS} DIRECTORY)

  list(APPEND OPENVDB_PYTHON_INCLUDES ${PYTHON_INCLUDE_DIR})
else()
  # To ensure consistent versions between components Interpreter, Compiler,
  # Development and NumPy, specify all components at the same time when using
  # FindPython
  if(USE_NUMPY)
    find_package(Python ${MINIMUM_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development NumPy)
    if(Python_NumPy_VERSION VERSION_LESS MINIMUM_NUMPY_VERSION)
      message(FATAL_ERROR "Could NOT find NumPy (Required is at least version "
        "\"${MINIMUM_NUMPY_VERSION}\")"
      )
    else()
      message(STATUS "Found NumPy: ${Python_NumPy_INCLUDE_DIRS} (found suitable "
        "version \"${Python_NumPy_VERSION}\", minimum required is "
        "\"${MINIMUM_NUMPY_VERSION}\")"
      )
    endif()
    list(APPEND OPENVDB_PYTHON_DEPS Python::NumPy)
  else()
    find_package(Python ${MINIMUM_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)
  endif()
  list(APPEND OPENVDB_PYTHON_DEPS Python::Python)
endif()

if(USE_NUMPY)
  if(Python_NumPy_VERSION VERSION_LESS FUTURE_MINIMUM_NUMPY_VERSION)
    message(DEPRECATION "Support for NumPy versions < ${FUTURE_MINIMUM_NUMPY_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

# Boost python handling - try and find both python and pythonXx (version suffixed).
# Prioritize the version suffixed library, failing if neither exist.

find_package(Boost ${MINIMUM_BOOST_VERSION}
  QUIET COMPONENTS python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}
)

if(TARGET Boost::python${Python_VERSION_MAJOR}${Python_VERSION_MINOR})
  set(BOOST_PYTHON_LIB "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
  message(STATUS "Found boost_python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
else()
  find_package(Boost ${MINIMUM_BOOST_VERSION}
    QUIET COMPONENTS python
  )
  if(TARGET Boost::python)
    set(BOOST_PYTHON_LIB "python")
    message(STATUS "Found non-suffixed boost_python, assuming to be python version "
      "\"${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}\" compatible"
    )
  else()
    message(FATAL_ERROR "Unable to find boost_python or "
      "boost_python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}."
    )
  endif()
endif()

set(OPENVDB_PYTHON_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  Boost::${BOOST_PYTHON_LIB}
  ${OPENVDB_PYTHON_DEPS}
)

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
  set(PYOPENVDB_INSTALL_DIRECTORY
    lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}
    CACHE STRING "The directory to install the pyopenvdb.so module."
  )
endif()

add_library(pyopenvdb SHARED
  ${OPENVDB_PYTHON_MODULE_SOURCE_FILES}
)

target_include_directories(pyopenvdb
  SYSTEM PUBLIC ${OPENVDB_PYTHON_INCLUDES}
)

if(OPENVDB_PYTHON_WRAP_ALL_GRID_TYPES)
  target_compile_definitions(pyopenvdb PRIVATE "-DPY_OPENVDB_WRAP_ALL_GRID_TYPES")
endif()

if(USE_NUMPY)
  target_compile_definitions(pyopenvdb PUBLIC "-DPY_OPENVDB_USE_NUMPY")
endif()

target_link_libraries(pyopenvdb
  ${OPENVDB_PYTHON_DEPENDENT_LIBS}
)

set_target_properties(pyopenvdb PROPERTIES
  PREFIX ""  # no 'lib' prefix
)

if(UNIX)
  # must be .so (not .dylib)
  set_target_properties(pyopenvdb PROPERTIES
    SUFFIX ".so"
  )
endif()

if(OPENVDB_ENABLE_RPATH)
  # @todo There is probably a better way to do this for imported targets
  set(RPATHS "")
  list(APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
    ${Python_LIBRARY_DIRS}
  )
  if(OPENVDB_BUILD_CORE)
    list(APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib)
  else()
    list(APPEND RPATHS ${OpenVDB_LIBRARY_DIRS})
  endif()

  list(REMOVE_DUPLICATES RPATHS)
  set_target_properties(pyopenvdb
    PROPERTIES INSTALL_RPATH "${RPATHS}"
  )
  unset(RPATHS)
endif()

set(PYTHON_PUBLIC_INCLUDE_NAMES
  pyopenvdb.h
)

install(TARGETS
  pyopenvdb
  DESTINATION
  ${PYOPENVDB_INSTALL_DIRECTORY}
)

install(FILES ${PYTHON_PUBLIC_INCLUDE_NAMES} DESTINATION include/openvdb/python)

# pytest

add_test(pytest ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/TestOpenVDB.py)
set_tests_properties(pytest PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")


