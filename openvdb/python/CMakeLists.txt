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

  CMake Configuration for OpenVDB Python bindings

#]=======================================================================]

PROJECT ( OpenVDBPython )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
# Monitoring <PackageName>_ROOT variables
IF ( POLICY CMP0074 )
  CMAKE_POLICY ( SET CMP0074 NEW )
ENDIF ()

###### OpenVDB Python Options

OPTION ( USE_NUMPY [=[
Build the python library with numpy support. Currently requires CMake 3.14.]=] OFF )
OPTION ( OPENVDB_PYTHON_WRAP_ALL_GRID_TYPES [=[
Expose (almost) all of the grid types in the python module. Otherwise, only FloatGrid, BoolGrid and
Vec3SGrid will be exposed (see, e.g., exportIntGrid() in python/pyIntGrid.cc). Compiling the Python
module with this ON can be very memory-intensive.]=] OFF )

SET ( PYTHON_VERSION "2.7" CACHE STRING "The python version to use when building the OpenVDB
  Python module. Note that the available version of Boost.Python should also match this." )

#########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "------------ Configuring OpenVDBPython -------------" )
MESSAGE ( STATUS "----------------------------------------------------" )

##########################################################################

# Collect and configure lib dependencies

IF ( NOT OPENVDB_BUILD_CORE )
  FIND_PACKAGE ( OpenVDB REQUIRED )
  SET ( OPENVDB_LIB OpenVDB::openvdb )
ELSE ()
  SET ( OPENVDB_LIB openvdb_shared )
  INCLUDE_DIRECTORIES ( ../../ )
ENDIF ()

FIND_PACKAGE ( IlmBase REQUIRED COMPONENTS Half )
FIND_PACKAGE ( TBB REQUIRED COMPONENTS tbb )
FIND_PACKAGE ( ZLIB REQUIRED )

IF ( USE_LOG4CPLUS )
  FIND_PACKAGE ( Log4cplus REQUIRED )
ENDIF ()

IF ( USE_BLOSC )
  FIND_PACKAGE ( Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED )
ENDIF ()

# NumPy requires CMake 3.14
IF ( USE_NUMPY AND ( ${CMAKE_VERSION} VERSION_LESS 3.14 ) )
  MESSAGE ( FATAL_ERROR "No CMake support for FindPackage ( NumPy ) currently "
    "available for OpenVDB. Requires CMake 3.14." )
ENDIF ()

SET ( OPENVDB_PYTHON_DEPS )

# If CMake is < 3.12, use the old style python search and alias the variables
# Note that the Interpreter component is only required for the python test
IF ( ${CMAKE_VERSION} VERSION_LESS 3.12 )
  FIND_PACKAGE ( PythonInterp ${PYTHON_VERSION} REQUIRED )
  FIND_PACKAGE ( PythonLibs ${PYTHON_VERSION} REQUIRED )

  SET ( Python_VERSION_MAJOR ${PYTHON_VERSION_MAJOR} )
  SET ( Python_VERSION_MINOR ${PYTHON_VERSION_MINOR} )
  SET ( Python_EXECUTABLE ${PYTHON_EXECUTABLE} )
  SET ( OPENVDB_PYTHON_DEPS ${PYTHON_LIBRARIES} )
  GET_FILENAME_COMPONENT ( Python_LIBRARY_DIRS ${OPENVDB_PYTHON_DEPS} DIRECTORY )

  INCLUDE_DIRECTORIES ( SYSTEM ${PYTHON_INCLUDE_DIR} )
ELSE ()
  # To ensure consistent versions between components Interpreter, Compiler,
  # Development and NumPy, specify all components at the same time when using
  # FindPython
  IF ( USE_NUMPY )
    FIND_PACKAGE ( Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development NumPy )
    LIST ( APPEND OPENVDB_PYTHON_DEPS Python::NumPy )
  ELSE ()
    FIND_PACKAGE ( Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development )
  ENDIF ()
  LIST ( APPEND OPENVDB_PYTHON_DEPS Python::Python )
ENDIF ()

IF ( BOOST_PYTHON_NAMESPACE_VERSIONING )
  SET ( BOOST_PYTHON_LIB "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}" )
ELSE ()
  SET ( BOOST_PYTHON_LIB "python" )
ENDIF ()

FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION}
  REQUIRED COMPONENTS iostreams system ${BOOST_PYTHON_LIB}
)

IF ( UNIX )
  FIND_PACKAGE ( Threads REQUIRED )
ENDIF ()

# Set core deps. Note that the order here is important. If we're building against
# Houdini 17.5 we must include OpenEXR and IlmBase deps first to ensure the normal
# namespaced headers are used over the Houdini 17.5 ones (which will be imported
# through targets like tbb which are still deployed with Houdini)

SET ( OPENVDB_PYTHON_DEPENDANT_LIBS
  ${OPENVDB_LIB}
  Boost::${BOOST_PYTHON_LIB}
  Boost::iostreams
  Boost::system
  IlmBase::Half
  ${OPENVDB_PYTHON_DEPS}
)

IF ( USE_LOG4CPLUS )
  LIST ( APPEND OPENVDB_PYTHON_DEPENDANT_LIBS Log4cplus::log4cplus )
ENDIF ()

IF ( USE_BLOSC )
  LIST ( APPEND OPENVDB_PYTHON_DEPENDANT_LIBS Blosc::blosc )
ENDIF ()

LIST ( APPEND OPENVDB_BINARIES_DEPENDANT_LIBS
  TBB::tbb
  ZLIB::ZLIB
  Threads::Threads
)

##########################################################################

SET ( OPENVDB_PYTHON_MODULE_SOURCE_FILES
  pyFloatGrid.cc
  pyIntGrid.cc
  pyMetadata.cc
  pyPointGrid.cc
  pyOpenVDBModule.cc
  pyPointGrid.cc
  pyTransform.cc
  pyVec3Grid.cc
  )

IF ( NOT DEFINED PYOPENVDB_INSTALL_DIRECTORY )
  SET ( PYOPENVDB_INSTALL_DIRECTORY
    lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}
    CACHE STRING "The directory to install the pyopenvdb.so module."
    )
ENDIF ( NOT DEFINED PYOPENVDB_INSTALL_DIRECTORY )

ADD_LIBRARY ( pyopenvdb SHARED
  ${OPENVDB_PYTHON_MODULE_SOURCE_FILES}
  )

IF ( OPENVDB_PYTHON_WRAP_ALL_GRID_TYPES )
  TARGET_COMPILE_DEFINITIONS ( pyopenvdb PRIVATE "-DPY_OPENVDB_WRAP_ALL_GRID_TYPES" )
ENDIF ()

IF ( USE_LOG4CPLUS )
  TARGET_COMPILE_DEFINITIONS ( pyopenvdb PUBLIC "-DOPENVDB_USE_LOG4CPLUS" )
ENDIF ()

IF ( USE_NUMPY )
  TARGET_COMPILE_DEFINITIONS ( pyopenvdb PUBLIC "-DPY_OPENVDB_USE_NUMPY" )
ENDIF ()

TARGET_LINK_LIBRARIES ( pyopenvdb
  ${OPENVDB_PYTHON_DEPENDANT_LIBS}
  )

SET_TARGET_PROPERTIES ( pyopenvdb PROPERTIES
  PREFIX ""
  )

IF ( OPENVDB_ENABLE_RPATH )
  # @todo There is probably a better way to do this for imported targets
  LIST ( APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
    ${Python_LIBRARY_DIRS}
  )
  IF ( OPENVDB_BUILD_CORE )
    LIST ( APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib )
  ELSE ()
    LIST ( APPEND RPATHS ${OpenVDB_LIBRARY_DIRS} )
  ENDIF ()

  LIST ( REMOVE_DUPLICATES RPATHS )
  SET_TARGET_PROPERTIES ( pyopenvdb
    PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  UNSET ( RPATHS )
ENDIF ()

SET ( PYTHON_PUBLIC_INCLUDE_NAMES
  pyopenvdb.h
  )

INSTALL ( TARGETS
  pyopenvdb
  DESTINATION
  ${PYOPENVDB_INSTALL_DIRECTORY}
  )

INSTALL ( FILES ${PYTHON_PUBLIC_INCLUDE_NAMES} DESTINATION include/openvdb/python )

# pytest

ADD_TEST ( pytest ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/TestOpenVDB.py )
SET_TESTS_PROPERTIES ( pytest PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}" )


