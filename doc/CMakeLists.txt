# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Documentation

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)

# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# If OPENVDB_DOXYGEN_HOUDINI is set, the Houdini documentation is also
# built. You can provide this module a value for Houdini_VERSION which will
# take into account any conditional compiled blocks. Whilst this doesn't
# require you to build the Houdini plugin, building from the root CMake
# with OPENVDB_BUILD_DOCS=ON and either OPENVDB_BUILD_HOUDINI_PLUGIN=ON or
# USE_HOUDINI=ON will define Houdini_VERSION. If Houdini_VERSION is not set,
# all documentation is included.
option(OPENVDB_DOXYGEN_HOUDINI "Build Houdini documentation" ON)

# Add the doxygen command if required - do this here so we guarantee
# not to error on unrelated build issues
find_package(Doxygen REQUIRED)
if(DOXYGEN_VERSION VERSION_LESS MINIMUM_DOXYGEN_VERSION)
  message(WARNING "The doxygen-config doxyfile has been generated with version "
    "\"${MINIMUM_DOXYGEN_VERSION}\". Found Doxygen version \"${DOXYGEN_VERSION}\". "
    "Documentation may contain errors."
  )
endif()

# @todo use cmake doxygen functions available from cmake 3.9
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-config DOXYGEN_CONFIG_CONTENT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config ${DOXYGEN_CONFIG_CONTENT})
if(OPENVDB_DOXYGEN_HOUDINI)
  # Append Houdini-specific settings to the Doxygen config file.
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-config-houdini DOXYGEN_CONFIG_CONTENT)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config ${DOXYGEN_CONFIG_CONTENT})
  if(Houdini_VERSION)
    # Extract the components of the Houdini version number into a list.
    string(REPLACE "." ";" HOUDINI_VERSION_COMPONENTS ${Houdini_VERSION})
    list(GET HOUDINI_VERSION_COMPONENTS 0 HVERS_MAJOR)
    list(GET HOUDINI_VERSION_COMPONENTS 1 HVERS_MINOR)
    list(GET HOUDINI_VERSION_COMPONENTS 2 HVERS_PATCH)
  else()
    set(HVERS_MAJOR 127)  # CMake treats (255 << 24) as a negative number
    set(HVERS_MINOR 0)
    set(HVERS_PATCH 0)
  endif()
  math(EXPR HVERS_INT "(((${HVERS_MAJOR} << 8) + ${HVERS_MINOR}) << 16) + ${HVERS_PATCH}")
  # Append Houdini version number macros to the Doxygen config file so that
  # the generated documentation reflects the API for that version
  # (i.e, taking into account any conditionally compiled blocks).
  set(DOXYGEN_CONFIG_CONTENT
    "PREDEFINED += UT_MAJOR_VERSION_INT=${HVERS_MAJOR}\n"
    "PREDEFINED += UT_MINOR_VERSION_INT=${HVERS_MINOR}\n"
    "PREDEFINED += UT_VERSION_INT=${HVERS_INT}\n\n")
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config
      ${DOXYGEN_CONFIG_CONTENT})
endif()
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config
  "QUIET=YES\nOUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/openvdb/doc\n"
)

add_custom_target(doc ALL
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/openvdb/cmake-doxygen-config
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Generating API documentation with Doxygen" VERBATIM
)

# Suppress "Installing..." messages for all but one of the hundreds of generated files.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openvdb/doc/html/index.html DESTINATION doc/html)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/openvdb/doc/html DESTINATION doc MESSAGE_NEVER)
