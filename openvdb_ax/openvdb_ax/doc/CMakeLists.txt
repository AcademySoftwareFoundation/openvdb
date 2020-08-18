# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB AX Doxygen

#]=======================================================================]

project(OpenVDBAXDocs LANGUAGES NONE)

find_package(Doxygen REQUIRED)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-config DOXYGEN_CONFIG_CONTENT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config ${DOXYGEN_CONFIG_CONTENT})
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config
  "QUIET=YES\nOUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}\n"
)

add_custom_target(doc ALL
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Generating API documentation with Doxygen" VERBATIM
  )

# Suppress "Installing..." messages for all but one of the hundreds of generated files.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/html/index.html DESTINATION doc/html)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION doc MESSAGE_NEVER)
