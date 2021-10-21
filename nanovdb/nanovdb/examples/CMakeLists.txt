# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Examples

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(NanoVDBExamples LANGUAGES CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "---------- Configuring NanoVDB Examples ------------")
message(STATUS "----------------------------------------------------")

###############################################################################

if(WIN32)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

#set(NANOVDB_INSTALL_EXAMPLEDIR ${CMAKE_INSTALL_PREFIX}/bin/nanovdb_examples)
set(NANOVDB_LOCAL_EXAMPLEDIR ${PROJECT_BINARY_DIR}/examples)

# -----------------------------------------------------------------------------`
# helper function
# -----------------------------------------------------------------------------`
function(NANOVDB_EXAMPLE)
  set(options OPENVDB)
  set(oneValueArgs NAME)
  set(multiValueArgs)
  cmake_parse_arguments(NANOVDB_EXAMPLE
    "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(EXAMPLE_NAME ${NANOVDB_EXAMPLE_NAME})
  set(EXAMPLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}")

  file(GLOB EXAMPLE_HEADER_FILES
    ${EXAMPLE_ROOT}/*.h
    ${EXAMPLE_ROOT}/*.inl
    ${EXAMPLE_ROOT}/*.hh
    ${EXAMPLE_ROOT}/*.hpp
    ${EXAMPLE_ROOT}/*.cuh)

  file(GLOB EXAMPLE_SOURCE_FILES
    ${EXAMPLE_ROOT}/*.c
    ${EXAMPLE_ROOT}/*.cpp
    ${EXAMPLE_ROOT}/*.cc)

  file(GLOB EXAMPLE_SOURCE_CUDA_FILES ${EXAMPLE_ROOT}/*.cu)

  # configure CUDA
  if(NOT NANOVDB_USE_CUDA AND EXAMPLE_SOURCE_CUDA_FILES)
    message(WARNING " - CUDA required to build ${EXAMPLE_NAME}. Skipping.")
    return()
  endif()

  if(NANOVDB_EXAMPLE_OPENVDB AND NOT NANOVDB_USE_OPENVDB)
    message(WARNING " - OpenVDB required to build ${EXAMPLE_NAME}. Skipping.")
    return()
  endif()

  add_executable(${EXAMPLE_NAME}
    ${EXAMPLE_SOURCE_FILES}
    ${EXAMPLE_SOURCE_CUDA_FILES}
    ${EXAMPLE_HEADER_FILES})

  target_include_directories(${EXAMPLE_NAME} PUBLIC
    ex_util
    ${CUDA_INCLUDE_DIRECTORY})

  target_link_libraries(${EXAMPLE_NAME} PRIVATE nanovdb ${CUDA_LIBRARIES})

  install(TARGETS ${EXAMPLE_NAME} DESTINATION ${NANOVDB_LOCAL_EXAMPLEDIR})
endfunction()

# -----------------------------------------------------------------------

if(NANOVDB_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

nanovdb_example(NAME "ex_make_custom_nanovdb")
nanovdb_example(NAME "ex_make_funny_nanovdb")
nanovdb_example(NAME "ex_make_typed_grids")
nanovdb_example(NAME "ex_make_nanovdb_sphere")
nanovdb_example(NAME "ex_write_nanovdb_grids")
nanovdb_example(NAME "ex_openvdb_to_nanovdb" OPENVDB)
nanovdb_example(NAME "ex_openvdb_to_nanovdb_accessor" OPENVDB)
nanovdb_example(NAME "ex_openvdb_to_nanovdb_cuda" OPENVDB)
nanovdb_example(NAME "ex_read_nanovdb_sphere")
nanovdb_example(NAME "ex_read_nanovdb_sphere_accessor")
nanovdb_example(NAME "ex_read_nanovdb_sphere_accessor_cuda")
nanovdb_example(NAME "ex_modify_nanovdb_thrust")
nanovdb_example(NAME "ex_map_pool_buffer")
nanovdb_example(NAME "ex_bump_pool_buffer")
nanovdb_example(NAME "ex_collide_level_set")
nanovdb_example(NAME "ex_raytrace_fog_volume")
nanovdb_example(NAME "ex_raytrace_level_set")

if(NANOVDB_USE_MAGICAVOXEL)
  nanovdb_example(NAME "ex_vox_to_nanovdb")
endif()
# add_subdirectory(ex_sph_benchmark)

# many of the sample projects depend on a data directory. This allows Debug
# launching from the cmake binary working directory.
file(MAKE_DIRECTORY "${NANOVDB_LOCAL_EXAMPLEDIR}/data")
