# Copyright Contributors to the OpenVDB Project SPDX-License-Identifier: MPL-2.0

# -----------------------------------------------------------------------
if(WIN32)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

# ---------------------------------------------------------------
function(_ADD_EXAMPLE)
  set(options "")
  set(oneValueArgs NAME PATH GENERATED GENPATHS)
  set(multiValueArgs OPTIONS COMPILE_FLAGS CUDA_FLAGS_DEBUG SOURCES LIBS
                     INCLUDE)
  cmake_parse_arguments(_ADD_EXAMPLE "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(EXAMPLE_NAME ${_ADD_EXAMPLE_NAME})
  set(EXAMPLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}")

  # many of the sample projects depend on a data directory. This allows Debug
  # launching from the cmake binary working directory.
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")

  file(
    GLOB
    EXAMPLE_HEADER_FILES
    ${EXAMPLE_ROOT}/*.h
    ${EXAMPLE_ROOT}/*.inl
    ${EXAMPLE_ROOT}/*.hh
    ${EXAMPLE_ROOT}/*.hpp
    ${EXAMPLE_ROOT}/*.cuh)

  file(GLOB EXAMPLE_SOURCE_FILES ${EXAMPLE_ROOT}/*.c ${EXAMPLE_ROOT}/*.cpp
       ${EXAMPLE_ROOT}/*.cc)

  file(GLOB EXAMPLE_SOURCE_CUDA_FILES ${EXAMPLE_ROOT}/*.cu)

  # message( STATUS "EXAMPLE_NAME = ${EXAMPLE_NAME}")
  message(STATUS "EXAMPLE_ROOT = ${EXAMPLE_ROOT}")

  if(EXAMPLE_SOURCE_CUDA_FILES AND NOT CUDA_FOUND)
    message(WARNING " - CUDA required to build ${EXAMPLE_NAME}. Skipping.")
    return()
  endif()

  add_executable(
    ${EXAMPLE_NAME} ${EXAMPLE_SOURCE_FILES} ${EXAMPLE_SOURCE_CUDA_FILES}
                    ${EXAMPLE_HEADER_FILES})

  target_include_directories(
    ${EXAMPLE_NAME} PUBLIC ex_util ${TBB_INCLUDE_DIRECTORY}
                           ${CUDA_INCLUDE_DIRECTORY} ${NANOVDB_OGT_INCLUDE_DIRECTORY})

  target_link_libraries(
    ${EXAMPLE_NAME}
    PRIVATE libnanovdb
            ${CUDA_LIBRARIES}
            ${NANOVDB_OPENVDB} # optional
            ${NANOVDB_TBB} # optional
            ${NANOVDB_ZLIB} # optional
            ${NANOVDB_BLOSC} # optional
  )

  target_compile_definitions(
    ${EXAMPLE_NAME}
    PRIVATE "${NANOVDB_USE_OPENVDB_FLAG}" "${NANOVDB_USE_CUDA_FLAG}" "${NANOVDB_USE_MAGICAVOXEL_FLAG}"
            "${NANOVDB_USE_TBB_FLAG}" "${NANOVDB_USE_BLOSC_FLAG}"
            "${NANOVDB_USE_ZLIB_FLAG}")

  install(TARGETS ${EXAMPLE_NAME} DESTINATION examples)

endfunction()

# ---------------------------------------------------------------
function(_ADD_EXAMPLE_OPENVDB)
  set(options "")
  set(oneValueArgs NAME PATH GENERATED GENPATHS)
  set(multiValueArgs OPTIONS COMPILE_FLAGS CUDA_FLAGS_DEBUG SOURCES LIBS
                     INCLUDE)
  cmake_parse_arguments(_ADD_EXAMPLE "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(EXAMPLE_NAME ${_ADD_EXAMPLE_NAME})
  set(EXAMPLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}")

  # many of the sample projects depend on a data directory. This allows Debug
  # launching from the cmake binary working directory.
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")

  file(
    GLOB
    EXAMPLE_HEADER_FILES
    ${EXAMPLE_ROOT}/*.h
    ${EXAMPLE_ROOT}/*.inl
    ${EXAMPLE_ROOT}/*.hh
    ${EXAMPLE_ROOT}/*.hpp
    ${EXAMPLE_ROOT}/*.cuh)

  file(GLOB EXAMPLE_SOURCE_FILES ${EXAMPLE_ROOT}/*.c ${EXAMPLE_ROOT}/*.cpp
       ${EXAMPLE_ROOT}/*.cc)

  file(GLOB EXAMPLE_SOURCE_CUDA_FILES ${EXAMPLE_ROOT}/*.cu)

  # message( STATUS "EXAMPLE_NAME = ${EXAMPLE_NAME}")
  message(STATUS "EXAMPLE_ROOT = ${EXAMPLE_ROOT}")

  if(EXAMPLE_SOURCE_CUDA_FILES AND NOT CUDA_FOUND)
    message(WARNING " - CUDA required to build ${EXAMPLE_NAME}. Skipping.")
    return()
  endif()

  if(NOT (OpenVDB_FOUND))
    message(WARNING " - OpenVDB required to build ${EXAMPLE_NAME}. Skipping.")
    return()
  endif()

  add_executable(
    ${EXAMPLE_NAME} ${EXAMPLE_SOURCE_FILES} ${EXAMPLE_SOURCE_CUDA_FILES}
                    ${EXAMPLE_HEADER_FILES})

  target_include_directories(
    ${EXAMPLE_NAME} PUBLIC ex_util ${OpenVDB_INCLUDE_DIRS}
                           ${TBB_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRECTORY} ${NANOVDB_OGT_INCLUDE_DIRECTORY})

  target_link_libraries(
    ${EXAMPLE_NAME}
    PRIVATE libnanovdb
            ${NANOVDB_OPENVDB} # optional
            ${NANOVDB_TBB} # optional
            ${NANOVDB_ZLIB} # optional
            ${NANOVDB_BLOSC} # optional
  )

  target_compile_definitions(
    ${EXAMPLE_NAME}
    PRIVATE "${NANOVDB_USE_OPENVDB_FLAG}" "${NANOVDB_USE_CUDA_FLAG}" "${NANOVDB_USE_MAGICAVOXEL_FLAG}"
            "${NANOVDB_USE_TBB_FLAG}" "${NANOVDB_USE_BLOSC_FLAG}"
            "${NANOVDB_USE_ZLIB_FLAG}")

  install(TARGETS ${EXAMPLE_NAME} DESTINATION examples)

endfunction()

# -----------------------------------------------------------------------
if(NANOVDB_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()
_add_example(NAME "ex_make_custom_nanovdb")
_add_example(NAME "ex_make_funny_nanovdb")
_add_example(NAME "ex_make_typed_grids")
_add_example(NAME "ex_make_nanovdb_sphere")
_add_example(NAME "ex_write_nanovdb_grids")
_add_example_openvdb(NAME "ex_openvdb_to_nanovdb")
_add_example_openvdb(NAME "ex_openvdb_to_nanovdb_accessor")
_add_example_openvdb(NAME "ex_openvdb_to_nanovdb_cuda")
_add_example(NAME "ex_read_nanovdb_sphere")
_add_example(NAME "ex_read_nanovdb_sphere_accessor")
_add_example(NAME "ex_read_nanovdb_sphere_accessor_cuda")
_add_example(NAME "ex_modify_nanovdb_thrust")
_add_example(NAME "ex_map_pool_buffer")
_add_example(NAME "ex_bump_pool_buffer")
_add_example(NAME "ex_collide_level_set")
_add_example(NAME "ex_raytrace_fog_volume")
_add_example(NAME "ex_raytrace_level_set")
if(NANOVDB_USE_MAGICAVOXEL)
_add_example(NAME "ex_vox_to_nanovdb")
endif()
# add_subdirectory(ex_sph_benchmark)

# ---------------------------------------------------------------
