cmake_minimum_required(VERSION 3.20)
project(teachme_tests LANGUAGES CXX CUDA)

# Compile-only gate: object libraries compile each generated translation unit
# to an object file without linking. That is exactly what we want — catch API
# errors at compile time; the fixture's extern stand-ins never need to resolve.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 6000 Ada; any arch suffices to compile
endif()

# NanoVDB headers pull in CUDA lambdas / relaxed-constexpr code paths.
# diag 177 = "function declared but never referenced": every wrapped block
# becomes an unreferenced static function on purpose, so silence it.
string(APPEND CMAKE_CUDA_FLAGS
  " --extended-lambda --expt-relaxed-constexpr --diag-suppress 177")

# NanoVDB include root: this file is doc/nanovdb/TEACHME/test/CMakeLists.txt, so
# the repo root is four dirs up. Headers are at nanovdb/nanovdb/NanoVDB.h, so the
# include root is the repo's `nanovdb/` subdir (a sibling of `doc/`).
get_filename_component(_teachme_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)  # TEACHME
get_filename_component(_docnv_dir   "${_teachme_dir}" DIRECTORY)              # doc/nanovdb
get_filename_component(_doc_dir     "${_docnv_dir}" DIRECTORY)                # doc
get_filename_component(REPO_ROOT    "${_doc_dir}" DIRECTORY)                  # repo root
set(NANOVDB_INCLUDE_ROOT "${REPO_ROOT}/nanovdb")
set(OPENVDB_INCLUDE_ROOT "${REPO_ROOT}/openvdb")

message(STATUS "NanoVDB include root: ${NANOVDB_INCLUDE_ROOT}")

# Generated TUs + manifest live in ${GEN_DIR} (passed in, defaults to ./gen).
if(NOT DEFINED GEN_DIR)
  set(GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gen")
endif()

set(_manifest "${GEN_DIR}/manifest.txt")
if(NOT EXISTS "${_manifest}")
  message(FATAL_ERROR "No manifest at ${_manifest}. Run the extractor first "
                      "(see run.sh).")
endif()

file(STRINGS "${_manifest}" _lines)
set(_core_srcs "")
set(_ovdb_srcs "")
foreach(_line IN LISTS _lines)
  if(_line STREQUAL "")
    continue()
  endif()
  separate_arguments(_parts UNIX_COMMAND "${_line}")
  list(GET _parts 0 _stem)
  list(GET _parts 1 _needs_ovdb)
  set(_src "${GEN_DIR}/${_stem}.cu")
  if(_needs_ovdb STREQUAL "1")
    list(APPEND _ovdb_srcs "${_src}")
  else()
    list(APPEND _core_srcs "${_src}")
  endif()
endforeach()

# Core (NanoVDB-only) translation units.
if(_core_srcs)
  add_library(teachme_blocks OBJECT ${_core_srcs})
  target_include_directories(teachme_blocks PRIVATE
    "${NANOVDB_INCLUDE_ROOT}" "${GEN_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  list(LENGTH _core_srcs _n_core)
  message(STATUS "teachme_blocks: ${_n_core} core TU(s)")
endif()

# OpenVDB-dependent translation units — only if OpenVDB is available.
if(_ovdb_srcs)
  find_package(OpenVDB QUIET)
  if(OpenVDB_FOUND)
    add_library(teachme_blocks_openvdb OBJECT ${_ovdb_srcs})
    target_compile_definitions(teachme_blocks_openvdb PRIVATE NANOVDB_USE_OPENVDB)
    target_include_directories(teachme_blocks_openvdb PRIVATE
      "${NANOVDB_INCLUDE_ROOT}" "${OPENVDB_INCLUDE_ROOT}"
      "${GEN_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(teachme_blocks_openvdb PRIVATE OpenVDB::openvdb)
    list(LENGTH _ovdb_srcs _n_ovdb)
    message(STATUS "teachme_blocks_openvdb: ${_n_ovdb} OpenVDB TU(s)")
  else()
    list(LENGTH _ovdb_srcs _n_ovdb)
    message(WARNING "OpenVDB not found — skipping ${_n_ovdb} OpenVDB TU(s). "
                    "These blocks are reference-checked but not compiled.")
  endif()
endif()
