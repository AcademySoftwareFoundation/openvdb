cmake_minimum_required(VERSION 3.20)
project(nanovdb_capstone LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # emit build/compile_commands.json
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 6000 Ada; override with -DCMAKE_CUDA_ARCHITECTURES=...
endif()

# NanoVDB headers pull in CUDA lambdas / relaxed-constexpr paths.
string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda --expt-relaxed-constexpr")

# This file is at doc/nanovdb/TEACHME/capstone/CMakeLists.txt; the NanoVDB
# include root (the dir containing the nanovdb/ header subdir) is the repo's
# `nanovdb/` subdir — repo root is four dirs up.
get_filename_component(_teachme "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY) # TEACHME
get_filename_component(_docnv   "${_teachme}" DIRECTORY)                 # doc/nanovdb
get_filename_component(_doc     "${_docnv}" DIRECTORY)                   # doc
get_filename_component(REPO_ROOT "${_doc}" DIRECTORY)                    # repo root
set(NANOVDB_INCLUDE_ROOT "${REPO_ROOT}/nanovdb")
message(STATUS "NanoVDB include root: ${NANOVDB_INCLUDE_ROOT}")

add_executable(raymarch raymarch.cu)
target_include_directories(raymarch PRIVATE "${NANOVDB_INCLUDE_ROOT}")
# Binary lands in the build dir (conventional). raymarch writes sphere.ppm to
# the current working directory, so run it from wherever you want the output.
