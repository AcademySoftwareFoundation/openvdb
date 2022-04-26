#!/usr/bin/env bash
set -e

# Various tests to test the FindOpenVDB CMake modules and
# general VDB installation

# 1) Test basic CMakeLists is able to build vdb_print with
# the expected VDB installation

cmakelists="
cmake_minimum_required(VERSION 3.15)
project(TestInstall LANGUAGES CXX)
find_package(OpenVDB REQUIRED COMPONENTS openvdb)
add_executable(test_vdb_print \"../openvdb_cmd/vdb_print/main.cc\")
target_link_libraries(test_vdb_print OpenVDB::openvdb)
"
mkdir tmp
cd tmp
echo -e "$cmakelists" > CMakeLists.txt
cmake -DCMAKE_MODULE_PATH=/usr/local/lib64/cmake/OpenVDB/ .
cmake --build . --config Release --target test_vdb_print
