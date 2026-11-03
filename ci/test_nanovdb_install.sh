#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
set -e

# Tests the installed NanoVDB CMake package config by building nanovdb_print
# against a NanoVDB installation.
#
# Usage: test_nanovdb_install.sh <install prefix>

prefix=$(realpath "$1")
root=$(realpath "$(dirname "$0")/..")

cmakelists="
cmake_minimum_required(VERSION 3.24)
project(TestNanoVDBInstall LANGUAGES CXX)
find_package(NanoVDB CONFIG REQUIRED)
add_executable(test_nanovdb_print \"$root/nanovdb/nanovdb/cmd/print/nanovdb_print.cc\")
target_link_libraries(test_nanovdb_print NanoVDB::nanovdb)
"
mkdir -p tmp_nanovdb_install
cd tmp_nanovdb_install
echo -e "$cmakelists" > CMakeLists.txt
cmake -DCMAKE_PREFIX_PATH="$prefix" .
cmake --build . --config Release --target test_nanovdb_print
