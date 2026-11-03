#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
set -e

# Configure-only regression tests for how NanoVDB finds CCCL and publishes
# its include directory, both from a CUDA-enabled install and in-tree.
# A fake CCCL package stands in for CCCL releases of any version.
#
# Usage: test_nanovdb_config.sh <install prefix>

prefix=$(realpath "$1")
root=$(realpath "$(dirname "$0")/..")
work=$(realpath -m tmp_nanovdb_config)
rm -rf "$work"
mkdir -p "$work"

# Writes a CCCL package of the given version that, like any package config
# using @PACKAGE_INIT@, overwrites PACKAGE_PREFIX_DIR. CMake 3.30 and newer
# restore PACKAGE_PREFIX_DIR after find_package, so the include_dir case only
# catches that problem on older CMake.
fake_cccl() {
  local dir="$work/fake_cccl_$1"
  mkdir -p "$dir"
  cat > "$dir/cccl-config.cmake" <<EOF
set(PACKAGE_PREFIX_DIR "$dir")
if(NOT TARGET CCCL::CCCL)
  add_library(CCCL::CCCL INTERFACE IMPORTED)
endif()
set(CCCL_VERSION $1)
set_property(GLOBAL PROPERTY FAKE_CCCL_USED TRUE)
EOF
  cat > "$dir/cccl-config-version.cmake" <<EOF
set(PACKAGE_VERSION $1)
set(PACKAGE_VERSION_COMPATIBLE TRUE)
EOF
  echo "$dir"
}
cccl_old=$(fake_cccl 2.7.0)
cccl_new=$(fake_cccl 2.9.0)

failures=0

# run_case <name> <pass|fail> <cmake body> [cmake args...]
run_case() {
  local name=$1 expect=$2 body=$3
  shift 3
  local dir="$work/$name"
  mkdir -p "$dir"
  printf 'cmake_minimum_required(VERSION 3.24)\nproject(%s LANGUAGES CXX)\n%s\n' "$name" "$body" > "$dir/CMakeLists.txt"
  local result=pass
  cmake -S "$dir" -B "$dir/build" -DCMAKE_PREFIX_PATH="$prefix" "$@" > "$dir/log.txt" 2>&1 || result=fail
  # A failing case must fail on the CCCL minimum, not on something else
  if [ "$result" = fail ] && ! grep -q "requires CCCL" "$dir/log.txt"; then
    result=error
  fi
  if [ "$result" = "$expect" ]; then
    echo "ok    $name"
  else
    echo "FAIL  $name (expected $expect, got $result)"
    cat "$dir/log.txt"
    failures=$((failures + 1))
  fi
}

intree="
set(NANOVDB_BUILD_TOOLS OFF CACHE BOOL \"\")
set(NANOVDB_USE_CUDA ON CACHE BOOL \"\")
add_subdirectory(\"$root/nanovdb/nanovdb\" nanovdb)"

run_case include_dir pass "
find_package(NanoVDB CONFIG REQUIRED)
get_target_property(dir NanoVDB::nanovdb NANOVDB_INCLUDE_DIR)
if(NOT dir STREQUAL \"$prefix/include\")
  message(FATAL_ERROR \"NANOVDB_INCLUDE_DIR is \${dir}\")
endif()" -DCCCL_DIR="$cccl_new"

run_case installed_too_old fail "
find_package(NanoVDB CONFIG REQUIRED)" -DCCCL_DIR="$cccl_old"

run_case preloaded_too_old fail "
find_package(CCCL CONFIG REQUIRED PATHS \"$cccl_old\" NO_DEFAULT_PATH)
find_package(NanoVDB CONFIG REQUIRED)"

run_case optional_then_required fail "
find_package(NanoVDB CONFIG QUIET)
find_package(NanoVDB CONFIG REQUIRED)" -DCCCL_DIR="$cccl_old"

run_case intree_preloaded_too_old fail "
find_package(CCCL CONFIG REQUIRED PATHS \"$cccl_old\" NO_DEFAULT_PATH)
$intree"

run_case intree_forced_fetch pass "
$intree
get_property(used GLOBAL PROPERTY FAKE_CCCL_USED)
if(used OR NOT TARGET CCCL::CCCL)
  message(FATAL_ERROR \"NANOVDB_FETCH_CCCL did not clone CCCL\")
endif()" -DNANOVDB_FETCH_CCCL=ON -DCPM_USE_LOCAL_PACKAGES=ON -DCCCL_DIR="$cccl_new"

if [ $failures -ne 0 ]; then
  echo "$failures NanoVDB config test(s) failed"
  exit 1
fi
