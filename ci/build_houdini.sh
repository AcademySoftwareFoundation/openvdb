#!/usr/bin/env bash

set -ex

RELEASE="$1"; shift
EXTRAS="$1"; shift
CMAKE_EXTRA="$@"

# print versions
bash --version
$CXX -v
cmake --version

# DebugNoInfo is a custom CMAKE_BUILD_TYPE - no optimizations, no symbols, asserts enabled

if [ -d "hou" ]; then
    cd hou
    source houdini_setup_bash
    cd -

    mkdir build
    cd build
    cmake \
        -DCMAKE_CXX_FLAGS_DebugNoInfo="" \
        -DCMAKE_BUILD_TYPE=${RELEASE} \
        -DOPENVDB_CXX_STRICT=ON \
        -DOPENVDB_USE_DEPRECATED_ABI_6=ON \
        -DOPENVDB_BUILD_HOUDINI_PLUGIN=ON \
        -DOPENVDB_BUILD_HOUDINI_ABITESTS=ON \
        -DOPENVDB_BUILD_AX=${EXTRAS} \
        -DOPENVDB_BUILD_AX_BINARIES=${EXTRAS} \
        -DOPENVDB_BUILD_AX_UNITTESTS=${EXTRAS} \
        -DOPENVDB_BUILD_BINARIES=${EXTRAS} \
        -DOPENVDB_BUILD_PYTHON_MODULE=${EXTRAS} \
        -DOPENVDB_BUILD_UNITTESTS=${EXTRAS} \
        -DOPENVDB_HOUDINI_INSTALL_PREFIX=/tmp \
        ${CMAKE_EXTRA} \
         ..

    # Can only build using one thread with GCC due to memory constraints
    if [ "$CXX" = "clang++" ]; then
        make -j2
    else
        make
    fi
fi
