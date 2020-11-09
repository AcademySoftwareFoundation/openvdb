#!/usr/bin/env bash

set -ex

COMPILER="$1"
RELEASE="$2"
EXTRAS="$3"

# DebugNoInfo is a custom CMAKE_BUILD_TYPE - no optimizations, no symbols, asserts enabled

if [ -d "hou" ]; then
    cd hou
    source houdini_setup_bash
    cd -

    mkdir build
    cd build
    cmake \
        -DCMAKE_CXX_FLAGS_DebugNoInfo="" \
        -DCMAKE_CXX_COMPILER=${COMPILER} \
        -DCMAKE_BUILD_TYPE=${RELEASE} \
        -DOPENVDB_CXX_STRICT=ON \
        -DOPENVDB_USE_DEPRECATED_ABI_5=ON \
        -DOPENVDB_BUILD_HOUDINI_PLUGIN=ON \
        -DOPENVDB_BUILD_HOUDINI_ABITESTS=ON \
        -DOPENVDB_BUILD_BINARIES=${EXTRAS} \
        -DOPENVDB_BUILD_PYTHON_MODULE=${EXTRAS} \
        -DOPENVDB_BUILD_UNITTESTS=${EXTRAS} \
        -DOPENVDB_HOUDINI_INSTALL_PREFIX=/tmp \
         ..

    # Can only build using one thread with GCC due to memory constraints
    if [ "$COMPILER" = "clang++" ]; then
        make -j2
    else
        make
    fi
fi
