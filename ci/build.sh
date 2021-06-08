#!/usr/bin/env bash

set -ex

# print versions
bash --version
if [ ! -z "$CXX" ]; then $CXX -v; fi
cmake --version

################################################

BUILD_TYPE="$1"; shift
# Command seperated list of components i.e. "core,ax,bin"
IN_COMPONENTS="$1"; shift
CMAKE_EXTRA="$@"

################################################
#
# Select components to build

# Available components. If a component is not provided it is
# explicitly set to OFF.
declare -A COMPONENTS
COMPONENTS['core']='OPENVDB_BUILD_CORE'
COMPONENTS['python']='OPENVDB_BUILD_PYTHON_MODULE'
COMPONENTS['test']='OPENVDB_BUILD_UNITTESTS'
COMPONENTS['bin']='OPENVDB_BUILD_BINARIES'
COMPONENTS['doc']='OPENVDB_BUILD_DOCS'
COMPONENTS['axcore']='OPENVDB_BUILD_AX'
COMPONENTS['axgr']='OPENVDB_BUILD_AX_GRAMMAR'
COMPONENTS['axbin']='OPENVDB_BUILD_AX_BINARIES'
COMPONENTS['axtest']='OPENVDB_BUILD_AX_UNITTESTS'

# Check inputs
IFS=', ' read -r -a IN_COMPONENTS <<< "$IN_COMPONENTS"
for comp in "${IN_COMPONENTS[@]}"; do
    if [ -z ${COMPONENTS[$comp]} ]; then
        echo "Invalid component passed to build \"$comp\""; exit -1
    fi
done

# Build CMake command
for comp in "${!COMPONENTS[@]}"; do
    found=false
    for in in "${IN_COMPONENTS[@]}"; do
        if [[ $comp == "$in" ]]; then
            found=true; break
        fi
    done

    if $found; then
        CMAKE_EXTRA+=" -D${COMPONENTS[$comp]}=ON "
    else
        CMAKE_EXTRA+=" -D${COMPONENTS[$comp]}=OFF "
    fi
done

#
################################################

# github actions runners have 2 threads
# https://help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners
export CMAKE_BUILD_PARALLEL_LEVEL=2

mkdir -p build
cd build

# Note: all sub binary options are always on and can be toggles with
# OPENVDB_BUILD_BINARIES=ON/OFF
# Using CMAKE_VERBOSE_MAKEFILE rather than `cmake --verbose` to support older
# versions of CMake. Always run verbose make to have the full logs available
cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DOPENVDB_USE_DEPRECATED_ABI_6=ON \
    -DOPENVDB_USE_FUTURE_ABI_9=ON \
    -DOPENVDB_BUILD_VDB_PRINT=ON \
    -DOPENVDB_BUILD_VDB_LOD=ON \
    -DOPENVDB_BUILD_VDB_RENDER=ON \
    -DOPENVDB_BUILD_VDB_VIEW=ON \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    ${CMAKE_EXTRA} \
    ..

cmake --build . --config $BUILD_TYPE --target install
