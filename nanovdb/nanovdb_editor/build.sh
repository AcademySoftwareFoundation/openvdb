#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Project variables
PROJECT_NAME="nanovdb_editor"
PROJECT_DIR="$(dirname "$(realpath $0)")"
BUILD_DIR="$PROJECT_DIR/build"
SLANG_DEBUG_OUTPUT=OFF
CLEAN_SHADERS=OFF
ASAN_DEBUG_BUILD=OFF

# Parse command line arguments
clean_build=false
release=false
debug=false
verbose=false
python=false
success=true
debug_python=false

usage() {
    echo "Usage: $0 [-x] [-r] [-d] [-v] [-a] [-p]"
    echo "  -x    Perform a clean build"
    echo "  -r    Build in release"
    echo "  -d    Build in debug"
    echo "  -v    Enable CMake verbose output"
    echo "  -s    Compile slang into ASM"
    echo "  -a    Build in debug with AddressSanitizer"
    echo "  -p    Build python module, set also -d to build in debug"
}

while getopts ":xrdvsaph" opt; do
    case ${opt} in
        x) clean_build=true ;;
        r) release=true ;;
        d) debug=true ;;
        v) verbose=true ;;
        s) SLANG_DEBUG_OUTPUT=ON; CLEAN_SHADERS=ON;;
        a) ASAN_DEBUG_BUILD=ON; debug=true ;;
        p) python=true ;;
        h) usage; exit 1 ;;
        \?) echo "Invalid option: $OPTARG" 1>&2; usage; exit 1 ;;
    esac
done

# Shift processed options
shift $((OPTIND -1))

# Set defaults
if [[ $release == false && $debug == false && $python == false ]]; then
    release=true
fi

if $verbose; then
    CMAKE_VERBOSE="--verbose"
else
    CMAKE_VERBOSE=""
fi

if $clean_build; then
    echo "-- Performing a clean build..."
    rm -rf $BUILD_DIR
    echo "-- Deleted $BUILD_DIR"
    CLEAN_SHADERS=ON
fi

function run_build() {
    CONFIG=$1
    BUILD_DIR_CONFIG=$BUILD_DIR/$CONFIG

    echo "-- Building config $CONFIG..."

    # Create build directory
    if [ ! -d $BUILD_DIR_CONFIG ]; then
        mkdir -p $BUILD_DIR_CONFIG
    fi

    # Configure
    cmake -G "Unix Makefiles" $PROJECT_DIR -B $BUILD_DIR_CONFIG \
    -DCMAKE_BUILD_TYPE=$CONFIG \
    -DNANOVDB_EDITOR_CLEAN_SHADERS=$CLEAN_SHADERS \
    -DNANOVDB_EDITOR_SLANG_DEBUG_OUTPUT=$SLANG_DEBUG_OUTPUT \
    -DNANOVDB_EDITOR_ASAN_DEBUG_BUILD=$ASAN_DEBUG_BUILD \
    -DNANOVDB_EDITOR_DEBUG_PYTHON=$debug_python

    # Build
    cmake --build $BUILD_DIR_CONFIG --config $CONFIG $CMAKE_VERBOSE

    # Check for errors
    if [ $? -ne 0 ]; then
        success=false
        echo "Failure while building $CONFIG" >&2
    else
        echo "-- Built config $CONFIG"
    fi
}

echo "-- Building $PROJECT_NAME..."

if $release; then
    run_build "Release"
fi
if $debug; then
    if [ "$release" == "false" ]; then
        debug_python=true
    fi
    run_build "Debug"
fi
if $python; then
    echo "-- Building python module..."

    cd ./pymodule || exit 1

    python -m pip install --upgrade build
    python -m build --wheel

    cd ..
fi

if $success; then
    echo "-- Build of $PROJECT_NAME completed"
else
    echo "Failure while building $PROJECT_NAME" >&2
fi
