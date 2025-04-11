#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# get the build type from the command line
BUILD_TYPE=${1:-install}

# Function to calculate and set the optimal number of parallel build jobs
setup_parallel_build_jobs() {
  # Calculate the optimal number of parallel build jobs based on available RAM
  RAM_GB=$(free -g | awk '/^Mem:/{print $4}')
  if [ -z "$RAM_GB" ]; then
      echo "Error: Unable to determine available RAM"
      exit 1
  fi
  JOB_RAM_GB=2.5

  # Calculate max jobs based on RAM
  RAM_JOBS=$(awk -v ram="$RAM_GB" -v job_ram="$JOB_RAM_GB" 'BEGIN { print int(ram / job_ram) }')

  # Get number of processors
  NPROC=$(nproc)

  # if CMAKE_BUILD_PARALLEL_LEVEL is set, use that
  if [ -n "$CMAKE_BUILD_PARALLEL_LEVEL" ]; then
    echo "Using CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
  else
    # Determine the minimum of RAM-based jobs and processor count
    if [ "$RAM_JOBS" -lt "$NPROC" ]; then
      CMAKE_BUILD_PARALLEL_LEVEL="$RAM_JOBS"
    else
      CMAKE_BUILD_PARALLEL_LEVEL="$NPROC"
    fi

    # Ensure at least 1 job
    if [ "$CMAKE_BUILD_PARALLEL_LEVEL" -lt 1 ]; then
      CMAKE_BUILD_PARALLEL_LEVEL=1
    fi

    echo "Setting CMAKE_BUILD_PARALLEL_LEVEL to $CMAKE_BUILD_PARALLEL_LEVEL based on available RAM to target $JOB_RAM_GB GB per translation unit"
    export CMAKE_BUILD_PARALLEL_LEVEL
  fi
}


## Build the project

# Ensure that the build is done with the conda environment
# Get any additional command line arguments after $1
shift
ADDITIONAL_ARGS="$@"
export PIP_ARGS="-v --no-build-isolation $ADDITIONAL_ARGS"

if [ "$BUILD_TYPE" != "ctest" ]; then
    setup_parallel_build_jobs
fi

# if the user specified 'wheel' as the build type, then we will build the wheel
if [ "$BUILD_TYPE" == "wheel" ]; then
    echo "Build wheel"
    echo "pip wheel . --wheel-dir dist/ $PIP_ARGS"
    pip wheel . --wheel-dir dist/ $PIP_ARGS
elif [ "$BUILD_TYPE" == "install" ]; then
    echo "Build and install package"
    echo "pip install  --force-reinstall $PIP_ARGS ."
    pip install --force-reinstall $PIP_ARGS .
# TODO: Fix editable install
# else
#     echo "Build and install editable package"
#     echo "pip install $PIP_ARGS -e .  "
#     pip install $PIP_ARGS -e .
elif [ "$BUILD_TYPE" == "ctest" ]; then
    BUILD_DIR=$(find build -name tests -type d)
    if [ -z "$BUILD_DIR" ]; then
        echo "Error: Could not find build directory with tests"
        echo "Please enable tests by building with pip argument"
        echo "-C cmake.define.FVDB_BUILD_TESTS=ON"
        exit 1
    fi
    cd "$BUILD_DIR"
    ctest --output-on-failure
else
    echo "Invalid build/run type: $BUILD_TYPE"
    echo "Valid build/run types are: wheel, install, ctest"
    exit 1
fi
