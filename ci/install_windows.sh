#!/usr/bin/env bash

set -x
set -e

# Required dependencies
VCPKG_INSTALL_CMD="vcpkg install
    zlib
    libpng
    openexr
    tbb
    gtest
    cppunit
    blosc
    glfw3
    glew
    python3
    jemalloc
    boost-iostreams
    boost-interprocess
    boost-algorithm
    pybind11
    lz4"

# if VCPKG_DEFAULT_TRIPLET ends with -static, then add ':x64-windows-static' to all the dependencies
if [[ $VCPKG_DEFAULT_TRIPLET == *"-static" ]]; then
    VCPKG_INSTALL_CMD="$VCPKG_INSTALL_CMD:x64-windows-static"
fi

VCPKG_INSTALL_CMD="$VCPKG_INSTALL_CMD  --clean-after-build"

# Update vcpkg
vcpkg update

# Allow the vcpkg command to fail once so we can retry with the latest
set +e
$VCPKG_INSTALL_CMD
STATUS=$?

# Subsequent commands cannot fail
set -x

if [ $STATUS -ne 0 ]; then
  # Try once more with latest ports
  echo "vcpkg install failed, retrying with latest ports..."
  cd $VCPKG_INSTALLATION_ROOT && git pull && cd-
  vcpkg update
  $VCPKG_INSTALL_CMD
fi

echo "vcpkg install completed successfully"
