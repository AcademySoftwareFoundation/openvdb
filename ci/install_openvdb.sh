#!/usr/bin/env bash

set -ex

# Download, build and install OpenVDB from source

OPENVDB_VERSION=$1
OPENVDB_ROOT_DIR=$2
if [ -z $OPENVDB_VERSION ]; then
    echo "No OPENVDB version provided for OPENVDB installation"
    exit -1
fi
if [ -z $OPENVDB_ROOT_DIR ]; then
    echo "No installation directory provided for OPENVDB installation"
    exit -1
fi

echo "Downloading OpenVDB $OPENVDB_VERSION src..."

git clone --branch v$OPENVDB_VERSION https://github.com/AcademySoftwareFoundation/openvdb.git openvdb-$OPENVDB_VERSION.src
cd openvdb-$OPENVDB_VERSION.src

echo "Building OpenVDB $OPENVDB_VERSION -> $OPENVDB_ROOT_DIR ..."

mkdir .build
cd .build

cmake \
    -D DISABLE_DEPENDENCY_VERSION_CHECKS=ON \
    -D OPENVDB_BUILD_CORE=ON \
    -D OPENVDB_CORE_STATIC=OFF \
    -D OPENVDB_BUILD_BINARIES=OFF \
    -D OPENVDB_BUILD_PYTHON_MODULE=OFF \
    -D OPENVDB_BUILD_UNITTESTS=OFF \
    -D OPENVDB_BUILD_DOCS=OFF \
    -D OPENVDB_BUILD_HOUDINI_PLUGIN=OFF \
    -D OPENVDB_BUILD_MAYA_PLUGIN=OFF \
    -D CMAKE_INSTALL_PREFIX=$OPENVDB_ROOT_DIR \
    ../

make -j2
make install
