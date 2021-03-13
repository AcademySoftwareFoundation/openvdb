#!/usr/bin/env bash

set -ex

BUILD_TYPE="$1"; shift
CMAKE_EXTRA="$@"

mkdir build
cd build

# See the following regarding boost installation:
#  https://github.com/actions/virtual-environments/issues/687
#  https://github.com/actions/virtual-environments/blob/master/images/win/Windows2019-Readme.md

# print version
cmake --version

cmake \
  -G "Visual Studio 16 2019" -A x64 \
  -DVCPKG_TARGET_TRIPLET="${VCPKG_DEFAULT_TRIPLET}" \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_INSTALLATION_ROOT}\scripts\buildsystems\vcpkg.cmake" \
  -DOPENVDB_BUILD_BINARIES=ON \
  -DOPENVDB_BUILD_VDB_PRINT=ON \
  -DOPENVDB_BUILD_VDB_LOD=ON \
  -DOPENVDB_BUILD_VDB_RENDER=ON \
  -DOPENVDB_BUILD_VDB_VIEW=ON \
  -DOPENVDB_BUILD_UNITTESTS=ON \
  -DBOOST_ROOT="${BOOST_ROOT_1_72_0}" \
  -DBOOST_INCLUDEDIR="${BOOST_ROOT_1_72_0}\boost\include" \
  -DBOOST_LIBRARYDIR="${BOOST_ROOT_1_72_0}\lib" \
  -DMSVC_MP_THREAD_COUNT=4 \
  ${CMAKE_EXTRA} \
  ..

# NOTE: --parallel only effects the number of projects build, not t-units.
# We support this with out own MSVC_MP_THREAD_COUNT option.
# Alternatively it is mentioned that the following should work:
#   cmake --build . --  /p:CL_MPcount=8
# However it does not seem to for our project.
# https://gitlab.kitware.com/cmake/cmake/-/issues/20564

cmake --build . --parallel 4 --config $BUILD_TYPE --target install
