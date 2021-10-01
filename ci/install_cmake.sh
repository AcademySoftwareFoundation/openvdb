#!/usr/bin/env bash

set -ex

CMAKE_VERSION="$1"
CMAKE_PACKAGE=cmake-${CMAKE_VERSION}-Linux-x86_64

wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_PACKAGE}.tar.gz

tar -xf ${CMAKE_PACKAGE}.tar.gz
cp ${CMAKE_PACKAGE}/bin/* /usr/local/bin/
cp -r ${CMAKE_PACKAGE}/share/* /usr/local/share/

rm -rf ${CMAKE_PACKAGE}
rm -rf ${CMAKE_PACKAGE}.tar.gz
