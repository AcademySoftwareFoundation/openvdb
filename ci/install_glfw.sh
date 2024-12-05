#!/usr/bin/env bash

set -ex
GLFW_VERSION="$1"

git clone https://github.com/glfw/glfw.git
cd glfw

if [ "$GLFW_VERSION" != "latest" ]; then
    git checkout tags/${GLFW_VERSION} -b ${GLFW_VERSION}
fi

mkdir build
cd build

cmake ..

make -j8
make install
