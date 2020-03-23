#!/usr/bin/env bash

set -ex

if [ -d "build" ]; then
    cd build
    ctest -V
fi
