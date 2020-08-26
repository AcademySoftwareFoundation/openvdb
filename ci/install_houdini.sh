#!/usr/bin/env bash

set -e

if [ -d "hou" ]; then
    # move hou tarball into top-level and untar
    cp hou/hou.tar.gz .
    tar -xzf hou.tar.gz
fi
