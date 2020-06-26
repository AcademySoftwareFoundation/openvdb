#!/usr/bin/env bash

set -e

if [ ! -f "hou/hou.tar.gz" ]; then
    echo "Could not find Houdini download"
    exit 1
fi
