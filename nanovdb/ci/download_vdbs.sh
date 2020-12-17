#!/usr/bin/env bash

set -ex

mkdir -p /repo/vdbs
pushd /repo/vdbs

wget https://artifacts.aswf.io/io/aswf/openvdb/models/bunny.vdb/1.0.0/bunny.vdb-1.0.0.zip
if [ -f bunny.vdb-1.0.0.zip ]; then
    unzip bunny.vdb-1.0.0.zip
    rm -f bunny.vdb-1.0.0.zip
fi

wget https://artifacts.aswf.io/io/aswf/openvdb/models/smoke1.vdb/1.0.0/smoke1.vdb-1.0.0.zip
if [ -f smoke1.vdb-1.0.0.zip ]; then
    unzip smoke1.vdb-1.0.0.zip
    rm -f smoke1.vdb-1.0.0.zip
fi

wget https://artifacts.aswf.io/io/aswf/openvdb/models/sphere_points.vdb/1.0.0/sphere_points.vdb-1.0.0.zip
if [ -f sphere_points.vdb-1.0.0.zip ]; then
    unzip sphere_points.vdb-1.0.0.zip
    rm -f sphere_points.vdb-1.0.0.zip
fi

popd

