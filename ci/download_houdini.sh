#!/usr/bin/env bash

set -ex

HOUDINI_MAJOR="$1"
GOLD="$2"

pip install --user requests

python ci/download_houdini.py $HOUDINI_MAJOR $GOLD

# create dir hierarchy
mkdir -p hou/bin
mkdir -p hou/houdini
mkdir -p hou/toolkit
mkdir -p hou/dsolib

# unpack hou.tar.gz and cleanup
tar -xzf hou.tar.gz
rm -rf hou.tar.gz
cd houdini*
tar -xzf houdini.tar.gz

# copy required files into hou dir
cp houdini_setup* ../hou/.

# report library names
ls -al dsolib/

# copy required libraries
cp -r toolkit/cmake ../hou/toolkit/.
cp -r toolkit/include ../hou/toolkit/.
cp -r dsolib/libHoudini* ../hou/dsolib/.
cp -r dsolib/libopenvdb_sesi* ../hou/dsolib/.
cp -r dsolib/libblosc* ../hou/dsolib/.
cp -r dsolib/libhboost* ../hou/dsolib/.
cp -r dsolib/libz* ../hou/dsolib/.
cp -r dsolib/libbz2* ../hou/dsolib/.
cp -r dsolib/libtbb* ../hou/dsolib/.
cp -r dsolib/libjemalloc* ../hou/dsolib/.
cp -r dsolib/liblzma* ../hou/dsolib/.
cp -r dsolib/libIex* ../hou/dsolib/.
cp -r dsolib/libImath* ../hou/dsolib/.
cp -r dsolib/libIlmThread* ../hou/dsolib/.

if [ "$HOUDINI_MAJOR" == "19.0" ]; then
    cp -r dsolib/libHalf* ../hou/dsolib/.
    cp -r dsolib/libIlmImf* ../hou/dsolib/.
fi

# write hou into hou.tar.gz and cleanup
cd ..
tar -czvf hou.tar.gz hou

# move hou.tar.gz into hou subdirectory
rm -rf hou/*
mv hou.tar.gz hou

# inspect size of tarball
ls -lart hou/hou.tar.gz
