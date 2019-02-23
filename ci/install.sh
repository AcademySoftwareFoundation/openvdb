#!/usr/bin/env bash
#
# CircleCI install script for OpenVDB
#
# HOUDINI_MAJOR (16.0/16.5/17.0) - the major version of Houdini
# COMPILER (g++/clang++) - build with GCC or Clang
#
# Author: Dan Bailey

set -ex

HOUDINI_MAJOR="$1"
COMPILER="$2"

apt-get update

apt-get install -y zlib1g-dev
apt-get install -y wget
apt-get install -y unzip
apt-get install -y curl
apt-get install -y cmake
apt-get install -y g++
apt-get install -y clang
apt-get install -y llvm
apt-get install -y libglu1-mesa-dev
apt-get install -y libgl1-mesa-dev

if [ "$HOUDINI_MAJOR" = "none" ]; then
    apt-get install -y libboost-iostreams-dev
    apt-get install -y libboost-system-dev
    apt-get install -y libboost-thread-dev
    apt-get install -y libtbb-dev
    apt-get install -y libilmbase-dev
    apt-get install -y libopenexr-dev
    apt-get install -y libcppunit-dev
    apt-get install -y liblog4cplus-dev
    apt-get install -y libglfw3-dev
    apt-get install -y python-dev
    apt-get install -y libboost-python-dev
    apt-get install -y python-numpy
    apt-get install -y python-epydoc
    apt-get install -y doxygen
    # download and build Blosc 1.5.0
    if [ ! -d $HOME/blosc/lib ]; then
        wget https://github.com/Blosc/c-blosc/archive/v1.5.0.zip
        unzip v1.5.0.zip
        cd c-blosc-1.5.0
        mkdir build
        cd build
        cmake -DCMAKE_CXX_COMPILER=$COMPILER ../.
        make
        make install
        cd -
    fi
else
    # install houdini pre-requisites
    apt-get install -y libxi-dev
    apt-get install -y csh
    apt-get install -y default-jre
    apt-get install -y python-dev
    # boost no longer shipped with Houdini from 16.5 onwards
    apt-get install -y bc
    HOUDINI_HAS_BOOST=$(echo "$HOUDINI_MAJOR < 16.5" | bc -l)
    if [ $HOUDINI_HAS_BOOST -eq 0 ]; then
        apt-get install -y libboost-iostreams-dev
        apt-get install -y libboost-system-dev
    fi
    apt-get install -y python-mechanize
    export PYTHONPATH=${PYTHONPATH}:/usr/lib/python2.7/dist-packages
    # download and unpack latest houdini headers and libraries from daily-builds
    python ci/download_houdini.py $HOUDINI_MAJOR
    tar -xzf hou.tar.gz
    ln -s houdini* hou
    cd hou
    tar -xzf houdini.tar.gz
    cd -
fi
