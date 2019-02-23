#!/usr/bin/env bash
#
# CircleCI build script for OpenVDB
#
# TASK (core/extras/test/run/houdini):
#        * core - compiling core library
#        * extras - compiling python module, tests and binaries
#        * test - compiling unit tests
#        * run - executing unit tests
#        * houdini - compiling houdini library and plugins
# ABI (3/4/5/6) - the ABI version of OpenVDB
# BLOSC (yes/no) - to build with Blosc support
# MODE (release/debug/header) - optimized build, debug build or header checks
# HOUDINI_MAJOR (16.0/16.5/17.0) - the major version of Houdini
# COMPILER (g++/clang++) - build with GCC or Clang
#
# Note that builds use two threads to ensure they stay within 4GB of memory
#
# Author: Dan Bailey

set -ex

TASK="$1"
ABI="$2"
BLOSC="$3"
MODE="$4"
HOUDINI_MAJOR="$5"
COMPILER="$6"

PARAMS="CXX=$COMPILER DESTDIR=/tmp/OpenVDB abi=$ABI strict=yes"

if [ "$MODE" = "debug" ]; then
    PARAMS+=" debug=yes"
fi

# Location of third-party dependencies for standalone and houdini builds
if [ "$HOUDINI_MAJOR" = "none" ]; then
    PARAMS+="       BOOST_LIB_DIR=/usr/lib/x86_64-linux-gnu\
                    EXR_INCL_DIR=/usr/include/OpenEXR\
                    EXR_LIB_DIR=/usr/local/lib\
                    TBB_LIB_DIR=/usr/lib\
                    CONCURRENT_MALLOC_LIB=\
                    CPPUNIT_INCL_DIR=$HOME/cppunit/include\
                    CPPUNIT_LIB_DIR=$HOME/cppunit/lib\
                    LOG4CPLUS_INCL_DIR=/usr/include\
                    LOG4CPLUS_LIB_DIR=/usr/lib/x86_64-linux-gnu\
                    GLFW_INCL_DIR=/usr/include/GL\
                    GLFW_LIB_DIR=/usr/lib/x86_64-linux-gnu\
                    PYTHON_INCL_DIR=/usr/include/python2.7\
                    PYTHON_LIB_DIR=/usr/lib/x86_64-linux-gnu\
                    BOOST_PYTHON_LIB_DIR=/usr/lib/x86_64-linux-gnu\
                    BOOST_PYTHON_LIB=-lboost_python\
                    NUMPY_INCL_DIR=/usr/lib/python2.7/dist-packages/numpy/core/include/numpy\
                    PYTHON_WRAP_ALL_GRID_TYPES=yes\
                    EPYDOC=/usr/bin/epydoc\
                    DOXYGEN=/usr/bin/doxygen"
    if [ "$BLOSC" = "yes" ]; then
        PARAMS+="   BLOSC_INCL_DIR=$HOME/blosc/include\
                    BLOSC_LIB_DIR=$HOME/blosc/lib"
    else
        PARAMS+="   BLOSC_INCL_DIR=\
                    BLOSC_LIB_DIR="
    fi
else
    # source houdini_setup
    cd hou
    source houdini_setup
    cd -
    PARAMS+="       BOOST_INCL_DIR=/root/project/hou/toolkit/include\
                    BOOST_LIB_DIR=/root/project/hou/dsolib\
                    TBB_LIB_DIR=/root/project/hou/dsolib\
                    EXR_INCL_DIR=/root/project/hou/toolkit/include\
                    EXR_LIB_DIR=/root/project/hou/dsolib\
                    LOG4CPLUS_INCL_DIR=\
                    GLFW_INCL_DIR=\
                    PYTHON_INCL_DIR=\
                    DOXYGEN="
    if [ "$BLOSC" = "yes" ]; then
        PARAMS+="   BLOSC_INCL_DIR=/root/project/hou/toolkit/include\
                    BLOSC_LIB_DIR=/root/project/hou/dsolib"
    else
        PARAMS+="   BLOSC_INCL_DIR=\
                    BLOSC_LIB_DIR="
    fi
fi

# fix to issue with -isystem includes using GCC 6.3.0
sed -E -i.bak "s/-isystem/-I/g" openvdb/Makefile

# fix to issue with using Clang 3.8 with hcustom
sed -E -i.bak "s/hcustom -c/hcustom -c | sed 's\/-fno-exceptions\/-DGCC4 -DGCC3 -Wno-deprecated\/g'/g" openvdb_houdini/Makefile

# fix to disable hcustom tagging to remove timestamps in Houdini DSOs
# note that this means the DSO can no longer be loaded in Houdini
sed -E -i.bak 's/\/hcustom/\/hcustom -t/g' openvdb_houdini/Makefile

if [ "$TASK" = "core" ]; then
    if [ "$MODE" = "header" ]; then
        # check for any indirect includes
        make -C openvdb $PARAMS header_test -j2
    else
        # build OpenVDB core library and OpenVDB houdini library
        make -C openvdb $PARAMS install_lib -j2
    fi
elif [ "$TASK" = "extras" ]; then
    # build OpenVDB core library, OpenVDB Python module and all binaries
    if [ "$COMPILER" = "g++" ]; then
        # Use just one thread for GCC as 2GB per core is insufficient for Boost Python
        make -C openvdb $PARAMS install
    else
        make -C openvdb $PARAMS install -j2
    fi
elif [ "$TASK" = "test" ]; then
    make -C openvdb $PARAMS vdb_test -j2
elif [ "$TASK" = "houdini" ]; then
    if [ "$MODE" = "header" ]; then
        # check for any indirect includes
        make -C openvdb_houdini $PARAMS header_test -j2
    else
        # build OpenVDB Houdini library and plugins
        make -C openvdb_houdini $PARAMS install -j2
    fi
elif [ "$TASK" = "run" ]; then
    # run unit tests
    make -C openvdb $PARAMS test verbose=yes
fi
