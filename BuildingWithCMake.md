# Overview

The CMake build infrastructure for OpenVDB is designed for out-of-source build. This is to avoid overwriting the existing Makefile that already exists in the source tree.

# Examples

Here are examples of build command for Linux, Windows and OS X

## OS X

```{r, engine='bash', count_lines}
#!/bin/sh
rm -f CMakeCache.txt
export GLFW3_ROOT=$HOME/systems/glfw/v3.1.1
export BOOST_ROOT=$HOME/systems/boost/v1.57.0
export TBB_ROOT=$HOME/systems/tbb/tbb44_20151115oss
export ILMBASE_ROOT=$HOME/systems/OpenEXR/v2.2.0
export OPENEXR_ROOT=$HOME/systems/OpenEXR/v2.2.0
export BLOSC_ROOT=$HOME/systems/blosc/v1.7.0
cmake -Wno-dev \
    -D OPENEXR_NAMESPACE_VERSIONING=OFF \
    -D CMAKE_CXX_FLAGS="-fPIC -std=c++11" \
    -D TBB_LIBRARY_DIR=$TBB_ROOT/lib \
    -D DOXYGEN_SKIP_DOT=ON \
    -D Blosc_USE_STATIC_LIBS=ON \
    -D USE_GLFW3=ON \
    -D GLFW3_USE_STATIC_LIBS=ON \
    -D _ECLIPSE_VERSION=4.6 \
    -D Boost_USE_STATIC_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=$HOME/systems/OpenVDB/v4.0.0 \
    -G "Eclipse CDT4 - Unix Makefiles" \
    ../nyue_openvdb_git
```


## CentOS 6.7

### Build Core
```{r, engine='bash', count_lines}
#!/bin/sh
rm -f CMakeCache.txt
export ILMBASE_ROOT=$RHOME/Systems/OpenEXR/v2.2.0
export OPENEXR_ROOT=$RHOME/Systems/OpenEXR/v2.2.0
export BOOST_ROOT=$RHOME/Systems/boost/v1.61.0
export TBB_ROOT=$RHOME/Systems/tbb/tbb44_20150728oss
export GLFW3_ROOT=$RHOME/Systems/glfw/v3.2.1
export BLOSC_ROOT=$RHOME/Systems/c-blosc/v1.7.0
cmake \
    -D PYTHON_LIBRARY=/sw/external/COS6/python/2.6.4/lib/libpython2.6.so \
    -D PYTHON_INCLUDE_DIR=/sw/external/COS6/python/2.6.4/include/python2.6 \
    -D CMAKE_CXX_COMPILER=/sw/external/COS6/gcc/4.8.2/bin/g++ \
    -D CMAKE_C_COMPILER=/sw/external/COS6/gcc/4.8.2/bin/gcc \
    -D CMAKE_CXX_FLAGS=-std=c++11 \
    -D MINIMUM_BOOST_VERSION=1.52 \
    -D ILMBASE_NAMESPACE_VERSIONING=OFF \
    -D OPENEXR_NAMESPACE_VERSIONING=OFF \
    -D USE_GLFW3=ON \
    -D Blosc_USE_STATIC_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=$RHOME/Systems/OpenVDB/4.0.0 \
    -D _ECLIPSE_VERSION=4.6 \
    -G "Eclipse CDT4 - Unix Makefiles" \
    ../nyue_openvdb_git
```
### Build Houdini SOPs
```{r, engine='bash', count_lines}
#!/bin/sh
rm -f CMakeCache.txt
export BLOSC_ROOT=$HT
export TBB_ROOT=$HT
export ILMBASE_ROOT=$HT
export OPENEXR_ROOT=$HT
export GLFW3_ROOT=$HOME/systems/glfw/v3.2.1
export CPPUNIT_ROOT=$HOME/systems/cppunit/v1.10.2
cmake \
    -D ZLIB_ROOT=$HT \
    -D BOOST_ROOT=$HT \
    -D BOOST_LIBRARYDIR=$HDSO \
    -D ILMBASE_LIBRARYDIR=$HDSO \
    -D OPENEXR_LIBRARYDIR=$HDSO \
    -D BLOSC_LIBRARYDIR=$HDSO \
    -D TBB_LIBRARYDIR=$HDSO \
    -D ZLIB_LIBRARY=$HDSO/libz.so \
    -D HDK_AUTO_GENERATE_SESITAG=OFF \
    -D OPENVDB_BUILD_HOUDINI_SOPS=ON \
    -D CMAKE_CXX_FLAGS="-fPIC -std=c++11" \
    -D USE_GLFW3=ON \
    -D GLFW3_USE_STATIC_LIBS=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=$HOME/systems/OpenVDB/v4.0.0 \
    -D _ECLIPSE_VERSION=4.6 \
    -G "Eclipse CDT4 - Unix Makefiles" \
    ../nyue_openvdb_git
```
### Build Maya Node
```{r, engine='bash', count_lines}
#!/bin/sh
rm -f CMakeCache.txt
export ILMBASE_ROOT=$RHOME/Systems/OpenEXR/v2.2.0
export OPENEXR_ROOT=$RHOME/Systems/OpenEXR/v2.2.0
export BOOST_ROOT=$RHOME/Systems/boost/v1.61.0
export GLFW3_ROOT=$RHOME/Systems/glfw/v3.2.1
export BLOSC_ROOT=$RHOME/Systems/c-blosc/v1.7.0
export CPPUNIT_ROOT=$RHOME/Systems-gcc48/cppunit/v1.10.2
export MAYA_LOCATION=/sw/external/autodesk/maya2016-sp5-x64
export TBB_ROOT=$MAYA_LOCATION
cmake \
    -D Tbb_TBB_LIBRARY=$MAYA_LOCATION/lib/libtbb.so \
    -D Tbb_TBBMALLOC_LIBRARY=$MAYA_LOCATION/lib/libtbbmalloc.so \
    -D OPENVDB_ENABLE_3_ABI_COMPATIBLE=ON \
    -D OPENVDB_BUILD_MAYA_PLUGIN=ON \
    -D OPENVDB_ENABLE_RPATH=ON \
    -D PYTHON_LIBRARY=/sw/external/COS6/python/2.6.4/lib/libpython2.6.so \
    -D PYTHON_INCLUDE_DIR=/sw/external/COS6/python/2.6.4/include/python2.6 \
    -D CMAKE_CXX_COMPILER=/sw/external/COS6/gcc/4.8.2/bin/g++ \
    -D CMAKE_C_COMPILER=/sw/external/COS6/gcc/4.8.2/bin/gcc \
    -D CMAKE_CXX_FLAGS=-std=c++11 \
    -D MINIMUM_BOOST_VERSION=1.52 \
    -D ILMBASE_NAMESPACE_VERSIONING=OFF \
    -D OPENEXR_NAMESPACE_VERSIONING=OFF \
    -D USE_GLFW3=ON \
    -D Boost_USE_STATIC_LIBS=ON \
    -D Blosc_USE_STATIC_LIBS=ON \
    -D CPPUnit_USE_STATIC_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=$RHOME/Systems-gcc48/OpenVDB/4.0.0 \
    ../nyue_openvdb_git
```


## Windows - Visual Studio 2015 x64

```{r, engine='bash', count_lines}
setlocal
del /f CMakeCache.txt
set BOOST_ROOT=C:\Systems\x64\vc14\boost\v1.62.0
set GLEW_ROOT=C:\Systems\x64\vc14\glew\v1.13.0
set GLFW3_ROOT=C:\Systems\x64\vc14\glfw\v3.2.1
set ILMBASE_ROOT=C:\Systems\x64\vc14\OpenEXR\v2.2.0-static
set OPENEXR_ROOT=C:\Systems\x64\vc14\OpenEXR\v2.2.0-static
set TBB_ROOT=C:\Systems\x64\tbb2017_20160916oss
set BLOSC_ROOT=C:\Systems\x64\vc14\c-blosc\v1.7.0
cmake ^
      -D DOXYGEN_SKIP_DOT=ON ^
      -D Blosc_USE_STATIC_LIBS=OFF ^
      -D USE_GLFW3=ON ^
      -D GLFW3_USE_STATIC_LIBS=ON ^
      -D Boost_USE_STATIC_LIBS=ON ^
      -D Boost_INCLUDE_DIR="C:\Systems\x64\vc14\boost\v1.62.0\include\boost-1_62" ^
      -D ZLIB_INCLUDE_DIR=C:/Systems/x64/vc14/zlib/v1.2.8/include ^
      -D ZLIB_LIBRARY=C:/Systems/x64/vc14/zlib/v1.2.8/lib/zlibstatic.lib ^
      -D TBB_LIBRARY_DIR=%TBB_ROOT%\lib\intel64\vc14 ^
      -D TBB_LIBRARY_PATH=%TBB_ROOT%\lib\intel64\vc14 ^
      -D Tbb_TBB_LIBRARY=%TBB_ROOT%\lib\intel64\vc14\tbb.lib ^
      -D Tbb_TBBMALLOC_LIBRARY=%TBB_ROOT%\lib\intel64\vc14\tbbmalloc.lib ^
      -D Tbb_TBB_PREVIEW_LIBRARY=%TBB_ROOT%\lib\intel64\vc14\tbb_preview.lib ^
      -D CMAKE_INSTALL_PREFIX="C:\Systems\x64\vc14\openvdb\v4.0.0" ^
      -G "Visual Studio 14 2015 Win64" ^
      ..\nyue_openvdb_git
```
