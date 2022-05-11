![OpenVDB](https://www.openvdb.org/images/openvdb_logo.png)

[![License](https://img.shields.io/github/license/AcademySoftwareFoundation/openvdb)](LICENSE)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2774/badge)](https://bestpractices.coreinfrastructure.org/projects/2774)
[![Slack](https://slack.aswf.io/badge.svg)](https://slack.aswf.io/)


| OpenVDB |   AX   |  Nano  | Houdini |
| :----:  | :----: | :----: |  :----: |
| [![core](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/build.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/build.yml) | [![ax](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/ax.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/ax.yml) | [![nano](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/nanovdb.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/nanovdb.yml) | [![hou](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/houdini.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/houdini.yml) |

[Website](https://www.openvdb.org) |
[Discussion Forum](https://github.com/AcademySoftwareFoundation/openvdb/discussions) |
[Documentation](https://www.openvdb.org/documentation/doxygen)

OpenVDB is an open source C++ library comprising a novel hierarchical data structure and a large suite of tools for the efficient storage and manipulation of sparse volumetric data discretized on three-dimensional grids. It was developed by DreamWorks Animation for use in volumetric applications typically encountered in feature film production.


### Development Repository

This GitHub repository hosts the trunk of the OpenVDB development. This implies that it is the newest public version with the latest features and bug fixes. However, it also means that it has not undergone a lot of testing and is generally less stable than the [production releases](https://github.com/AcademySoftwareFoundation/openvdb/releases).


### License

OpenVDB is released under the [Mozilla Public License Version 2.0](https://www.mozilla.org/MPL/2.0/), which is a free, open source software license developed and maintained by the Mozilla Foundation.

The trademarks of any contributor to this project may not be used in association with the project without the contributor's express permission.

### Contributing

OpenVDB welcomes contributions to the OpenVDB project. Please refer to the [contribution guidelines](CONTRIBUTING.md) for details on how to make a contribution.

### Developer Quick Start

The following provides basic installation examples for the core OpenVDB library. Other components, such as the python module, OpenVDB AX, NanoVDB and various executables, may require additional dependencies. See the [build documentation](https://www.openvdb.org/documentation/doxygen/build.html) for help with installations.

#### Linux
##### Installing Dependencies (Boost, TBB, Blosc)

```bash
apt-get install -y libboost-iostreams-dev
apt-get install -y libtbb-dev
apt-get install -y libblosc-dev
```

##### Building OpenVDB
```bash
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake ..
make -j4 && make install
```
#### macOS
##### Installing Dependencies (Boost, TBB, Blosc)
```bash
brew install boost
brew install tbb
brew install c-blosc
```
##### Building OpenVDB
```bash
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake ..
make -j4 && make install
```
#### Windows
##### Installing Dependencies (Boost, TBB, Blosc)

Note that the following commands have only been tested for 64bit systems/libraries.
It is recommended to set the `VCPKG_DEFAULT_TRIPLET` environment variable to
`x64-windows` to use 64-bit libraries by default. You will also require
[Git](https://git-scm.com/downloads), [vcpkg](https://github.com/microsoft/vcpkg)
and [CMake](https://cmake.org/download/) to be installed.

```bash
vcpkg install zlib:x64-windows
vcpkg install blosc:x64-windows
vcpkg install tbb:x64-windows
vcpkg install boost-iostreams:x64-windows
vcpkg install boost-any:x64-windows
vcpkg install boost-algorithm:x64-windows
vcpkg install boost-uuid:x64-windows
vcpkg install boost-interprocess:x64-windows
```
##### Building OpenVDB
```bash
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_VCPKG>\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -A x64 ..
cmake --build . --parallel 4 --config Release --target install
```

#### Building OpenVDB AX

OpenVDB AX depends on the core OpenVDB library. See the [build documentation](https://www.openvdb.org/documentation/doxygen/build.html) for all available AX component options:

```bash
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake -DOPENVDB_BUILD_AX=ON ..
make -j4 && make install
```

#### Building NanoVDB

NanoVDB can be built with and without OpenVDB support. To see full build instructions
see the [NanoVDB build documentation](https://www.openvdb.org/documentation/doxygen/NanoVDB_HowToBuild.html)

#### Building Without OpenVDB Support

```bash
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb/nanovdb/nanovdb  # Build from the subdirectory
mkdir build
cd build
cmake ..
make -j4 && make install
```

#### Building With OpenVDB Support

```bash
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake -DOPENVDB_BUILD_NANOVDB=ON ..
make -j4 && make install
```
