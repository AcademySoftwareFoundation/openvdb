![OpenVDB](https://www.openvdb.org/images/openvdb_logo.png)

[![License](https://img.shields.io/github/license/AcademySoftwareFoundation/openvdb)](LICENSE)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/2774/badge)](https://bestpractices.coreinfrastructure.org/projects/2774)
[![Slack](https://slack.aswf.io/badge.svg)](https://slack.aswf.io/)


| OpenVDB |   AX   |  Nano  | Houdini |
| :----:  | :----: | :----: |  :----: |
| [![core](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/build.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/build.yml) | [![ax](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/ax.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/ax.yml) | [![nano](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/nanovdb.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/nanovdb.yml) | [![hou](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/houdini.yml/badge.svg)](https://github.com/AcademySoftwareFoundation/openvdb/actions/workflows/houdini.yml) |

[Website](https://www.openvdb.org) |
[Discussion Forum](https://github.com/AcademySoftwareFoundation/openvdb/discussions) |
[Documentation](https://academysoftwarefoundation.github.io/openvdb)

OpenVDB is an open source C++ library comprising a novel hierarchical data structure and a large suite of tools for the efficient storage and manipulation of sparse volumetric data discretized on three-dimensional grids. It was developed by DreamWorks Animation for use in volumetric applications typically encountered in feature film production.


### Development Repository

This GitHub repository hosts the trunk of the OpenVDB development. This implies that it is the newest public version with the latest features and bug fixes. However, it also means that it has not undergone a lot of testing and is generally less stable than the [production releases](https://github.com/AcademySoftwareFoundation/openvdb/releases).


### License

OpenVDB is released under the [Mozilla Public License Version 2.0](https://www.mozilla.org/MPL/2.0/), which is a free, open source software license developed and maintained by the Mozilla Foundation.

The trademarks of any contributor to this project may not be used in association with the project without the contributor's express permission.

### Contributing

OpenVDB welcomes contributions to the OpenVDB project. Please refer to the [contribution guidelines](CONTRIBUTING.md) for details on how to make a contribution.

### Developer Quick Start

The following provides basic installation examples for the core OpenVDB library. Other components, such as the python module, OpenVDB AX and various executables, may require additional dependencies. See the [build documentation](https://academysoftwarefoundation.github.io/openvdb/build.html) for help with installations.

#### Linux
##### Installing Dependencies (Boost, TBB, Blosc)

```
apt-get install -y libboost-iostreams-dev
apt-get install -y libboost-system-dev
apt-get install -y libtbb-dev
apt-get install -y libblosc-dev
```

##### Building OpenVDB
```
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake ..
make -j4
make install
```
#### macOS
##### Installing Dependencies (Boost, TBB, Blosc)
```
brew install boost
brew install tbb
brew install c-blosc
```
##### Building OpenVDB
```
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake ..
make -j4
make install
```
#### Windows
##### Installing Dependencies (Boost, TBB, Blosc)

Note that the following commands have only been tested for 64bit systems/libraries.
It is recommended to set the `VCPKG_DEFAULT_TRIPLET` environment variable to
`x64-windows` to use 64-bit libraries by default. You will also require
[Git](https://git-scm.com/downloads), [vcpkg](https://github.com/microsoft/vcpkg)
and [CMake](https://cmake.org/download/) to be installed.

```
vcpkg install zlib:x64-windows
vcpkg install blosc:x64-windows
vcpkg install tbb:x64-windows
vcpkg install boost-iostreams:x64-windows
vcpkg install boost-system:x64-windows
vcpkg install boost-any:x64-windows
vcpkg install boost-algorithm:x64-windows
vcpkg install boost-uuid:x64-windows
vcpkg install boost-interprocess:x64-windows
```
##### Building OpenVDB
```
git clone git@github.com:AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_VCPKG>\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -A x64 ..
cmake --build . --parallel 4 --config Release --target install
```

#### Building NanoVDB

**First example: building NanoVDB and OpenVDB core**

NanoVDB is now a module of the larger OpenVDB project. A user can build both libraries together. This will build the OpenVDB core library, install the NanoVDB header files, and build the NanoVDB command-line tools in the `build/nanovdb/cmd` directory. From the 'root' OpenVDB project directory (change the dependency paths to match your environment):
  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DUSE_NANOVDB=ON -DTBB_ROOT=/path/to/tbb -DBOOST_ROOT=/path/to/boost -DBLOSC_ROOT=/path/to/blosc -DCMAKE_INSTALL_PREFIX=/install/path
  foo@bar:~$ make -j 4 && make install
  ```
Note that the default value of `NANOVDB_USE_OPENVDB` is linked to `OPENVDB_BUILD_CORE` option and can be overriden by passing on `-DNANOVDB_USE_OPENVDB=OFF`. The `Boost` library is included because it is a requirement for building OpenVDB.

In general, CMake will try to find every optional dependency when a user opts to add an additional dependency. Be sure to check the CMake log to see what dependencies were **not** found.

**Second example: NanoVDB with no dependencies**

From the 'root' OpenVDB project directory:
  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DUSE_NANOVDB=ON -DOPENVDB_BUILD_CORE=OFF -DOPENVDB_BUILD_BINARIES=OFF -DNANOVDB_USE_TBB=OFF -DNANOVDB_USE_BLOSC=OFF -DNANOVDB_USE_ZLIB=OFF -DCMAKE_INSTALL_PREFIX=/install/path
  foo@bar:~$ make -j 4 && make install
  ```

Another option is to build it from the NanoVDB directory itself, which is much simpler:
  ```console
  foo@bar:~$ cd nanovdb/nanovdb
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path
  foo@bar:~$ make -j 4 && make install
  ```
Both options will install the NanoVDB header files to the `/install/path` as well as building `nanovdb_print` and `nanovdb_validate` executable. The path where these executables are installed will be different: in the first option they will be under `build/nanovdb/cmd` directory; whilst in the second option they will be under the `build/cmd/` directory.

**Third example: build 'everything' in NanoVDB along with OpenVDB core**

From the root OpenVDB directory:
  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DUSE_NANOVDB=ON -DNANOVDB_BUILD_UNITTESTS=ON -DNANOVDB_BUILD_EXAMPLES=ON -DNANOVDB_BUILD_BENCHMARK=ON -DNANOVDB_USE_INTRINSICS=ON -DNANOVDB_USE_CUDA=ON -DNANOVDB_CUDA_KEEP_PTX=ON -DTBB_ROOT=/path/to/tbb -DBOOST_ROOT=/path/to/boost -DBLOSC_ROOT=/path/to/blosc -DGTEST_ROOT=/path/to/gtest -DCMAKE_INSTALL_PREFIX=/install/path
  foo@bar:~$ make -j 4 && make install
  ```
Note that if you already have the correct version of OpenVDB pre-installed, you can configure CMake to link against that library by passing the arguments `-DOPENVDB_BUILD_CORE=OFF -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_ROOT=/path/to/openvdb` when invoking `cmake`.
