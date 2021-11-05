# Build instructions {#NanoVDB_HowToBuild}

Note that NanoVDB can be built in many different ways, in the sense that its many external dependencies are optional. This includes OpenVDB, TBB, graphics APIs like CUDA, OpenGL, and OpenCL, as well as compression like ZLIB and BLOSC.
However, since the core data structure in [NanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/NanoVDB.h) is self-contained, client code may only depend on this single header file.

Also, note that if the environment variable `VDB_DATA_PATH` is set to a directory containing the OpenVDB files from [the OpenVDB web site](http://www.openvdb.org/download), they will be used for the unit and the benchmark tests. Else a simple narrow-band level set sphere is used. Set the environment variable `VDB_SCRATCH_PATH` to define the path to where images generated in the benchmark tests are stored (defaults to the build directory).

[//]: # (TODO: add back OPTIX, OPENCL, and OPENGL when these options are added back)
To explicitly remove a dependency, `X`, add the argument `-DNANOVDB_USE_X=OFF` where `X = {OPENVDB, TBB, BLOSC, ZLIB, CUDA}`. These options are set in the [NanoVDB root CMakeLists.txt](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/CMakeLists.txt). The same file also defines options to build NanoVDB examples, benchmarks, unit-tests, and command-line tools. The default values for these options are `OFF`. A user can override each of these options by passing, for example, `-DNANOVDB_BUILD_UNITTESTS=ON` flag while running cmake from the command line.

To see the compiler arguments add `-DCMAKE_VERBOSE_MAKEFILE=ON`.

## First example: building NanoVDB and OpenVDB core

NanoVDB is now a module of the larger OpenVDB project. A user can build both libraries together. This will build the OpenVDB core library, install the NanoVDB header files, and build the NanoVDB command-line tools in the `build/nanovdb/cmd` directory. From the 'root' OpenVDB project directory (change the dependency paths to match your environment):
  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DUSE_NANOVDB=ON -DTBB_ROOT=/path/to/tbb -DBOOST_ROOT=/path/to/boost -DBLOSC_ROOT=/path/to/blosc -DCMAKE_INSTALL_PREFIX=/install/path
  foo@bar:~$ make -j 4 && make install
  ```
Note that the default value of `NANOVDB_USE_OPENVDB` is linked to `OPENVDB_BUILD_CORE` option and can be overriden by passing on `-DNANOVDB_USE_OPENVDB=OFF`. The `Boost` library is included because it is a requirement for building OpenVDB.

In general, CMake will try to find every optional dependency when a user opts to add an additional dependency. Be sure to check the CMake log to see what dependencies were **not** found.

## Second example: NanoVDB with no dependencies

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

## Third example: build 'everything' in NanoVDB along with OpenVDB core

From the root OpenVDB directory:
  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DUSE_NANOVDB=ON -DNANOVDB_BUILD_UNITTESTS=ON -DNANOVDB_BUILD_EXAMPLES=ON -DNANOVDB_BUILD_BENCHMARK=ON -DNANOVDB_USE_INTRINSICS=ON -DNANOVDB_USE_CUDA=ON -DNANOVDB_CUDA_KEEP_PTX=ON -DTBB_ROOT=/path/to/tbb -DBOOST_ROOT=/path/to/boost -DBLOSC_ROOT=/path/to/blosc -DGTEST_ROOT=/path/to/gtest -DCMAKE_INSTALL_PREFIX=/install/path
  foo@bar:~$ make -j 4 && make install
  ```
Note that if you already have the correct version of OpenVDB pre-installed, you can configure CMake to link against that library by passing the arguments `-DOPENVDB_BUILD_CORE=OFF -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_ROOT=/path/to/openvdb` when invoking `cmake`.
