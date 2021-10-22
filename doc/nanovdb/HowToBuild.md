# Build instructions {#NanoVDB_HowToBuild}

Note that NanoVDB can be built in many different ways, in the sense that its many external dependencies are optional. This includes OpenVDB, TBB, graphics APIs like CUDA, OpenGL, and OpenCL, as well as compression like ZLIB and BLOSC.
However, since the core data structure in [NanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/NanoVDB.h) is self-contained, client code may only depend on this single header file.

Also, note that if the environment variable `VDB_DATA_PATH` is set to a directory containing the OpenVDB files from [the OpenVDB web site](http://www.openvdb.org/download), they will be used for the unit and the benchmark tests. Else a simple narrow-band level set sphere is used. Set the environment variable `VDB_SCRATCH_PATH` to define the path to where images generated in the benchmark tests are stored (defaults to the build directory).

### Using CMake:

[//]: # (TODO: add back OPTIX, OPENCL, and OPENGL when these options are added back)
To explicitly remove a dependency, `X`, add the argument `-DNANOVDB_USE_X=OFF` where `X = {OPENVDB, TBB, BLOSC, ZLIB, CUDA}`. These options are set in the [NanoVDB root CMakeLists.txt](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/CMakeLists.txt). The same file also defines options to build NanoVDB examples, benchmarks, unit-tests, and command line tools. The default values for these options are `OFF`. A user can override each of these options by passing, for example, `-DNANOVDB_BUILD_UNITTESTS=ON` flag while running cmake from the command line.

To see the compiler arguments add `-DCMAKE_VERBOSE_MAKEFILE=ON`.

NanoVDB is now a module of the larger OpenVDB project. To build NanoVDB without any dependencies and additional options, a user can run the build commands from the root OpenVDB directory:

  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DUSE_NANOVDB=ON -DOPENVDB_BUILD_CORE=OFF -DOPENVDB_BUILD_BINARIES=OFF -DUSE_TBB=OFF -DUSE_BLOSC=OFF -DUSE_ZLIB=OFF  -DCMAKE_INSTALL_PREFIX=/install/path
  foo@bar:~$ make -j 4 && make install
  ```
In the example above, we need to turn off the default values for building the core OpenVDB library (`DOPENVEB_BUILD_CORE=OFF`), the OpenVDB binaries (`-DOPENVEB_BUILD_BINARIES=OFF`), as well as a few other OpenVDB default dependencies (`-DUSE_TBB=OFF`, `DUSE_BLOSC=OFF`, `DUSE_ZLIB=OFF`). This simple example installs the NanoVDB header files in the `/install/path/include/nanovdb` directory.

CMake will try to find every optional dependency when a user opts to add an additional dependency. Be sure to check the CMake log to see what dependencies were **not** found.

Here is an example of how to build NanoVDB along with the core OpenVDB library and the NanoVDB unit-tests (change the dependency paths to match your environment):

  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DUSE_NANOVDB=ON -DNANOVDB_BUILD_UNITTESTS=ON -DOPENVDB_BUILD_CORE=ON -DOPENVDB_BUILD_BINARIES=OFF -DNANOVDB_USE_OPENVDB=ON  -DUSE_ZLIB=OFF -DBOOST_ROOT=/path/to/boost -DTBB_ROOT=/path/to/tbb -DBLOSC_ROOT=/path/to/blosc -DGTEST_ROOT=/path/to/gtest -DCMAKE_INSTALL_PREFIX=/install/pathfoo@bar:~$ make -j 4 && make install
  ```
The NanoVDB unit-test will then be installed in `build/nanovdb/nanovdb/unittest` directory.