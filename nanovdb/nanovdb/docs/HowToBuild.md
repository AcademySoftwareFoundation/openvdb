# Build instructions {#HowToBuild}

Note that NanoVDB can be built in many different ways, in the sense that its many external dependencies are optional. This includes OpenVDB, TBB, graphics APIs like CUDA, OpenGL, and OpenCL, as well as compression like ZLIB and BLOSC.
However, since the core data structure in [NanoVDB.h](../include/nanovdb/NanoVDB.h) is self-contained, client code may only depend on this single header file.

Also, note that if the environment variable `VDB_DATA_PATH` is set to a directory containing the OpenVDB files from [the OpenVDB web site](http://www.openvdb.org/download), they will be used for the unit and the benchmark tests. Else a simple narrow-band level set sphere is used. Set the environment variable `VDB_SCRATCH_PATH` to define the path to where images generated in the benchmark tests are stored (defaults to the build directory).

### Using CMake:

To explicitly remove a dependency, X, add the argument "-DNANOVDB_USE_X=false" where X = {BLOSC, CUDA, OPENCL, OPENVDB, OPTIX, TBB, ZLIB}.
To see the compiler arguments add "-DCMAKE_VERBOSE_MAKEFILE=true".

By far the simplest way to build the library is to ignore the options entirely:
  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake ..
  foo@bar:~$ make -j
  ```
This will build using every optional dependency CMake can successfully find. Be sure to check the CMake log to see what dependencies were **not** found.

Here is an example of how to build a typical configuration using CMake (change the dependency paths to match your environment):
  ```console
  foo@bar:~$ mkdir build
  foo@bar:~$ cd build
  foo@bar:~$ cmake .. -DOpenVDB_ROOT=/home/kmu/local -DIlmBase_ROOT=/home/kmu/local -DTBB_ROOT=/home/kmu/local -DBlosc_ROOT=/home/kmu/local -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_VERBOSE_MAKEFILE=ON
  foo@bar:~$ make -j
  ```

### Copyright Contributors to the OpenVDB Project
### SPDX-License-Identifier: MPL-2.0