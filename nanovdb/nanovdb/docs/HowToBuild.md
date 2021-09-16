# Build instructions {#HowToBuild}

This project offers two different build systems, CMake and GNU make. The former is by far the most convenient and up-to-date, but we include a Makefile as a fallback. Note that NanoVDB can be built in many different ways, in the sense that its many external dependencies are optional. This includes OpenVDB, TBB, graphics APIs like CUDA, OpenGL, and OpenCL, as well as compression like ZLIB and BLOSC.
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

### Using Makefile:

Alternatively, you can use the included makefile, but first you'll have to edit it and set the correct paths to external libraries. While NanoVDB doesn't depend on OpenVDB, the [OpenToNanoVDB](../include/nanovdb/util/OpenToNanoVDB.h), which converts an OpenVDB grid into a NanoVDB grid, obviously does! The same is true for the [NanoToOpenVDB](../include/nanovdb/util/NanoToOpenVDB.h).

* To build and run the included examples:

  ```console
  foo@bar:~$ make examples
  ```

* Build and run unit tests:

  ```console
  foo@bar:~$ make test
  ```

* Build and run benchmark test:

  ```console
  foo@bar:~$ setenv VDB_DATA_PATH /home/kmu/dev/data/vdb
  foo@bar:~$ setenv VDB_SCRATCH_PATH /mnt/ramdisk
  foo@bar:~$ make benchmark
  ```

* Clean up build directory
 
  ```console
  foo@bar:~$ make clean
  ```

* Create a tar-ball of the source code
 
  ```console
  foo@bar:~$ make archive
  ```

* Build and run everything
 
  ```console
  foo@bar:~$ make all
  ```

### Copyright Contributors to the OpenVDB Project
### SPDX-License-Identifier: MPL-2.0