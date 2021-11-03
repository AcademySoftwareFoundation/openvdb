# Source tree structure {#NanoVDB_SourceTree}

#### Overview of the directory tree of NanoVDB.

* [NanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/NanoVDB.h) C++11 implementation of the core data structure and its access methods.
* [CNanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/CNanoVDB.h) Incomplete but useable C99 implementation of the core data structure and its access methods.  Designed in particular for use in OpenCL kernels.  Note that this relies on zero-sized arrays for the _reserved padding, so will not work on all compilers (with MSVC being a particular example)
* [PNanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/PNanoVDB.h) C99 implementation of the core data structure and its access methods. More complete coverage than CNanoVDB.  This version is pointer-less and supports virtually all graphics APIs.
* [util/GridHandle.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/util/GridHandle.h) defines a handler for the memory allocated to a NanoVDB grid.
* [util/IO.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/util/IO.h) implements I/O support.
* [util/OpenToNanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/util/OpenToNanoVDB.h) defines the converter from OpenVDB to NanoVDB and obviously depends on the OpenVDB library (as the only header file).
* [Ray.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/util/Ray.h) Ray class.
* [HDDA.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/util/HDDA.h) HDDA related.
* [SampleFromVoxels.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/util/SampleFromVoxels.h) interpolation.

```bash
foo@bar:~$ tree
.
├── nanovdb
│   ├── CNanoVDB.h
│   ├── NanoVDB.h
│   ├── PNanoVDB.h
│   └── util
│       ├── CSampleFromVoxels.h
│       ├── CudaDeviceBuffer.h
|       ├── DitherLUT.h
│       ├── ForEach.h
│       ├── GridBuilder.h
│       ├── GridChecksum.h
│       ├── GridHandle.h
│       ├── GridStats.h
│       ├── GridValidator.h
│       ├── HDDA.h
│       ├── HostBuffer.h
│       ├── Invoke.h
│       ├── IO.h
│       ├── NanoToOpenVDB.h
│       ├── OpenToNanoVDB.h
|       ├── Primitives.h
│       ├── Range.h
│       ├── Ray.h
|       ├── Reduce.h
|       ├── SampleFromVoxels.h
|       └── Stencils.h
```
