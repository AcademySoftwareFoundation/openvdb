# Source tree structure {#SourceTree}

#### Overview of the directory tree of NanoVDB.

* [NanoVDB.h](../NanoVDB.h) C++11 implementation of the core data structure and its access methods.
*  [CNanoVDB.h](../CNanoVDB.h) C99 implementation of the core data structure and its access methods.
*  [PNanoVDB.h](../PNanoVDB.h) C99 implementation of the core data structure and its access methods. Unlike CNanoVDB this version is pointer-less and supports virtually all graphics APIs.
* [util/GridHandle.h](../util/GridHandle.h) defines a handler for the memory allocated to a NanoVDB grid.
* [util/IO.h](../util/IO.h) implements I/O support. 
* [util/OpenToNanoVDB.h](../util/OpenToNanoVDB.h) defines the converter from OpenVDB to NanoVDB and obviously depends on the OpenVDB library (as the only header file). 
* [Ray.h](../util/Ray.h) Ray class.
* [HDDA.h](../util/HDDA.h) HDDA related.
* [SampleFromVoxels.h](../util/SampleFromVoxels.h) interpolation.

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

### Copyright Contributors to the OpenVDB Project
### SPDX-License-Identifier: MPL-2.0