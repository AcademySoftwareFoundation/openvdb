# Source tree structure {#SourceTree}

#### Overview of the directory tree of NanoVDB.

* [NanoVDB.h](../nanovdb/NanoVDB.h) defines the core data structure and its access methods. 
* [GridHandle.h](../nanovdb/util/GridHandle.h) defines a handler for the memory allocated to a NanoVDB grid.
* [IO.h](../nanovdb/util/IO.h) implements I/O support. 
* [OpenToNanoVDB.h](../include/nanovdb/OpenToNanoVDB.h) defines the converter from OpenVDB to NanoVDB and obviously depends on the OpenVDB library (as the only header file). 
* [Ray.h](../nanovdb/util/Ray.h) Ray class.
* [HDDA.h](../nanovdb/util/HDDA.h) HDDA related.
* [SampleFromVoxels.h](../nanovdb/util/SampleFromVoxels.h) interpolation.

```bash
foo@bar:~$ tree
.
├── 
├── nanovdb
│   ├── CNanoVDB.h
│   ├── NanoVDB.h
│   └── util
│       ├── CSampleFromVoxels.h
│       ├── CudaDeviceBuffer.h
│       ├── GridBuilder.h
│       ├── GridHandle.h
│       ├── HDDA.h
│       ├── HostBuffer.h
│       ├── IO.h
│       ├── NanoToOpenVDB.h
│       ├── OpenToNanoVDB.h
│       ├── Ray.h
│       └── SampleFromVoxels.h
```

### Copyright Contributors to the OpenVDB Project
### SPDX-License-Identifier: MPL-2.0