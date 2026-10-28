# Source tree structure {#NanoVDB_SourceTree}

#### Overview of the directory tree of NanoVDB.

* [NanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/NanoVDB.h) C++11 implementation of the core data structure and its access methods.
* [CNanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/CNanoVDB.h) Incomplete but useable C99 implementation of the core data structure and its access methods.  Designed in particular for use in OpenCL kernels.  Note that this relies on zero-sized arrays for the _reserved padding, so will not work on all compilers (with MSVC being a particular example)
* [PNanoVDB.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/PNanoVDB.h) C99 implementation of the core data structure and its access methods. More complete coverage than CNanoVDB.  This version is pointer-less and supports virtually all graphics APIs.
* [GridHandle.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/GridHandle.h) defines a handler for the memory allocated to a NanoVDB grid.
* [io/IO.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/io/IO.h) implements I/O support.
* [tools/CreateNanoGrid.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/tools/CreateNanoGrid.h) defines the converter from OpenVDB to NanoVDB and obviously depends on the OpenVDB library (as the only header file).
* [math/Ray.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/math/Ray.h) Ray class.
* [math/HDDA.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/math/HDDA.h) HDDA related.
* [math/SampleFromVoxels.h](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/math/SampleFromVoxels.h) interpolation.

```
$ tree
.
├── CMakeLists.txt
├── cmd
│   ├── CMakeLists.txt
│   ├── convert
│   │   └── nanovdb_convert.cc
│   ├── print
│   │   └── nanovdb_print.cc
│   ├── updateFiles.sh
│   └── validate
│       └── nanovdb_validate.cc
├── CNanoVDB.h
├── cuda
│   ├── DeviceBuffer.h
│   ├── GridHandle.cuh
│   └── NodeManager.cuh
├── docs
│   ├── CMakeLists.txt
│   ├── codingstyle.txt
│   └── doxygen-config
├── examples
│   ├── CMakeLists.txt
│   ├── ex_bump_pool_buffer
│   │   └── bump_pool_buffer.cc
│   ├── ex_collide_level_set
│   │   ├── common.h
│   │   ├── main.cc
│   │   ├── nanovdb.cu
│   │   └── openvdb.cc
│   ├── ex_index_grid_cuda
│   │   ├── index_grid_cuda.cc
│   │   └── index_grid_cuda_kernel.cu
│   ├── ex_make_custom_nanovdb
│   │   └── make_custom_nanovdb.cc
│   ├── ex_make_custom_nanovdb_cuda
│   │   ├── make_custom_nanovdb_cuda.cc
│   │   └── make_custom_nanovdb_cuda_kernel.cu
│   ├── ex_make_funny_nanovdb
│   │   └── make_funny_nanovdb.cc
│   ├── ex_make_nanovdb_sphere
│   │   └── make_nanovdb_sphere.cc
│   ├── ex_make_typed_grids
│   │   └── make_typed_grids.cc
│   ├── ex_map_pool_buffer
│   │   └── map_pool_buffer.cc
│   ├── ex_modify_nanovdb_thrust
│   │   ├── modify_nanovdb_thrust.cc
│   │   └── modify_nanovdb_thrust.cu
│   ├── ex_nodemanager_cuda
│   │   ├── nodemanager_cuda.cc
│   │   └── nodemanager_cuda_kernel.cu
│   ├── ex_openvdb_to_nanovdb
│   │   └── openvdb_to_nanovdb.cc
│   ├── ex_openvdb_to_nanovdb_accessor
│   │   └── openvdb_to_nanovdb_accessor.cc
│   ├── ex_openvdb_to_nanovdb_cuda
│   │   ├── openvdb_to_nanovdb_cuda.cc
│   │   └── openvdb_to_nanovdb_cuda_kernel.cu
│   ├── ex_raytrace_fog_volume
│   │   ├── common.h
│   │   ├── main.cc
│   │   ├── nanovdb.cu
│   │   └── openvdb.cc
│   ├── ex_raytrace_level_set
│   │   ├── common.h
│   │   ├── main.cc
│   │   ├── nanovdb.cu
│   │   └── openvdb.cc
│   ├── ex_read_nanovdb_sphere
│   │   └── read_nanovdb_sphere.cc
│   ├── ex_read_nanovdb_sphere_accessor
│   │   └── read_nanovdb_sphere_accessor.cc
│   ├── ex_read_nanovdb_sphere_accessor_cuda
│   │   ├── read_nanovdb_sphere_accessor_cuda.cu
│   │   └── read_nanovdb_sphere_accessor_cuda_kernel.cu
│   ├── ex_util
│   │   └── ComputePrimitives.h
│   ├── ex_voxels_to_grid_cuda
│   │   └── ex_voxels_to_grid_cuda.cu
│   ├── ex_vox_to_nanovdb
│   │   ├── vox_to_nanovdb.cc
│   │   └── VoxToNanoVDB.h
│   └── ex_write_nanovdb_grids
│       └── write_nanovdb_grids.cc
├── GridHandle.h
├── HostBuffer.h
├── io
│   └── IO.h
├── math
│   ├── CSampleFromVoxels.h
│   ├── DitherLUT.h
│   ├── HDDA.h
│   ├── Math.h
│   ├── Ray.h
│   ├── SampleFromVoxels.h
│   └── Stencils.h
├── NanoVDB.h
├── NodeManager.h
├── PNanoVDB.h
├── Readme.md
├── tools
│   ├── CreateNanoGrid.h
│   ├── CreatePrimitives.h
│   ├── cuda
│   │   ├── AddBlindData.cuh
│   │   ├── GridChecksum.cuh
│   │   ├── GridStats.cuh
│   │   ├── GridValidator.cuh
│   │   ├── IndexToGrid.cuh
│   │   ├── PointsToGrid.cuh
│   │   └── SignedFloodFill.cuh
│   ├── GridBuilder.h
│   ├── GridChecksum.h
│   ├── GridStats.h
│   ├── GridValidator.h
│   └── NanoToOpenVDB.h
├── unittest
│   ├── CMakeLists.txt
│   ├── pnanovdb_validate_strides.h
│   ├── TestNanoVDB.cc
│   ├── TestNanoVDB.cu
│   └── TestOpenVDB.cc
└── util
    ├── CpuTimer.h
    ├── CreateNanoGrid.h
    ├── cuda
    │   ├── CudaAddBlindData.cuh
    │   ├── CudaDeviceBuffer.h
    │   ├── CudaGridChecksum.cuh
    │   ├── CudaGridHandle.cuh
    │   ├── CudaGridStats.cuh
    │   ├── CudaGridValidator.cuh
    │   ├── CudaIndexToGrid.cuh
    │   ├── CudaNodeManager.cuh
    │   ├── CudaPointsToGrid.cuh
    │   ├── CudaSignedFloodFill.cuh
    │   ├── CudaUtils.h
    │   ├── GpuTimer.h
    │   ├── Timer.h
    │   └── Util.h
    ├── DitherLUT.h
    ├── ForEach.h
    ├── GridBuilder.h
    ├── GridChecksum.h
    ├── GridStats.h
    ├── GridValidator.h
    ├── HDDA.h
    ├── HostBuffer.h
    ├── Invoke.h
    ├── IO.h
    ├── NanoToOpenVDB.h
    ├── NodeManager.h
    ├── OpenToNanoVDB.h
    ├── PrefixSum.h
    ├── Primitives.h
    ├── Range.h
    ├── Ray.h
    ├── Reduce.h
    ├── SampleFromVoxels.h
    ├── Stencils.h
    ├── Timer.h
    └── Util.h
```
