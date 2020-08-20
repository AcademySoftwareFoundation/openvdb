# Missing features {#ToDo}

- [x] **Volume ray-tracing**

    Currently, NanoVDB only supports ray-level-set intersections. Volume-ray-tracing will be added soon (similarly to that found in OpenVDB).

- [ ] **Use AABB to accelerate ray marching**

  One of the differences between the data structure in OpenVDB and NanoVDB is the axis-aligned bounding-boxes of the active values encoded into the tree nodes of the latter. This allows for early ray termination but is currently disabled due to some precision issues that will (hopefully) be resolved in the near future. The expectation is that this will improve the performance of the HDDA for both surface and volume ray-tracing.

 - [x] **Sequential access**

   ~~NanoVDB currently offers only random (i.e. coordinate-based) vs sequential (i.e. offset-based) access to tree nodes and voxel values. The former is useful for applications like ray-tracing and stencil lookups, where the latter is useful for computations on grid, like filtering.~~   

- [ ]  **Adopt high-order interpolation in benchmark test**
  
   The included benchmark test is designed to measure the performance of the ray-marching and the [HDDA](../include/nanovdb/util/HDDA.h). As a result, the interpolation schemes are left as zero-order which can result in aliasing artifacts. As such, we will probably change the benchmark test to use at least tri-linear interpolation in the near future. However, note that all the higher-order interpolation schemes in [SampleFromVoxels.h](../include/nanovdb/util/SampleFromVoxels.h), have been unit-tested and should already be ready for adoption in client code.

- [x]  **Interpolation of gradients**

     ~~Currently SampleFromVoxels only interpolates the scalar values of a grid, but we need interpolated gradients as well.~~


- [x] **Conversion of OpenVDB FOG volumes to NanoVDB**

     ~~Currently we only support conversion of OpenVDB level set volumes to NanoVDB. We obviously need to support FOG volumes as well.~~

- [x] **Conversion from NanoVDB to OpenVDB**

     ~~Currently we can only convert an OpenVDB grid into a NanoVDB grid, but we wih to support conversion in the other direction as well, which will be very useful once we can modify grids on the GPU.~~ 

- [x] **Support for vector grids**

    ~~Currently, NanoVDB does only support scalar grid, but vector grids will be added shortly (should be straightforward since the voxel type is templated as in OpenVDB.~~

- [x] **Optional compression of file format**

    ~~NanoVDB has it's only (simple) streaming file format that is currently now compressed, resulting is a large file size. We plan to add optional compression of the grid data.~~

 - [x] **Multi-grid support in file format**

    ~~NanoVDB has it's only (simple) streaming file format that is currently limited to a single (scalar) grid. In the future we expect to add support for multiple grids of different types (as in OpenVDB).~~

 - [x] **Support reading of files regardless of grid type**

    ~~Currently the grid type has to be known before a nanovdb can be read. By adding a register of supported grid types we can all any file format to be read, and the correct gtid type can subsequently be derived.~~    

 - [ ] **Processing of voxel values on the GPU**

    While the tree topology of NanoVDB is assumed to be fixed, there are no restrictions on the actual voxel values. As such it should be straightforward to implement fast processing of grid values on the GPU, e.g. convolution and filtering of densities and even SDFs. If topology changes are captured during the construction of the NanoVDB (on the CPU) this technique could even be used for other types of compute-intensive grid operations, e.g. solving PDEs.

 - [x] **Improve memory layout with space-filling curves**

    ~~Currently, tree nodes in NanoVDB are laid out in memory in the same order they are visited (during breadth-first tree traversal). We expect that a re-ordering of the tree nodes according to a space-filling curve (e.g. based on Morton-codes) will improve spatial coherence during stencil operations like interpolation and finite-differencing.~~

 - [ ] **Motion blurring**

    Support for motion blurring of sparse volumetric data can potentially be added by encoding multiple temporal values in the tree structure.

### Copyright Contributors to the OpenVDB Project
### SPDX-License-Identifier: MPL-2.0