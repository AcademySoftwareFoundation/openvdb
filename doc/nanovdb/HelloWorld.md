# NanoVDB Hello World Examples {#NanoVDB_HelloWorld}

The example snippets below are pulled directly from compiled example source
files via Doxygen's `\snippet` mechanism, so the code shown here is always
in sync with what is built and tested in CI. Enable
`-DNANOVDB_BUILD_EXAMPLES=ON` to build them locally.

## Convert an OpenVDB grid to NanoVDB and use accessors

[The first example](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_openvdb_to_nanovdb_accessor/openvdb_to_nanovdb_accessor.cc)
shows how to convert an OpenVDB level set sphere into a NanoVDB level set, use
accessors to print out multiple values from both grids, and save the NanoVDB
grid to file. Note that **this example depends on both OpenVDB and NanoVDB**.

\snippet nanovdb/nanovdb/examples/ex_openvdb_to_nanovdb_accessor/openvdb_to_nanovdb_accessor.cc openvdb_to_nanovdb_accessor

## Read a NanoVDB grid from a file (host only)

[The second example](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor/read_nanovdb_sphere_accessor.cc)
reads a NanoVDB grid from a file (the one saved in the previous example) and
prints out multiple values. Note that **this example does not depend on
OpenVDB (nor CUDA), only NanoVDB**.

\snippet nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor/read_nanovdb_sphere_accessor.cc read_nanovdb_sphere_accessor

## Read a NanoVDB grid and access it on the CPU and GPU

[The third example](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda.cu)
reads a NanoVDB grid from a file (the one saved in the first example) and
prints out multiple values on both the CPU and GPU. Note that **this example
does NOT depend on OpenVDB, only NanoVDB and CUDA**.

\snippet nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda.cu read_nanovdb_sphere_accessor_cuda

NVCC requires the CUDA kernel to be defined in
[a separate .cu file](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda_kernel.cu):

\snippet nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda_kernel.cu read_nanovdb_sphere_accessor_cuda_kernel
