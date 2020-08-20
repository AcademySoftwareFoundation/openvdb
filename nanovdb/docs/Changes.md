# April 29

* Header files are now located in include/nanovdb and include/cnanovdb.
* The former contains C++11 implementaitons of data structures and tools.
* The latter contains C implementations of the core data structure and some of the interpolators.
* GridManager is not longer templeted on the grid type and can read any valid nvdb file.
* The root node optionally employs a single uint64_t key vs three int32_t coord key.
* Optional sorting of leaf nodes by means of morton codes.
* ReadAccessor now has a much smaller memory footprint (44 vs 68 bytes).
* Improved robustness of Serializer with more checks for unsupported inputs.
* Fixed bugs in TrilinearSampler and TricubicSampler.
* The affine Map in the nanovdb::Grid now encodes both a double and float representation.
* Minor optimization of the HDDA.
  
 # May 1

* Introduced Allocator class to switch between gcc, nvcc or other memory allocators. 
* Eliminated CUDA dependencies in main data stucture and core tools (limited to DefaultCudaAllocator.h).
* Added GridClass, from OpenVDB, to lable "level sets", "fog volumes", "staggered vector fields" and "unknown".  
* Each grid now encodes an optional c-string, with its name or blind meta data of arbitrary size.
* Replaced GridManager with two new classes, "File" and "GridHandler", which seperates I/O and memory management. 
* File format now supports an arbitary number of grids with different value types.
* The grid and file format now supports scalar and vector grids of type float, double, int32, int64, Vec3f, and Vec3d.

# May 11

* Fast "deserialization" (conversion) from a NanoVDB grid to an OpenVDB grid.
* Computation of interpolated gradients (TriLinearSampler::gradient)
* Writing and reading of nanovdb files now (optionally) support both ZIP and BLOSC compression.

# May 18

* File format now allows for multiple file headers, resulting from multiple write operations to a stream
* Added several improvements and bug fixes to File class (e.g. hash key collisions of grid names are now resolved)
* Grid now encodes information about its value-type, e.g. GridType::Float for NanoGrid<float>, which improves type safely
* Added hooks to support frustrum transforms in the future.
* Added the command-line tool nanovdb_print that prints meta information about grids in NanoVDB files.

# May 25

* Added the command-line tool nanovdb_convert that convertes between OpenVDB and NanoVDB files (both ways).
* Improved file layout for fast and easy extraction of meta data.
* Fix a bug in ZeroCrossing that could cause lockup in rare cases.

# June 1

* Added cmake build system
* 32 byte aligment
* nanovdb_viewer that perform real-time ray-tracing
* CNanoVDB.h is renamed CBabyVDB.h since it will be distributed to certain customers under NDA
* Added the ultra memory compact MaskGrid, which encodes just 1 bit per voxel!

# June 8

* nanovdb_viewer now supports five modes: CPU(serial), CPU(TBB), OpenGL, OpenCL and CUDA.
* 32 Byte alignment of in-core and out-of-core represnetaiton (disregarding optional compresion)
* Improvements to HDDA

# June 15

* Added support for blind-data associated with voxels in a NanoVDB grid.
* Serializer now has full support for PointIndexGrid, i.e. it can generate NanoVDB files of indices to points.
* Viewer has sample code for point rendering.
* Volume path-tracing sample optimizations and improvements.
* Generic TreeMarcher with a ray-node intersection API.

# June 22

* Serializer now has full support for PointDataGrid, i.e. it can generate NanoVDB files of points with arbitrary attributes.
* Added PointAccessor.
* Moved this respository from GitLab to a private GitHub (June 25).