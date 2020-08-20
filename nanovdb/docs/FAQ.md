# FAQ {#FAQ}

* **What is NanoVDB?**

  As the name indicates it's a mini-version of the much bigger OpenVDB library, both in terms of functionality and scope. In fact, a stand-alone C++11 implementation of NanoVDB is available in the file [NanoVDB.h](../include/nanovdb/NanoVDB.h) and the C99 equivalent in the file [CNanoVDB.h](../include/cnanovdb/CNanoVDB.h). However, NanoVDB offers one major advantage over OpenVDB, namely support for GPUs. In short, NanoVDB is a standalone read-only implementation of the well-known sparse volumetric VDB data structure.

* **What are the advantages of NanoVDB?**

  Other then the fact that it runs on the GPU, NanoVDB offers several other advantages over OpenVDB. For its limited applications it's generally faster, uses slightly less memory, is implemented in few header files, and has virtually no dependencies on external libraries (other then C++11 or C99). While NanoVDB inherits most of the memory and data access advantages of OpenVDB (e.g. fast random-access by means of inverse tree-traversal), the improved memory layout of NanoVDB actually results in faster read access performance.

* **What are the limitations of NanoVDB?**

  Other than the fact that NanoVDB obviously lacks many of the features and tools of OpenVDB, the most important limitation of NanoVDB is the fact that it *assumes the topology of the tree structure to be static*. In other words, NanoVDB is currently a read-only data structure which can only be constructed from existing OpenVDB volumes (on the CPU). This limitation will be relaxed in the future.

* **Does NanoVDB work on the CPU as well as the GPU?**

  Yes! In fact, it provides performance advantages on the CPU over OpenVDB, primarily due to the defragmented and more cache-friendly memory layout for spatially coherent voxel access.

* **What is NanoVDB intended for?**

  Rendering (e.g. ray-tracing) on the GPU or CPU, and other applications that require fast random access into a sparse volumetric data structure (e.g. collision detection for cloth simulations or static boundary conditions in fluid simulations). Additionally, since NanoVDB is essentially a slimmed-down version of the core data structure in OpenVDB, NanoVDB also serves as a great tool to learn about the implementation details of VDB, and can even serve as a starting point for your own sparse data structure.

* **Does NanoVDB depend on and require CUDA?**

  No! The data structure itself (defined in [NanoVDB.h](../include/nanovdb/NanoVDB.h)) only depends on C++11, but some of the auxiliary math classes can (and have been) improved with intrinsic CUDA calls (e.g. fma in Vec3) and the [Allocator](../include/nanovdb/Util.h) uses CUDA for explicit memory management. It should be straightforward to replace (or remove) these CUDA dependencies if you so desire. Having said that, we have only tested NanoVDB on the GPU with CUDA, and as such we cannot speak to the performance benefits of NanoVDB in other contexts than CUDA.

* **How does the data structure of NanoVDB compare to that of OpenVDB?**

  Conceptually they are essentially identical (NanoVDB uses the exact same tree configuration), but there are some implementation differences to minimize dependencies on external libraries in NanoVDB (e.g. no boost and std containers), and more importantly, NanoVDB has additional meta-data encoded in tree nodes specifically to accelerate ray-tracing (e.g. min/max values and AABB of active values). Also, NanoVDB is pointer-less and occupies a contiguous block of memory, i.e. de-fragmentized relative to OpenVDB. Effectively, NanoVDB is a linearized version of the well-known OpenVDB data structure with a few additional information baked into the tree nodes!

* **Are there known issues with NanoVDB?**

  While the are currently no known bugs, there are certainly lots of room for improvements and enhancements. Some optimizations (e.g. AABB intersection) are currently disabled due to issues and important features are still missing (e.g. volume vs surface ray-intersections). Some of these issues and missing features are listed in the [Missing Features](#-Missing-features) section.

* **Is NanoVDB production-ready?**

  That very much depends on your application and requirements. While the core NanoVDB data structure is robust and unlikely to change, its API (and ABI) might change based on the feedback we are soliciting from clients. The file format, on the other hand, is almost certainly going to change in order to support multiple grids and vector types. However, if your question is simply "is NanoVDB working today", then the answer is "yes to the best of our knowledge"! While the NanoVDB project is only a few months old, its underlying VDB data structure is over a decade old and more to the point it has been battle-tested since it was open-sourced in 2012. However, if you encounter any bugs please please let us know :)

* **Will NanoVDB be open-sourced?**

  While NanoVDB is currently shared with a restrictive license under NDA, there is consensus at NVIDIA to eventually contribute parts of NanoVDB back into to the OpenVDB library. When this happens and exactly what parts of NanoVDB depends on the feedback we receive from the few vendors we have decided to share it with. As such, it is important that you let us know if you continue to be interested in adopting NanoVDB and of course if there are missing features or blocking issues (including licensing concerns).

### Copyright Contributors to the OpenVDB Project
### SPDX-License-Identifier: MPL-2.0