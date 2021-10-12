# FAQ {#FAQ}

* **What is NanoVDB?**

  As the name indicates it's a mini-version of the much bigger OpenVDB library, both in terms of functionality and scope. In fact, a stand-alone C++11 implementation of NanoVDB is available in the file [NanoVDB.h](../include/nanovdb/NanoVDB.h) and the C99 equivalent in the files [CNanoVDB.h](../include/cnanovdb/CNanoVDB.h) and [PNanoVDB.h](../include/cnanovdb/PNanoVDB.h). However, NanoVDB offers one major advantage over OpenVDB, namely support for GPUs. In short, NanoVDB is a standalone static-topology implementation of the well-known sparse volumetric VDB data structure. In other words, while values can be modified in a NanoVDB grid its tree topology cannot.

  **What graphics API does NanoVDB support?**

  We have test NanoVDB with CUDA, OpenCL, OpenGL, DirectX 12, OptiX, HLSL, and GLSL. We are also working on adding support for WebGL.

* **What are the advantages of NanoVDB?**

  Other then the fact that it runs on the GPU, NanoVDB offers several other advantages over OpenVDB. For its limited applications it's generally faster, uses slightly less memory, is implemented in few header files, and has virtually no dependencies on external libraries (other then C++11 or C99). While NanoVDB inherits most of the memory and data access advantages of OpenVDB (e.g. fast random-access by means of inverse tree-traversal), the improved memory layout of NanoVDB actually results in faster read access performance.

* **What are the limitations of NanoVDB?**

  Other than the fact that NanoVDB obviously lacks many of the features and tools of OpenVDB, the most important limitation of NanoVDB is the fact that it *assumes the topology of the tree structure to be static*. Thus, it is possible to modify the values in a NanoVDB grid, but not its topology.

* **Does NanoVDB work on the CPU as well as the GPU?**

  Yes! In fact, it provides performance advantages on the CPU over OpenVDB, primarily due to the defragmented and more cache-friendly memory layout for spatially coherent voxel access.

* **What is NanoVDB intended for?**

  Rendering (e.g. ray-tracing) on the GPU or CPU, and other applications that require fast random access into a sparse volumetric data structure (e.g. collision detection for cloth simulations or static boundary conditions in fluid simulations). Additionally, since NanoVDB is essentially a slimmed-down version of the core data structure in OpenVDB, NanoVDB also serves as a great tool to learn about the implementation details of VDB, and can even serve as a starting point for your own sparse data structure.

* **Does NanoVDB depend on and require CUDA?**

  No, NanoVDB is by design agnostic and works with most graphics APIs, e.g. OpenGL, OpenCL, OptiX, CUDA, HLSL, GLSL, DirectX, and even the CPU! The data structure itself (defined in [NanoVDB.h](../include/nanovdb/NanoVDB.h) and [PNanoVDB.h](../include/nanovdb/PNanoVDB.h)) only depends on C++11 or C99. However, some of the auxiliary math classes are optionally optimized with intrinsic CUDA calls (e.g. fma in Vec3) and some of the [Allocator](../include/nanovdb/Util.h) uses CUDA for explicit memory management. It should be straightforward to replace (or remove) these CUDA dependencies if you so desire.

* **How does the data structure of NanoVDB compare to that of OpenVDB?**

  Conceptually they are essentially identical (NanoVDB uses the exact same tree configuration), but there are some implementation differences to minimize dependencies on external libraries in NanoVDB (e.g. no boost and std containers), and more importantly, NanoVDB has additional meta-data encoded in tree nodes specifically to accelerate ray-tracing (e.g. min/max values and AABB of active values). Also, NanoVDB is pointer-less and occupies a contiguous block of memory, i.e. de-fragmentized relative to OpenVDB. Effectively, NanoVDB is a linearized version of the well-known OpenVDB data structure with a few additional information baked into the tree nodes!

* **Are there known issues with NanoVDB?**

  While the are currently no known bugs, there are certainly lots of room for improvements and enhancements. If you have issues or ideas for new feature in
  NanoVDB please let us know, e.g email ken.museth@gmail.com.

* **Is NanoVDB production-ready?**

  While NanoVDB is still a beta release, it is stable and useful enough to be used
  in production environments. In fact, NanoVDB is already part of commercial software
  like Houdini version 18.5. The only aspect of NanoVDB that is still considered volatile
  is the .nvdb file format. The reason is simply that in this file format NanoVDB grids are encoded exactly the same way out-of-core as in-core. Consequently, any changes in the in-memory representation of NanoVDB grids will also change the .nvdb file format. We typically capture this with a bump of the major version number, e.g. 1.3.4 -> 2.0.0. So for now we do not recommend using the NanoVDB file format for long-time storage, though it works great for temporary caching!

### Copyright Contributors to the OpenVDB Project
### SPDX-License-Identifier: MPL-2.0