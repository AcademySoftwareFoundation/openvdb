# NanoVDB Hello World Examples {#NanoVDB_HelloWorld}

[The first example](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_openvdb_to_nanovdb_accessor/openvdb_to_nanovdb_accessor.cc) shows how to convert an OpenVDB level set sphere into a NanoVDB level set, use accessors to print out multiple values from both grids, and save the NanoVDB grid to file. Note that **this example depends on both OpenVDB and NanoVDB**.

```cpp
#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/util/CreateNanoGrid.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/IO.h>

// Convert an openvdb level set sphere into a nanovdb, use accessors to print out multiple values from both
// grids and save the NanoVDB grid to file.
// Note, main depends on BOTH OpenVDB and NanoVDB.
int main()
{
    try {
        // Create an OpenVDB grid (here a level set surface but replace this with your own code)
        auto srcGrid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(100.0f, openvdb::Vec3f(0.0f), 1.0f);

        // Convert the OpenVDB grid, srcGrid, into a NanoVDB grid handle.
        auto handle = nanovdb::createNanoGrid(*srcGrid);

        // Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
        auto* dstGrid = handle.grid<float>();

        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        // Get accessors for the two grids. Note that accessors only accelerate repeated access!
        auto dstAcc = dstGrid->getAccessor();
        auto srcAcc = srcGrid->getAccessor();

        // Access and print out a cross-section of the narrow-band level set from the two grids
        for (int i = 97; i < 104; ++i) {
            printf("(%3i,0,0) OpenVDB cpu: % -4.2f, NanoVDB cpu: % -4.2f\n", i, srcAcc.getValue(openvdb::Coord(i, 0, 0)), dstAcc.getValue(nanovdb::Coord(i, 0, 0)));
        }

        nanovdb::io::writeGrid("data/sphere.nvdb", handle); // Write the NanoVDB grid to file and throw if writing fails
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
```

[The second example](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor/read_nanovdb_sphere_accessor.cc) reads a NanoVDB grid from a file (the one saved in the previous example) and prints out multiple values. Note that **this example does not depend on OpenVDB (nor CUDA), only NanoVDB**.

```cpp
#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host

/// @brief Read a NanoVDB grid from a file and print out multiple values.
///
/// @note Note This example does NOT depend on OpenVDB (nor CUDA), only NanoVDB.
int main()
{
    try {
        auto handle = nanovdb::io::readGrid("data/sphere.nvdb"); // reads first grid from file

        auto* grid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float

        if (!grid)
            throw std::runtime_error("File did not contain a grid with value type float");

        auto acc = grid->getAccessor(); // create an accessor for fast access to multiple values
        for (int i = 97; i < 104; ++i) {
            printf("(%3i,0,0) NanoVDB cpu: % -4.2f\n", i, acc.getValue(nanovdb::Coord(i, 0, 0)));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
```

[The third example](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda.cc) reads a NanoVDB grid from a file (the one saved in the first example) and prints out multiple values on both the CPU and GPU. Note that **this example does NOT depend on OpenVDB, only NanoVDB and CUDA**.

```cpp
#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/util/cuda/CudaDeviceBuffer.h> // required for CUDA memory management

extern "C" void launch_kernels(const nanovdb::NanoGrid<float>*,
                               const nanovdb::NanoGrid<float>*,
                               cudaStream_t stream);

/// @brief Read a NanoVDB grid from a file and print out multiple values on both the cpu and gpu.
///
/// @note Note This example does NOT depend on OpenVDB, only NanoVDB and CUDA.
int main()
{
    try {
        // returns a GridHandle using CUDA for memory management.
        auto handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>("data/sphere.nvdb");

        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);

        handle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously

        auto* cpuGrid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* deviceGrid = handle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!deviceGrid || !cpuGrid)
            throw std::runtime_error("GridHandle did not contain a grid with value type float");

        launch_kernels(deviceGrid, cpuGrid, stream); // Call a host method to print a grid values on both the CPU and GPU

        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}
```

NVCC requires the CUDA kernel to be defined in [a separate .cu file](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda.cu):

```cpp
#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessible on both the host and device
#include <stdio.h> // for printf

// This is called by the host only
void cpu_kernel(const nanovdb::NanoGrid<float>* cpuGrid)
{
    auto cpuAcc = cpuGrid->getAccessor();
    for (int i = 97; i < 104; ++i) {
        printf("(%3i,0,0) NanoVDB cpu: % -4.2f\n", i, cpuAcc.getValue(nanovdb::Coord(i, 0, 0)));
    }
}

// This is called by the device only
__global__ void gpu_kernel(const nanovdb::NanoGrid<float>* deviceGrid)
{
    if (threadIdx.x > 6)
        return;
    int  i = 97 + threadIdx.x;
    auto gpuAcc = deviceGrid->getAccessor();
    printf("(%3i,0,0) NanoVDB gpu: % -4.2f\n", i, gpuAcc.getValue(nanovdb::Coord(i, 0, 0)));
}

// This is called by the client code on the host
extern "C" void launch_kernels(const nanovdb::NanoGrid<float>* deviceGrid,
                               const nanovdb::NanoGrid<float>* cpuGrid,
                               cudaStream_t                    stream)
{
    // Launch the device kernel asynchronously
    gpu_kernel<<<1, 64, 0, stream>>>(deviceGrid);

    // Launch the host "kernel" (synchronously)
    cpu_kernel(cpuGrid);
}
```
