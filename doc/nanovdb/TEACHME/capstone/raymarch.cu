// Capstone scaffold: GPU level-set ray-march renderer.
//
// The host plumbing is DONE (build SDF, upload to device, launch, read back,
// write PPM) and the light direction + ambient level are handed to the kernel
// as parameters. Your job is the kernel body — the per-pixel ray-march + shade.
// As shipped it compiles and runs but only writes a placeholder gradient.
//
// Build:  cmake -S . -B build && cmake --build build
// Run:    ./build/raymarch [ambient] [lx ly lz]   (writes sphere.ppm)
// View:   python3 ppm2png.py sphere.ppm sphere.png   (then open sphere.png)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>        // nanovdb::math::ZeroCrossing
#include <nanovdb/math/Stencils.h>   // nanovdb::math::GradStencil

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using GridT = nanovdb::NanoGrid<float>;

__global__ void render(const GridT* grid, unsigned char* img, int W, int H,
                       nanovdb::Vec3f lightDir, float ambient)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pid = (y * W + x) * 3;

    // ===================== YOUR WORK STARTS HERE =====================
    // Placeholder so it compiles & runs: a horizontal grey gradient.
    unsigned char shade = (unsigned char)(255.f * x / W);

    // The aesthetic inputs are given to you as parameters: `lightDir` and
    // `ambient` (set in main, overridable on the command line). You write the
    // ray-march and the shading math. Signatures are in the cheat sheet.
    //
    //   1. Camera ray (WORLD space). Pinhole: an eye point and, per pixel, a
    //      direction through the image plane. You decide the math — remember
    //      aspect ratio and that image y grows downward. Wrap it in a Ray.
    //
    //   2. The SDF lives in INDEX space, so convert your world ray to index
    //      space before you trace it. (Ray has a method for this.)
    //
    //   3. Find where the ray first crosses the surface (the SDF sign change).
    //      Module 5 gave you a helper in math/HDDA.h that drives an HDDA and
    //      reports the hit voxel, the value, and the ray parameter.
    //
    //   4. On a hit, the surface normal is the (normalized) SDF gradient at the
    //      hit voxel. Module 5's gradient stencil computes it. (Watch its
    //      template parameter.)
    //
    //   5. Shade: a Lambertian intensity from the normal and the provided
    //      `lightDir`, lifted off the floor by `ambient`; write it as greyscale.
    //      On a MISS, leave the background.
    // ====================== YOUR WORK ENDS HERE ======================

    img[pid] = shade; img[pid + 1] = shade; img[pid + 2] = shade;
}

int main(int argc, char** argv)
{
    // Shading parameters passed into the kernel. Defaults, overridable on the
    // command line:  ./raymarch [ambient] [lx ly lz]
    float ambient = 0.15f;
    nanovdb::Vec3f lightDir(-0.577f, 0.577f, -0.577f);   // (-1,1,-1) normalized
    if (argc >= 2) ambient = (float)std::atof(argv[1]);
    if (argc >= 5) lightDir = nanovdb::Vec3f((float)std::atof(argv[2]),
                                             (float)std::atof(argv[3]),
                                             (float)std::atof(argv[4]));
    lightDir.normalize();

    // Build a level-set sphere SDF on the host (radius 100, voxel size 1).
    auto handle = nanovdb::tools::createLevelSetSphere<float>(
        /*radius=*/100.0, /*center=*/nanovdb::Vec3d(0.0),
        /*voxelSize=*/1.0, /*halfWidth=*/3.0);

    // Move to the device. copy<DeviceBuffer>() fills the HOST side of the dual
    // buffer; deviceUpload() pushes it to the GPU — skip it and deviceGrid()
    // returns nullptr.
    auto devHandle = handle.copy<nanovdb::cuda::DeviceBuffer>();
    devHandle.deviceUpload();
    const GridT* dGrid = devHandle.deviceGrid<float>();
    if (!dGrid) { std::printf("no device grid\n"); return 1; }

    const int W = 512, H = 512;
    unsigned char* dImg = nullptr;
    cudaMalloc(&dImg, size_t(W) * H * 3);
    const dim3 block(16, 16), gridDim((W + 15) / 16, (H + 15) / 16);
    render<<<gridDim, block>>>(dGrid, dImg, W, H, lightDir, ambient);
    cudaDeviceSynchronize();
    if (auto e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    std::vector<unsigned char> img(size_t(W) * H * 3);
    cudaMemcpy(img.data(), dImg, img.size(), cudaMemcpyDeviceToHost);
    cudaFree(dImg);

    FILE* f = std::fopen("sphere.ppm", "wb");
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::fwrite(img.data(), 1, img.size(), f);
    std::fclose(f);
    std::printf("wrote sphere.ppm (%dx%d)\n", W, H);
    return 0;
}
