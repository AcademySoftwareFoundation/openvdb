// Capstone: GPU level-set ray-march renderer.
// Build an SDF sphere -> move to device -> one thread per pixel ->
// ZeroCrossing to find the surface -> Lambertian shade via SDF gradient
// normal -> write PPM.
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>        // ZeroCrossing lives here
#include <nanovdb/math/Stencils.h>   // GradStencil

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

    // --- pinhole camera in WORLD space ---
    const nanovdb::Vec3f eye(0.f, 0.f, -300.f);
    const float aspect     = float(W) / float(H);
    const float tanHalfFov = tanf(0.5f * 45.f * 3.14159265f / 180.f);
    const float u = (2.f * ((x + 0.5f) / W) - 1.f) * tanHalfFov * aspect;
    const float v = (1.f - 2.f * ((y + 0.5f) / H)) * tanHalfFov;  // flip y
    nanovdb::Vec3f dir(u, v, 1.f);
    dir.normalize();

    nanovdb::math::Ray<float> wRay(eye, dir);
    auto iRay = wRay.worldToIndexF(*grid);          // ray in index space

    auto acc = grid->getAccessor();
    nanovdb::Coord ijk;
    float t = 0.f, val = 0.f;

    unsigned char shade = 0;                        // background (black)
    if (nanovdb::math::ZeroCrossing(iRay, acc, ijk, val, t)) {
        nanovdb::math::GradStencil<GridT> stencil(*grid);
        stencil.moveTo(ijk);
        nanovdb::Vec3f n = stencil.gradient();      // SDF gradient = outward normal
        n.normalize();
        const float diff = fmaxf(0.f, n.dot(lightDir));
        const float I = fminf(1.f, ambient + (1.f - ambient) * diff);
        shade = (unsigned char)(255.f * I);         // greyscale
    }
    img[pid] = shade; img[pid + 1] = shade; img[pid + 2] = shade;
}

int main(int argc, char** argv)
{
    // Shading parameters (passed into the kernel). Defaults, overridable on the
    // command line:  ./raymarch [ambient] [lx ly lz]
    float ambient = 0.15f;
    nanovdb::Vec3f lightDir(-0.577f, 0.577f, -0.577f);   // (-1,1,-1) normalized
    if (argc >= 2) ambient = (float)std::atof(argv[1]);
    if (argc >= 5) lightDir = nanovdb::Vec3f((float)std::atof(argv[2]),
                                             (float)std::atof(argv[3]),
                                             (float)std::atof(argv[4]));
    lightDir.normalize();

    // 1. Build a level-set sphere SDF on the host (radius 100, voxel size 1).
    auto handle = nanovdb::tools::createLevelSetSphere<float>(
        /*radius=*/100.0, /*center=*/nanovdb::Vec3d(0.0),
        /*voxelSize=*/1.0, /*halfWidth=*/3.0);

    // 2. Move it to the device. copy<DeviceBuffer>() fills the HOST side of the
    //    dual buffer; deviceUpload() pushes the bytes to the device. Only then
    //    is deviceGrid() non-null.
    auto devHandle = handle.copy<nanovdb::cuda::DeviceBuffer>();
    devHandle.deviceUpload();
    const GridT* dGrid = devHandle.deviceGrid<float>();
    if (!dGrid) { std::printf("no device grid\n"); return 1; }

    // 3. Render.
    const int W = 512, H = 512;
    unsigned char* dImg = nullptr;
    cudaMalloc(&dImg, size_t(W) * H * 3);
    const dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    render<<<grid, block>>>(dGrid, dImg, W, H, lightDir, ambient);
    cudaDeviceSynchronize();
    if (auto e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // 4. Copy back and write PPM.
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
