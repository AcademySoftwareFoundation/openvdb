#include "Ops.h"

#include "detail/utils/cuda/Utils.cuh"

#include <c10/cuda/CUDACachingAllocator.h>

#include <cub/cub.cuh>


namespace fvdb {
namespace detail {
namespace ops {


/// @brief forEachJaggedElementCUDA callback to count the number of voxels to generate inside each triangle
///        and save it to a buffer
/// @param bidx Batch index
/// @param eidx Element index
/// @param transforms Array of transforms for each batch element
/// @param vertices JaggedAccessor for vertex positions in each mesh in the batch
/// @param triangles JaggedAccessor for triangle indices in each mesh in the batch
/// @param outNumSamplesPerTri Output buffer for the number of voxels to generate in each triangle
template <typename ScalarF, typename ScalarI>
__device__ void countVoxelsPerTriToCheck(int32_t bidx, int32_t eidx,
                                         const VoxelCoordTransform* transforms,
                                         const JaggedRAcc32<ScalarF, 2> vertices,
                                         const JaggedRAcc32<ScalarI, 2> triangles,
                                         TorchRAcc32<int32_t, 1> outNumSamplesPerTri) {
    using Vec3F = nanovdb::math::Vec3<ScalarF>;
    using Vec3I = nanovdb::math::Vec3<ScalarI>;

    const VoxelCoordTransform tx = transforms[bidx];
    const TorchRAcc32<ScalarI, 2> faces = triangles.data();
    const TorchRAcc32<ScalarF, 2> verts = vertices.data();

    // Voxel space triangle vertices
    const int64_t off = vertices.offsetStart(bidx);
    const Vec3F v1 = tx.apply(verts[off+faces[eidx][0]][0], verts[off+faces[eidx][0]][1], verts[off+faces[eidx][0]][2]);
    const Vec3F v2 = tx.apply(verts[off+faces[eidx][1]][0], verts[off+faces[eidx][1]][1], verts[off+faces[eidx][1]][2]);
    const Vec3F v3 = tx.apply(verts[off+faces[eidx][2]][0], verts[off+faces[eidx][2]][1], verts[off+faces[eidx][2]][2]);

    // Edges of triangle in voxel space
    const Vec3F e1 = v2 - v1;
    const Vec3F e2 = v3 - v1;

    // Spacing between samples to ensure coverage
    const ScalarF spacing = sqrt(3.0) / 3.0;  // This is very conservative spacing but fine for now

    // Number of samples to generate for this triangle
    const int32_t numU = ceil((e1.length() + spacing) / spacing);
    const int32_t numV = ceil((e2.length() + spacing) / spacing);
    const int32_t numVoxels = numU * numV;

    // Store a zero in the first element so we can do a cumulative sum later
    outNumSamplesPerTri[eidx+1] = numVoxels;
    if (eidx == 0) {
        outNumSamplesPerTri[0] = 0;
    }
}



template <typename ScalarF, typename ScalarI, template <typename T, int32_t D> typename TensorAccessor>
__global__ void generateSurfaceSamples(const VoxelCoordTransform* transforms,
                                       const JaggedRAcc32<ScalarF, 2> vertices,
                                       const JaggedRAcc32<ScalarI, 2> triangles,
                                       const TorchRAcc32<int64_t, 1> cumSamplesPerTri,
                                       TensorAccessor<int32_t, 2> outIJK,
                                       TensorAccessor<int16_t, 1> outJIdx) {
    using Vec3F = nanovdb::math::Vec3<ScalarF>;
    using Vec3I = nanovdb::math::Vec3<ScalarI>;

    const int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t numTris = cumSamplesPerTri.size(0) - 1;     // Total number of triangles in the mesh
    const int32_t totalSamples = cumSamplesPerTri[numTris];   // Total number of ijk samples we're going to generate

    if (tid >= totalSamples) {
        return;
    }

    // Binary search for the triangle index that we're in
    // If most threads in a warp are in the same triangle, there is no execution divergence,
    // and this should be pretty fast
    // cumSamplesPerTri has size numTris + 1
    // startIdx and endIdx are the ranges to search for the triangle index
    int64_t triIdx = numTris / 2;
    {
        int64_t startIdx = 0;
        int64_t endIdx = numTris - 1;
        int64_t rangeSize = endIdx - startIdx;
        int32_t count = 0;
        while (count < 8192) {
            const int64_t rangeStart = cumSamplesPerTri[triIdx];
            const int64_t rangeEnd = cumSamplesPerTri[triIdx+1];

            if (tid < rangeStart) {
                endIdx = triIdx;
            } else if (tid >= rangeEnd) {
                startIdx = triIdx;
            } else {
                break;
            }
            triIdx = startIdx + (rangeSize / 2);
            rangeSize = endIdx - startIdx;
            count += 1;
        }

        if (count == 8192) {
            printf("Binary search failed. This is a big deal\n");
            return;
        }
        if (triIdx >= numTris) {
            printf("Triangle index out of bounds (overflow). This is a big deal\n");
            return;
        }
        if (triIdx < 0) {
            printf("Triangle index out of bounds (negative). This is a big deal\n");
            return;
        }
    }

    // Compute the voxel coordinate vertices and edges of the triangle containing this sample
    const int16_t triJIdx = triangles.batchIdx(triIdx);
    const VoxelCoordTransform tx = transforms[triJIdx];
    const TorchRAcc32<ScalarI, 2> faces = triangles.data();
    const TorchRAcc32<ScalarF, 2> verts = vertices.data();

    // Voxel space vertices of the triangle containing this sample
    const int64_t off = vertices.offsetStart(triJIdx);
    const Vec3F v1 = tx.apply(verts[off+faces[triIdx][0]][0], verts[off+faces[triIdx][0]][1], verts[off+faces[triIdx][0]][2]);
    const Vec3F v2 = tx.apply(verts[off+faces[triIdx][1]][0], verts[off+faces[triIdx][1]][1], verts[off+faces[triIdx][1]][2]);
    const Vec3F v3 = tx.apply(verts[off+faces[triIdx][2]][0], verts[off+faces[triIdx][2]][1], verts[off+faces[triIdx][2]][2]);

    // Voxel space edges of the triangle containing this sample
    const Vec3F e1 = v2 - v1;
    const Vec3F e2 = v3 - v1;
    const ScalarF spacing = sqrt(3.0) / 3.0;  // This is very conservative spacing but fine for now

    // Number of points to generate per axis
    const int32_t numU = ceil((e1.length() + spacing) / spacing);
    const int32_t numV = ceil((e2.length() + spacing) / spacing);

    // Compute the position of this sample in the triangle (reflecting it if necessary)
    const int64_t base = cumSamplesPerTri[triIdx];
    const int64_t offsetInTri = tid - cumSamplesPerTri[triIdx];
    const int64_t i0 = offsetInTri / numV;
    const int64_t j0 = offsetInTri - i0 * numV;
    ScalarF u = ScalarF(i0) / (ScalarF(max(numU - 1, 1)));
    ScalarF v = ScalarF(j0) / (ScalarF(max(numV - 1, 1)));
    if (u + v >= 1.0) {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    const Vec3F sample = v1 + e1 * u + e2 * v;
    const nanovdb::Coord ijk = sample.round();

    // Round the sample down to the nearest voxel and write it out
    outIJK[tid][0] = ijk[0];
    outIJK[tid][1] = ijk[1];
    outIJK[tid][2] = ijk[2];
    outJIdx[tid] = triJIdx;
}



template <>
JaggedTensor dispatchIJKForMesh<torch::kCUDA>(const JaggedTensor& meshVertices,
                                              const JaggedTensor& meshFaces,
                                              const std::vector<VoxelCoordTransform>& transforms) {
    TORCH_CHECK(meshVertices.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(meshVertices.device().has_index(), "GridBatchImpl must have a valid index");
    TORCH_CHECK(meshFaces.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(meshFaces.device().has_index(), "GridBatchImpl must have a valid index");

    RAIIRawDeviceBuffer<VoxelCoordTransform> transformsDVec(transforms.size(), meshVertices.device());
    transformsDVec.setData((VoxelCoordTransform*) transforms.data(), true /* blocking */);
    const VoxelCoordTransform* transformDevPtr = transformsDVec.devicePtr;

    const torch::TensorOptions optsI32 = torch::TensorOptions().dtype(torch::kInt32).device(meshFaces.device());
    const torch::TensorOptions optsI16 = torch::TensorOptions().dtype(torch::kInt16).device(meshFaces.device());

    return AT_DISPATCH_INTEGRAL_TYPES(meshFaces.scalar_type(), "ijkForMesh", [&]() {
        using scalar_i = scalar_t;
        return AT_DISPATCH_FLOATING_TYPES_AND_HALF(meshVertices.scalar_type(), "countVoxelsPerTriToCheckVertices", [&]() {
            using scalar_f = scalar_t;

            // First count the total number of samples to generate in each triangle
            torch::Tensor samplesPerTri = torch::empty({meshFaces.jdata().size(0) + 1}, optsI32);
            auto samplesPerTriAcc = samplesPerTri.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>();
            auto verticesAcc = meshVertices.packed_accessor32<scalar_f, 2, torch::RestrictPtrTraits>();
            auto facesAcc = meshFaces.packed_accessor32<scalar_i, 2, torch::RestrictPtrTraits>();
            auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_i, 2> acc) {
                countVoxelsPerTriToCheck<scalar_f, scalar_i>(bidx, eidx, transformDevPtr,
                                                             verticesAcc, facesAcc,
                                                             samplesPerTriAcc);
            };
            forEachJaggedElementChannelCUDA<scalar_i, 2>(1024, 1, meshFaces, cb);

            // Compute the cumulative sum of the number of samples per triangle so each thread can figure out which triangle it's in
            torch::Tensor samplesPerTriCumSum = samplesPerTri.cumsum(0);
            auto samplesPerTriCumSumAcc = samplesPerTriCumSum.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>();
            const int64_t totalSurfaceSamples = samplesPerTriCumSum[-1].item<int64_t>();

            // Now write out the surface samples
            const int64_t threadsPerBlock = 1024;
            const int64_t numBlocks = GET_BLOCKS(totalSurfaceSamples, threadsPerBlock);
            torch::Tensor outIJK = torch::empty({totalSurfaceSamples, 3}, optsI32);
            torch::Tensor outJidx = torch::empty({totalSurfaceSamples}, optsI16);

            if (outIJK.numel() >= 1 << 31) {
                auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
                auto outJidxKAcc = outJidx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();
                generateSurfaceSamples<scalar_f, scalar_i, TorchRAcc64><<<numBlocks, threadsPerBlock>>>(transformDevPtr,
                                                                                        verticesAcc, facesAcc,
                                                                                        samplesPerTriCumSumAcc,
                                                                                        outIJKAcc, outJidxKAcc);
            } else {

                auto outIJKAcc = outIJK.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();
                auto outJidxKAcc = outJidx.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>();
                generateSurfaceSamples<scalar_f, scalar_i, TorchRAcc32><<<numBlocks, threadsPerBlock>>>(transformDevPtr,
                                                                                        verticesAcc, facesAcc,
                                                                                        samplesPerTriCumSumAcc,
                                                                                        outIJKAcc, outJidxKAcc);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return fvdb::JaggedTensor::from_data_and_jidx(
                outIJK, outJidx, meshFaces.batch_size());
        });
    });

}


} // namespace ops
} // namespace detail
} // namespace fvdb
