#include <nanovdb/NanoVDB.h>
#include <torch/torch.h>


namespace fvdb {

/// @brief Convert a torch tensor with exactly 3 elements into a nanovdb::Vec3d
/// @param inVec3Tensor A torch tensor containing exactly 3 elements of any type.
/// @return A nanovdb::Vec3d with the same values as the input tensor casted to double.
inline nanovdb::Vec3d tensorToVec3d(const torch::Tensor& inVec3Tensor) {
    const torch::Tensor vec3Tensor = inVec3Tensor.squeeze().cpu();
    TORCH_CHECK(vec3Tensor.numel() == 3, "tensor must be a vec3");
    TORCH_CHECK(vec3Tensor.size(0) == 3, "tensor must be a vec3");

    return nanovdb::Vec3d(vec3Tensor[0].item().toDouble(),
                          vec3Tensor[1].item().toDouble(),
                          vec3Tensor[2].item().toDouble());
}

/// @brief Convert a torch tensor with exactly 3 integral-type (int, long, etc...) elements into a nanovdb::Coord
/// @param inVec3Tensor A torch tensor containing exactly 3 elements of an integral type.
/// @return A nanovdb::Coord with the same values as the input tensor casted to long.
inline nanovdb::Coord tensorToCoord(const torch::Tensor& inVec3Tensor) {
    const torch::Tensor vec3Tensor = inVec3Tensor.squeeze().cpu();
    TORCH_CHECK(vec3Tensor.numel() == 3, "tensor must be a vec3");
    TORCH_CHECK(vec3Tensor.size(0) == 3, "tensor must be a vec3");
    TORCH_CHECK(at::isIntegralType(vec3Tensor.scalar_type(), false /*includeBool*/), "tensor must have an integer type");

    return nanovdb::Coord(vec3Tensor[0].item().toLong(),
                          vec3Tensor[1].item().toLong(),
                          vec3Tensor[2].item().toLong());
}

/// @brief Convert a torch tensor with exactly 4 integral-type (int, long, etc...) elements into a nanovdb::Vec4i
/// @param inVec4Tensor A torch tensor containing exactly 4 elements of an integral type.
/// @return A nanovdb::Vec4i with the same values as the input tensor casted to long.
inline nanovdb::Vec4i tensorToCoord4(const torch::Tensor& inVec3Tensor) {
    const torch::Tensor vec3Tensor = inVec3Tensor.squeeze().cpu();
    TORCH_CHECK(vec3Tensor.numel() == 4, "tensor must be a vec4");
    TORCH_CHECK(vec3Tensor.size(0) == 4, "tensor must be a vec4");
    TORCH_CHECK(at::isIntegralType(vec3Tensor.scalar_type(), false /*includeBool*/), "tensor must have an integer type");

    return nanovdb::Vec4i(vec3Tensor[0].item().toLong(),
                          vec3Tensor[1].item().toLong(),
                          vec3Tensor[2].item().toLong(),
                          vec3Tensor[3].item().toLong());
}

/// @brief Convert a nanovdb::coord into a (cpu) torch tensor with exactly 3 long elements
/// @param inCoord The nanovdb::Coord to convert
/// @return A torch tensor with exactly 3 long elements
inline torch::Tensor coordToTensor(const nanovdb::Coord& inCoord) {
    auto opts = torch::TensorOptions().dtype(torch::kLong);
    torch::Tensor ret = torch::empty(3, opts);
    auto acc = ret.accessor<int64_t, 1>();
    for (int i = 0; i < 3; i += 1) { acc[i] = inCoord[i]; }
    return ret;
}


/// @brief Convert a tensor of shape [B, 3] or [3] into a vector of nanovdb::Vec3d of length B. If the input tensor has
///        shape [3,], then it is duplicated B times.
/// @param batchSize The size of the batch
/// @param vec3ToConvert A tensor of shape [3,] or [B, 3]
/// @param allowNegative If true, then negative values are allowed in the tensor
/// @return A vector of nanovdb::Vec3d of length B
inline std::vector<nanovdb::Vec3d> tensorToVec3dBatch(int64_t batchSize, const torch::Tensor& vec3ToConvert, bool allowNegative = true, std::string name = "tensor") {
    torch::Tensor vec3In = vec3ToConvert.squeeze();
    std::vector<nanovdb::Vec3d> returnVec;
    returnVec.reserve(batchSize);
    if (vec3In.dim() == 1) {
        TORCH_CHECK_VALUE(vec3In.size(0) == 3, "Expected ", name, " to have shape [3,] or [B, 3] but got shape = [" +
                    std::to_string(vec3In.size(0)) + ",]");
        const nanovdb::Vec3d voxS = fvdb::tensorToVec3d(vec3In);
        if (!allowNegative) {
            TORCH_CHECK_VALUE(voxS[0] > 0, "voxelSize[0] must be > 0");
            TORCH_CHECK_VALUE(voxS[1] > 0, "voxelSize[1] must be > 0");
            TORCH_CHECK_VALUE(voxS[2] > 0, "voxelSize[2] must be > 0");
        }
        for (int64_t i = 0; i < batchSize; ++i) {
            returnVec.push_back(voxS);
        }
    } else if (vec3In.dim() == 2) {
        TORCH_CHECK(vec3In.size(0) == batchSize, "Expected ", name, " to have shape [3,] or [B, 3] but got shape = [" +
                    std::to_string(vec3In.size(0)) + ", " + std::to_string(vec3In.size(1)) + "]");
        TORCH_CHECK(vec3In.size(0) == batchSize, "Expected ", name, " to have shape [3,] or [B, 3] but got shape = [" +
                    std::to_string(vec3In.size(0)) + ", " + std::to_string(vec3In.size(1)) + "]");
        for (int64_t i = 0; i < batchSize; ++i) {
            const nanovdb::Vec3d voxS = fvdb::tensorToVec3d(vec3In[i]);
            if (!allowNegative) {
                TORCH_CHECK_VALUE(voxS[0] > 0, "voxelSize[0] must be > 0");
                TORCH_CHECK_VALUE(voxS[1] > 0, "voxelSize[1] must be > 0");
                TORCH_CHECK_VALUE(voxS[2] > 0, "voxelSize[2] must be > 0");
            }
            returnVec.push_back(voxS);
        }
    }
    return returnVec;
}

} // namespace fvdb