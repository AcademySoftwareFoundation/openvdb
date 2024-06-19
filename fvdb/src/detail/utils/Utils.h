#pragma once

#include <nanovdb/NanoVDB.h>

#include <type_traits>
#include <iostream> // for std::ostream

#include <c10/util/Half.h>
#include <torch/extension.h>

#include "nanovdb/ActiveVoxelIterator.h"
#include "nanovdb/HDDAIterators.h"
#include "nanovdb/CustomAccessors.h"
#include "nanovdb/Printing.h"
#include "nanovdb/TorchNanoConversions.h"


// A bunch of things defined to make intellisense work with nvcc
#if defined(NDEVELOP_IDE_ONLY)
namespace torch {
    template <typename T>
    struct RestrictPtrTraits {
    typedef T* __restrict__ PtrType;
    };
}
#endif


/// @brief Given a fvdb::GridBatchImpl, dispatch to the correct grid type.
///        This macro calls the passed in function with the typedef GridType to the correct grid type.
///        (i.e. ValueOnIndex or ValueOnIndexMask)
#define FVDB_DISPATCH_GRID_TYPES(GRID_HDL, ...)           \
    [&]() {                                               \
        if ((GRID_HDL).isMutable()) {                     \
            using GridType = nanovdb::ValueOnIndexMask;   \
            return __VA_ARGS__();                         \
        } else {                                          \
            using GridType = nanovdb::ValueOnIndex;       \
            return __VA_ARGS__();                         \
        }                                                 \
    }()

/// @brief Given a torch::Device, define DeviceTag to torch::kCPU or torch::kCUDA.
///        This macro calls the passed in function with the typedef DeviceTag to the correct device tag.
#define FVDB_DISPATCH_KERNEL_DEVICE(DEVICE, ...)                           \
    [&]() {                                                                \
        if (DEVICE.is_cpu()) {                                             \
            static constexpr c10::DeviceType DeviceTag = torch::kCPU;             \
            return __VA_ARGS__();                                          \
        } else if (DEVICE.is_cuda()) {                                     \
            static constexpr c10::DeviceType DeviceTag = torch::kCUDA;            \
            return __VA_ARGS__();                                          \
        } else {                                                           \
            TORCH_CHECK(false, "Only CUDA and CPU devices are supported"); \
        }                                                                  \
    }()

/// @brief Given a boolean for whether a grid is mutable, dispatch to the correct grid type.
///        This macro calls the passed in function with the typedef GridType to the correct grid type.
///        (i.e. ValueOnIndex or ValueOnIndexMask)
#define FVDB_DISPATCH_GRID_TYPES_MUTABLE(IS_MUTABLE, ...) \
    [&]() {                                               \
        if (IS_MUTABLE) {                                 \
            using GridType = nanovdb::ValueOnIndexMask;   \
            return __VA_ARGS__();                         \
        } else {                                          \
            using GridType = nanovdb::ValueOnIndex;       \
            return __VA_ARGS__();                         \
        }                                                 \
    }()



namespace fvdb {
namespace detail {

/// @brief A helper struct to determine if a type is a floating-point type or a half-precision floating-point type.
/// @tparam T The type to check.
template< class T >
struct is_floating_point_or_half
    : std::integral_constant<
        bool,
        // Note: standard floating-point types
        std::is_same<float, typename std::remove_cv<T>::type>::value
        || std::is_same<double, typename std::remove_cv<T>::type>::value
        || std::is_same<long double, typename std::remove_cv<T>::type>::value
        // Note: extended floating-point types (C++23, if supported)
        || std::is_same<c10::Half, typename std::remove_cv<T>::type>::value
    > {};


/// @brief Convert a 1d tensor of integer values into an std:vector<int64_t>
/// @param shapeTensor a 1D tensor of integer values
/// @return An std::vector<int64_t> with the same values as the input tensor
inline std::vector<int64_t> intTensor1DToStdVector(torch::Tensor shapeTensor) {
    return AT_DISPATCH_INTEGRAL_TYPES(shapeTensor.scalar_type(), "tensorToShape", [&]() {
        TORCH_CHECK(shapeTensor.dim() == 1, "shapeTensor must be a 1D tensor");
        TORCH_CHECK(!shapeTensor.is_floating_point(), "shapeTensor must be an integer tensor");
        auto acc = shapeTensor.accessor<scalar_t, 1>();
        std::vector<int64_t> outShape(acc.size(0));
        for (int64_t i = 0; i < acc.size(0); i += 1) {
            outShape[i] = (int64_t) acc[i];
        }
        return outShape;
    });
}


/// @brief Return an std::vector<int64_t> representing the shape of a tensor which is forned by removing the first
///        N dimensions of the input tensor and replacing them with s0
///        For example lets' say inTensor has shape [X, Y, Z], then spliceShape({A, B}, inTensor, 2) will return [A, B, Z]
///        and spliceShape({A, B}, inTensor, 1) will return [A, B, Y, Z]
/// @param s0 The shape values to splice in
/// @param inTensor The tensor whose shape to splice
/// @param start How many dimensions to remove from the shape of inTensor
/// @return An std::vector<int64_t> representing the shape of the spliced tensor
inline std::vector<int64_t> spliceShape(std::vector<int64_t> s0, const torch::Tensor& inTensor, int start = 1) {
    TORCH_CHECK(start >= 0 && start <= inTensor.dim(), "start must be in range [0, inTensor.dim()]");
    std::vector<int64_t> outSize(s0.size() + inTensor.dim() - start);
    for (size_t i = 0; i < s0.size(); i+= 1) {
        outSize[i] = s0[i];
    }

    for (int64_t i = start; i < inTensor.dim(); i += 1) {
        outSize[i + s0.size() - start] = inTensor.size(i);
    }
    return outSize;
}


/// @brief Return a view of the input tensor with all but first ndim dimensions coalesced into a single dimension
///        this is similar to inTensor.view({inTensor.size(0), ..., inTensor.size(ndim - 1), -1})
///        but it handles the case when inTensor.size(0)*...*inTensor.size(ndim - 1) == 0
/// @param inTensor The tensor to coalesce
/// @return A view of the input tensor with all but first dimensions coalesced into a single dimension
inline torch::Tensor featureCoalescedView(const torch::Tensor& inTensor, int64_t ndim = 1) {
    std::vector<int64_t> outSize;
    for (int64_t i = 0; i < ndim; ++i) {
        outSize.push_back(inTensor.size(i));
    }
    int64_t numOthers = 1;
    for (int64_t i = ndim; i < inTensor.dim(); ++i) {
        numOthers *= inTensor.size(i);
    }
    outSize.push_back(numOthers);
    torch::Tensor outTensor = inTensor.view(outSize);
    TORCH_CHECK(inTensor.storage().is_alias_of(outTensor.storage()), "output should be a view!");
    return outTensor;
}


/// @brief Convert a tensor of shape [B, 3] or [3] representing a batch of coordinates or a single coordinate into a
///        tensor of shape [B, 3] (if the input has shape [B, 3], this is a no-op)
/// @param coordOrBatch A tensor of shape [B, 3] or [3]
/// @param batchSize The size of the batch
/// @return A tensor of shape [B, 3]
inline torch::Tensor coordTensorToBatch(const torch::Tensor& coordOrBatch, int64_t batchSize) {
    if (coordOrBatch.dim() == 1) {
        TORCH_CHECK_VALUE(coordOrBatch.size(0) == 3, "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                          std::to_string(coordOrBatch.size(0)) + ",]");
        return coordOrBatch.unsqueeze(0).repeat({batchSize, 1});
    } else {
        TORCH_CHECK_VALUE(coordOrBatch.dim() == 2, "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                          std::to_string(coordOrBatch.size(0)) + ", " + std::to_string(coordOrBatch.size(1)) + "]");
        TORCH_CHECK_VALUE(coordOrBatch.size(0) == batchSize, "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                          std::to_string(coordOrBatch.size(0)) + ", " + std::to_string(coordOrBatch.size(1)) + "]");
        TORCH_CHECK_VALUE(coordOrBatch.size(1) == 3, "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                          std::to_string(coordOrBatch.size(0)) + ", " + std::to_string(coordOrBatch.size(1)) + "]");
        return coordOrBatch;
    }
}

/// @brief Convert a torch::ScalarType to its corresponding string (e.g. torch::kInt32 -> "int32")
/// @param stype The scalar type to convert to a string
/// @return The string representation of the scalar type
inline const std::string TorchScalarTypeToStr(torch::ScalarType stype) {
    switch (stype) {
        case torch::kInt8:
            return "int8";
        case torch::kUInt8:
            return "uint8";
        case torch::kInt16:
            return "int16";
        case torch::kInt32:
            return "int32";
        case torch::kInt64:
            return "int64";

        case torch::kFloat16:
            return "float16";
        case torch::kFloat32:
            return "float32";
        case torch::kFloat64:
            return "float64";

        case torch::kComplexHalf:
            return "complex32";
        case torch::kComplexFloat:
            return "complex64";
        case torch::kComplexDouble:
            return "complex128";

        case torch::kBool:
            return "bool";

        case torch::kQInt8:
            return "qint8";
        case torch::kQUInt8:
            return "quint8";
        case torch::kQInt32:
            return "qint32";

        case torch::kBFloat16:
            return "bfloat16";

        case torch::kQUInt2x4:
            return "quint2x4";
        case torch::kQUInt4x2:
            return "quint4x2";

        default:
            TORCH_CHECK_VALUE(false, "Unsupported scalar type");

        //Byte, Char, Short, Int, Long, Half, Float, Double, ComplexHalf, ComplexFloat, ComplexDouble, Bool, QInt8, QUInt8, QInt32, BFloat16, QUInt4x2, QUInt2x4
    }
}


/// @brief Convert a string representation of a scalar type into a torch::ScalarType (or throw an exception if the type is not valid)
/// @param dtypeStr The string representation of the scalar type
/// @return The torch::ScalarType corresponding to the string
inline torch::ScalarType StringToTorchScalarType(std::string dtypeStr) {
    if (dtypeStr == "int8") {
        return torch::kInt8;
    } else if (dtypeStr == "uint8") {
        return torch::kUInt8;
    } else if (dtypeStr == "int16") {
        return torch::kInt16;
    } else if (dtypeStr == "int32") {
        return torch::kInt32;
    } else if (dtypeStr == "int64") {
        return torch::kInt64;
    }

    else if (dtypeStr == "float16") {
        return torch::kFloat16;
    } else if (dtypeStr == "float32") {
        return torch::kFloat32;
    } else if (dtypeStr == "float64") {
        return torch::kFloat64;
    }

    else if (dtypeStr == "complex32") {
        return torch::kComplexHalf;
    } else if (dtypeStr == "complex64") {
        return torch::kComplexFloat;
    } else if (dtypeStr == "complex128") {
        return torch::kComplexDouble;
    }

    else if(dtypeStr == "bool") {
        return torch::kBool;
    }

    else if (dtypeStr == "qint8") {
        return torch::kQInt8;
    } else if (dtypeStr == "quint8") {
        return torch::kQUInt8;
    } else if (dtypeStr == "qint32") {
        return torch::kQInt32;
    } else if (dtypeStr == "bfloat16") {
        return torch::kBFloat16;
    } else if (dtypeStr == "quint2x4") {
        return torch::kQUInt2x4;
    } else if (dtypeStr == "quint4x2") {
        return torch::kQUInt4x2;
    }

    TORCH_CHECK(false, "Invalid dtype string " + dtypeStr);

}


} // namespace detail
} // namespace fvdb


// std::cout and std::cerr for shapes
inline std::ostream& operator<<(std::ostream& os, at::IntArrayRef c) {
    os << "[";
    for (size_t i = 0; i < c.size(); i += 1) {
        os << c[i];
        if (i < c.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}