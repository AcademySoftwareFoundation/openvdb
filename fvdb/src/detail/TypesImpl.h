#include <nanovdb/NanoVDB.h>


namespace fvdb {
namespace detail {

template <bool AllowScalar>
class Vec3dImpl {
    nanovdb::Vec3d mValue;
    bool mWasScalar = false;

public:
    static constexpr bool SupportsScalarCast = AllowScalar;
    using ValueType = nanovdb::Vec3d::ValueType;

    Vec3dImpl() : mValue(0.0, 0.0, 0.0) {}
    Vec3dImpl(const nanovdb::Vec3d& coord) : mValue(coord) {}
    Vec3dImpl(const nanovdb::Vec3f& coord) : mValue(coord[0], coord[1], coord[2]) {}
    Vec3dImpl(const torch::Tensor& coordTensor) : mValue(fvdb::tensorToVec3d(coordTensor)) {}
    template <typename T>
    Vec3dImpl(const std::vector<T>& coordVec) {
        static_assert(std::is_arithmetic<T>::value, "Coord3D can only be constructed from integral types");
        TORCH_CHECK_VALUE(coordVec.size() == 3, "Coord3D can only be constructed from a vector of size 3");
        mValue = nanovdb::Vec3d(coordVec[0], coordVec[1], coordVec[2]);
    }

    template <typename T>
    Vec3dImpl(T scalar) {
        static_assert(AllowScalar, "Vec3d can only be constructed from a scalar if AllowScalar is true");
        static_assert(std::is_arithmetic<T>::value, "Vec3d can only be constructed from numeric types");
        mValue = nanovdb::Vec3d(scalar, scalar, scalar);
        mWasScalar = true;
    }

    const nanovdb::Vec3d& value() const {
        if constexpr (!AllowScalar) {
            TORCH_CHECK_VALUE(!mWasScalar, "Expected a vector, but got a scalar");
        }
        return mValue;
    }
};


template <bool AllowScalar>
class Coord3Impl {
    nanovdb::Coord mValue;
    bool mWasScalar = false;

public:
    static constexpr bool SupportsScalarCast = AllowScalar;
    using ValueType = nanovdb::Coord::ValueType;

    Coord3Impl() : mValue(0, 0, 0) {}
    Coord3Impl(const nanovdb::Coord& coord) : mValue(coord) {}
    Coord3Impl(const nanovdb::Vec3i& coord) : mValue(coord[0], coord[1], coord[2]) {}
    Coord3Impl(const nanovdb::Vec3u& coord) : mValue(coord[0], coord[1], coord[2]) {}
    Coord3Impl(const torch::Tensor& coordTensor) : mValue(fvdb::tensorToCoord(coordTensor)) {}
    template <typename T>
    Coord3Impl(const std::vector<T>& coordVec) {
        static_assert(std::is_integral<T>::value, "Coord can only be constructed from integral types");
        TORCH_CHECK_VALUE(coordVec.size() == 3, "Coord can only be constructed from a vector of size 3");
        mValue = nanovdb::Coord(coordVec[0], coordVec[1], coordVec[2]);
    }
    template <typename T>
    Coord3Impl(T scalar) {
        static_assert(AllowScalar, "Coord3 can only be constructed from a scalar if AllowScalar is true");
        static_assert(std::is_integral<T>::value, "Coord3D can only be constructed from integral types");
        mValue = nanovdb::Coord(scalar, scalar, scalar);
        mWasScalar = true;
    }

    const nanovdb::Coord& value() const {
        if constexpr (!AllowScalar) {
            TORCH_CHECK_VALUE(!mWasScalar, "Expected a vector, but got a scalar");
        }
        return mValue;
    }

    std::string toString() const {
        return "{" + std::to_string(mValue[0]) + ", " + std::to_string(mValue[1]) + ", " + std::to_string(mValue[2]) + "}";
    }
};


template <bool AllowScalar>
class Coord4Impl {
    nanovdb::Vec4i mValue;
    static_assert(!AllowScalar, "Coord does not allow scalar conversion. We may wish to change this in the future.");

public:
    static constexpr bool SupportsScalarCast = AllowScalar;
    using ValueType = nanovdb::Coord::ValueType;

    Coord4Impl() : mValue(0, 0, 0, 0) {}
    Coord4Impl(const nanovdb::Vec4i& coord) : mValue(coord) {}
    Coord4Impl(const torch::Tensor& coordTensor) : mValue(fvdb::tensorToCoord4(coordTensor)) {}
    template <typename T>
    Coord4Impl(const std::vector<T>& coordVec) {
        static_assert(std::is_integral<T>::value, "Vec4i can only be constructed from integral types");
        TORCH_CHECK_VALUE(coordVec.size() == 4, "Vec4i can only be constructed from a vector of size 4");
        mValue = nanovdb::Vec4i(coordVec[0], coordVec[1], coordVec[2], coordVec[3]);
    }

    const nanovdb::Vec4i& value() const {
        return mValue;
    }
};


template <typename VecT, bool AllowScalar, bool AllowBroadcast>
class Vec3BatchImpl {
private:
    std::vector<VecT> mValue;
    bool isScalar = false;
    bool isSingle = false;

    std::vector<VecT> repeatIt(int64_t batchSize, bool onlyPositive) const {
        if (onlyPositive) {
            TORCH_CHECK_VALUE(mValue[0][0] > 0 && mValue[0][1] > 0 && mValue[0][2] > 0, "Expected all coordinates to be positive");
        }
        std::vector<VecT> result;
        result.reserve(batchSize);
        for (int64_t i = 0; i < batchSize; ++i) {
            result.push_back(mValue[0]);
        }
        return result;
    }

public:
    static constexpr bool SupportsBroadcast = AllowBroadcast;
    static constexpr bool SupportsScalarCast = AllowScalar;

    using ValueType = typename VecT::ValueType;
    using VecType = VecT;

    Vec3BatchImpl() : mValue() {}

    Vec3BatchImpl(const torch::Tensor& tensor) {
        torch::Tensor squeezed = tensor.squeeze().cpu();

        if constexpr (AllowScalar) {
            if (squeezed.numel() == 1) {
                mValue.push_back(VecT(squeezed.item<double>()));
                isScalar = true;
                return;
            }
        }

        if constexpr (AllowBroadcast) {
            if (squeezed.numel() == 3) {
                mValue.push_back(VecT(squeezed[0].item<double>(), squeezed[1].item<double>(), squeezed[2].item<double>()));
                isSingle = true;
                return;
            }
        }

        TORCH_CHECK_VALUE(squeezed.dim() == 2, "Expected a batch of 3D coordinates with size [B, 3]");
        TORCH_CHECK_VALUE(squeezed.size(1) == 3, "Expected a batch of 3D coordinates with size [B, 3]");
        mValue.reserve(squeezed.size(0));
        for (int i = 0; i < squeezed.size(0); ++i) {
            mValue.push_back(VecT(squeezed[i][0].item<double>(), squeezed[i][1].item<double>(), squeezed[i][2].item<double>()));
        }
    }

    template <typename T>
    Vec3BatchImpl(const std::vector<std::vector<T>>& vectorData) {
        if constexpr (nanovdb::util::is_same<VecT, nanovdb::Coord>::value) {
            static_assert(std::is_integral<T>::value, "Vec3Batch can only be constructed from integral types");
        }
        static_assert(std::is_arithmetic<T>::value, "Vec3Batch can only be constructed from numeric types");
        size_t batchSize = vectorData.size();
        TORCH_CHECK_VALUE(batchSize > 0, "Expected a batch of coordinates with size [B, 3]");
        for (size_t i = 0; i < batchSize; i += 1) {
            TORCH_CHECK_VALUE(vectorData[i].size() == 3, "Expected a batch of 3D coordinates with size [B, 3]");
            mValue.push_back(VecT(vectorData[i][0], vectorData[i][1], vectorData[i][2]));
        }
    }

    template <typename T>
    Vec3BatchImpl(const T& scalar) {
        static_assert(AllowScalar, "Cannot construct Vec3Batch from scalar when AllowScalar is set to false");

        if constexpr (nanovdb::util::is_same<VecT, nanovdb::Coord>::value) {
            static_assert(std::is_integral<T>::value, "Vec3Batch can only be constructed from integral types");
        }
        static_assert(std::is_arithmetic<T>::value, "Vec3Batch can only be constructed from numeric types");
        mValue.push_back(VecT((double) scalar));
        isScalar = true;
    }

    template <typename T>
    Vec3BatchImpl(const std::vector<T>& vec) {
        static_assert(AllowBroadcast, "Cannot construct Vec3Batch from single vector when AllowBroadcast is set to false");

        if constexpr (nanovdb::util::is_same<VecT, nanovdb::Coord>::value) {
            static_assert(std::is_integral<T>::value, "Vec3Batch can only be constructed from integral types");
        }
        static_assert(std::is_arithmetic<T>::value, "Vec3Batch can only be constructed from numeric types");
        TORCH_CHECK_VALUE(vec.size() == 3, "Expected a batch of 3D coordinates with size [B, 3] or a single coordinate of size [3,]");
        mValue.push_back(VecT(vec[0], vec[1], vec[2]));
        isSingle = true;
    }

    std::vector<VecT> value(uint64_t batchSize, bool onlyPositive, std::string name) const {
        TORCH_CHECK(batchSize > 0, "Can't request empty batch of coordinates");
        TORCH_CHECK(mValue.size() > 0, "Can't request empty batch of coordinates");

        if constexpr (AllowScalar) {
            if (isScalar) {
                return repeatIt(batchSize, onlyPositive);
            }

        }
        if constexpr (AllowBroadcast) {
            if (isSingle && batchSize != 1) {
                return repeatIt(batchSize, onlyPositive);
            }
        }

        if (onlyPositive) {
            for (size_t i = 0; i < mValue.size(); ++i) {
                TORCH_CHECK_VALUE(mValue[i][0] > 0 && mValue[i][1] > 0 && mValue[i][2] > 0, "Expected all coordinates of " + name + " to be positive");
            }
        }
        TORCH_CHECK_VALUE(batchSize == mValue.size(), "Expected " + name + " batch of 3D coordinates to have size [" + std::to_string(batchSize) + ", 3]");
        return mValue;
    }

    torch::Tensor tensorValue(uint64_t batchSize, bool onlyPositive, std::string name) const {
        std::vector<VecT> vec = value(batchSize, onlyPositive, name);

        if constexpr (nanovdb::util::is_same<VecT, nanovdb::Coord>::value) {
            return torch::from_blob(vec.data(), { (int64_t) vec.size(), 3 }, torch::kInt32).clone();
        } else if constexpr (nanovdb::util::is_same<VecT, nanovdb::Vec3d>::value) {
            return torch::from_blob(vec.data(), { (int64_t) vec.size(), 3 }, torch::kDouble).clone();
        } else {
            static_assert("Only Coord and Vec3d are supported for now");
        }
    }
};

} // namespace detail
} // namespace fvdb
