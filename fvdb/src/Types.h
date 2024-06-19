#pragma once

#include <torch/all.h>
#include <c10/cuda/CUDAFunctions.h>
#include <nanovdb/NanoVDB.h>

#include "detail/TypesImpl.h"


namespace fvdb {


// These are union types that can be constructed from nanovdb types, torch tensors, std::vectors, single scalars, etc...
// They are used to allow the user to pass in a variety of types to the API, and then convert them to the correct type
using Vec3i = detail::Coord3Impl<false>;
using Vec3iOrScalar = detail::Coord3Impl<true>;
using Vec4i = detail::Coord4Impl<false>;
using Vec3d = detail::Vec3dImpl<false>;
using Vec3dOrScalar = detail::Vec3dImpl<true>;

// These are union types that can be constructed from nanovdb types, torch tensors, std::vectors, single scalars, etc...
// and resolve to a batch of values. They are used to allow the user to pass in a single vector (or scalar) and have
// it be broadcast to a whole batch of values.
// E.g. if you are constructing a batch of grids, you can pass in a single scalar 1.0 to have a voxel size of [1, 1, 1]
//      for every grid in the batch. Or a user can pass in a vector [1, 2, 3] to have each grid have a voxel
//       size of [1, 2, 3]. Alternatively, a user can specify a voxel size for each grid in the batch
//       [[v1x, v1y, v1z], ..., [vnx, vny, vnz]]. The Vec3dBatchOrScalar will accept all these inputs
//       and resolve them to a batch of values.
using Vec3dBatchOrScalar = detail::Vec3BatchImpl<nanovdb::Vec3d, true /*AllowScalar*/, true /*AllowBroadcast*/>;
using Vec3dBatch = detail::Vec3BatchImpl<nanovdb::Vec3d, false /*AllowScalar*/, true /*AllowBroadcast*/>;
using Vec3iBatch = detail::Vec3BatchImpl<nanovdb::Coord, false /*AllowScalar*/, true /*AllowBroadcast*/>;


/// @brief A class that can be constructed from a torch::Device or a string.
///        Calling value() returns a torch::device
class TorchDeviceOrString {
    torch::Device mValue;
    void setIndex() {
        if (mValue.is_cuda() && ! mValue.has_index()) {
            mValue.set_index(c10::cuda::current_device());
        }
    }
public:
    TorchDeviceOrString() : mValue(torch::kCPU) { setIndex(); }
    TorchDeviceOrString(torch::Device device) : mValue(device) { setIndex(); }
    TorchDeviceOrString(c10::DeviceType deviceType) : mValue(deviceType) { setIndex(); }
    TorchDeviceOrString(std::string& str) : mValue(str) { setIndex(); }

    const torch::Device& value() const {
        return mValue;
    }
};


/// @brief A class that con be constructed from a string or a list of strings but always returns a list of strings
///        Used to enable broadcasting for arguments that specify a single value or a list of values for a whole batch
class StringOrListOfStrings {
    std::vector<std::string> mValue;
public:
    StringOrListOfStrings() : mValue() {}
    StringOrListOfStrings(std::string str) : mValue({str}) {}
    StringOrListOfStrings(std::vector<std::string> str) : mValue(str) {}

    const std::vector<std::string>& value() const {
        return mValue;
    }
};


/// @brief A class representing a set of unique IDs for a nanovdb grid (used to specify which grids to load
///        from an .nvdb file). You can specify the set of grids to load as a integer index, a single string name,
///        a vector of integer indices, or a vector of string names
class NanoVDBFileGridIdentifier {
    std::vector<uint64_t> mIndices;
    std::vector<std::string> mGridNames;

public:
    NanoVDBFileGridIdentifier() : mIndices(), mGridNames() {};
    NanoVDBFileGridIdentifier(uint64_t index) : mIndices({index}) {};
    NanoVDBFileGridIdentifier(std::vector<uint64_t> indices) : mIndices(indices) {};
    NanoVDBFileGridIdentifier(std::string gridName) : mGridNames({gridName}) {};
    NanoVDBFileGridIdentifier(std::vector<std::string> gridNames) : mGridNames(gridNames) {};

    std::string toString() const {
        std::stringstream ss;
        if (specifiesIndices()) {
            for(auto idx : mIndices) {
                ss << idx << ", ";
            }
            return "NanoVDBFileGridIdentifier indices: " + ss.str();
        } else {
            for(auto idx : mGridNames) {
                ss << idx << ", ";
            }
            return "NanoVDBFileGridIdentifier gridNames: " + ss.str();
        }
    }

    bool isValid() const {
        return (mIndices.empty() != mGridNames.empty());
    }

    bool specifiesIndices() const {
        return !mIndices.empty();
    }

    bool specifiesNames() const {
        return !mGridNames.empty();
    }

    const std::vector<uint64_t>& indicesValue() const {
        return mIndices;
    }

    const std::vector<std::string>& namesValue() const {
        return mGridNames;
    }

    bool empty() const {
        return (mIndices.empty() && mGridNames.empty());
    }

    size_t size() const {
        if (specifiesIndices()) {
            return mIndices.size();
        } else {
            return mGridNames.size();
        }
    }

};


} // namespace fvdb