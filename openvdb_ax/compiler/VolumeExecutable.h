// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/VolumeExecutable.h
///
/// @authors Nick Avramoussis, Francisco Gochez, Richard Jones
///
/// @brief The VolumeExecutable, produced by the OpenVDB AX Compiler for
///   execution over Numerical OpenVDB Grids.
///

#ifndef OPENVDB_AX_COMPILER_VOLUME_EXECUTABLE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_VOLUME_EXECUTABLE_HAS_BEEN_INCLUDED

#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/compiler/AttributeRegistry.h>

#include <openvdb/Grid.h>

#include <unordered_map>

// Forward declaration of LLVM types which persist on the Executables

namespace llvm {

class ExecutionEngine;
class LLVMContext;

}

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

/// @brief Object that encapsulates compiled AX code which can be executed on a collection of
///        VDB volume grids
class VolumeExecutable
{
public:
    using Ptr = std::shared_ptr<VolumeExecutable>;

    enum class IterType {
        ON,
        OFF,
        ALL
    };

    /// @brief Constructor
    /// @param exeEngine Shared pointer to an llvm::ExecutionEngine object used to build functions.
    ///        context should be the associated llvm context
    /// @param context Shared pointer to an llvm:context object associated with exeEngine
    /// @param accessRegistry Registry of volumes accessed by AX code
    /// @param customData Custom data object which will be shared by this executable.  It can be
    ///        used to retrieve external data from within the AX code
    /// @param functionAddresses A Vector of maps of function names to physical memory addresses which were built
    ///        by llvm using exeEngine
    /// @note  This object is normally be constructed by the Compiler::compile method, rather
    ///        than directly
    VolumeExecutable(const std::shared_ptr<const llvm::ExecutionEngine>& exeEngine,
                     const std::shared_ptr<const llvm::LLVMContext>& context,
                     const AttributeRegistry::ConstPtr& accessRegistry,
                     const CustomData::ConstPtr& customData,
                     const std::unordered_map<std::string, uint64_t>& functionAddresses)
        : mContext(context)
        , mExecutionEngine(exeEngine)
        , mAttributeRegistry(accessRegistry)
        , mCustomData(customData)
        , mFunctionAddresses(functionAddresses) {}

    ~VolumeExecutable() = default;

    /// @brief Execute AX code on target grids
    void execute(openvdb::GridPtrVec& grids,
        const IterType iterType = IterType::ON,
        const bool createMissing = false) const;

private:

    // The Context and ExecutionEngine must exist _only_ for object lifetime
    // management. The ExecutionEngine must be destroyed before the Context
    const std::shared_ptr<const llvm::LLVMContext> mContext;
    const std::shared_ptr<const llvm::ExecutionEngine> mExecutionEngine;
    const AttributeRegistry::ConstPtr mAttributeRegistry;
    const CustomData::ConstPtr mCustomData;
    const std::unordered_map<std::string, uint64_t> mFunctionAddresses;
};

}
}
}

#endif // OPENVDB_AX_COMPILER_VOLUME_EXECUTABLE_HAS_BEEN_INCLUDED

