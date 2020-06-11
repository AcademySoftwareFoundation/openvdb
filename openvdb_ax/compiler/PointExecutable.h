// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/PointExecutable.h
///
/// @authors Nick Avramoussis, Francisco Gochez, Richard Jones
///
/// @brief The PointExecutable, produced by the OpenVDB AX Compiler for
///   execution over OpenVDB Points Grids.
///

#ifndef OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED

#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/compiler/AttributeRegistry.h>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>

#include <unordered_map>

//forward
namespace llvm {

class ExecutionEngine;
class LLVMContext;

}

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {


/// @brief Object that encapsulates compiled AX code which can be executed on a target point grid
class PointExecutable
{
public:
    using Ptr = std::shared_ptr<PointExecutable>;

    /// @brief Constructor
    /// @param exeEngine Shared pointer to an llvm::ExecutionEngine object used to build functions.
    ///        context should be the associated llvm context
    /// @param context Shared pointer to an llvm:context object associated with exeEngine
    /// @param attributeRegistry Registry of point attributes accessed by AX code
    /// @param customData Custom data object which will be shared by this executable.  It can be
    ///        used to retrieve external data from within the AX code
    /// @param functions A map of function names to physical memory addresses which were built
    ///        by llvm using exeEngine
    /// @note  This object is normally be constructed by the Compiler::compile method, rather
    ///        than directly
    PointExecutable(const std::shared_ptr<const llvm::ExecutionEngine>& exeEngine,
                    const std::shared_ptr<const llvm::LLVMContext>& context,
                    const AttributeRegistry::ConstPtr& attributeRegistry,
                    const CustomData::ConstPtr& customData,
                    const std::unordered_map<std::string, uint64_t>& functions)
        : mContext(context)
        , mExecutionEngine(exeEngine)
        , mAttributeRegistry(attributeRegistry)
        , mCustomData(customData)
        , mFunctionAddresses(functions) {}

    ~PointExecutable() = default;

    /// @brief executes compiled AX code on target grid
    /// @param grid Grid to apply code to
    /// @param group Optional name of a group for filtering.  If this is not NULL,
    ///        the code will only be applied to points in this group
    /// @param createMissing If true, any attribute which has been used but does
    ///        not exist will be created. Otherwise, a missing attribute runtime
    ///        error will be thrown.
    void execute(points::PointDataGrid& grid,
                 const std::string* const group = nullptr,
                 const bool createMissing = true) const;

private:

    // The Context and ExecutionEngine must exist _only_ for object lifetime
    // management. The ExecutionEngine must be destroyed before the Context
    const std::shared_ptr<const llvm::LLVMContext> mContext;
    const std::shared_ptr<const llvm::ExecutionEngine> mExecutionEngine;
    const AttributeRegistry::ConstPtr mAttributeRegistry;
    const CustomData::ConstPtr mCustomData;
    // addresses of actual compiled code
    const std::unordered_map<std::string, uint64_t> mFunctionAddresses;
};

}
}
}

#endif // OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED

