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

class TestPointExecutable;

namespace llvm {
class ExecutionEngine;
class LLVMContext;
}

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {

class Compiler;

/// @brief Object that encapsulates compiled AX code which can be executed on a target point grid
class PointExecutable
{
public:
    using Ptr = std::shared_ptr<PointExecutable>;
    ~PointExecutable();

    /// @brief  Copy constructor. Shares the LLVM constructs but deep copies the
    ///   settings. Multiple copies of an executor can be used at the same time
    ///   safely.
    PointExecutable(const PointExecutable& other);

    ////////////////////////////////////////////////////////

    /// @brief executes compiled AX code on target grid
    void execute(points::PointDataGrid& grid) const;

    ////////////////////////////////////////////////////////

    /// @brief  Set a specific point group to execute over. The default is none,
    ///   which corresponds to all points. Note that this can also be compiled
    ///   into the AX function using the ingroup("mygroup") method.
    /// @warning  If the group does not exist during execute, a runtime error
    ///   will be thrown.
    /// @param name  The name of the group to execute over
    void setGroupExecution(const std::string& name);
    /// @return  The points group to be processed. Default is empty, which is
    ///   all points.
    const std::string& getGroupExecution() const;

    /// @brief  Set the behaviour when missing point attributes are accessed.
    ///    Default behaviour is true, which creates them with default initial
    ///    values. If false, a missing attribute runtime error will be thrown
    ///    on missing accesses.
    /// @param flag  Enables or disables the creation of missing attributes
    void setCreateMissing(const bool flag);
    /// @return  Whether this executable will generate new point attributes.
    bool getCreateMissing() const;

    /// @brief  Set the threading grain size. Default is 1. A value of 0 has the
    ///   effect of disabling multi-threading.
    /// @param grain The grain size
    void setGrainSize(const size_t grain);
    /// @return  The current grain size
    size_t getGrainSize() const;

    ////////////////////////////////////////////////////////

    // @brief deprecated methods
    OPENVDB_DEPRECATED void
    execute(points::PointDataGrid& grid,
        const std::string* const group,
        const bool create) const
    {
        PointExecutable copy(*this);
        if (group) copy.setGroupExecution(*group);
        copy.setCreateMissing(create);
        copy.execute(grid);
    }

    OPENVDB_DEPRECATED void
    execute(points::PointDataGrid& grid,
        const std::string* const group) const
    {
        PointExecutable copy(*this);
        if (group) copy.setGroupExecution(*group);
        copy.execute(grid);
    }

    ////////////////////////////////////////////////////////

    // foward declaration of settings for this executable
    struct Settings;

private:
    friend class Compiler;
    friend class ::TestPointExecutable;

    /// @brief Constructor, expected to be invoked by the compiler. Should not
    ///   be invoked directly.
    /// @param context Shared pointer to an llvm:LLVMContext associated with the
    ///   execution engine
    /// @param engine Shared pointer to an llvm::ExecutionEngine used to build
    ///   functions. Context should be the associated LLVMContext
    /// @param attributeRegistry Registry of attributes accessed by AX code
    /// @param customData Custom data which will be shared by this executable.
    ///   It can be used to retrieve external data from within the AX code
    /// @param functions A map of function names to physical memory addresses
    ///   which were built by llvm using engine
    PointExecutable(const std::shared_ptr<const llvm::LLVMContext>& context,
                    const std::shared_ptr<const llvm::ExecutionEngine>& engine,
                    const AttributeRegistry::ConstPtr& attributeRegistry,
                    const CustomData::ConstPtr& customData,
                    const std::unordered_map<std::string, uint64_t>& functions);

private:
    // The Context and ExecutionEngine must exist _only_ for object lifetime
    // management. The ExecutionEngine must be destroyed before the Context
    const std::shared_ptr<const llvm::LLVMContext> mContext;
    const std::shared_ptr<const llvm::ExecutionEngine> mExecutionEngine;
    const AttributeRegistry::ConstPtr mAttributeRegistry;
    const CustomData::ConstPtr mCustomData;
    const std::unordered_map<std::string, uint64_t> mFunctionAddresses;
    std::unique_ptr<Settings> mSettings;
};

}
}
}

#endif // OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED

