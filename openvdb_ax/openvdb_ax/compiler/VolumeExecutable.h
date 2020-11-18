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

#include "CustomData.h"
#include "AttributeRegistry.h"

#include <openvdb/version.h>
#include <openvdb/Grid.h>

#include <unordered_map>

class TestVolumeExecutable;

namespace llvm {
class ExecutionEngine;
class LLVMContext;
}

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {

class Compiler;

/// @brief Object that encapsulates compiled AX code which can be executed on a collection of
///        VDB volume grids
class VolumeExecutable
{
public:
    using Ptr = std::shared_ptr<VolumeExecutable>;
    ~VolumeExecutable();

    /// @brief  Copy constructor. Shares the LLVM constructs but deep copies the
    ///   settings. Multiple copies of an executor can be used at the same time
    ///   safely.
    VolumeExecutable(const VolumeExecutable& other);

    ////////////////////////////////////////////////////////

    /// @brief Execute AX code on target grids
    void execute(openvdb::GridPtrVec& grids) const;
    void execute(openvdb::GridBase& grid) const;

    ////////////////////////////////////////////////////////

    /// @brief  Set the behaviour when missing grids are accessed. Default
    ///    behaviour is true, which creates them with default transforms and
    ///    background values
    /// @param flag  Enables or disables the creation of missing attributes
    void setCreateMissing(const bool flag);
    /// @return  Whether this executable will generate new grids.
    bool getCreateMissing() const;

    /// @brief  Set the execution level for this executable. This controls what
    ///   nodes are processed when execute is called. Possible values depend on
    ///   the OpenVDB configuration in use however a value of 0 is the default
    ///   and  will always correspond to the lowest level (leaf-level).
    /// @note A value larger that the number of levels in the tree (i.e. larger
    ///   than the tree depth) will cause this method to throw a runtime error.
    /// @warning  Executing over tiles with compiled code designed for voxel
    ///   level access may produce incorrect results. This is typically the
    ///   case when accessing VDBs with mismatching topology. Consider
    ///   voxelizing tiles where necessary.
    /// @param level The tree execution level to set
    void setTreeExecutionLevel(const Index level);
    /// @return  The tree execution level. Default is 0 i.e. the leaf level
    Index getTreeExecutionLevel() const;

    enum class IterType { ON, OFF, ALL };
    /// @brief  Set the value iterator type to use with this executable. Options
    ///  are ON, OFF, ALL. Default is ON.
    /// @param iter The value iterator type to set
    void setValueIterator(const IterType& iter);
    /// @return  The current value iterator type
    IterType getValueIterator() const;

    /// @brief  Set the threading grain size. Default is 1. A value of 0 has the
    ///   effect of disabling multi-threading.
    /// @param grain The grain size
    void setGrainSize(const size_t grain);
    /// @return  The current grain size
    size_t getGrainSize() const;

    ////////////////////////////////////////////////////////

    // @brief deprecated methods
    OPENVDB_DEPRECATED void
    execute(openvdb::GridPtrVec& grids,
        const IterType iter,
        const bool create) const
    {
        VolumeExecutable copy(*this);
        copy.setValueIterator(iter);
        copy.setCreateMissing(create);
        copy.execute(grids);
    }

    OPENVDB_DEPRECATED void
    execute(openvdb::GridPtrVec& grids,
        const IterType iter) const
    {
        VolumeExecutable copy(*this);
        copy.setValueIterator(iter);
        copy.execute(grids);
    }

    ////////////////////////////////////////////////////////

    // foward declaration of settings for this executable
    struct Settings;

private:
    friend class Compiler;
    friend class ::TestVolumeExecutable;

    /// @brief Constructor, expected to be invoked by the compiler. Should not
    ///   be invoked directly.
    /// @param context Shared pointer to an llvm:LLVMContext associated with the
    ///   execution engine
    /// @param engine Shared pointer to an llvm::ExecutionEngine used to build
    ///   functions. Context should be the associated LLVMContext
    /// @param accessRegistry Registry of volumes accessed by AX code
    /// @param customData Custom data which will be shared by this executable.
    ///   It can be used to retrieve external data from within the AX code
    /// @param functions A map of function names to physical memory addresses
    ///   which were built by llvm using engine
    VolumeExecutable(const std::shared_ptr<const llvm::LLVMContext>& context,
        const std::shared_ptr<const llvm::ExecutionEngine>& engine,
        const AttributeRegistry::ConstPtr& accessRegistry,
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

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_VOLUME_EXECUTABLE_HAS_BEEN_INCLUDED

