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

#include "CustomData.h"
#include "AttributeRegistry.h"
#include "AttributeBindings.h"

#include <openvdb/openvdb.h>
#include <openvdb/version.h>
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

/// @brief Object that encapsulates compiled AX code which can be executed on a
///   collection of VDB Point Data grids. Executables are created by the
///   compiler and hold the final immutable JIT compiled function and context.
/// @details  The PointExecutable is returned from the ax::Compiler when
///   compiling AX code for point execution. The class represents a typical AX
///   executable object; immutable except for execution settings and implements
///   'execute' functions which can be called multiple times for arbitrary sets
///   of inputs. The intended usage of these executables is to configure their
///   runtime arguments and then call PointExecutable::execute with your VDBs.
///   For example:
/// @code
///   PointExecutable::Ptr exe = compiler.compile<PointExecutable>("@a += 1");
///   exe->setCreateMissing(false); // fail on missing attributes
///   exe->setGroupExecution("group1"); // only process points in group1
///   exe->execute(vdbs); // run on a set of vdb point data grids
///   exe->execute(points); // run on a single point data grid
/// @endcode
///
///   The setCreateMissing is initialised with specific configurable settings:
///   - Create Missing: True
///       By default, create any missing attributes that were accessed
///       @sa setCreateMissing
///   - Group Execution: All
///       By default, process all points regardless of their group membership
///       @sa setGroupExecution
///   - Grain size: 1
///       The default grain sizes passed to the tbb partitioner for leaf level
///       processing.
///       @sa setGrainSize
///
///  For more in depth information, see the @ref vdbaxcompilerexe documentation.
class OPENVDB_AX_API PointExecutable
{
public:
    using Ptr = std::shared_ptr<PointExecutable>;
    ~PointExecutable();

    /// @brief  Copy constructor. Shares the LLVM constructs but deep copies the
    ///   settings. Multiple copies of an executor can be used at the same time
    ///   safely.
    PointExecutable(const PointExecutable& other);

    ////////////////////////////////////////////////////////

    /// @brief  Run this point executable binary on a target PointDataGrid.
    /// @details  This method reads from the stored settings on the executable
    ///   to determine certain behaviour and runs the JIT compiled function
    ///   across every valid point. Point attributes may be created, deleted
    ///   collapsed or expanded, and points themselves may be added, deleted
    ///   or moved.
    ///
    ///   This method is thread safe; it can be run concurrently from the same
    ///   PointExecutable instance on different inputs.
    ///
    /// @param grid  The PointDataGrid to process
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

    /// @brief  Set attribute bindings.
    /// @param bindings A map of attribute bindings to expected names on
    ///   the geometry to be executed over. By default the AX attributes will be
    ///   bound to point attributes of the same name. Supplying bindings
    ///   for a subset of the attributes will leave the others unchanged.
    ///   AX attributes can only bind to a single point attribute and vice versa.
    ///   However, in a single set call these can be swapped e.g. a -> b and b -> a.
    ///   When bindings are overriden through subsequent calls to this function,
    ///   any dangling point attributes will be automatically bound by name.
    ///   To reset these bindings call get function and create a target set of bindings
    ///   for each attribute of name -> name.
    void setAttributeBindings(const AttributeBindings& bindings);
    /// @return  The current attribute bindings map
    const AttributeBindings& getAttributeBindings() const;

    ////////////////////////////////////////////////////////

    // foward declaration of settings for this executable
    template <bool> struct Settings;

    /// @brief Command Line Interface handling for the PointExecutable.
    /// @details  This class wraps the logic for converting commands specific
    ///   to the PointExecutable to the internal Settings. Subsequent
    ///   executables can be initialized from the CLI object that gets created
    struct OPENVDB_AX_API CLI
    {
        ~CLI();
        CLI(CLI&&);
        CLI& operator=(CLI&&);
        static CLI create(size_t argc, const char* argv[], bool* used=nullptr);
        static void usage(std::ostream& os, const bool verbose);
    private:
        friend class PointExecutable;
        CLI();
        std::unique_ptr<Settings<true>> mSettings;
    };

    /// @brief  Intialize the Settings of this executables from the CLI object
    /// @param cli The CLI object
    void setSettingsFromCLI(const CLI& cli);

    ////////////////////////////////////////////////////////

private:
    friend class Compiler;
    friend class ::TestPointExecutable;

    /// @brief  Private method used in the unit tests
    bool usesAcceleratedKernel(const points::PointDataTree& tree) const;

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
    /// @param tree The AST linked to this executable. The AST is not stored
    ///   after compilation, but can be used during construction of the exe to
    ///   infer some pre/post processing optimisations.
    PointExecutable(const std::shared_ptr<const llvm::LLVMContext>& context,
                    const std::shared_ptr<const llvm::ExecutionEngine>& engine,
                    const AttributeRegistry::ConstPtr& attributeRegistry,
                    const CustomData::ConstPtr& customData,
                    const std::unordered_map<std::string, uint64_t>& functions,
                    const ast::Tree& tree);

private:
    // The Context and ExecutionEngine must exist _only_ for object lifetime
    // management. The ExecutionEngine must be destroyed before the Context
    const std::shared_ptr<const llvm::LLVMContext> mContext;
    const std::shared_ptr<const llvm::ExecutionEngine> mExecutionEngine;
    const AttributeRegistry::ConstPtr mAttributeRegistry;
    const CustomData::ConstPtr mCustomData;
    const std::unordered_map<std::string, uint64_t> mFunctionAddresses;
    std::unique_ptr<Settings<false>> mSettings;
};

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED

