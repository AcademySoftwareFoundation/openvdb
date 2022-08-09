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
#include "AttributeBindings.h"

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

/// @brief Object that encapsulates compiled AX code which can be executed on a
///   collection of VDB volume grids. Executables are created by the compiler
///   and hold the final immutable JIT compiled function and context.
/// @details  The VolumeExecutable is returned from the ax::Compiler when
///   compiling AX code for volume execution. The class represents a typical AX
///   executable object; immutable except for execution settings and implements
///   'execute' functions which can be called multiple times for arbitrary sets
///   of inputs. The intended usage of these executables is to configure their
///   runtime arguments and then call VolumeExecutable::execute with your VDBs.
///   For example:
/// @code
///   VolumeExecutable::Ptr exe = compiler.compile<VolumeExecutable>("@a += 1");
///   exe->setTreeExecutionLevel(0); // only process leaf nodes
///   exe->setValueIterator(VolumeExecutable::IterType::ALL); // process all values
///   exe->execute(vdbs); // run on a set of vdbs
///   exe->execute(grid); // run on a single vdb
/// @endcode
///
///   The Volume executable is initialised with specific configurable settings:
///   - Iteration Level: min=0, max=RootNode::Level.
///       By default, processes the entire VDB tree hierarchy.
///       @sa setTreeExecutionLevel
///   - Iteration Type: ON
///       By default, processes ACTIVE values.
///       @sa setValueIterator
///   - Active Tile Streaming: ON, OFF or AUTO depending on AX code.
///       By default, if AX detects that the AX program may produce unique
///       values for leaf level voxels that would otherwise comprise a
///       given active tile, this setting is set to ON or AUTO. Otherwise it is
///       set to OFF.
///       @sa setActiveTileStreaming
///   - Grain sizes: 1:32
///       The default grain sizes passed to the tbb partitioner for leaf level
///       processing and active tile processing.
///       @sa setGrainSize
///       @sa setActiveTileStreamingGrainSize
///   - AttributeBindings: None
///       Whether to indriect any AX accesses to different grid names.
///       @sa setAttributeBindings
///
///  For more in depth information, see the @ref vdbaxcompilerexe documentation.
class OPENVDB_AX_API VolumeExecutable
{
public:
    using Ptr = std::shared_ptr<VolumeExecutable>;
    ~VolumeExecutable();

    /// @brief  Copy constructor. Shares the LLVM constructs but deep copies the
    ///   settings. Multiple copies of an executor can be used at the same time
    ///   safely.
    VolumeExecutable(const VolumeExecutable& other);

    ////////////////////////////////////////////////////////

    ///@{
    /// @brief  Run this volume executable binary on target volumes.
    /// @details  This method reads from the stored settings on the executable
    ///   to determine certain behaviour and runs the JIT compiled function
    ///   across every valid VDB value. Topology may be changed, deleted or
    ///   created.
    ///
    ///   This method is thread safe; it can be run concurrently from the same
    ///   VolumeExecutable instance on different inputs.
    ///
    /// @param grids  The VDB Volumes to process
    void execute(openvdb::GridPtrVec& grids) const;
    void execute(openvdb::GridBase& grids) const;
    ///@}

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
    ///   the OpenVDB configuration in use, however a value of 0 will always
    ///   correspond to the lowest level (leaf-level). By default, the min
    ///   level is zero (LeafNodeType::LEVEL) and the max level is the root
    ///   node's level (RootNodeType::LEVEL). In other words, the default
    ///   execution level settings process the whole of the tree.
    /// @note A value larger that the number of levels in the tree (i.e. larger
    ///   than the root node's level) will cause this method to throw a runtime
    ///   error.
    /// @param min The minimum tree execution level to set
    /// @param max The maximum tree execution level to set
    void setTreeExecutionLevel(const Index min, const Index max);
    /// @param level The tree execution level to set. Calls setTreeExecutionLevel
    ///   with min and max arguments as level.
    void setTreeExecutionLevel(const Index level);
    /// @brief  Get the tree execution levels.
    /// @param min The minimum tree execution level
    /// @param max The maximum tree execution level
    void getTreeExecutionLevel(Index& min, Index& max) const;

    /// @brief  The streaming type of active tiles during execution.
    /// @param  ON active tiles are temporarily densified (converted to leaf
    ///     level voxels) on an "as needed" basis and the subsequent voxel
    ///     values are processed. The temporarily densified node is added to the
    ///     tree only if a non constant voxel buffer is produced. Otherwise a
    ///     child tile may be created or the original tile's value may simply be
    ///     modified.
    /// @param  OFF tile topologies are left unchanged and their single value is
    ///     processed.
    /// @param  AUTO the volume executable analyzes the compiled kernel and
    ///     attempts to determine if expansion of active tiles would lead to
    ///     different, non-constant values in the respective voxels. This is
    ///     done on a per grid basis; ultimately each execution will be set to
    ///     ON or OFF. This option will always fall back to ON if there is any
    ///     chance the kernel may produce child nodes
    ///
    /// @note The volume executable always runs an AUTO check on creation and
    ///   will set itself to ON (if all grids always need child nodes), OFF (if
    ///   grids never need child nodes) or remains as AUTO (if this depends on
    ///   which grid is being processed).
    ///
    /// @details When an AX kernel is run over coarser levels of the tree (i.e.
    ///   not leaf voxels), it is often desirable to densify these areas into
    ///   unique voxels such that they can each receive a unique value. For
    ///   example, consider the following AX code which assigns a vector volume
    ///   to the world space position of each voxel:
    /// @code
    ///      v@v = getvoxelpws();
    /// @endcode
    ///   Active tiles hold a single value but comprise an area greater than
    ///   that of a single voxel. As the above kernel varies with respect to
    ///   a nodes position, we'd need to replace these tiles with leaf voxels
    ///   to get unique per node values. The stream flag is initialised to ON
    ///   in this case.
    ///
    ///   This behaviour, however, is not always desirable .i.e:
    /// @code
    ///      v@v = {1,2,3};
    /// @endcode
    ///   In this instance, all values within a volume receive the same value
    ///   and are not dependent on any spatially or iteratively varying
    ///   metrics. The stream flag is set to OFF.
    ///
    ///   The AUTO flag is set in cases where the runtime access pattern of the
    ///   inputs determines streaming:
    /// @code
    ///     f@density = f@mask;
    ///     f@mask = 0;
    /// @endcode
    ///   In this instance, the runtime topology and values of \@mask determines
    ///   whether child topology needs to be created in \@density, but \@mask
    ///   itself does not need streaming. Streaming will be set to ON for
    ///   density but OFF for mask.
    ///
    /// @note This behaviour is only applied to active tiles. If the value
    ///   iterator is set to OFF, this option is ignored.
    /// @warning  This option can generate large amounts of leaf level voxels.
    ///   It is recommended to use a good concurrent memory allocator (such as
    ///   jemalloc) for the best performance.
    enum class Streaming { ON, OFF, AUTO };
    /// @brief  Controls the behaviour of expansion of active tiles.
    /// @param s The behaviour to set
    void setActiveTileStreaming(const Streaming& s);
    /// @return  The current stream behaviour.
    Streaming getActiveTileStreaming() const;
    /// @return  The current stream behaviour for a particular grid. This is
    ///   either ON or OFF.
    /// @param name The name of the grid to query
    /// @param type The grids type
    Streaming getActiveTileStreaming(const std::string& name,
        const ast::tokens::CoreType& type) const;

    enum class IterType { ON, OFF, ALL };
    /// @brief  Set the value iterator type to use with this executable. Options
    ///  are ON, OFF, ALL. Default is ON.
    /// @param iter The value iterator type to set
    void setValueIterator(const IterType& iter);
    /// @return  The current value iterator type
    IterType getValueIterator() const;

    ///@{
    /// @brief  Set the threading grain sizes used when iterating over nodes
    ///  in a VDB.
    /// @details  Two grain sizes are provided, the first of which (g1) is used
    ///  to determine the chunk size of nodes which are not being streamed (see
    ///  setActiveTileStream). Leaf node execution always uses this grain size.
    ///  The default value for g1 is 1 which is typically appropriate for most
    ///  AX kernels.
    ///  The second grain size is used when streaming execution over active
    ///  tiles in a VDB. This execution model differs significantly from
    ///  typical leaf node execution due to the potential for substantially
    ///  more memory to be allocated. The default value is 32, which works well
    ///  for the default configuration of OpenVDB. If streaming is disabled,
    ///  this value has no effect.
    /// @note Setting g1 or g2 to zero has the effect of disabling
    ///  multi-threading for the respective node executions. Setting both to
    ///  zero will disable all multi-threading performed by the execute method.
    void setGrainSize(const size_t g1);
    void setActiveTileStreamingGrainSize(const size_t g2);
    /// @return  The current g1 grain size
    /// @sa setGrainSize
    size_t getGrainSize() const;
    /// @return  The current g2 grain size
    /// @sa setActiveTileStreamingGrainSize
    size_t getActiveTileStreamingGrainSize() const;
    ///@}

    /// @brief  Set attribute bindings.
    /// @param bindings A map of attribute bindings to expected names on
    ///   the geometry to be executed over. By default the AX attributes will be
    ///   bound to volumes of the same name. Supplying bindings
    ///   for a subset of the attributes will leave the others unchanged.
    ///   AX attributes can only bind to a single volume and vice versa.
    ///   However, in a single set call these can be swapped e.g. a -> b and b -> a.
    ///   When bindings are overriden through subsequent calls to this function,
    ///   any dangling volumes will be automatically bound by name.
    ///   To reset these bindings call get function and create a target set of bindings
    ///   for each attribute of name -> name.
    void setAttributeBindings(const AttributeBindings& bindings);
    /// @return  The current attribute bindings map
    const AttributeBindings& getAttributeBindings() const;

    ////////////////////////////////////////////////////////

    // foward declaration of settings for this executable
    template <bool> struct Settings;

    /// @brief Command Line Interface handling for the VolumeExecutable.
    /// @details  This class wraps the logic for converting commands specific
    ///   to the VolumeExecutable to the internal Settings. Subsequent
    ///   executables can be initialized from the CLI object that gets created.
    struct OPENVDB_AX_API CLI
    {
        ~CLI();
        CLI(CLI&&);
        CLI& operator=(CLI&&);
        static CLI create(size_t argc, const char* argv[], bool* used=nullptr);
        static void usage(std::ostream& os, const bool verbose);
    private:
        friend class VolumeExecutable;
        CLI();
        std::unique_ptr<Settings<true>> mSettings;
    };

    /// @brief  Intialize the Settings of this executables from the CLI object
    /// @param cli The CLI object
    /// @{
    void setSettingsFromCLI(const CLI& cli);
    /// @}

    ////////////////////////////////////////////////////////

    /// @return  The tree execution level.
    OPENVDB_DEPRECATED_MESSAGE("Use getTreeExecutionLevel(Index&, Index&)")
    Index getTreeExecutionLevel() const;

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
    /// @param tree The AST linked to this executable. The AST is not stored
    ///   after compilation but can be used during construction of the exe to
    ///   infer some pre/post processing optimisations.
    VolumeExecutable(const std::shared_ptr<const llvm::LLVMContext>& context,
        const std::shared_ptr<const llvm::ExecutionEngine>& engine,
        const AttributeRegistry::ConstPtr& accessRegistry,
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

#endif // OPENVDB_AX_COMPILER_VOLUME_EXECUTABLE_HAS_BEEN_INCLUDED

