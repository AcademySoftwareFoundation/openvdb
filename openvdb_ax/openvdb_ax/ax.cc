// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "ax.h"
#include "ast/AST.h"
#include "compiler/Compiler.h"
#include "compiler/PointExecutable.h"
#include "compiler/VolumeExecutable.h"

#include <openvdb/util/Assert.h>

#include <llvm/Support/TargetSelect.h> // InitializeNativeTarget
#include <llvm/Support/ManagedStatic.h> // llvm_shutdown

#include <mutex>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {

/// @note Implementation for initialize, isInitialized and uninitialized
///       reamins in compiler/Compiler.cc

void run(const char* ax, openvdb::GridBase& grid, const AttributeBindings& bindings)
{
    // Construct a generic compiler
    openvdb::ax::Compiler compiler;

    if (grid.isType<points::PointDataGrid>()) {
        // Compile for Point support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::PointExecutable::Ptr exe =
            compiler.compile<openvdb::ax::PointExecutable>(ax);
        OPENVDB_ASSERT(exe);

        //Set the attribute bindings
        exe->setAttributeBindings(bindings);
        // Execute on the provided points
        // @note  Throws on invalid point inputs such as mismatching types
        exe->execute(static_cast<points::PointDataGrid&>(grid));
    }
    else {
        // Compile for numerical grid support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::VolumeExecutable::Ptr exe =
            compiler.compile<openvdb::ax::VolumeExecutable>(ax);
        OPENVDB_ASSERT(exe);

        // Set the attribute bindings
        exe->setAttributeBindings(bindings);
        // Execute on the provided numerical grid
        // @note  Throws on invalid grid inputs such as mismatching types
        exe->execute(grid);
    }
}

void run(const char* ax, openvdb::GridPtrVec& grids, const AttributeBindings& bindings)
{
    if (grids.empty()) return;
    // Check the type of all grids. If they are all points, run for point data.
    // Otherwise, run for numerical volumes. Throw if the container has both.
    const bool points = grids.front()->isType<points::PointDataGrid>();
    for (auto& grid : grids) {
        if (points ^ grid->isType<points::PointDataGrid>()) {
            OPENVDB_THROW(AXCompilerError,
                "Unable to process both OpenVDB Points and OpenVDB Volumes in "
                "a single invocation of ax::run()");
        }
    }
    // Construct a generic compiler
    openvdb::ax::Compiler compiler;

    if (points) {
        // Compile for Point support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::PointExecutable::Ptr exe =
            compiler.compile<openvdb::ax::PointExecutable>(ax);
        OPENVDB_ASSERT(exe);

        //Set the attribute bindings
        exe->setAttributeBindings(bindings);
        // Execute on the provided points individually
        // @note  Throws on invalid point inputs such as mismatching types
        for (auto& grid : grids) {
            exe->execute(static_cast<points::PointDataGrid&>(*grid));
        }
    }
    else {
        // Compile for Volume support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::VolumeExecutable::Ptr exe =
            compiler.compile<openvdb::ax::VolumeExecutable>(ax);
        OPENVDB_ASSERT(exe);

        //Set the attribute bindings
        exe->setAttributeBindings(bindings);
        // Execute on the provided volumes
        // @note  Throws on invalid grid inputs such as mismatching types
        exe->execute(grids);
    }
}

namespace {
inline std::mutex& GetInitMutex()
{
    static std::mutex sInitMutex;
    return sInitMutex;
}
bool sIsInitialized = false;
bool sShutdown = false;
}

bool isInitialized()
{
    std::lock_guard<std::mutex> lock(GetInitMutex());
    return sIsInitialized;
}

void initialize()
{
    std::lock_guard<std::mutex> lock(GetInitMutex());
    if (sIsInitialized) return;

    if (sShutdown) {
        OPENVDB_THROW(AXCompilerError,
            "Unable to re-initialize LLVM target after uninitialize has been called.");
    }

    // Init JIT
    if (llvm::InitializeNativeTarget() ||
        llvm::InitializeNativeTargetAsmPrinter() ||
        llvm::InitializeNativeTargetAsmParser())
    {
        OPENVDB_THROW(AXCompilerError,
            "Failed to initialize LLVM target for JIT");
    }

    sIsInitialized = true;
}

void uninitialize()
{
    std::lock_guard<std::mutex> lock(GetInitMutex());
    if (!sIsInitialized) return;

    // @todo consider replacing with storage to Support/InitLLVM
    llvm::llvm_shutdown();

    sIsInitialized = false;
    sShutdown = true;
}

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

