// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "ax.h"
#include "ast/AST.h"
#include "compiler/Logger.h"
#include "compiler/Compiler.h"
#include "compiler/PointExecutable.h"
#include "compiler/VolumeExecutable.h"

#include <llvm/InitializePasses.h>
#include <llvm/PassRegistry.h>
#include <llvm/Config/llvm-config.h> // version numbers
#include <llvm/Support/TargetSelect.h> // InitializeNativeTarget
#include <llvm/Support/ManagedStatic.h> // llvm_shutdown
#include <llvm/ExecutionEngine/MCJIT.h> // LLVMLinkInMCJIT

#include <tbb/mutex.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {

/// @note Implementation for initialize, isInitialized and unitialized
///       reamins in compiler/Compiler.cc

void run(const char* ax, openvdb::GridBase& grid)
{
    // Construct a logger that will output errors to cerr and suppress warnings
    openvdb::ax::Logger logger;
    // Construct a generic compiler
    openvdb::ax::Compiler compiler;
    // Parse the provided code and produce an abstract syntax tree
    // @note  Throws with parser errors if invalid. Parsable code does not
    //        necessarily equate to compilable code
    const openvdb::ax::ast::Tree::ConstPtr
        ast = openvdb::ax::ast::parse(ax, logger);

    if (grid.isType<points::PointDataGrid>()) {
        // Compile for Point support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::PointExecutable::Ptr exe =
            compiler.compile<openvdb::ax::PointExecutable>(*ast, logger);
        // Execute on the provided points
        // @note  Throws on invalid point inputs such as mismatching types
        exe->execute(static_cast<points::PointDataGrid&>(grid));
    }
    else {
        // Compile for numerical grid support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::VolumeExecutable::Ptr exe =
            compiler.compile<openvdb::ax::VolumeExecutable>(*ast, logger);
        // Execute on the provided numerical grid
        // @note  Throws on invalid grid inputs such as mismatching types
        exe->execute(grid);
    }
}

void run(const char* ax, openvdb::GridPtrVec& grids)
{
    if (grids.empty()) return;
    // Check the type of all grids. If they are all points, run for point data.
    // Otherwise, run for numerical volumes.
    bool points = true;
    for (auto& grid : grids) {
        if (!grid->isType<points::PointDataGrid>()) {
            points = false;
            break;
        }
    }
    // Construct a logger that will output errors to cerr and suppress warnings
    openvdb::ax::Logger logger;
    // Construct a generic compiler
    openvdb::ax::Compiler compiler;
    // Parse the provided code and produce an abstract syntax tree
    // @note  Throws with parser errors if invalid. Parsable code does not
    //        necessarily equate to compilable code
    const openvdb::ax::ast::Tree::ConstPtr
        ast = openvdb::ax::ast::parse(ax, logger);
    if (points) {
        // Compile for Point support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::PointExecutable::Ptr exe =
            compiler.compile<openvdb::ax::PointExecutable>(*ast, logger);
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
            compiler.compile<openvdb::ax::VolumeExecutable>(*ast, logger);
        // Execute on the provided volumes
        // @note  Throws on invalid grid inputs such as mismatching types
        exe->execute(grids);
    }
}

namespace {
// Declare this at file scope to ensure thread-safe initialization.
tbb::mutex sInitMutex;
bool sIsInitialized = false;
bool sShutdown = false;
}

bool isInitialized()
{
    tbb::mutex::scoped_lock lock(sInitMutex);
    return sIsInitialized;
}

void initialize()
{
    tbb::mutex::scoped_lock lock(sInitMutex);
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

    // required on some systems
    LLVMLinkInMCJIT();

    // Initialize passes
    llvm::PassRegistry& registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(registry);
    llvm::initializeScalarOpts(registry);
    llvm::initializeObjCARCOpts(registry);
    llvm::initializeVectorization(registry);
    llvm::initializeIPO(registry);
    llvm::initializeAnalysis(registry);
    llvm::initializeTransformUtils(registry);
    llvm::initializeInstCombine(registry);
#if LLVM_VERSION_MAJOR > 6
    llvm::initializeAggressiveInstCombine(registry);
#endif
    llvm::initializeInstrumentation(registry);
    llvm::initializeTarget(registry);
    // For codegen passes, only passes that do IR to IR transformation are
    // supported.
    llvm::initializeExpandMemCmpPassPass(registry);
    llvm::initializeScalarizeMaskedMemIntrinPass(registry);
    llvm::initializeCodeGenPreparePass(registry);
    llvm::initializeAtomicExpandPass(registry);
    llvm::initializeRewriteSymbolsLegacyPassPass(registry);
    llvm::initializeWinEHPreparePass(registry);
    llvm::initializeDwarfEHPreparePass(registry);
    llvm::initializeSafeStackLegacyPassPass(registry);
    llvm::initializeSjLjEHPreparePass(registry);
    llvm::initializePreISelIntrinsicLoweringLegacyPassPass(registry);
    llvm::initializeGlobalMergePass(registry);
#if LLVM_VERSION_MAJOR > 6
    llvm::initializeIndirectBrExpandPassPass(registry);
#endif
#if LLVM_VERSION_MAJOR > 7
    llvm::initializeInterleavedLoadCombinePass(registry);
#endif
    llvm::initializeInterleavedAccessPass(registry);
    llvm::initializeEntryExitInstrumenterPass(registry);
    llvm::initializePostInlineEntryExitInstrumenterPass(registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(registry);
    llvm::initializeExpandReductionsPass(registry);
#if LLVM_VERSION_MAJOR > 6
    llvm::initializeWasmEHPreparePass(registry);
#endif
    llvm::initializeWriteBitcodePassPass(registry);

    sIsInitialized = true;
}

void uninitialize()
{
    tbb::mutex::scoped_lock lock(sInitMutex);
    if (!sIsInitialized) return;

    // @todo consider replacing with storage to Support/InitLLVM
    llvm::llvm_shutdown();

    sIsInitialized = false;
    sShutdown = true;
}

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
