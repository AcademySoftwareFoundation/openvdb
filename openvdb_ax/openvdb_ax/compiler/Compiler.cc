// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/Compiler.cc

#include "Compiler.h"

#include "PointExecutable.h"
#include "VolumeExecutable.h"

#include "../ast/Scanners.h"
#include "../codegen/Functions.h"
#include "../codegen/PointComputeGenerator.h"
#include "../codegen/VolumeComputeGenerator.h"
#include "../Exceptions.h"

#include <openvdb/Exceptions.h>

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h> // SMDiagnostic
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

// @note  As of adding support for LLVM 5.0 we not longer explicitly
// perform standrd compiler passes (-std-compile-opts) based on the changes
// to the opt binary in the llvm codebase (tools/opt.cpp). We also no
// longer explicitly perform:
//  - llvm::createStripSymbolsPass()
// And have never performed any specific target machine analysis passes
//
// @todo  Properly identify the IPO passes that we would benefit from using
// as well as what user controls would otherwise be appropriate

#include <llvm/Transforms/IPO.h> // Inter-procedural optimization passes
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

namespace
{

/// @brief  Initialize a target machine for the host platform. Returns a nullptr
///         if a target could not be created.
/// @note   This logic is based off the Kaleidoscope tutorial below with extensions
///         for CPU and CPU featrue set targetting
///         https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl08.html
inline std::unique_ptr<llvm::TargetMachine> initializeTargetMachine()
{
    const std::string TargetTriple = llvm::sys::getDefaultTargetTriple();
    std::string Error;
    const llvm::Target* Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
    if (!Target) {
        OPENVDB_LOG_DEBUG_RUNTIME("Unable to retrieve target machine information. "
            "No target specific optimization will be performed: " << Error);
        return nullptr;
    }

    // default cpu with no additional features = "generic"
    const llvm::StringRef& CPU = llvm::sys::getHostCPUName();

    llvm::SubtargetFeatures Features;
    llvm::StringMap<bool> HostFeatures;
    if (llvm::sys::getHostCPUFeatures(HostFeatures))
      for (auto &F : HostFeatures)
        Features.AddFeature(F.first(), F.second);

    // default options
    llvm::TargetOptions opt;
    const llvm::Optional<llvm::Reloc::Model> RM =
        llvm::Optional<llvm::Reloc::Model>();

    std::unique_ptr<llvm::TargetMachine> TargetMachine(
        Target->createTargetMachine(TargetTriple, CPU, Features.getString(), opt, RM));

    return TargetMachine;
}

#ifndef USE_NEW_PASS_MANAGER

void addStandardLinkPasses(llvm::legacy::PassManagerBase& passes)
{
    llvm::PassManagerBuilder builder;
    builder.VerifyInput = true;
    builder.Inliner = llvm::createFunctionInliningPass();
    builder.populateLTOPassManager(passes);
}

/// This routine adds optimization passes based on selected optimization level
///
void addOptimizationPasses(llvm::legacy::PassManagerBase& passes,
                           llvm::legacy::FunctionPassManager& functionPasses,
                           llvm::TargetMachine* targetMachine,
                           const unsigned optLevel,
                           const unsigned sizeLevel,
                           const bool disableInline = false,
                           const bool disableUnitAtATime = false,
                           const bool disableLoopUnrolling = false,
                           const bool disableLoopVectorization = false,
                           const bool disableSLPVectorization = false)
{
    llvm::PassManagerBuilder builder;
    builder.OptLevel = optLevel;
    builder.SizeLevel = sizeLevel;

    if (disableInline) {
        // No inlining pass
    } else if (optLevel > 1) {
        builder.Inliner =
            llvm::createFunctionInliningPass(optLevel, sizeLevel,
                /*DisableInlineHotCallSite*/false);
    } else {
        builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
    }

#if LLVM_VERSION_MAJOR < 9
    // Enable IPO. This corresponds to gcc's -funit-at-a-time
    builder.DisableUnitAtATime = disableUnitAtATime;
#else
    // unused from llvm 9
    (void)(disableUnitAtATime);
#endif

    // Disable loop unrolling in all relevant passes
    builder.DisableUnrollLoops =
        disableLoopUnrolling ? disableLoopUnrolling : optLevel == 0;

    // See the following link for more info on vectorizers
    // http://llvm.org/docs/Vectorizers.html

    // (-vectorize-loops, -loop-vectorize)
    builder.LoopVectorize =
        disableLoopVectorization ? false : optLevel > 1 && sizeLevel < 2;
    builder.SLPVectorize =
        disableSLPVectorization ? false : optLevel > 1 && sizeLevel < 2;

    // If a target machine is provided, allow the target to modify the pass manager
    // e.g. by calling PassManagerBuilder::addExtension.
    if (targetMachine) {
        targetMachine->adjustPassManager(builder);
    }

    builder.populateFunctionPassManager(functionPasses);
    builder.populateModulePassManager(passes);
}

void LLVMoptimise(llvm::Module* module,
                  const unsigned optLevel,
                  const unsigned sizeLevel,
                  const bool verify,
                  llvm::TargetMachine* TM)
{
    // Pass manager setup and IR optimisations - Do target independent optimisations
    // only - i.e. the following do not require an llvm TargetMachine analysis pass

    llvm::legacy::PassManager passes;
    llvm::TargetLibraryInfoImpl TLII(llvm::Triple(module->getTargetTriple()));
    passes.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

    // Add internal analysis passes from the target machine.
    if (TM) passes.add(llvm::createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));
    else    passes.add(llvm::createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));

    llvm::legacy::FunctionPassManager functionPasses(module);
    if (TM) functionPasses.add(llvm::createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));
    else    functionPasses.add(llvm::createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));

    if (verify) functionPasses.add(llvm::createVerifierPass());

    addStandardLinkPasses(passes);
    addOptimizationPasses(passes, functionPasses, TM, optLevel, sizeLevel);

    functionPasses.doInitialization();
    for (llvm::Function& function : *module) {
      functionPasses.run(function);
    }
    functionPasses.doFinalization();

    if (verify) passes.add(llvm::createVerifierPass());
    passes.run(*module);
}

void LLVMoptimise(llvm::Module* module,
                  const llvm::PassBuilder::OptimizationLevel opt,
                  const bool verify,
                  llvm::TargetMachine* TM)
{
    unsigned optLevel = 0, sizeLevel = 0;

    // llvm::PassBuilder::OptimizationLevel is an enum in llvm 10
    // and earlier, a class in llvm 11 and later (which holds
    // various member data about the optimization level)
#if LLVM_VERSION_MAJOR < 11
    switch (opt) {
        case llvm::PassBuilder::OptimizationLevel::O0 : {
            optLevel = 0; sizeLevel = 0;
            break;
        }
        case llvm::PassBuilder::OptimizationLevel::O1 : {
            optLevel = 1; sizeLevel = 0;
            break;
        }
        case llvm::PassBuilder::OptimizationLevel::O2 : {
            optLevel = 2; sizeLevel = 0;
            break;
        }
        case llvm::PassBuilder::OptimizationLevel::Os : {
            optLevel = 2; sizeLevel = 1;
            break;
        }
        case llvm::PassBuilder::OptimizationLevel::Oz : {
            optLevel = 2; sizeLevel = 2;
            break;
        }
        case llvm::PassBuilder::OptimizationLevel::O3 : {
            optLevel = 3; sizeLevel = 0;
            break;
        }
        default : {}
    }
#else
    optLevel = opt.getSpeedupLevel();
    sizeLevel = opt.getSizeLevel();
#endif

    LLVMoptimise(module, optLevel, sizeLevel, verify, TM);
}

#else

void LLVMoptimise(llvm::Module* module,
                  const llvm::PassBuilder::OptimizationLevel optLevel,
                  const bool verify,
                  llvm::TargetMachine* TM)
{
    // use the PassBuilder for optimisation pass management
    // see llvm's llvm/Passes/PassBuilder.h, tools/opt/NewPMDriver.cpp
    // and clang's CodeGen/BackEndUtil.cpp for more info/examples
    llvm::PassBuilder PB(TM);

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager cGSCCAM;
    llvm::ModuleAnalysisManager MAM;

    // register all of the analysis passes available by default
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(cGSCCAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);

    // the analysis managers above are interdependent so
    // register dependent managers with each other via proxies
    PB.crossRegisterProxies(LAM, FAM, cGSCCAM, MAM);

    // the PassBuilder does not produce -O0 pipelines, so do that ourselves
    if (optLevel == llvm::PassBuilder::OptimizationLevel::O0) {
        // matching clang -O0, only add inliner pass
        // ref: clang CodeGen/BackEndUtil.cpp EmitAssemblyWithNewPassManager
        llvm::ModulePassManager MPM;
        MPM.addPass(llvm::AlwaysInlinerPass());
        if (verify) MPM.addPass(llvm::VerifierPass());
        MPM.run(*module, MAM);
    }
    else {
        // create a clang-like optimisation pipeline for -O1, 2,  s, z, 3
        llvm::ModulePassManager MPM =
            PB.buildPerModuleDefaultPipeline(optLevel);
        if (verify) MPM.addPass(llvm::VerifierPass());
        MPM.run(*module, MAM);
    }
}
#endif

void optimiseAndVerify(llvm::Module* module,
        const bool verify,
        const CompilerOptions::OptLevel optLevel,
        llvm::TargetMachine* TM)
{
    if (verify) {
        llvm::raw_os_ostream out(std::cout);
        if (llvm::verifyModule(*module, &out)) {
            OPENVDB_THROW(AXCompilerError, "Generated LLVM IR is not valid.");
        }
    }

    switch (optLevel) {
        case CompilerOptions::OptLevel::O0 : {
            LLVMoptimise(module, llvm::PassBuilder::OptimizationLevel::O0, verify, TM);
            break;
        }
        case CompilerOptions::OptLevel::O1 : {
            LLVMoptimise(module, llvm::PassBuilder::OptimizationLevel::O1, verify, TM);
            break;
        }
        case CompilerOptions::OptLevel::O2 : {
            LLVMoptimise(module, llvm::PassBuilder::OptimizationLevel::O2, verify, TM);
            break;
        }
        case CompilerOptions::OptLevel::Os : {
            LLVMoptimise(module, llvm::PassBuilder::OptimizationLevel::Os, verify, TM);
            break;
        }
        case CompilerOptions::OptLevel::Oz : {
            LLVMoptimise(module, llvm::PassBuilder::OptimizationLevel::Oz, verify, TM);
            break;
        }
        case CompilerOptions::OptLevel::O3 : {
            LLVMoptimise(module, llvm::PassBuilder::OptimizationLevel::O3, verify, TM);
            break;
        }
        case CompilerOptions::OptLevel::NONE :
        default             : {}
    }
}

void initializeGlobalFunctions(const codegen::FunctionRegistry& registry,
                               llvm::ExecutionEngine& engine,
                               llvm::Module& module)
{
    /// @note  This is a copy of ExecutionEngine::getMangledName. LLVM's ExecutionEngine
    ///   provides two signatures for updating global mappings, one which takes a void* and
    ///   another which takes a uint64_t address. When providing function mappings,
    ///   it is potentially unsafe to cast pointers-to-functions to pointers-to-objects
    ///   as they are not guaranteed to have the same size on some (albiet non "standard")
    ///   platforms. getMangledName is protected, so a copy exists here to allows us to
    ///   call the uint64_t method.
    /// @note  This is only caught by -pendantic so this work around may be overkill
    auto getMangledName = [](const llvm::GlobalValue* GV,
                        const llvm::ExecutionEngine& E) -> std::string
    {
        llvm::SmallString<128> FullName;
        const llvm::DataLayout& DL =
            GV->getParent()->getDataLayout().isDefault()
                ? E.getDataLayout()
                : GV->getParent()->getDataLayout();
        llvm::Mangler::getNameWithPrefix(FullName, GV->getName(), DL);
        return std::string(FullName.str());
    };

    /// @note  Could use InstallLazyFunctionCreator here instead as follows:
    ///
    /// engine.InstallLazyFunctionCreator([](const std::string& name) -> void * {
    ///    // Loop through register and find matching symbol
    /// });
    ///
    /// However note that if functions have been compiled with mLazyFunctions that the
    /// below code using addGlobalMapping() only adds mapping for instantiated
    /// functions anyway.
    ///
    /// @note  Depending on how functions are inserted into LLVM (Linkage Type) in
    ///        the future, InstallLazyFunctionCreator may be required
    for (const auto& iter : registry.map()) {
        const codegen::FunctionGroup* const function = iter.second.function();
        if (!function) continue;

        const codegen::FunctionGroup::FunctionList& list = function->list();
        for (const codegen::Function::Ptr& decl : list) {

            // llvmFunction may not exists if compiled without mLazyFunctions
            const llvm::Function* llvmFunction = module.getFunction(decl->symbol());

            // if the function has an entry block, it's not a C binding - this is a
            // quick check to improve performance (so we don't call virtual methods
            // for every function)
            if (!llvmFunction) continue;
            if (llvmFunction->size() > 0) continue;

            const codegen::CFunctionBase* binding =
                dynamic_cast<const codegen::CFunctionBase*>(decl.get());
            if (!binding) {
                OPENVDB_LOG_WARN("Function with symbol \"" << decl->symbol() << "\" has "
                    "no function body and is not a C binding.");
                continue;
            }

            const uint64_t address = binding->address();
            if (address == 0) {
                OPENVDB_THROW(AXCompilerError, "No available mapping for C Binding "
                    "with symbol \"" << decl->symbol() << "\"");
            }
            const std::string mangled =
                getMangledName(llvm::cast<llvm::GlobalValue>(llvmFunction), engine);

            // error if updateGlobalMapping returned a previously mapped address, as
            // we've overwritten something
            const uint64_t oldAddress = engine.updateGlobalMapping(mangled, address);
            if (oldAddress != 0 && oldAddress != address) {
                OPENVDB_THROW(AXCompilerError, "Function registry mapping error - "
                    "multiple functions are using the same symbol \"" << decl->symbol()
                    << "\".");
            }
        }
    }

#ifndef NDEBUG
    // Loop through all functions and check to see if they have valid engine mappings.
    // This can occur if lazy functions don't initialize their dependencies properly.
    // @todo  Really we should just loop through the module functions to begin with
    //  to init engine mappings - it would probably be faster but we'd have to do
    //  some string manip and it would assume function names have been set up
    //  correctly
    const auto& list = module.getFunctionList();
    for (const auto& F : list) {
        if (F.size() > 0) continue;
        // Some LLVM functions may also not be defined at this stage which is expected
        if (!F.getName().startswith("ax")) continue;
        const std::string mangled =
            getMangledName(llvm::cast<llvm::GlobalValue>(&F), engine);
        const uint64_t address =
            engine.getAddressToGlobalIfAvailable(mangled);
        assert(address != 0 && "Unbound function!");
    }
#endif
}

void verifyTypedAccesses(const ast::Tree& tree, openvdb::ax::Logger& logger)
{
    // verify the attributes and external variables requested in the syntax tree
    // only have a single type. Note that the executer will also throw a runtime
    // error if the same attribute is accessed with different types, but as that's
    // currently not a valid state on a PointDataGrid, error in compilation as well
    // @todo - introduce a framework for supporting custom preprocessors

    std::unordered_map<std::string, std::string> nameType;

    auto attributeOp =
        [&nameType, &logger](const ast::Attribute& node) -> bool {
            auto iter = nameType.find(node.name());
            if (iter == nameType.end()) {
                nameType[node.name()] = node.typestr();
            }
            else if (iter->second != node.typestr()) {
                logger.error("failed to compile ambiguous @ parameters. "
                    "\"" + node.name() + "\" has been accessed with different type elsewhere.", &node);
            }
            return true;
        };

    ast::visitNodeType<ast::Attribute>(tree, attributeOp);

    nameType.clear();

    auto externalOp =
        [&nameType, &logger](const ast::ExternalVariable& node) -> bool {
            auto iter = nameType.find(node.name());
            if (iter == nameType.end()) {
                nameType[node.name()] = node.typestr();
            }
            else if (iter->second != node.typestr()) {
                logger.error("failed to compile ambiguous $ parameters. "
                   "\"" + node.name() + "\" has been accessed with different type elsewhere.", &node);
            }
            return true;
        };

    ast::visitNodeType<ast::ExternalVariable>(tree, externalOp);
}

inline void
registerAccesses(const codegen::SymbolTable& globals, const AttributeRegistry& registry)
{
    std::string name, type;

    for (const auto& global : globals.map()) {

        // detect if this global variable is an attribute access
        const std::string& token = global.first;
        if (!ast::Attribute::nametypeFromToken(token, &name, &type)) continue;

        const ast::tokens::CoreType typetoken =
            ast::tokens::tokenFromTypeString(type);

        // add the access to the registry - this will force the executables
        // to always request or create the data type

        const size_t index = registry.accessIndex(name, typetoken);

        // should always be a GlobalVariable.
        assert(llvm::isa<llvm::GlobalVariable>(global.second));

        // Assign the attribute index global a valid index.
        // @note executionEngine->addGlobalMapping() can also be used if the indices
        // ever need to vary positions without having to force a recompile (previously
        // was used unnecessarily)

        llvm::GlobalVariable* variable =
            llvm::cast<llvm::GlobalVariable>(global.second);
        assert(variable->getValueType()->isIntegerTy(64));

        variable->setInitializer(llvm::ConstantInt::get(variable->getValueType(), index));
        variable->setConstant(true); // is not written to at runtime
    }
}

template <typename T, typename MetadataType = TypedMetadata<T>>
inline llvm::Constant*
initializeMetadataPtr(CustomData& data,
    const std::string& name,
    llvm::LLVMContext& C)
{
    MetadataType* meta = data.getOrInsertData<MetadataType>(name);
    if (meta) return codegen::LLVMType<T>::get(C, &(meta->value()));
    return nullptr;
}

inline void
registerExternalGlobals(const codegen::SymbolTable& globals, CustomData::Ptr& dataPtr, llvm::LLVMContext& C)
{
    auto initializerFromToken =
        [&](const ast::tokens::CoreType type, const std::string& name, CustomData& data) -> llvm::Constant* {
        switch (type) {
            case ast::tokens::BOOL    : return initializeMetadataPtr<bool>(data, name, C);
            case ast::tokens::INT32   : return initializeMetadataPtr<int32_t>(data, name, C);
            case ast::tokens::INT64   : return initializeMetadataPtr<int64_t>(data, name, C);
            case ast::tokens::FLOAT   : return initializeMetadataPtr<float>(data, name, C);
            case ast::tokens::DOUBLE  : return initializeMetadataPtr<double>(data, name, C);
            case ast::tokens::VEC2I   : return initializeMetadataPtr<math::Vec2<int32_t>>(data, name, C);
            case ast::tokens::VEC2F   : return initializeMetadataPtr<math::Vec2<float>>(data, name, C);
            case ast::tokens::VEC2D   : return initializeMetadataPtr<math::Vec2<double>>(data, name, C);
            case ast::tokens::VEC3I   : return initializeMetadataPtr<math::Vec3<int32_t>>(data, name, C);
            case ast::tokens::VEC3F   : return initializeMetadataPtr<math::Vec3<float>>(data, name, C);
            case ast::tokens::VEC3D   : return initializeMetadataPtr<math::Vec3<double>>(data, name, C);
            case ast::tokens::VEC4I   : return initializeMetadataPtr<math::Vec4<int32_t>>(data, name, C);
            case ast::tokens::VEC4F   : return initializeMetadataPtr<math::Vec4<float>>(data, name, C);
            case ast::tokens::VEC4D   : return initializeMetadataPtr<math::Vec4<double>>(data, name, C);
            case ast::tokens::MAT3F   : return initializeMetadataPtr<math::Mat3<float>>(data, name, C);
            case ast::tokens::MAT3D   : return initializeMetadataPtr<math::Mat3<double>>(data, name, C);
            case ast::tokens::MAT4F   : return initializeMetadataPtr<math::Mat4<float>>(data, name, C);
            case ast::tokens::MAT4D   : return initializeMetadataPtr<math::Mat4<double>>(data, name, C);
            case ast::tokens::STRING  : return initializeMetadataPtr<ax::AXString, ax::AXStringMetadata>(data, name, C);
            case ast::tokens::UNKNOWN :
            default      : {
                // grammar guarantees this is unreachable as long as all types are supported
                OPENVDB_THROW(AXCompilerError, "Attribute type unsupported or not recognised");
            }
        }
    };

    std::string name, typestr;
    for (const auto& global : globals.map()) {

        const std::string& token = global.first;
        if (!ast::ExternalVariable::nametypeFromToken(token, &name, &typestr)) continue;

        const ast::tokens::CoreType typetoken =
            ast::tokens::tokenFromTypeString(typestr);

        // if we have any external variables, the custom data must be initialized to at least hold
        // zero values (initialized by the default metadata types)
        if (!dataPtr) dataPtr.reset(new CustomData);

        // should always be a GlobalVariable.
        assert(llvm::isa<llvm::GlobalVariable>(global.second));

        llvm::GlobalVariable* variable = llvm::cast<llvm::GlobalVariable>(global.second);
        assert(variable->getValueType() == codegen::LLVMType<uintptr_t>::get(C));

        llvm::Constant* initializer = initializerFromToken(typetoken, name, *dataPtr);

        if (!initializer) {
            OPENVDB_THROW(AXCompilerError, "Custom data \"" + name + "\" already exists with a "
                "different type.");
        }

        variable->setInitializer(initializer);
        variable->setConstant(true); // is not written to at runtime
    }
}

struct PointDefaultModifier :
    public openvdb::ax::ast::Visitor<PointDefaultModifier, /*non-const*/false>
{
    using openvdb::ax::ast::Visitor<PointDefaultModifier, false>::traverse;
    using openvdb::ax::ast::Visitor<PointDefaultModifier, false>::visit;

    PointDefaultModifier() = default;

    virtual ~PointDefaultModifier() = default;

    const std::set<std::string> autoVecAttribs {"P", "v", "N", "Cd"};

    bool visit(ast::Attribute* attrib) {
        if (!attrib->inferred()) return true;
        if (autoVecAttribs.find(attrib->name()) == autoVecAttribs.end()) return true;

        openvdb::ax::ast::Attribute::UniquePtr
            replacement(new openvdb::ax::ast::Attribute(attrib->name(), ast::tokens::VEC3F, true));
        if (!attrib->replace(replacement.get())) {
            OPENVDB_THROW(AXCompilerError,
                "Auto conversion of inferred attributes failed.");
        }
        replacement.release();

        return true;
    }
};

} // anonymous namespace

/////////////////////////////////////////////////////////////////////////////

Compiler::Compiler(const CompilerOptions& options)
    : mContext()
    , mCompilerOptions(options)
    , mFunctionRegistry()
{
    mContext.reset(new llvm::LLVMContext);
    mFunctionRegistry = codegen::createDefaultRegistry(&options.mFunctionOptions);
}

Compiler::UniquePtr Compiler::create(const CompilerOptions &options)
{
    UniquePtr compiler(new Compiler(options));
    return compiler;
}

void Compiler::setFunctionRegistry(std::unique_ptr<codegen::FunctionRegistry>&& functionRegistry)
{
    mFunctionRegistry = std::move(functionRegistry);
}

template<>
PointExecutable::Ptr
Compiler::compile<PointExecutable>(const ast::Tree& syntaxTree,
                                   Logger& logger,
                                   const CustomData::Ptr customData)
{
    openvdb::SharedPtr<ast::Tree> tree(syntaxTree.copy());
    PointDefaultModifier modifier;
    modifier.traverse(tree.get());

    verifyTypedAccesses(*tree, logger);
    // initialize the module and generate LLVM IR
    std::unique_ptr<llvm::Module> module(new llvm::Module("module", *mContext));
    std::unique_ptr<llvm::TargetMachine> TM = initializeTargetMachine();
    if (TM) {
        module->setDataLayout(TM->createDataLayout());
        module->setTargetTriple(TM->getTargetTriple().normalize());
    }

    codegen::codegen_internal::PointComputeGenerator
        codeGenerator(*module, mCompilerOptions.mFunctionOptions,
            *mFunctionRegistry, logger);
    AttributeRegistry::Ptr attributes = codeGenerator.generate(*tree);

    // if there has been a compilation error through user error, exit
    if (!attributes) {
        assert(logger.hasError());
        return nullptr;
    }

    // map accesses (always do this prior to optimising as globals may be removed)
    registerAccesses(codeGenerator.globals(), *attributes);

    CustomData::Ptr validCustomData(customData);
    registerExternalGlobals(codeGenerator.globals(), validCustomData, *mContext);

    // optimise

    llvm::Module* modulePtr = module.get();
    optimiseAndVerify(modulePtr, mCompilerOptions.mVerify, mCompilerOptions.mOptLevel, TM.get());

    // @todo re-constant fold!! although constant folding will work with constant
    //       expressions prior to optimisation, expressions like "int a = 1; cosh(a);"
    //       will still keep a call to cosh. This is because the current AX folding
    //       only checks for an immediate constant expression and for C bindings,
    //       like cosh, llvm its unable to optimise the call out (as it isn't aware
    //       of the function body). What llvm can do, however, is change this example
    //       into "cosh(1)" which we can then handle.

    // create the llvm execution engine which will build our function pointers

    std::string error;
    std::shared_ptr<llvm::ExecutionEngine>
        executionEngine(llvm::EngineBuilder(std::move(module))
            .setEngineKind(llvm::EngineKind::JIT)
            .setErrorStr(&error)
            .create());

    if (!executionEngine) {
        OPENVDB_THROW(AXCompilerError, "Failed to create LLVMExecutionEngine: " + error);
    }

    // map functions

    initializeGlobalFunctions(*mFunctionRegistry, *executionEngine, *modulePtr);

    // finalize mapping

    executionEngine->finalizeObject();

    // get the built function pointers

    const std::vector<std::string> functionNames {
        codegen::PointKernel::getDefaultName(),
        codegen::PointRangeKernel::getDefaultName()
    };

    std::unordered_map<std::string, uint64_t> functionMap;

    for (const std::string& name : functionNames) {
        const uint64_t address = executionEngine->getFunctionAddress(name);
        if (!address) {
            OPENVDB_THROW(AXCompilerError, "Failed to compile compute function \"" + name + "\"");
        }
        functionMap[name] = address;
    }

    // create final executable object
    PointExecutable::Ptr
        executable(new PointExecutable(mContext,
            executionEngine,
            attributes,
            validCustomData,
            functionMap));

    return executable;
}

template<>
VolumeExecutable::Ptr
Compiler::compile<VolumeExecutable>(const ast::Tree& syntaxTree,
                                    Logger& logger,
                                    const CustomData::Ptr customData)
{
    verifyTypedAccesses(syntaxTree, logger);

    // initialize the module and generate LLVM IR

    std::unique_ptr<llvm::Module> module(new llvm::Module("module", *mContext));
    std::unique_ptr<llvm::TargetMachine> TM = initializeTargetMachine();
    if (TM) {
        module->setDataLayout(TM->createDataLayout());
        module->setTargetTriple(TM->getTargetTriple().normalize());
    }

    codegen::codegen_internal::VolumeComputeGenerator
        codeGenerator(*module, mCompilerOptions.mFunctionOptions,
            *mFunctionRegistry, logger);
    AttributeRegistry::Ptr attributes = codeGenerator.generate(syntaxTree);

    // if there has been a compilation error through user error, exit
    if (!attributes) {
        assert(logger.hasError());
        return nullptr;
    }

    // map accesses (always do this prior to optimising as globals may be removed)
    registerAccesses(codeGenerator.globals(), *attributes);

    CustomData::Ptr validCustomData(customData);
    registerExternalGlobals(codeGenerator.globals(), validCustomData, *mContext);

    llvm::Module* modulePtr = module.get();
    optimiseAndVerify(modulePtr, mCompilerOptions.mVerify, mCompilerOptions.mOptLevel, TM.get());

    std::string error;
    std::shared_ptr<llvm::ExecutionEngine>
        executionEngine(llvm::EngineBuilder(std::move(module))
            .setEngineKind(llvm::EngineKind::JIT)
            .setErrorStr(&error)
            .create());


    if (!executionEngine) {
        OPENVDB_THROW(AXCompilerError, "Failed to create LLVMExecutionEngine: " + error);
    }
    // map functions

    initializeGlobalFunctions(*mFunctionRegistry, *executionEngine,
        *modulePtr);

    // finalize mapping

    executionEngine->finalizeObject();

    const std::string name = codegen::VolumeKernel::getDefaultName();
    const uint64_t address = executionEngine->getFunctionAddress(name);
    if (!address) {
        OPENVDB_THROW(AXCompilerError, "Failed to compile compute function \"" + name + "\"");
    }

    std::unordered_map<std::string, uint64_t> functionMap;
    functionMap[name] = address;

    // create final executable object
    VolumeExecutable::Ptr
        executable(new VolumeExecutable(mContext,
            executionEngine,
            attributes,
            validCustomData,
            functionMap));
    return executable;
}


} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

