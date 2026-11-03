// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file compiler/Compiler.cc

#include "Compiler.h"

#include "PointExecutable.h"
#include "VolumeExecutable.h"

#include "openvdb_ax/ast/Scanners.h"
#include "openvdb_ax/codegen/Functions.h"
#include "openvdb_ax/codegen/PointComputeGenerator.h"
#include "openvdb_ax/codegen/VolumeComputeGenerator.h"
#include "openvdb_ax/codegen/VolumeKernelFunctions.h"
#include "openvdb_ax/Exceptions.h"

#include <openvdb/Exceptions.h>
#include <openvdb/util/Assert.h>

#include <llvm/Passes/PassBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

namespace
{

bool verify(const llvm::Module& module, Logger& logger)
{
    std::ostringstream os;
    llvm::raw_os_ostream out(os);
    if (llvm::verifyModule(module, &out)) {
        out.flush();
        logger.error("Fatal AX Compiler error; the generated IR was invalid:\n" + os.str());
        return false;
    }
    return true;
}

inline std::unique_ptr<llvm::orc::LLJIT>
initializeExecutionEngine(Logger& logger)
{
    auto LLJIT = llvm::orc::LLJITBuilder().create();
    if (!LLJIT) {
        logger.error("Fatal AX Compiler error; the LLVM Execution Engine could "
            "not be initialized\n");
        llvm::logAllUnhandledErrors(LLJIT.takeError(), llvm::errs());
        return nullptr;
    }
    return std::move(*LLJIT);
}

void LLVMoptimise(llvm::Module& M,
    const llvm::OptimizationLevel opt,
    llvm::TargetMachine* TM)
{
    // Create the analysis managers.
    // These must be declared in this order so that they are destroyed in the
    // correct order due to inter-analysis-manager references.
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    llvm::PassBuilder PB(TM);

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt);

    // Optimize the IR!
    MPM.run(M, MAM);
}

void optimise(llvm::Module& module,
    const CompilerOptions::OptLevel optLevel,
    llvm::TargetMachine* TM)
{
    switch (optLevel) {
        case CompilerOptions::OptLevel::O0 : {
            LLVMoptimise(module, llvm::OptimizationLevel::O0, TM);
            break;
        }
        case CompilerOptions::OptLevel::O1 : {
            LLVMoptimise(module, llvm::OptimizationLevel::O1, TM);
            break;
        }
        case CompilerOptions::OptLevel::O2 : {
            LLVMoptimise(module, llvm::OptimizationLevel::O2, TM);
            break;
        }
        case CompilerOptions::OptLevel::Os : {
            LLVMoptimise(module, llvm::OptimizationLevel::Os, TM);
            break;
        }
        case CompilerOptions::OptLevel::Oz : {
            LLVMoptimise(module, llvm::OptimizationLevel::Oz, TM);
            break;
        }
        case CompilerOptions::OptLevel::O3 : {
            LLVMoptimise(module, llvm::OptimizationLevel::O3, TM);
            break;
        }
        case CompilerOptions::OptLevel::NONE :
        default             : {}
    }
}

bool initializeGlobalFunctions(const codegen::FunctionRegistry& registry,
                               llvm::orc::LLJIT& EE,
                               llvm::Module& module,
                               Logger& logger)
{
    const size_t count = logger.errors();

    llvm::orc::SymbolMap Symbols;

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
#ifdef OPENVDB_ENABLE_ASSERTS
                // some internally supported LLVm symbols (malloc, free, etc) are
                // not prefixed with ax. and we don't generated a function body
                if (llvmFunction->getName().starts_with("ax.")) {
                    OPENVDB_LOG_WARN("Function with symbol \"" << decl->symbol() << "\" has "
                        "no function body and is not a C binding.");
                }
#endif
                continue;
            }

            const uint64_t address = binding->address();
            if (address == 0) {
                logger.error("Fatal AX Compiler error; No available mapping for C Binding "
                    "with symbol \"" + std::string(decl->symbol()) + "\"");
                continue;
            }

            // Constuct a llvm::orc::ExecutorSymbolDef in the symbol map
            const auto result =
                Symbols.try_emplace(EE.mangleAndIntern(llvmFunction->getName()),
                    llvm::orc::ExecutorAddr(address),
                    llvm::JITSymbolFlags::Exported);

            if (!result.second) {
                logger.error("Fatal AX Compiler error; multiple functions are using the "
                    "same symbol \"" + std::string(decl->symbol()) + "\".");
            }
        }
    }

    auto Err = EE.getMainJITDylib().define(llvm::orc::absoluteSymbols(Symbols));
    if (Err) {
        logger.error("Fatal AX Compiler error; the LLVM Execution Engine could "
            "not be initialized\n");
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs());
        return false;
    }

#ifdef OPENVDB_ENABLE_ASSERTS
    // Loop through all functions and check to see if they have valid engine mappings.
    // This can occur if lazy functions don't initialize their dependencies properly.
    // @todo  Really we should just loop through the module functions to begin with
    //  to init engine mappings - it would probably be faster but we'd have to do
    //  some string manipulation and it would assume function names have been set up
    //  correctly
    const auto& list = module.getFunctionList();
    for (const auto& F : list) {
        if (F.size() > 0) continue;
        // Some LLVM functions may also not be defined at this stage which is expected
        if (!F.getName().starts_with("ax.")) continue;
        auto EntrySym = EE.lookup(F.getName());
        OPENVDB_ASSERT(bool(EntrySym));
        OPENVDB_ASSERT_MESSAGE(EntrySym->getValue() != 0, "Unbound function!");
    }
#endif

    return count == logger.errors();
}

bool verifyTypedAccesses(const ast::Tree& tree, openvdb::ax::Logger& logger)
{
    // verify the attributes and external variables requested in the syntax tree
    // only have a single type. Note that the executer will also throw a runtime
    // error if the same attribute is accessed with different types, but as that's
    // currently not a valid state on a PointDataGrid, error in compilation as well
    // @todo - introduce a framework for supporting custom preprocessors

    const size_t errs = logger.errors();

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

    return logger.errors() == errs;
}

inline void
registerAccesses(const codegen::SymbolTable<llvm::Value*>& globals, const AttributeRegistry& registry)
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
        OPENVDB_ASSERT(llvm::isa<llvm::GlobalVariable>(global.second));

        // Assign the attribute index global a valid index.
        // @note executionEngine->addGlobalMapping() can also be used if the indices
        // ever need to vary positions without having to force a recompile (previously
        // was used unnecessarily)

        llvm::GlobalVariable* variable =
            llvm::cast<llvm::GlobalVariable>(global.second);
        OPENVDB_ASSERT(variable->getValueType()->isIntegerTy(64));

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

inline bool
registerExternalGlobals(const codegen::SymbolTable<llvm::Value*>& globals,
            CustomData::Ptr& dataPtr,
            llvm::LLVMContext& C,
            Logger& logger)
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
            // @note could be const char*, but not all functions have support for const char* args
            case ast::tokens::STRING  : return initializeMetadataPtr<ax::codegen::String>(data, name, C);
            case ast::tokens::UNKNOWN :
            default      : {
                // grammar guarantees this is unreachable as long as all types are supported
                OPENVDB_ASSERT(false && "Attribute type unsupported or not recognised");
                return nullptr;
            }
        }
    };

    bool success = true;
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
        OPENVDB_ASSERT(llvm::isa<llvm::GlobalVariable>(global.second));

        llvm::GlobalVariable* variable = llvm::cast<llvm::GlobalVariable>(global.second);
        OPENVDB_ASSERT(variable->getValueType() == codegen::LLVMType<uintptr_t>::get(C));

        llvm::Constant* initializer = initializerFromToken(typetoken, name, *dataPtr);

        if (!initializer) {
            logger.error("Custom data \"" + name + "\" already exists with a different type.");
            success = false;
            continue;
        }

        variable->setInitializer(initializer);
        variable->setConstant(true); // is not written to at runtime
    }

    return success;
}

struct PointDefaultModifier :
    public openvdb::ax::ast::Visitor<PointDefaultModifier, /*non-const*/false>
{
    using openvdb::ax::ast::Visitor<PointDefaultModifier, false>::traverse;
    using openvdb::ax::ast::Visitor<PointDefaultModifier, false>::visit;

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
    : mCompilerOptions(options)
    , mFunctionRegistry()
{
    mFunctionRegistry = codegen::createDefaultRegistry(&options.mFunctionOptions);
}

Compiler::~Compiler() = default;
Compiler::Compiler(Compiler&&) = default;
Compiler& Compiler::operator=(Compiler&&) = default;

Compiler::UniquePtr Compiler::create(const CompilerOptions &options)
{
    UniquePtr compiler(new Compiler(options));
    return compiler;
}

void Compiler::setFunctionRegistry(std::unique_ptr<codegen::FunctionRegistry>&& functionRegistry)
{
    mFunctionRegistry = std::move(functionRegistry);
}


template <typename ExeT, typename GenT>
inline typename ExeT::Ptr
Compiler::compile(const ast::Tree& tree,
        const std::string& moduleName,
        const std::vector<std::string>& functions,
        CustomData::Ptr data,
        Logger& logger)
{
    // @todo  Not technically necessary for volumes but does the
    //   executer/bindings handle this?
    if (!verifyTypedAccesses(tree, logger)) {
        return nullptr;
    }

    auto C = std::make_unique<llvm::LLVMContext>();
    auto M = std::make_unique<llvm::Module>(moduleName, *C);

    GenT codeGenerator(*M, mCompilerOptions.mFunctionOptions, *mFunctionRegistry, logger);
    AttributeRegistry::Ptr attributes = codeGenerator.generate(tree);

    // if there has been a compilation error through user error, exit
    if (!attributes) {
        OPENVDB_ASSERT(logger.hasError());
        return nullptr;
    }

    // map accesses (always do this prior to optimising as globals may be removed)

    registerAccesses(codeGenerator.globals(), *attributes);

    if (!registerExternalGlobals(codeGenerator.globals(), data, *C, logger)) {
        return nullptr;
    }

    // create the target machine for optimization passes in the new PM.
    // The LLJIT engine creates this in the same way (there doesn't seem to be
    // an easy way to retrieve the TM from the EE - could instead pass out JITTM
    // to the EE to be clearer that these are expected to match).

    std::unique_ptr<llvm::TargetMachine> TM{nullptr};
    if (auto JITTM = llvm::orc::JITTargetMachineBuilder::detectHost()) {
        llvm::Expected<std::unique_ptr<llvm::TargetMachine>> result = JITTM->createTargetMachine();
        if (result) TM = std::move(*result);
    }
    if (!TM) {
        logger.warning("Unable to determine CPU host features");
    }

    // optimise and verify

    if (mCompilerOptions.mVerify && !verify(*M, logger)) return nullptr;
    optimise(*M, mCompilerOptions.mOptLevel, TM.get());
    if (mCompilerOptions.mOptLevel != CompilerOptions::OptLevel::NONE) {
        if (mCompilerOptions.mVerify && !verify(*M, logger)) return nullptr;
    }

    // @todo re-constant fold!! although constant folding will work with constant
    //       expressions prior to optimisation, expressions like "int a = 1; cosh(a);"
    //       will still keep a call to cosh. This is because the current AX folding
    //       only checks for an immediate constant expression and for C bindings,
    //       like cosh, llvm its unable to optimise the call out (as it isn't aware
    //       of the function body). What llvm can do, however, is change this example
    //       into "cosh(1)" which we can then handle.

    // map functions

    std::shared_ptr<llvm::orc::LLJIT> EE = initializeExecutionEngine(logger);
    if (!EE) return nullptr;


    if (!initializeGlobalFunctions(*mFunctionRegistry, *EE, *M, logger)) {
        return nullptr;
    }

    llvm::orc::ThreadSafeModule module(std::move(M), std::move(C));
    if (auto Err = EE->addIRModule(std::move(module))) {
        logger.error("Fatal AX Compiler error; the LLVM Execution Engine could "
            "not be initialized\n");
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs());
        return nullptr;
    }

    // get the built function pointers

    std::unordered_map<std::string, uint64_t> functionMap;

    for (const std::string& name : functions) {
        auto EntrySym = EE->lookup(name);
        if (!EntrySym) {
            logger.error("Fatal AX Compiler error; Unable to compile compute "
                "function \"" + name + "\"");
            return nullptr;
        }
        functionMap[name] = EntrySym->getValue();
    }

    // create final executable object
    // @note relies on friend access so can't use make_shared
    return typename ExeT::Ptr(new ExeT(
        std::move(EE),
        std::move(attributes),
        std::move(data),
        std::move(functionMap),
        tree));
}

template<>
OPENVDB_AX_API PointExecutable::Ptr
Compiler::compile<PointExecutable>(const ast::Tree& syntaxTree,
                                   Logger& logger,
                                   const CustomData::Ptr customData)
{
    using GenT = codegen::codegen_internal::PointComputeGenerator;

    openvdb::SharedPtr<ast::Tree> tree(syntaxTree.copy());
    PointDefaultModifier modifier;
    modifier.traverse(tree.get());

    const std::vector<std::string> functionNames {
        codegen::PointKernelBufferRange::getDefaultName(),
        codegen::PointKernelAttributeArray::getDefaultName()
    };

    return this->compile<PointExecutable, GenT>(*tree, "ax.point.module",
        functionNames, customData, logger);
}

template<>
OPENVDB_AX_API VolumeExecutable::Ptr
Compiler::compile<VolumeExecutable>(const ast::Tree& syntaxTree,
                                    Logger& logger,
                                    const CustomData::Ptr customData)
{
    using GenT = codegen::codegen_internal::VolumeComputeGenerator;

    const std::vector<std::string> functionNames {
        // codegen::VolumeKernelValue::getDefaultName(), // currently unused directly
        codegen::VolumeKernelBuffer::getDefaultName(),
        codegen::VolumeKernelNode::getDefaultName()
    };

    return this->compile<VolumeExecutable, GenT>(syntaxTree, "ax.volume.module",
        functionNames, customData, logger);
}


} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

