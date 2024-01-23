// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/PointFunctions.cc
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  Contains the function objects that define the functions used in
///   point compute function generation, to be inserted into the
///   FunctionRegistry. These define the functions available when operating
///   on points. Also includes the definitions for the point attribute
///   retrieval and setting.
///

#include "Functions.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"
#include "PointLeafLocalData.h"

#include "openvdb_ax/ast/Tokens.h"
#include "openvdb_ax/compiler/CompilerOptions.h"
#include "openvdb_ax/Exceptions.h"

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/util/Assert.h>

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

namespace
{

#define OPENVDB_AX_CHECK_MODULE_CONTEXT(B) \
    { \
        const llvm::Function* F = B.GetInsertBlock()->getParent(); \
        const llvm::Module* M = F ? F->getParent() : nullptr; \
        if (!M || M->getName() != "ax.point.module") { \
            OPENVDB_THROW(AXCompilerError, "Function \"" << (F ? F->getName().str() : "unknown") << \
                "\" cannot be called for the current target:\"" << (M ? M->getName().str() : "unknown") << \
                "\". This function only runs on OpenVDB Point Grids."); \
        } \
    }

/// @brief  Retrieve a group handle from an expected vector of handles using the offset
///         pointed to by the engine data. Note that HandleT should only ever be a GroupHandle
///         or GroupWriteHandle object
template <typename HandleT>
inline HandleT*
groupHandle(const std::string& name, void** groupHandles, const void* const data)
{
    const openvdb::points::AttributeSet* const attributeSet =
        static_cast<const openvdb::points::AttributeSet*>(data);

    const size_t groupIdx = attributeSet->groupOffset(name);
    if (groupIdx == openvdb::points::AttributeSet::INVALID_POS) return nullptr;

    return static_cast<HandleT*>(groupHandles[groupIdx]);
}

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

inline FunctionGroup::UniquePtr ax_ingroup(const FunctionOptions& op)
{
    static auto ingroup =
        [](const codegen::String* const name,
           const uint64_t index,
           void** groupHandles,
           const void* const leafDataPtr,
           const void* const data) -> bool
    {
        OPENVDB_ASSERT(name);
        OPENVDB_ASSERT(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        if (name->size() == 0) return false;
        if (!groupHandles) return false;

        const std::string nameStr = name->str();
        const openvdb::points::GroupHandle* handle =
            groupHandle<openvdb::points::GroupHandle>(nameStr, groupHandles, data);
        if (handle) return handle->get(static_cast<openvdb::Index>(index));

        // If the handle doesn't exist, check to see if any new groups have
        // been added
        const codegen_internal::PointLeafLocalData* const leafData =
            static_cast<const codegen_internal::PointLeafLocalData*>(leafDataPtr);
        handle = leafData->get(nameStr);
        return handle ? handle->get(static_cast<openvdb::Index>(index)) : false;
    };

    using InGroup = bool(const codegen::String* const,
       const uint64_t,
       void**,
       const void* const,
       const void* const);

    return FunctionBuilder("_ingroup")
        .addSignature<InGroup>(ingroup)
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(2, llvm::Attribute::ReadOnly)
        .addParameterAttribute(2, llvm::Attribute::NoAlias)
        .addParameterAttribute(3, llvm::Attribute::ReadOnly)
        .addParameterAttribute(3, llvm::Attribute::NoAlias)
        .addParameterAttribute(4, llvm::Attribute::ReadOnly)
        .addParameterAttribute(4, llvm::Attribute::NoAlias)
        .addFunctionAttribute(llvm::Attribute::ReadOnly)
        .addFunctionAttribute(llvm::Attribute::NoRecurse)
        // @note  handle->get can throw, so no unwind. Maybe use getUnsafe?
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for querying point group data")
        .get();
}

inline FunctionGroup::UniquePtr axingroup(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>& args,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        llvm::Value* point_index = extractArgument(compute, "point_index");
        llvm::Value* group_handles = extractArgument(compute, "group_handles");
        llvm::Value* leaf_data = extractArgument(compute, "leaf_data");
        llvm::Value* attribute_set = extractArgument(compute, "attribute_set");
        OPENVDB_ASSERT(point_index);
        OPENVDB_ASSERT(group_handles);
        OPENVDB_ASSERT(leaf_data);
        OPENVDB_ASSERT(attribute_set);

        std::vector<llvm::Value*> input(args);
        input.emplace_back(point_index);
        input.emplace_back(group_handles);
        input.emplace_back(leaf_data);
        input.emplace_back(attribute_set);
        return ax_ingroup(op)->execute(input, B);
    };

    return FunctionBuilder("ingroup")
        .addSignature<bool(const codegen::String* const)>(generate)
        .addDependency("_ingroup")
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation( "Return whether or not the current point is "
            "a member of the given group name. This returns false if the group does "
            "not exist.")
        .get();
}

inline FunctionGroup::UniquePtr axeditgroup(const FunctionOptions& op)
{
    static auto editgroup =
        [](const codegen::String* const name,
           const uint64_t index,
           void** groupHandles,
           void* const leafDataPtr,
           const void* const data,
           const bool flag)
    {
        OPENVDB_ASSERT(name);
        if (name->size() == 0) return;

        // Get the group handle out of the pre-existing container of handles if they
        // exist

        const std::string nameStr = name->str();
        openvdb::points::GroupWriteHandle* handle = nullptr;
        if (groupHandles) {
            handle = groupHandle<openvdb::points::GroupWriteHandle>(nameStr, groupHandles, data);
        }

        if (!handle) {
            codegen_internal::PointLeafLocalData* const leafData =
                static_cast<codegen_internal::PointLeafLocalData*>(leafDataPtr);

            // If we are setting membership and the handle doesn't exist, create in in
            // the set of new data thats being added
            if (!flag && !leafData->hasGroup(nameStr)) return;
            handle = leafData->getOrInsert(nameStr);
            OPENVDB_ASSERT(handle);
        }

        // set the group membership
        handle->set(static_cast<openvdb::Index>(index), flag);
    };

    static auto editgroupcstar =
        [](const char* const name,
           const uint64_t index,
           void** groupHandles,
           void* const leafDataPtr,
           const void* const data,
           const bool flag)
    {
        const codegen::String str(name);
        editgroup(&str, index, groupHandles, leafDataPtr, data, flag);
    };

    using EditGroup = void(const codegen::String* const,
           const uint64_t,
           void**,
           void* const,
           const void* const,
           const bool);

    using EditGroupCstar = void(const char* const,
           const uint64_t,
           void**,
           void* const,
           const void* const,
           const bool);

    return FunctionBuilder("editgroup")
        .addSignature<EditGroup>(editgroup)
        .addSignature<EditGroupCstar>(editgroupcstar)
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(2, llvm::Attribute::ReadOnly)
        .addParameterAttribute(3, llvm::Attribute::ReadOnly)
        .addParameterAttribute(4, llvm::Attribute::ReadOnly)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for setting point group data")
        .get();
}

inline FunctionGroup::UniquePtr axaddtogroup(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>& args,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        llvm::Value* point_index = extractArgument(compute, "point_index");
        llvm::Value* group_handles = extractArgument(compute, "group_handles");
        llvm::Value* leaf_data = extractArgument(compute, "leaf_data");
        llvm::Value* attribute_set = extractArgument(compute, "attribute_set");
        OPENVDB_ASSERT(point_index);
        OPENVDB_ASSERT(group_handles);
        OPENVDB_ASSERT(leaf_data);
        OPENVDB_ASSERT(attribute_set);

        std::vector<llvm::Value*> input(args);
        input.emplace_back(point_index);
        input.emplace_back(group_handles);
        input.emplace_back(leaf_data);
        input.emplace_back(attribute_set);
        input.emplace_back(llvm::ConstantInt::get(LLVMType<bool>::get(B.getContext()), true));
        return axeditgroup(op)->execute(input, B);
    };

    return FunctionBuilder("addtogroup")
        .addSignature<void(const codegen::String* const)>(generate)
        .addSignature<void(const char* const)>(generate) // to support axdeletepoint() @todo fix
        .addDependency("editgroup")
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Add the current point to the given group "
            "name, effectively setting its membership to true. If the group does not "
            "exist, it is implicitly created. This function has no effect if the point "
            "already belongs to the given group.")
        .get();
}

inline FunctionGroup::UniquePtr axremovefromgroup(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>& args,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        llvm::Value* point_index = extractArgument(compute, "point_index");
        llvm::Value* group_handles = extractArgument(compute, "group_handles");
        llvm::Value* leaf_data = extractArgument(compute, "leaf_data");
        llvm::Value* attribute_set = extractArgument(compute, "attribute_set");
        OPENVDB_ASSERT(point_index);
        OPENVDB_ASSERT(group_handles);
        OPENVDB_ASSERT(leaf_data);
        OPENVDB_ASSERT(attribute_set);

        std::vector<llvm::Value*> input(args);
        input.emplace_back(point_index);
        input.emplace_back(group_handles);
        input.emplace_back(leaf_data);
        input.emplace_back(attribute_set);
        input.emplace_back(llvm::ConstantInt::get(LLVMType<bool>::get(B.getContext()), false));
        return axeditgroup(op)->execute(input, B);
    };

    return FunctionBuilder("removefromgroup")
        .addSignature<void(const codegen::String* const)>(generate)
        .addDependency("editgroup")
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Remove the current point from the "
            "given group name, effectively setting its membership to false. This "
            "function has no effect if the group does not exist.")
        .get();
}

inline FunctionGroup::UniquePtr axdeletepoint(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>&,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // args guaranteed to be empty
        const std::string deadGroup = "__ax_dead";
        llvm::Constant* loc = llvm::cast<llvm::Constant>(B.CreateGlobalStringPtr(deadGroup.c_str())); // char*
        return axaddtogroup(op)->execute({loc}, B);
    };

    return FunctionBuilder("deletepoint")
        .addSignature<void()>(generate)
        .addDependency("addtogroup")
        .addFunctionAttribute(llvm::Attribute::AlwaysInline)
        .setEmbedIR(true) // axaddtogroup needs access to parent function arguments
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Delete the current point from the point set. Note that this does not "
            "stop AX execution - any additional AX commands will be executed on the "
            "point and it will remain accessible until the end of execution.")
        .get();
}

inline FunctionGroup::UniquePtr axsetattribute(const FunctionOptions& op)
{
    static auto setattribptr =
        [](void* attributeHandle, uint64_t index, const auto value)
    {
        using ValueType = typename std::remove_const
            <typename std::remove_pointer
                <decltype(value)>::type>::type;
        using AttributeHandleType = openvdb::points::AttributeWriteHandle<ValueType>;

        OPENVDB_ASSERT(attributeHandle);
        OPENVDB_ASSERT(value);
        OPENVDB_ASSERT(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        AttributeHandleType* handle = static_cast<AttributeHandleType*>(attributeHandle);
        handle->set(static_cast<openvdb::Index>(index), *value);
    };

    static auto setattribstr =
        [](void* attributeHandle,
           const uint64_t index,
           const codegen::String* value,
           void* const leafDataPtr)
    {
        using AttributeHandleType = openvdb::points::StringAttributeWriteHandle;

        OPENVDB_ASSERT(attributeHandle);
        OPENVDB_ASSERT(value);
        OPENVDB_ASSERT(leafDataPtr);
        OPENVDB_ASSERT(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        const std::string s = value->str();
        AttributeHandleType* const handle =
            static_cast<AttributeHandleType*>(attributeHandle);
        codegen_internal::PointLeafLocalData* const leafData =
            static_cast<codegen_internal::PointLeafLocalData*>(leafDataPtr);

        // Check to see if the string exists in the metadata cache. If so, set the string and
        // remove any new data associated with it, otherwise set the new data

        if (handle->contains(s)) {
            handle->set(static_cast<openvdb::Index>(index), s);
            leafData->removeNewStringData(&(handle->array()), index);
        }
        else {
            leafData->setNewStringData(&(handle->array()), index, s);
        }
    };

    static auto setattrib =
        [](void* attributeHandle,
           uint64_t index,
           const auto value) {
        setattribptr(attributeHandle, index, &value);
    };

    using SetAttribD = void(void*, uint64_t, const double);
    using SetAttribF = void(void*, uint64_t, const float);
    using SetAttribI64 = void(void*, uint64_t, const int64_t);
    using SetAttribI32 = void(void*, uint64_t, const int32_t);
    using SetAttribI16 = void(void*, uint64_t, const int16_t);
    using SetAttribB = void(void*, uint64_t, const bool);
    using SetAttribV2D = void(void*, uint64_t, const openvdb::math::Vec2<double>*);
    using SetAttribV2F = void(void*, uint64_t, const openvdb::math::Vec2<float>*);
    using SetAttribV2I = void(void*, uint64_t, const openvdb::math::Vec2<int32_t>*);
    using SetAttribV3D = void(void*, uint64_t, const openvdb::math::Vec3<double>*);
    using SetAttribV3F = void(void*, uint64_t, const openvdb::math::Vec3<float>*);
    using SetAttribV3I = void(void*, uint64_t, const openvdb::math::Vec3<int32_t>*);
    using SetAttribV4D = void(void*, uint64_t, const openvdb::math::Vec4<double>*);
    using SetAttribV4F = void(void*, uint64_t, const openvdb::math::Vec4<float>*);
    using SetAttribV4I = void(void*, uint64_t, const openvdb::math::Vec4<int32_t>*);
    using SetAttribM3D = void(void*, uint64_t, const openvdb::math::Mat3<double>*);
    using SetAttribM3F = void(void*, uint64_t, const openvdb::math::Mat3<float>*);
    using SetAttribM4D = void(void*, uint64_t, const openvdb::math::Mat4<double>*);
    using SetAttribM4F = void(void*, uint64_t, const openvdb::math::Mat4<float>*);
    using SetAttribStr = void(void*, uint64_t, const codegen::String*, void* const);

    return FunctionBuilder("setattribute")
        .addSignature<SetAttribD>((SetAttribD*)(setattrib))
        .addSignature<SetAttribF>((SetAttribF*)(setattrib))
        .addSignature<SetAttribI64>((SetAttribI64*)(setattrib))
        .addSignature<SetAttribI32>((SetAttribI32*)(setattrib))
        .addSignature<SetAttribI16>((SetAttribI16*)(setattrib))
        .addSignature<SetAttribB>((SetAttribB*)(setattrib))
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .addSignature<SetAttribV2D>((SetAttribV2D*)(setattribptr))
        .addSignature<SetAttribV2F>((SetAttribV2F*)(setattribptr))
        .addSignature<SetAttribV2I>((SetAttribV2I*)(setattribptr))
        .addSignature<SetAttribV3D>((SetAttribV3D*)(setattribptr))
        .addSignature<SetAttribV3F>((SetAttribV3F*)(setattribptr))
        .addSignature<SetAttribV3I>((SetAttribV3I*)(setattribptr))
        .addSignature<SetAttribV4D>((SetAttribV4D*)(setattribptr))
        .addSignature<SetAttribV4F>((SetAttribV4F*)(setattribptr))
        .addSignature<SetAttribV4I>((SetAttribV4I*)(setattribptr))
        .addSignature<SetAttribM3D>((SetAttribM3D*)(setattribptr))
        .addSignature<SetAttribM3F>((SetAttribM3F*)(setattribptr))
        .addSignature<SetAttribM4D>((SetAttribM4D*)(setattribptr))
        .addSignature<SetAttribM4F>((SetAttribM4F*)(setattribptr))
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addParameterAttribute(2, llvm::Attribute::ReadOnly)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .addSignature<SetAttribStr>((SetAttribStr*)(setattribstr))
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addParameterAttribute(2, llvm::Attribute::ReadOnly)
            .addParameterAttribute(3, llvm::Attribute::ReadOnly)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for setting the value of a point attribute.")
        .get();
}

inline FunctionGroup::UniquePtr axgetattribute(const FunctionOptions& op)
{
    static auto getattrib =
        [](void* attributeHandle, uint64_t index, auto value)
    {
        using ValueType = typename std::remove_const
            <typename std::remove_pointer
                <decltype(value)>::type>::type;
        // typedef is a read handle. As write handles are derived types this
        // is okay and lets us define the handle types outside IR for attributes that are
        // only being read!
        using AttributeHandleType = openvdb::points::AttributeHandle<ValueType>;

        OPENVDB_ASSERT(value);
        OPENVDB_ASSERT(attributeHandle);
        OPENVDB_ASSERT(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        AttributeHandleType* handle = static_cast<AttributeHandleType*>(attributeHandle);
        (*value) = handle->get(static_cast<openvdb::Index>(index));
    };

    static auto getattribstr =
        [](void* attributeHandle,
           uint64_t index,
           codegen::String* value,
           const void* const leafDataPtr)
    {
        using AttributeHandleType = openvdb::points::StringAttributeHandle;

        OPENVDB_ASSERT(value);
        OPENVDB_ASSERT(attributeHandle);
        OPENVDB_ASSERT(leafDataPtr);
        OPENVDB_ASSERT(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        AttributeHandleType* const handle =
            static_cast<AttributeHandleType*>(attributeHandle);
        const codegen_internal::PointLeafLocalData* const leafData =
            static_cast<const codegen_internal::PointLeafLocalData*>(leafDataPtr);

        std::string data;
        if (!leafData->getNewStringData(&(handle->array()), index, data)) {
            handle->get(data, static_cast<openvdb::Index>(index));
        }

        *value = data;
    };

    using GetAttribD = void(void*, uint64_t, double*);
    using GetAttribF = void(void*, uint64_t, float*);
    using GetAttribI64 = void(void*, uint64_t, int64_t*);
    using GetAttribI32 = void(void*, uint64_t, int32_t*);
    using GetAttribI16 = void(void*, uint64_t, int16_t*);
    using GetAttribB = void(void*, uint64_t, bool*);
    using GetAttribV2D = void(void*, uint64_t, openvdb::math::Vec2<double>*);
    using GetAttribV2F = void(void*, uint64_t, openvdb::math::Vec2<float>*);
    using GetAttribV2I = void(void*, uint64_t, openvdb::math::Vec2<int32_t>*);
    using GetAttribV3D = void(void*, uint64_t, openvdb::math::Vec3<double>*);
    using GetAttribV3F = void(void*, uint64_t, openvdb::math::Vec3<float>*);
    using GetAttribV3I = void(void*, uint64_t, openvdb::math::Vec3<int32_t>*);
    using GetAttribV4D = void(void*, uint64_t, openvdb::math::Vec4<double>*);
    using GetAttribV4F = void(void*, uint64_t, openvdb::math::Vec4<float>*);
    using GetAttribV4I = void(void*, uint64_t, openvdb::math::Vec4<int32_t>*);
    using GetAttribM3D = void(void*, uint64_t, openvdb::math::Mat3<double>*);
    using GetAttribM3F = void(void*, uint64_t, openvdb::math::Mat3<float>*);
    using GetAttribM4D = void(void*, uint64_t, openvdb::math::Mat4<double>*);
    using GetAttribM4F = void(void*, uint64_t, openvdb::math::Mat4<float>*);
    using GetAttribStr = void(void*, uint64_t, codegen::String*, const void* const);

    return FunctionBuilder("getattribute")
        .addSignature<GetAttribD>((GetAttribD*)(getattrib))
        .addSignature<GetAttribF>((GetAttribF*)(getattrib))
        .addSignature<GetAttribI64>((GetAttribI64*)(getattrib))
        .addSignature<GetAttribI32>((GetAttribI32*)(getattrib))
        .addSignature<GetAttribI16>((GetAttribI16*)(getattrib))
        .addSignature<GetAttribB>((GetAttribB*)(getattrib))
        .addSignature<GetAttribV2D>((GetAttribV2D*)(getattrib))
        .addSignature<GetAttribV2F>((GetAttribV2F*)(getattrib))
        .addSignature<GetAttribV2I>((GetAttribV2I*)(getattrib))
        .addSignature<GetAttribV3D>((GetAttribV3D*)(getattrib))
        .addSignature<GetAttribV3F>((GetAttribV3F*)(getattrib))
        .addSignature<GetAttribV3I>((GetAttribV3I*)(getattrib))
        .addSignature<GetAttribV4D>((GetAttribV4D*)(getattrib))
        .addSignature<GetAttribV4F>((GetAttribV4F*)(getattrib))
        .addSignature<GetAttribV4I>((GetAttribV4I*)(getattrib))
        .addSignature<GetAttribM3D>((GetAttribM3D*)(getattrib))
        .addSignature<GetAttribM3F>((GetAttribM3F*)(getattrib))
        .addSignature<GetAttribM4D>((GetAttribM4D*)(getattrib))
        .addSignature<GetAttribM4F>((GetAttribM4F*)(getattrib))
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .addSignature<GetAttribStr>((GetAttribStr*)(getattribstr))
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addParameterAttribute(3, llvm::Attribute::ReadOnly)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for getting the value of a point attribute.")
        .get();
}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


void insertVDBPointFunctions(FunctionRegistry& registry,
    const FunctionOptions* options)
{
    const bool create = options && !options->mLazyFunctions;
    auto add = [&](const std::string& name,
        const FunctionRegistry::ConstructorT creator,
        const bool internal = false)
    {
        if (create) registry.insertAndCreate(name, creator, *options, internal);
        else        registry.insert(name, creator, internal);
    };

    // point functions

    add("addtogroup", axaddtogroup);
    add("ingroup", axingroup);
    add("removefromgroup",axremovefromgroup);
    add("deletepoint", axdeletepoint);
    add("_ingroup", ax_ingroup, true);
    add("editgroup", axeditgroup, true);
    add("getattribute", axgetattribute, true);
    add("setattribute", axsetattribute, true);
}

} // namespace codegen
} // namespace ax
} // namespace openvdb_version
} // namespace openvdb


