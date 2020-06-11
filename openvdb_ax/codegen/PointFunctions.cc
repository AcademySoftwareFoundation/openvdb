///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

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

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>

#include <openvdb_ax/ast/Tokens.h>
#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/compiler/LeafLocalData.h>
#include <openvdb_ax/Exceptions.h>
#include <openvdb_ax/version.h>

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

namespace
{

/// @brief  Retrieve a group handle from an expected vector of handles using the offset
///         pointed to by the engine data. Note that HandleT should only ever be a GroupHandle
///         or GroupWriteHandle object
template <typename HandleT>
inline HandleT*
groupHandle(const std::string& name, void** groupHandles, const void* const data)
{
    const openvdb::points::AttributeSet* const attributeSet =
        static_cast<const openvdb::points::AttributeSet* const>(data);

    const size_t groupIdx = attributeSet->groupOffset(name);
    if (groupIdx == openvdb::points::AttributeSet::INVALID_POS) return nullptr;

    return static_cast<HandleT*>(groupHandles[groupIdx]);
}

}

inline FunctionGroup::Ptr ax_ingroup(const FunctionOptions& op)
{
    static auto ingroup =
        [](const AXString* const name,
           const uint64_t index,
           void** groupHandles,
           const void* const leafDataPtr,
           const void* const data) -> bool
    {
        assert(name);
        assert(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        if (name->size == 0) return false;
        if (!groupHandles) return false;

        const std::string nameStr(name->ptr, name->size);
        const openvdb::points::GroupHandle* handle =
            groupHandle<openvdb::points::GroupHandle>(nameStr, groupHandles, data);
        if (handle) return handle->get(static_cast<openvdb::Index>(index));

        // If the handle doesn't exist, check to see if any new groups have
        // been added
        const openvdb::ax::compiler::LeafLocalData* const leafData =
            static_cast<const openvdb::ax::compiler::LeafLocalData* const>(leafDataPtr);
        handle = leafData->get(nameStr);
        return handle ? handle->get(static_cast<openvdb::Index>(index)) : false;
    };

    using InGroup = bool(const AXString* const,
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

inline FunctionGroup::Ptr axingroup(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>& args,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        assert(compute);
        assert(compute->getName() == "ax.compute.point");
        llvm::Value* point_index = extractArgument(compute, "point_index");
        llvm::Value* group_handles = extractArgument(compute, "group_handles");
        llvm::Value* leaf_data = extractArgument(compute, "leaf_data");
        llvm::Value* attribute_set = extractArgument(compute, "attribute_set");
        assert(point_index);
        assert(group_handles);
        assert(leaf_data);
        assert(attribute_set);

        std::vector<llvm::Value*> input(args);
        input.emplace_back(point_index);
        input.emplace_back(group_handles);
        input.emplace_back(leaf_data);
        input.emplace_back(attribute_set);
        return ax_ingroup(op)->execute(input, B);
    };

    return FunctionBuilder("ingroup")
        .addSignature<bool(const AXString* const)>(generate)
        .addDependency("_ingroup")
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation( "Return whether or not the current point is "
            "a member of the given group name. This returns false if the group does "
            "not exist.")
        .get();
}

inline FunctionGroup::Ptr axeditgroup(const FunctionOptions& op)
{
    static auto editgroup =
        [](const AXString* const name,
           const uint64_t index,
           void** groupHandles,
           void* const leafDataPtr,
           const void* const data,
           const bool flag)
    {
        assert(name);
        if (name->size == 0) return;

        // Get the group handle out of the pre-existing container of handles if they
        // exist

        const std::string nameStr(name->ptr, name->size);
        openvdb::points::GroupWriteHandle* handle = nullptr;
        if (groupHandles) {
            handle = groupHandle<openvdb::points::GroupWriteHandle>(nameStr, groupHandles, data);
        }

        if (!handle) {
            openvdb::ax::compiler::LeafLocalData* const leafData =
                static_cast<openvdb::ax::compiler::LeafLocalData* const>(leafDataPtr);

            // If we are setting membership and the handle doesnt exist, create in in
            // the set of new data thats being added
            if (!flag && !leafData->hasGroup(nameStr)) return;
            handle = leafData->getOrInsert(nameStr);
            assert(handle);
        }

        // set the group membership
        handle->set(static_cast<openvdb::Index>(index), flag);
    };

    using EditGroup = void(const AXString* const,
           const uint64_t,
           void**,
           void* const,
           const void* const,
           const bool);

    return FunctionBuilder("editgroup")
        .addSignature<EditGroup>(editgroup)
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(2, llvm::Attribute::ReadOnly)
        .addParameterAttribute(3, llvm::Attribute::ReadOnly)
        .addParameterAttribute(4, llvm::Attribute::ReadOnly)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for setting point group data")
        .get();
}

inline FunctionGroup::Ptr axaddtogroup(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>& args,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        assert(compute);
        assert(compute->getName() == "ax.compute.point");
        llvm::Value* point_index = extractArgument(compute, "point_index");
        llvm::Value* group_handles = extractArgument(compute, "group_handles");
        llvm::Value* leaf_data = extractArgument(compute, "leaf_data");
        llvm::Value* attribute_set = extractArgument(compute, "attribute_set");
        assert(point_index);
        assert(group_handles);
        assert(leaf_data);
        assert(attribute_set);

        std::vector<llvm::Value*> input(args);
        input.emplace_back(point_index);
        input.emplace_back(group_handles);
        input.emplace_back(leaf_data);
        input.emplace_back(attribute_set);
        input.emplace_back(llvm::ConstantInt::get(LLVMType<bool>::get(B.getContext()), true));
        return axeditgroup(op)->execute(input, B);
    };

    return FunctionBuilder("addtogroup")
        .addSignature<void(const AXString* const)>(generate)
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

inline FunctionGroup::Ptr axremovefromgroup(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>& args,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        assert(compute);
        assert(compute->getName() == "ax.compute.point");
        llvm::Value* point_index = extractArgument(compute, "point_index");
        llvm::Value* group_handles = extractArgument(compute, "group_handles");
        llvm::Value* leaf_data = extractArgument(compute, "leaf_data");
        llvm::Value* attribute_set = extractArgument(compute, "attribute_set");
        assert(point_index);
        assert(group_handles);
        assert(leaf_data);
        assert(attribute_set);

        std::vector<llvm::Value*> input(args);
        input.emplace_back(point_index);
        input.emplace_back(group_handles);
        input.emplace_back(leaf_data);
        input.emplace_back(attribute_set);
        input.emplace_back(llvm::ConstantInt::get(LLVMType<bool>::get(B.getContext()), false));
        return axeditgroup(op)->execute(input, B);
    };

    return FunctionBuilder("removefromgroup")
        .addSignature<void(const AXString* const)>(generate)
        .addDependency("editgroup")
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Remove the current point from the "
            "given group name, effectively setting its membership to false. This "
            "function has no effect if the group does not exist.")
        .get();
}

inline FunctionGroup::Ptr axdeletepoint(const FunctionOptions& op)
{
    static auto generate =
        [op](const std::vector<llvm::Value*>&,
             llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // args guaranteed to be empty
        llvm::Constant* loc = llvm::cast<llvm::Constant>(B.CreateGlobalStringPtr("dead")); // char*
        llvm::Constant* size = LLVMType<AXString::SizeType>::get(B.getContext(), 4);
        llvm::Value* str = LLVMType<AXString>::get(B.getContext(), loc, size);
        // Always allocate an AXString here for easier passing to functions
        // @todo shouldn't need an AXString for char* literals
        llvm::Value* alloc =
            B.CreateAlloca(LLVMType<AXString>::get(B.getContext()));
        B.CreateStore(str, alloc);
        return axaddtogroup(op)->execute({alloc}, B);
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

inline FunctionGroup::Ptr axsetattribute(const FunctionOptions& op)
{
    static auto setattribptr =
        [](void* attributeHandle, uint64_t index, const auto value)
    {
        using ValueType = typename std::remove_const
            <typename std::remove_pointer
                <decltype(value)>::type>::type;
        using AttributeHandleType = openvdb::points::AttributeWriteHandle<ValueType>;

        assert(attributeHandle);
        assert(value);
        assert(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        AttributeHandleType* handle = static_cast<AttributeHandleType*>(attributeHandle);
        handle->set(static_cast<openvdb::Index>(index), *value);
    };

    static auto setattribstr =
        [](void* attributeHandle,
           const uint64_t index,
           const AXString* value,
           void* const leafDataPtr)
    {
        using AttributeHandleType = openvdb::points::StringAttributeWriteHandle;

        assert(attributeHandle);
        assert(value);
        assert(leafDataPtr);
        assert(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        const std::string s(value->ptr, value->size);
        AttributeHandleType* const handle =
            static_cast<AttributeHandleType*>(attributeHandle);
        openvdb::ax::compiler::LeafLocalData* const leafData =
            static_cast<openvdb::ax::compiler::LeafLocalData* const>(leafDataPtr);

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
    using SetAttribStr = void(void*, uint64_t, const AXString*, void* const);

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

inline FunctionGroup::Ptr axgetattribute(const FunctionOptions& op)
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

        assert(value);
        assert(attributeHandle);
        assert(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        AttributeHandleType* handle = static_cast<AttributeHandleType*>(attributeHandle);
        (*value) = handle->get(static_cast<openvdb::Index>(index));
    };

    static auto getattribstr =
        [](void* attributeHandle,
           uint64_t index,
           AXString* value,
           const void* const leafDataPtr)
    {
        using AttributeHandleType = openvdb::points::StringAttributeHandle;

        assert(value);
        assert(attributeHandle);
        assert(leafDataPtr);
        assert(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        AttributeHandleType* const handle =
            static_cast<AttributeHandleType*>(attributeHandle);
        const openvdb::ax::compiler::LeafLocalData* const leafData =
            static_cast<const openvdb::ax::compiler::LeafLocalData* const>(leafDataPtr);

        std::string data;
        if (!leafData->getNewStringData(&(handle->array()), index, data)) {
            handle->get(data, static_cast<openvdb::Index>(index));
        }

        assert(value->size == static_cast<AXString::SizeType>(data.size()));
        strcpy(const_cast<char*>(value->ptr), data.c_str());
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
    using GetAttribStr = void(void*, uint64_t, AXString*, const void* const);

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

inline FunctionGroup::Ptr axstrattribsize(const FunctionOptions& op)
{
    static auto strattribsize =
        [](void* attributeHandle,
           uint64_t index,
           const void* const leafDataPtr) -> AXString::SizeType
    {
        using AttributeHandleType = openvdb::points::StringAttributeHandle;

        assert(attributeHandle);
        assert(leafDataPtr);
        assert(index < static_cast<uint64_t>(std::numeric_limits<openvdb::Index>::max()));

        const AttributeHandleType* const handle =
            static_cast<AttributeHandleType*>(attributeHandle);
        const openvdb::ax::compiler::LeafLocalData* const leafData =
            static_cast<const openvdb::ax::compiler::LeafLocalData* const>(leafDataPtr);

        std::string data;
        if (!leafData->getNewStringData(&(handle->array()), index, data)) {
            handle->get(data, static_cast<openvdb::Index>(index));
        }

        return static_cast<AXString::SizeType>(data.size());
    };

    using StrAttribSize = AXString::SizeType(void*, uint64_t, const void* const);

    return FunctionBuilder("strattribsize")
        .addSignature<StrAttribSize>((StrAttribSize*)(strattribsize))
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for querying the size of a points string attribute")
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
    add("strattribsize", axstrattribsize, true);
}

} // namespace codegen
} // namespace ax
} // namespace openvdb_version
} // namespace openvdb


// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
