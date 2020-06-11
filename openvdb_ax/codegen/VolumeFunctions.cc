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

/// @file codegen/VolumeFunctions.cc
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  Contains the function objects that define the functions used in
///   volume compute function generation, to be inserted into the FunctionRegistry.
///   These define the functions available when operating on volumes.
///   Also includes the definitions for the volume value retrieval and setting.
///

#include "Functions.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/Exceptions.h>

#include <openvdb_ax/version.h>

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

inline FunctionGroup::Ptr axgetvoxelpws(const FunctionOptions& op)
{
    static auto generate = [](const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        assert(compute);
        assert(compute->getName() == "ax.compute.voxel");
        llvm::Value* coordws = extractArgument(compute, "coord_ws");
        assert(coordws);
        return coordws;
    };

    return FunctionBuilder("getvoxelpws")
        .addSignature<openvdb::math::Vec3<float>*()>(generate)
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Returns the current voxel's position in world space as a vector float.")
        .get();
}

template <size_t Index>
inline FunctionGroup::Ptr axgetcoord(const FunctionOptions& op)
{
    static_assert(Index <= 2, "Invalid index for axgetcoord");

    static auto generate = [](const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        assert(compute);
        assert(compute->getName() == "ax.compute.voxel");
        llvm::Value* coordis = extractArgument(compute, "coord_is");
        assert(coordis);
        return B.CreateLoad(B.CreateConstGEP2_64(coordis, 0, Index));
    };

    return FunctionBuilder((Index == 0 ? "getcoordx" : Index == 1 ? "getcoordy" : "getcoordz"))
        .addSignature<int()>(generate)
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation((
             Index == 0 ? "Returns the current voxel's X index value in index space as an integer." :
             Index == 1 ? "Returns the current voxel's Y index value in index space as an integer." :
                          "Returns the current voxel's Z index value in index space as an integer."))
        .get();
}

inline FunctionGroup::Ptr axsetvoxel(const FunctionOptions& op)
{
    static auto setvoxelptr =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           const auto value)
    {
        using ValueType = typename std::remove_const
            <typename std::remove_pointer
                <decltype(value)>::type>::type;
        using GridType = typename openvdb::BoolGrid::ValueConverter<ValueType>::Type;
        using AccessorType = typename GridType::Accessor;

        assert(accessor);
        assert(coord);

        // set value only to avoid changing topology
        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);
        AccessorType* const accessorPtr = static_cast<AccessorType* const>(accessor);
        accessorPtr->setValueOnly(*ijk, *value);
    };

    static auto setvoxelstr =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           const AXString* value)
    {
        const std::string copy(value->ptr, value->size);
        setvoxelptr(accessor, coord, &copy);
    };

    static auto setvoxel =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           const auto value) {
        setvoxelptr(accessor, coord, &value);
    };

    using SetVoxelD = void(void*, const openvdb::math::Vec3<int32_t>*, const double);
    using SetVoxelF = void(void*, const openvdb::math::Vec3<int32_t>*, const float);
    using SetVoxelI64 = void(void*, const openvdb::math::Vec3<int32_t>*, const int64_t);
    using SetVoxelI32 = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t);
    using SetVoxelI16 = void(void*, const openvdb::math::Vec3<int32_t>*, const int16_t);
    using SetVoxelB = void(void*, const openvdb::math::Vec3<int32_t>*, const bool);
    using SetVoxelV2D = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec2<double>*);
    using SetVoxelV2F = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec2<float>*);
    using SetVoxelV2I = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec2<int32_t>*);
    using SetVoxelV3D = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec3<double>*);
    using SetVoxelV3F = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec3<float>*);
    using SetVoxelV3I = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec3<int32_t>*);
    using SetVoxelV4D = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec4<double>*);
    using SetVoxelV4F = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec4<float>*);
    using SetVoxelV4I = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Vec4<int32_t>*);
    using SetVoxelM3D = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Mat3<double>*);
    using SetVoxelM3F = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Mat3<float>*);
    using SetVoxelM4D = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Mat4<double>*);
    using SetVoxelM4F = void(void*, const openvdb::math::Vec3<int32_t>*, const openvdb::math::Mat4<float>*);
    using SetVoxelStr = void(void*, const openvdb::math::Vec3<int32_t>*, const AXString*);

    return FunctionBuilder("setvoxel")
        .addSignature<SetVoxelD>((SetVoxelD*)(setvoxel))
        .addSignature<SetVoxelF>((SetVoxelF*)(setvoxel))
        .addSignature<SetVoxelI64>((SetVoxelI64*)(setvoxel))
        .addSignature<SetVoxelI32>((SetVoxelI32*)(setvoxel))
        .addSignature<SetVoxelI16>((SetVoxelI16*)(setvoxel))
        .addSignature<SetVoxelB>((SetVoxelB*)(setvoxel))
            .addParameterAttribute(0, llvm::Attribute::NoAlias)
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addParameterAttribute(1, llvm::Attribute::ReadOnly)
            .addFunctionAttribute(llvm::Attribute::NoUnwind)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .addSignature<SetVoxelV2D>((SetVoxelV2D*)(setvoxelptr))
        .addSignature<SetVoxelV2F>((SetVoxelV2F*)(setvoxelptr))
        .addSignature<SetVoxelV2I>((SetVoxelV2I*)(setvoxelptr))
        .addSignature<SetVoxelV3D>((SetVoxelV3D*)(setvoxelptr))
        .addSignature<SetVoxelV3F>((SetVoxelV3F*)(setvoxelptr))
        .addSignature<SetVoxelV3I>((SetVoxelV3I*)(setvoxelptr))
        .addSignature<SetVoxelV4D>((SetVoxelV4D*)(setvoxelptr))
        .addSignature<SetVoxelV4F>((SetVoxelV4F*)(setvoxelptr))
        .addSignature<SetVoxelV4I>((SetVoxelV4I*)(setvoxelptr))
        .addSignature<SetVoxelM3D>((SetVoxelM3D*)(setvoxelptr))
        .addSignature<SetVoxelM3F>((SetVoxelM3F*)(setvoxelptr))
        .addSignature<SetVoxelM4D>((SetVoxelM4D*)(setvoxelptr))
        .addSignature<SetVoxelM4F>((SetVoxelM4F*)(setvoxelptr))
        .addSignature<SetVoxelStr>((SetVoxelStr*)(setvoxelstr))
            .addParameterAttribute(0, llvm::Attribute::NoAlias)
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addParameterAttribute(1, llvm::Attribute::ReadOnly)
            .addParameterAttribute(2, llvm::Attribute::ReadOnly)
            .addFunctionAttribute(llvm::Attribute::NoUnwind)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for setting the value of a voxel.")
        .get();
}

inline FunctionGroup::Ptr axgetvoxel(const FunctionOptions& op)
{
    static auto getvoxel =
        [](void* accessor,
           void* transform,
           const openvdb::math::Vec3<float>* wspos,
           auto value)
    {
        using ValueType = typename std::remove_pointer<decltype(value)>::type;
        using GridType = typename openvdb::BoolGrid::ValueConverter<ValueType>::Type;
        using AccessorType = typename GridType::Accessor;

        assert(accessor);
        assert(wspos);
        assert(transform);

        const AccessorType* const accessorPtr = static_cast<const AccessorType* const>(accessor);
        const openvdb::math::Transform* const transformPtr =
                static_cast<const openvdb::math::Transform* const>(transform);
        const openvdb::Coord coordIS = transformPtr->worldToIndexCellCentered(*wspos);
        (*value) = accessorPtr->getValue(coordIS);
    };

    // @todo  This is inherently flawed as we're not allocating the data in IR when passing
    //        this back. When, in the future, grids can be written to and read from at the
    //        same time we might need to revisit string accesses.
    static auto getvoxelstr =
        [](void* accessor,
           void* transform,
           const openvdb::math::Vec3<float>* wspos,
           AXString* value)
    {
        using GridType = typename openvdb::BoolGrid::ValueConverter<std::string>::Type;
        using AccessorType = typename GridType::Accessor;

        assert(accessor);
        assert(wspos);
        assert(transform);

        const AccessorType* const accessorPtr = static_cast<const AccessorType* const>(accessor);
        const openvdb::math::Transform* const transformPtr =
                static_cast<const openvdb::math::Transform* const>(transform);
        openvdb::Coord coordIS = transformPtr->worldToIndexCellCentered(*wspos);
        const std::string& str = accessorPtr->getValue(coordIS);
        value->ptr = str.c_str();
        value->size = static_cast<AXString::SizeType>(str.size());
    };

    using GetVoxelD = void(void*, void*, const openvdb::math::Vec3<float>*, double*);
    using GetVoxelF = void(void*, void*, const openvdb::math::Vec3<float>*, float*);
    using GetVoxelI64 = void(void*, void*, const openvdb::math::Vec3<float>*, int64_t*);
    using GetVoxelI32 = void(void*, void*, const openvdb::math::Vec3<float>*, int32_t*);
    using GetVoxelI16 = void(void*, void*, const openvdb::math::Vec3<float>*, int16_t*);
    using GetVoxelB = void(void*, void*, const openvdb::math::Vec3<float>*, bool*);
    using GetVoxelV2D = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec2<double>*);
    using GetVoxelV2F = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec2<float>*);
    using GetVoxelV2I = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec2<int32_t>*);
    using GetVoxelV3D = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec3<double>*);
    using GetVoxelV3F = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec3<float>*);
    using GetVoxelV3I = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec3<int32_t>*);
    using GetVoxelV4D = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec4<double>*);
    using GetVoxelV4F = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec4<float>*);
    using GetVoxelV4I = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Vec4<int32_t>*);
    using GetVoxelM3D = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Mat3<double>*);
    using GetVoxelM3F = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Mat3<float>*);
    using GetVoxelM4D = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Mat4<double>*);
    using GetVoxelM4F = void(void*, void*, const openvdb::math::Vec3<float>*, openvdb::math::Mat4<float>*);
    using GetVoxelStr = void(void*, void*, const openvdb::math::Vec3<float>*, AXString*);

    return FunctionBuilder("getvoxel")
        .addSignature<GetVoxelD>((GetVoxelD*)(getvoxel))
        .addSignature<GetVoxelF>((GetVoxelF*)(getvoxel))
        .addSignature<GetVoxelI64>((GetVoxelI64*)(getvoxel))
        .addSignature<GetVoxelI32>((GetVoxelI32*)(getvoxel))
        .addSignature<GetVoxelI16>((GetVoxelI16*)(getvoxel))
        .addSignature<GetVoxelB>((GetVoxelB*)(getvoxel))
        .addSignature<GetVoxelV2D>((GetVoxelV2D*)(getvoxel))
        .addSignature<GetVoxelV2F>((GetVoxelV2F*)(getvoxel))
        .addSignature<GetVoxelV2I>((GetVoxelV2I*)(getvoxel))
        .addSignature<GetVoxelV3D>((GetVoxelV3D*)(getvoxel))
        .addSignature<GetVoxelV3F>((GetVoxelV3F*)(getvoxel))
        .addSignature<GetVoxelV3I>((GetVoxelV3I*)(getvoxel))
        .addSignature<GetVoxelV4D>((GetVoxelV4D*)(getvoxel))
        .addSignature<GetVoxelV4F>((GetVoxelV4F*)(getvoxel))
        .addSignature<GetVoxelV4I>((GetVoxelV4I*)(getvoxel))
        .addSignature<GetVoxelM3F>((GetVoxelM3F*)(getvoxel))
        .addSignature<GetVoxelM3D>((GetVoxelM3D*)(getvoxel))
        .addSignature<GetVoxelM4F>((GetVoxelM4F*)(getvoxel))
        .addSignature<GetVoxelM4D>((GetVoxelM4D*)(getvoxel))
        .addSignature<GetVoxelStr>((GetVoxelStr*)(getvoxelstr))
            .addParameterAttribute(0, llvm::Attribute::NoAlias)
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addParameterAttribute(1, llvm::Attribute::NoAlias)
            .addParameterAttribute(1, llvm::Attribute::ReadOnly)
            .addParameterAttribute(2, llvm::Attribute::ReadOnly)
            .addParameterAttribute(3, llvm::Attribute::WriteOnly)
            .addParameterAttribute(3, llvm::Attribute::NoAlias)
            .addFunctionAttribute(llvm::Attribute::NoUnwind)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for setting the value of a voxel.")
        .get();
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

void insertVDBVolumeFunctions(FunctionRegistry& registry,
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

    // volume functions

    add("getcoordx", axgetcoord<0>);
    add("getcoordy", axgetcoord<1>);
    add("getcoordz", axgetcoord<2>);
    add("getvoxelpws", axgetvoxelpws);
    add("getvoxel", axgetvoxel, true);
    add("setvoxel", axsetvoxel, true);
}

}
}
}
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
