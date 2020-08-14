// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

#include "../compiler/CompilerOptions.h"
#include "../Exceptions.h"
#include "../version.h"

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {


namespace {

/// @todo  Provide more framework for functions such that they can only
///        be registered against compatible code generators.
inline void verifyContext(const llvm::Function* const F, const std::string& name)
{
    if (!F || F->getName() != "ax.compute.voxel") {
        OPENVDB_THROW(AXCompilerError, "Function \"" << name << "\" cannot be called for "
            "the current target. This function only runs on OpenVDB Grids (not OpenVDB Point Grids).");
    }
}

}


inline FunctionGroup::Ptr axgetvoxelpws(const FunctionOptions& op)
{
    static auto generate = [](const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        verifyContext(compute, "getvoxelpws");
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
        verifyContext(compute, (Index == 0 ? "getcoordx" : Index == 1 ? "getcoordy" : "getcoordz"));
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
        using RootNodeType = typename GridType::TreeType::RootNodeType;
        using AccessorType = typename GridType::Accessor;

        assert(accessor);
        assert(coord);

        // set value only to avoid changing topology
        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);
        AccessorType* const accessorPtr = static_cast<AccessorType* const>(accessor);

        // Check the depth to avoid creating voxel topology for higher levels
        // @todo  As this option is not configurable outside of the executable, we
        // should be able to avoid this branching by setting the depth as a global
        const int depth = accessorPtr->getValueDepth(*ijk);
        if (depth == static_cast<int>(RootNodeType::LEVEL)) {
            accessorPtr->setValueOnly(*ijk, *value);
        }
        else {
            // If the current depth is not the maximum (i.e voxel/leaf level) then
            // we're iterating over tiles of an internal node (NodeT0 is the leaf level).
            // We can't call setValueOnly or other variants as this will forcer voxel
            // topology to be created. Whilst the VolumeExecutables runs in such a
            // way that this is safe, it's not desriable; we just want to change the
            // tile value. There is no easy way to do this; we have to set a new tile
            // with the same active state.
            // @warning This code assume that getValueDepth() is always called to force
            // a node cache.
            using NodeT1 = typename AccessorType::NodeT1;
            using NodeT2 = typename AccessorType::NodeT2;
            if (NodeT1* node = accessorPtr->template getNode<NodeT1>()) {
                const openvdb::Index index = node->coordToOffset(*ijk);
                assert(node->isChildMaskOff(index));
                node->addTile(index, *value, node->isValueOn(index));
            }
            else if (NodeT2* node = accessorPtr->template getNode<NodeT2>()) {
                const openvdb::Index index = node->coordToOffset(*ijk);
                assert(node->isChildMaskOff(index));
                node->addTile(index, *value, node->isValueOn(index));
            }
            else {
                const int level = RootNodeType::LEVEL - depth;
                accessorPtr->addTile(level, *ijk, *value, accessorPtr->isValueOn(*ijk));
            }
        }
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

