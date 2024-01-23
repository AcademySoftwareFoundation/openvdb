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

#include "openvdb_ax/compiler/CompilerOptions.h"
#include "openvdb_ax/Exceptions.h"

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include <unordered_map>
#include <cstdlib>
#include <cstring>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {


namespace {

#define OPENVDB_AX_CHECK_MODULE_CONTEXT(B) \
    { \
        const llvm::Function* F = B.GetInsertBlock()->getParent(); \
        const llvm::Module* M = F ? F->getParent() : nullptr; \
        if (!M || M->getName() != "ax.volume.module") { \
            OPENVDB_THROW(AXCompilerError, "Function \"" << (F ? F->getName().str() : "unknown") << \
                "\" cannot be called for the current target:\"" << (M ? M->getName().str() : "unknown") << \
                "\". This function only runs on OpenVDB Grids (not OpenVDB Point Grids)."); \
        } \
    }

}

inline FunctionGroup::UniquePtr axcoordtooffset(const FunctionOptions& op)
{
    using LeafNodeT = openvdb::BoolGrid::TreeType::LeafNodeType;

    /// @warning This function assumes that the node in question is a LeafNode!
    ///   This means that the result of this method is ONLY correct if the
    ///   origin points to an existing leaf node, OR if the offset is zero.
    ///   Currently the VolumeExectuable processes non-leaf nodes (active tiles)
    ///   individually, so the offset for these nodes is always zero. Should
    ///   we need to processes a non-leaf node with a non-zero offset, this
    ///   function should be extended to take a "level" param from the parent
    ///   which identifies the node level and can thus be used to call the
    ///   appropriate offset logic.

    static auto generate = [](const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_ASSERT(args.size() == 1);
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);
        llvm::Value* x = ir_constgep2_64(B, args[0], 0, 0);
        llvm::Value* y = ir_constgep2_64(B, args[0], 0, 1);
        llvm::Value* z = ir_constgep2_64(B, args[0], 0, 2);
        llvm::Value* dimmin1 = LLVMType<int32_t>::get(B.getContext(), int32_t(LeafNodeT::DIM-1u));
        llvm::Value* l2d2 = LLVMType<int32_t>::get(B.getContext(), int32_t(2*LeafNodeT::LOG2DIM));
        llvm::Value* l2d = LLVMType<int32_t>::get(B.getContext(), int32_t(LeafNodeT::LOG2DIM));

        // ((xyz[0] & (DIM-1u)) << 2*Log2Dim)
        x = ir_load(B, x);
        x = binaryOperator(x, dimmin1, ast::tokens::BITAND, B);
        x = binaryOperator(x, l2d2, ast::tokens::SHIFTLEFT, B);

        // ((xyz[1] & (DIM-1u)) << Log2Dim)
        y = ir_load(B, y);
        y = binaryOperator(y, dimmin1, ast::tokens::BITAND, B);
        y = binaryOperator(y, l2d, ast::tokens::SHIFTLEFT, B);

        // (xyz[2] & (DIM-1u))
        z = ir_load(B, z);
        z = binaryOperator(z, dimmin1, ast::tokens::BITAND, B);

        return
            binaryOperator(z,
                binaryOperator(x, y, ast::tokens::PLUS, B),
                    ast::tokens::PLUS, B);
    };

    static auto coordtooffset =
        [](const openvdb::math::Vec3<int32_t>* iscoord)
    {
        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(iscoord);
        return int32_t(LeafNodeT::coordToOffset(*ijk));
    };

    return FunctionBuilder("coordtooffset")
        .addSignature<int32_t(const openvdb::math::Vec3<int32_t>*)>(generate,
                (int32_t(*)(const openvdb::math::Vec3<int32_t>*))(coordtooffset))
        .setArgumentNames({"coord"})
        .addFunctionAttribute(llvm::Attribute::ReadOnly)
        .addFunctionAttribute(llvm::Attribute::NoRecurse)
        .addFunctionAttribute(llvm::Attribute::NoUnwind)
        .addFunctionAttribute(llvm::Attribute::AlwaysInline)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Return the linear table offset of the given global or local coordinates.")
        .get();
}

inline FunctionGroup::UniquePtr axoffsettocoord(const FunctionOptions& op)
{
    using LeafNodeT = openvdb::BoolGrid::TreeType::LeafNodeType;

    /// @warning This function assumes that the node in question is a LeafNode!
    ///   This means that the result of this method is ONLY correct if the
    ///   origin points to an existing leaf node, OR if the offset is zero.
    ///   Currently the VolumeExectuable processes non-leaf nodes (active tiles)
    ///   individually, so the offset for these nodes is always zero. Should
    ///   we need to processes a non-leaf node with a non-zero offset, this
    ///   function should be extended to take a "level" param from the parent
    ///   which identifies the node level and can thus be used to call the
    ///   appropriate offset logic.

    static auto generate = [](const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_ASSERT(args.size() == 2);
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);

        llvm::Value* ijk = args[0];
        llvm::Value* offset = args[1];

        llvm::Value* l2d2 = LLVMType<int32_t>::get(B.getContext(), int32_t(2*LeafNodeT::LOG2DIM));
        llvm::Value* l2d = LLVMType<int32_t>::get(B.getContext(), int32_t(LeafNodeT::LOG2DIM));

        // (offset >> 2*Log2Dim)
        llvm::Value* x = binaryOperator(offset, l2d2, ast::tokens::SHIFTRIGHT, B);
        B.CreateStore(x, ir_constgep2_64(B, ijk, 0, 0));

        // (offset &= ((1<<2*Log2Dim)-1))
        static constexpr int32_t ymask = ((1<<2*LeafNodeT::LOG2DIM)-1);
        offset = binaryOperator(offset, B.getInt32(ymask), ast::tokens::BITAND, B);

        // (n >> Log2Dim)
        llvm::Value* y = binaryOperator(offset, l2d, ast::tokens::SHIFTRIGHT, B);
        B.CreateStore(y, ir_constgep2_64(B, ijk, 0, 1));

        // (n & ((1<<Log2Dim)-1))
        static constexpr int32_t zmask = ((1<<LeafNodeT::LOG2DIM)-1);
        llvm::Value* z = binaryOperator(offset, B.getInt32(zmask), ast::tokens::BITAND, B);
        B.CreateStore(z, ir_constgep2_64(B, ijk, 0, 2));
        return nullptr;
    };

    static auto offsetToCoord =
        [](openvdb::math::Vec3<int32_t>* out, const int32_t offset)
    {
        *out = LeafNodeT::offsetToLocalCoord(offset).asVec3i();
    };

    using OffsetToCoordT = void(openvdb::math::Vec3<int32_t>*, const int32_t);

    return FunctionBuilder("offsettocoord")
        .addSignature<OffsetToCoordT, true>(generate, (OffsetToCoordT*)(offsetToCoord))
        .setArgumentNames({"offset"})
        .addParameterAttribute(0, llvm::Attribute::NoAlias)
        .addParameterAttribute(0, llvm::Attribute::WriteOnly)
        .addParameterAttribute(0, llvm::Attribute::NoCapture)
        .addFunctionAttribute(llvm::Attribute::NoUnwind)
        .addFunctionAttribute(llvm::Attribute::NoRecurse)
        .addFunctionAttribute(llvm::Attribute::AlwaysInline)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axoffsettoglobalcoord(const FunctionOptions& op)
{
    using LeafNodeT = openvdb::BoolGrid::TreeType::LeafNodeType;

    /// @warning This function assumes that the node in question is a LeafNode!
    ///   This means that the result of this method is ONLY correct if the
    ///   origin points to an existing leaf node, OR if the offset is zero.
    ///   Currently the VolumeExectuable processes non-leaf nodes (active tiles)
    ///   individually, so the offset for these nodes is always zero. Should
    ///   we need to processes a non-leaf node with a non-zero offset, this
    ///   function should be extended to take a "level" param from the parent
    ///   which identifies the node level and can thus be used to call the
    ///   appropriate offset logic.

    auto generate = [op](const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_ASSERT(args.size() == 3);
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);

        llvm::Value* result = args[0];
        llvm::Value* offset = args[1];
        llvm::Value* origin = args[2];

        llvm::Value* local = axoffsettocoord(op)->execute({offset}, B);

        for (size_t i = 0; i < 3; ++i){
            llvm::Value* lx = ir_constgep2_64(B, local, 0, i);
            llvm::Value* ox = ir_constgep2_64(B, origin, 0, i);
            ox = binaryOperator(ir_load(B, ox), ir_load(B, lx), ast::tokens::PLUS, B);
            B.CreateStore(ox, ir_constgep2_64(B, result, 0, i));
        }

        return nullptr;
    };

    static auto offsetToGlobalCoord =
        [](openvdb::math::Vec3<int32_t>* out, const int32_t offset, const openvdb::math::Vec3<int32_t>* in)
    {
        auto coord = LeafNodeT::offsetToLocalCoord(offset);
        out->x() = coord.x() + in->x();
        out->y() = coord.y() + in->y();
        out->z() = coord.z() + in->z();
    };

    using OffsetToGlobalCoordT = void(openvdb::math::Vec3<int32_t>*,const int32_t,const openvdb::math::Vec3<int32_t>*);

    return FunctionBuilder("offsettoglobalcoord")
        .addSignature<OffsetToGlobalCoordT, true>(generate, (OffsetToGlobalCoordT*)(offsetToGlobalCoord))
        .setArgumentNames({"offset", "coord"})
        .addParameterAttribute(0, llvm::Attribute::NoAlias)
        .addParameterAttribute(0, llvm::Attribute::WriteOnly)
        .addParameterAttribute(2, llvm::Attribute::NoAlias)
        .addParameterAttribute(2, llvm::Attribute::ReadOnly)
        .addFunctionAttribute(llvm::Attribute::NoUnwind)
        .addFunctionAttribute(llvm::Attribute::AlwaysInline)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axindextoworld(const FunctionOptions& op)
{
    static auto indexToWorld =
        [](openvdb::math::Vec3<double>* out,
           const openvdb::math::Vec3<int32_t>* coord,
           const void* transform)
    {
        const openvdb::math::Transform* const transformPtr =
                static_cast<const openvdb::math::Transform*>(transform);
        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);
        *out = transformPtr->indexToWorld(*ijk);
    };

    using IndexToWorldT = void(openvdb::math::Vec3<double>*, const openvdb::math::Vec3<int32_t>*, const void*);

    return FunctionBuilder("indextoworld")
        .addSignature<IndexToWorldT, true>((IndexToWorldT*)(indexToWorld))
        .setArgumentNames({"coord", "transform"})
        .addParameterAttribute(0, llvm::Attribute::NoAlias)
        .addParameterAttribute(0, llvm::Attribute::WriteOnly)
        .addParameterAttribute(1, llvm::Attribute::NoAlias)
        .addParameterAttribute(1, llvm::Attribute::ReadOnly)
        .addFunctionAttribute(llvm::Attribute::NoUnwind)
        .addFunctionAttribute(llvm::Attribute::AlwaysInline)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Converted the given index space coordiante to a world space value based on the currently executing volume.")
        .get();
}

inline FunctionGroup::UniquePtr axgetcoord(const FunctionOptions& op)
{
    auto generate = [op](const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);
        llvm::Value* origin = extractArgument(compute, "origin");
        llvm::Value* offset = extractArgument(compute, "offset");
        return axoffsettoglobalcoord(op)->execute({offset, origin}, B);
    };

    return FunctionBuilder("getcoord")
        .addSignature<openvdb::math::Vec3<int32_t>*()>(generate)
        .setEmbedIR(true)
        .setConstantFold(false)
        .addDependency("offsettoglobalcoord")
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Returns the current voxel's ijk index space coordiante.")
        .get();
}

template <size_t Index>
inline FunctionGroup::UniquePtr axgetcoord(const FunctionOptions& op)
{
    static_assert(Index <= 2, "Invalid index for axgetcoord");

    auto generate = [op](const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        llvm::Value* coord = axgetcoord(op)->execute({}, B);
        return ir_load(B, ir_constgep2_64(B, coord, 0, Index));
    };

    return FunctionBuilder((Index == 0 ? "getcoordx" : Index == 1 ? "getcoordy" : "getcoordz"))
        .addSignature<int32_t()>(generate)
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .addDependency("getcoord")
        .setDocumentation((
             Index == 0 ? "Returns the current voxel's X index value in index space as an integer." :
             Index == 1 ? "Returns the current voxel's Y index value in index space as an integer." :
                          "Returns the current voxel's Z index value in index space as an integer."))
        .get();
}

inline FunctionGroup::UniquePtr axgetvoxelpws(const FunctionOptions& op)
{
    auto generate = [op](const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        llvm::Value* transform = extractArgument(compute, "transforms");
        llvm::Value* wi = extractArgument(compute, "write_index");
        transform = ir_gep(B, transform, wi);
        transform = ir_load(B, transform);
        llvm::Value* coord = axgetcoord(op)->execute({}, B);
        return axindextoworld(op)->execute({coord, transform}, B);
    };

    return FunctionBuilder("getvoxelpws")
        .addSignature<openvdb::math::Vec3<double>*()>(generate)
        .setEmbedIR(true)
        .setConstantFold(false)
        .addDependency("getcoord")
        .addDependency("indextoworld")
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Returns the current voxel's position in world space as a vector float.")
        .get();
}

inline FunctionGroup::UniquePtr axisactive(const FunctionOptions& op)
{
    static auto generate = [](const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_AX_CHECK_MODULE_CONTEXT(B);
        // Pull out parent function arguments
        llvm::Function* compute = B.GetInsertBlock()->getParent();
        return extractArgument(compute, "active");
    };

    return FunctionBuilder("isactive")
        .addSignature<bool()>(generate)
        .setEmbedIR(true)
        .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Returns whether the current voxel or tile is active.")
        .get();
}

inline FunctionGroup::UniquePtr axsetvoxel(const FunctionOptions& op)
{
    static auto setvoxelptr =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           const int32_t level,
           const bool ison,
           const auto value)
    {
        using ValueType = typename std::remove_const
            <typename std::remove_pointer
                <decltype(value)>::type>::type;
        using GridType = typename openvdb::BoolGrid::ValueConverter<ValueType>::Type;
        using RootNodeType = typename GridType::TreeType::RootNodeType;
        using AccessorType = typename GridType::Accessor;

        OPENVDB_ASSERT(accessor);
        OPENVDB_ASSERT(coord);

        // set value only to avoid changing topology
        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);
        AccessorType* const accessorPtr = static_cast<AccessorType*>(accessor);

        if (level != -1) {
            OPENVDB_ASSERT(level >= 0);
            accessorPtr->addTile(Index(level), *ijk, *value, ison);
        }
        else {
            // Check the depth to avoid creating voxel topology for higher levels
            // @note  This option is not configurable outside of the executable
            const int depth = accessorPtr->getValueDepth(*ijk);
            if (depth == static_cast<int>(RootNodeType::LEVEL)) {
                // voxel/leaf level
                OPENVDB_ASSERT(accessorPtr->probeConstLeaf(*ijk));
                if (ison) accessorPtr->setValueOn(*ijk, *value);
                else      accessorPtr->setValueOff(*ijk, *value);
            }
            else {
                // If the current depth is not the maximum (i.e voxel/leaf level) then
                // we're iterating over tiles of an internal node (NodeT0 is the leaf level).
                // We can't call setValueOnly or other variants as this will forcer voxel
                // topology to be created. Whilst the VolumeExecutables runs in such a
                // way that this is safe, it's not desirable; we just want to change the
                // tile value. There is no easy way to do this; we have to set a new tile
                // with the same active state.
                // @warning This code assume that getValueDepth() is always called to force
                // a node cache.
                using NodeT1 = typename AccessorType::template NodeTypeAtLevel<1>;
                using NodeT2 = typename AccessorType::template NodeTypeAtLevel<2>;
                if (NodeT1* node = accessorPtr->template getNode<NodeT1>()) {
                    const openvdb::Index index = node->coordToOffset(*ijk);
                    OPENVDB_ASSERT(node->isChildMaskOff(index));
                    node->addTile(index, *value, ison);
                }
                else if (NodeT2* node = accessorPtr->template getNode<NodeT2>()) {
                    const openvdb::Index index = node->coordToOffset(*ijk);
                    OPENVDB_ASSERT(node->isChildMaskOff(index));
                    node->addTile(index, *value, ison);
                }
                else {
                    const int level = RootNodeType::LEVEL - depth;
                    accessorPtr->addTile(level, *ijk, *value, ison);
                }
            }
        }
    };

    static auto setvoxelstr =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           const int32_t level,
           const bool ison,
           codegen::String* value)
    {
        const std::string copy(value->str());
        setvoxelptr(accessor, coord, level, ison, &copy);
    };

    static auto setvoxel =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           const int32_t level,
           const bool ison,
           const auto value) {
        setvoxelptr(accessor, coord, level, ison, &value);
    };

    using SetVoxelD = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const double);
    using SetVoxelF = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const float);
    using SetVoxelI64 = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const int64_t);
    using SetVoxelI32 = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const int32_t);
    using SetVoxelI16 = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const int16_t);
    using SetVoxelB = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const bool);
    using SetVoxelV2D = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec2<double>*);
    using SetVoxelV2F = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec2<float>*);
    using SetVoxelV2I = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec2<int32_t>*);
    using SetVoxelV3D = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec3<double>*);
    using SetVoxelV3F = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec3<float>*);
    using SetVoxelV3I = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec3<int32_t>*);
    using SetVoxelV4D = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec4<double>*);
    using SetVoxelV4F = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec4<float>*);
    using SetVoxelV4I = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Vec4<int32_t>*);
    using SetVoxelM3D = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Mat3<double>*);
    using SetVoxelM3F = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Mat3<float>*);
    using SetVoxelM4D = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Mat4<double>*);
    using SetVoxelM4F = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, const openvdb::math::Mat4<float>*);
    using SetVoxelStr = void(void*, const openvdb::math::Vec3<int32_t>*, const int32_t, const bool, codegen::String*);

    return FunctionBuilder("setvoxel")
        .addSignature<SetVoxelD>((SetVoxelD*)(setvoxel))
        .addSignature<SetVoxelF>((SetVoxelF*)(setvoxel))
        .addSignature<SetVoxelI64>((SetVoxelI64*)(setvoxel))
        .addSignature<SetVoxelI32>((SetVoxelI32*)(setvoxel))
        .addSignature<SetVoxelI16>((SetVoxelI16*)(setvoxel))
        .addSignature<SetVoxelB>((SetVoxelB*)(setvoxel))
            .addParameterAttribute(0, llvm::Attribute::NoAlias)
            .addParameterAttribute(0, llvm::Attribute::ReadOnly)
            .addParameterAttribute(0, llvm::Attribute::NoCapture)
            .addParameterAttribute(1, llvm::Attribute::ReadOnly)
            .addParameterAttribute(1, llvm::Attribute::NoCapture)
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
            .addParameterAttribute(0, llvm::Attribute::NoCapture)
            .addParameterAttribute(1, llvm::Attribute::ReadOnly)
            .addParameterAttribute(1, llvm::Attribute::NoCapture)
            .addParameterAttribute(4, llvm::Attribute::NoAlias)
            .addParameterAttribute(4, llvm::Attribute::ReadOnly)
            .addParameterAttribute(4, llvm::Attribute::NoCapture)
            .addFunctionAttribute(llvm::Attribute::NoUnwind)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for setting the value of a voxel.")
        .get();
}

inline FunctionGroup::UniquePtr axgetvoxel(const FunctionOptions& op)
{
    static auto getvoxel =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           auto value)
    {
        using ValueType = typename std::remove_pointer<decltype(value)>::type;
        using GridType = typename openvdb::BoolGrid::ValueConverter<ValueType>::Type;
        using AccessorType = typename GridType::Accessor;

        OPENVDB_ASSERT(accessor);
        OPENVDB_ASSERT(coord);
        OPENVDB_ASSERT(value);

        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);
        (*value) = static_cast<const AccessorType*>(accessor)->getValue(*ijk);
    };

    static auto getvoxelstr =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           codegen::String* value)
    {
        using GridType = openvdb::BoolGrid::ValueConverter<std::string>::Type;
        using AccessorType = GridType::Accessor;

        OPENVDB_ASSERT(accessor);
        OPENVDB_ASSERT(coord);
        OPENVDB_ASSERT(value);

        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);
        const std::string& str = static_cast<const AccessorType*>(accessor)->getValue(*ijk);
        // Copy the string to AX's required representation
        *value = str;
    };

    static auto getvoxel_s2t =
        [](void* accessor,
           void* sourceTransform,
           void* targetTransform,
           const openvdb::math::Vec3<int32_t>* origin,
           const int32_t offset,
           auto value)
    {
        using ValueType = typename std::remove_pointer<decltype(value)>::type;
        using GridType = typename openvdb::BoolGrid::ValueConverter<ValueType>::Type;
        using LeafNodeT = typename GridType::TreeType::LeafNodeType;
        using AccessorType = typename GridType::Accessor;

        OPENVDB_ASSERT(accessor);
        OPENVDB_ASSERT(origin);
        OPENVDB_ASSERT(sourceTransform);
        OPENVDB_ASSERT(targetTransform);

        const AccessorType* const accessorPtr = static_cast<const AccessorType*>(accessor);
        const openvdb::math::Transform* const sourceTransformPtr =
                static_cast<const openvdb::math::Transform*>(sourceTransform);
        const openvdb::math::Transform* const targetTransformPtr =
                static_cast<const openvdb::math::Transform*>(targetTransform);

        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(origin);
        auto coord = *ijk + LeafNodeT::offsetToLocalCoord(offset);
        coord = targetTransformPtr->worldToIndexCellCentered(sourceTransformPtr->indexToWorld(coord));
        (*value) = accessorPtr->getValue(coord);
    };

    static auto getvoxelstr_s2t =
        [](void* accessor,
           void* sourceTransform,
           void* targetTransform,
           const openvdb::math::Vec3<int32_t>* origin,
           const int32_t offset,
           codegen::String* value)
    {
        using GridType = typename openvdb::BoolGrid::ValueConverter<std::string>::Type;
        using LeafNodeT = typename GridType::TreeType::LeafNodeType;
        using AccessorType = typename GridType::Accessor;

        OPENVDB_ASSERT(accessor);
        OPENVDB_ASSERT(origin);
        OPENVDB_ASSERT(sourceTransform);
        OPENVDB_ASSERT(targetTransform);

        const AccessorType* const accessorPtr = static_cast<const AccessorType*>(accessor);
        const openvdb::math::Transform* const sourceTransformPtr =
                static_cast<const openvdb::math::Transform*>(sourceTransform);
        const openvdb::math::Transform* const targetTransformPtr =
                static_cast<const openvdb::math::Transform*>(targetTransform);

        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(origin);
        auto coord = *ijk + LeafNodeT::offsetToLocalCoord(offset);
        coord = targetTransformPtr->worldToIndexCellCentered(sourceTransformPtr->indexToWorld(coord));
        const std::string& str = accessorPtr->getValue(coord);
        // Copy the string to AX's required representation
        *value = str;
    };

    using GetVoxelS2T_D = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, double*);
    using GetVoxelS2T_F = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, float*);
    using GetVoxelS2T_I64 = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, int64_t*);
    using GetVoxelS2T_I32 = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, int32_t*);
    using GetVoxelS2T_I16 = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, int16_t*);
    using GetVoxelS2T_B = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, bool*);
    using GetVoxelS2T_V2D = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec2<double>*);
    using GetVoxelS2T_V2F = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec2<float>*);
    using GetVoxelS2T_V2I = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec2<int32_t>*);
    using GetVoxelS2T_V3D = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec3<double>*);
    using GetVoxelS2T_V3F = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec3<float>*);
    using GetVoxelS2T_V3I = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec3<int32_t>*);
    using GetVoxelS2T_V4D = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec4<double>*);
    using GetVoxelS2T_V4F = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec4<float>*);
    using GetVoxelS2T_V4I = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Vec4<int32_t>*);
    using GetVoxelS2T_M3D = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Mat3<double>*);
    using GetVoxelS2T_M3F = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Mat3<float>*);
    using GetVoxelS2T_M4D = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Mat4<double>*);
    using GetVoxelS2T_M4F = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, openvdb::math::Mat4<float>*);
    using GetVoxelS2T_Str = void(void*, void*, void*, const openvdb::math::Vec3<int32_t>*, int32_t, codegen::String*);

    using GetVoxelD = void(void*, const openvdb::math::Vec3<int32_t>*, double*);
    using GetVoxelF = void(void*, const openvdb::math::Vec3<int32_t>*, float*);
    using GetVoxelI64 = void(void*, const openvdb::math::Vec3<int32_t>*, int64_t*);
    using GetVoxelI32 = void(void*, const openvdb::math::Vec3<int32_t>*, int32_t*);
    using GetVoxelI16 = void(void*, const openvdb::math::Vec3<int32_t>*, int16_t*);
    using GetVoxelB = void(void*, const openvdb::math::Vec3<int32_t>*, bool*);
    using GetVoxelV2D = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec2<double>*);
    using GetVoxelV2F = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec2<float>*);
    using GetVoxelV2I = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec2<int32_t>*);
    using GetVoxelV3D = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec3<double>*);
    using GetVoxelV3F = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec3<float>*);
    using GetVoxelV3I = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec3<int32_t>*);
    using GetVoxelV4D = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec4<double>*);
    using GetVoxelV4F = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec4<float>*);
    using GetVoxelV4I = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Vec4<int32_t>*);
    using GetVoxelM3D = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Mat3<double>*);
    using GetVoxelM3F = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Mat3<float>*);
    using GetVoxelM4D = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Mat4<double>*);
    using GetVoxelM4F = void(void*, const openvdb::math::Vec3<int32_t>*, openvdb::math::Mat4<float>*);
    using GetVoxelStr = void(void*, const openvdb::math::Vec3<int32_t>*, codegen::String*);

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
            .addParameterAttribute(1, llvm::Attribute::NoAlias)
            .addParameterAttribute(1, llvm::Attribute::ReadOnly)
            .addParameterAttribute(2, llvm::Attribute::WriteOnly)
            .addParameterAttribute(2, llvm::Attribute::NoAlias)
            .addFunctionAttribute(llvm::Attribute::NoUnwind)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .addSignature<GetVoxelS2T_D>((GetVoxelS2T_D*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_F>((GetVoxelS2T_F*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_I64>((GetVoxelS2T_I64*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_I32>((GetVoxelS2T_I32*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_I16>((GetVoxelS2T_I16*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_B>((GetVoxelS2T_B*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V2D>((GetVoxelS2T_V2D*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V2F>((GetVoxelS2T_V2F*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V2I>((GetVoxelS2T_V2I*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V3D>((GetVoxelS2T_V3D*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V3F>((GetVoxelS2T_V3F*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V3I>((GetVoxelS2T_V3I*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V4D>((GetVoxelS2T_V4D*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V4F>((GetVoxelS2T_V4F*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_V4I>((GetVoxelS2T_V4I*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_M3F>((GetVoxelS2T_M3F*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_M3D>((GetVoxelS2T_M3D*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_M4F>((GetVoxelS2T_M4F*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_M4D>((GetVoxelS2T_M4D*)(getvoxel_s2t))
        .addSignature<GetVoxelS2T_Str>((GetVoxelS2T_Str*)(getvoxelstr_s2t))
            .addParameterAttribute(0, llvm::Attribute::NoAlias)
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

inline FunctionGroup::UniquePtr axprobevalue(const FunctionOptions& op)
{
    static auto probe =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           bool* ison,
           auto value)
    {
        using ValueType = typename std::remove_pointer<decltype(value)>::type;
        using GridType = typename openvdb::BoolGrid::ValueConverter<ValueType>::Type;
        using AccessorType = typename GridType::Accessor;

        OPENVDB_ASSERT(accessor);
        OPENVDB_ASSERT(coord);
        OPENVDB_ASSERT(value);
        OPENVDB_ASSERT(ison);

        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);
        *ison = static_cast<const AccessorType*>(accessor)->probeValue(*ijk, *value);
    };

    static auto probestr =
        [](void* accessor,
           const openvdb::math::Vec3<int32_t>* coord,
           bool* ison,
           codegen::String* value)
    {
        using GridType = openvdb::BoolGrid::ValueConverter<std::string>::Type;
        using AccessorType = GridType::Accessor;

        OPENVDB_ASSERT(accessor);
        OPENVDB_ASSERT(coord);
        OPENVDB_ASSERT(value);
        OPENVDB_ASSERT(ison);

        const openvdb::Coord* ijk = reinterpret_cast<const openvdb::Coord*>(coord);

        std::string str;
        *ison = static_cast<const AccessorType*>(accessor)->probeValue(*ijk, str);
        // Copy the string to AX's required representation
        *value = str;
    };

    using ProbeValueD = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, double*);
    using ProbeValueF = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, float*);
    using ProbeValueI64 = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, int64_t*);
    using ProbeValueI32 = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, int32_t*);
    using ProbeValueI16 = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, int16_t*);
    using ProbeValueB = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, bool*);
    using ProbeValueV2D = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec2<double>*);
    using ProbeValueV2F = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec2<float>*);
    using ProbeValueV2I = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec2<int32_t>*);
    using ProbeValueV3D = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec3<double>*);
    using ProbeValueV3F = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec3<float>*);
    using ProbeValueV3I = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec3<int32_t>*);
    using ProbeValueV4D = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec4<double>*);
    using ProbeValueV4F = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec4<float>*);
    using ProbeValueV4I = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Vec4<int32_t>*);
    using ProbeValueM3D = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Mat3<double>*);
    using ProbeValueM3F = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Mat3<float>*);
    using ProbeValueM4D = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Mat4<double>*);
    using ProbeValueM4F = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, openvdb::math::Mat4<float>*);
    using ProbeValueStr = void(void*, const openvdb::math::Vec3<int32_t>*, bool*, codegen::String*);

    return FunctionBuilder("probevalue")
        .addSignature<ProbeValueD>((ProbeValueD*)(probe))
        .addSignature<ProbeValueF>((ProbeValueF*)(probe))
        .addSignature<ProbeValueI64>((ProbeValueI64*)(probe))
        .addSignature<ProbeValueI32>((ProbeValueI32*)(probe))
        .addSignature<ProbeValueI16>((ProbeValueI16*)(probe))
        .addSignature<ProbeValueB>((ProbeValueB*)(probe))
        .addSignature<ProbeValueV2D>((ProbeValueV2D*)(probe))
        .addSignature<ProbeValueV2F>((ProbeValueV2F*)(probe))
        .addSignature<ProbeValueV2I>((ProbeValueV2I*)(probe))
        .addSignature<ProbeValueV3D>((ProbeValueV3D*)(probe))
        .addSignature<ProbeValueV3F>((ProbeValueV3F*)(probe))
        .addSignature<ProbeValueV3I>((ProbeValueV3I*)(probe))
        .addSignature<ProbeValueV4D>((ProbeValueV4D*)(probe))
        .addSignature<ProbeValueV4F>((ProbeValueV4F*)(probe))
        .addSignature<ProbeValueV4I>((ProbeValueV4I*)(probe))
        .addSignature<ProbeValueM3F>((ProbeValueM3F*)(probe))
        .addSignature<ProbeValueM3D>((ProbeValueM3D*)(probe))
        .addSignature<ProbeValueM4F>((ProbeValueM4F*)(probe))
        .addSignature<ProbeValueM4D>((ProbeValueM4D*)(probe))
        .addSignature<ProbeValueStr>((ProbeValueStr*)(probestr))
            .addParameterAttribute(0, llvm::Attribute::NoAlias)
            .addParameterAttribute(1, llvm::Attribute::NoAlias)
            .addParameterAttribute(1, llvm::Attribute::ReadOnly)
            .addParameterAttribute(2, llvm::Attribute::WriteOnly)
            .addParameterAttribute(2, llvm::Attribute::NoAlias)
            .addParameterAttribute(3, llvm::Attribute::WriteOnly)
            .addParameterAttribute(3, llvm::Attribute::NoAlias)
            .addFunctionAttribute(llvm::Attribute::NoUnwind)
            .addFunctionAttribute(llvm::Attribute::NoRecurse)
            .setConstantFold(false)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("Internal function for getting the value of a voxel and its active state.")
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

    add("coordtooffset", axcoordtooffset, true);
    add("offsettocoord", axoffsettocoord, true);
    add("offsettoglobalcoord", axoffsettoglobalcoord, true);
    add("indextoworld", axindextoworld, true);

    add("getcoord", axgetcoord);
    add("getcoordx", axgetcoord<0>);
    add("getcoordy", axgetcoord<1>);
    add("getcoordz", axgetcoord<2>);
    add("getvoxelpws", axgetvoxelpws);
    add("isactive", axisactive, true); // needs tests

    add("getvoxel", axgetvoxel, true);
    add("setvoxel", axsetvoxel, true);
    add("probevalue", axprobevalue, true);
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

