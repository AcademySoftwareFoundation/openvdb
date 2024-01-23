// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/VolumeExecutable.cc

#include "cli.h" // from vdb_ax command line tool

#include "VolumeExecutable.h"
#include "Logger.h"

#include "openvdb_ax/Exceptions.h"
// @TODO refactor so we don't have to include VolumeComputeGenerator.h,
// but still have the functions defined in one place
#include "openvdb_ax/codegen/VolumeComputeGenerator.h"
#include "openvdb_ax/codegen/String.h"

#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/NodeManager.h>
#include <openvdb/util/Assert.h>

#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_vector.h>

#include <memory>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

std::ostream& operator<<(std::ostream& os, const VolumeExecutable::Streaming& v)
{
    if (v == VolumeExecutable::Streaming::ON)        os << "ON";
    else if (v == VolumeExecutable::Streaming::OFF)  os << "OFF";
    else if (v == VolumeExecutable::Streaming::AUTO) os << "AUTO";
    return os;
}

std::ostream& operator<<(std::ostream& os, const VolumeExecutable::IterType& v)
{
    if (v == VolumeExecutable::IterType::ON)        os << "ON";
    else if (v == VolumeExecutable::IterType::OFF)  os << "OFF";
    else if (v == VolumeExecutable::IterType::ALL)  os << "ALL";
    return os;
}

/// @brief Settings which are stored on the volume executer
///   and are configurable by the user.
template <bool IsCLI>
struct VolumeExecutable::Settings
{
    template <typename T>
    using ParamT = typename std::conditional<IsCLI,
        cli::Param<T>,
        cli::BasicParam<T>>::type;

    template <typename T>
    using ParamBuilderT = typename std::conditional<IsCLI,
        cli::ParamBuilder<T>,
        cli::BasicParamBuilder<T>>::type;

    ///////////////////////////////////////////////////////////////////////////

    inline std::vector<cli::ParamBase*> optional()
    {
        OPENVDB_ASSERT(IsCLI);
        std::vector<cli::ParamBase*> params {
            &this->mCreateMissing,
            &this->mTreeExecutionLevel,
            &this->mValueIterator,
            &this->mActiveTileStreaming,
            &this->mGrainSizes,
            &this->mBindings
        };
        return params;
    }

    inline void initialize(const VolumeExecutable::Settings<true>& S)
    {
        if (S.mCreateMissing.isInit())       mCreateMissing = S.mCreateMissing.get();
        if (S.mTreeExecutionLevel.isInit())  mTreeExecutionLevel = S.mTreeExecutionLevel.get();
        if (S.mValueIterator.isInit())       mValueIterator = S.mValueIterator.get();
        if (S.mActiveTileStreaming.isInit()) mActiveTileStreaming = S.mActiveTileStreaming.get();
        if (S.mGrainSizes.isInit())          mGrainSizes = S.mGrainSizes.get();
        if (S.mBindings.isInit())            mBindings = S.mBindings.get();
    }

    ///////////////////////////////////////////////////////////////////////////

    ParamT<bool> mCreateMissing =
        ParamBuilderT<bool>()
            .addOpt("--create-missing [ON|OFF]")
            .setDoc("whether to implicitly create volumes (Default: ON). Volumes are created "
                    "if they are referenced in the AX program but do not exist on the input "
                    "geometry.")
            .setDefault(true)
            .setCB([](bool& v, const char* arg) {
                if (std::strcmp(arg, "ON") == 0)       v = true;
                else if (std::strcmp(arg, "OFF") == 0) v = false;
                else OPENVDB_THROW(CLIError, "Invalid option passed to --create-missing: '" << arg << "'");
            })
            .get();

    ParamT<std::pair<Index,Index>> mTreeExecutionLevel =
        ParamBuilderT<std::pair<Index,Index>>()
            .addOpt("--tree-level [l1|l1:l2]")
            .setDoc("a node level or node range to process (Default: 0:4). If [l2] is not "
                    "provided only a single level [l1] is processed. By default AX processes the "
                    "entire VDB tree.")
            .setDefault({0, FloatTree::DEPTH-1})
            .setCB([](std::pair<Index,Index>& v, const char* arg) {
                cli::DefaultCallback<Index>::get()(v.first, arg);
                if (const char* sep = std::strchr(arg, ':')) {
                    cli::DefaultCallback<Index>::get()(v.second, sep+1);
                }
            })
            .get();

    ParamT<IterType> mValueIterator =
        ParamBuilderT<IterType>()
            .addOpt("--node-iter [ON|OFF|ALL]")
            .setDoc("the active state type of nodes to process (Default: ON). [ON] processes "
                    "active values, [OFF] processes inactive values, ALL processes both. The default "
                    "is [ALL].")
            .setDefault(IterType::ON)
            .setCB([](IterType& v, const char* arg) {
                if (std::strcmp(arg, "ON") == 0)       v = IterType::ON;
                else if (std::strcmp(arg, "OFF") == 0) v = IterType::OFF;
                else if (std::strcmp(arg, "ALL") == 0) v = IterType::ALL;
                else OPENVDB_THROW(CLIError, "Invalid option passed to --node-iter: '" << arg << "'");
            })
            .get();

   ParamT<Streaming> mActiveTileStreaming =
        ParamBuilderT<Streaming>()
            .addOpt("--tile-stream [ON|OFF|AUTO]")
            .setDoc("set the active tile streaming behaviour (Default: AUTO). Active tiles "
                "are constant tree nodes that can potentially span a large area of the volume. By default "
                "AX will 'stream' these tiles only when it detects that a given program may produce non-"
                "constant values. This involves densifying tiles into potentially finer child topology. "
                "You can explicitly set this behaviour to always be [ON], or disable it with [OFF]. The "
                "latter ensures that each active tiles single value is only processed once.")
            .setDefault(Streaming::AUTO)
            .setCB([](Streaming& v, const char* arg) {
                if (std::strcmp(arg, "ON") == 0)        v = Streaming::ON;
                else if (std::strcmp(arg, "OFF") == 0)  v = Streaming::OFF;
                else if (std::strcmp(arg, "AUTO") == 0) v = Streaming::AUTO;
                else OPENVDB_THROW(CLIError, "Invalid option passed to --tile-stream: '" << arg << "'");
            })
            .get();

   ParamT<std::pair<size_t,size_t>> mGrainSizes =
        ParamBuilderT<std::pair<size_t,size_t>>()
            .addOpt("--volume-grain [g1|g1:g2]")
            .setDoc("threading grain size for processing nodes (Default: 1:32). [g1] controls the outer layer's "
                    "grain size. The outer layer visits each individual node in a VDB. [g2] controls the inner "
                    "layer's grain size. This is used for Volumes during task splitting of active tile streaming. "
                    "A value of 0 disables threading for the respective layer.")
            .setDefault({ 1, 32 })
            .setCB([](std::pair<size_t,size_t>& v, const char* arg) {
                cli::DefaultCallback<size_t>::get()(v.first, arg);
                if (const char* sep = std::strchr(arg, ':')) {
                    cli::DefaultCallback<size_t>::get()(v.second, sep+1);
                }
            })
            .get();

   ParamT<AttributeBindings> mBindings =
        ParamBuilderT<AttributeBindings>()
            .addOpt("--bindings [\"ax_name:volume_name,...\"]")
            .setDoc("attribute bindings for volumes. The argument accepts a quoted string list of "
                    "AX (source code) name to data (vdb attribute) name pairs joined by colons and "
                    "seperated by commas. For example:\n"
                    "  --bindings \"velocity:v,density:s\"\n"
                    "binds velocity AX accesses to a 'v' attribute and density AX accesses to a 's' "
                    "attribute. The following snippet would then alias these attributes:\n"
                    "  v@velocity *= 5;   // actually accesses 'v' volume\n"
                    "   @density += 1.0f; // actually accesses 's' volume")
            .setDefault(AttributeBindings{})
            .setCB([](AttributeBindings& bindings, const char* c) {
                std::string source, target;
                std::string* active = &source, *other = &target;
                while (*c != '\0') {
                    if (*c == ':') std::swap(active, other);
                    else if (*c == ',') {
                        std::swap(active, other);
                        if (source.empty() || target.empty()) {
                            OPENVDB_THROW(CLIError, "invalid string passed to --bindings: '" << c << "'");
                        }
                        bindings.set(source, target);
                        source.clear();
                        target.clear();
                    }
                    else {
                        active->push_back(*c);
                    }
                    ++c;
                }

                if (source.empty() || target.empty()) {
                    OPENVDB_THROW(CLIError, "invalid string passed to --bindings: '" << c << "'");
                }
                bindings.set(source, target);
            })
            .get();
};

namespace {

/// @brief Volume Kernel types
///
using KernelBufferFunctionPtr = std::add_pointer<codegen::VolumeKernelBuffer::Signature>::type;
using KernelNodeFunctionPtr = std::add_pointer<codegen::VolumeKernelNode::Signature>::type;

template <typename ValueT>
using ConverterT = typename openvdb::BoolGrid::ValueConverter<ValueT>::Type;
using SupportedTypeList = openvdb::TypeList<
    ConverterT<double>,
    ConverterT<float>,
    ConverterT<int64_t>,
    ConverterT<int32_t>,
    ConverterT<int16_t>,
    ConverterT<bool>,
    ConverterT<openvdb::math::Vec2<double>>,
    ConverterT<openvdb::math::Vec2<float>>,
    ConverterT<openvdb::math::Vec2<int32_t>>,
    ConverterT<openvdb::math::Vec3<double>>,
    ConverterT<openvdb::math::Vec3<float>>,
    ConverterT<openvdb::math::Vec3<int32_t>>,
    ConverterT<openvdb::math::Vec4<double>>,
    ConverterT<openvdb::math::Vec4<float>>,
    ConverterT<openvdb::math::Vec4<int32_t>>,
    ConverterT<openvdb::math::Mat3<double>>,
    ConverterT<openvdb::math::Mat3<float>>,
    ConverterT<openvdb::math::Mat4<double>>,
    ConverterT<openvdb::math::Mat4<float>>,
    ConverterT<std::string>>;

inline bool supported(const ast::tokens::CoreType type)
{
    switch (type) {
        case ast::tokens::BOOL    : return true;
        case ast::tokens::INT16   : return true;
        case ast::tokens::INT32   : return true;
        case ast::tokens::INT64   : return true;
        case ast::tokens::FLOAT   : return true;
        case ast::tokens::DOUBLE  : return true;
        case ast::tokens::VEC2I   : return true;
        case ast::tokens::VEC2F   : return true;
        case ast::tokens::VEC2D   : return true;
        case ast::tokens::VEC3I   : return true;
        case ast::tokens::VEC3F   : return true;
        case ast::tokens::VEC3D   : return true;
        case ast::tokens::VEC4I   : return true;
        case ast::tokens::VEC4F   : return true;
        case ast::tokens::VEC4D   : return true;
        case ast::tokens::MAT3F   : return true;
        case ast::tokens::MAT3D   : return true;
        case ast::tokens::MAT4F   : return true;
        case ast::tokens::MAT4D   : return true;
        case ast::tokens::STRING  : return true;
        case ast::tokens::UNKNOWN :
        default                   : return false;
    }
}

inline openvdb::GridBase::Ptr
createGrid(const ast::tokens::CoreType& type)
{
    // assert so the executer can be marked as noexcept (assuming nothing throws in compute)
    OPENVDB_ASSERT(supported(type) && "Could not retrieve accessor from unsupported type");
    switch (type) {
        case ast::tokens::BOOL    : return ConverterT<bool>::create();
        case ast::tokens::INT16   : return ConverterT<int16_t>::create();
        case ast::tokens::INT32   : return ConverterT<int32_t>::create();
        case ast::tokens::INT64   : return ConverterT<int64_t>::create();
        case ast::tokens::FLOAT   : return ConverterT<float>::create();
        case ast::tokens::DOUBLE  : return ConverterT<double>::create();
        case ast::tokens::VEC2D   : return ConverterT<openvdb::math::Vec2<double>>::create();
        case ast::tokens::VEC2F   : return ConverterT<openvdb::math::Vec2<float>>::create();
        case ast::tokens::VEC2I   : return ConverterT<openvdb::math::Vec2<int32_t>>::create();
        case ast::tokens::VEC3D   : return ConverterT<openvdb::math::Vec3<double>>::create();
        case ast::tokens::VEC3F   : return ConverterT<openvdb::math::Vec3<float>>::create();
        case ast::tokens::VEC3I   : return ConverterT<openvdb::math::Vec3<int32_t>>::create();
        case ast::tokens::VEC4D   : return ConverterT<openvdb::math::Vec4<double>>::create();
        case ast::tokens::VEC4F   : return ConverterT<openvdb::math::Vec4<float>>::create();
        case ast::tokens::VEC4I   : return ConverterT<openvdb::math::Vec4<int32_t>>::create();
        case ast::tokens::MAT3D   : return ConverterT<openvdb::math::Mat3<double>>::create();
        case ast::tokens::MAT3F   : return ConverterT<openvdb::math::Mat3<float>>::create();
        case ast::tokens::MAT4D   : return ConverterT<openvdb::math::Mat4<double>>::create();
        case ast::tokens::MAT4F   : return ConverterT<openvdb::math::Mat4<float>>::create();
        case ast::tokens::STRING  : return ConverterT<std::string>::create();
        case ast::tokens::UNKNOWN :
        default                   : return nullptr;
    }
}

struct GridCache
{
    inline void addWriteGrid(GridBase& grid)
    {
        mWrite.emplace_back(&grid);
    }

    inline void addReadGrid(GridBase& grid, const bool copy)
    {
        if (copy) {
            // deepCopyGrid returns a shared pointer
            mReadCache.emplace_back(grid.deepCopyGrid());
            mRead.emplace_back(mReadCache.back().get());
        }
        else {
            mRead.emplace_back(&grid);
        }
    }

    std::vector<GridBase*> mWrite;
    std::vector<GridBase*> mRead;
    std::vector<GridBase::Ptr> mReadCache;
};

/// @brief  Shared data for the parallel operator
struct OpData
{
    KernelBufferFunctionPtr mKernelValueBuffer;
    KernelNodeFunctionPtr mKernelNode;
    const CustomData* mCustomData;
    const AttributeRegistry* mAttributeRegistry;
    std::vector<void*> mVoidTransforms;
    openvdb::GridBase** mGrids;
    size_t mActiveIndex;
    Index mTreeLevelMin; // only used with NodeManagers
    Index mTreeLevelMax; // only used with NodeManagers
    size_t mIterMode; // 0 = OFF, 1 = ON, 2 = ALL
    bool mActiveTileStreaming;
    size_t mTileGrainSize;
};

/// The arguments of the generated function
struct VolumeFunctionArguments
{
    struct Accessors
    {
        using UniquePtr = std::unique_ptr<Accessors>;
        virtual ~Accessors() = default;
    };

    template <typename TreeT>
    struct TypedAccessor final : public Accessors
    {
        using UniquePtr = std::unique_ptr<TypedAccessor<TreeT>>;
        TypedAccessor(TreeT& tree) : mAccessor(new tree::ValueAccessor<TreeT>(tree)) {}
        ~TypedAccessor() override final = default;
        const std::unique_ptr<tree::ValueAccessor<TreeT>> mAccessor;
    };

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    VolumeFunctionArguments(const OpData& data, openvdb::GridBase** read, const AttributeRegistry& reg)
        : mData(data)
        , mAccessors()
        , mVoidAccessors()
        {
            mAccessors.reserve(data.mVoidTransforms.size());
            mVoidAccessors.reserve(data.mVoidTransforms.size());
            for (const auto& regdata : reg.data()) {
                this->addAccessor(*read, regdata.type());
                ++read;
            }
        }

    /// @brief  Given a built version of the function signature, automatically
    ///         bind the current arguments and return a callable function
    ///         which takes no arguments
    inline auto bindBufferKernel()
    {
        using FunctionTraitsT = codegen::VolumeKernelBuffer::FunctionTraitsT;
        using ReturnT = FunctionTraitsT::ReturnType;

        return [&](const openvdb::Coord& origin, void* buffer, Index64* mask, const size_t size) -> ReturnT {
            return mData.mKernelValueBuffer(static_cast<FunctionTraitsT::Arg<0>::Type>(mData.mCustomData),
                reinterpret_cast<FunctionTraitsT::Arg<1>::Type>(origin.data()),
                static_cast<FunctionTraitsT::Arg<2>::Type>(buffer),
                static_cast<FunctionTraitsT::Arg<3>::Type>(mask),
                static_cast<FunctionTraitsT::Arg<4>::Type>(size),
                static_cast<FunctionTraitsT::Arg<5>::Type>(mData.mIterMode),
                static_cast<FunctionTraitsT::Arg<6>::Type>(mVoidAccessors.data()),
                static_cast<FunctionTraitsT::Arg<7>::Type>(mData.mVoidTransforms.data()),
                static_cast<FunctionTraitsT::Arg<8>::Type>(mData.mActiveIndex));
        };
    }

    inline auto bindNodeKernel(void* activeAccessor)
    {
        using FunctionTraitsT = codegen::VolumeKernelNode::FunctionTraitsT;
        using ReturnT = FunctionTraitsT::ReturnType;

        return [&, activeAccessor](const openvdb::Coord& ijk) -> ReturnT {
            return mData.mKernelNode(static_cast<FunctionTraitsT::Arg<0>::Type>(mData.mCustomData),
                reinterpret_cast<FunctionTraitsT::Arg<1>::Type>(ijk.data()),
                static_cast<FunctionTraitsT::Arg<2>::Type>(mVoidAccessors.data()),
                static_cast<FunctionTraitsT::Arg<3>::Type>(mData.mVoidTransforms.data()),
                static_cast<FunctionTraitsT::Arg<4>::Type>(mData.mActiveIndex),
                static_cast<FunctionTraitsT::Arg<5>::Type>(activeAccessor));
        };
    }

private:
    template <typename TreeT>
    inline void
    addAccessor(TreeT& tree)
    {
        typename TypedAccessor<TreeT>::UniquePtr accessor(new TypedAccessor<TreeT>(tree));
        mVoidAccessors.emplace_back(accessor->mAccessor.get());
        mAccessors.emplace_back(std::move(accessor));
    }

    inline void
    addAccessor(openvdb::GridBase* grid, const ast::tokens::CoreType& type)
    {
        OPENVDB_ASSERT(grid);
        // assert so the executer can be marked as noexcept (assuming nothing throws in compute)
        OPENVDB_ASSERT(supported(type) && "Could not retrieve accessor from unsupported type");
        switch (type) {
            case ast::tokens::BOOL    : { this->addAccessor(static_cast<ConverterT<bool>*>(grid)->tree()); return; }
            case ast::tokens::INT16   : { this->addAccessor(static_cast<ConverterT<int16_t>*>(grid)->tree()); return; }
            case ast::tokens::INT32   : { this->addAccessor(static_cast<ConverterT<int32_t>*>(grid)->tree()); return; }
            case ast::tokens::INT64   : { this->addAccessor(static_cast<ConverterT<int64_t>*>(grid)->tree()); return; }
            case ast::tokens::FLOAT   : { this->addAccessor(static_cast<ConverterT<float>*>(grid)->tree()); return; }
            case ast::tokens::DOUBLE  : { this->addAccessor(static_cast<ConverterT<double>*>(grid)->tree()); return; }
            case ast::tokens::VEC2D   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec2<double>>*>(grid)->tree()); return; }
            case ast::tokens::VEC2F   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec2<float>>*>(grid)->tree()); return; }
            case ast::tokens::VEC2I   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec2<int32_t>>*>(grid)->tree()); return; }
            case ast::tokens::VEC3D   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec3<double>>*>(grid)->tree()); return; }
            case ast::tokens::VEC3F   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec3<float>>*>(grid)->tree()); return; }
            case ast::tokens::VEC3I   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec3<int32_t>>*>(grid)->tree()); return; }
            case ast::tokens::VEC4D   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec4<double>>*>(grid)->tree()); return; }
            case ast::tokens::VEC4F   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec4<float>>*>(grid)->tree()); return; }
            case ast::tokens::VEC4I   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Vec4<int32_t>>*>(grid)->tree()); return; }
            case ast::tokens::MAT3D   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Mat3<double>>*>(grid)->tree()); return; }
            case ast::tokens::MAT3F   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Mat3<float>>*>(grid)->tree()); return; }
            case ast::tokens::MAT4D   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Mat4<double>>*>(grid)->tree()); return; }
            case ast::tokens::MAT4F   : { this->addAccessor(static_cast<ConverterT<openvdb::math::Mat4<float>>*>(grid)->tree()); return; }
            case ast::tokens::STRING  : { this->addAccessor(static_cast<ConverterT<std::string>*>(grid)->tree()); return; }
            case ast::tokens::UNKNOWN :
            default                   : return;
        }
    }

private:
    const OpData& mData;
    std::vector<Accessors::UniquePtr> mAccessors;
    std::vector<void*> mVoidAccessors;
};

struct ValueOnIter
{
    template<typename NodeT>
    using IterTraitsT = typename tree::IterTraits<NodeT, typename NodeT::ValueOnIter>;
};

struct ValueAllIter
{
    template<typename NodeT>
    using IterTraitsT = typename tree::IterTraits<NodeT, typename NodeT::ValueAllIter>;
};

struct ValueOffIter
{
    template<typename NodeT>
    using IterTraitsT = typename tree::IterTraits<NodeT, typename NodeT::ValueOffIter>;
};

template <typename TreeT, typename IterT>
struct VolumeExecuterOp
{
    using LeafManagerT = tree::LeafManager<TreeT>;
    using LeafRangeT = typename LeafManagerT::LeafRange;
    using RootNodeT =  typename TreeT::RootNodeType;
    using LeafNodeT = typename TreeT::LeafNodeType;
    using ValueT =  typename TreeT::ValueType;
    using IterType = typename IterT::template IterTraitsT<LeafNodeT>;

    /// @brief  Small struct that represents a single VDB value
    struct Tile {
        Tile() = default;
        Tile(const Index offset, const ValueT& v, const bool on)
            : mOffset(offset), mValue(v), mActive(on) {}
        Index mOffset; ValueT mValue; bool mActive;
    };

    VolumeExecuterOp(const OpData& data, TreeT& tree)
        : mData(data)
        , mTree(tree) {}

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Public operator for all node types except LeafNodes (i.e.
    ///   RootNode types and InternalNode types). Active values can be streamed
    ///   when enabled. Otherwise, depending on the iterator type, Active and
    ///   Inactive values are processed.
    /// @note  The steaming of active values can produce child topology, but
    ///   the non-streamed (standard value iteration) execution does not
    ///   descend into lower levels of the tree.
    template <typename NodeType, typename =
        typename std::enable_if<!std::is_same<NodeType, LeafRangeT>::value>::type>
    void operator()(NodeType& node) const
    {
        // if the current node level does not match, skip
        const Index level = node.getLevel();
        OPENVDB_ASSERT(level > 0);
        if (level < mData.mTreeLevelMin) return;
        if (level > mData.mTreeLevelMax) return;

        if (mData.mActiveTileStreaming) {
            // Expect that the iterator type is NOT ValueOff if we are
            // streaming ACTIVE tiles (this is an artificial limitation to stop
            // typical VDBs memory exploding when things like inactive root
            // node tiles are streamed).
            OPENVDB_ASSERT((!std::is_same<ValueOffIter, IterT>::value));
            // Process ACTIVE values
            this->process(node);

            if (std::is_same<ValueAllIter, IterT>::value) {
                // If we're streaming active values but processing both active
                // and inactive (all) values, manually process OFF value tiles
                /// @warning The value off iterator visits child nodes, so skip them
                using IterType = ValueOffIter::IterTraitsT<NodeType>;
                auto it = IterType::begin(node);
                if (!it) return; // only possible if this node only has active tiles
                VolumeFunctionArguments args(mData, mData.mGrids, *mData.mAttributeRegistry);
                tree::ValueAccessor<TreeT> acc(mTree);
                /// @note  the node kernel works for any VDB configuration and value type
                auto kernel = args.bindNodeKernel(static_cast<void*>(&acc));
                for (; it; ++it) {
                    // Manually skip child topology (not-skipped by the ValueOff iterator)
                    if (this->isChildMaskOn(node, it.pos())) continue;
                    kernel(it.getCoord());
                }
            }
        }
        else {
            // If we're here, we are not streaming active values - simply process
            // each individual tile value pointed at by the value iterator
            /// @warning The value off iterator visits child nodes, so skip them
            using IterType = typename IterT::template IterTraitsT<NodeType>;
            auto it = IterType::begin(node);
            if (!it) return;
            VolumeFunctionArguments args(mData, mData.mGrids, *mData.mAttributeRegistry);
            tree::ValueAccessor<TreeT> acc(mTree);
            /// @note  the node kernel works for any VDB configuration and value type
            auto kernel = args.bindNodeKernel(static_cast<void*>(&acc));
            for (; it; ++it) {
                // Manually skip child topology (not-skipped by the ValueOff iterator)
                if (std::is_same<ValueOffIter, IterT>::value &&
                    this->isChildMaskOn(node, it.pos())) continue;
                OPENVDB_ASSERT(!this->isChildMaskOn(node, it.pos()));
                kernel(it.getCoord());
            }
        }
    }

    // For use with a LeafManager, when the target execution level is 0
    /// @note  We currently don't use the same node manager to process leaf
    ///   nodes. Instead, we use a unique leaf manager which calls the
    ///   LeafRangeT operator.
    void operator()(const LeafRangeT& range) const
    {
        this->processLeafNodes<ValueT>(range);
    }

private:

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Trivial methods that alias for a RootNode when querying the
    ///  the state of the child MASK. This is NOT the same as asking the
    ///  RootNode if a tile is active/inactive - these methods specifically
    ///  return whether a Node's ChildMask is ON or OFF. As a RootNode does not
    ///  hold a ChildMask, it always returns false. This allows the main
    ///  operator of this class to handle Internal and Root node types with
    ///  the same iterators.
    //@{
    template <typename NodeType>
    inline static bool isChildMaskOn(const NodeType& node, const Index n) {
        return node.isChildMaskOn(n);
    }
    inline static bool isChildMaskOn(const RootNodeT&, const Index) {
        return false;
    }
    //@}

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// Kernel leaf exec methods for different ValueTypes

    /// @brief  Default kernel execution for all values types except bools and strings
    template <typename ValueType, typename std::enable_if<
        !std::is_same<ValueType, bool>::value &&
        !std::is_same<ValueType, std::string>::value
        >::type* = nullptr>
    void processLeafNodes(const LeafRangeT& range) const
    {
        VolumeFunctionArguments args(mData, mData.mGrids, *mData.mAttributeRegistry);
        auto kernel = args.bindBufferKernel();
        for (auto leaf = range.begin(); leaf; ++leaf) {
            void* buffer = static_cast<void*>(leaf->buffer().data());
            Index64* masks = &(leaf->getValueMask().template getWord<Index64>(0));
            kernel(leaf->origin(), buffer, masks, size_t(LeafNodeT::NUM_VOXELS));
        }
    }

    /// @brief Overload for bool and string value types
    template <typename ValueType, typename std::enable_if<
        std::is_same<ValueType, bool>::value ||
        std::is_same<ValueType, std::string>::value
        >::type* = nullptr>
    void processLeafNodes(const LeafRangeT& range) const
    {
        using TempBufferT = typename std::conditional<
            std::is_same<std::string, ValueType>::value,
            ax::codegen::String, bool>::type;

        VolumeFunctionArguments args(mData, mData.mGrids, *mData.mAttributeRegistry);
        auto kernel = args.bindBufferKernel();
        TempBufferT values[LeafNodeT::NUM_VOXELS];

        for (auto leaf = range.begin(); leaf; ++leaf) {
            Index64* masks = &(leaf->getValueMask().template getWord<Index64>(0));
            auto& buffer = leaf->buffer();
            for (auto it = IterType::begin(*leaf); it; ++it) {
                values[it.pos()] = *it;
            }
            kernel(leaf->origin(), values, masks, size_t(LeafNodeT::NUM_VOXELS));
            for (auto it = IterType::begin(*leaf); it; ++it) {
                buffer.setValue(it.pos(), ValueT(values[it.pos()]));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Recursively process an active or inactive tile with a LEVEL
    ///  greater than 1.
    /// @details This algorithm recursively breaks up tile domains into child
    ///  domains until a node which represents a leaf parent (LEVEL == 1) is
    ///  reached. Each child node domain is looped over and the node results
    ///  are returned. If any child nodes were created, this method returns a
    ///  new node of NodeT which the child nodes added. If any tiles were
    ///  created which do NOT match the value and active state of the parent
    ///  domain, a new node of NodeT is created and returned with the unique
    ///  tiles added. If, however, all child domain values and active states
    ///  are constant, a nullptr is returned and the input value and active
    ///  arguments are set to the new states (possibly the same).
    template <typename NodeT, typename std::enable_if<(NodeT::LEVEL > 1)>::type* = nullptr>
    NodeT* processTile(const Coord& origin, ValueT& value, bool& active) const
    {
        using ChildNodeT = typename NodeT::ChildNodeType;

        tbb::concurrent_vector<ChildNodeT*> childs;
        tbb::concurrent_vector<Tile> tiles;

        const tbb::blocked_range<Index>
            totalRange(0, NodeT::NUM_VALUES, mData.mTileGrainSize);

        if (mData.mTileGrainSize > 0) {
            tbb::parallel_for(totalRange, [&](const auto& range) {
                Coord ijk;
                for (Index n = range.begin(), N = range.end(); n < N; ++n) {
                    NodeT::offsetToLocalCoord(n, ijk);
                    ijk <<= ChildNodeT::TOTAL;
                    ijk += origin;
                    ValueT _value = value; // possible new value
                    bool _active = true; // possible new state, starts always active
                    ChildNodeT* child = this->processTile<ChildNodeT>(ijk, _value, _active);
                    if (child) { childs.emplace_back(child); }
                    else if (_value != value || _active != true) {
                        tiles.emplace_back(n, _value, _active);
                    }
                }
            });
        }
        else {
            Coord ijk;
            for (Index n = totalRange.begin(), N = totalRange.end(); n < N; ++n) {
                NodeT::offsetToLocalCoord(n, ijk);
                ijk <<= ChildNodeT::TOTAL;
                ijk += origin;
                ValueT _value = value; // possible new value
                bool _active = true; // possible new state, starts always active
                ChildNodeT* child = this->processTile<ChildNodeT>(ijk, _value, _active);
                if (child) { childs.emplace_back(child); }
                else if (_value != value || _active != true) {
                    tiles.emplace_back(n, _value, _active);
                }
            }
        }

        std::unique_ptr<NodeT> node;

        // If we have child nodes then we have to create a parent
        if (!childs.empty()) {
            node.reset(new NodeT(origin, value, /*active=*/true)); // always starts as active
            for (auto n : childs) node->addChild(n);
        }

        if (!tiles.empty()) {
            const auto& tile = tiles.front();

            // if !node, we didn't create any child nodes. If we have NodeT::NUM_VALUES
            // number of tiles, check their values - they may all be constant, in which
            // case we don't need to create a parent at all
            if (!node) {
                if (tiles.size() == NodeT::NUM_VALUES) {
                    const bool constant =
                        (std::find_if(tiles.cbegin() + 1, tiles.cend(), [&tile](const auto& iter) {
                            return iter.mValue != tile.mValue || iter.mActive != tile.mActive;
                        }) == tiles.cend());
                    if (!constant) {
                        node.reset(new NodeT(origin, value, /*active=*/true)); // always starts as active
                    }
                }
            }

            // If the node exists, add all the tiles, otherwise we are replacing this
            // entire node with a new value and state
            if (node) {
                for (auto& n : tiles) {
                    node->addTile(n.mOffset, n.mValue, n.mActive);
                }
            }
            else {
                value = tile.mValue;
                active = tile.mActive;
            }
        }

        return node.release();
    }

    /// @brief Specialization of the recursive domain splitting to execute on
    ///  a leaf node parent node (i.e. LEVEL==1).
    template <typename NodeT, typename std::enable_if<(NodeT::LEVEL == 1)>::type* = nullptr>
    NodeT* processTile(const Coord& origin, ValueT& value, bool& active) const
    {
        // @note  Always starts off active as streaming only streams active tiles
        // @todo  This tile create is redundant if process does not create child
        //   nodes or non-uniform tiles. Could replace this with the 2xvector
        //   storage solution, needs profiling.
        std::unique_ptr<NodeT> cache(new NodeT(origin, value, /*active=*/true));
        this->process(*cache);

        ValueT first;
        bool state;
        bool constant = cache->isConstant(first, state);
        if (!constant) return cache.release();

        value = first;
        active = state;
        return nullptr;
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Process a nodes ValueOn/Off/All tiles, where the node's level.
    ///   is greater than 1. This method calls the above recursive algorithms
    ///   to evaluate each child tile.
    template <typename NodeT, typename std::enable_if<(NodeT::LEVEL > 1)>::type* = nullptr>
    void process(NodeT& parent) const
    {
        using ChildNodeT = typename NodeT::ChildNodeType;
        OPENVDB_ASSERT((!std::is_same<ValueOffIter, IterT>::value));

        // Explicitly use a ValueOn Iterator (only stream ON Values)
        for (auto it = ValueOnIter::IterTraitsT<NodeT>::begin(parent); it; ++it) {
            const Coord origin = it.getCoord();
            const ValueT value = it.getValue();
            // ValueIter should never point to a child node - only time this is
            // possible is with a ValueOff iter, but this code only ever invoked
            // with a ValueOnIter
            OPENVDB_ASSERT(!this->isChildMaskOn(parent, it.pos()));
            // only processes active tiles
            OPENVDB_ASSERT(it.isValueOn());

            ValueT _value = value;
            bool _active = true;
            if (ChildNodeT* child = this->processTile<ChildNodeT>(origin, _value, _active)) {
                parent.addChild(child);
            }
            else {
                if (_value != value) it.setValue(_value);
                if (_active != true) it.setValueOn(_active);
            }
        }
    }

    /// @brief  Process a leaf parent (i.e LEVEL==1) node's ValueOn tiles
    ///   densifying tiles into leaf nodes and vice versa where necessary.
    template <typename NodeT, typename std::enable_if<(NodeT::LEVEL == 1)>::type* = nullptr>
    void process(NodeT& parent) const
    {
        using ChildNodeT = typename NodeT::ChildNodeType;
        static_assert(ChildNodeT::DIM == LeafNodeT::DIM,
            "Expected the parent node type of LeafNodeT to have a "
            "CHILD_DIM equal to the DIM of a LeafNodeT.");
        OPENVDB_ASSERT((!std::is_same<ValueOffIter, IterT>::value));

        // only process active tiles when streaming
        if (parent.getValueMask().isOff()) return;

        const tbb::blocked_range<Index>
            totalRange(0, NodeT::NUM_VALUES, mData.mTileGrainSize);

        if (mData.mTileGrainSize > 0) {
            tbb::concurrent_vector<LeafNodeT*> leafs;
            tbb::concurrent_vector<Tile> tiles;
            tbb::parallel_for(totalRange, [&](const tbb::blocked_range<Index>& range) {
                std::vector<LeafNodeT*> lleafs;
                std::vector<Tile> ltiles;
                lleafs.reserve(range.size());
                ltiles.reserve(range.size());

                this->processLeafParent(parent, range, lleafs, ltiles);

                if (!lleafs.empty()) {
                    std::move(lleafs.begin(), lleafs.end(), leafs.grow_by(lleafs.size()));
                }
                if (!ltiles.empty()) {
                    std::move(ltiles.begin(), ltiles.end(), tiles.grow_by(ltiles.size()));
                }
            });
            for (auto n : leafs)  parent.addLeaf(n);
            for (auto& n : tiles) parent.addTile(n.mOffset, n.mValue, n.mActive);
        }
        else {
            std::vector<LeafNodeT*> leafs;
            std::vector<Tile> tiles;

            this->processLeafParent(parent, totalRange, leafs, tiles);

            for (auto n : leafs)  parent.addLeaf(n);
            for (auto& n : tiles) parent.addTile(n.mOffset, n.mValue, n.mActive);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// Kernel Leaf parent exec methods for different ValueTypes

    /// @brief  Default kernel execution for all values types except bools and strings
    template <typename NodeT, typename std::enable_if<
        !std::is_same<typename NodeT::ValueType, bool>::value &&
        !std::is_same<typename NodeT::ValueType, std::string>::value
        >::type* = nullptr>
    void processLeafParent(NodeT& parent,
        const tbb::blocked_range<Index>& range,
        std::vector<LeafNodeT*>& leafs,
        std::vector<Tile>& tiles) const
    {
        // ValueOff iterators should explicitly disable tile streaming
        OPENVDB_ASSERT((!std::is_same<ValueOffIter, IterT>::value));
        // @todo update to new InternalNode API methods when available
        auto* const table = const_cast<typename NodeT::UnionType*>(parent.getTable());
        const auto& mask = parent.getValueMask();

        VolumeFunctionArguments args(mData, mData.mGrids, *mData.mAttributeRegistry);
        auto kernel = args.bindBufferKernel();

        Index64* word;
        ValueT* data, *end;
        std::unique_ptr<LeafNodeT> leaf;

        for (Index n = range.begin(), N = range.end(); n < N; ++n) {
            // explicitly only process active tiles when streaming
            if (!mask.isOn(n)) continue;
            OPENVDB_ASSERT(!this->isChildMaskOn(parent, n));

            const Coord& ijk = parent.offsetToGlobalCoord(n);
            const ValueT& value = table[n].getValue();

            if (!leaf) {
                leaf.reset(new LeafNodeT(ijk, value, /*active*/true)); // always starts as active
                data = leaf->buffer().data();
                end = data + LeafNodeT::NUM_VOXELS;
                word = &(leaf->getValueMask().template getWord<Index64>(0));
            }
            else {
                leaf->setOrigin(ijk);
                leaf->getValueMask().set(true); // always starts as active
                std::fill(data, end, value);
            }

            kernel(ijk, data, word, size_t(LeafNodeT::NUM_VOXELS));

            bool ison;
            bool constant = leaf->getValueMask().isConstant(ison);
            if (constant) {
                constant = (std::find_if(data + 1, end, [&](const auto& v) { return v != data[0]; }) == end);
            }
            if (!constant) {
                leafs.emplace_back(leaf.release());
            }
            else {
                if (ison != true) {
                    // mask changed, we always have to postpone this update
                    // @todo  could update the value here and the mask in serial
                    tiles.emplace_back(n, data[0], ison);
                }
                else if (data[0] != value) {
                    table[n].setValue(data[0]);
                }
            }
        }
    }

    /// @brief Overload for bool value types
    template <typename NodeT, typename std::enable_if<
        std::is_same<typename NodeT::ValueType, bool>::value ||
        std::is_same<typename NodeT::ValueType, std::string>::value
        >::type* = nullptr>
    void processLeafParent(NodeT& parent,
        const tbb::blocked_range<Index>& range,
        std::vector<LeafNodeT*>& leafs,
        std::vector<Tile>& tiles) const
    {
        // ValueOff iterators should explicitly disable tile streaming
        OPENVDB_ASSERT((!std::is_same<ValueOffIter, IterT>::value));
        using TempBufferT = typename std::conditional<
            std::is_same<std::string, ValueT>::value,
                ax::codegen::String, bool>::type;

        // @todo update to new InternalNode API methods when available
        auto* const table = const_cast<typename NodeT::UnionType*>(parent.getTable());
        const auto& mask = parent.getValueMask();

        VolumeFunctionArguments args(mData, mData.mGrids, *mData.mAttributeRegistry);
        auto kernel = args.bindBufferKernel();

        TempBufferT values[LeafNodeT::NUM_VOXELS];
        TempBufferT* data = values, *end = &values[LeafNodeT::NUM_VOXELS];
        typename LeafNodeT::NodeMaskType leafmask;
        Index64* word = &(leafmask.template getWord<Index64>(0));
        Coord ijk;
        bool reset = true;

        for (Index n = range.begin(), N = range.end(); n < N; ++n) {
            // explicitly only process active tiles when streaming
            if (!mask.isOn(n)) continue;
            OPENVDB_ASSERT(!this->isChildMaskOn(parent, n));

            const Coord& ijk = parent.offsetToGlobalCoord(n);
            const TempBufferT value = table[n].getValue();

            if (reset) {
                std::fill(data, end, value);
                leafmask.set(true); // always starts as active
            }

            kernel(ijk, data, word, size_t(LeafNodeT::NUM_VOXELS));

            bool ison;
            bool constant = leafmask.isConstant(ison);
            if (constant) {
                constant = (std::find_if(data + 1, end, [&](const auto& v) { return v != data[0]; }) == end);
            }

            if (!constant) {
                std::unique_ptr<LeafNodeT> leaf(new LeafNodeT(ijk));
                leaf->getValueMask() = leafmask;
                for (Index i = 0; i < LeafNodeT::NUM_VOXELS; ++i) {
                    leaf->buffer().setValue(i, ValueT(values[i]));
                }
                leafs.emplace_back(leaf.release());
                reset = true;
            }
            else {
                if (ison != true) {
                    // mask changed, we always have to postpone this update
                    // @todo  could update the value here and the mask in serial
                    tiles.emplace_back(n, ValueT(data[0]), ison);
                    reset = true;
                }
                else if (data[0] != value) {
                    table[n].setValue(ValueT(data[0]));
                    reset = true;
                }
                else {
                    reset = false;
                }
            }
        }
    }

private:
    const OpData& mData;
    TreeT& mTree;
};

/// @brief  register volumes, create missing and build a GridCache which
///   stores read/write grids with deep copied ownership where necessary
std::unique_ptr<GridCache>
registerVolumes(GridPtrVec& grids,
    const AttributeRegistry& registry,
    const AttributeBindings& bindings,
    const bool createMissing,
    Logger& logger)
{
    std::unique_ptr<GridCache> cache(new GridCache);

    for (auto& iter : registry.data()) {

        openvdb::GridBase* matchedGrid = nullptr;
        bool matchedName = false;
        ast::tokens::CoreType type = ast::tokens::UNKNOWN;
        const std::string& iterName = iter.name();
        const std::string* volumeNamePtr = nullptr;
        volumeNamePtr = bindings.isBoundAXName(iterName) ?  bindings.dataNameBoundTo(iterName) : &iterName;
        OPENVDB_ASSERT(volumeNamePtr);
        const std::string& volumeName = *volumeNamePtr;
        for (const auto& grid : grids) {
            if (grid->getName() != volumeName) continue;
            matchedName = true;
            type = ast::tokens::tokenFromTypeString(grid->valueType());
            if (type != iter.type()) continue;
            matchedGrid = grid.get();
            break;
        }

        if (createMissing && !matchedGrid) {
            auto created = createGrid(iter.type());
            if (created) {
                created->setName(volumeName);
                grids.emplace_back(created);
                type = iter.type();
                matchedName = true;
                matchedGrid = created.get();
            }
        }

        if (!matchedName && !matchedGrid) {
            logger.error("Missing grid \"" + ast::tokens::typeStringFromToken(iter.type()) +
                "@" + iter.name() + "\".");
        }
        else if (matchedName && !matchedGrid) {
            logger.error("Mismatching grid access type. \"@" + volumeName +
                (volumeName != iter.name() ? "\" [bound to \"" + iter.name() + "\"]" : "\"") +
                " exists but has been accessed with type \"" +
                ast::tokens::typeStringFromToken(iter.type()) + "\".");
        }

        if (!matchedGrid) continue;

        if (!supported(type)) {
            logger.error("Could not register volume '" + matchedGrid->getName() +
                "' as it has an unknown or unsupported value type '" +
                matchedGrid->valueType() + "'");
            continue;
        }

        // Populate the write/read grids based on the access registry. If a
        // grid is being written to and has non self usage, (influences
        // another grids value which isn't it's own) it must be deep copied

        // @todo implement better execution order detection which could minimize
        // the number of deep copies required

        if (iter.writes() && iter.affectsothers()) {
            // if affectsothers(), it's also read from at some point
            OPENVDB_ASSERT(iter.reads());
            cache->addReadGrid(*matchedGrid, /*copy=*/true);
            cache->addWriteGrid(*matchedGrid);
        }
        else {
            if (iter.writes()) {
                cache->addWriteGrid(*matchedGrid);
            }
            cache->addReadGrid(*matchedGrid, /*copy=*/false);
        }
    }

    return cache;
}

template <class IterT, typename GridT>
inline void run(GridT& grid, OpData& data, const VolumeExecutable& E)
{
    using TreeType = typename GridT::TreeType;

    // Get the active index of the grid being executed
    const ast::tokens::CoreType type =
        ast::tokens::tokenFromTypeString(grid.valueType());
    OPENVDB_ASSERT(data.mActiveIndex >= 0);

    // Set the active tile streaming behaviour for this grid if
    // the behaviour is set to AUTO (otherwise it's assigned the
    // same value)
    const bool cache = data.mActiveTileStreaming;
    // data.mActiveTileStreaming might be OFF if IterT is an OFF iterator
    data.mActiveTileStreaming &=
        (E.getActiveTileStreaming(grid.getName(), type) == VolumeExecutable::Streaming::ON);

    VolumeExecuterOp<TreeType, IterT> exec(data, grid.tree());

    // If either grain size is > 0, enable overlapping task execution and other
    // multi threaded operations (such as leaf/node manager constructions)
    const size_t g1 = E.getGrainSize();
    const size_t g2 = E.getActiveTileStreamingGrainSize();
    const bool threadOtherOps = g1 > 0 || g2 > 0;
    OPENVDB_ASSERT(data.mTreeLevelMin <= data.mTreeLevelMax);

    // Cache any existing leaf node pointers before doing any execution
    std::unique_ptr<tree::LeafManager<TreeType>> leafManager;
    if (data.mTreeLevelMin == 0) {
        leafManager.reset(new tree::LeafManager<TreeType>
            (grid.tree(), /*aux=*/0, /*serial=*/!threadOtherOps));
    }

    tbb::task_group tasks;

    if (data.mTreeLevelMin == 0) {
        // It's significantly faster to process leaf nodes using a leaf manager
        // so process them first. Additionally, processing tiles can generate
        // new leaf nodes if active tile streaming is enabled and we don't want
        // to process them twice!
        if (threadOtherOps) {
            tasks.run([&]() {
                if (g1 > 0) {
                    tbb::parallel_for(leafManager->leafRange(g1), exec);
                }
                else {
                    exec(leafManager->leafRange());
                }
            });
        }
        else {
            exec(leafManager->leafRange());
        }
    }

    if (data.mTreeLevelMax > 0) {
        // Only cache to TreeType::RootNodeType::LEVEL-1
        using NodeManagerT =
            tree::NodeManager<TreeType, TreeType::RootNodeType::LEVEL-1>;
        if (threadOtherOps) {
            tasks.run([&]() {
                NodeManagerT manager(grid.tree(), /*serial=*/!threadOtherOps);
                if (!data.mActiveTileStreaming) manager.foreachBottomUp(exec, g1 > 0, g1);
                else                         manager.foreachBottomUp(exec, g2 > 0, /*parent-gs=*/1);
            });
        }
        else {
            // If not thread, don't call through to any tbb primitives
            NodeManagerT manager(grid.tree(), /*serial*/true);
            manager.foreachBottomUp(exec, /*thread*/false);
        }
    }

    if (threadOtherOps) tasks.wait();

    data.mActiveTileStreaming = cache;
}

template <class IterT>
inline void run(GridCache& cache,
                const std::unordered_map<std::string, uint64_t>& functions,
                const AttributeRegistry& registry,
                const CustomData* const custom,
                const VolumeExecutable::Settings<false>& S,
                const VolumeExecutable& E,
                Logger& logger)
{
    OPENVDB_ASSERT(cache.mRead.size() == registry.data().size());

    // Initialize the shared op data

    OpData data;
    data.mKernelValueBuffer =
        reinterpret_cast<KernelBufferFunctionPtr>
            (functions.at(codegen::VolumeKernelBuffer::getDefaultName()));
    data.mKernelNode =
        reinterpret_cast<KernelNodeFunctionPtr>
            (functions.at(codegen::VolumeKernelNode::getDefaultName()));
    data.mCustomData = custom;
    data.mAttributeRegistry = &registry;
    data.mGrids = cache.mRead.data();
    data.mTreeLevelMin = S.mTreeExecutionLevel.get().first;
    data.mTreeLevelMax = S.mTreeExecutionLevel.get().second;
    data.mIterMode =
        std::is_same<IterT, ValueOnIter>::value  ? 1 :
        std::is_same<IterT, ValueOffIter>::value ? 0 :
        std::is_same<IterT, ValueAllIter>::value ? 2 : 2;
    data.mTileGrainSize = S.mGrainSizes.get().second;
    // @note If Streaming::AUTO, this value can be temporarily
    // changed by the next invocation of run().
    data.mActiveTileStreaming = ((data.mIterMode == 1 || data.mIterMode == 2) &&
        (S.mActiveTileStreaming.get() != VolumeExecutable::Streaming::OFF));

    openvdb::GridBase** read = cache.mRead.data();
    data.mVoidTransforms.reserve(cache.mRead.size());
    for (size_t i = 0; i < registry.data().size(); ++i, ++read) {
        OPENVDB_ASSERT(read);
        data.mVoidTransforms.emplace_back(static_cast<void*>(&(*read)->transform()));
    }

    size_t regidx = 0, cacheidx = 0;
    for (const auto& attr : registry.data()) {
        if (attr.writes()) {
            const auto& grid = cache.mWrite[cacheidx];
            const bool success = grid->apply<SupportedTypeList>([&](auto& typed) {
                using GridType = typename std::decay<decltype(typed)>::type;
                data.mActiveIndex = regidx;
                run<IterT, GridType>(typed, data, E);
            });
            if (!success) {
                logger.error("Could not write to volume '" + grid->getName()
                    + "' as it has an unknown or unsupported value type '" + grid->valueType()
                    + "'");
            }
            ++cacheidx;
        }
        ++regidx;
    }
}
} // anonymous namespace

VolumeExecutable::VolumeExecutable(const std::shared_ptr<const llvm::LLVMContext>& context,
                    const std::shared_ptr<const llvm::ExecutionEngine>& engine,
                    const AttributeRegistry::ConstPtr& accessRegistry,
                    const CustomData::ConstPtr& customData,
                    const std::unordered_map<std::string, uint64_t>& functionAddresses,
                    const ast::Tree& ast)
    : mContext(context)
    , mExecutionEngine(engine)
    , mAttributeRegistry(accessRegistry)
    , mCustomData(customData)
    , mFunctionAddresses(functionAddresses)
    , mSettings(new Settings<false>)
{
    OPENVDB_ASSERT(mContext);
    OPENVDB_ASSERT(mExecutionEngine);
    OPENVDB_ASSERT(mAttributeRegistry);

    // Determine if this kernel needs automatic streaming

    // @todo  These functions need to be catalogues in the same way as we
    //  catalogue attribute dependencies as streaming only needs to be enabled
    //  for grids which are directly affected by them.
    bool stream = ast::callsFunction(ast, "getvoxelpws") ||
        ast::callsFunction(ast, "getcoord") ||
        ast::callsFunction(ast, "getcoordx") ||
        ast::callsFunction(ast, "getcoordy") ||
        ast::callsFunction(ast, "getcoordz") ||
        ast::callsFunction(ast, "rand");

    if (stream) {
        mSettings->mActiveTileStreaming = Streaming::ON;
    }
    else {
        for (const auto& data : mAttributeRegistry->data()) {
            stream = data.affectsothers();
            if (stream) break;
        }
        if (stream) mSettings->mActiveTileStreaming = Streaming::AUTO;
        else        mSettings->mActiveTileStreaming = Streaming::OFF;
    }

    // Set up the default attribute bindings
     for (const auto& iter : mAttributeRegistry->data()) {
        mSettings->mBindings.get().set(iter.name(), iter.name());
    }
}

VolumeExecutable::VolumeExecutable(const VolumeExecutable& other)
    : mContext(other.mContext)
    , mExecutionEngine(other.mExecutionEngine)
    , mAttributeRegistry(other.mAttributeRegistry)
    , mCustomData(other.mCustomData)
    , mFunctionAddresses(other.mFunctionAddresses)
    , mSettings(new Settings<false>(*other.mSettings)) {}

VolumeExecutable::~VolumeExecutable() {}

void VolumeExecutable::execute(openvdb::GridPtrVec& grids) const
{
    Logger* logger;
    std::unique_ptr<Logger> log;
    if (true) {
        /// @note  This branch exists for forwards compatibility with upcoming
        ///   changes to allow a logger to be provided to the executables
        log.reset(new Logger([](const std::string& error) {
            OPENVDB_THROW(AXExecutionError, error);
        }));
        logger = log.get();
    }

    std::unique_ptr<GridCache> cache =
        registerVolumes(grids, *mAttributeRegistry, mSettings->mBindings.get(), mSettings->mCreateMissing, *logger);

    if (logger->hasError()) return;

    if (mSettings->mValueIterator.get() == IterType::ON) {
        run<ValueOnIter>(*cache, mFunctionAddresses, *mAttributeRegistry, mCustomData.get(), *mSettings, *this, *logger);
    }
    else if (mSettings->mValueIterator.get() == IterType::OFF) {
        run<ValueOffIter>(*cache, mFunctionAddresses, *mAttributeRegistry, mCustomData.get(), *mSettings, *this, *logger);
    }
    else if (mSettings->mValueIterator.get() == IterType::ALL) {
        run<ValueAllIter>(*cache, mFunctionAddresses, *mAttributeRegistry, mCustomData.get(), *mSettings, *this, *logger);
    }
    else {
        OPENVDB_ASSERT(false && "Unrecognised voxel iterator.");
    }
}

void VolumeExecutable::execute(openvdb::GridBase& grid) const
{
    // wrap the input with a no delete shared pointer so we can call the main
    // invocation. execute is guaranteed to not take ownership of the inputs
    // (ideally GridPtrVec would not be a vector of shared pointers)
    GridBase::Ptr gridptr(&grid, [](GridBase*){});
    std::vector<GridBase::Ptr> grids { gridptr };
    this->execute(grids);
}


/////////////////////////////////////////////
/////////////////////////////////////////////


void VolumeExecutable::setCreateMissing(const bool flag)
{
    mSettings->mCreateMissing = flag;
}

bool VolumeExecutable::getCreateMissing() const
{
    return mSettings->mCreateMissing;
}

void VolumeExecutable::setTreeExecutionLevel(const Index level)
{
    this->setTreeExecutionLevel(level, level);
}

void VolumeExecutable::setTreeExecutionLevel(const Index min, const Index max)
{
    // use the default implementation of FloatTree for reference
    if (min >= FloatTree::DEPTH || max >= FloatTree::DEPTH || min > max) {
        OPENVDB_THROW(RuntimeError,
            "Invalid tree execution level in VolumeExecutable.");
    }
    mSettings->mTreeExecutionLevel.set({ min, max});
}

void VolumeExecutable::getTreeExecutionLevel(Index& min, Index& max) const
{
    min = mSettings->mTreeExecutionLevel.get().first;
    max = mSettings->mTreeExecutionLevel.get().second;
}

void VolumeExecutable::setActiveTileStreaming(const Streaming& s)
{
    mSettings->mActiveTileStreaming = s;
}

VolumeExecutable::Streaming VolumeExecutable::getActiveTileStreaming() const
{
    return mSettings->mActiveTileStreaming;
}

VolumeExecutable::Streaming
VolumeExecutable::getActiveTileStreaming(const std::string& name,
                     const ast::tokens::CoreType& type) const
{
    OPENVDB_ASSERT(mAttributeRegistry);
    if (mSettings->mActiveTileStreaming.get() == VolumeExecutable::Streaming::AUTO) {
        const ax::AttributeRegistry::AccessData* accessData =
            mAttributeRegistry->get(name, type);
        if (!accessData) return this->getActiveTileStreaming();
        // If this grids access depends on other grids, we must stream it.
        // if it only has one access and this is its own access, streaming
        // is off.
        if (accessData->deps().size() == 0) return Streaming::OFF;
        if (accessData->deps().size() == 1) {
            if (accessData->dependson(accessData)) return Streaming::OFF;
            else                                   return Streaming::ON;
        }
        else {
            return Streaming::ON;
        }
    }
    else {
        return this->getActiveTileStreaming();
    }
}

void VolumeExecutable::setValueIterator(const VolumeExecutable::IterType& iter)
{
    mSettings->mValueIterator = iter;
}

VolumeExecutable::IterType VolumeExecutable::getValueIterator() const
{
    return mSettings->mValueIterator;
}

void VolumeExecutable::setGrainSize(const size_t grain)
{
    mSettings->mGrainSizes.get().first = grain;
}

size_t VolumeExecutable::getGrainSize() const
{
    return mSettings->mGrainSizes.get().first;
}

void VolumeExecutable::setActiveTileStreamingGrainSize(const size_t grain)
{
    mSettings->mGrainSizes.get().second = grain;
}

size_t VolumeExecutable::getActiveTileStreamingGrainSize() const
{
    return mSettings->mGrainSizes.get().second;
}

Index VolumeExecutable::getTreeExecutionLevel() const
{
    return mSettings->mTreeExecutionLevel.get().first;
}

void VolumeExecutable::setAttributeBindings(const AttributeBindings& bindings)
{
    for (const auto& binding : bindings.axToDataMap()) {
        mSettings->mBindings.get().set(binding.first, binding.second);
    }
    // check the registry to make sure everything is bound
    for (const auto& access : mAttributeRegistry->data()) {
        if (!mSettings->mBindings.get().isBoundAXName(access.name())) {
            if (bindings.isBoundDataName(access.name())) {
                OPENVDB_THROW(AXExecutionError, "AX attribute \"@"
                    + access.name() + "\" not bound to any volume."
                    " Volume \"" + access.name() + "\" bound to \"@"
                    + *bindings.axNameBoundTo(access.name()) + "\".");
            }
            else {
                // rebind to itself as it may have been unbound
                // by a previous set call
                mSettings->mBindings.get().set(access.name(), access.name());
            }
        }
    }
}

const AttributeBindings& VolumeExecutable::getAttributeBindings() const
{
    return mSettings->mBindings.get();
}


////////////////////////////////////////////////////////
////////////////////////////////////////////////////////


VolumeExecutable::CLI::CLI()
    : mSettings(new VolumeExecutable::Settings<true>) {}
VolumeExecutable::CLI::~CLI() {}
VolumeExecutable::CLI::CLI(CLI&& other) {
    mSettings = std::move(other.mSettings);
}
VolumeExecutable::CLI& VolumeExecutable::CLI::operator=(CLI&& other) {
    mSettings = std::move(other.mSettings);
    return *this;
}

VolumeExecutable::CLI
VolumeExecutable::CLI::create(size_t argc, const char* argv[], bool* flags)
{
    CLI cli;
    openvdb::ax::cli::init(argc, argv, {}, cli.mSettings->optional(), flags);
    return cli;
}

void VolumeExecutable::CLI::usage(std::ostream& os, const bool verbose)
{
    VolumeExecutable::Settings<true> S;
    for (const auto& P : S.optional()) {
        ax::cli::usage(os, P->opts(), P->doc(), verbose);
    }
}

void VolumeExecutable::setSettingsFromCLI(const VolumeExecutable::CLI& cli)
{
    mSettings->initialize(*cli.mSettings);
}


} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

