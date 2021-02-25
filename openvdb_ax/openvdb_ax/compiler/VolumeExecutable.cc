// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/VolumeExecutable.cc

#include "VolumeExecutable.h"

#include "../Exceptions.h"
// @TODO refactor so we don't have to include VolumeComputeGenerator.h,
// but still have the functions defined in one place
#include "../codegen/VolumeComputeGenerator.h"

#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/NodeManager.h>

#include <tbb/parallel_for.h>

#include <memory>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

struct VolumeExecutable::Settings
{
    Index mTreeExecutionLevel = 0;
    bool mCreateMissing = true;
    IterType mValueIterator = IterType::ON;
    size_t mGrainSize = 1;
};

namespace {

/// @brief Volume Kernel types
///
using KernelFunctionPtr = std::add_pointer<codegen::VolumeKernel::Signature>::type;
using FunctionTraitsT = codegen::VolumeKernel::FunctionTraitsT;
using ReturnT = FunctionTraitsT::ReturnType;

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
        TypedAccessor(TreeT& tree)
            : mAccessor(new tree::ValueAccessor<TreeT>(tree)) {}
        ~TypedAccessor() override final = default;

        inline void* get() const { return static_cast<void*>(mAccessor.get()); }
        const std::unique_ptr<tree::ValueAccessor<TreeT>> mAccessor;
    };

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    VolumeFunctionArguments(const KernelFunctionPtr function,
            const size_t index,
            void* const accessor,
            const CustomData* const customData)
        : mFunction(function)
        , mIdx(index)
        , mAccessor(accessor)
        , mCustomData(customData)
        , mVoidAccessors()
        , mAccessors()
        , mVoidTransforms() {}

    /// @brief  Given a built version of the function signature, automatically
    ///         bind the current arguments and return a callable function
    ///         which takes no arguments
    inline auto bind()
    {
        return [&](const openvdb::Coord& ijk, const openvdb::Vec3f& pos) -> ReturnT {
            return mFunction(static_cast<FunctionTraitsT::Arg<0>::Type>(mCustomData),
                reinterpret_cast<FunctionTraitsT::Arg<1>::Type>(ijk.data()),
                reinterpret_cast<FunctionTraitsT::Arg<2>::Type>(pos.asV()),
                static_cast<FunctionTraitsT::Arg<3>::Type>(mVoidAccessors.data()),
                static_cast<FunctionTraitsT::Arg<4>::Type>(mVoidTransforms.data()),
                static_cast<FunctionTraitsT::Arg<5>::Type>(mIdx),
                mAccessor);
        };
    }

    template <typename TreeT>
    inline void
    addAccessor(TreeT& tree)
    {
        typename TypedAccessor<TreeT>::UniquePtr accessor(new TypedAccessor<TreeT>(tree));
        mVoidAccessors.emplace_back(accessor->get());
        mAccessors.emplace_back(std::move(accessor));
    }

    inline void
    addTransform(math::Transform& transform)
    {
        mVoidTransforms.emplace_back(static_cast<void*>(&transform));
    }

private:
    const KernelFunctionPtr mFunction;
    const size_t mIdx;
    void* const mAccessor;
    const CustomData* const mCustomData;
    std::vector<void*> mVoidAccessors;
    std::vector<Accessors::UniquePtr> mAccessors;
    std::vector<void*> mVoidTransforms;
};

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

inline void
retrieveAccessor(VolumeFunctionArguments& args,
                 openvdb::GridBase* grid,
                 const ast::tokens::CoreType& type)
{
    // assert so the executer can be marked as noexcept (assuming nothing throws in compute)
    assert(supported(type) && "Could not retrieve accessor from unsupported type");
    switch (type) {
        case ast::tokens::BOOL    : { args.addAccessor(static_cast<ConverterT<bool>*>(grid)->tree()); return; }
        case ast::tokens::INT16   : { args.addAccessor(static_cast<ConverterT<int16_t>*>(grid)->tree()); return; }
        case ast::tokens::INT32   : { args.addAccessor(static_cast<ConverterT<int32_t>*>(grid)->tree()); return; }
        case ast::tokens::INT64   : { args.addAccessor(static_cast<ConverterT<int64_t>*>(grid)->tree()); return; }
        case ast::tokens::FLOAT   : { args.addAccessor(static_cast<ConverterT<float>*>(grid)->tree()); return; }
        case ast::tokens::DOUBLE  : { args.addAccessor(static_cast<ConverterT<double>*>(grid)->tree()); return; }
        case ast::tokens::VEC2D   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec2<double>>*>(grid)->tree()); return; }
        case ast::tokens::VEC2F   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec2<float>>*>(grid)->tree()); return; }
        case ast::tokens::VEC2I   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec2<int32_t>>*>(grid)->tree()); return; }
        case ast::tokens::VEC3D   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec3<double>>*>(grid)->tree()); return; }
        case ast::tokens::VEC3F   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec3<float>>*>(grid)->tree()); return; }
        case ast::tokens::VEC3I   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec3<int32_t>>*>(grid)->tree()); return; }
        case ast::tokens::VEC4D   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec4<double>>*>(grid)->tree()); return; }
        case ast::tokens::VEC4F   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec4<float>>*>(grid)->tree()); return; }
        case ast::tokens::VEC4I   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Vec4<int32_t>>*>(grid)->tree()); return; }
        case ast::tokens::MAT3D   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Mat3<double>>*>(grid)->tree()); return; }
        case ast::tokens::MAT3F   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Mat3<float>>*>(grid)->tree()); return; }
        case ast::tokens::MAT4D   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Mat4<double>>*>(grid)->tree()); return; }
        case ast::tokens::MAT4F   : { args.addAccessor(static_cast<ConverterT<openvdb::math::Mat4<float>>*>(grid)->tree()); return; }
        case ast::tokens::STRING  : { args.addAccessor(static_cast<ConverterT<std::string>*>(grid)->tree()); return; }
        case ast::tokens::UNKNOWN :
        default                   : return;
    }
}

inline openvdb::GridBase::Ptr
createGrid(const ast::tokens::CoreType& type)
{
    // assert so the executer can be marked as noexcept (assuming nothing throws in compute)
    assert(supported(type) && "Could not retrieve accessor from unsupported type");
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

template <typename TreeT, typename LeafIterTraitsT>
struct VolumeExecuterOp
{
    using LeafManagerT = tree::LeafManager<TreeT>;
    using LeafRangeT = typename LeafManagerT::LeafRange;

    VolumeExecuterOp(const AttributeRegistry& attributeRegistry,
                     const CustomData* const customData,
                     const math::Transform& assignedVolumeTransform,
                     const KernelFunctionPtr computeFunction,
                     openvdb::GridBase** grids,
                     TreeT& tree,
                     const size_t idx,
                     const Index level)
        : mAttributeRegistry(attributeRegistry)
        , mCustomData(customData)
        , mComputeFunction(computeFunction)
        , mTransform(assignedVolumeTransform)
        , mGrids(grids)
        , mIdx(idx)
        , mTree(tree)
        , mLevel(level) {
            assert(mGrids);
        }

    // For use with a NodeManager
    // @note The enable_if shouldn't be necessary but the partitioner
    //  through TBB refuses to call the LeafRange operator with a
    //   const reference argument.
    template <typename NodeType, typename =
        typename std::enable_if<!std::is_same<NodeType, LeafRangeT>::value>::type>
    void operator()(NodeType& node) const
    {
        // if the current node level does not match, skip
        // @todo only run over the given level, avoid caching other nodes
        assert(node.getLevel() > 0);
        if (node.getLevel() != mLevel) return;
        openvdb::tree::ValueAccessor<TreeT> acc(mTree);
        VolumeFunctionArguments args(mComputeFunction,
            mIdx, static_cast<void*>(&acc), mCustomData);

        openvdb::GridBase** read = mGrids;
        for (const auto& iter : mAttributeRegistry.data()) {
            assert(read);
            retrieveAccessor(args, *read, iter.type());
            args.addTransform((*read)->transform());
            ++read;
        }

        const auto run = args.bind();
        (*this)(node, run);
    }

    // For use with a LeafManager, when the target execution level is 0
    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        openvdb::tree::ValueAccessor<TreeT> acc(mTree);
        VolumeFunctionArguments args(mComputeFunction,
            mIdx, static_cast<void*>(&acc), mCustomData);

        openvdb::GridBase** read = mGrids;
        for (const auto& iter : mAttributeRegistry.data()) {
            assert(read);
            retrieveAccessor(args, *read, iter.type());
            args.addTransform((*read)->transform());
            ++read;
        }

        const auto run = args.bind();
        for (auto leaf = range.begin(); leaf; ++leaf) {
            (*this)(*leaf, run);
        }
    }

    template <typename NodeType, typename FuncT>
    void operator()(NodeType& node, const FuncT& axfunc) const
    {
        using IterT = typename LeafIterTraitsT::template NodeConverter<NodeType>::Type;
        using IterTraitsT = tree::IterTraits<NodeType, IterT>;

        for (auto iter = IterTraitsT::begin(node); iter; ++iter) {
            const openvdb::Coord& coord = iter.getCoord();
            const openvdb::Vec3f& pos = mTransform.indexToWorld(coord);
            axfunc(coord, pos);
        }
    }

private:
    const AttributeRegistry&  mAttributeRegistry;
    const CustomData* const   mCustomData;
    const KernelFunctionPtr   mComputeFunction;
    const math::Transform&    mTransform;
    openvdb::GridBase** const mGrids;
    const size_t mIdx;
    TreeT& mTree;
    const Index mLevel; // only used with NodeManagers
};

void registerVolumes(GridPtrVec& grids,
    GridPtrVec& writeableGrids,
    GridPtrVec& readGrids,
    const AttributeRegistry& registry,
    const bool createMissing)
{
    for (auto& iter : registry.data()) {

        openvdb::GridBase::Ptr matchedGrid;
        bool matchedName(false);
        ast::tokens::CoreType type = ast::tokens::UNKNOWN;

        for (const auto& grid : grids) {
            if (grid->getName() != iter.name()) continue;
            matchedName = true;
            type = ast::tokens::tokenFromTypeString(grid->valueType());
            if (type != iter.type()) continue;
            matchedGrid = grid;
            break;
        }

        if (createMissing && !matchedGrid) {
            matchedGrid = createGrid(iter.type());
            if (matchedGrid) {
                matchedGrid->setName(iter.name());
                grids.emplace_back(matchedGrid);
                matchedName = true;
                type = iter.type();
            }
        }
        if (!matchedName && !matchedGrid) {
            OPENVDB_THROW(AXExecutionError, "Missing grid \"" +
                ast::tokens::typeStringFromToken(iter.type()) + "@" + iter.name() + "\".");
        }
        if (matchedName && !matchedGrid) {
            OPENVDB_THROW(AXExecutionError, "Mismatching grid access type. \"@" + iter.name() +
                "\" exists but has been accessed with type \"" +
                ast::tokens::typeStringFromToken(iter.type()) + "\".");
        }

        assert(matchedGrid);

        if (!supported(type)) {
            OPENVDB_THROW(AXExecutionError, "Could not register volume '"
                + matchedGrid->getName() + "' as it has an unknown or unsupported value type '"
                + matchedGrid->valueType() + "'");
        }

        // Populate the write/read grids based on the access registry. If a
        // grid is being written to and has non self usage, (influences
        // another grids value which isn't it's own) it must be deep copied

        // @todo implement better execution order detection which could minimize
        // the number of deep copies required

        if (iter.writes() && iter.affectsothers()) {
            // if affectsothers(), it's also read from at some point
            assert(iter.reads());
            readGrids.push_back(matchedGrid->deepCopyGrid());
            writeableGrids.push_back(matchedGrid);
        }
        else {
            if (iter.writes()) {
                writeableGrids.push_back(matchedGrid);
            }
            readGrids.push_back(matchedGrid);
        }
    }
}

template<typename LeafT> struct ValueOnIter  { using IterTraitsT = typename tree::IterTraits<LeafT, typename LeafT::ValueOnIter>;  };
template<typename LeafT> struct ValueAllIter { using IterTraitsT = typename tree::IterTraits<LeafT, typename LeafT::ValueAllIter>; };
template<typename LeafT> struct ValueOffIter { using IterTraitsT = typename tree::IterTraits<LeafT, typename LeafT::ValueOffIter>; };

template <template <typename> class IterT, typename GridT>
inline void run(openvdb::GridBase& grid,
    openvdb::GridBase** readptrs,
    const KernelFunctionPtr kernel,
    const AttributeRegistry& registry,
    const CustomData* const custom,
    const VolumeExecutable::Settings& S)
{
    using TreeType = typename GridT::TreeType;
    using IterType = IterT<typename TreeType::LeafNodeType>;

    const ast::tokens::CoreType type =
        ast::tokens::tokenFromTypeString(grid.valueType());
    const int64_t idx = registry.accessIndex(grid.getName(), type);
    assert(idx >= 0);

    GridT& typed = static_cast<GridT&>(grid);
    VolumeExecuterOp<TreeType, typename IterType::IterTraitsT>
        executerOp(registry, custom, grid.transform(),
            kernel, readptrs, typed.tree(), idx, S.mTreeExecutionLevel);

    const bool thread = S.mGrainSize > 0;

    if (S.mTreeExecutionLevel == 0) {
        // execute over the topology of the grid currently being modified.
        tree::LeafManager<TreeType> leafManager(typed.tree());
        if (thread) tbb::parallel_for(leafManager.leafRange(S.mGrainSize), executerOp);
        else        executerOp(leafManager.leafRange());
    }
    else {
        // no leaf nodes
        tree::NodeManager<TreeType, TreeType::RootNodeType::LEVEL-1> manager(typed.tree());
        manager.foreachBottomUp(executerOp, thread, S.mGrainSize);
    }
}

template <template <typename> class IterT>
inline void run(const openvdb::GridPtrVec& writeableGrids,
                const openvdb::GridPtrVec& readGrids,
                const KernelFunctionPtr kernel,
                const AttributeRegistry& registry,
                const CustomData* const custom,
                const VolumeExecutable::Settings& S)
{
    // extract grid pointers from shared pointer container
    assert(readGrids.size() == registry.data().size());
    std::vector<openvdb::GridBase*> readptrs;
    readptrs.reserve(readGrids.size());
    for (auto& grid : readGrids) readptrs.emplace_back(grid.get());

    for (const auto& grid : writeableGrids) {
        const bool success = grid->apply<SupportedTypeList>([&](auto& typed) {
            using GridType = typename std::decay<decltype(typed)>::type;
            run<IterT, GridType>(*grid, readptrs.data(), kernel, registry, custom, S);
        });
        if (!success) {
            OPENVDB_THROW(AXExecutionError, "Could not retrieve volume '" + grid->getName()
                + "' as it has an unknown or unsupported value type '" + grid->valueType()
                + "'");
        }
    }
}
} // anonymous namespace

VolumeExecutable::VolumeExecutable(const std::shared_ptr<const llvm::LLVMContext>& context,
                    const std::shared_ptr<const llvm::ExecutionEngine>& engine,
                    const AttributeRegistry::ConstPtr& accessRegistry,
                    const CustomData::ConstPtr& customData,
                    const std::unordered_map<std::string, uint64_t>& functionAddresses)
    : mContext(context)
    , mExecutionEngine(engine)
    , mAttributeRegistry(accessRegistry)
    , mCustomData(customData)
    , mFunctionAddresses(functionAddresses)
    , mSettings(new Settings)
{
    assert(mContext);
    assert(mExecutionEngine);
    assert(mAttributeRegistry);
}

VolumeExecutable::VolumeExecutable(const VolumeExecutable& other)
    : mContext(other.mContext)
    , mExecutionEngine(other.mExecutionEngine)
    , mAttributeRegistry(other.mAttributeRegistry)
    , mCustomData(other.mCustomData)
    , mFunctionAddresses(other.mFunctionAddresses)
    , mSettings(new Settings(*other.mSettings)) {}

VolumeExecutable::~VolumeExecutable() {}

void VolumeExecutable::execute(openvdb::GridPtrVec& grids) const
{
    openvdb::GridPtrVec readGrids, writeableGrids;

    registerVolumes(grids, writeableGrids, readGrids, *mAttributeRegistry, mSettings->mCreateMissing);

    const auto iter = mFunctionAddresses.find(codegen::VolumeKernel::getDefaultName());
    KernelFunctionPtr kernel = nullptr;
    if (iter != mFunctionAddresses.end()) {
        kernel = reinterpret_cast<KernelFunctionPtr>(iter->second);
    }
    if (kernel == nullptr) {
        OPENVDB_THROW(AXCompilerError,
            "No AX kernel found for execution.");
    }

    if (mSettings->mValueIterator == IterType::ON)
        run<ValueOnIter>(writeableGrids, readGrids, kernel, *mAttributeRegistry, mCustomData.get(), *mSettings);
    else if (mSettings->mValueIterator == IterType::OFF)
        run<ValueOffIter>(writeableGrids, readGrids, kernel, *mAttributeRegistry, mCustomData.get(), *mSettings);
    else if (mSettings->mValueIterator == IterType::ALL)
        run<ValueAllIter>(writeableGrids, readGrids, kernel, *mAttributeRegistry, mCustomData.get(), *mSettings);
    else {
        OPENVDB_THROW(AXExecutionError,
            "Unrecognised voxel iterator.");
    }
}

void VolumeExecutable::execute(openvdb::GridBase& grid) const
{
    const auto data = mAttributeRegistry->data();
    if (data.empty()) return;

    for (auto& iter : mAttributeRegistry->data()) {
        if (grid.getName() != iter.name()) {
            OPENVDB_THROW(LookupError, "Missing grid \"" +
                ast::tokens::typeStringFromToken(iter.type()) + "@" + iter.name() + "\".");
        }
        const ast::tokens::CoreType type = ast::tokens::tokenFromTypeString(grid.valueType());
        if (type != iter.type()) {
            OPENVDB_THROW(TypeError, "Mismatching grid access type. \"@" + iter.name() +
                "\" exists but has been accessed with type \"" +
                ast::tokens::typeStringFromToken(iter.type()) + "\".");
        }
        if (!supported(type)) {
            OPENVDB_THROW(TypeError, "Could not register volume '"
                + grid.getName() + "' as it has an unknown or unsupported value type '"
                + grid.valueType() + "'");
        }
    }
    assert(mAttributeRegistry->data().size() == 1);

    const auto iter = mFunctionAddresses.find(codegen::VolumeKernel::getDefaultName());
    KernelFunctionPtr kernel = nullptr;
    if (iter != mFunctionAddresses.end()) {
        kernel = reinterpret_cast<KernelFunctionPtr>(iter->second);
    }
    if (kernel == nullptr) {
        OPENVDB_THROW(AXCompilerError,
            "No code has been successfully compiled for execution.");
    }

    const bool success = grid.apply<SupportedTypeList>([&](auto& typed) {
        using GridType = typename std::decay<decltype(typed)>::type;
        openvdb::GridBase* grids = &grid;
        if (mSettings->mValueIterator == IterType::ON)
            run<ValueOnIter, GridType>(grid, &grids, kernel, *mAttributeRegistry, mCustomData.get(), *mSettings);
        else if (mSettings->mValueIterator == IterType::OFF)
            run<ValueOffIter, GridType>(grid, &grids, kernel, *mAttributeRegistry, mCustomData.get(), *mSettings);
        else if (mSettings->mValueIterator == IterType::ALL)
            run<ValueAllIter, GridType>(grid, &grids, kernel, *mAttributeRegistry, mCustomData.get(), *mSettings);
        else
            OPENVDB_THROW(AXExecutionError,"Unrecognised voxel iterator.");
    });
    if (!success) {
        OPENVDB_THROW(TypeError, "Could not retrieve volume '" + grid.getName()
            + "' as it has an unknown or unsupported value type '" + grid.valueType()
            + "'");
    }
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
    // use the default implementation of FloatTree for reference
    if (level >= FloatTree::DEPTH) {
        OPENVDB_THROW(RuntimeError,
            "Invalid tree execution level in VolumeExecutable.");
    }
    mSettings->mTreeExecutionLevel = level;
}

Index VolumeExecutable::getTreeExecutionLevel() const
{
    return mSettings->mTreeExecutionLevel;
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
    mSettings->mGrainSize = grain;
}

size_t VolumeExecutable::getGrainSize() const
{
    return mSettings->mGrainSize;
}


} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

