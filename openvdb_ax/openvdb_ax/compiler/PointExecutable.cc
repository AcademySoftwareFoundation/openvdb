// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/PointExecutable.cc

#include "PointExecutable.h"
#include "Logger.h"

#include "../ast/Scanners.h"
#include "../Exceptions.h"
// @TODO refactor so we don't have to include PointComputeGenerator.h,
// but still have the functions defined in one place
#include "../codegen/PointComputeGenerator.h"
#include "../codegen/PointLeafLocalData.h"

#include <openvdb/Types.h>

#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h> // ConversionTraits
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointMask.h>
#include <openvdb/points/PointMove.h>
#include <openvdb/points/PointDelete.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

struct PointExecutable::Settings
{
    bool mCreateMissing = true;
    size_t mGrainSize = 1;
    std::string mGroup = "";
    bool mPostDelete = false;
};

namespace {

/// @brief Point Kernel types
///
using KernelFunctionPtr = std::add_pointer<codegen::PointKernel::Signature>::type;
using FunctionTraitsT = codegen::PointKernel::FunctionTraitsT;
using ReturnT = FunctionTraitsT::ReturnType;
using PointLeafLocalData = codegen::codegen_internal::PointLeafLocalData;

/// @brief  The arguments of the generated function
///
struct PointFunctionArguments
{
    using LeafT = points::PointDataTree::LeafNodeType;

    /// @brief  Base untyped handle struct for container storage
    struct Handles
    {
        using UniquePtr = std::unique_ptr<Handles>;
        virtual ~Handles() = default;
    };

    /// @brief  A wrapper around a VDB Points Attribute Handle, allowing for
    ///         typed storage of a read or write handle. This is used for
    ///         automatic memory management and void pointer passing into the
    ///         generated point functions
    template <typename ValueT>
    struct TypedHandle final : public Handles
    {
        using UniquePtr = std::unique_ptr<TypedHandle<ValueT>>;
        using HandleTraits = points::point_conversion_internal::ConversionTraits<ValueT>;
        using HandleT = typename HandleTraits::Handle;

        ~TypedHandle() override final = default;

        inline void*
        initReadHandle(const LeafT& leaf, const size_t pos) {
            mHandle = HandleTraits::handleFromLeaf(leaf, static_cast<Index>(pos));
            return static_cast<void*>(mHandle.get());
        }

        inline void*
        initWriteHandle(LeafT& leaf, const size_t pos) {
            mHandle = HandleTraits::writeHandleFromLeaf(leaf, static_cast<Index>(pos));
            return static_cast<void*>(mHandle.get());
        }

    private:
        typename HandleT::Ptr mHandle;
    };

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    PointFunctionArguments(const KernelFunctionPtr function,
                           const CustomData* const customData,
                           const points::AttributeSet& attributeSet,
                           PointLeafLocalData* const leafLocalData)
        : mFunction(function)
        , mCustomData(customData)
        , mAttributeSet(&attributeSet)
        , mVoidAttributeHandles()
        , mAttributeHandles()
        , mVoidGroupHandles()
        , mGroupHandles()
        , mLeafLocalData(leafLocalData) {}

    /// @brief  Given a built version of the function signature, automatically
    ///         bind the current arguments and return a callable function
    ///         which takes no arguments
    inline auto bind()
    {
        return [&](const uint64_t index) -> ReturnT {
            return mFunction(static_cast<FunctionTraitsT::Arg<0>::Type>(mCustomData),
                static_cast<FunctionTraitsT::Arg<1>::Type>(mAttributeSet),
                static_cast<FunctionTraitsT::Arg<2>::Type>(index),
                static_cast<FunctionTraitsT::Arg<3>::Type>(mVoidAttributeHandles.data()),
                static_cast<FunctionTraitsT::Arg<4>::Type>(mVoidGroupHandles.data()),
                static_cast<FunctionTraitsT::Arg<5>::Type>(mLeafLocalData));
        };
    }

    template <typename ValueT>
    inline void addHandle(const LeafT& leaf, const size_t pos)
    {
        typename TypedHandle<ValueT>::UniquePtr handle(new TypedHandle<ValueT>());
        mVoidAttributeHandles.emplace_back(handle->initReadHandle(leaf, pos));
        mAttributeHandles.emplace_back(std::move(handle));
    }

    template <typename ValueT>
    inline void addWriteHandle(LeafT& leaf, const size_t pos)
    {
        typename TypedHandle<ValueT>::UniquePtr handle(new TypedHandle<ValueT>());
        mVoidAttributeHandles.emplace_back(handle->initWriteHandle(leaf, pos));
        mAttributeHandles.emplace_back(std::move(handle));
    }

    inline void addGroupHandle(const LeafT& leaf, const std::string& name)
    {
        assert(leaf.attributeSet().descriptor().hasGroup(name));
        mGroupHandles.emplace_back(new points::GroupHandle(leaf.groupHandle(name)));
        mVoidGroupHandles.emplace_back(static_cast<void*>(mGroupHandles.back().get()));
    }

    inline void addGroupWriteHandle(LeafT& leaf, const std::string& name)
    {
        assert(leaf.attributeSet().descriptor().hasGroup(name));
        mGroupHandles.emplace_back(new points::GroupWriteHandle(leaf.groupWriteHandle(name)));
        mVoidGroupHandles.emplace_back(static_cast<void*>(mGroupHandles.back().get()));
    }

    inline void addNullGroupHandle() { mVoidGroupHandles.emplace_back(nullptr); }
    inline void addNullAttribHandle() { mVoidAttributeHandles.emplace_back(nullptr); }

private:
    const KernelFunctionPtr mFunction;
    const CustomData* const mCustomData;
    const points::AttributeSet* const mAttributeSet;
    std::vector<void*> mVoidAttributeHandles;
    std::vector<Handles::UniquePtr> mAttributeHandles;
    std::vector<void*> mVoidGroupHandles;
    std::vector<points::GroupHandle::UniquePtr> mGroupHandles;
    PointLeafLocalData* const mLeafLocalData;
};


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


template<typename FilterT = openvdb::points::NullFilter>
struct PointExecuterDeformer
{
    PointExecuterDeformer(const std::string& positionAttribute,
        const FilterT& filter = FilterT())
        : mFilter(filter)
        , mPws(nullptr)
        , mPositionAttribute(positionAttribute) {}

    PointExecuterDeformer(const PointExecuterDeformer& other)
        : mFilter(other.mFilter)
        , mPws(nullptr)
        , mPositionAttribute(other.mPositionAttribute) {}

    template <typename LeafT>
    void reset(const LeafT& leaf, const size_t)
    {
        mFilter.reset(leaf);
        mPws.reset(new points::AttributeHandle<Vec3f>(leaf.constAttributeArray(mPositionAttribute)));
    }

    template <typename IterT>
    void apply(Vec3d& position, const IterT& iter) const
    {
        if (mFilter.valid(iter)) {
            assert(mPws);
            position = Vec3d(mPws->get(*iter));
        }
    }

    FilterT mFilter;
    points::AttributeHandle<Vec3f>::UniquePtr mPws;
    const std::string& mPositionAttribute;
};


template <typename ValueType>
inline void
addAttributeHandleTyped(PointFunctionArguments& args,
                        openvdb::points::PointDataTree::LeafNodeType& leaf,
                        const std::string& name,
                        const bool write)
{
    const openvdb::points::AttributeSet& attributeSet = leaf.attributeSet();
    const size_t pos = attributeSet.find(name);
    assert(pos != openvdb::points::AttributeSet::INVALID_POS);

    if (write) args.addWriteHandle<ValueType>(leaf, pos);
    else       args.addHandle<ValueType>(leaf, pos);
}

#ifndef NDEBUG
inline bool supported(const ast::tokens::CoreType type)
{
    switch (type) {
        case ast::tokens::BOOL    : return true;
        case ast::tokens::CHAR    : return true;
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
#endif

inline void
addAttributeHandle(PointFunctionArguments& args,
                   openvdb::points::PointDataTree::LeafNodeType& leaf,
                   const std::string& name,
                   const ast::tokens::CoreType type,
                   const bool write)
{
    // assert so the executer can be marked as noexcept (assuming nothing throws in compute)
    assert(supported(type) && "Could not retrieve attribute handle from unsupported type");
    switch (type) {
        case ast::tokens::BOOL    : return addAttributeHandleTyped<bool>(args, leaf, name, write);
        case ast::tokens::CHAR    : return addAttributeHandleTyped<char>(args, leaf, name, write);
        case ast::tokens::INT16   : return addAttributeHandleTyped<int16_t>(args, leaf, name, write);
        case ast::tokens::INT32   : return addAttributeHandleTyped<int32_t>(args, leaf, name, write);
        case ast::tokens::INT64   : return addAttributeHandleTyped<int64_t>(args, leaf, name, write);
        case ast::tokens::FLOAT   : return addAttributeHandleTyped<float>(args, leaf, name, write);
        case ast::tokens::DOUBLE  : return addAttributeHandleTyped<double>(args, leaf, name, write);
        case ast::tokens::VEC2I   : return addAttributeHandleTyped<math::Vec2<int32_t>>(args, leaf, name, write);
        case ast::tokens::VEC2F   : return addAttributeHandleTyped<math::Vec2<float>>(args, leaf, name, write);
        case ast::tokens::VEC2D   : return addAttributeHandleTyped<math::Vec2<double>>(args, leaf, name, write);
        case ast::tokens::VEC3I   : return addAttributeHandleTyped<math::Vec3<int32_t>>(args, leaf, name, write);
        case ast::tokens::VEC3F   : return addAttributeHandleTyped<math::Vec3<float>>(args, leaf, name, write);
        case ast::tokens::VEC3D   : return addAttributeHandleTyped<math::Vec3<double>>(args, leaf, name, write);
        case ast::tokens::VEC4I   : return addAttributeHandleTyped<math::Vec4<int32_t>>(args, leaf, name, write);
        case ast::tokens::VEC4F   : return addAttributeHandleTyped<math::Vec4<float>>(args, leaf, name, write);
        case ast::tokens::VEC4D   : return addAttributeHandleTyped<math::Vec4<double>>(args, leaf, name, write);
        case ast::tokens::MAT3F   : return addAttributeHandleTyped<math::Mat3<float>>(args, leaf, name, write);
        case ast::tokens::MAT3D   : return addAttributeHandleTyped<math::Mat3<double>>(args, leaf, name, write);
        case ast::tokens::MAT4F   : return addAttributeHandleTyped<math::Mat4<float>>(args, leaf, name, write);
        case ast::tokens::MAT4D   : return addAttributeHandleTyped<math::Mat4<double>>(args, leaf, name, write);
        case ast::tokens::STRING  : return addAttributeHandleTyped<std::string>(args, leaf, name, write);
        case ast::tokens::UNKNOWN :
        default                   : return;
    }
}

/// @brief  VDB Points executer for a compiled function pointer
struct PointExecuterOp
{
    using LeafManagerT = openvdb::tree::LeafManager<openvdb::points::PointDataTree>;
    using LeafNode = openvdb::points::PointDataTree::LeafNodeType;

    using Descriptor = openvdb::points::AttributeSet::Descriptor;
    using GroupFilter = openvdb::points::GroupFilter;
    using GroupIndex = Descriptor::GroupIndex;

    PointExecuterOp(const AttributeRegistry& attributeRegistry,
               const CustomData* const customData,
               const KernelFunctionPtr computeFunction,
               const math::Transform& transform,
               const GroupIndex& groupIndex,
               std::vector<PointLeafLocalData::UniquePtr>& leafLocalData,
               const std::string& positionAttribute,
               const std::pair<bool,bool>& positionAccess)
        : mAttributeRegistry(attributeRegistry)
        , mCustomData(customData)
        , mComputeFunction(computeFunction)
        , mTransform(transform)
        , mGroupIndex(groupIndex)
        , mLeafLocalData(leafLocalData)
        , mPositionAttribute(positionAttribute)
        , mPositionAccess(positionAccess) {}

    template<typename FilterT = openvdb::points::NullFilter>
    inline std::unique_ptr<points::AttributeWriteHandle<Vec3f>>
    initPositions(LeafNode& leaf, const FilterT& filter = FilterT()) const
    {
        const points::AttributeHandle<Vec3f>::UniquePtr
            positions(new points::AttributeHandle<Vec3f>(leaf.constAttributeArray("P")));
        std::unique_ptr<points::AttributeWriteHandle<Vec3f>>
            pws(new points::AttributeWriteHandle<Vec3f>(leaf.attributeArray(mPositionAttribute)));

        for (auto iter = leaf.beginIndexAll(filter); iter; ++iter) {
            const Index idx = *iter;
            const openvdb::Vec3f pos = positions->get(idx) + iter.getCoord().asVec3s();
            pws->set(idx, mTransform.indexToWorld(pos));
        }

        return pws;
    }

    void operator()(LeafNode& leaf, size_t idx) const
    {
        const size_t count = leaf.getLastValue();
        const points::AttributeSet& set = leaf.attributeSet();
        auto& leafLocalData = mLeafLocalData[idx];
        leafLocalData.reset(new PointLeafLocalData(count));

        PointFunctionArguments args(mComputeFunction, mCustomData, set, leafLocalData.get());

        // add attributes based on the order and existence in the attribute registry
        for (const auto& iter : mAttributeRegistry.data()) {
            const std::string& name = (iter.name() == "P" ? mPositionAttribute : iter.name());
            addAttributeHandle(args, leaf, name, iter.type(), iter.writes());
        }

        // add groups
        const auto& map = set.descriptor().groupMap();
        if (!map.empty()) {
            // add all groups based on their offset within the attribute set - the offset can
            // then be used as a key when retrieving groups from the linearized array, which
            // is provided by the attribute set argument
            std::map<size_t, std::string> orderedGroups;
            for (const auto& iter : map) {
                orderedGroups[iter.second] = iter.first;
            }

            // add a handle at every offset up to and including the max offset. If the
            // offset is not in use, we just use a null pointer as this will never be
            // accessed
            const size_t maxOffset = orderedGroups.crbegin()->first;
            auto iter = orderedGroups.begin();
            for (size_t i = 0; i <= maxOffset; ++i) {
                if (iter->first == i) {
                    args.addGroupWriteHandle(leaf, iter->second);
                    ++iter;
                }
                else {
                    // empty handle at this index
                    args.addNullGroupHandle();
                }
            }
        }

        const bool group = mGroupIndex.first != points::AttributeSet::INVALID_POS;

        // if we are using position we need to initialise the world space storage
        std::unique_ptr<points::AttributeWriteHandle<Vec3f>> pws;
        if (mPositionAccess.first || mPositionAccess.second) {
            if (group) {
                const GroupFilter filter(mGroupIndex);
                pws = this->initPositions(leaf, filter);
            }
            else {
                pws = this->initPositions(leaf);
            }
        }

        const auto run = args.bind();

        if (group) {
            const GroupFilter filter(mGroupIndex);
            auto iter = leaf.beginIndex<LeafNode::ValueAllCIter, GroupFilter>(filter);
            for (; iter; ++iter) run(*iter);
        }
        else {
            // the Compute function performs unsigned integer arithmetic and will wrap
            // if count == 0 inside ComputeGenerator::genComputeFunction()
            if (count > 0) run(count);
        }

        // if not writing to position (i.e. post sorting) collapse the temporary attribute

        if (pws && !mPositionAccess.second) {
            pws->collapse();
            pws.reset();
        }

        // as multiple groups can be stored in a single array, attempt to compact the
        // arrays directly so that we're not trying to call compact multiple times
        // unsuccessfully
        leafLocalData->compact();
    }

    void operator()(const LeafManagerT::LeafRange& range) const
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {
            (*this)(*leaf, leaf.pos());
        }
    }

private:
    const AttributeRegistry&  mAttributeRegistry;
    const CustomData* const   mCustomData;
    const KernelFunctionPtr   mComputeFunction;
    const math::Transform&    mTransform;
    const GroupIndex&         mGroupIndex;
    std::vector<PointLeafLocalData::UniquePtr>& mLeafLocalData;
    const std::string&          mPositionAttribute;
    const std::pair<bool,bool>& mPositionAccess;
};

void processAttributes(points::PointDataGrid& grid,
                       const AttributeRegistry& registry,
                       const bool createMissing,
                       Logger& logger)
{
    auto typePairFromToken =
        [](const ast::tokens::CoreType type) -> NamePair {
        switch (type) {
            case ast::tokens::BOOL    : return points::TypedAttributeArray<bool>::attributeType();
            case ast::tokens::CHAR    : return points::TypedAttributeArray<char>::attributeType();
            case ast::tokens::INT16   : return points::TypedAttributeArray<int16_t>::attributeType();
            case ast::tokens::INT32   : return points::TypedAttributeArray<int32_t>::attributeType();
            case ast::tokens::INT64   : return points::TypedAttributeArray<int64_t>::attributeType();
            case ast::tokens::FLOAT   : return points::TypedAttributeArray<float>::attributeType();
            case ast::tokens::DOUBLE  : return points::TypedAttributeArray<double>::attributeType();
            case ast::tokens::VEC2I   : return points::TypedAttributeArray<math::Vec2<int32_t>>::attributeType();
            case ast::tokens::VEC2F   : return points::TypedAttributeArray<math::Vec2<float>>::attributeType();
            case ast::tokens::VEC2D   : return points::TypedAttributeArray<math::Vec2<double>>::attributeType();
            case ast::tokens::VEC3I   : return points::TypedAttributeArray<math::Vec3<int32_t>>::attributeType();
            case ast::tokens::VEC3F   : return points::TypedAttributeArray<math::Vec3<float>>::attributeType();
            case ast::tokens::VEC3D   : return points::TypedAttributeArray<math::Vec3<double>>::attributeType();
            case ast::tokens::VEC4I   : return points::TypedAttributeArray<math::Vec4<int32_t>>::attributeType();
            case ast::tokens::VEC4F   : return points::TypedAttributeArray<math::Vec4<float>>::attributeType();
            case ast::tokens::VEC4D   : return points::TypedAttributeArray<math::Vec4<double>>::attributeType();
            case ast::tokens::MAT3F   : return points::TypedAttributeArray<math::Mat3<float>>::attributeType();
            case ast::tokens::MAT3D   : return points::TypedAttributeArray<math::Mat3<double>>::attributeType();
            case ast::tokens::MAT4F   : return points::TypedAttributeArray<math::Mat4<float>>::attributeType();
            case ast::tokens::MAT4D   : return points::TypedAttributeArray<math::Mat4<double>>::attributeType();
            case ast::tokens::STRING  : return points::StringAttributeArray::attributeType();
            case ast::tokens::UNKNOWN :
            default      : {
                return NamePair();
            }
        }
    };

    const auto leafIter = grid.tree().cbeginLeaf();
    assert(leafIter);

    // append attributes

    for (const auto& iter : registry.data()) {

        const std::string& name = iter.name();
        const points::AttributeSet::Descriptor& desc = leafIter->attributeSet().descriptor();
        const size_t pos = desc.find(name);

        if (!createMissing && pos == points::AttributeSet::INVALID_POS) {
            logger.error("Attribute \"" + name + "\" does not exist on grid \"" + grid.getName() + "\"");
            continue;
        }

        if (pos != points::AttributeSet::INVALID_POS) {
            const points::AttributeArray* const array = leafIter->attributeSet().getConst(pos);
            assert(array);
            if (array->stride() > 1) {
                logger.warning("Attribute \"" + name + "\" on grid \"" + grid.getName() + "\" "
                    "is a strided (array) attribute. Reading or writing to this attribute with @" +
                    name + " will only access the first element.");
            }

            const NamePair& type = desc.type(pos);
            const ast::tokens::CoreType typetoken =
                ast::tokens::tokenFromTypeString(type.first);

            if (typetoken != iter.type() &&
                !(type.second == "str" && iter.type() == ast::tokens::STRING)) {
                logger.error("Mismatching attributes types. Attribute \"" + name +
                    "\" on grid \"" + grid.getName() + "\" exists of type \"" + type.first +
                    "\" but has been accessed with type \"" +
                    ast::tokens::typeStringFromToken(iter.type()) + "\"");
            }
            continue;
        }

        assert(supported(iter.type()));
        const NamePair type = typePairFromToken(iter.type());
        points::appendAttribute(grid.tree(), name, type);
    }
}

} // anonymous namespace

PointExecutable::PointExecutable(const std::shared_ptr<const llvm::LLVMContext>& context,
                const std::shared_ptr<const llvm::ExecutionEngine>& engine,
                const AttributeRegistry::ConstPtr& attributeRegistry,
                const CustomData::ConstPtr& customData,
                const std::unordered_map<std::string, uint64_t>& functions,
                const ast::Tree& ast)
    : mContext(context)
    , mExecutionEngine(engine)
    , mAttributeRegistry(attributeRegistry)
    , mCustomData(customData)
    , mFunctionAddresses(functions)
    , mSettings(new Settings)
{
    assert(mContext);
    assert(mExecutionEngine);
    assert(mAttributeRegistry);

    // parse the AST for known functions which require pre/post processing
    mSettings->mPostDelete = ast::callsFunction(ast, "deletepoint");
}

PointExecutable::PointExecutable(const PointExecutable& other)
    : mContext(other.mContext)
    , mExecutionEngine(other.mExecutionEngine)
    , mAttributeRegistry(other.mAttributeRegistry)
    , mCustomData(other.mCustomData)
    , mFunctionAddresses(other.mFunctionAddresses)
    , mSettings(new Settings(*other.mSettings)) {}

PointExecutable::~PointExecutable() {}

void PointExecutable::execute(openvdb::points::PointDataGrid& grid) const
{
    using LeafManagerT = openvdb::tree::LeafManager<openvdb::points::PointDataTree>;

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

    const auto leafIter = grid.tree().cbeginLeaf();
    if (!leafIter) {
        logger->warning("PointDataGrid \"" + grid.getName() + "\" is empty.");
        return;
    }

    // create any missing attributes
    processAttributes(grid, *mAttributeRegistry, mSettings->mCreateMissing, *logger);

    const std::pair<bool,bool> positionAccess =
        mAttributeRegistry->accessPattern("P", ast::tokens::VEC3F);
    const bool usingPosition = positionAccess.first || positionAccess.second;

    // create temporary world space position attribute if P is being accessed
    // @todo  should avoid actually adding this attribute to the tree as its temporary

    std::string positionAttribute = "P";
    if (usingPosition /*mAttributeRegistry->isWritable("P", ast::tokens::VEC3F)*/) {
        const points::AttributeSet::Descriptor& desc =
            leafIter->attributeSet().descriptor();
        positionAttribute = desc.uniqueName("__P");
        points::appendAttribute<openvdb::Vec3f>(grid.tree(), positionAttribute);
    }

    // init the internal dead group if necessary

    if (mSettings->mPostDelete) {
        // @todo  This group is hard coded in the deletepoint function call - we
        //  should perhaps instead pass in a unique group string to the point kernel
        if (leafIter->attributeSet().descriptor().hasGroup("__ax_dead")) {
            logger->warning("Input points grid \"" + grid.getName() + "\" contains the "
                "internal point group \"__ax_dead\" which is used for deleting points. "
                "points in this group may also be removed.");
        }
        else {
            points::appendGroup(grid.tree(), "__ax_dead");
        }
    }

    openvdb::points::AttributeSet::Descriptor::GroupIndex groupIndex;
    groupIndex.first = openvdb::points::AttributeSet::INVALID_POS;

    const bool usingGroup = !mSettings->mGroup.empty();
    if (usingGroup) {
        if (!leafIter->attributeSet().descriptor().hasGroup(mSettings->mGroup)) {
            logger->error("Requested point group \"" + mSettings->mGroup +
                "\" on grid \"" + grid.getName() + "\" does not exist.");
        }
        else {
            groupIndex = leafIter->attributeSet().groupIndex(mSettings->mGroup);
        }
    }

    // extract appropriate function pointer
    KernelFunctionPtr compute = nullptr;
    const auto iter = usingGroup ?
        mFunctionAddresses.find(codegen::PointKernel::getDefaultName()) :
        mFunctionAddresses.find(codegen::PointRangeKernel::getDefaultName());
    if (iter != mFunctionAddresses.end()) {
        compute = reinterpret_cast<KernelFunctionPtr>(iter->second);
    }
    if (!compute) {
        logger->error("No AX kernel found for execution.");
    }
    if (logger->hasError()) return;

    const math::Transform& transform = grid.transform();
    LeafManagerT leafManager(grid.tree());
    std::vector<PointLeafLocalData::UniquePtr> leafLocalData(leafManager.leafCount());
    const bool threaded = mSettings->mGrainSize > 0;

    PointExecuterOp executerOp(*mAttributeRegistry,
        mCustomData.get(), compute, transform, groupIndex,
        leafLocalData, positionAttribute, positionAccess);
    leafManager.foreach(executerOp, threaded, mSettings->mGrainSize);

    // Check to see if any new data has been added and apply it accordingly

    std::set<std::string> groups;
    bool newStrings = false;

    {
        points::StringMetaInserter
            inserter(leafIter->attributeSet().descriptorPtr()->getMetadata());
        for (const auto& data : leafLocalData) {
            data->getGroups(groups);
            newStrings |= data->insertNewStrings(inserter);
        }
    }

    // append and copy over newly created groups
    // @todo  We should just be able to steal the arrays and compact
    // groups but the API for this isn't very nice at the moment

    for (const auto& name : groups) {
        points::appendGroup(grid.tree(), name);
    }

    // add new groups and set strings

    leafManager.foreach(
        [&groups, &leafLocalData, newStrings] (auto& leaf, size_t idx) {

            PointLeafLocalData::UniquePtr& data = leafLocalData[idx];

            for (const auto& name : groups) {

                // Attempt to get the group handle out of the leaf local data form this
                // leaf. This may not exist as although all of the unique set are appended
                // to the tree (above), not every leaf may have been directly touched
                // by every new group. Some leaf nodes may not require any bit mask copying

                points::GroupWriteHandle* tmpHandle = data->get(name);
                if (!tmpHandle) continue;

                points::GroupWriteHandle handle = leaf.groupWriteHandle(name);
                if (tmpHandle->isUniform()) {
                    handle.collapse(tmpHandle->get(0));
                }
                else {
                    const openvdb::Index size = tmpHandle->size();
                    for (openvdb::Index i = 0; i < size; ++i) {
                        handle.set(i, tmpHandle->get(i));
                    }
                }
            }

            if (newStrings) {
                const MetaMap& metadata = leaf.attributeSet().descriptor().getMetadata();
                const PointLeafLocalData::StringArrayMap& stringArrayMap = data->getStringArrayMap();

                for (const auto& arrayIter : stringArrayMap) {
                    points::StringAttributeWriteHandle::Ptr handle =
                        points::StringAttributeWriteHandle::create(*(arrayIter.first), metadata);

                    for (const auto& iter : arrayIter.second) {
                        handle->set(static_cast<Index>(iter.first), iter.second);
                    }
                }
            }
    }, threaded, mSettings->mGrainSize);

    if (positionAccess.second) {
        // if position is writable, sort the points
        if (usingGroup) {
            openvdb::points::GroupFilter filter(groupIndex);
            PointExecuterDeformer<openvdb::points::GroupFilter> deformer(positionAttribute, filter);
            openvdb::points::movePoints(grid, deformer);
        }
        else {
            PointExecuterDeformer<> deformer(positionAttribute);
            openvdb::points::movePoints(grid, deformer);
        }
    }

    if (usingPosition) {
        // remove temporary world space storage
        points::dropAttribute(grid.tree(), positionAttribute);
    }

    if (mSettings->mPostDelete) {
        points::deleteFromGroup(grid.tree(), "__ax_dead", false, /*drop=*/true);
    }
}


/////////////////////////////////////////////
/////////////////////////////////////////////


void PointExecutable::setCreateMissing(const bool flag)
{
    mSettings->mCreateMissing = flag;
}

bool PointExecutable::getCreateMissing() const
{
    return mSettings->mCreateMissing;
}

void PointExecutable::setGrainSize(const size_t grain)
{
    mSettings->mGrainSize = grain;
}

size_t PointExecutable::getGrainSize() const
{
    return mSettings->mGrainSize;
}

void PointExecutable::setGroupExecution(const std::string& group)
{
    mSettings->mGroup = group;
}

const std::string& PointExecutable::getGroupExecution() const
{
    return mSettings->mGroup;
}

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

