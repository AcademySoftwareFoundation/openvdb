// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/PointExecutable.cc

#include "cli.h" // from vdb_ax command line tool

#include "PointExecutable.h"
#include "Logger.h"

#include "openvdb_ax/Exceptions.h"
#include "openvdb_ax/ast/Scanners.h"
// @TODO refactor so we don't have to include PointComputeGenerator.h,
// but still have the functions defined in one place
#include "openvdb_ax/codegen/PointComputeGenerator.h"
#include "openvdb_ax/codegen/PointLeafLocalData.h"
#include "openvdb_ax/codegen/Codecs.h"

#include <openvdb/Types.h>
#include <openvdb/util/Assert.h>

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

/// @brief Settings which are stored on the point executer
///   and are configurable by the user.
template <bool IsCLI>
struct PointExecutable::Settings
{
    //IterType mValueIterator = IterType::ON;

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
            &this->mGroup,
            &this->mGrainSize,
            &this->mBindings
        };
        return params;
    }

    inline void init(const PointExecutable::Settings<true>& S)
    {
        if (S.mCreateMissing.isInit())  mCreateMissing = S.mCreateMissing.get();
        if (S.mGroup.isInit())          mGroup = S.mGroup.get();
        if (S.mGrainSize.isInit())      mGrainSize = S.mGrainSize.get();
        if (S.mBindings.isInit())       mBindings = S.mBindings.get();
    }

    ///////////////////////////////////////////////////////////////////////////

    ParamT<bool> mCreateMissing =
        ParamBuilderT<bool>()
            .addOpt("--create-missing [ON|OFF]")
            .setDoc("whether to implicitly point attributes (Default: ON). Attributes are "
                    "created if they are referenced in the AX program but do not exist on the "
                    "input geometry.")
            .setDefault(true)
            .setCB([](bool& v, const char* arg) {
                if (std::strcmp(arg, "ON") == 0)       v = true;
                else if (std::strcmp(arg, "OFF") == 0) v = false;
                else OPENVDB_THROW(CLIError, "Invalid option passed to --create-missing: '" << arg << "'");
            })
            .get();

    ParamT<std::string> mGroup =
        ParamBuilderT<std::string>()
            .addOpt("--group [name]")
            .setDoc("a point group to process. Only points that belong to this group are "
                    "processed by the AX program. Note that this is equivalent to using:\n "
                    "  if (!ingroup(\"name\")) return;\n"
                    "at the start of the AX program.")
            .setDefault("")
            .get();

    ParamT<size_t> mGrainSize =
        ParamBuilderT<size_t>()
            .addOpt("--points-grain [n]")
            .setDoc("the threading grain size for processing nodes (Default: 1). A value of 0 "
                    "disables threading.")
            .setDefault(1)
            .get();

    ParamT<AttributeBindings> mBindings =
        ParamBuilderT<AttributeBindings>()
            .addOpt("--bindings [\"ax_name:point_attr_name,...\"]")
            .setDoc("attribute bindings for points. The argument accepts a quoted string list of "
                    "AX (source code) name to data (vdb attribute) name pairs joined by colons and "
                    "seperated by commas. For example:\n"
                    "  --bindings \"velocity:v,density:s\"\n"
                    "binds velocity AX accesses to a 'v' attribute and density AX accesses to a 's' "
                    "attribute. The following snippet would then alias these attributes:\n"
                    "  v@velocity *= 5;   // actually accesses 'v' points\n"
                    "   @density += 1.0f; // actually accesses 's' points")
            .setDefault(AttributeBindings{})
            .setCB([](AttributeBindings& bindings, const char* c) {
                std::string source, target;
                std::string* active = &source, *other = &target;
                while (*c != '\0') {
                    if (*c == ':') std::swap(active, other);
                    else if (*c == ',') {
                        std::swap(active, other);
                        if (source.empty() || target.empty()) {
                            OPENVDB_THROW(CLIError, "invalid string passed to --bindings. '" << c << "'");
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

    cli::BasicParam<bool> mPostDelete = false;
};

namespace {

/// @brief Point Kernel types
///
using KernelValueFunctionPtr = std::add_pointer<codegen::PointKernelAttributeArray::Signature>::type;
using KernelBufferRangeFunctionPtr = std::add_pointer<codegen::PointKernelBufferRange::Signature>::type;
using PointLeafLocalData = codegen::codegen_internal::PointLeafLocalData;

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

struct PointAttributeInfo
{
    PointAttributeInfo(const std::string& name,
                       const ast::tokens::CoreType type,
                       const bool write)
        : mName(name)
        , mType(type)
        , mWrite(write) {}
    std::string mName;
    ast::tokens::CoreType mType;
    bool mWrite;
};

/// @brief  Shared data for the parallel operator
struct OpData
{
    KernelValueFunctionPtr mKernelAttributeArray;
    KernelBufferRangeFunctionPtr mKernelBufferRange;
    const CustomData* mCustomData;
    const AttributeRegistry* mAttributeRegistry;
    size_t mIterMode; // 0 = OFF, 1 = ON, 2 = ALL
    const math::Transform* mTransform;
    points::AttributeSet::Descriptor::GroupIndex mGroupIndex;
    std::string mPositionAttribute;
    std::pair<bool,bool> mPositionAccess;
    bool mUseBufferKernel;
    std::vector<PointAttributeInfo> mAttributeInfo;
};

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
    struct ReadHandle final : public Handles
    {
        using UniquePtr = std::unique_ptr<ReadHandle<ValueT>>;
        using HandleTraits = points::point_conversion_internal::ConversionTraits<ValueT>;
        using HandleT = typename HandleTraits::Handle;
        ReadHandle(const LeafT& leaf, const Index idx)
            : mHandle(HandleTraits::handleFromLeaf(leaf, idx)) {}
        ~ReadHandle() override final = default;
        const std::unique_ptr<HandleT> mHandle;
    };

    template <typename ValueT>
    struct WriteHandle final : public Handles
    {
        using UniquePtr = std::unique_ptr<WriteHandle<ValueT>>;
        using HandleTraits = points::point_conversion_internal::ConversionTraits<ValueT>;
        using HandleT = typename HandleTraits::WriteHandle;
        WriteHandle(LeafT& leaf, const Index idx)
            : mHandle(HandleTraits::writeHandleFromLeaf(leaf, idx)) {}
        ~WriteHandle() override final = default;
        const std::unique_ptr<HandleT> mHandle;
    };

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    PointFunctionArguments(const OpData& data,
                           LeafT& leaf,
                           PointLeafLocalData* const leafLocalData)
            : mData(data)
            , mAttributeSet(&leaf.attributeSet())
            , mHandlesOrBuffers()
            , mAttributeHandles()
            , mVoidGroupHandles()
            , mGroupHandles()
            , mLeafLocalData(leafLocalData)
        {
            // add attributes based on the order and existence in the attribute registry
            for (const auto& iter : mData.mAttributeInfo) {
                this->addAttributeHandle(leaf, iter.mName, iter.mType, iter.mWrite);
            }

            // add groups
            const auto& map = mAttributeSet->descriptor().groupMap();
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
                        this->addGroupWriteHandle(leaf, iter->second);
                        ++iter;
                    }
                    else {
                        // empty handle at this index
                        this->addNullGroupHandle();
                    }
                }
            }
        }

    inline auto bindValueKernel()
    {
        using FunctionTraitsT = codegen::PointKernelAttributeArray::FunctionTraitsT;
        using ReturnT = FunctionTraitsT::ReturnType;

        return [&](const openvdb::Coord& origin, void* buffer, bool active, const size_t index) -> ReturnT {
            mData.mKernelAttributeArray(static_cast<FunctionTraitsT::Arg<0>::Type>(mData.mCustomData),
                reinterpret_cast<FunctionTraitsT::Arg<1>::Type>(origin.data()),
                static_cast<FunctionTraitsT::Arg<2>::Type>(buffer),
                static_cast<FunctionTraitsT::Arg<3>::Type>(active),
                static_cast<FunctionTraitsT::Arg<4>::Type>(index),
                static_cast<FunctionTraitsT::Arg<5>::Type>(nullptr/*mData.mVoidTransforms.data()*/),
                static_cast<FunctionTraitsT::Arg<6>::Type>(mHandlesOrBuffers.data()),
                static_cast<FunctionTraitsT::Arg<7>::Type>(mFlags.data()),
                static_cast<FunctionTraitsT::Arg<8>::Type>(mAttributeSet),
                static_cast<FunctionTraitsT::Arg<9>::Type>(mVoidGroupHandles.data()),
                static_cast<FunctionTraitsT::Arg<10>::Type>(mLeafLocalData));
        };
    }

    inline auto bindRangeKernel()
    {
        using FunctionTraitsT = codegen::PointKernelBufferRange::FunctionTraitsT;
        using ReturnT = FunctionTraitsT::ReturnType;

        OPENVDB_ASSERT(mData.mUseBufferKernel);

        return [&](const openvdb::Coord& origin, void* buffer, Index64* mask, const size_t size) -> ReturnT {
            mData.mKernelBufferRange(static_cast<FunctionTraitsT::Arg<0>::Type>(mData.mCustomData),
                reinterpret_cast<FunctionTraitsT::Arg<1>::Type>(origin.data()),
                static_cast<FunctionTraitsT::Arg<2>::Type>(buffer),
                static_cast<FunctionTraitsT::Arg<3>::Type>(mask),
                static_cast<FunctionTraitsT::Arg<4>::Type>(size),
                static_cast<FunctionTraitsT::Arg<5>::Type>(2/*mData.mIterMode*/),
                static_cast<FunctionTraitsT::Arg<6>::Type>(nullptr/*mData.mVoidTransforms.data()*/),
                static_cast<FunctionTraitsT::Arg<7>::Type>(mHandlesOrBuffers.data()),
                static_cast<FunctionTraitsT::Arg<8>::Type>(mFlags.data()),
                static_cast<FunctionTraitsT::Arg<9>::Type>(mAttributeSet),
                static_cast<FunctionTraitsT::Arg<10>::Type>(mVoidGroupHandles.data()),
                static_cast<FunctionTraitsT::Arg<11>::Type>(mLeafLocalData));
        };
    }

    template <typename ValueT>
    inline void addHandle(LeafT& leaf, const size_t pos)
    {
        uint64_t flag = 0;
        const points::AttributeArray& array = leaf.constAttributeArray(pos);
        if (array.isUniform()) flag |= uint64_t(1) << 63;

        // @todo  if the array is shared we should probably make it unique?

        if (mData.mUseBufferKernel) {
            const_cast<points::AttributeArray&>(array).loadData();
            const char* data = array.constDataAsByteArray();
            void* ptr = static_cast<void*>(const_cast<char*>(data));
            mHandlesOrBuffers.emplace_back(ptr);
            const codegen::Codec* codec =
                codegen::getCodec(ast::tokens::tokenFromTypeString(array.valueType()), array.codecType());
            if (codec) flag |= codec->flag();
        }
        else {
            typename ReadHandle<ValueT>::UniquePtr handle(new ReadHandle<ValueT>(leaf, Index(pos)));
            mHandlesOrBuffers.emplace_back(handle->mHandle.get());
            mAttributeHandles.emplace_back(std::move(handle));
        }

        mFlags.emplace_back(flag);
    }

    template <typename ValueT>
    inline void addWriteHandle(LeafT& leaf, const size_t pos)
    {
        uint64_t flag = 0;
        points::AttributeArray& array = leaf.attributeArray(pos);
        array.expand();

        if (mData.mUseBufferKernel) {
            array.loadData();
            const char* data = array.constDataAsByteArray();
            void* ptr = static_cast<void*>(const_cast<char*>(data));
            mHandlesOrBuffers.emplace_back(ptr);
            const codegen::Codec* codec =
                codegen::getCodec(ast::tokens::tokenFromTypeString(array.valueType()), array.codecType());
            if (codec) flag |= codec->flag();
            OPENVDB_ASSERT(array.isDataLoaded() && !array.isUniform());
        }
        else {
            typename WriteHandle<ValueT>::UniquePtr handle(new WriteHandle<ValueT>(leaf, Index(pos)));
            mHandlesOrBuffers.emplace_back(handle->mHandle.get());
            mAttributeHandles.emplace_back(std::move(handle));
        }

        mFlags.emplace_back(flag);
    }

    inline void addGroupHandle(const LeafT& leaf, const std::string& name)
    {
        OPENVDB_ASSERT(leaf.attributeSet().descriptor().hasGroup(name));
        mGroupHandles.emplace_back(new points::GroupHandle(leaf.groupHandle(name)));
        mVoidGroupHandles.emplace_back(static_cast<void*>(mGroupHandles.back().get()));
    }

    inline void addGroupWriteHandle(LeafT& leaf, const std::string& name)
    {
        OPENVDB_ASSERT(leaf.attributeSet().descriptor().hasGroup(name));
        mGroupHandles.emplace_back(new points::GroupWriteHandle(leaf.groupWriteHandle(name)));
        mVoidGroupHandles.emplace_back(static_cast<void*>(mGroupHandles.back().get()));
    }

    inline void addNullGroupHandle() { mVoidGroupHandles.emplace_back(nullptr); }

    inline void
    addAttributeHandle(LeafT& leaf, const std::string& name, const ast::tokens::CoreType type, const bool write)
    {
        // assert so the executer can be marked as noexcept (assuming nothing throws in compute)
        OPENVDB_ASSERT(supported(type) && "Could not retrieve attribute handle from unsupported type");
        switch (type) {
            case ast::tokens::BOOL    : return this->addAttributeHandleTyped<bool>(leaf, name, write);
            case ast::tokens::CHAR    : return this->addAttributeHandleTyped<char>(leaf, name, write);
            case ast::tokens::INT16   : return this->addAttributeHandleTyped<int16_t>(leaf, name, write);
            case ast::tokens::INT32   : return this->addAttributeHandleTyped<int32_t>(leaf, name, write);
            case ast::tokens::INT64   : return this->addAttributeHandleTyped<int64_t>(leaf, name, write);
            case ast::tokens::FLOAT   : return this->addAttributeHandleTyped<float>(leaf, name, write);
            case ast::tokens::DOUBLE  : return this->addAttributeHandleTyped<double>(leaf, name, write);
            case ast::tokens::VEC2I   : return this->addAttributeHandleTyped<math::Vec2<int32_t>>(leaf, name, write);
            case ast::tokens::VEC2F   : return this->addAttributeHandleTyped<math::Vec2<float>>(leaf, name, write);
            case ast::tokens::VEC2D   : return this->addAttributeHandleTyped<math::Vec2<double>>(leaf, name, write);
            case ast::tokens::VEC3I   : return this->addAttributeHandleTyped<math::Vec3<int32_t>>(leaf, name, write);
            case ast::tokens::VEC3F   : return this->addAttributeHandleTyped<math::Vec3<float>>(leaf, name, write);
            case ast::tokens::VEC3D   : return this->addAttributeHandleTyped<math::Vec3<double>>(leaf, name, write);
            case ast::tokens::VEC4I   : return this->addAttributeHandleTyped<math::Vec4<int32_t>>(leaf, name, write);
            case ast::tokens::VEC4F   : return this->addAttributeHandleTyped<math::Vec4<float>>(leaf, name, write);
            case ast::tokens::VEC4D   : return this->addAttributeHandleTyped<math::Vec4<double>>(leaf, name, write);
            case ast::tokens::MAT3F   : return this->addAttributeHandleTyped<math::Mat3<float>>(leaf, name, write);
            case ast::tokens::MAT3D   : return this->addAttributeHandleTyped<math::Mat3<double>>(leaf, name, write);
            case ast::tokens::MAT4F   : return this->addAttributeHandleTyped<math::Mat4<float>>(leaf, name, write);
            case ast::tokens::MAT4D   : return this->addAttributeHandleTyped<math::Mat4<double>>(leaf, name, write);
            case ast::tokens::STRING  : return this->addAttributeHandleTyped<std::string>(leaf, name, write);
            case ast::tokens::UNKNOWN :
            default                   : return;
        }
    }

private:
    template <typename ValueType>
    inline void
    addAttributeHandleTyped(LeafT& leaf, const std::string& name, const bool write)
    {
        const size_t pos = leaf.attributeSet().find(name);
        //assert(!leaf.attributeSet().isShared(pos));
        OPENVDB_ASSERT(pos != openvdb::points::AttributeSet::INVALID_POS);
        if (write) this->addWriteHandle<ValueType>(leaf, pos);
        else       this->addHandle<ValueType>(leaf, pos);
    }

private:
    const OpData& mData;
    const points::AttributeSet* const mAttributeSet;
    std::vector<void*> mHandlesOrBuffers;
    std::vector<Handles::UniquePtr> mAttributeHandles;
    std::vector<uint64_t> mFlags;
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
            OPENVDB_ASSERT(mPws);
            position = Vec3d(mPws->get(*iter));
        }
    }

    FilterT mFilter;
    points::AttributeHandle<Vec3f>::UniquePtr mPws;
    const std::string& mPositionAttribute;
};

/// @brief  VDB Points executer for a compiled function pointer
struct PointExecuterOp
{
    using LeafManagerT = openvdb::tree::LeafManager<openvdb::points::PointDataTree>;
    using LeafNode = openvdb::points::PointDataTree::LeafNodeType;
    using GroupFilter = openvdb::points::GroupFilter;

    PointExecuterOp(const OpData& data,
               std::vector<PointLeafLocalData::UniquePtr>& leafLocalData)
        : mData(data)
        , mLeafLocalData(leafLocalData) {}

    template<typename FilterT = openvdb::points::NullFilter>
    inline std::unique_ptr<points::AttributeWriteHandle<Vec3f>>
    initPositions(LeafNode& leaf, const FilterT& filter = FilterT()) const
    {
        points::AttributeHandle<Vec3f> positions(leaf.constAttributeArray("P"));
        std::unique_ptr<points::AttributeWriteHandle<Vec3f>>
            pws(new points::AttributeWriteHandle<Vec3f>(leaf.attributeArray(mData.mPositionAttribute)));

        for (auto iter = leaf.beginIndexAll(filter); iter; ++iter) {
            const Index idx = *iter;
            const openvdb::Vec3f pos = positions.get(idx) + iter.getCoord().asVec3s();
            pws->set(idx, mData.mTransform->indexToWorld(pos));
        }

        return pws;
    }

    void operator()(LeafNode& leaf, size_t idx) const
    {
        auto& leafLocalData = mLeafLocalData[idx];
        leafLocalData.reset(new PointLeafLocalData(leaf.getLastValue()));

        const bool group = mData.mGroupIndex.first != points::AttributeSet::INVALID_POS;

        // if we are using position we need to initialise the world space storage
        std::unique_ptr<points::AttributeWriteHandle<Vec3f>> pws;
        if (mData.mPositionAccess.first || mData.mPositionAccess.second) {
            if (group) {
                const GroupFilter filter(mData.mGroupIndex);
                pws = this->initPositions(leaf, filter);
            }
            else {
                pws = this->initPositions(leaf);
            }
        }

        PointFunctionArguments args(mData, leaf, leafLocalData.get());
        void* buffer = static_cast<void*>(leaf.buffer().data());

        if (group) {
            const auto kernel = args.bindValueKernel();
            const GroupFilter filter(mData.mGroupIndex);
            auto iter = leaf.beginIndex<LeafNode::ValueAllCIter, GroupFilter>(filter);
            for (; iter; ++iter) kernel(leaf.origin(), buffer, /*active*/true, *iter);
        }
        else if (!mData.mUseBufferKernel) {
            const auto kernel = args.bindValueKernel();
            auto iter = leaf.beginIndexAll();
            for (; iter; ++iter) kernel(leaf.origin(), buffer, /*active*/true, *iter);
        }
        else {
            const auto kernel = args.bindRangeKernel();
            Index64* masks = &(leaf.getValueMask().template getWord<Index64>(0));
            kernel(leaf.origin(), buffer, masks, size_t(LeafNode::NUM_VOXELS));
        }

        // if not writing to position (i.e. post sorting) collapse the temporary attribute

        if (pws && !mData.mPositionAccess.second) {
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
    const OpData& mData;
    std::vector<PointLeafLocalData::UniquePtr>& mLeafLocalData;
};

inline NamePair typePairFromToken(const ast::tokens::CoreType type)
{
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
}

void processAttributes(points::PointDataGrid& grid,
                       std::vector<PointAttributeInfo>& attributeInfo,
                       std::pair<bool,bool>& positionAccess,
                       std::string& posWS,
                       const AttributeRegistry& registry,
                       const AttributeBindings& bindings,
                       const bool createMissing,
                       Logger& logger)
{
    const auto leafIter = grid.tree().cbeginLeaf();
    OPENVDB_ASSERT(leafIter);

    attributeInfo.reserve(registry.data().size());

    // append attributes

    for (const auto& iter : registry.data()) {
        // get the corresponding point attributes
        const std::string* nameptr = bindings.dataNameBoundTo(iter.name());
        if (!nameptr) continue;
        const std::string& name = *nameptr;
        const points::AttributeSet::Descriptor& desc = leafIter->attributeSet().descriptor();

        // handle position separately as we need to create temp storage during execution
        // create temporary world space position attribute if P is being accessed
        // @todo  should avoid actually adding this attribute to the tree as its temporary
        if (name == "P") {
            positionAccess = {iter.reads(), iter.writes()};
            posWS = desc.uniqueName("__P");
            points::appendAttribute<openvdb::Vec3f>(grid.tree(), posWS);
            attributeInfo.emplace_back(posWS, ast::tokens::VEC3F, positionAccess.second);
            continue;
        }
        // add the data name to the attribute info
        attributeInfo.emplace_back(name, iter.type(), iter.writes());

        const size_t pos = desc.find(name);
        if (!createMissing && pos == points::AttributeSet::INVALID_POS) {
            logger.error("Attribute \"" + name + "\" does not exist on grid \"" + grid.getName() + "\"."
                         + (name != iter.name() ? "[bound to \"" + iter.name() + " \"]" : ""));
            continue;
        }

        if (pos != points::AttributeSet::INVALID_POS) {
            const points::AttributeArray* const array = leafIter->attributeSet().getConst(pos);
            OPENVDB_ASSERT(array);
            if (array->stride() > 1) {
                logger.warning("Attribute \"" + name + (name != iter.name() ? "\" [bound to \"" + iter.name() + "\"]" : "\"")
                    + " on grid \"" + grid.getName() + "\"is a strided (array) attribute. "
                    "Reading or writing to this attribute with @" + name + " will only access the first element.");
            }

            const NamePair& type = desc.type(pos);
            const ast::tokens::CoreType typetoken =
                ast::tokens::tokenFromTypeString(type.first);

            if (typetoken != iter.type() &&
                !(type.second == "str" && iter.type() == ast::tokens::STRING)) {
                logger.error("Mismatching attributes types. Attribute \"" + name +
                    (name != iter.name() ? "\" [bound to \"" + iter.name() + "\"]" : "\"") +
                    " on grid \"" + grid.getName() + "\" exists of type \"" + type.first +
                    "\" but has been accessed with type \"" +
                    ast::tokens::typeStringFromToken(iter.type()) + "\"");
            }
            continue;
        }

        OPENVDB_ASSERT(supported(iter.type()));
        const NamePair type = typePairFromToken(iter.type());
        points::appendAttribute(grid.tree(), name, type);
    }
}

bool checkCodecs(const points::AttributeSet::Descriptor& desc,
    const AttributeRegistry& registry,
    const AttributeBindings& bindings,
    const std::string& posWS)
{
    for (const auto& iter : registry.data()) {
        // no support for strings yet (we should just use the cbind)
        if (iter.type() == ast::tokens::STRING) return false;

        // if this type has no possible codec, continue
        const auto* codecs = codegen::getTypeSupportedCodecs(iter.type());
        if (!codecs) continue;

        std::string name = iter.name();
        const std::string* nameptr = bindings.dataNameBoundTo(name);
        if (nameptr)          name = *nameptr;
        else if (name == "P") name = posWS;

        const std::string& codec = desc.type(desc.find(name)).second;

        // null codecs can be ignored
        if (codec == std::string(points::NullCodec::name())) continue;

        // unknown codec in use, fall back to cbinding
        if (codecs->find(codec) == codecs->cend()) return false;
    }

    return true;
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
    , mSettings(new Settings<false>)
{
    OPENVDB_ASSERT(mContext);
    OPENVDB_ASSERT(mExecutionEngine);
    OPENVDB_ASSERT(mAttributeRegistry);

    // parse the AST for known functions which require pre/post processing
    mSettings->mPostDelete = ast::callsFunction(ast, "deletepoint");

    // Set up the default attribute bindings
    for (const auto& iter : mAttributeRegistry->data()) {
        mSettings->mBindings.get().set(iter.name(), iter.name());
    }
}

PointExecutable::PointExecutable(const PointExecutable& other)
    : mContext(other.mContext)
    , mExecutionEngine(other.mExecutionEngine)
    , mAttributeRegistry(other.mAttributeRegistry)
    , mCustomData(other.mCustomData)
    , mFunctionAddresses(other.mFunctionAddresses)
    , mSettings(new Settings<false>(*other.mSettings)) {}

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

    // Initialize the shared op data
    OpData data;

    // create any missing attributes and handle temp ws position storage
    data.mPositionAccess = {false, false};
    processAttributes(grid, data.mAttributeInfo, data.mPositionAccess, data.mPositionAttribute,
                      *mAttributeRegistry, mSettings->mBindings.get(),
                      mSettings->mCreateMissing, *logger);

    data.mKernelAttributeArray =
        reinterpret_cast<KernelValueFunctionPtr>
            (mFunctionAddresses.at(codegen::PointKernelAttributeArray::getDefaultName()));
    data.mKernelBufferRange =
        reinterpret_cast<KernelBufferRangeFunctionPtr>
            (mFunctionAddresses.at(codegen::PointKernelBufferRange::getDefaultName()));
    data.mTransform = &grid.transform();
    data.mCustomData = mCustomData.get();
    data.mGroupIndex.first = openvdb::points::AttributeSet::INVALID_POS;
    data.mAttributeRegistry = mAttributeRegistry.get();
    //data.mPositionAccess = mAttributeRegistry->accessPattern("P", ast::tokens::VEC3F);

    // init the internal dead group if necessary

    if (mSettings->mPostDelete.get()) {
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

    const bool usingGroup = !mSettings->mGroup.get().empty();
    if (usingGroup) {
        if (!leafIter->attributeSet().descriptor().hasGroup(mSettings->mGroup.get())) {
            logger->error("Requested point group \"" + mSettings->mGroup.get() +
                "\" on grid \"" + grid.getName() + "\" does not exist.");
        }
        else {
            data.mGroupIndex =
                leafIter->attributeSet().groupIndex(mSettings->mGroup.get());
        }
    }

    if (logger->hasError()) return;

    // Compute whether we can use the accelerated kernel
    // @note  Assumes attributes are valid (i.e. has errored out if they are not)

    if (!usingGroup) {
        const auto& desc = leafIter->attributeSet().descriptor();
        data.mUseBufferKernel = checkCodecs(desc, *mAttributeRegistry,
            mSettings->mBindings.get(),
            data.mPositionAttribute);
    }
    else {
        // if a group has been specified we can't use the buffer range yet
        data.mUseBufferKernel = false;
    }

    // execute

    LeafManagerT leafManager(grid.tree());
    std::vector<PointLeafLocalData::UniquePtr> leafLocalData(leafManager.leafCount());
    const bool threaded = mSettings->mGrainSize.get() > 0;

    PointExecuterOp executerOp(data, leafLocalData);
    leafManager.foreach(executerOp, threaded, mSettings->mGrainSize.get());

    // Check to see if any new data has been added and apply it accordingly

    std::set<std::string> groups;
    bool newStrings = false;

    {
        points::StringMetaInserter
            inserter(leafIter->attributeSet().descriptorPtr()->getMetadata());
        for (const auto& leafData : leafLocalData) {
            leafData->getGroups(groups);
            newStrings |= leafData->insertNewStrings(inserter);
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

            PointLeafLocalData::UniquePtr& leafData = leafLocalData[idx];

            for (const auto& name : groups) {

                // Attempt to get the group handle out of the leaf local data form this
                // leaf. This may not exist as although all of the unique set are appended
                // to the tree (above), not every leaf may have been directly touched
                // by every new group. Some leaf nodes may not require any bit mask copying

                points::GroupWriteHandle* tmpHandle = leafData->get(name);
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
                const PointLeafLocalData::StringArrayMap& stringArrayMap = leafData->getStringArrayMap();

                for (const auto& arrayIter : stringArrayMap) {
                    points::StringAttributeWriteHandle::Ptr handle =
                        points::StringAttributeWriteHandle::create(*(arrayIter.first), metadata);

                    for (const auto& iter : arrayIter.second) {
                        handle->set(static_cast<Index>(iter.first), iter.second);
                    }
                }
            }
    }, threaded, mSettings->mGrainSize);

    if (data.mPositionAccess.second) {
        // if position is writable, sort the points
        if (usingGroup) {
            openvdb::points::GroupFilter filter(data.mGroupIndex);
            PointExecuterDeformer<openvdb::points::GroupFilter> deformer(data.mPositionAttribute, filter);
            openvdb::points::movePoints(grid, deformer);
        }
        else {
            PointExecuterDeformer<> deformer(data.mPositionAttribute);
            openvdb::points::movePoints(grid, deformer);
        }
    }

    if (data.mPositionAccess.first || data.mPositionAccess.second) {
        // remove temporary world space storage
        points::dropAttribute(grid.tree(), data.mPositionAttribute);
    }

    if (mSettings->mPostDelete.get()) {
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

// void PointExecutable::setValueIterator(const PointExecutable::IterType& iter)
// {
//     mSettings->mValueIterator = iter;
// }

// PointExecutable::IterType PointExecutable::getValueIterator() const
// {
//     return mSettings->mValueIterator;
// }

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

void PointExecutable::setAttributeBindings(const AttributeBindings& bindings)
{
    //@todo: warn when inferred P, Cd, N etc are bound they default to vec3f
    for (const auto& binding : bindings.axToDataMap()) {
        mSettings->mBindings.get().set(binding.first, binding.second);
    }

    // check the registry to make sure everything is bound
    for (const auto& access : mAttributeRegistry->data()) {
        if (!mSettings->mBindings.get().isBoundAXName(access.name())) {
            if (bindings.isBoundDataName(access.name())) {
                OPENVDB_THROW(AXExecutionError, "AX attribute \"@"
                    + access.name() + "\" not bound to any point attribute."
                    " Point attribute \"" + access.name() + "\" bound to \"@"
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

const AttributeBindings& PointExecutable::getAttributeBindings() const
{
    return mSettings->mBindings.get();
}

//

bool PointExecutable::usesAcceleratedKernel(const points::PointDataTree& tree) const
{
    if (tree.empty()) return true;

    // deep copy the Descriptor

    const auto leafIter = tree.cbeginLeaf();
    points::AttributeSet::Descriptor::Ptr desc(
        new points::AttributeSet::Descriptor(leafIter->attributeSet().descriptor()));

    // add any attributes which don't exist for the codec check

    for (const auto& iter : mAttributeRegistry->data()) {
        // get the corresponding point attributes
        const std::string* nameptr = mSettings->mBindings.get().dataNameBoundTo(iter.name());
        if (!nameptr) continue;
        const std::string& name = *nameptr;
        if (points::AttributeSet::INVALID_POS == desc->find(name)) {
            const NamePair type = typePairFromToken(iter.type());
            desc = desc->duplicateAppend(name, type);
        }
    }

    // If P is being written or read from it is implicitly converted
    // to an uncompressed attribute, so make sure P is not compressed

    desc = desc->duplicateDrop({desc->find("P")});
    desc = desc->duplicateAppend("P", typePairFromToken(ast::tokens::CoreType::VEC3F));

    return checkCodecs(*desc, *mAttributeRegistry,
        mSettings->mBindings.get(), "P");
}


////////////////////////////////////////////////////////
////////////////////////////////////////////////////////


PointExecutable::CLI::CLI()
    : mSettings(new PointExecutable::Settings<true>) {}
PointExecutable::CLI::~CLI() {}
PointExecutable::CLI::CLI(CLI&& other) {
    mSettings = std::move(other.mSettings);
}
PointExecutable::CLI& PointExecutable::CLI::operator=(CLI&& other) {
    mSettings = std::move(other.mSettings);
    return *this;
}

PointExecutable::CLI
PointExecutable::CLI::create(size_t argc, const char* argv[], bool* flags)
{
    CLI cli;
    openvdb::ax::cli::init(argc, argv, {}, cli.mSettings->optional(), flags);
    return cli;
}

void PointExecutable::CLI::usage(std::ostream& os, const bool verbose)
{
    PointExecutable::Settings<true> S;
    for (const auto& P : S.optional()) {
        ax::cli::usage(os, P->opts(), P->doc(), verbose);
    }
}

void PointExecutable::setSettingsFromCLI(const PointExecutable::CLI& cli)
{
    mSettings->init(*cli.mSettings);
}


} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

