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

/// @file compiler/PointExecutable.cc

#include "PointExecutable.h"

#include <openvdb/Types.h>

#include <openvdb_ax/Exceptions.h>

// @TODO refactor so we don't have to include PointComputeGenerator.h,
// but still have the functions defined in one place
#include <openvdb_ax/codegen/PointComputeGenerator.h>
#include <openvdb_ax/compiler/LeafLocalData.h>

#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h> // ConversionTraits
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointMask.h>
#include <openvdb/points/PointMove.h>

#include <type_traits> // std::enable_if

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

namespace {

/// @brief Point Kernel types
///
using KernelFunctionPtr = std::add_pointer<codegen::PointKernel::Signature>::type;
using FunctionTraitsT = codegen::PointKernel::FunctionTraitsT;
using ReturnT = FunctionTraitsT::ReturnType;

/// @brief  The arguments of the generated function
///
struct PointFunctionArguments
{
    /// @brief  Base untyped handle struct for container storage
    ///
    struct Handles
    {
        using UniquePtr = std::unique_ptr<Handles>;
        virtual ~Handles() = default;
    };

    /// @brief  A wrapper around a VDB Points Attribute Handle, allowing for
    ///         typed storage of a read or write handle. This is used for
    ///         automatic memory management and void pointer passing into the
    ///         generated point functions
    ///
    template <typename ValueT>
    struct TypedHandle final : public Handles
    {
        using UniquePtr = std::unique_ptr<TypedHandle<ValueT>>;
        using HandleTraits = points::point_conversion_internal::ConversionTraits<ValueT>;
        using HandleT = typename HandleTraits::Handle;

        using LeafT = points::PointDataTree::LeafNodeType;

        ~TypedHandle() override final = default;

        inline void*
        initReadHandle(const LeafT& leaf, const size_t pos) {
            mHandle = HandleTraits::handleFromLeaf(const_cast<LeafT&>(leaf), static_cast<Index>(pos));
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


    PointFunctionArguments(const CustomData::ConstPtr& customData,
                           const points::AttributeSet& attributeSet,
                           const size_t pointCount)
        : mCustomData(customData)
        , mAttributeSet(&attributeSet)
        , mIndex(0)
        , mLeafLocalData(new compiler::LeafLocalData(pointCount))
        , mVoidAttributeHandles()
        , mAttributeHandles()
        , mVoidGroupHandles()
        , mGroupHandles() {}

    /// @brief  Given a built version of the function signature, automatically
    ///         bind the current arguments and return a callable function
    ///         which takes no arguments
    ///
    /// @param  function  The fully generated function built from the
    ///                   PointComputeGenerator
    ///
    inline std::function<ReturnT()>
    bind(KernelFunctionPtr function)
    {
        return std::bind(function,
            static_cast<FunctionTraitsT::Arg<0>::Type>(mCustomData.get()),
            static_cast<FunctionTraitsT::Arg<1>::Type>(mAttributeSet),
            static_cast<FunctionTraitsT::Arg<2>::Type>(mIndex),
            static_cast<FunctionTraitsT::Arg<3>::Type>(mVoidAttributeHandles.data()),
            static_cast<FunctionTraitsT::Arg<4>::Type>(mVoidGroupHandles.data()),
            static_cast<FunctionTraitsT::Arg<5>::Type>(mLeafLocalData.get()));
    }

    template <typename ValueT>
    inline void
    addHandle(const points::PointDataTree::LeafNodeType& leaf,
              const size_t pos)
    {
        typename TypedHandle<ValueT>::UniquePtr handle(new TypedHandle<ValueT>());
        mVoidAttributeHandles.emplace_back(handle->initReadHandle(leaf, pos));
        mAttributeHandles.emplace_back(std::move(handle));
    }

    template <typename ValueT>
    inline void
    addWriteHandle(points::PointDataTree::LeafNodeType& leaf,
                   const size_t pos)
    {
        typename TypedHandle<ValueT>::UniquePtr handle(new TypedHandle<ValueT>());
        mVoidAttributeHandles.emplace_back(handle->initWriteHandle(leaf, pos));
        mAttributeHandles.emplace_back(std::move(handle));
    }

    inline void
    addGroupHandle(const points::PointDataTree::LeafNodeType& leaf,
                   const std::string& name)
    {
        assert(leaf.attributeSet().descriptor().hasGroup(name));
        mGroupHandles.emplace_back(new points::GroupHandle(leaf.groupHandle(name)));
        mVoidGroupHandles.emplace_back(static_cast<void*>(mGroupHandles.back().get()));
    }

    inline void
    addGroupWriteHandle(points::PointDataTree::LeafNodeType& leaf,
                        const std::string& name)
    {
        assert(leaf.attributeSet().descriptor().hasGroup(name));
        mGroupHandles.emplace_back(new points::GroupWriteHandle(leaf.groupWriteHandle(name)));
        mVoidGroupHandles.emplace_back(static_cast<void*>(mGroupHandles.back().get()));
    }

    inline void addNullGroupHandle() { mVoidGroupHandles.emplace_back(nullptr); }
    inline void addNullAttribHandle() { mVoidAttributeHandles.emplace_back(nullptr); }

    const CustomData::ConstPtr mCustomData;
    const points::AttributeSet* const mAttributeSet;
    uint64_t mIndex;
    compiler::LeafLocalData::UniquePtr mLeafLocalData;

private:
    std::vector<void*> mVoidAttributeHandles;
    std::vector<Handles::UniquePtr> mAttributeHandles;
    std::vector<void*> mVoidGroupHandles;
    std::vector<points::GroupHandle::Ptr> mGroupHandles;
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

    template <typename LeafT>
    void reset(const LeafT& leaf, const size_t)
    {
        mFilter.reset(leaf);
        mPws = points::AttributeHandle<Vec3f>::
            create(leaf.constAttributeArray(mPositionAttribute));
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
    points::AttributeHandle<Vec3f>::Ptr mPws;
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
        case ast::tokens::SHORT   : return true;
        case ast::tokens::INT     : return true;
        case ast::tokens::LONG    : return true;
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
        case ast::tokens::SHORT   : return addAttributeHandleTyped<int16_t>(args, leaf, name, write);
        case ast::tokens::INT     : return addAttributeHandleTyped<int32_t>(args, leaf, name, write);
        case ast::tokens::LONG    : return addAttributeHandleTyped<int64_t>(args, leaf, name, write);
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
template<bool UseGroup>
struct PointExecuterOp
{
    using LeafManagerT = openvdb::tree::LeafManager<openvdb::points::PointDataTree>;
    using LeafNode = openvdb::points::PointDataTree::LeafNodeType;

    using Descriptor = openvdb::points::AttributeSet::Descriptor;
    using GroupFilter = openvdb::points::GroupFilter;
    using GroupIndex = Descriptor::GroupIndex;

    PointExecuterOp(const AttributeRegistry& attributeRegistry,
               const CustomData::ConstPtr& customData,
               KernelFunctionPtr computeFunction,
               const math::Transform& transform,
               const GroupIndex* const groupIndex,
               std::vector<compiler::LeafLocalData::UniquePtr>& leafLocalData,
               const std::string& positionAttribute,
               const std::pair<bool,bool>& positionAccess)
        : mComputeFunction(computeFunction)
        , mCustomData(customData)
        , mTransform(transform)
        , mGroupIndex(groupIndex)
        , mAttributeRegistry(attributeRegistry)
        , mLeafLocalData(leafLocalData)
        , mPositionAttribute(positionAttribute)
        , mPositionAccess(positionAccess) {}

    template<typename FilterT = openvdb::points::NullFilter>
    inline points::AttributeWriteHandle<Vec3f>::Ptr
    initPositions(LeafNode& leaf, const FilterT& filter = FilterT()) const
    {
        const points::AttributeHandle<Vec3f>::Ptr
            positions = points::AttributeHandle<Vec3f>::
                create(leaf.constAttributeArray("P"));
        const points::AttributeWriteHandle<Vec3f>::Ptr
            pws = points::AttributeWriteHandle<Vec3f>::
                create(leaf.attributeArray(mPositionAttribute));

        for (auto iter = leaf.beginIndexAll(filter); iter; ++iter) {
            const Index idx = *iter;
            const openvdb::Vec3f pos = positions->get(idx) + iter.getCoord().asVec3s();
            pws->set(idx, mTransform.indexToWorld(pos));
        }

        return pws;
    }

    // UseGroup = true
    template<bool UseG>
    typename std::enable_if<UseG, void>::type
    execute(LeafNode& leaf, PointFunctionArguments& args) const
    {
        using IndexIterT = openvdb::points::IndexIter<LeafNode::ValueAllCIter, GroupFilter>;

        assert(mGroupIndex);
        GroupFilter filter(*mGroupIndex);
        IndexIterT iter = leaf.beginIndex<LeafNode::ValueAllCIter, GroupFilter>(filter);

        for (; iter; ++iter) {
            args.mIndex = *iter;
            args.bind(mComputeFunction)();
        }
    }

    // UseGroup = false
    template<bool UseG>
    typename std::enable_if<!UseG, void>::type
    execute(LeafNode& leaf, PointFunctionArguments& args) const
    {
        // the Compute function performs unsigned integer arithmetic and will wrap
        // if count <= 0 inside ComputeGenerator::genComputeFunction()

        const Index count = leaf.getLastValue();
        if (count <= 0) return;

        args.mIndex = count;
        args.bind(mComputeFunction)();
    }


    void operator()(LeafNode& leaf, size_t idx) const
    {
        PointFunctionArguments args(mCustomData, leaf.attributeSet(), leaf.getLastValue());

        // add attributes based on the order and existence in the attribute registry

        for (const auto& iter : mAttributeRegistry.data()) {
            const std::string& name = (iter.name() == "P" ? mPositionAttribute : iter.name());
            addAttributeHandle(args, leaf, name, iter.type(), iter.writes());
        }

        const auto& map = leaf.attributeSet().descriptor().groupMap();

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
                    args.addNullGroupHandle(); // empty handle at this index
                }
            }
        }

        // if we are using position we need to initialise the world space storage

        points::AttributeWriteHandle<Vec3f>::Ptr pws;
        if (mPositionAccess.first || mPositionAccess.second) {
            if (UseGroup) {
                const GroupFilter filter(*mGroupIndex);
                pws = this->initPositions(leaf, filter);
            }
            else {
                pws = this->initPositions(leaf);
            }
        }

        execute<UseGroup>(leaf, args);

        // if not writing to position (i.e. post sorting) collapse the temporary attribute

        if (pws && !mPositionAccess.second) {
            pws->collapse();
            pws.reset();
        }

        // as multiple groups can be stored in a single array, attempt to compact the
        // arrays directly so that we're not trying to call compact multiple times
        // unsuccessfully

        args.mLeafLocalData->compact();

        mLeafLocalData[idx] = std::move(args.mLeafLocalData);
    }

    void operator()(const LeafManagerT::LeafRange& range) const
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {
            (*this)(*leaf, leaf.pos());
        }
    }

private:
    KernelFunctionPtr           mComputeFunction;
    const CustomData::ConstPtr  mCustomData;
    const math::Transform&      mTransform;
    const GroupIndex* const     mGroupIndex;
    const AttributeRegistry&    mAttributeRegistry;
    std::vector<compiler::LeafLocalData::UniquePtr>& mLeafLocalData;
    const std::string&          mPositionAttribute;
    const std::pair<bool,bool>& mPositionAccess;
};

void appendMissingAttributes(points::PointDataGrid& grid,
                             const AttributeRegistry& registry)
{
    const auto leafIter = grid.tree().cbeginLeaf();
    assert(leafIter);

    // append attributes

    for (const auto& iter : registry.data()) {
        const std::string& name = iter.name();

        const points::AttributeSet::Descriptor& desc = leafIter->attributeSet().descriptor();

        const size_t pos = desc.find(name);

        if (pos != points::AttributeSet::INVALID_POS) {

            const NamePair& type = desc.type(pos);
            const ast::tokens::CoreType typetoken =
                ast::tokens::tokenFromTypeString(type.first);

            if (typetoken != iter.type() &&
                !(type.second == "str" && iter.type() == ast::tokens::STRING)) {
                OPENVDB_THROW(TypeError, "Mismatching attributes types. \"" + name +
                    "\" exists of type \"" + type.first + "\" but has been "
                    "accessed with type \"" + ast::tokens::typeStringFromToken(iter.type()) + "\"");
            }
            continue;
        }

        auto typePairFromToken =
            [](const ast::tokens::CoreType type) -> NamePair {
            switch (type) {
                case ast::tokens::BOOL    : return points::TypedAttributeArray<bool>::attributeType();
                case ast::tokens::CHAR    : return points::TypedAttributeArray<char>::attributeType();
                case ast::tokens::SHORT   : return points::TypedAttributeArray<int16_t>::attributeType();
                case ast::tokens::INT     : return points::TypedAttributeArray<int32_t>::attributeType();
                case ast::tokens::LONG    : return points::TypedAttributeArray<int64_t>::attributeType();
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

        assert(supported(iter.type()));
        const NamePair type = typePairFromToken(iter.type());
        points::appendAttribute(grid.tree(), name, type);
    }
}

void checkAttributesExist(const points::PointDataGrid& grid,
                          const AttributeRegistry& registry)
{
    const auto leafIter = grid.tree().cbeginLeaf();
    assert(leafIter);

    const points::AttributeSet::Descriptor& desc = leafIter->attributeSet().descriptor();

    for (const auto& iter : registry.data()) {
        const std::string& name = iter.name();

        const size_t pos = desc.find(name);

        if (pos == points::AttributeSet::INVALID_POS) {
            OPENVDB_THROW(openvdb::LookupError, "Attribute \"" + name +
                "\" does not exist on grid \"" + grid.getName() + "\"");
        }
    }
}

} // anonymous namespace


void PointExecutable::execute(openvdb::points::PointDataGrid& grid,
                              const std::string* const group,
                              const bool createMissing) const
{
    using LeafManagerT = openvdb::tree::LeafManager<openvdb::points::PointDataTree>;

    const auto leafIter = grid.tree().cbeginLeaf();
    if (!leafIter) return;

    // create any missing attributes
    if (createMissing) appendMissingAttributes(grid, *mAttributeRegistry);
    else checkAttributesExist(grid, *mAttributeRegistry);

    const std::pair<bool,bool> positionAccess =
        mAttributeRegistry->accessPattern("P", ast::tokens::VEC3F);
    const bool usingPosition = positionAccess.first || positionAccess.second;

    // create temporary world space position attribute if P is being accessed

    std::string positionAttribute = "P";
    if (usingPosition /*mAttributeRegistry->isWritable("P", ast::tokens::VEC3F)*/) {
        const points::AttributeSet::Descriptor& desc =
            leafIter->attributeSet().descriptor();
        positionAttribute = desc.uniqueName("__P");
        points::appendAttribute<openvdb::Vec3f>(grid.tree(), positionAttribute);
    }

    const bool usingGroup(static_cast<bool>(group) ? !group->empty() : false);
    const math::Transform& transform = grid.transform();

    openvdb::points::AttributeSet::Descriptor::GroupIndex groupIndex;
    if (usingGroup) {
        groupIndex = leafIter->attributeSet().groupIndex(*group);
    }

    LeafManagerT leafManager(grid.tree());

    std::vector<compiler::LeafLocalData::UniquePtr> leafLocalData(leafManager.leafCount());

    if (!usingGroup) {
        using FunctionType = codegen::PointRangeKernel;

        KernelFunctionPtr compute = nullptr;
        const auto iter = mFunctionAddresses.find(FunctionType::getDefaultName());
        if (iter != mFunctionAddresses.end()) {
            compute = reinterpret_cast<KernelFunctionPtr>(iter->second);
        }
        if (!compute) {
            OPENVDB_THROW(AXCompilerError,
                "No code has been successfully compiled for execution.");
        }

        PointExecuterOp</*UseGroup*/false>
            executerOp(*mAttributeRegistry, mCustomData, compute, transform, &groupIndex,
                leafLocalData, positionAttribute, positionAccess);
        leafManager.foreach(executerOp);
    }
    else {
        using FunctionType = codegen::PointKernel;

        KernelFunctionPtr compute = nullptr;
        const auto iter = mFunctionAddresses.find(FunctionType::getDefaultName());
        if (iter != mFunctionAddresses.end()) {
            compute = reinterpret_cast<KernelFunctionPtr>(iter->second);
        }
        if (!compute) {
            OPENVDB_THROW(AXCompilerError,
                "No code has been successfully compiled for execution.");
        }

        PointExecuterOp</*UseGroup*/true>
            executerOp(*mAttributeRegistry, mCustomData, compute, transform, &groupIndex,
                leafLocalData, positionAttribute, positionAccess);
        leafManager.foreach(executerOp);
    }

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
        [&groups, &leafLocalData, newStrings] (LeafManagerT::LeafNodeType& leaf, size_t idx) {

            compiler::LeafLocalData::UniquePtr& data = leafLocalData[idx];

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
                const compiler::LeafLocalData::StringArrayMap& stringArrayMap = data->getStringArrayMap();

                for (const auto& arrayIter : stringArrayMap) {
                    points::StringAttributeWriteHandle::Ptr handle =
                        points::StringAttributeWriteHandle::create(*(arrayIter.first), metadata);

                    for (const auto& iter : arrayIter.second) {
                        handle->set(static_cast<Index>(iter.first), iter.second);
                    }
                }
            }
    });

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
}

}
}
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
