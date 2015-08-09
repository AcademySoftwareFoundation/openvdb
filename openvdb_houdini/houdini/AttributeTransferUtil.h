///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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
//
/// @file AttributeTransferUtil.h
/// @author FX R&D Simulation team
/// @brief Utility methods used by the From Mesh and From Particle SOPs

#ifndef OPENVDB_HOUDINI_ATTRIBUTE_TRANSFER_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_ATTRIBUTE_TRANSFER_UTIL_HAS_BEEN_INCLUDED

#include <openvdb_houdini/Utils.h>

#include <openvdb/openvdb.h>
#include <openvdb/math/Proximity.h>
#include <openvdb/util/Util.h>

#include <GA/GA_PageIterator.h>
#include <GA/GA_SplittableRange.h>
#include <SYS/SYS_Types.h>
#include <UT/UT_Version.h>
#include <iostream>
#include <sstream>

namespace openvdb_houdini {

////////////////////////////////////////

/// Get OpenVDB specific value, by calling GA_AIFTuple::get()
/// with appropriate arguments.

template <typename ValueType> inline ValueType
evalAttr(const GA_Attribute* atr, const GA_AIFTuple* aif,
    GA_Offset off, int idx)
{
    fpreal64 value;
    aif->get(atr, off, value, idx);
    return ValueType(value);
}

template <> inline float
evalAttr<float>(const GA_Attribute* atr, const GA_AIFTuple* aif,
    GA_Offset off, int idx)
{
    fpreal32 value;
    aif->get(atr, off, value, idx);
    return float(value);
}

template <> inline openvdb::Int32
evalAttr<openvdb::Int32>(const GA_Attribute* atr, const GA_AIFTuple* aif,
    GA_Offset off, int idx)
{
    int32 value;
    aif->get(atr, off, value, idx);
    return openvdb::Int32(value);
}

template <> inline openvdb::Int64
evalAttr<openvdb::Int64>(const GA_Attribute* atr, const GA_AIFTuple* aif,
    GA_Offset off, int idx)
{
    int64 value;
    aif->get(atr, off, value, idx);
    return openvdb::Int64(value);
}

template <> inline openvdb::Vec3i
evalAttr<openvdb::Vec3i>(const GA_Attribute* atr, const GA_AIFTuple* aif,
    GA_Offset off, int)
{
    openvdb::Vec3i vec;

    int32 comp;
    aif->get(atr, off, comp, 0);
    vec[0] = openvdb::Int32(comp);

    aif->get(atr, off, comp, 1);
    vec[1] = openvdb::Int32(comp);

    aif->get(atr, off, comp, 2);
    vec[2] = openvdb::Int32(comp);

    return vec;
}

template <> inline openvdb::Vec3s
evalAttr<openvdb::Vec3s>(const GA_Attribute* atr, const GA_AIFTuple* aif,
    GA_Offset off, int)
{
    openvdb::Vec3s vec;

    fpreal32 comp;
    aif->get(atr, off, comp, 0);
    vec[0] = float(comp);

    aif->get(atr, off, comp, 1);
    vec[1] = float(comp);

    aif->get(atr, off, comp, 2);
    vec[2] = float(comp);

    return vec;
}

template <> inline openvdb::Vec3d
evalAttr<openvdb::Vec3d>(const GA_Attribute* atr, const GA_AIFTuple* aif,
    GA_Offset off, int)
{
    openvdb::Vec3d vec;

    fpreal64 comp;
    aif->get(atr, off, comp, 0);
    vec[0] = double(comp);

    aif->get(atr, off, comp, 1);
    vec[1] = double(comp);

    aif->get(atr, off, comp, 2);
    vec[2] = double(comp);

    return vec;
}


////////////////////////////////////////

/// Combine differnet value types.

template <typename ValueType> inline ValueType
combine(const ValueType& v0, const ValueType& v1, const ValueType& v2,
    const openvdb::Vec3d& w)
{
    return ValueType(v0 * w[0] + v1 * w[1] + v2 * w[2]);
}

template <> inline openvdb::Int32
combine(const openvdb::Int32& v0, const openvdb::Int32& v1,
    const openvdb::Int32& v2, const openvdb::Vec3d& w)
{
    if (w[2] > w[0] && w[2] > w[1]) return v2;
    if (w[1] > w[0] && w[1] > w[2]) return v1;
    return v0;
}

template <> inline openvdb::Int64
combine(const openvdb::Int64& v0, const openvdb::Int64& v1,
    const openvdb::Int64& v2, const openvdb::Vec3d& w)
{
    if (w[2] > w[0] && w[2] > w[1]) return v2;
    if (w[1] > w[0] && w[1] > w[2]) return v1;
    return v0;
}

template <> inline openvdb::Vec3i
combine(const openvdb::Vec3i& v0, const openvdb::Vec3i& v1,
    const openvdb::Vec3i& v2, const openvdb::Vec3d& w)
{
    if (w[2] > w[0] && w[2] > w[1]) return v2;
    if (w[1] > w[0] && w[1] > w[2]) return v1;
    return v0;
}

template <> inline openvdb::Vec3s
combine(const openvdb::Vec3s& v0, const openvdb::Vec3s& v1,
    const openvdb::Vec3s& v2, const openvdb::Vec3d& w)
{
    openvdb::Vec3s vec;

    vec[0] = float(v0[0] * w[0] + v1[0] * w[1] + v2[0] * w[2]);
    vec[1] = float(v0[1] * w[0] + v1[1] * w[1] + v2[1] * w[2]);
    vec[2] = float(v0[2] * w[0] + v1[2] * w[1] + v2[2] * w[2]);

    return vec;
}

template <> inline openvdb::Vec3d
combine(const openvdb::Vec3d& v0, const openvdb::Vec3d& v1,
    const openvdb::Vec3d& v2, const openvdb::Vec3d& w)
{
    openvdb::Vec3d vec;

    vec[0] = v0[0] * w[0] + v1[0] * w[1] + v2[0] * w[2];
    vec[1] = v0[1] * w[0] + v1[1] * w[1] + v2[1] * w[2];
    vec[2] = v0[2] * w[0] + v1[2] * w[1] + v2[2] * w[2];

    return vec;
}


////////////////////////////////////////

/// Gets OpenVDB specific value by evaluating GA_Default::get()
/// with appropriate arguments.

template <typename ValueType> inline ValueType
evalAttrDefault(const GA_Defaults& defaults, int idx)
{
    fpreal64 value;
    defaults.get(idx, value);
    return ValueType(value);
}

template <> inline float
evalAttrDefault<float>(const GA_Defaults& defaults, int /*idx*/)
{
    fpreal32 value;
    defaults.get(0, value);
    return float(value);
}

template <> inline openvdb::Int32
evalAttrDefault<openvdb::Int32>(const GA_Defaults& defaults, int idx)
{
    int32 value;
    defaults.get(idx, value);
    return openvdb::Int32(value);
}

template <> inline openvdb::Int64
evalAttrDefault<openvdb::Int64>(const GA_Defaults& defaults, int idx)
{
    int64 value;
    defaults.get(idx, value);
    return openvdb::Int64(value);
}

template <> inline openvdb::Vec3i
evalAttrDefault<openvdb::Vec3i>(const GA_Defaults& defaults, int)
{
    openvdb::Vec3i vec;
    int32 value;

    defaults.get(0, value);
    vec[0] = openvdb::Int32(value);

    defaults.get(1, value);
    vec[1] = openvdb::Int32(value);

    defaults.get(2, value);
    vec[2] = openvdb::Int32(value);

    return vec;
}

template <> inline openvdb::Vec3s
evalAttrDefault<openvdb::Vec3s>(const GA_Defaults& defaults, int)
{
    openvdb::Vec3s vec;
    fpreal32 value;

    defaults.get(0, value);
    vec[0] = float(value);

    defaults.get(1, value);
    vec[1] = float(value);

    defaults.get(2, value);
    vec[2] = float(value);

    return vec;
}

template <> inline openvdb::Vec3d
evalAttrDefault<openvdb::Vec3d>(const GA_Defaults& defaults, int)
{
    openvdb::Vec3d vec;
    fpreal64 value;

    defaults.get(0, value);
    vec[0] = double(value);

    defaults.get(1, value);
    vec[1] = double(value);

    defaults.get(2, value);
    vec[2] = double(value);

    return vec;
}


////////////////////////////////////////


class AttributeDetailBase
{
public:
    typedef boost::shared_ptr<AttributeDetailBase> Ptr;

    virtual ~AttributeDetailBase() {}

    virtual void set(const openvdb::Coord& ijk, const GA_Offset (&offsets)[3],
        const openvdb::Vec3d& weights) = 0;

    virtual void set(const openvdb::Coord& ijk, GA_Offset offset) = 0;

    virtual openvdb::GridBase::Ptr& grid() = 0;
    virtual std::string& name() = 0;

    virtual AttributeDetailBase::Ptr copy() = 0;

protected:
    AttributeDetailBase() {}
};


typedef std::vector<AttributeDetailBase::Ptr> AttributeDetailList;


////////////////////////////////////////


template <class VDBGridType>
class AttributeDetail: public AttributeDetailBase
{
public:
    typedef typename VDBGridType::ValueType ValueType;

    AttributeDetail(
        openvdb::GridBase::Ptr grid,
        const GA_Attribute* attribute,
        const GA_AIFTuple* tupleAIF,
        const int tupleIndex,
        const bool isVector = false);

    void set(const openvdb::Coord& ijk, const GA_Offset (&offsets)[3],
        const openvdb::Vec3d& weights);

    void set(const openvdb::Coord& ijk, GA_Offset offset);

    openvdb::GridBase::Ptr& grid() { return mGrid; }
    std::string& name() { return mName; }

    AttributeDetailBase::Ptr copy();

protected:
    AttributeDetail();

private:
    openvdb::GridBase::Ptr mGrid;
    typename VDBGridType::Accessor mAccessor;

    const GA_Attribute* mAttribute;
    const GA_AIFTuple*  mTupleAIF;
    const int           mTupleIndex;
    std::string         mName;
};


template <class VDBGridType>
AttributeDetail<VDBGridType>::AttributeDetail():
    mAttribute(NULL),
    mTupleAIF(NULL),
    mTupleIndex(0)
{
}


template <class VDBGridType>
AttributeDetail<VDBGridType>::AttributeDetail(
    openvdb::GridBase::Ptr grid,
    const GA_Attribute* attribute,
    const GA_AIFTuple* tupleAIF,
    const int tupleIndex,
    const bool isVector):
    mGrid(grid),
    mAccessor(openvdb::GridBase::grid<VDBGridType>(mGrid)->getAccessor()),
    mAttribute(attribute),
    mTupleAIF(tupleAIF),
    mTupleIndex(tupleIndex)
{
    std::ostringstream name;
    name << mAttribute->getName();

    const int tupleSize = mTupleAIF->getTupleSize(mAttribute);

    if(!isVector && tupleSize != 1) {
        name << "_" << mTupleIndex;
    }

    mName = name.str();
}


template <class VDBGridType>
void
AttributeDetail<VDBGridType>::set(const openvdb::Coord& ijk,
    const GA_Offset (&offsets)[3], const openvdb::Vec3d& weights)
{
    ValueType v0 = evalAttr<ValueType>(
        mAttribute, mTupleAIF, offsets[0], mTupleIndex);

    ValueType v1 = evalAttr<ValueType>(
        mAttribute, mTupleAIF, offsets[1], mTupleIndex);

    ValueType v2 = evalAttr<ValueType>(
        mAttribute, mTupleAIF, offsets[2], mTupleIndex);

    mAccessor.setValue(ijk, combine<ValueType>(v0, v1, v2, weights));
}

template <class VDBGridType>
void
AttributeDetail<VDBGridType>::set(const openvdb::Coord& ijk, GA_Offset offset)
{
    mAccessor.setValue(ijk,
        evalAttr<ValueType>(mAttribute, mTupleAIF, offset, mTupleIndex));
}

template <class VDBGridType>
AttributeDetailBase::Ptr
AttributeDetail<VDBGridType>::copy()
{
    return AttributeDetailBase::Ptr(new AttributeDetail<VDBGridType>(*this));
}


////////////////////////////////////////


// TBB object to transfer mesh attributes.
// Only quads and/or triangles are supported
// NOTE: This class has all code in the header and so it cannot have
//       OPENVDB_HOUDINI_API.
class MeshAttrTransfer
{
public:
    typedef openvdb::tree::IteratorRange<openvdb::Int32Tree::LeafCIter> IterRange;

    inline
    MeshAttrTransfer(
        AttributeDetailList &pointAttributes,
        AttributeDetailList &vertexAttributes,
        AttributeDetailList &primitiveAttributes,
        const openvdb::Int32Grid& closestPrimGrid,
        const openvdb::math::Transform& transform,
        const GU_Detail& meshGdp);

    inline
    MeshAttrTransfer(const MeshAttrTransfer &other);

    inline
    ~MeshAttrTransfer() {}

    /// Main calls
    inline void runParallel();
    inline void runSerial();

    inline void operator()(IterRange &range) const;

private:
    AttributeDetailList mPointAttributes, mVertexAttributes, mPrimitiveAttributes;
    const openvdb::Int32Grid& mClosestPrimGrid;

    const openvdb::math::Transform& mTransform;

    const GA_Detail &mMeshGdp;
};


MeshAttrTransfer::MeshAttrTransfer(
    AttributeDetailList& pointAttributes,
    AttributeDetailList& vertexAttributes,
    AttributeDetailList& primitiveAttributes,
    const openvdb::Int32Grid& closestPrimGrid,
    const openvdb::math::Transform& transform,
    const GU_Detail& meshGdp):
    mPointAttributes(pointAttributes),
    mVertexAttributes(vertexAttributes),
    mPrimitiveAttributes(primitiveAttributes),
    mClosestPrimGrid(closestPrimGrid),
    mTransform(transform),
    mMeshGdp(meshGdp)
{
}


MeshAttrTransfer::MeshAttrTransfer(const MeshAttrTransfer &other):
    mPointAttributes(other.mPointAttributes.size()),
    mVertexAttributes(other.mVertexAttributes.size()),
    mPrimitiveAttributes(other.mPrimitiveAttributes.size()),
    mClosestPrimGrid(other.mClosestPrimGrid),
    mTransform(other.mTransform),
    mMeshGdp(other.mMeshGdp)
{
    // Deep-copy the AttributeDetail arrays, to construct unique tree
    // accessors per thread.

    for (size_t i = 0, N = other.mPointAttributes.size(); i < N; ++i) {
       mPointAttributes[i] = other.mPointAttributes[i]->copy();
    }

    for (size_t i = 0, N = other.mVertexAttributes.size(); i < N; ++i) {
       mVertexAttributes[i] = other.mVertexAttributes[i]->copy();
    }

    for (size_t i = 0, N = other.mPrimitiveAttributes.size(); i < N; ++i) {
       mPrimitiveAttributes[i] =  other.mPrimitiveAttributes[i]->copy();
    }
}


void
MeshAttrTransfer::runParallel()
{
    IterRange range(mClosestPrimGrid.tree().beginLeaf());
    tbb::parallel_for(range, *this);
}

void
MeshAttrTransfer::runSerial()
{
    IterRange range(mClosestPrimGrid.tree().beginLeaf());
    (*this)(range);
}


void
MeshAttrTransfer::operator()(IterRange &range) const
{
    openvdb::Int32Tree::LeafNodeType::ValueOnCIter iter;

    openvdb::Coord ijk;
    unsigned int vtx;
    GA_Size vtxn;

    const bool ptnAttrTransfer = mPointAttributes.size() > 0;
    const bool vtxAttrTransfer = mVertexAttributes.size() > 0;

    GA_Primitive::const_iterator it;
    GA_Offset vtxOffsetList[4], ptnOffsetList[4], vtxOffsets[3], ptnOffsets[3], prmOffset;
    openvdb::Vec3d ptnList[4], xyz, cpt, cpt2, uvw, uvw2;

    for ( ; range; ++range) {
        iter = range.iterator()->beginValueOn();
        for ( ; iter; ++iter) {

            ijk = iter.getCoord();

            const GA_Index prmIndex = iter.getValue();
            prmOffset = mMeshGdp.primitiveOffset(prmIndex);

            // Transfer primitive attributes
            for (size_t i = 0, N = mPrimitiveAttributes.size(); i < N; ++i) {
                mPrimitiveAttributes[i]->set(ijk, prmOffset);
            }

            if (!ptnAttrTransfer && !vtxAttrTransfer) continue;

            // Transfer vertex and point attributes
            const GA_Primitive * primRef = mMeshGdp.getPrimitiveList().get(prmOffset);

            vtxn = primRef->getVertexCount();

            // Get vertex and point offests
            primRef->beginVertex(it);

            for (vtx = 0; !it.atEnd(); ++it, ++vtx) {

                ptnOffsetList[vtx] = it.getPointOffset();
                vtxOffsetList[vtx] = it.getVertexOffset();

                UT_Vector3 p = mMeshGdp.getPos3(ptnOffsetList[vtx]);
                ptnList[vtx][0] = double(p[0]);
                ptnList[vtx][1] = double(p[1]);
                ptnList[vtx][2] = double(p[2]);
            }

            xyz = mTransform.indexToWorld(ijk);

            // Compute barycentric coordinates

            cpt = closestPointOnTriangleToPoint(
                    ptnList[0], ptnList[2], ptnList[1], xyz, uvw);

            vtxOffsets[0] = vtxOffsetList[0]; // cpt offsets
            ptnOffsets[0] = ptnOffsetList[0];
            vtxOffsets[1] = vtxOffsetList[2];
            ptnOffsets[1] = ptnOffsetList[2];
            vtxOffsets[2] = vtxOffsetList[1];
            ptnOffsets[2] = ptnOffsetList[1];

            if (4 == vtxn) {
                cpt2 = closestPointOnTriangleToPoint(
                        ptnList[0], ptnList[3], ptnList[2], xyz, uvw2);

                if ((cpt2 - xyz).lengthSqr() < (cpt - xyz).lengthSqr()) {
                    uvw = uvw2;
                    vtxOffsets[1] = vtxOffsetList[3];
                    ptnOffsets[1] = ptnOffsetList[3];
                    vtxOffsets[2] = vtxOffsetList[2];
                    ptnOffsets[2] = ptnOffsetList[2];
                }
            }

            // Transfer vertex attributes
            for (size_t i = 0, N = mVertexAttributes.size(); i < N; ++i) {
                mVertexAttributes[i]->set(ijk, vtxOffsets, uvw);
            }

            // Transfer point attributes
            for (size_t i = 0, N = mPointAttributes.size(); i < N; ++i) {
                mPointAttributes[i]->set(ijk, ptnOffsets, uvw);
            }

        } // end sparse voxel iteration.
    } // end leaf-node iteration
}


////////////////////////////////////////


// TBB object to transfer mesh attributes.
// Only quads and/or triangles are supported
// NOTE: This class has all code in the header and so it cannot have
//       OPENVDB_HOUDINI_API.
class PointAttrTransfer
{
public:
    typedef openvdb::tree::IteratorRange<openvdb::Int32Tree::LeafCIter> IterRange;

    inline
    PointAttrTransfer(
        AttributeDetailList &pointAttributes,
        const openvdb::Int32Grid& closestPtnIdxGrid,
        const GU_Detail& ptGeop);

    inline
    PointAttrTransfer(const PointAttrTransfer &other);

    inline
    ~PointAttrTransfer() {}

    /// Main calls
    inline void runParallel();
    inline void runSerial();

    inline void operator()(IterRange &range) const;

private:
    AttributeDetailList mPointAttributes;
    const openvdb::Int32Grid& mClosestPtnIdxGrid;
    const GA_Detail &mPtGeo;
};


PointAttrTransfer::PointAttrTransfer(
    AttributeDetailList& pointAttributes,
    const openvdb::Int32Grid& closestPtnIdxGrid,
    const GU_Detail& ptGeop):
    mPointAttributes(pointAttributes),
    mClosestPtnIdxGrid(closestPtnIdxGrid),
    mPtGeo(ptGeop)
{
}


PointAttrTransfer::PointAttrTransfer(const PointAttrTransfer &other):
    mPointAttributes(other.mPointAttributes.size()),
    mClosestPtnIdxGrid(other.mClosestPtnIdxGrid),
    mPtGeo(other.mPtGeo)
{
    // Deep-copy the AttributeDetail arrays, to construct unique tree
    // accessors per thread.

    for (size_t i = 0, N = other.mPointAttributes.size(); i < N; ++i) {
       mPointAttributes[i] = other.mPointAttributes[i]->copy();
    }
}


void
PointAttrTransfer::runParallel()
{
    IterRange range(mClosestPtnIdxGrid.tree().beginLeaf());
    tbb::parallel_for(range, *this);
}

void
PointAttrTransfer::runSerial()
{
    IterRange range(mClosestPtnIdxGrid.tree().beginLeaf());
    (*this)(range);
}


void
PointAttrTransfer::operator()(IterRange &range) const
{
    openvdb::Int32Tree::LeafNodeType::ValueOnCIter iter;
    openvdb::Coord ijk;

    for ( ; range; ++range) {
        iter = range.iterator()->beginValueOn();
        for ( ; iter; ++iter) {

            ijk = iter.getCoord();

            const GA_Index pointIndex = iter.getValue();
            const GA_Offset pointOffset = mPtGeo.pointOffset(pointIndex);

            // Transfer point attributes
            for (size_t i = 0, N = mPointAttributes.size(); i < N; ++i) {
                mPointAttributes[i]->set(ijk, pointOffset);
            }
        } // end sparse voxel iteration.
    } // end leaf-node iteration
}


////////////////////////////////////////

// Mesh to Mesh Attribute Transfer Utils


struct AttributeCopyBase
{
    typedef boost::shared_ptr<AttributeCopyBase> Ptr;

    virtual ~AttributeCopyBase() {}
    virtual void copy(GA_Offset /*source*/, GA_Offset /*target*/) = 0;
    virtual void copy(GA_Offset&, GA_Offset&, GA_Offset&, GA_Offset /*target*/,
        const openvdb::Vec3d& /*uvw*/) = 0;
protected:
    AttributeCopyBase() {}
};


template<class ValueType>
struct AttributeCopy: public AttributeCopyBase
{
public:
    AttributeCopy(const GA_Attribute& sourceAttr, GA_Attribute& targetAttr)
        : mSourceAttr(sourceAttr)
        , mTargetAttr(targetAttr)
        , mAIFTuple(*mSourceAttr.getAIFTuple())
        , mTupleSize(mAIFTuple.getTupleSize(&mSourceAttr))
    {
    }

    void copy(GA_Offset source, GA_Offset target)
    {
        ValueType data;
        for (int i = 0; i < mTupleSize; ++i) {
            mAIFTuple.get(&mSourceAttr, source, data, i);
            mAIFTuple.set(&mTargetAttr, target, data, i);
        }
    }

    void copy(GA_Offset& v0, GA_Offset& v1, GA_Offset& v2, GA_Offset target,
        const openvdb::Vec3d& uvw)
    {
        doCopy<ValueType>(v0, v1, v2, target, uvw);
    }

private:

    template <typename T>
    typename boost::enable_if<boost::is_integral<T>, void>::type
    doCopy(GA_Offset& v0, GA_Offset& v1, GA_Offset& v2, GA_Offset target, const openvdb::Vec3d& uvw)
    {
        GA_Offset source = v0;
        double min = uvw[0];

        if (uvw[1] < min) {
            min = uvw[1];
            source = v1;
        }
        if (uvw[2] < min) source = v2;


        ValueType data;
        for (int i = 0; i < mTupleSize; ++i) {
            mAIFTuple.get(&mSourceAttr, source, data, i);
            mAIFTuple.set(&mTargetAttr, target, data, i);
        }
    }

    template <typename T>
    typename boost::enable_if<boost::is_floating_point<T>, void>::type
    doCopy(GA_Offset& v0, GA_Offset& v1, GA_Offset& v2, GA_Offset target, const openvdb::Vec3d& uvw)
    {
        ValueType a, b, c;
        for (int i = 0; i < mTupleSize; ++i) {
            mAIFTuple.get(&mSourceAttr, v0, a, i);
            mAIFTuple.get(&mSourceAttr, v1, b, i);
            mAIFTuple.get(&mSourceAttr, v2, c, i);
            mAIFTuple.set(&mTargetAttr, target, a*uvw[0] + b*uvw[1] + c*uvw[2], i);
        }
    }


    const GA_Attribute& mSourceAttr;
    GA_Attribute& mTargetAttr;
    const GA_AIFTuple& mAIFTuple;
    int mTupleSize;
};


struct StrAttributeCopy: public AttributeCopyBase
{
public:
    StrAttributeCopy(const GA_Attribute& sourceAttr, GA_Attribute& targetAttr)
        : mSourceAttr(sourceAttr)
        , mTargetAttr(targetAttr)
        , mAIF(*mSourceAttr.getAIFSharedStringTuple())
        , mTupleSize(mAIF.getTupleSize(&mSourceAttr))
    {
    }

    void copy(GA_Offset source, GA_Offset target)
    {
        for (int i = 0; i < mTupleSize; ++i) {
            mAIF.setString(&mTargetAttr, target, mAIF.getString(&mSourceAttr, source, i), i);
        }
    }

    void copy(GA_Offset& v0, GA_Offset& v1, GA_Offset& v2, GA_Offset target,
        const openvdb::Vec3d& uvw)
    {
        GA_Offset source = v0;
        double min = uvw[0];

        if (uvw[1] < min) {
            min = uvw[1];
            source = v1;
        }
        if (uvw[2] < min) source = v2;

        for (int i = 0; i < mTupleSize; ++i) {
             mAIF.setString(&mTargetAttr, target, mAIF.getString(&mSourceAttr, source, i), i);
        }
    }

protected:
    const GA_Attribute& mSourceAttr;
    GA_Attribute& mTargetAttr;
    const GA_AIFSharedStringTuple& mAIF;
    int mTupleSize;
};


////////////////////////////////////////


inline AttributeCopyBase::Ptr
createAttributeCopier(const GA_Attribute& sourceAttr, GA_Attribute& targetAttr)
{
    const GA_AIFTuple * aifTuple = sourceAttr.getAIFTuple();
    AttributeCopyBase::Ptr attr;

    if (aifTuple) {
        const GA_Storage sourceStorage = aifTuple->getStorage(&sourceAttr);
        const GA_Storage targetStorage = aifTuple->getStorage(&targetAttr);

        const int sourceTupleSize = aifTuple->getTupleSize(&sourceAttr);
        const int targetTupleSize = aifTuple->getTupleSize(&targetAttr);

        if (sourceStorage == targetStorage && sourceTupleSize == targetTupleSize) {
            switch (sourceStorage)
            {
                case GA_STORE_INT16:
                case GA_STORE_INT32:
                    attr = AttributeCopyBase::Ptr(
                        new AttributeCopy<int32>(sourceAttr, targetAttr));
                    break;
                case GA_STORE_INT64:
                    attr = AttributeCopyBase::Ptr(
                        new AttributeCopy<int64>(sourceAttr, targetAttr));
                    break;
                case GA_STORE_REAL16:
                case GA_STORE_REAL32:
                    attr = AttributeCopyBase::Ptr(
                        new AttributeCopy<fpreal32>(sourceAttr, targetAttr));
                    break;
                case GA_STORE_REAL64:
                    attr = AttributeCopyBase::Ptr(
                        new AttributeCopy<fpreal64>(sourceAttr, targetAttr));
                    break;
                default:
                    break;
            }
        }
    } else {
        const GA_AIFSharedStringTuple * aifString = sourceAttr.getAIFSharedStringTuple();
        if (aifString) {
            attr = AttributeCopyBase::Ptr(new StrAttributeCopy(sourceAttr, targetAttr));
        }
    }

    return attr;
}


////////////////////////////////////////


inline GA_Offset
findClosestPrimitiveToPoint(
    const GU_Detail& geo, const std::set<GA_Index>& primitives, const openvdb::Vec3d& p,
    GA_Offset& vert0, GA_Offset& vert1, GA_Offset& vert2, openvdb::Vec3d& uvw)
{
    std::set<GA_Index>::const_iterator it = primitives.begin();

    GA_Offset primOffset = GA_INVALID_OFFSET;
    const GA_Primitive * primRef = NULL;
    double minDist = std::numeric_limits<double>::max();

    openvdb::Vec3d a, b, c, d, tmpUVW;
    UT_Vector3 tmpPoint;

    for (; it != primitives.end(); ++it) {

        const GA_Offset offset = geo.primitiveOffset(*it);
        primRef = geo.getPrimitiveList().get(offset);

        const GA_Size vertexCount = primRef->getVertexCount();


        if (vertexCount == 3 || vertexCount == 4) {

            tmpPoint = geo.getPos3(primRef->getPointOffset(0));
            a[0] = tmpPoint.x();
            a[1] = tmpPoint.y();
            a[2] = tmpPoint.z();

            tmpPoint = geo.getPos3(primRef->getPointOffset(1));
            b[0] = tmpPoint.x();
            b[1] = tmpPoint.y();
            b[2] = tmpPoint.z();

            tmpPoint = geo.getPos3(primRef->getPointOffset(2));
            c[0] = tmpPoint.x();
            c[1] = tmpPoint.y();
            c[2] = tmpPoint.z();

            double tmpDist =
                (p - openvdb::math::closestPointOnTriangleToPoint(a, c, b, p, tmpUVW)).lengthSqr();

            if (tmpDist < minDist) {
                minDist = tmpDist;
                primOffset = offset;
                uvw = tmpUVW;
                vert0 = primRef->getVertexOffset(0);
                vert1 = primRef->getVertexOffset(2);
                vert2 = primRef->getVertexOffset(1);
            }

            if (vertexCount == 4) {
                tmpPoint = geo.getPos3(primRef->getPointOffset(3));
                d[0] = tmpPoint.x();
                d[1] = tmpPoint.y();
                d[2] = tmpPoint.z();

                tmpDist = (p - openvdb::math::closestPointOnTriangleToPoint(
                    a, d, c, p, tmpUVW)).lengthSqr();
                if (tmpDist < minDist) {
                    minDist = tmpDist;
                    primOffset = offset;
                    uvw = tmpUVW;
                    vert0 = primRef->getVertexOffset(0);
                    vert1 = primRef->getVertexOffset(3);
                    vert2 = primRef->getVertexOffset(2);
                }
            }

        }
    }

    return primOffset;
}


// Faster for small primitive counts
inline GA_Offset
findClosestPrimitiveToPoint(
    const GU_Detail& geo, std::vector<GA_Index>& primitives, const openvdb::Vec3d& p,
    GA_Offset& vert0, GA_Offset& vert1, GA_Offset& vert2, openvdb::Vec3d& uvw)
{
    GA_Offset primOffset = GA_INVALID_OFFSET;
    const GA_Primitive * primRef = NULL;
    double minDist = std::numeric_limits<double>::max();

    openvdb::Vec3d a, b, c, d, tmpUVW;
    UT_Vector3 tmpPoint;

    std::sort(primitives.begin(), primitives.end());

    GA_Index lastPrim = -1;
    for (size_t n = 0, N = primitives.size(); n < N; ++n) {
        if (primitives[n] == lastPrim) continue;
        lastPrim = primitives[n];

        const GA_Offset offset = geo.primitiveOffset(lastPrim);
        primRef = geo.getPrimitiveList().get(offset);

        const GA_Size vertexCount = primRef->getVertexCount();


        if (vertexCount == 3 || vertexCount == 4) {

            tmpPoint = geo.getPos3(primRef->getPointOffset(0));
            a[0] = tmpPoint.x();
            a[1] = tmpPoint.y();
            a[2] = tmpPoint.z();

            tmpPoint = geo.getPos3(primRef->getPointOffset(1));
            b[0] = tmpPoint.x();
            b[1] = tmpPoint.y();
            b[2] = tmpPoint.z();

            tmpPoint = geo.getPos3(primRef->getPointOffset(2));
            c[0] = tmpPoint.x();
            c[1] = tmpPoint.y();
            c[2] = tmpPoint.z();

            double tmpDist =
                (p - openvdb::math::closestPointOnTriangleToPoint(a, c, b, p, tmpUVW)).lengthSqr();

            if (tmpDist < minDist) {
                minDist = tmpDist;
                primOffset = offset;
                uvw = tmpUVW;
                vert0 = primRef->getVertexOffset(0);
                vert1 = primRef->getVertexOffset(2);
                vert2 = primRef->getVertexOffset(1);
            }

            if (vertexCount == 4) {
                tmpPoint = geo.getPos3(primRef->getPointOffset(3));
                d[0] = tmpPoint.x();
                d[1] = tmpPoint.y();
                d[2] = tmpPoint.z();

                tmpDist = (p - openvdb::math::closestPointOnTriangleToPoint(
                    a, d, c, p, tmpUVW)).lengthSqr();
                if (tmpDist < minDist) {
                    minDist = tmpDist;
                    primOffset = offset;
                    uvw = tmpUVW;
                    vert0 = primRef->getVertexOffset(0);
                    vert1 = primRef->getVertexOffset(3);
                    vert2 = primRef->getVertexOffset(2);
                }
            }

        }
    }

    return primOffset;
}

////////////////////////////////////////


template<class GridType>
class TransferPrimitiveAttributesOp
{
public:
    TransferPrimitiveAttributesOp(
        const GU_Detail& sourceGeo, GU_Detail& targetGeo, const GridType& indexGrid,
        std::vector<AttributeCopyBase::Ptr>& primAttributes,
        std::vector<AttributeCopyBase::Ptr>& vertAttributes);

    void operator()(const GA_SplittableRange&) const;
private:
    const GU_Detail& mSourceGeo;
    GU_Detail& mTargetGeo;
    const GridType& mIndexGrid;
    std::vector<AttributeCopyBase::Ptr>& mPrimAttributes;
    std::vector<AttributeCopyBase::Ptr>& mVertAttributes;
};

template<class GridType>
TransferPrimitiveAttributesOp<GridType>::TransferPrimitiveAttributesOp(
    const GU_Detail& sourceGeo, GU_Detail& targetGeo, const GridType& indexGrid,
    std::vector<AttributeCopyBase::Ptr>& primAttributes,
    std::vector<AttributeCopyBase::Ptr>& vertAttributes)
    : mSourceGeo(sourceGeo)
    , mTargetGeo(targetGeo)
    , mIndexGrid(indexGrid)
    , mPrimAttributes(primAttributes)
    , mVertAttributes(vertAttributes)
{

}

template<class GridType>
void
TransferPrimitiveAttributesOp<GridType>::operator()(const GA_SplittableRange& range) const
{
    typedef typename GridType::ValueType IndexT;

    GA_Offset start, end, source, target, v0, v1, v2;
    const GA_Primitive * primRef = NULL;
    GA_Primitive::const_iterator vtxIt;

    typename GridType::ConstAccessor acc = mIndexGrid.getConstAccessor();
    const openvdb::math::Transform& transform = mIndexGrid.transform();
    openvdb::Vec3d pos, indexPos, uvw;
    openvdb::Coord ijk, coord;
    std::vector<GA_Index> primitives(8), similar_primitives(8);
    int count;

    UT_Vector3 targetN, sourceN;

    for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
        for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
            for (target = start; target < end; ++target) {

                primRef = mTargetGeo.getPrimitiveList().get(target);
                targetN = mTargetGeo.getGEOPrimitive(target)->computeNormal();

                if (mPrimAttributes.size() != 0 ) {

                    // Compute avg. vertex position
                    pos[0] = 0.0;
                    pos[1] = 0.0;
                    pos[2] = 0.0;
                    count = 0;

                    for (primRef->beginVertex(vtxIt); !vtxIt.atEnd(); ++vtxIt, ++count) {
                        const UT_Vector3 p = mTargetGeo.getPos3(vtxIt.getPointOffset());
                        pos[0] += p.x();
                        pos[1] += p.y();
                        pos[2] += p.z();
                    }

                    if (count > 1) pos *= (1.0 / float(count));
                    indexPos = transform.worldToIndex(pos);

                    // Find closest source primitive to current avg. vertex position.
                    coord[0] = int(std::floor(indexPos[0]));
                    coord[1] = int(std::floor(indexPos[1]));
                    coord[2] = int(std::floor(indexPos[2]));

                    primitives.clear();
                    similar_primitives.clear();
                    IndexT primIndex;

                    for (int d = 0; d < 8; ++d) {
                        ijk[0] = coord[0] + (((d & 0x02) >> 1) ^ (d & 0x01));
                        ijk[1] = coord[1] + ((d & 0x02) >> 1);
                        ijk[2] = coord[2] + ((d & 0x04) >> 2);

                        if (acc.probeValue(ijk, primIndex) &&
                            openvdb::Index32(primIndex) != openvdb::util::INVALID_IDX) {

                            GA_Offset tmpOffset = mSourceGeo.primitiveOffset(primIndex);
                            sourceN = mSourceGeo.getGEOPrimitive(tmpOffset)->computeNormal();

                            if (sourceN.dot(targetN) > 0.5) {
                                similar_primitives.push_back(primIndex);
                            } else {
                                primitives.push_back(primIndex);
                            }
                        }
                    }

                    if (!primitives.empty() || !similar_primitives.empty()) {

                        if (!similar_primitives.empty()) {
                            source = findClosestPrimitiveToPoint(
                                mSourceGeo, similar_primitives, pos, v0, v1, v2, uvw);
                        } else {
                            source = findClosestPrimitiveToPoint(
                                mSourceGeo, primitives, pos, v0, v1, v2, uvw);
                        }

                        // Transfer attributes
                        for (size_t n = 0, N = mPrimAttributes.size(); n < N; ++n) {
                            mPrimAttributes[n]->copy(source, target);
                        }
                    }
                }

                if (mVertAttributes.size() != 0) {
                    for (primRef->beginVertex(vtxIt); !vtxIt.atEnd(); ++vtxIt) {


                        const UT_Vector3 p = mTargetGeo.getPos3(vtxIt.getPointOffset());
                        pos[0] = p.x();
                        pos[1] = p.y();
                        pos[2] = p.z();

                        indexPos = transform.worldToIndex(pos);
                        coord[0] = int(std::floor(indexPos[0]));
                        coord[1] = int(std::floor(indexPos[1]));
                        coord[2] = int(std::floor(indexPos[2]));

                        int primIndex;
                        primitives.clear();
                        similar_primitives.clear();

                        for (int d = 0; d < 8; ++d) {
                            ijk[0] = coord[0] + (((d & 0x02) >> 1) ^ (d & 0x01));
                            ijk[1] = coord[1] + ((d & 0x02) >> 1);
                            ijk[2] = coord[2] + ((d & 0x04) >> 2);

                            if (acc.probeValue(ijk, primIndex) &&
                                openvdb::Index32(primIndex) != openvdb::util::INVALID_IDX) {

                                GA_Offset tmpOffset = mSourceGeo.primitiveOffset(primIndex);
                                sourceN = mSourceGeo.getGEOPrimitive(tmpOffset)->computeNormal();

                                if (sourceN.dot(targetN) > 0.5) {
                                    primitives.push_back(primIndex);
                                }
                            }
                        }

                        if (!primitives.empty() || !similar_primitives.empty()) {

                            if (!similar_primitives.empty()) {
                                findClosestPrimitiveToPoint(
                                    mSourceGeo, similar_primitives, pos, v0, v1, v2, uvw);
                            } else {
                                findClosestPrimitiveToPoint(
                                    mSourceGeo, primitives, pos, v0, v1, v2, uvw);
                            }

                            for (size_t n = 0, N = mVertAttributes.size(); n < N; ++n) {
                                mVertAttributes[n]->copy(v0, v1, v2, vtxIt.getVertexOffset(), uvw);
                            }
                        }
                    }
                }
            }
        }
    }
}


////////////////////////////////////////

template<class GridType>
class TransferPointAttributesOp
{
public:
    TransferPointAttributesOp(
        const GU_Detail& sourceGeo, GU_Detail& targetGeo, const GridType& indexGrid,
        std::vector<AttributeCopyBase::Ptr>& pointAttributes,
        const GA_PrimitiveGroup* surfacePrims = NULL);

    void operator()(const GA_SplittableRange&) const;
private:
    const GU_Detail& mSourceGeo;
    GU_Detail& mTargetGeo;
    const GridType& mIndexGrid;
    std::vector<AttributeCopyBase::Ptr>& mPointAttributes;
    const GA_PrimitiveGroup* mSurfacePrims;
};

template<class GridType>
TransferPointAttributesOp<GridType>::TransferPointAttributesOp(
    const GU_Detail& sourceGeo, GU_Detail& targetGeo, const GridType& indexGrid,
    std::vector<AttributeCopyBase::Ptr>& pointAttributes,
    const GA_PrimitiveGroup* surfacePrims)
    : mSourceGeo(sourceGeo)
    , mTargetGeo(targetGeo)
    , mIndexGrid(indexGrid)
    , mPointAttributes(pointAttributes)
    , mSurfacePrims(surfacePrims)
{
}

template<class GridType>
void
TransferPointAttributesOp<GridType>::operator()(const GA_SplittableRange& range) const
{
    typedef typename GridType::ValueType IndexT;

    GA_Offset start, end, vtxOffset, primOffset, target, v0, v1, v2;

    typename GridType::ConstAccessor acc = mIndexGrid.getConstAccessor();
    const openvdb::math::Transform& transform = mIndexGrid.transform();
    openvdb::Vec3d pos, indexPos, uvw;
    std::vector<GA_Index> primitives(8);
    openvdb::Coord ijk, coord;

    for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
        for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
            for (target = start; target < end; ++target) {


                vtxOffset = mTargetGeo.pointVertex(target);

                // Check if point is referenced by a surface primitive.
                if (mSurfacePrims) {
                    bool surfacePrim = false;

                    while (GAisValid(vtxOffset)) {

                        primOffset = mTargetGeo.vertexPrimitive(vtxOffset);

                        if (mSurfacePrims->containsIndex(mTargetGeo.primitiveIndex(primOffset))) {
                            surfacePrim = true;
                            break;
                        }

                        vtxOffset = mTargetGeo.vertexToNextVertex(vtxOffset);
                    }

                    if (!surfacePrim) continue;
                }

                const UT_Vector3 p = mTargetGeo.getPos3(target);
                pos[0] = p.x();
                pos[1] = p.y();
                pos[2] = p.z();

                indexPos = transform.worldToIndex(pos);
                coord[0] = int(std::floor(indexPos[0]));
                coord[1] = int(std::floor(indexPos[1]));
                coord[2] = int(std::floor(indexPos[2]));

                primitives.clear();
                IndexT primIndex;

                for (int d = 0; d < 8; ++d) {
                    ijk[0] = coord[0] + (((d & 0x02) >> 1) ^ (d & 0x01));
                    ijk[1] = coord[1] + ((d & 0x02) >> 1);
                    ijk[2] = coord[2] + ((d & 0x04) >> 2);

                    if (acc.probeValue(ijk, primIndex) &&
                        openvdb::Index32(primIndex) != openvdb::util::INVALID_IDX) {
                        primitives.push_back(primIndex);
                    }
                }

                if (!primitives.empty()) {
                    findClosestPrimitiveToPoint(mSourceGeo, primitives, pos, v0, v1, v2, uvw);

                    v0 = mSourceGeo.vertexPoint(v0);
                    v1 = mSourceGeo.vertexPoint(v1);
                    v2 = mSourceGeo.vertexPoint(v2);

                    for (size_t n = 0, N = mPointAttributes.size(); n < N; ++n) {
                        mPointAttributes[n]->copy(v0, v1, v2, target, uvw);
                    }
                }
            }
        }
    }
}


////////////////////////////////////////


template<class GridType>
inline void
transferPrimitiveAttributes(
    const GU_Detail& sourceGeo,
    GU_Detail& targetGeo,
    GridType& indexGrid,
    Interrupter& boss,
    const GA_PrimitiveGroup *primitives = NULL)
{
    // Match public primitive attributes
    GA_AttributeDict::iterator it = sourceGeo.primitiveAttribs().begin(GA_SCOPE_PUBLIC);

    if (indexGrid.activeVoxelCount() == 0) return;

    std::vector<AttributeCopyBase::Ptr> primAttributeList;

    // Primitive attributes
    for (; !it.atEnd(); ++it) {
        const GA_Attribute* sourceAttr = it.attrib();
#if (UT_VERSION_INT >= 0x0e0000b0) // 14.0.176 or later
        if (NULL == targetGeo.findPrimitiveAttribute(it.name())) {
            targetGeo.addPrimAttrib(sourceAttr);
        }
        GA_Attribute* targetAttr = targetGeo.findPrimitiveAttribute(it.name());
#else
        if (!targetGeo.findPrimitiveAttribute(it.name()).isValid()) {
            targetGeo.addPrimAttrib(sourceAttr);
        }
        GA_Attribute* targetAttr = targetGeo.findPrimitiveAttribute(it.name()).getAttribute();
#endif

        if (sourceAttr && targetAttr) {
            AttributeCopyBase::Ptr att = createAttributeCopier(*sourceAttr, *targetAttr);
            if(att) primAttributeList.push_back(att);
        }
    }

    if (boss.wasInterrupted()) return;

    std::vector<AttributeCopyBase::Ptr> vertAttributeList;

    it = sourceGeo.vertexAttribs().begin(GA_SCOPE_PUBLIC);

    // Vertex attributes
    for (; !it.atEnd(); ++it) {
        const GA_Attribute* sourceAttr = it.attrib();
#if (UT_VERSION_INT >= 0x0e0000b0) // 14.0.176 or later
        if (NULL == targetGeo.findVertexAttribute(it.name())) {
            targetGeo.addVertexAttrib(sourceAttr);
        }
        GA_Attribute* targetAttr = targetGeo.findVertexAttribute(it.name());
#else
        if (!targetGeo.findVertexAttribute(it.name()).isValid()) {
            targetGeo.addVertexAttrib(sourceAttr);
        }
        GA_Attribute* targetAttr = targetGeo.findVertexAttribute(it.name()).getAttribute();
#endif

        if (sourceAttr && targetAttr) {
#if (UT_VERSION_INT >= 0x0c01007D) // 12.1.125 or later
            targetAttr->hardenAllPages();
#endif
            AttributeCopyBase::Ptr att = createAttributeCopier(*sourceAttr, *targetAttr);
            if(att) vertAttributeList.push_back(att);
        }
    }

    if (!boss.wasInterrupted() && (!primAttributeList.empty() || !vertAttributeList.empty())) {

        UTparallelFor(GA_SplittableRange(targetGeo.getPrimitiveRange(primitives)),
            TransferPrimitiveAttributesOp<GridType>(sourceGeo, targetGeo, indexGrid,
                primAttributeList, vertAttributeList));
    }

    if (!boss.wasInterrupted()) {
        std::vector<AttributeCopyBase::Ptr> pointAttributeList;
        it = sourceGeo.pointAttribs().begin(GA_SCOPE_PUBLIC);

        // Point attributes
        for (; !it.atEnd(); ++it) {
            if (std::string(it.name()) == "P") continue; // Ignore previous point positions.

            const GA_Attribute* sourceAttr = it.attrib();
#if (UT_VERSION_INT >= 0x0e0000b0) // 14.0.176 or later
            if (NULL == targetGeo.findPointAttribute(it.name())) {
                targetGeo.addPointAttrib(sourceAttr);
            }
            GA_Attribute* targetAttr = targetGeo.findPointAttribute(it.name());
#else
            if (!targetGeo.findPointAttribute(it.name()).isValid()) {
                targetGeo.addPointAttrib(sourceAttr);
            }
            GA_Attribute* targetAttr = targetGeo.findPointAttribute(it.name()).getAttribute();
#endif

            if (sourceAttr && targetAttr) {
                AttributeCopyBase::Ptr att = createAttributeCopier(*sourceAttr, *targetAttr);
                if(att) pointAttributeList.push_back(att);
            }
        }

        if (!boss.wasInterrupted() && !pointAttributeList.empty()) {
            UTparallelFor(GA_SplittableRange(targetGeo.getPointRange()),
               TransferPointAttributesOp<GridType>(sourceGeo, targetGeo, indexGrid,
                    pointAttributeList, primitives));

        }
    }
}

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_ATTRIBUTE_TRANSFER_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
