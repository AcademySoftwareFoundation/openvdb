///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file GU_VDBPointTools.h
/// @author FX R&D OpenVDB team
///
/// @brief Collection of PointIndexGrid helpers for Houdini

#ifndef __GU_VDBPOINTTOOLS_H_HAS_BEEN_INCLUDED__
#define __GU_VDBPOINTTOOLS_H_HAS_BEEN_INCLUDED__

#if defined(SESI_OPENVDB)
    #include "GU_Detail.h"
#else
    #include <GU/GU_Detail.h>
#endif
#include <GA/GA_ElementGroup.h>
#include <UT/UT_VectorTypes.h>

#include <openvdb/Platform.h>
#include <openvdb/tools/PointIndexGrid.h>


/// @brief Houdini point attribute wrapper
template <typename VectorType>
struct GU_VDBPointList {
    typedef VectorType ValueType;   // OpenVDB convention.
    typedef VectorType value_type;  // STL convention.

    GU_VDBPointList(const GU_Detail& detail, const GA_PointGroup* group = NULL)
        : mPositionHandle(detail.getP())
        , mVelocityHandle()
        , mRadiusHandle()
        , mIndexMap(&detail.getP()->getIndexMap())
        , mOffsets()
        , mSize(mIndexMap->indexSize())
    {
        if (group) {
            mSize = group->entries();
            mOffsets.reserve(mSize);

            GA_Offset start, end;
            GA_Range range(*group);
            for (GA_Iterator it = range.begin(); it.blockAdvance(start, end); ) {
                for (GA_Offset off = start; off < end; ++off) {
                    mOffsets.push_back(off);
                }
            }

            getOffset = &GU_VDBPointList::offsetFromGroupMap;
        } else if (mIndexMap->isTrivialMap()) {
            getOffset = &GU_VDBPointList::offsetFromIndexCast;
        } else {
            getOffset = &GU_VDBPointList::offsetFromGeoMap;
        }

        // Bind optional attributes

        GA_ROAttributeRef velRef = detail.findFloatTuple(GA_ATTRIB_POINT, GEO_STD_ATTRIB_VELOCITY, 3);
        if (velRef.isValid()) {
            mVelocityHandle.bind(velRef.getAttribute());
        }

        GA_ROAttributeRef radRef = detail.findFloatTuple(GA_ATTRIB_POINT, GEO_STD_ATTRIB_PSCALE);
        if (radRef.isValid()) {
            mRadiusHandle.bind(radRef.getAttribute());
        }
    }


    size_t size() const { return mSize; }

    bool hasVelocity() const { return mVelocityHandle.isValid(); }
    bool hasRadius() const { return mRadiusHandle.isValid(); }

    // Index access methods

    void getPos(size_t n, VectorType& xyz) const {
        getPosFromOffset((this->*getOffset)(n), xyz);
    }

    void getVelocity(size_t n, VectorType& v) const {
        getVelocityFromOffset((this->*getOffset)(n), v);
    }

    void getRadius(size_t n, float& r) const {
        getRadiusFromOffset((this->*getOffset)(n), r);
    }

    // Offset access methods

    GA_Offset offsetFromIndex(size_t n) const {
        return (this->*getOffset)(n);
    }

    void getPosFromOffset(const GA_Offset offset, VectorType& xyz) const {
        const UT_Vector3 data = mPositionHandle.get(offset);
        xyz[0] = typename VectorType::ValueType(data[0]);
        xyz[1] = typename VectorType::ValueType(data[1]);
        xyz[2] = typename VectorType::ValueType(data[2]);
    }

    void getVelocityFromOffset(const GA_Offset offset, VectorType& v) const {
        const UT_Vector3 data = mVelocityHandle.get(offset);
        v[0] = typename VectorType::ValueType(data[0]);
        v[1] = typename VectorType::ValueType(data[1]);
        v[2] = typename VectorType::ValueType(data[2]);
    }

    void getRadiusFromOffset(const GA_Offset offset, float& r) const {
        r = mRadiusHandle.get(offset);
    }


private:
    // Disallow copying
    GU_VDBPointList(const GU_VDBPointList&);
    GU_VDBPointList& operator=(const GU_VDBPointList&);

    GA_Offset (GU_VDBPointList::* getOffset)(const size_t) const;

    GA_Offset offsetFromGeoMap(const size_t n) const {
        return mIndexMap->offsetFromIndex(GA_Index(n));
    }

    GA_Offset offsetFromGroupMap(const size_t n) const {
        return mOffsets[n];
    }

    GA_Offset offsetFromIndexCast(const size_t n) const {
        return GA_Offset(n);
    }

    GA_ROHandleV3 mPositionHandle, mVelocityHandle;
    GA_ROHandleF mRadiusHandle;
    GA_IndexMap const * const mIndexMap;
    std::vector<GA_Offset> mOffsets;
    size_t mSize;
}; // GU_VDBPointList


////////////////////////////////////////


// PointIndexGrid utility methods


namespace GU_VDBPointToolsInternal {

template<typename PointArrayType>
struct IndexToOffsetOp {
    IndexToOffsetOp(const PointArrayType& points): mPointList(&points) {}

    template <typename LeafT>
    void operator()(LeafT &leaf, size_t /*leafIndex*/) const {
        typename LeafT::IndexArray& indices = leaf.indices();
        for (size_t n = 0, N = indices.size(); n < N; ++n) {
             indices[n] = typename LeafT::ValueType(mPointList->offsetFromIndex(size_t(indices[n])));
        }
    }
    PointArrayType const * const mPointList;
};

} // namespace GU_VDBPointToolsInternal


/// @brief Utility method to change point indices into Houdini geometry offsets
/// @note PointIndexGrid's that store Houdini geometry offsets are not
///       safe to write to disk, offsets are not guaranteed to be immutable
///       under defragmentation operations or I/O.
template<typename PointIndexTreeType, typename PointArrayType>
inline void
GUvdbConvertIndexToOffset(PointIndexTreeType& tree, const PointArrayType& points)
{
    openvdb::tree::LeafManager<PointIndexTreeType> leafnodes(tree);
    leafnodes.foreach(GU_VDBPointToolsInternal::IndexToOffsetOp<PointArrayType>(points));
}


/// Utility method to construct a PointIndexGrid
inline openvdb::tools::PointIndexGrid::Ptr
GUvdbCreatePointIndexGrid(
    const openvdb::math::Transform& xform,
    const GU_Detail& detail,
    const GA_PointGroup* pointGroup = NULL)
{
    GU_VDBPointList<openvdb::Vec3s> points(detail, pointGroup);
    return openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(points, xform);
}


/// @brief  Utility method to construct a PointIndexGrid that stores
///         Houdini geometry offsets.
///
/// @note PointIndexGrid's that store Houdini geometry offsets are not
///       safe to write to disk, offsets are not guaranteed to be immutable
///       under defragmentation operations or I/O.
inline openvdb::tools::PointIndexGrid::Ptr
GUvdbCreatePointOffsetGrid(
    const openvdb::math::Transform& xform,
    const GU_Detail& detail,
    const GA_PointGroup* pointGroup = NULL)
{
    GU_VDBPointList<openvdb::Vec3s> points(detail, pointGroup);

    openvdb::tools::PointIndexGrid::Ptr grid =
        openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(points, xform);

    GUvdbConvertIndexToOffset(grid->tree(), points);

    return grid;
}


#endif // __GU_VDBPOINTTOOLS_H_HAS_BEEN_INCLUDED__

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
