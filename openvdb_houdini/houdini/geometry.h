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
/// @file geoemetry.h
/// @author FX R&D OpenVDB team
///
/// @brief A collection of Houdini geometry related methods and helper functions.

#ifndef HOUDINI_UTILS_GEOMETRY_HAS_BEEN_INCLUDED
#define HOUDINI_UTILS_GEOMETRY_HAS_BEEN_INCLUDED

#include <UT/UT_VectorTypes.h>
#include <GU/GU_Detail.h>
#include <GA/GA_ElementGroup.h>

#if defined(PRODDEV_BUILD) || defined(DWREAL_IS_DOUBLE)
  // OPENVDB_HOUDINI_API, which has no meaning in a DWA build environment but
  // must at least exist, is normally defined by including openvdb/Platform.h.
  // For DWA builds (i.e., if either PRODDEV_BUILD or DWREAL_IS_DOUBLE exists),
  // that introduces an unwanted and unnecessary library dependency.
  #ifndef OPENVDB_HOUDINI_API
    #define OPENVDB_HOUDINI_API
  #endif
#else
  #include <openvdb/Platform.h>
#endif

namespace houdini_utils {


/// @brief Add geometry to the given GU_Detail to create a box with the given corners.
/// @param corners  the eight corners of the box
/// @param color    an optional color for the added geometry
/// @param shaded   if false, generate a wireframe box; otherwise, generate a solid box
/// @param alpha    an optional opacity for the added geometry
OPENVDB_HOUDINI_API void createBox(GU_Detail&, UT_Vector3 corners[8],
    const UT_Vector3* color = NULL, bool shaded = false, float alpha = 1.0);


/// @brief Houdini point attribute wrapper
template <typename VectorType>
struct PointList {
    typedef VectorType ValueType;   // OpenVDB convention.
    typedef VectorType value_type;  // STL convention.

    PointList(const GU_Detail& detail, const GA_PointGroup* group = NULL)
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

            getOffset = &PointList::offsetFromGroupMap;
        } else if (mIndexMap->isTrivialMap()) {
            getOffset = &PointList::offsetFromIndexCast;
        } else {
            getOffset = &PointList::offsetFromGeoMap;
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
        UT_Vector3 data = mPositionHandle.get(offset);
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
    PointList(const PointList&);
    PointList& operator=(const PointList&);

    GA_Offset (PointList::* getOffset)(const size_t) const;

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
}; // PointList


} // namespace houdini_utils

#endif // HOUDINI_UTILS_GEOMETRY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
