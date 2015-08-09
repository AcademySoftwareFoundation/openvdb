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
/// @file Utils.h
/// @author FX R&D Simulation team
/// @brief Utility classes and functions for OpenVDB plugins

#ifndef OPENVDB_HOUDINI_UTILS_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_UTILS_HAS_BEEN_INCLUDED

#include "GU_PrimVDB.h"
#include <UT/UT_Interrupt.h>
#include <openvdb/openvdb.h>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_const.hpp>


#ifdef SESI_OPENVDB
#ifdef OPENVDB_HOUDINI_API
    #undef OPENVDB_HOUDINI_API
    #define OPENVDB_HOUDINI_API
#endif
#endif

class GEO_PrimVDB;
class GU_Detail;
class UT_String;

namespace openvdb_houdini {

typedef openvdb::GridBase           Grid;
typedef openvdb::GridBase::Ptr      GridPtr;
typedef openvdb::GridBase::ConstPtr GridCPtr;
typedef openvdb::GridBase&          GridRef;
typedef const openvdb::GridBase&    GridCRef;


/// @brief Iterator over const VDB primitives on a geometry detail
///
/// @details At least until @c GEO_PrimVDB becomes a built-in primitive type
/// (that can be used as the mask for a @c GA_GBPrimitiveIterator), use this
/// iterator to iterate over all VDB grids belonging to a gdp and, optionally,
/// belonging to a particular group.
class OPENVDB_HOUDINI_API VdbPrimCIterator
{
public:
    typedef boost::function<bool (const GU_PrimVDB&)> FilterFunc;

    /// @param gdp
    ///     the geometry detail over which to iterate
    /// @param group
    ///     a group in the detail over which to iterate (if @c NULL,
    ///     iterate over all VDB primitives)
    /// @param filter
    ///     an optional function or functor that takes a const reference
    ///     to a GU_PrimVDB and returns a boolean specifying whether
    ///     that primitive should be visited (@c true) or not (@c false)
    explicit VdbPrimCIterator(const GEO_Detail* gdp, const GA_PrimitiveGroup* group = NULL,
        FilterFunc filter = FilterFunc());

    VdbPrimCIterator(const VdbPrimCIterator&);
    VdbPrimCIterator& operator=(const VdbPrimCIterator&);

    //@{
    /// Advance to the next VDB primitive.
    void advance();
    VdbPrimCIterator& operator++() { advance(); return *this; }
    //@}

    //@{
    /// Return a pointer to the current VDB primitive (@c NULL if at end).
    const GU_PrimVDB* getPrimitive() const;
    const GU_PrimVDB* operator*() const { return getPrimitive(); }
    const GU_PrimVDB* operator->() const { return getPrimitive(); }
    //@}

    //@{
    GA_Offset getOffset() const { return getPrimitive()->getMapOffset(); }
    GA_Index getIndex() const { return getPrimitive()->getMapIndex(); }
    //@}

    /// Return @c false if there are no more VDB primitives.
    operator bool() const { return getPrimitive() != NULL; }

    /// @brief Return the value of the current VDB primitive's @c name attribute.
    /// @param defaultName
    ///     if the current primitive has no @c name attribute
    ///     or its name is empty, return this name instead
    UT_String getPrimitiveName(const UT_String& defaultName = "") const;

    /// @brief Return the value of the current VDB primitive's @c name attribute
    /// or, if the name is empty, the primitive's index (as a UT_String).
    UT_String getPrimitiveNameOrIndex() const;

protected:
    /// Allow primitives to be deleted during iteration.
    VdbPrimCIterator(const GEO_Detail*, GA_Range::safedeletions,
        const GA_PrimitiveGroup* = NULL, FilterFunc = FilterFunc());

    boost::shared_ptr<GA_GBPrimitiveIterator> mIter;
    FilterFunc mFilter;
}; // class VdbPrimCIterator


/// @brief Iterator over non-const VDB primitives on a geometry detail
///
/// @details At least until @c GEO_PrimVDB becomes a built-in primitive type
/// (that can be used as the mask for a @c GA_GBPrimitiveIterator), use this
/// iterator to iterate over all VDB grids belonging to a gdp and, optionally,
/// belonging to a particular group.
class OPENVDB_HOUDINI_API VdbPrimIterator: public VdbPrimCIterator
{
public:
    /// @param gdp
    ///     the geometry detail over which to iterate
    /// @param group
    ///     a group in the detail over which to iterate (if @c NULL,
    ///     iterate over all VDB primitives)
    /// @param filter
    ///     an optional function or functor that takes a @c const reference
    ///     to a GU_PrimVDB and returns a boolean specifying whether
    ///     that primitive should be visited (@c true) or not (@c false)
    explicit VdbPrimIterator(GEO_Detail* gdp, const GA_PrimitiveGroup* group = NULL,
        FilterFunc filter = FilterFunc()):
        VdbPrimCIterator(gdp, group, filter) {}
    /// @brief Allow primitives to be deleted during iteration.
    /// @param gdp
    ///     the geometry detail over which to iterate
    /// @param group
    ///     a group in the detail over which to iterate (if @c NULL,
    ///     iterate over all VDB primitives)
    /// @param filter
    ///     an optional function or functor that takes a @c const reference
    ///     to a GU_PrimVDB and returns a boolean specifying whether
    ///     that primitive should be visited (@c true) or not (@c false)
    VdbPrimIterator(GEO_Detail* gdp, GA_Range::safedeletions,
        const GA_PrimitiveGroup* group = NULL, FilterFunc filter = FilterFunc()):
        VdbPrimCIterator(gdp, GA_Range::safedeletions(), group, filter) {}

    VdbPrimIterator(const VdbPrimIterator&);
    VdbPrimIterator& operator=(const VdbPrimIterator&);

    /// Advance to the next VDB primitive.
    VdbPrimIterator& operator++() { advance(); return *this; }

    //@{
    /// Return a pointer to the current VDB primitive (@c NULL if at end).
    GU_PrimVDB* getPrimitive() const {
        return const_cast<GU_PrimVDB*>(VdbPrimCIterator::getPrimitive());
    }
    GU_PrimVDB* operator*() const { return getPrimitive(); }
    GU_PrimVDB* operator->() const { return getPrimitive(); }
    //@}
}; // class VdbPrimIterator


////////////////////////////////////////


/// @brief Wrapper class that adapts a Houdini @c UT_Interrupt object
/// for use with OpenVDB library routines
/// @sa openvdb/util/NullInterrupter.h
class Interrupter
{
public:
    Interrupter(const char* title = NULL):
        mUTI(UTgetInterrupt()), mRunning(false)
    {
        if (title) mUTI->setAppTitle(title);
    }
    ~Interrupter() { if (mRunning) this->end(); }

    /// @brief Signal the start of an interruptible operation.
    /// @param name  an optional descriptive name for the operation
    void start(const char* name = NULL) { if (!mRunning) { mRunning=true; mUTI->opStart(name); } }
    /// Signal the end of an interruptible operation.
    void end() { if (mRunning) { mUTI->opEnd(); mRunning = false; } }

    /// @brief Check if an interruptible operation should be aborted.
    /// @param percent  an optional (when >= 0) percentage indicating
    ///     the fraction of the operation that has been completed
    bool wasInterrupted(int percent=-1) { return mUTI->opInterrupt(percent); }

private:
    UT_Interrupt* mUTI;
    bool mRunning;
};


////////////////////////////////////////


// Utility methods

/// @brief Store a VDB grid in a new VDB primitive and add the primitive
/// to a geometry detail.
/// @return the newly-created VDB primitive.
/// @param gdp        the detail to which to add the primitive
/// @param grid       the VDB grid to be added
/// @param name       if non-null, set the new primitive's @c name attribute to this string
/// @note This operation clears the input grid's metadata.
OPENVDB_HOUDINI_API
GU_PrimVDB* createVdbPrimitive(GU_Detail& gdp, GridPtr grid, const char* name = NULL);


/// @brief Replace an existing VDB primitive with a new primitive that contains
/// the given grid.
/// @return the newly-created VDB primitive.
/// @param gdp        the detail to which to add the primitive
/// @param grid       the VDB grid to be added
/// @param src        replace this primitive with the newly-created primitive
/// @param copyAttrs  if @c true, copy attributes and group membership from the @a src primitive
/// @param name       if non-null, set the new primitive's @c name attribute to this string;
///                   otherwise, if @a copyAttrs is @c true, copy the name from @a src
/// @note This operation clears the input grid's metadata.
OPENVDB_HOUDINI_API
GU_PrimVDB* replaceVdbPrimitive(GU_Detail& gdp, GridPtr grid, GEO_PrimVDB& src,
    const bool copyAttrs = true, const char* name = NULL);


/// @brief Return in @a corners the corners of the given grid's active voxel bounding box.
/// @return @c false if the grid has no active voxels.
OPENVDB_HOUDINI_API
bool evalGridBBox(GridCRef grid, UT_Vector3 corners[8], bool expandHalfVoxel = false);


/// Construct an index-space CoordBBox from a UT_BoundingBox.
OPENVDB_HOUDINI_API
openvdb::CoordBBox makeCoordBBox(const UT_BoundingBox&, const openvdb::math::Transform&);


////////////////////////////////////////


/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType, bool IsConst/*=false*/>
struct GridProcessor {
    static inline void call(OpType& op, GridPtr grid) {
#ifdef _MSC_VER
        op.operator()<GridType>(openvdb::gridPtrCast<GridType>(grid));
#else
        op.template operator()<GridType>(openvdb::gridPtrCast<GridType>(grid));
#endif
    }
};

/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType>
struct GridProcessor<GridType, OpType, /*IsConst=*/true> {
    static inline void call(OpType& op, GridCPtr grid) {
#ifdef _MSC_VER
        op.operator()<GridType>(openvdb::gridConstPtrCast<GridType>(grid));
#else
        op.template operator()<GridType>(openvdb::gridConstPtrCast<GridType>(grid));
#endif
    }
};


/// Helper function used internally by processTypedGrid()
template<typename GridType, typename OpType, typename GridPtrType>
inline void
doProcessTypedGrid(GridPtrType grid, OpType& op)
{
    GridProcessor<GridType, OpType,
        boost::is_const<typename GridPtrType::element_type>::value>::call(op, grid);
}


////////////////////////////////////////


/// @brief Utility function that, given a generic grid pointer,
/// calls a functor on the fully-resolved grid
///
/// @par Example:
/// @code
/// using openvdb::Coord;
/// using openvdb::CoordBBox;
///
/// struct FillOp {
///     const CoordBBox bbox;
///
///     FillOp(const CoordBBox& b): bbox(b) {}
///
///     template<typename GridT>
///     void operator()(typename GridT::Ptr grid) const {
///         typedef typename GridT::ValueType ValueT;
///         grid->fill(bbox, ValueT(1));
///     }
/// };
///
/// CoordBBox bbox(Coord(0,0,0), Coord(10,10,10));
/// processTypedGrid(myGridPtr, FillOp(bbox));
/// @endcode
///
/// @return @c false if the grid type is unknown or unhandled.
/// @deprecated Use UTvdbProcessTypedGrid() or GEOvdbProcessTypedGrid() instead.
template<typename GridPtrType, typename OpType>
OPENVDB_DEPRECATED
bool
processTypedGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<BoolGrid>())        doProcessTypedGrid<BoolGrid>(grid, op);
    else if (grid->template isType<FloatGrid>())  doProcessTypedGrid<FloatGrid>(grid, op);
    else if (grid->template isType<DoubleGrid>()) doProcessTypedGrid<DoubleGrid>(grid, op);
    else if (grid->template isType<Int32Grid>())  doProcessTypedGrid<Int32Grid>(grid, op);
    else if (grid->template isType<Int64Grid>())  doProcessTypedGrid<Int64Grid>(grid, op);
    else if (grid->template isType<Vec3IGrid>())  doProcessTypedGrid<Vec3IGrid>(grid, op);
    else if (grid->template isType<Vec3SGrid>())  doProcessTypedGrid<Vec3SGrid>(grid, op);
    else if (grid->template isType<Vec3DGrid>())  doProcessTypedGrid<Vec3DGrid>(grid, op);
    else return false; ///< @todo throw exception ("unknown grid type")
    return true;
}


/// @brief Utility function that, given a generic grid pointer, calls
/// a functor on the fully-resolved grid, provided that the grid's
/// voxel values are 3-vectors (vec3i, vec3s or vec3d)
///
/// Usage:
/// @code
/// struct NormalizeOp {
///     template<typename GridT>
///     void operator()(typename GridT::Ptr grid) const { normalizeVectors(*grid); }
/// };
///
/// processTypedVec3Grid(myGridPtr, NormalizeOp());
/// @endcode
///
/// @return @c false if the grid type is unknown or non-vector.
/// @sa UTvdbProcessTypedGridVec3
/// @deprecated Use UTvdbProcessTypedGridVec3() or GEOvdbProcessTypedGridVec3() instead.
template<typename GridPtrType, typename OpType>
OPENVDB_DEPRECATED
bool
processTypedVec3Grid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<Vec3IGrid>())       doProcessTypedGrid<Vec3IGrid>(grid, op);
    else if (grid->template isType<Vec3SGrid>())  doProcessTypedGrid<Vec3SGrid>(grid, op);
    else if (grid->template isType<Vec3DGrid>())  doProcessTypedGrid<Vec3DGrid>(grid, op);
    else return false; ///< @todo throw exception ("grid type is not vec3")
    return true;
}


/// @brief Utility function that, given a generic grid pointer,
/// calls a functor on the fully-resolved grid
///
/// @par Example:
/// @code
/// using openvdb::Coord;
/// using openvdb::CoordBBox;
///
/// struct FillOp {
///     const CoordBBox bbox;
///
///     FillOp(const CoordBBox& b): bbox(b) {}
///
///     template<typename GridT>
///     void operator()(typename GridT::Ptr grid) const {
///         typedef typename GridT::ValueType ValueT;
///         grid->fill(bbox, ValueT(1));
///     }
/// };
///
/// CoordBBox bbox(Coord(0,0,0), Coord(10,10,10));
/// processTypedScalarGrid(myGridPtr, FillOp(bbox));
/// @endcode
///
/// @return @c false if the grid type is unknown or non-scalar.
/// @deprecated Use UTvdbProcessTypedGridScalar() or GEOvdbProcessTypedGridScalar() instead.
template<typename GridPtrType, typename OpType>
OPENVDB_DEPRECATED
bool
processTypedScalarGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<FloatGrid>())       doProcessTypedGrid<FloatGrid>(grid, op);
    else if (grid->template isType<DoubleGrid>()) doProcessTypedGrid<DoubleGrid>(grid, op);
    else if (grid->template isType<Int32Grid>())  doProcessTypedGrid<Int32Grid>(grid, op);
    else if (grid->template isType<Int64Grid>())  doProcessTypedGrid<Int64Grid>(grid, op);
    else return false; ///< @todo throw exception ("grid type is not scalar")
    return true;
}

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_UTILS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
