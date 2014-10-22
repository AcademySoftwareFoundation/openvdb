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
/// @file Composite.h
///
/// @brief Functions to efficiently perform various compositing operations on grids
///
/// @author Peter Cucka

#ifndef OPENVDB_TOOLS_COMPOSITE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_COMPOSITE_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h> // for isExactlyEqual()
#include "ValueTransformer.h" // for transformValues()
#include "Prune.h"// for prune
#include <boost/utility/enable_if.hpp>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Given two level set grids, replace the A grid with the union of A and B.
/// @throw ValueError if the background value of either grid is not greater than zero.
/// @note This operation always leaves the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void csgUnion(GridOrTreeT& a, GridOrTreeT& b, bool prune = true);
/// @brief Given two level set grids, replace the A grid with the intersection of A and B.
/// @throw ValueError if the background value of either grid is not greater than zero.
/// @note This operation always leaves the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void csgIntersection(GridOrTreeT& a, GridOrTreeT& b, bool prune = true);
/// @brief Given two level set grids, replace the A grid with the difference A / B.
/// @throw ValueError if the background value of either grid is not greater than zero.
/// @note This operation always leaves the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void csgDifference(GridOrTreeT& a, GridOrTreeT& b, bool prune = true);

/// @brief Given grids A and B, compute max(a, b) per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compMax(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute min(a, b) per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compMin(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute a + b per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compSum(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute a * b per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compMul(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute a / b per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compDiv(GridOrTreeT& a, GridOrTreeT& b);

/// Copy the active voxels of B into A.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compReplace(GridOrTreeT& a, const GridOrTreeT& b);


////////////////////////////////////////


namespace composite {

// composite::min() and composite::max() for non-vector types compare with operator<().
template<typename T> inline
const typename boost::disable_if_c<VecTraits<T>::IsVec, T>::type& // = T if T is not a vector type
min(const T& a, const T& b) { return std::min(a, b); }

template<typename T> inline
const typename boost::disable_if_c<VecTraits<T>::IsVec, T>::type&
max(const T& a, const T& b) { return std::max(a, b); }


// composite::min() and composite::max() for OpenVDB vector types compare by magnitude.
template<typename T> inline
const typename boost::enable_if_c<VecTraits<T>::IsVec, T>::type& // = T if T is a vector type
min(const T& a, const T& b)
{
    const typename T::ValueType aMag = a.lengthSqr(), bMag = b.lengthSqr();
    return (aMag < bMag ? a : (bMag < aMag ? b : std::min(a, b)));
}

template<typename T> inline
const typename boost::enable_if_c<VecTraits<T>::IsVec, T>::type&
max(const T& a, const T& b)
{
    const typename T::ValueType aMag = a.lengthSqr(), bMag = b.lengthSqr();
    return (aMag < bMag ? b : (bMag < aMag ? a : std::max(a, b)));
}


template<typename T> inline
typename boost::disable_if<boost::is_integral<T>, T>::type // = T if T is not an integer type
divide(const T& a, const T& b) { return a / b; }

template<typename T> inline
typename boost::enable_if<boost::is_integral<T>, T>::type // = T if T is an integer type
divide(const T& a, const T& b)
{
    const T zero(0);
    if (b != zero) return a / b;
    if (a == zero) return 0;
    return (a > 0 ? std::numeric_limits<T>::max() : -std::numeric_limits<T>::max());
}

// If b is true, return a / 1 = a.
// If b is false and a is true, return 1 / 0 = inf = MAX_BOOL = 1 = a.
// If b is false and a is false, return 0 / 0 = NaN = 0 = a.
inline bool divide(bool a, bool /*b*/) { return a; }

} // namespace composite


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compMax(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT>    Adapter;
    typedef typename Adapter::TreeType  TreeT;
    typedef typename TreeT::ValueType   ValueT;
    struct Local {
        static inline void op(CombineArgs<ValueT>& args) {
            args.setResult(composite::max(args.a(), args.b()));
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compMin(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT>    Adapter;
    typedef typename Adapter::TreeType  TreeT;
    typedef typename TreeT::ValueType   ValueT;
    struct Local {
        static inline void op(CombineArgs<ValueT>& args) {
            args.setResult(composite::min(args.a(), args.b()));
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compSum(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    struct Local {
        static inline void op(CombineArgs<typename TreeT::ValueType>& args) {
            args.setResult(args.a() + args.b());
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compMul(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    struct Local {
        static inline void op(CombineArgs<typename TreeT::ValueType>& args) {
            args.setResult(args.a() * args.b());
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compDiv(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    struct Local {
        static inline void op(CombineArgs<typename TreeT::ValueType>& args) {
            args.setResult(composite::divide(args.a(), args.b()));
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


////////////////////////////////////////


template<typename TreeT>
struct CompReplaceOp
{
    TreeT* const aTree;

    CompReplaceOp(TreeT& _aTree): aTree(&_aTree) {}

    void operator()(const typename TreeT::ValueOnCIter& iter) const
    {
        CoordBBox bbox;
        iter.getBoundingBox(bbox);
        aTree->fill(bbox, *iter);
    }

    void operator()(const typename TreeT::LeafCIter& leafIter) const
    {
        tree::ValueAccessor<TreeT> acc(*aTree);
        for (typename TreeT::LeafCIter::LeafNodeT::ValueOnCIter iter =
            leafIter->cbeginValueOn(); iter; ++iter)
        {
            acc.setValue(iter.getCoord(), *iter);
        }
    }
};


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compReplace(GridOrTreeT& aTree, const GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    typedef typename TreeT::ValueOnCIter ValueOnCIterT;

    // Copy active states (but not values) from B to A.
    Adapter::tree(aTree).topologyUnion(Adapter::tree(bTree));

    CompReplaceOp<TreeT> op(Adapter::tree(aTree));

    // Copy all active tile values from B to A.
    ValueOnCIterT iter = bTree.cbeginValueOn();
    iter.setMaxDepth(iter.getLeafDepth() - 1); // don't descend into leaf nodes
    foreach(iter, op);

    // Copy all active voxel values from B to A.
    foreach(Adapter::tree(bTree).cbeginLeaf(), op);
}


////////////////////////////////////////


/// Base visitor class for CSG operations
/// (not intended to be used polymorphically, so no virtual functions)
template<typename TreeType>
class CsgVisitorBase
{
public:
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = 3 };

    CsgVisitorBase(const TreeT& aTree, const TreeT& bTree):
        mAOutside(aTree.background()),
        mAInside(math::negative(mAOutside)),
        mBOutside(bTree.background()),
        mBInside(math::negative(mBOutside))
    {
        const ValueT zero = zeroVal<ValueT>();
        if (!(mAOutside > zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid A outside value > 0, got " << mAOutside);
        }
        if (!(mAInside < zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid A inside value < 0, got " << mAInside);
        }
        if (!(mBOutside > zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid B outside value > 0, got " << mBOutside);
        }
        if (!(mBInside < zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid B outside value < 0, got " << mBOutside);
        }
    }

protected:
    ValueT mAOutside, mAInside, mBOutside, mBInside;
};


////////////////////////////////////////


template<typename TreeType>
struct CsgUnionVisitor: public CsgVisitorBase<TreeType>
{
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = CsgVisitorBase<TreeT>::STOP };

    CsgUnionVisitor(const TreeT& a, const TreeT& b): CsgVisitorBase<TreeT>(a, b) {}

    /// Don't process nodes that are at different tree levels.
    template<typename AIterT, typename BIterT>
    inline int operator()(AIterT&, BIterT&) { return 0; }

    /// Process root and internal nodes.
    template<typename IterT>
    inline int operator()(IterT& aIter, IterT& bIter)
    {
        ValueT aValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* aChild = aIter.probeChild(aValue);
        if (!aChild && aValue < zeroVal<ValueT>()) {
            // A is an inside tile.  Leave it alone and stop traversing this branch.
            return STOP;
        }

        ValueT bValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* bChild = bIter.probeChild(bValue);
        if (!bChild && bValue < zeroVal<ValueT>()) {
            // B is an inside tile.  Make A an inside tile and stop traversing this branch.
            aIter.setValue(this->mAInside);
            aIter.setValueOn(bIter.isValueOn());
            delete aChild;
            return STOP;
        }

        if (!aChild && aValue > zeroVal<ValueT>()) {
            // A is an outside tile.  If B has a child, transfer it to A,
            // otherwise leave A alone.
            if (bChild) {
                bIter.setValue(this->mBOutside);
                bIter.setValueOff();
                bChild->resetBackground(this->mBOutside, this->mAOutside);
                aIter.setChild(bChild); // transfer child
                delete aChild;
            }
            return STOP;
        }

        // If A has a child and B is an outside tile, stop traversing this branch.
        // Continue traversal only if A and B both have children.
        return (aChild && bChild) ? 0 : STOP;
    }

    /// Process leaf node values.
    inline int operator()(ChildIterT& aIter, ChildIterT& bIter)
    {
        ValueT aValue, bValue;
        aIter.probeValue(aValue);
        bIter.probeValue(bValue);
        if (aValue > bValue) { // a = min(a, b)
            aIter.setValue(bValue);
            aIter.setValueOn(bIter.isValueOn());
        }
        return 0;
    }
};



////////////////////////////////////////


template<typename TreeType>
struct CsgIntersectVisitor: public CsgVisitorBase<TreeType>
{
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = CsgVisitorBase<TreeT>::STOP };

    CsgIntersectVisitor(const TreeT& a, const TreeT& b): CsgVisitorBase<TreeT>(a, b) {}

    /// Don't process nodes that are at different tree levels.
    template<typename AIterT, typename BIterT>
    inline int operator()(AIterT&, BIterT&) { return 0; }

    /// Process root and internal nodes.
    template<typename IterT>
    inline int operator()(IterT& aIter, IterT& bIter)
    {
        ValueT aValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* aChild = aIter.probeChild(aValue);
        if (!aChild && !(aValue < zeroVal<ValueT>())) {
            // A is an outside tile.  Leave it alone and stop traversing this branch.
            return STOP;
        }

        ValueT bValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* bChild = bIter.probeChild(bValue);
        if (!bChild && !(bValue < zeroVal<ValueT>())) {
            // B is an outside tile.  Make A an outside tile and stop traversing this branch.
            aIter.setValue(this->mAOutside);
            aIter.setValueOn(bIter.isValueOn());
            delete aChild;
            return STOP;
        }

        if (!aChild && aValue < zeroVal<ValueT>()) {
            // A is an inside tile.  If B has a child, transfer it to A,
            // otherwise leave A alone.
            if (bChild) {
                bIter.setValue(this->mBOutside);
                bIter.setValueOff();
                bChild->resetBackground(this->mBOutside, this->mAOutside);
                aIter.setChild(bChild); // transfer child
                delete aChild;
            }
            return STOP;
        }

        // If A has a child and B is an outside tile, stop traversing this branch.
        // Continue traversal only if A and B both have children.
        return (aChild && bChild) ? 0 : STOP;
    }

    /// Process leaf node values.
    inline int operator()(ChildIterT& aIter, ChildIterT& bIter)
    {
        ValueT aValue, bValue;
        aIter.probeValue(aValue);
        bIter.probeValue(bValue);
        if (aValue < bValue) { // a = max(a, b)
            aIter.setValue(bValue);
            aIter.setValueOn(bIter.isValueOn());
        }
        return 0;
    }
};


////////////////////////////////////////


template<typename TreeType>
struct CsgDiffVisitor: public CsgVisitorBase<TreeType>
{
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = CsgVisitorBase<TreeT>::STOP };

    CsgDiffVisitor(const TreeT& a, const TreeT& b): CsgVisitorBase<TreeT>(a, b) {}

    /// Don't process nodes that are at different tree levels.
    template<typename AIterT, typename BIterT>
    inline int operator()(AIterT&, BIterT&) { return 0; }

    /// Process root and internal nodes.
    template<typename IterT>
    inline int operator()(IterT& aIter, IterT& bIter)
    {
        ValueT aValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* aChild = aIter.probeChild(aValue);
        if (!aChild && !(aValue < zeroVal<ValueT>())) {
            // A is an outside tile.  Leave it alone and stop traversing this branch.
            return STOP;
        }

        ValueT bValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* bChild = bIter.probeChild(bValue);
        if (!bChild && bValue < zeroVal<ValueT>()) {
            // B is an inside tile.  Make A an inside tile and stop traversing this branch.
            aIter.setValue(this->mAOutside);
            aIter.setValueOn(bIter.isValueOn());
            delete aChild;
            return STOP;
        }

        if (!aChild && aValue < zeroVal<ValueT>()) {
            // A is an inside tile.  If B has a child, transfer it to A,
            // otherwise leave A alone.
            if (bChild) {
                bIter.setValue(this->mBOutside);
                bIter.setValueOff();
                bChild->resetBackground(this->mBOutside, this->mAOutside);
                aIter.setChild(bChild); // transfer child
                bChild->negate();
                delete aChild;
            }
            return STOP;
        }

        // If A has a child and B is an outside tile, stop traversing this branch.
        // Continue traversal only if A and B both have children.
        return (aChild && bChild) ? 0 : STOP;
    }

    /// Process leaf node values.
    inline int operator()(ChildIterT& aIter, ChildIterT& bIter)
    {
        ValueT aValue, bValue;
        aIter.probeValue(aValue);
        bIter.probeValue(bValue);
        bValue = math::negative(bValue);
        if (aValue < bValue) { // a = max(a, -b)
            aIter.setValue(bValue);
            aIter.setValueOn(bIter.isValueOn());
        }
        return 0;
    }
};


////////////////////////////////////////


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
csgUnion(GridOrTreeT& a, GridOrTreeT& b, bool prune)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    TreeT &aTree = Adapter::tree(a), &bTree = Adapter::tree(b);
    CsgUnionVisitor<TreeT> visitor(aTree, bTree);
    aTree.visit2(bTree, visitor);
    if (prune) tools::pruneLevelSet(aTree);
}

template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
csgIntersection(GridOrTreeT& a, GridOrTreeT& b, bool prune)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    TreeT &aTree = Adapter::tree(a), &bTree = Adapter::tree(b);
    CsgIntersectVisitor<TreeT> visitor(aTree, bTree);
    aTree.visit2(bTree, visitor);
    if (prune) tools::pruneLevelSet(aTree);
}

template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
csgDifference(GridOrTreeT& a, GridOrTreeT& b, bool prune)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    TreeT &aTree = Adapter::tree(a), &bTree = Adapter::tree(b);
    CsgDiffVisitor<TreeT> visitor(aTree, bTree);
    aTree.visit2(bTree, visitor);
    if (prune) tools::pruneLevelSet(aTree);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_COMPOSITE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
