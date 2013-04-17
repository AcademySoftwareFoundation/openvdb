///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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
/// @file ValueTransformer.h
///
/// tools::foreach() and tools::transformValues() transform the values in a grid
/// by iterating over the grid with a user-supplied iterator and applying a
/// user-supplied functor at each step of the iteration.  With tools::foreach(),
/// the transformation is done in-place on the input grid, whereas with
/// tools::transformValues(), transformed values are written to an output grid
/// (which can, for example, have a different value type than the input grid).
/// Both functions can optionally transform multiple values of the grid in
/// parallel.

#ifndef OPENVDB_TOOLS_VALUETRANSFORMER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VALUETRANSFORMER_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// Iterate over a grid and at each step call @c op(iter).
/// @param iter      an iterator over a grid or its tree (@c Grid::ValueOnCIter,
///                  @c Tree::NodeIter, etc.)
/// @param op        a functor of the form <tt>void op(const IterT&)</tt>, where @c IterT is
///                  the type of @a iter
/// @param threaded  if true, transform multiple values of the grid in parallel
/// @param shareOp   if true and @a threaded is true, all threads use the same functor;
///                  otherwise, each thread gets its own copy of the functor
///
/// @par Example:
/// Multiply all values (both set and unset) of a scalar, floating-point grid by two.
/// @code
/// struct Local {
///     static inline void op(const FloatGrid::ValueAllIter& iter) {
///         iter.setValue(*iter * 2);
///     }
/// };
/// FloatGrid grid;
/// tools::foreach(grid.beginValueAll(), Local::op);
/// @endcode
///
/// @par Example:
/// Rotate all active vectors of a vector grid by 45 degrees about the y axis.
/// @code
/// namespace {
///     struct MatMul {
///         math::Mat3s M;
///         MatMul(const math::Mat3s& mat): M(mat) {}
///         inline void operator()(const VectorGrid::ValueOnIter& iter) const {
///             iter.setValue(M.transform(*iter));
///         }
///     };
/// }
/// {
///     VectorGrid grid;
///     tools::foreach(grid.beginValueOn(),
///         MatMul(math::rotation<math::Mat3s>(math::Y, M_PI_4)));
/// }
/// @endcode
///
/// @note For more complex operations that require finer control over threading,
/// consider using @c tbb::parallel_for() or @c tbb::parallel_reduce() in conjunction
/// with a tree::IteratorRange that wraps a grid or tree iterator.
template<typename IterT, typename XformOp>
inline void foreach(const IterT& iter, XformOp& op,
    bool threaded = true, bool shareOp = true);

template<typename IterT, typename XformOp>
inline void foreach(const IterT& iter, const XformOp& op,
    bool threaded = true, bool shareOp = true);


/// Iterate over a grid and at each step call <tt>op(iter, accessor)</tt> to
/// populate (via the accessor) the given output grid, whose @c ValueType
/// need not be the same as the input grid's.
/// @param inIter    a non-<tt>const</tt> or (preferably) @c const iterator over an
///                  input grid or its tree (@c Grid::ValueOnCIter, @c Tree::NodeIter, etc.)
/// @param outGrid   an empty grid to be populated
/// @param op        a functor of the form
///                  <tt>void op(const InIterT&, OutGridT::ValueAccessor&)</tt>,
///                  where @c InIterT is the type of @a inIter
/// @param threaded  if true, transform multiple values of the input grid in parallel
/// @param shareOp   if true and @a threaded is true, all threads use the same functor;
///                  otherwise, each thread gets its own copy of the functor
///
/// @par Example:
/// Populate a scalar floating-point grid with the lengths of the vectors from all
/// active voxels of a vector-valued input grid.
/// @code
/// struct Local {
///     static void op(
///         const Vec3fGrid::ValueOnCIter& iter,
///         FloatGrid::ValueAccessor& accessor)
///     {
///         if (iter.isVoxelValue()) { // set a single voxel
///             accessor.setValue(iter.getCoord(), iter->length());
///         } else { // fill an entire tile
///             CoordBBox bbox;
///             iter.getBoundingBox(bbox);
///             accessor.getTree()->fill(bbox, iter->length());
///         }
///     }
/// };
/// Vec3fGrid inGrid;
/// FloatGrid outGrid;
/// tools::transformValues(inGrid.cbeginValueOn(), outGrid, Local::op);
/// @endcode
///
/// @note For more complex operations that require finer control over threading,
/// consider using @c tbb::parallel_for() or @c tbb::parallel_reduce() in conjunction
/// with a tree::IteratorRange that wraps a grid or tree iterator.
template<typename InIterT, typename OutGridT, typename XformOp>
inline void transformValues(const InIterT& inIter, OutGridT& outGrid,
    XformOp& op, bool threaded = true, bool shareOp = true);

#ifndef _MSC_VER
template<typename InIterT, typename OutGridT, typename XformOp>
inline void transformValues(const InIterT& inIter, OutGridT& outGrid,
    const XformOp& op, bool threaded = true, bool shareOp = true);
#endif


////////////////////////////////////////


namespace valxform {

template<typename IterT, typename OpT>
class SharedOpApplier
{
public:
    typedef typename tree::IteratorRange<IterT> IterRange;

    SharedOpApplier(const IterT& iter, OpT& op): mIter(iter), mOp(op) {}

    void process(bool threaded = true)
    {
        IterRange range(mIter);
        if (threaded) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }

    void operator()(IterRange& r) const { for ( ; r; ++r) mOp(r.iterator()); }

private:
    IterT mIter;
    OpT& mOp;
};


template<typename IterT, typename OpT>
class CopyableOpApplier
{
public:
    typedef typename tree::IteratorRange<IterT> IterRange;

    CopyableOpApplier(const IterT& iter, const OpT& op): mIter(iter), mOp(op), mOrigOp(&op) {}

    // When splitting this task, give the subtask a copy of the original functor,
    // not of this task's functor, which might have been modified arbitrarily.
    CopyableOpApplier(const CopyableOpApplier& other):
        mIter(other.mIter), mOp(*other.mOrigOp), mOrigOp(other.mOrigOp) {}

    void process(bool threaded = true)
    {
        IterRange range(mIter);
        if (threaded) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }

    void operator()(IterRange& r) const { for ( ; r; ++r) mOp(r.iterator()); }

private:
    IterT mIter;
    OpT mOp; // copy of original functor
    OpT const * const mOrigOp; // pointer to original functor
};

} // namespace valxform


template<typename IterT, typename XformOp>
inline void
foreach(const IterT& iter, XformOp& op, bool threaded, bool shared)
{
    if (shared) {
        typename valxform::SharedOpApplier<IterT, XformOp> proc(iter, op);
        proc.process(threaded);
    } else {
        typedef typename valxform::CopyableOpApplier<IterT, XformOp> Processor;
        Processor proc(iter, op);
        proc.process(threaded);
    }
}

template<typename IterT, typename XformOp>
inline void
foreach(const IterT& iter, const XformOp& op, bool threaded, bool /*shared*/)
{
    // Const ops are shared across threads, not copied.
    typename valxform::SharedOpApplier<IterT, const XformOp> proc(iter, op);
    proc.process(threaded);
}


////////////////////////////////////////


namespace valxform {

template<typename InIterT, typename OutTreeT, typename OpT>
class SharedOpTransformer
{
public:
    typedef typename InIterT::TreeT InTreeT;
    typedef typename tree::IteratorRange<InIterT> IterRange;
    typedef typename OutTreeT::ValueType OutValueT;

    SharedOpTransformer(const InIterT& inIter, OutTreeT& outTree, OpT& op):
        mIsRoot(true),
        mInputIter(inIter),
        mInputTree(inIter.getTree()),
        mOutputTree(&outTree),
        mOp(op)
    {
        if (static_cast<const void*>(mInputTree) == static_cast<void*>(mOutputTree)) {
            OPENVDB_LOG_INFO("use tools::foreach(), not transformValues(),"
                " to transform a grid in place");
        }
    }

    /// Splitting constructor
    SharedOpTransformer(SharedOpTransformer& other, tbb::split):
        mIsRoot(false),
        mInputIter(other.mInputIter),
        mInputTree(other.mInputTree),
        mOutputTree(new OutTreeT(zeroVal<OutValueT>())),
        mOp(other.mOp)
        {}

    ~SharedOpTransformer()
    {
        // Delete the output tree only if it was allocated locally
        // (the top-level output tree was supplied by the caller).
        if (!mIsRoot) {
            delete mOutputTree;
            mOutputTree = NULL;
        }
    }

    void process(bool threaded = true)
    {
        if (!mInputTree || !mOutputTree) return;

        IterRange range(mInputIter);

        // Independently transform elements in the iterator range,
        // either in parallel or serially.
        if (threaded) {
            tbb::parallel_reduce(range, *this);
        } else {
            (*this)(range);
        }
    }

    /// Transform each element in the given range.
    void operator()(IterRange& range) const
    {
        if (!mOutputTree) return;
        typename tree::ValueAccessor<OutTreeT> outAccessor(*mOutputTree);
        for ( ; range; ++range) {
            mOp(range.iterator(), outAccessor);
        }
    }

    void join(const SharedOpTransformer& other)
    {
        if (mOutputTree && other.mOutputTree) {
            mOutputTree->merge(*other.mOutputTree);
        }
    }

private:
    bool mIsRoot;
    InIterT mInputIter;
    const InTreeT* mInputTree;
    OutTreeT* mOutputTree;
    OpT& mOp;
}; // class SharedOpTransformer


template<typename InIterT, typename OutTreeT, typename OpT>
class CopyableOpTransformer
{
public:
    typedef typename InIterT::TreeT InTreeT;
    typedef typename tree::IteratorRange<InIterT> IterRange;
    typedef typename OutTreeT::ValueType OutValueT;

    CopyableOpTransformer(const InIterT& inIter, OutTreeT& outTree, const OpT& op):
        mIsRoot(true),
        mInputIter(inIter),
        mInputTree(inIter.getTree()),
        mOutputTree(&outTree),
        mOp(op),
        mOrigOp(&op)
    {
        if (static_cast<const void*>(mInputTree) == static_cast<void*>(mOutputTree)) {
            OPENVDB_LOG_INFO("use tools::foreach(), not transformValues(),"
                " to transform a grid in place");
        }
    }

    // When splitting this task, give the subtask a copy of the original functor,
    // not of this task's functor, which might have been modified arbitrarily.
    CopyableOpTransformer(CopyableOpTransformer& other, tbb::split):
        mIsRoot(false),
        mInputIter(other.mInputIter),
        mInputTree(other.mInputTree),
        mOutputTree(new OutTreeT(zeroVal<OutValueT>())),
        mOp(*other.mOrigOp),
        mOrigOp(other.mOrigOp)
        {}

    ~CopyableOpTransformer()
    {
        // Delete the output tree only if it was allocated locally
        // (the top-level output tree was supplied by the caller).
        if (!mIsRoot) {
            delete mOutputTree;
            mOutputTree = NULL;
        }
    }

    void process(bool threaded = true)
    {
        if (!mInputTree || !mOutputTree) return;

        IterRange range(mInputIter);

        // Independently transform elements in the iterator range,
        // either in parallel or serially.
        if (threaded) {
            tbb::parallel_reduce(range, *this);
        } else {
            (*this)(range);
        }
    }

    /// Transform each element in the given range.
    void operator()(IterRange& range)
    {
        if (!mOutputTree) return;
        typename tree::ValueAccessor<OutTreeT> outAccessor(*mOutputTree);
        for ( ; range; ++range) {
            mOp(range.iterator(), outAccessor);
        }
    }

    void join(const CopyableOpTransformer& other)
    {
        if (mOutputTree && other.mOutputTree) {
            mOutputTree->merge(*other.mOutputTree);
        }
    }

private:
    bool mIsRoot;
    InIterT mInputIter;
    const InTreeT* mInputTree;
    OutTreeT* mOutputTree;
    OpT mOp; // copy of original functor
    OpT const * const mOrigOp; // pointer to original functor
}; // class CopyableOpTransformer

} // namespace valxform


////////////////////////////////////////


template<typename InIterT, typename OutGridT, typename XformOp>
inline void
transformValues(const InIterT& inIter, OutGridT& outGrid, XformOp& op,
    bool threaded, bool shared)
{
    typedef TreeAdapter<OutGridT> Adapter;
    typedef typename Adapter::TreeType OutTreeT;
    if (shared) {
        typedef typename valxform::SharedOpTransformer<InIterT, OutTreeT, XformOp> Processor;
        Processor proc(inIter, Adapter::tree(outGrid), op);
        proc.process(threaded);
    } else {
        typedef typename valxform::CopyableOpTransformer<InIterT, OutTreeT, XformOp> Processor;
        Processor proc(inIter, Adapter::tree(outGrid), op);
        proc.process(threaded);
    }
}

#ifndef _MSC_VER
template<typename InIterT, typename OutGridT, typename XformOp>
inline void
transformValues(const InIterT& inIter, OutGridT& outGrid, const XformOp& op,
    bool threaded, bool /*share*/)
{
    typedef TreeAdapter<OutGridT> Adapter;
    typedef typename Adapter::TreeType OutTreeT;
    // Const ops are shared across threads, not copied.
    typedef typename valxform::SharedOpTransformer<InIterT, OutTreeT, const XformOp> Processor;
    Processor proc(inIter, Adapter::tree(outGrid), op);
    proc.process(threaded);
}
#endif

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VALUETRANSFORMER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
