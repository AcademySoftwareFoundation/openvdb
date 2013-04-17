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
/// @author Ken Museth
///
/// @file Filter.h

#ifndef OPENVDB_TOOLS_FILTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_FILTER_HAS_BEEN_INCLUDED

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/Grid.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Filtering of VDB volumes
template<typename GridT, typename InterruptT = util::NullInterrupter>
class Filter
{
public:
    typedef GridT                                  GridType;
    typedef typename GridType::TreeType            TreeType;
    typedef typename TreeType::LeafNodeType        LeafType;
    typedef typename LeafType::ValueType           ValueType;
    typedef typename tree::LeafManager<TreeType>   LeafManagerType;
    typedef typename LeafManagerType::LeafRange    RangeType;
    typedef typename LeafManagerType::BufferType   BufferType;

    /// Constructor
    Filter(GridT& grid, InterruptT* interrupt = NULL) :
        mGrid(grid), mTask(0), mInterrupter(interrupt)
    {
    }

    /// @brief One iteration of a fast separable mean-value (i.e. box) filter.
    void mean(int width = 1, int iterations = 1, bool serial = false);

    /// @brief One iteration of a fast separable gaussian filter.
    ///
    /// @note This is approximated as 4 iterations of a separable mean filter
    /// which typically leads an approximation that's better than 95%!
    void gaussian(int width = 1, int iterations = 1, bool serial = false);

    /// @brief One iteration of a median-value filter
    ///
    /// @note This filter is not separable and is hence relatively slow!
    void median(int width = 1, int iterations = 1, bool serial = false);

    /// Offsets (i.e. adds) a constant value to all active voxels.
    void offset(float offset, bool serial = false);

    /// @brief Used internally by tbb::parallel_for()
    ///
    /// @note Never call this method directly!
    void operator()(const RangeType& r) const
    {
        if (mTask) mTask(const_cast<Filter*>(this), r);
        else OPENVDB_THROW(ValueError, "task is undefined - call median(), mean(), etc.");
    }

private:
    typedef typename boost::function<void (Filter*, const RangeType&)> FuncType;

    void cook(bool serial, LeafManagerType& leafs);

    // Private filter methods called by tbb::parallel_for threads
    void doBoxX(const RangeType&, Int32);
    void doBoxY(const RangeType&, Int32);
    void doBoxZ(const RangeType&, Int32);
    void doMedian(const RangeType&, int);
    void doOffset(const RangeType&, float);
    /// @return true if the process was interrupted
    bool wasInterrupted();

    GridType&         mGrid;
    FuncType          mTask;
    InterruptT*       mInterrupter;
}; // end of Filter class

////////////////////////////////////////

template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::mean(int width, int iterations, bool serial)
{
    if (mInterrupter) mInterrupter->start("Applying mean filter");

    const int w = std::max(1, width);

    LeafManagerType leafs(mGrid.tree(), 1, serial);

    for (int i=0; i<iterations && !this->wasInterrupted(); ++i) {
        mTask = boost::bind(&Filter::doBoxX, _1, _2, w);
        this->cook(serial, leafs);

        mTask = boost::bind(&Filter::doBoxY, _1, _2, w);
        this->cook(serial, leafs);

        mTask = boost::bind(&Filter::doBoxZ, _1, _2, w);
        this->cook(serial, leafs);
    }

    if (mInterrupter) mInterrupter->end();
}

template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::gaussian(int width, int iterations, bool serial)
{
    if (mInterrupter) mInterrupter->start("Applying gaussian filter");

    const int w = std::max(1, width);

    LeafManagerType leafs(mGrid.tree(), 1, serial);

    for (int i=0; i<iterations; ++i) {
        for (int n=0; n<4 && !this->wasInterrupted(); ++n) {
            mTask = boost::bind(&Filter::doBoxX, _1, _2, w);
            this->cook(serial, leafs);

            mTask = boost::bind(&Filter::doBoxY, _1, _2, w);
            this->cook(serial, leafs);

            mTask = boost::bind(&Filter::doBoxZ, _1, _2, w);
            this->cook(serial, leafs);
        }
    }

    if (mInterrupter) mInterrupter->end();
}


template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::median(int width, int iterations, bool serial)
{
    if (mInterrupter) mInterrupter->start("Applying median filter");

    LeafManagerType leafs(mGrid.tree(), 1, serial);

    mTask = boost::bind(&Filter::doMedian, _1, _2, std::max(1, width));
    for (int i=0; i<iterations && !this->wasInterrupted(); ++i) this->cook(serial, leafs);

    if (mInterrupter) mInterrupter->end();
}

template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::offset(float value, bool serial)
{
    if (mInterrupter) mInterrupter->start("Applying offset");

    LeafManagerType leafs(mGrid.tree(), 0, serial);

    mTask = boost::bind(&Filter::doOffset, _1, _2, value);
    this->cook(serial, leafs);

    if (mInterrupter) mInterrupter->end();
}

////////////////////////////////////////


/// Private method to perform the task (serial or threaded) and
/// subsequently swap the leaf buffers.
template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::cook(bool serial, LeafManagerType& leafs)
{
    if (serial) {
        (*this)(leafs.leafRange());
    } else {
        tbb::parallel_for(leafs.leafRange(), *this);
    }
    leafs.swapLeafBuffer(1, serial);
}

/// X convolution of a separable box filter
template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::doBoxX(const RangeType& range, Int32 w)
{
    this->wasInterrupted();
    const ValueType frac = ValueType(1)/ValueType(2*w+1);
    typename GridT::ConstAccessor acc = mGrid.getConstAccessor();
    for (typename RangeType::Iterator lIter=range.begin(); lIter; ++lIter) {
        BufferType& buffer = lIter.buffer(1);
        for (typename LeafType::ValueOnCIter vIter = lIter->cbeginValueOn(); vIter; ++vIter) {
            ValueType sum = zeroVal<ValueType>();
            math::Coord xyz = vIter.getCoord();
            for (Int32 x = xyz.x()-w, xLast = xyz.x()+w; x <= xLast; ++x) {
                sum += acc.getValue(xyz.setX(x));
            }
            buffer.setValue(vIter.pos(), sum*frac);
        }
    }
}

/// Y convolution of a separable box filter
template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::doBoxY(const RangeType& range, Int32 w)
{
    this->wasInterrupted();
    const ValueType frac = ValueType(1)/ValueType(2*w+1);
    typename GridT::ConstAccessor acc = mGrid.getConstAccessor();
    for (typename RangeType::Iterator lIter=range.begin(); lIter; ++lIter) {
        BufferType& buffer = lIter.buffer(1);
        for (typename LeafType::ValueOnCIter vIter = lIter->cbeginValueOn(); vIter; ++vIter) {
            ValueType sum = zeroVal<ValueType>();
            math::Coord xyz = vIter.getCoord();
            for (Int32 y = xyz.y()-w, yLast = xyz.y()+w; y <= yLast; ++y) {
                sum += acc.getValue(xyz.setY(y));
            }
            buffer.setValue(vIter.pos(), sum*frac);
        }
    }
}

/// Z convolution of a separable box filter
template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::doBoxZ(const RangeType& range, Int32 w)
{
    this->wasInterrupted();
    const ValueType frac = ValueType(1)/ValueType(2*w+1);
    typename GridT::ConstAccessor acc = mGrid.getConstAccessor();
    for (typename RangeType::Iterator lIter=range.begin(); lIter; ++lIter) {
        BufferType& buffer = lIter.buffer(1);
        for (typename LeafType::ValueOnCIter vIter = lIter->cbeginValueOn(); vIter; ++vIter) {
            ValueType sum = zeroVal<ValueType>();
            math::Coord xyz = vIter.getCoord();
            for (Int32 z = xyz.z()-w, zLast = xyz.z()+w; z <= zLast; ++z) {
                sum += acc.getValue(xyz.setZ(z));
            }
            buffer.setValue(vIter.pos(), sum*frac);
        }
    }
}

/// Performs simple but slow median-value diffusion
template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::doMedian(const RangeType& range, int width)
{
    this->wasInterrupted();
    typename math::DenseStencil<GridType> stencil(mGrid, width);//creates local cache!
    for (typename RangeType::Iterator lIter=range.begin(); lIter; ++lIter) {
        BufferType& buffer = lIter.buffer(1);
        for (typename LeafType::ValueOnCIter vIter = lIter->cbeginValueOn(); vIter; ++vIter) {
            stencil.moveTo(vIter);
            buffer.setValue(vIter.pos(), stencil.median());
        }
    }
}

/// Offsets the values by a constant
template<typename GridT, typename InterruptT>
inline void
Filter<GridT, InterruptT>::doOffset(const RangeType& range, float floatVal)
{
    const ValueType value = static_cast<ValueType>(floatVal);
    for (typename RangeType::Iterator lIter=range.begin(); lIter; ++lIter) lIter->addValue(value);
}

template<typename GridT, typename InterruptT>
inline bool
Filter<GridT, InterruptT>::wasInterrupted()
{
    if (util::wasInterrupted(mInterrupter)) {
        tbb::task::self().cancel_group_execution();
        return true;
    }
    return false;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_FILTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
