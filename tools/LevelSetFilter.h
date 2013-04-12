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
/// @file LevelSetFilter.h
///
/// @brief Performs various types of level set deformations with
/// interface tracking. These unrestricted deformations include
/// surface smoothing (e.g., Laplacian flow), filtering (e.g., mean
/// value) and morphological operations (e.g., morphological opening).

#ifndef OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED

#include "LevelSetTracker.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Filtering (i.e. diffusion) of narrow-band level sets
///
/// @note This class performs propper interface tracking which allows
/// for unrestricted surface deformations
template<typename GridT,
         typename InterruptT = util::NullInterrupter>
class LevelSetFilter : public LevelSetTracker<GridT, InterruptT>
{
public:
    typedef boost::shared_ptr<LevelSetFilter>     Ptr;
    typedef LevelSetTracker<GridT, InterruptT>    BaseType;
    typedef GridT                                 GridType;
    typedef typename GridType::TreeType           TreeType;
    typedef typename TreeType::LeafNodeType       LeafType;
    typedef typename LeafType::ValueType          ValueType;
    typedef typename tree::LeafManager<TreeType>  LeafManagerType;
    typedef typename LeafManagerType::RangeType   RangeType;
    typedef typename LeafManagerType::BufferType  BufferType;

    /// Main constructor
    LevelSetFilter(GridType& grid, InterruptT* interrupt = NULL)
        : BaseType(grid, interrupt), mTask(0)
    {
    }
    /// Shallow copy constructor called by tbb::parallel_for() threads during filtering
    LevelSetFilter(const LevelSetFilter& other)
        : BaseType(other), mTask(other.mTask)
    {
    }
    virtual ~LevelSetFilter() {};

    /// Used internally by tbb::parallel_for()
    void operator()(const RangeType& r) const
    {
        if (mTask) mTask(const_cast<LevelSetFilter*>(this), r);
        else OPENVDB_THROW(ValueError, "task is undefined - call offset(), etc");
    }

    /// @brief One iteration of mean-curvature flow of the level set
    void meanCurvature();

    /// @brief One iteration of laplacian flow of the level set
    void laplacian();

    /// @brief One iteration of a fast separable gaussian filter.
    ///
    /// @note This is approximated as 4 iterations of a separable mean filter
    /// which typically leads an approximation that's better than 95%!
    void gaussian(int width = 1);

    /// @brief Offset the level set by the specified (world) distance
    void offset(ValueType offset);

    /// @brief One iteration of median-value flow of the level set
    ///
    /// @note This filter is not separable and is hence relatively slow!
    void median(int width = 1);

    /// @brief One iteration of mean-value flow of the level set
    ///
    /// @note This filter is separable so it's fast!
    void mean(int width = 1);

private:
    typedef typename boost::function<void (LevelSetFilter*, const RangeType&)> FuncType;

    FuncType mTask;

    // Private cook method calling tbb::parallel_for
    void cook(bool swap)
    {
        const int n = BaseType::getGrainSize();
        if (n>0) {
            tbb::parallel_for(BaseType::mLeafs->getRange(n), *this);
        } else {
            (*this)(BaseType::mLeafs->getRange());
        }
        if (swap) BaseType::mLeafs->swapLeafBuffer(1, n==0);
    }

    // Private methods called by tbb::parallel_for threads
    void doBoxX(const RangeType&, Int32);
    void doBoxY(const RangeType&, Int32);
    void doBoxZ(const RangeType&, Int32);
    void doMedian(const RangeType&, int);
    void doMeanCurvature(const RangeType&);
    void doLaplacian(const RangeType&);
    void doOffset(const RangeType&, ValueType);

}; // end of LevelSetFilter class


////////////////////////////////////////

template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::median(int width)
{
    BaseType::startInterrupter("Median-value flow of level set");

    BaseType::mLeafs->rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    mTask = boost::bind(&LevelSetFilter::doMedian, _1, _2, std::max(1, width));
    this->cook(true);

    BaseType::track();

    BaseType::endInterrupter();
}


template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::mean(int width)
{
    BaseType::startInterrupter("Mean-value flow of level set");

    width = std::max(1, width);

    BaseType::mLeafs->rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    mTask = boost::bind(&LevelSetFilter::doBoxX, _1, _2, width);
    this->cook(true);

    mTask = boost::bind(&LevelSetFilter::doBoxY, _1, _2, width);
    this->cook(true);

    mTask = boost::bind(&LevelSetFilter::doBoxZ, _1, _2, width);
    this->cook(true);

    BaseType::track();

    BaseType::endInterrupter();
}

template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::gaussian(int width)
{
    BaseType::startInterrupter("Gaussian flow of level set");

    width = std::max(1, width);

    BaseType::mLeafs->rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    for (int n=0; n<4; ++n) {

        mTask = boost::bind(&LevelSetFilter::doBoxX, _1, _2, width);
        this->cook(true);

        mTask = boost::bind(&LevelSetFilter::doBoxY, _1, _2, width);
        this->cook(true);

        mTask = boost::bind(&LevelSetFilter::doBoxZ, _1, _2, width);
        this->cook(true);
    }

    BaseType::track();

    BaseType::endInterrupter();
}



template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::meanCurvature()
{
    BaseType::startInterrupter("Mean-curvature flow of level set");

    BaseType::mLeafs->rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    mTask = boost::bind(&LevelSetFilter::doMeanCurvature, _1, _2);
    this->cook(true);

    BaseType::track();

    BaseType::endInterrupter();
}

template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::laplacian()
{
    BaseType::startInterrupter("Laplacian flow of level set");

    BaseType::mLeafs->rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    mTask = boost::bind(&LevelSetFilter::doLaplacian, _1, _2);
    this->cook(true);

    BaseType::track();

    BaseType::endInterrupter();
}


template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::offset(ValueType value)
{
    BaseType::startInterrupter("Offsetting level set");

    BaseType::mLeafs->removeAuxBuffers();// no auxiliary buffers required

    const ValueType CFL = ValueType(0.5) * BaseType::voxelSize(), offset = openvdb::math::Abs(value);
    ValueType dist = 0.0;
    while (offset-dist > ValueType(0.001)*CFL && BaseType::checkInterrupter()) {
        const ValueType delta = openvdb::math::Min(offset-dist, CFL);
        dist += delta;

        mTask = boost::bind(&LevelSetFilter::doOffset, _1, _2, copysign(delta,value));
        this->cook(false);

        BaseType::track();
    }

    BaseType::endInterrupter();
}


///////////////////////// PRIVATE METHODS //////////////////////

/// Performs parabolic mean-curvature diffusion
template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::doMeanCurvature(const RangeType& range)
{
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    BaseType::checkInterrupter();
    //const float CFL = 0.9f, dt = CFL * mDx * mDx / 6.0f;
    const ValueType dx = BaseType::voxelSize(),dt = math::Pow2(dx) / ValueType(3.0);
    math::CurvatureStencil<GridType> stencil(BaseType::mGrid, dx);
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        BufferType& buffer = BaseType::mLeafs->getBuffer(n,1);
        for (VoxelIterT iter = BaseType::mLeafs->leaf(n).cbeginValueOn(); iter; ++iter) {
            stencil.moveTo(iter);
            buffer.setValue(iter.pos(), stencil.getValue() + dt * stencil.meanCurvatureNormGrad());
        }
    }
}

/// Performs laplacian diffusion. Note if the grids contains a true
/// signed distance field (e.g. a solution to the Eikonal equation)
/// Laplacian diffusions (e.g. geometric heat equation) is actually
/// identical to mean curvature diffusion, yet less computationally
/// expensive! In other words if you're performing renormalization
/// anyway (e.g. rebuilding the narrow-band) you should consider
/// performing laplacian diffusion over mean curvature flow!
template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::doLaplacian(const RangeType& range)
{
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    BaseType::checkInterrupter();
    //const float CFL = 0.9f, half_dt = CFL * mDx * mDx / 12.0f;
    const ValueType dx = BaseType::voxelSize(), dt = math::Pow2(dx) / ValueType(6.0);
    math::GradStencil<GridType> stencil(BaseType::mGrid, dx);
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        BufferType& buffer = BaseType::mLeafs->getBuffer(n,1);
        for (VoxelIterT iter = BaseType::mLeafs->leaf(n).cbeginValueOn(); iter; ++iter) {
            stencil.moveTo(iter);
            buffer.setValue(iter.pos(), stencil.getValue() + dt * stencil.laplacian());
        }
    }
}

/// Offsets the values by a constant
template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::doOffset(const RangeType& range, ValueType value)
{
    BaseType::checkInterrupter();
    for (size_t n=range.begin(), e=range.end(); n != e; ++n)
        BaseType::mLeafs->leaf(n).addValue(value);
}

/// Performs simple but slow median-value diffusion
template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::doMedian(const RangeType& range, int width)
{
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    BaseType::checkInterrupter();
    typename math::DenseStencil<GridType> stencil(BaseType::mGrid, width);//creates local cache!
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        BufferType& buffer = BaseType::mLeafs->getBuffer(n,1);
        for (VoxelIterT iter=BaseType::mLeafs->leaf(n).cbeginValueOn(); iter; ++iter) {
            stencil.moveTo(iter);
            buffer.setValue(iter.pos(), stencil.median());
        }
    }
}

/// X convolution of a separable box filter
template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::doBoxX(const RangeType& range, Int32 w)
{
    this->checkInterrupter();
    const ValueType frac = ValueType(1)/ValueType(2*w+1);
    typename GridT::ConstAccessor acc = BaseType::mGrid.getConstAccessor();
    for (size_t n=range.begin(), nLast = range.end(); n != nLast; ++n) {
        const LeafType& leaf = BaseType::mLeafs->leaf(n);
        BufferType& buffer   = BaseType::mLeafs->getBuffer(n, 1);
        for (typename LeafType::ValueOnCIter iter = leaf.cbeginValueOn(); iter; ++iter) {
            ValueType sum = zeroVal<ValueType>();
            math::Coord xyz = iter.getCoord();
            for (Int32 x = xyz.x()-w, xLast = xyz.x()+w; x <= xLast; ++x) {
                sum += acc.getValue(xyz.setX(x));
            }
            buffer.setValue(iter.pos(), sum*frac);
        }
    }
}

/// Y convolution of a separable box filter
template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::doBoxY(const RangeType& range, Int32 w)
{
    this->checkInterrupter();
    const ValueType frac = ValueType(1)/ValueType(2*w+1);
    typename GridT::ConstAccessor acc = BaseType::mGrid.getConstAccessor();
    for (size_t n=range.begin(), nLast = range.end(); n != nLast; ++n) {
        const LeafType& leaf = BaseType::mLeafs->leaf(n);
        BufferType& buffer   = BaseType::mLeafs->getBuffer(n, 1);
        for (typename LeafType::ValueOnCIter iter = leaf.cbeginValueOn(); iter; ++iter) {
            ValueType sum = zeroVal<ValueType>();
            math::Coord xyz = iter.getCoord();
            for (Int32 y = xyz.y()-w, yLast = xyz.y()+w; y <= yLast; ++y) {
                sum += acc.getValue(xyz.setY(y));
            }
            buffer.setValue(iter.pos(), sum*frac);
        }
    }
}

/// Z convolution of a separable box filter
template<typename GridT, typename InterruptT>
inline void
LevelSetFilter<GridT, InterruptT>::doBoxZ(const RangeType& range, Int32 w)
{
    this->checkInterrupter();
    const ValueType frac = ValueType(1)/ValueType(2*w+1);
    typename GridT::ConstAccessor acc = BaseType::mGrid.getConstAccessor();
    for (size_t n=range.begin(), nLast = range.end(); n != nLast; ++n) {
        const LeafType& leaf = BaseType::mLeafs->leaf(n);
        BufferType& buffer   = BaseType::mLeafs->getBuffer(n, 1);
        for (typename LeafType::ValueOnCIter iter = leaf.cbeginValueOn(); iter; ++iter) {
            ValueType sum = zeroVal<ValueType>();
            math::Coord xyz = iter.getCoord();
            for (Int32 z = xyz.z()-w, zLast = xyz.z()+w; z <= zLast; ++z) {
                sum += acc.getValue(xyz.setZ(z));
            }
            buffer.setValue(iter.pos(), sum*frac);
        }
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
