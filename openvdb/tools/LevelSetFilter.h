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
/// @author Ken Museth
///
/// @file LevelSetFilter.h
///
/// @brief Performs various types of level set deformations with
/// interface tracking. These unrestricted deformations include
/// surface smoothing (e.g., Laplacian flow), filtering (e.g., mean
/// value) and morphological operations (e.g., morphological opening).
/// All these operations can optionally be masked with another grid that
/// acts as an alpha-mask.

#ifndef OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED

#include <assert.h>
#include <boost/type_traits/is_floating_point.hpp>
#include "LevelSetTracker.h"
#include "Interpolation.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Filtering (e.g. diffusion) of narrow-band level sets. An
/// optional scalar field can be used to produce a (smooth) alpha mask
/// for the filtering.
///
/// @note This class performs propper interface tracking which allows
/// for unrestricted surface deformations
template<typename GridT,
         typename MaskT = typename GridT::template ValueConverter<float>::Type,
         typename InterruptT = util::NullInterrupter>
class LevelSetFilter : public LevelSetTracker<GridT, InterruptT>
{
public:
    typedef LevelSetTracker<GridT, InterruptT>              BaseType;
    typedef GridT                                           GridType;
    typedef MaskT                                           MaskType;
    typedef typename GridType::TreeType                     TreeType;
    typedef typename TreeType::ValueType                    ValueType;
    typedef typename MaskType::ValueType                    AlphaType;
    typedef typename tree::LeafManager<TreeType>::LeafRange RangeType;
    BOOST_STATIC_ASSERT(boost::is_floating_point<AlphaType>::value);

    /// @brief Main constructor from a grid
    /// @param grid The level set to be filtered.
    /// @param interrupt Optional interrupter.
    LevelSetFilter(GridType& grid, InterruptT* interrupt = NULL)
        : BaseType(grid, interrupt)
        , mTask(0)
        , mMask(NULL)
        , mMinMask(0)
        , mMaxMask(1)
        , mInvertMask(false)
    {
    }
    /// @brief Shallow copy constructor called by tbb::parallel_for()
    /// threads during filtering.
    /// @param other The other LevelSetFilter from which to copy.
    LevelSetFilter(const LevelSetFilter& other)
        : BaseType(other)
        , mTask(other.mTask)
        , mMask(other.mMask)
        , mMinMask(other.mMinMask)
        , mMaxMask(other.mMaxMask)
        , mInvertMask(other.mInvertMask)
    {
    }
    /// @brief Destructor
    virtual ~LevelSetFilter() {};

    /// @brief Used internally by tbb::parallel_for().
    /// @param range The range over which to perform multi-threading.
    /// @warning Never call this method directly!
    void operator()(const RangeType& range) const
    {
        if (mTask) mTask(const_cast<LevelSetFilter*>(this), range);
        else OPENVDB_THROW(ValueError, "task is undefined - call offset(), etc");
    }

    /// @brief Return the minimum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    AlphaType minMask() const { return mMinMask; }
    /// @brief Return the maximum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    AlphaType maxMask() const { return mMaxMask; }
    /// @brief Define the range for the (optional) scalar mask.
    /// @param min Minimum value of the range.
    /// @param max Maximum value of the range.
    /// @details Mask values outside the range maps to alpha values of
    /// respectfully zero and one, and values inside the range maps
    /// smoothly to 0->1 (unless of course the mask is inverted).
    /// @throw ValueError if @a min is not smaller then @a max.
    void setMaskRange(AlphaType min, AlphaType max)
    {
        if (!(min < max)) OPENVDB_THROW(ValueError, "Invalid mask range (expects min < max)");
        mMinMask = min;
        mMaxMask = max;
    }

    /// @brief Return true if the mask is inverted, i.e. min->max in the
    /// original mask maps to 1->0 in the inverted alpha mask.
    bool isMaskInverted() const { return mInvertMask; }
    /// @brief Invert the optional mask, i.e. min->max in the original
    /// mask maps to 1->0 in the inverted alpha mask.
    void invertMask(bool invert=true) { mInvertMask = invert; }

    /// @brief One iteration of mean-curvature flow of the level set.
    /// @param mask Optional alpha mask.
    void meanCurvature(const MaskType* mask = NULL);

    /// @brief One iteration of laplacian flow of the level set.
    /// @param mask Optional alpha mask.
    void laplacian(const MaskType* mask = NULL);

    /// @brief One iteration of a fast separable gaussian filter.
    /// @param width Width of the gaussian kernel in voxel units.
    /// @param mask Optional alpha mask.
    ///
    /// @note This is approximated as 4 iterations of a separable mean filter
    /// which typically leads an approximation that's better than 95%!
    void gaussian(int width = 1, const MaskType* mask = NULL);

    /// @brief Offset the level set by the specified (world) distance.
    /// @param offset Value of the offset.
    /// @param mask Optional alpha mask.
    void offset(ValueType offset, const MaskType* mask = NULL);

    /// @brief One iteration of median-value flow of the level set.
    /// @param width Width of the median-value kernel in voxel units.
    /// @param mask Optional alpha mask.
    ///
    /// @warning This filter is not separable and is hence relatively
    /// slow!
    void median(int width = 1, const MaskType* mask = NULL);

    /// @brief One iteration of mean-value flow of the level set.
    /// @param width Width of the mean-value kernel in voxel units.
    /// @param mask Optional alpha mask.
    ///
    /// @note This filter is separable so it's fast!
    void mean(int width = 1, const MaskType* mask = NULL);

private:
    typedef typename TreeType::LeafNodeType                  LeafT;
    typedef typename LeafT::ValueOnIter                      VoxelIterT;
    typedef typename LeafT::ValueOnCIter                     VoxelCIterT;
    typedef typename tree::LeafManager<TreeType>::BufferType BufferT;
    typedef typename RangeType::Iterator                     LeafIterT;
    typedef tools::AlphaMask<GridT, MaskT>                   AlphaMaskT;

    // Only two private member data
    typename boost::function<void (LevelSetFilter*, const RangeType&)> mTask;
    const MaskType* mMask;
    AlphaType       mMinMask, mMaxMask;
    bool            mInvertMask;

    // Private cook method calling tbb::parallel_for
    void cook(bool swap)
    {
        const int n = BaseType::getGrainSize();
        if (n>0) {
            tbb::parallel_for(BaseType::leafs().leafRange(n), *this);
        } else {
            (*this)(BaseType::leafs().leafRange());
        }
        if (swap) BaseType::leafs().swapLeafBuffer(1, n==0);
    }

    // Private driver method for mean and gaussian filtering
    void box(int width);

    template <size_t Axis>
    struct Avg {
        Avg(const GridT& grid, Int32 w) :
            acc(grid.tree()), width(w), frac(1/ValueType(2*w+1))
        {
        }
        ValueType operator()(Coord xyz)
        {
            ValueType sum = zeroVal<ValueType>();
            Int32& i = xyz[Axis], j = i + width;
            for (i -= width; i <= j; ++i) sum += acc.getValue(xyz);
            return sum*frac;
        }
        typename GridT::ConstAccessor acc;
        const Int32 width;
        const ValueType frac;
    };

    // Private methods called by tbb::parallel_for threads
    template <typename AvgT>
    void doBox( const RangeType& r, Int32 w);
    void doBoxX(const RangeType& r, Int32 w) { this->doBox<Avg<0> >(r,w); }
    void doBoxZ(const RangeType& r, Int32 w) { this->doBox<Avg<1> >(r,w); }
    void doBoxY(const RangeType& r, Int32 w) { this->doBox<Avg<2> >(r,w); }
    void doMedian(const RangeType&, int);
    void doMeanCurvature(const RangeType&);
    void doLaplacian(const RangeType&);
    void doOffset(const RangeType&, ValueType);

}; // end of LevelSetFilter class


////////////////////////////////////////

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::median(int width, const MaskType* mask)
{
    mMask = mask;

    BaseType::startInterrupter("Median-value flow of level set");

    BaseType::leafs().rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    mTask = boost::bind(&LevelSetFilter::doMedian, _1, _2, std::max(1, width));
    this->cook(true);

    BaseType::track();

    BaseType::endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::mean(int width, const MaskType* mask)
{
    mMask = mask;

    BaseType::startInterrupter("Mean-value flow of level set");

    this->box(width);

    BaseType::endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::gaussian(int width, const MaskType* mask)
{
    mMask = mask;

    BaseType::startInterrupter("Gaussian flow of level set");

    for (int n=0; n<4; ++n) this->box(width);

    BaseType::endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::box(int width)
{
    BaseType::leafs().rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    width = std::max(1, width);

    mTask = boost::bind(&LevelSetFilter::doBoxX, _1, _2, width);
    this->cook(true);

    mTask = boost::bind(&LevelSetFilter::doBoxY, _1, _2, width);
    this->cook(true);

    mTask = boost::bind(&LevelSetFilter::doBoxZ, _1, _2, width);
    this->cook(true);

    BaseType::track();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::meanCurvature(const MaskType* mask)
{
    mMask = mask;

    BaseType::startInterrupter("Mean-curvature flow of level set");

    BaseType::leafs().rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    mTask = boost::bind(&LevelSetFilter::doMeanCurvature, _1, _2);
    this->cook(true);

    BaseType::track();

    BaseType::endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::laplacian(const MaskType* mask)
{
    mMask = mask;

    BaseType::startInterrupter("Laplacian flow of level set");

    BaseType::leafs().rebuildAuxBuffers(1, BaseType::getGrainSize()==0);

    mTask = boost::bind(&LevelSetFilter::doLaplacian, _1, _2);
    this->cook(true);

    BaseType::track();

    BaseType::endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::offset(ValueType value, const MaskType* mask)
{
    mMask = mask;

    BaseType::startInterrupter("Offsetting level set");

    BaseType::leafs().removeAuxBuffers();// no auxiliary buffers required

    const ValueType CFL = ValueType(0.5) * BaseType::voxelSize(), offset = openvdb::math::Abs(value);
    ValueType dist = 0.0;
    while (offset-dist > ValueType(0.001)*CFL && BaseType::checkInterrupter()) {
        const ValueType delta = openvdb::math::Min(offset-dist, CFL);
        dist += delta;

        mTask = boost::bind(&LevelSetFilter::doOffset, _1, _2, copysign(delta, value));
        this->cook(false);

        BaseType::track();
    }

    BaseType::endInterrupter();
}


///////////////////////// PRIVATE METHODS //////////////////////

/// Performs parabolic mean-curvature diffusion
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::doMeanCurvature(const RangeType& range)
{
    BaseType::checkInterrupter();
    //const float CFL = 0.9f, dt = CFL * mDx * mDx / 6.0f;
    const ValueType dx = BaseType::voxelSize(), dt = math::Pow2(dx) / ValueType(3.0);
    math::CurvatureStencil<GridType> stencil(BaseType::grid(), dx);
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(BaseType::grid(), *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    stencil.moveTo(iter);
                    const ValueType phi0 = *iter, phi1 = phi0 + dt*stencil.meanCurvatureNormGrad();
                    buffer.setValue(iter.pos(), b*phi0 + a*phi1);
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                buffer.setValue(iter.pos(), *iter + dt*stencil.meanCurvatureNormGrad());
            }
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
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::doLaplacian(const RangeType& range)
{
    BaseType::checkInterrupter();
    //const float CFL = 0.9f, half_dt = CFL * mDx * mDx / 12.0f;
    const ValueType dx = BaseType::voxelSize(), dt = math::Pow2(dx) / ValueType(6.0);
    math::GradStencil<GridType> stencil(BaseType::grid(), dx);
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(BaseType::grid(), *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    stencil.moveTo(iter);
                    const ValueType phi0 = *iter, phi1 = phi0 + dt*stencil.laplacian();
                    buffer.setValue(iter.pos(), b*phi0 + a*phi1);
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                buffer.setValue(iter.pos(), *iter + dt*stencil.laplacian());
            }
        }
    }
}

/// Offsets the values by a constant
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::doOffset(const RangeType& range, ValueType offset)
{
    BaseType::checkInterrupter();
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(BaseType::grid(), *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            for (VoxelIterT iter = leafIter->beginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) iter.setValue(*iter + a*offset);
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            for (VoxelIterT iter = leafIter->beginValueOn(); iter; ++iter) {
                iter.setValue(*iter + offset);
            }
        }
    }
}

/// Performs simple but slow median-value diffusion
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::doMedian(const RangeType& range, int width)
{
    BaseType::checkInterrupter();
    typename math::DenseStencil<GridType> stencil(BaseType::grid(), width);//creates local cache!
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(BaseType::grid(), *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    stencil.moveTo(iter);
                    buffer.setValue(iter.pos(), b*(*iter) + a*stencil.median());
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                buffer.setValue(iter.pos(), stencil.median());
            }
        }
    }
}

/// One dimensional convolution of a separable box filter
template<typename GridT, typename MaskT, typename InterruptT>
template <typename AvgT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::doBox(const RangeType& range, Int32 w)
{
    BaseType::checkInterrupter();
    AvgT avg(BaseType::grid(), w);
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(BaseType::grid(), *mMask, mMinMask, mMaxMask, mInvertMask);
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                const Coord xyz = iter.getCoord();
                if (alpha(xyz, a, b)) buffer.setValue(iter.pos(), b*(*iter)+ a*avg(xyz));
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            BufferT& buffer = leafIter.buffer(1);
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                buffer.setValue(iter.pos(), avg(iter.getCoord()));
            }
        }
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
