// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_IMAGE_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_IMAGE_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <openvdb/tools/GridTransformer.h>
#include <openvdb/math/Transform.h>

#include <vector>
#include <math.h>


/* openvdbmma::image members

 struct GridImage3D

 struct DepthMap

 struct GridSliceImage

 All are templated and intended for use with DynamicNodeManager.

*/


namespace openvdbmma {
namespace image {

//////////// utilities

namespace internal {

template<typename T>
inline unsigned char
to_uchar(const T& f)
{
    return static_cast<unsigned char>(255*f);
}

} // namespace internal

template<typename T>
inline mma::ImageRef<T> makeEmptyImage(const mint& a, const mint& b)
{
    mma::ImageRef<T> im = mma::makeImage<T>(a, b);
    T* im_data = im.data();

    mma::check_abort();

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, a*b),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i)
                im_data[i] = 0;
        }
    );

    return im;
}

template<typename T>
inline mma::Image3DRef<T> makeEmptyImage3D(const mint& a, const mint& b, const mint& c)
{
    mma::Image3DRef<T> im = mma::makeImage3D<T>(a, b, c);
    T* im_data = im.data();

    mma::check_abort();

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, a*b*c),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i)
                im_data[i] = 0;
        }
    );

    return im;
}


template<typename GridT>
struct pixelExtrema
{
    using ValueT = typename GridT::ValueType;

    pixelExtrema(typename GridT::Ptr grid)
    : mGrid(grid)
    {
        findExtrema();
    }

    ValueT min, max;

private:

    template<typename V = ValueT>
    inline typename std::enable_if_t<scalar_type<V>::value, void>
    findExtrema()
    {
        switch (mGrid->getGridClass()) {
            case GRID_FOG_VOLUME: {
                min = (ValueT)0;
                max = (ValueT)1;
                break;
            }
            case GRID_LEVEL_SET: {
                max = mGrid->background();
                min = -max;
                break;
            }
            default: {
                const openvdb::math::MinMax<ValueT> extrema = openvdb::tools::minMax(mGrid->tree());
                min = extrema.min();
                max = extrema.max();
                break;
            }
        }
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<bool_type<V>::value, void>
    findExtrema()
    {
        min = false;
        max = true;
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<!scalar_type<V>::value && !bool_type<V>::value, void>
    findExtrema()
    {
        min = (ValueT)0;
        max = (ValueT)1;
    }

    typename GridT::Ptr mGrid;
};


//////////// voxel value functors

template<typename TreeType>
struct GridImage3D
{
    using ValueT = typename TreeType::ValueType;
    using RootT  = typename TreeType::RootNodeType;
    using LeafT  = typename TreeType::LeafNodeType;

    using PixelT = typename std::conditional_t<
            mask_type<ValueT>::value || bool_type<ValueT>::value,
            mma::im_bit_t,
            mma::im_byte_t
        >;

    using mmaImageT = typename mma::Image3DRef<PixelT>;

    explicit GridImage3D(const CoordBBox& bbox, const ValueT& vmn, const ValueT& vmx)
    : im(openvdbmma::image::makeEmptyImage3D<PixelT>(bbox.dim().z(), bbox.dim().x(), bbox.dim().y()))
    , mBBox(bbox), mVmin(vmn), mVmax(vmx), mFac(rangeNormalization(vmn, vmx))
    , mXlen(bbox.dim().x() - 1), mOx(bbox.min().x())
    , mYlen(bbox.dim().y() - 1), mOy(bbox.max().y())
    , mZlen(bbox.dim().z() - 1), mOz(bbox.max().z())
    {
        openvdbmma::types::pixel_type_assert<ValueT>();
    }

    GridImage3D(const GridImage3D& other, tbb::split)
    : im(other.im), mBBox(other.mBBox), mOx(other.mOx), mOy(other.mOy), mOz(other.mOz)
    , mVmin(other.mVmin), mVmax(other.mVmax), mFac(other.mFac)
    , mXlen(other.mXlen), mYlen(other.mYlen), mZlen(other.mZlen)
    {
    }

    void operator()(const RootT& node) {}

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if (!mBBox.hasOverlap(node.getNodeBoundingBox()))
            return false;

        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            const CoordBBox bbox(
                CoordBBox::createCube(iter.getCoord(), NodeT::ChildNodeType::DIM));

            if (bbox.hasOverlap(mBBox)) {
                const PixelT ival = nodeValue(*iter);

                const int xstart = xLeft(bbox), xend = xRight(bbox);
                const int ystart = yLeft(bbox), yend = yRight(bbox);
                const int zstart = zLeft(bbox), zend = zRight(bbox);

                // Cache friendly way to iterate since im(k, j, i) is image_data[k*x*y + j*x + i]
                for(int k = zstart; k <= zend; k++)
                    for(int j = ystart; j <= yend; j++)
                        for(int i = xstart; i <= xend; i++)
                            im(k, j, i) = ival;
            }
        }

        return true;
    }

    inline bool operator()(const LeafT& leaf, size_t)
    {
        if (!mBBox.hasOverlap(leaf.getNodeBoundingBox()))
            return false;

        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            const Coord p = iter.getCoord();
            if (mBBox.isInside(p))
                im(zPos(p), yPos(p), xPos(p)) = leafValue(*iter);
        }

        return false;
    }

    void join(const GridImage3D& other) {}

    mmaImageT im;

private:

    template<typename V = ValueT>
    inline typename std::enable_if_t<scalar_type<V>::value, PixelT>
    nodeValue(const ValueT& x)
    {
        return internal::to_uchar<ValueT>(mFac * (x - mVmin));
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<!scalar_type<V>::value, PixelT>
    nodeValue(const ValueT& x)
    {
        return x;
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<scalar_type<V>::value, PixelT>
    leafValue(const ValueT& x)
    {
        return internal::to_uchar<ValueT>(mVmin < 0.0 && math::Abs(x) == mVmax ? (ValueT)0 : mFac * (x - mVmin));
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<!scalar_type<V>::value, PixelT>
    leafValue(const ValueT& x)
    {
        return x;
    }

    inline int xLeft (const CoordBBox& bbox) const { return math::Max(0, bbox.min().x() - mOx); }
    inline int xRight(const CoordBBox& bbox) const { return math::Min(mXlen, bbox.max().x() - mOx); }

    inline int yLeft (const CoordBBox& bbox) const { return math::Max(0, mOy - bbox.max().y()); }
    inline int yRight(const CoordBBox& bbox) const { return math::Min(mYlen, mOy - bbox.min().y()); }

    inline int zLeft (const CoordBBox& bbox) const { return math::Max(0, mOz - bbox.max().z()); }
    inline int zRight(const CoordBBox& bbox) const { return math::Min(mZlen, mOz - bbox.min().z()); }

    inline int xPos(const Coord& p) const { return p.x() - mOx; }
    inline int yPos(const Coord& p) const { return mOy - p.y(); }
    inline int zPos(const Coord& p) const { return mOz - p.z(); }

    template<typename V = ValueT>
    inline typename std::enable_if<scalar_type<V>::value, ValueT>::type
    rangeNormalization(const ValueT& vmn, const ValueT& vmx)
    {
        return vmn != vmx ? (ValueT)1/(vmx-vmn) : (ValueT)1;
    }

    template<typename V = ValueT>
    inline typename std::enable_if<!scalar_type<V>::value, ValueT>::type
    rangeNormalization(const ValueT&, const ValueT&)
    {
        return (ValueT)1;
    }

    //////////// private members

    const CoordBBox mBBox;
    const int mOx, mOy, mOz;
    const int mXlen, mYlen, mZlen;
    const ValueT mVmin, mVmax, mFac;
};


template<typename TreeType>
struct DepthMap
{
    using ValueT = typename TreeType::ValueType;
    using RootT  = typename TreeType::RootNodeType;
    using LeafT  = typename TreeType::LeafNodeType;

    explicit DepthMap(const CoordBBox& bbox, std::vector<float> ints, const bool& mv)
    : im(openvdbmma::image::makeEmptyImage<mma::im_real32_t>(bbox.dim().x(), bbox.dim().y()))
    , mBBox(bbox), mIntensities(ints), mMultiply(mv), mOz(bbox.min().z())
    , mXlen(bbox.max().x() - bbox.min().x()), mOx(bbox.min().x())
    , mYlen(bbox.max().y() - bbox.min().y()), mOy(bbox.max().y())
    {
        openvdbmma::types::pixel_type_assert<ValueT>();
    }

    DepthMap(const DepthMap& other, tbb::split)
    : im(other.im), mBBox(other.mBBox), mXlen(other.mXlen), mYlen(other.mYlen)
    , mOx(other.mOx), mOy(other.mOy), mOz(other.mOz)
    , mIntensities(other.mIntensities), mMultiply(other.mMultiply)
    {
    }

    void operator()(const RootT& node) {}

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if (!mBBox.hasOverlap(node.getNodeBoundingBox()))
            return false;

        const int child_dim = NodeT::ChildNodeType::DIM;

        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            const CoordBBox bbox(
                CoordBBox::createCube(iter.getCoord(), NodeT::ChildNodeType::DIM));

            if (mBBox.hasOverlap(bbox)) {
                const float ival = nodeValue(*iter, bbox);

                const int xstart = xLeft(bbox), xend = xRight(bbox);
                const int ystart = yLeft(bbox), yend = yRight(bbox);

                for(int j = ystart; j <= yend; j++)
                    for(int i = xstart; i <= xend; i++)
                        if(ival > im(j, i))
                            im(j, i) = ival;
            }
        }

        return true;
    }

    inline bool operator()(const LeafT& leaf, size_t)
    {
        if (!mBBox.hasOverlap(leaf.getNodeBoundingBox()))
            return false;

        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            const Coord p = iter.getCoord();
            if (mBBox.isInside(p)) {
                const float ival = leafValue(*iter, p);
                const int i = xPos(p);
                const int j = yPos(p);

                if(ival > im(j, i))
                    im(j, i) = ival;
            }
        }

        return false;
    }

    void join(const DepthMap& other) {}

    mma::ImageRef<mma::im_real32_t> im;

private:

    inline float nodeValue(const float& x, const CoordBBox& bbox)
    {
        const int zclip = math::Min(mBBox.max().z(), bbox.max().z()) - mOz;
        return (mMultiply ? x : 1.0) * mIntensities[zclip];
    }

    inline float leafValue(const float& x, const Coord& p)
    {
        return (mMultiply ? x : 1.0) * mIntensities[p.z() - mOz];
    }

    inline int xLeft (const CoordBBox& bbox) const { return math::Max(0, bbox.min().x() - mOx); }
    inline int xRight(const CoordBBox& bbox) const { return math::Min(mXlen, bbox.max().x() - mOx); }

    inline int yLeft (const CoordBBox& bbox) const { return math::Max(0, mOy - bbox.max().y()); }
    inline int yRight(const CoordBBox& bbox) const { return math::Min(mYlen, mOy - bbox.min().y()); }

    inline int xPos(const Coord& p) const { return p.x() - mOx; }
    inline int yPos(const Coord& p) const { return mOy - p.y(); }

    //////////// private members

    const CoordBBox mBBox;
    const int mOx, mOy, mOz;
    const int mXlen, mYlen;
    const std::vector<float> mIntensities;
    const bool mMultiply;
};


template<typename TreeType>
struct GridSliceImage
{
    using ValueT = typename TreeType::ValueType;
    using RootT  = typename TreeType::RootNodeType;
    using LeafT  = typename TreeType::LeafNodeType;

    using PixelT = typename std::conditional_t<
            mask_type<ValueT>::value || bool_type<ValueT>::value,
            mma::im_bit_t,
            mma::im_byte_t
        >;

    using mmaImageT = typename mma::ImageRef<PixelT>;

    explicit GridSliceImage(const int& slice, const int& xmn, const int& xmx,
        const int& ymn, const int& ymx, const ValueT& vmn, const ValueT& vmx, const bool& mirror)
    : im(openvdbmma::image::makeEmptyImage<PixelT>(xmx-xmn+1, ymx-ymn+1))
    , mZ(slice), mXmin(xmn), mXmax(xmx), mYmin(ymn), mYmax(ymx)
    , mXlen(xmx - xmn), mYlen(ymx - ymn), mMirror(mirror)
    , mVmin(vmn), mVmax(vmx), mFac(rangeNormalization(vmn, vmx))
    {
        openvdbmma::types::pixel_type_assert<ValueT>();
    }

    GridSliceImage(const GridSliceImage& other, tbb::split)
    : im(other.im), mZ(other.mZ), mXmin(other.mXmin), mXmax(other.mXmax)
    , mYmin(other.mYmin), mYmax(other.mYmax), mXlen(other.mXlen), mYlen(other.mYlen)
    , mVmin(other.mVmin), mVmax(other.mVmax), mFac(other.mFac), mMirror(other.mMirror)
    {
    }

    void operator()(const RootT& node) {}

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if (noOverlap(node.getNodeBoundingBox()))
            return false;

        const int child_dim = NodeT::ChildNodeType::DIM;

        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            const Coord p = iter.getCoord();

            if (hasOverlap(p, child_dim)) {
                const PixelT ival = nodeValue(*iter);

                const int xstart = xLeft(p, child_dim), xend = xRight(p, child_dim);
                const int ystart = yLeft(p, child_dim), yend = yRight(p, child_dim);

                for(int j = ystart; j <= yend; j++)
                    for(int i = xstart; i <= xend; i++)
                        im(j, i) = ival;
            }
        }

        return true;
    }

    inline bool operator()(const LeafT& leaf, size_t)
    {
        if (noOverlap(leaf.getNodeBoundingBox()))
            return false;

        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            const Coord p = iter.getCoord();
            if (isInside(p))
                im(yPos(p), xPos(p)) = leafValue(*iter);
        }

        return false;
    }

    void join(const GridSliceImage& other) {}

    mmaImageT im;

private:

    inline bool hasOverlap(const Coord& p, const int& len) const
    {
        return p.z() <= mZ && p.x() <= mXmax && p.y() <= mYmax
            && mZ <= p.z() + len && mXmin <= p.x() + len && mYmin <= p.y() + len;
    }

    inline bool noOverlap(const CoordBBox& bbox) const
    {
        return bbox.min().z() > mZ || bbox.max().z() < mZ
            || bbox.min().x() > mXmax || bbox.max().x() < mXmin
            || bbox.min().y() > mYmax || bbox.max().y() < mYmin;
    }

    inline bool isInside(const Coord& p) const
    {
        return p.z() == mZ && mXmin <= p.x() && p.x() <= mXmax
            && mYmin <= p.y() && p.y() <= mYmax;
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<scalar_type<V>::value, PixelT>
    nodeValue(const ValueT& x)
    {
        return internal::to_uchar<ValueT>(mFac * (x - mVmin));
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<!scalar_type<V>::value, PixelT>
    nodeValue(const ValueT& x)
    {
        return x;
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<scalar_type<V>::value, PixelT>
    leafValue(const ValueT& x)
    {
        return internal::to_uchar<ValueT>(mVmin < 0.0 && math::Abs(x) == mVmax ? (ValueT)0 : mFac * (x - mVmin));
    }

    template<typename V = ValueT>
    inline typename std::enable_if_t<!scalar_type<V>::value, PixelT>
    leafValue(const ValueT& x)
    {
        return x;
    }

    inline int xLeft (const Coord& p, const int& len) const
    {
        return mMirror ? math::Max(0, mXmax - p.x() - len) : math::Max(0, p.x() - mXmin);
    }
    inline int xRight(const Coord& p, const int& len) const
    {
        return mMirror ? math::Min(mXlen, mXmax - p.x()) : math::Min(mXlen, p.x() - mXmin + len);
    }

    inline int yLeft (const Coord& p, const int& len) const { return math::Max(0, mYmax - p.y() - len); }
    inline int yRight(const Coord& p, const int& len) const { return math::Min(mYlen, mYmax - p.y()); }

    inline int xPos(const Coord& p) const { return mMirror ? mXmax - p.x() : p.x() - mXmin; }
    inline int yPos(const Coord& p) const { return mYmax - p.y(); }

    template<typename V = ValueT>
    inline typename std::enable_if<scalar_type<V>::value, ValueT>::type
    rangeNormalization(const ValueT& vmn, const ValueT& vmx)
    {
        return vmn != vmx ? (ValueT)1/(vmx-vmn) : (ValueT)1;
    }

    template<typename V = ValueT>
    inline typename std::enable_if<!scalar_type<V>::value, ValueT>::type
    rangeNormalization(const ValueT&, const ValueT&)
    {
        return (ValueT)1;
    }

    //////////// private members

    const int mZ, mXmin, mXmax, mYmin, mYmax;
    const int mXlen, mYlen;
    const ValueT mVmin, mVmax, mFac;
    const bool mMirror;
};

} // namespace image
} // namespace openvdbmma

#endif // OPENVDBLINK_UTILITIES_IMAGE_HAS_BEEN_INCLUDED
