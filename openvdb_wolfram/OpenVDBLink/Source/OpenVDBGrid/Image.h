// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_IMAGE_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_IMAGE_HAS_BEEN_INCLUDED

#include "../Utilities/Image.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <openvdb/tools/GridTransformer.h>
#include <openvdb/math/Transform.h>

#include <vector>
#include <math.h>


/* OpenVDBGrid public member function list

mma::ImageRef<mma::im_real32_t> depthMap(mma::IntBounds3DRef bds,
    const double gamma, const double imin, const double imax)

 mma::GenericImageRef gridSliceImage(const mint z, mma::IntBounds2DRef bds,
    const bool mirror_image, const bool threaded)

 mma::GenericImageRef gridImage3D(mma::IntBounds3DRef bds)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
mma::ImageRef<mma::im_real32_t>
openvdbmma::OpenVDBGrid<V>::depthMap(mma::IntBounds3DRef bds,
    const double gamma, const double imin, const double imax) const
{
    pixel_type_assert<V>();

    using ValueT = typename wlGridType::ValueType;

    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    // avoid possible invalid exponentiation
    if(gamma <= 0.0 || imin < 0.0 || imax < 0.0)
        throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

    const CoordBBox bbox(bds.toCoordBBox());

    const int x = bds.xDim();
    const int y = bds.yDim();
    const int z = bds.zDim();

    std::vector<float> intensities(z);
    const double igmin = math::Pow(imin, 1/gamma), igmax = math::Pow(imax, 1/gamma);
    const double idelta = (igmax - igmin)/(z > 1 ? z-1 : 1);

    const bool multiply_value = !mask_type<V>::value && !bool_type<V>::value
        && (!scalar_type<V>::value || grid()->getGridClass() == GRID_FOG_VOLUME);

    intensities[0] = imin;
    intensities[z-1] = imax;
    for(int k = 1; k < z-1; k++) {
        intensities[k] = math::Pow(igmin + k * idelta, gamma);
    }

    if (multiply_value && !scalar_type<V>::value) {
        const float fac = 1.0f/((float)std::numeric_limits<ValueT>::max());
        for(int k = 1; k < z-1; k++) {
            intensities[k] *= fac;
        }
    }

    openvdbmma::image::DepthMap<wlTreeType> op(bbox, intensities, multiply_value);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());

    // Can we parallelize? Perhaps an array of mutexes keyed on (x, y), or z?
    nodeManager.reduceTopDown(op, false);

    return op.im;
}

template<typename V>
mma::GenericImageRef
openvdbmma::OpenVDBGrid<V>::gridSliceImage(const mint z, mma::IntBounds2DRef bds,
    const bool mirror_image, const bool threaded) const
{
    pixel_type_assert<V>();

    using ValueT = typename wlGridType::ValueType;

    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    // No built in 2D CoordBBox, so just pass in ordinates
    const int xmin = bds.xmin();
    const int xmax = bds.xmax();
    const int ymin = bds.ymin();
    const int ymax = bds.ymax();

    openvdbmma::image::pixelExtrema<wlGridType> extrema(grid());
    const ValueT vmin = extrema.min, vmax = extrema.max;

    openvdbmma::image::GridSliceImage<wlTreeType> op(
        z, xmin, xmax, ymin, ymax, vmin, vmax, mirror_image);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, threaded);

    return op.im;
}

template<typename V>
mma::GenericImage3DRef
openvdbmma::OpenVDBGrid<V>::gridImage3D(mma::IntBounds3DRef bds) const
{
    pixel_type_assert<V>();

    using ValueT = typename wlGridType::ValueType;

    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    openvdbmma::image::pixelExtrema<wlGridType> extrema(grid());
    const ValueT vmin = extrema.min, vmax = extrema.max;

    openvdbmma::image::GridImage3D<wlTreeType> op(bds.toCoordBBox(), vmin, vmax);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, true);

    return op.im;
}

#endif // OPENVDBLINK_OPENVDBGRID_IMAGE_HAS_BEEN_INCLUDED
