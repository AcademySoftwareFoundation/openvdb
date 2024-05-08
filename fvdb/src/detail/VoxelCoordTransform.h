#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>

#include "detail/utils/Utils.h"


namespace fvdb {
namespace detail {

/// @brief A class representing the the transformation from world space (xyz) to voxel space (ijk)
///        its inverse, and gradient. It can be applied to points, vectors and rays. It stores the transformation in
///        float16, float32 and float64 precision, using the appropriate representation depending on the
///        input types.
/// @note  This class currently only supports translation and non-uniform scaling transformations.
struct VoxelCoordTransform {

    /// @brief Construct a voxel coordinate transform with identity transformation
    /// @return The voxel coordinate transform
    __hostdev__ VoxelCoordTransform() {};

    /// @brief Construct a voxel coordinate transform that scales and translates each input point when mappint to voxel coordinates
    /// @param scale The 3D scale to apply to each input point
    /// @param translate The 3D translation to apply to each input point
    __hostdev__ VoxelCoordTransform(const nanovdb::Vec3d& scale, const nanovdb::Vec3d& translate) : mTransform(scale, translate) {}

    /// @brief Apply the gradient of the transformation (from xyz to ijk) to an input point xyz
    /// @tparam ScalarT The scalar type of the input point xyz
    /// @param xyz The input point to apply the gradient to
    /// @return The gradient dT/dxyz of the transformation applied to xyz
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> applyGrad(const nanovdb::math::Vec3<ScalarT>& xyz) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        return mTransform.scale<ScalarT>();
    }

    /// @brief Apply the gradient of the transformation (from xyz to ijk) to an input point (x, y, z)
    /// @tparam ScalarT The scalar type of the input point (x, y, z)
    /// @param x The x component of the input point to apply the gradient to
    /// @param y The y component of the input point to apply the gradient to
    /// @param z The z component of the input point to apply the gradient to
    /// @return The gradient dT/d(x, y, z) of the transformation applied to (x, y, z)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> applyGrad(ScalarT x, ScalarT y, ScalarT z) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        return mTransform.scale<ScalarT>();
    }


    /// @brief Apply the gradient of the inverse transformation (from ijk to xyz) to an input coordinate ijk
    /// @tparam ScalarT The scalar type of the input coordinate ijk
    /// @param ijk The input point to apply the gradient to
    /// @return The gradient dT^-1/dijk of the inverse transformation applied to ijk
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> applyInvGrad(const nanovdb::math::Vec3<ScalarT>& ijk) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        return nanovdb::math::Vec3<ScalarT>(1.0, 1.0, 1.0) / mTransform.scale<ScalarT>();
    }

    /// @brief Apply the gradient of the inverse transformation (from ijk to xyz) to an input coordinate (i, j, k)
    /// @tparam ScalarT ScalarT The scalar type of the input coordinate (i, j, k)
    /// @param i The i component of the input coordinate to apply the gradient to
    /// @param j The j component of the input coordinate to apply the gradient to
    /// @param k The k component of the input coordinate to apply the gradient to
    /// @return The gradient dT^-1/d(i, j, k) of the inverse transformation applied to (i, j, k)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> applyInvGrad(ScalarT i, ScalarT j, ScalarT k) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        return nanovdb::math::Vec3<ScalarT>(1.0, 1.0, 1.0) / mTransform.scale<ScalarT>();
    }



    /// @brief Apply the transformation (from xyz to ijk) to an input point xyz
    /// @tparam ScalarT The scalar type of the input point xyz
    /// @param xyz The input point to apply the transformation to
    /// @return The transformed point T(xyz)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> apply(const nanovdb::math::Vec3<ScalarT>& xyz) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        return xyz * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>();
    }

    /// @brief Apply the transformation (from xyz to ijk) to an input point (x, y, z)
    /// @tparam ScalarT The scalar type of the input point (x, y, z)
    /// @param x The x component of the input point to apply the transformation to
    /// @param y The y component of the input point to apply the transformation to
    /// @param z The z component of the input point to apply the transformation to
    /// @return The transformed point T(x, y, z)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> apply(ScalarT x, ScalarT y, ScalarT z) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        const nanovdb::math::Vec3<ScalarT> xyz(x, y, z);
        return xyz * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>();
    }

    /// @brief Apply the transformation (from xyz to ijk) to an input point xyz which is of an indexable type
    /// @tparam ScalarT The scalar type of the input point xyz
    /// @param xyz The input point to apply the transformation to (must support indexing with [0], [1], [2])
    /// @return The transformed point T(xyz)
    template <typename ScalarT, typename InVec3T>
    __hostdev__ nanovdb::math::Vec3<ScalarT> apply(const InVec3T& xyz) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        const nanovdb::math::Vec3<ScalarT> pt(xyz[0], xyz[1], xyz[2]);
        return pt * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>();
    }



    /// @brief Apply the inverse transformation (from ijk to xyz) to an input coordinate ijk
    /// @tparam ScalarT The scalar type of the input coordinate ijk
    /// @param ijk The input coordinate to apply the inverse transformation to
    /// @return The transformed coordinate T^-1(ijk)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> applyInv(const nanovdb::math::Vec3<ScalarT>& ijk) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        return (ijk - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>();
    }

    /// @brief Apply the inverse transformation (from ijk to xyz) to an input coordinate (i, j, k)
    /// @tparam ScalarT The scalar type of the input coordinate (i, j, k)
    /// @param i The i component of the input coordinate to apply the inverse transformation to
    /// @param j The j component of the input coordinate to apply the inverse transformation to
    /// @param k The k component of the input coordinate to apply the inverse transformation to
    /// @return The transformed coordinate T^-1(i, j, k)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> applyInv(ScalarT i, ScalarT j, ScalarT k) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        const nanovdb::math::Vec3<ScalarT> ijk(i, j, k);
        return (ijk - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>();
    }

    /// @brief Apply the inverse transformation (from ijk to xyz) to an input coordinate ijk which is of an indexable type
    /// @tparam ScalarT The scalar type of the input coordinate ijk
    /// @param ijk The input coordinate to apply the inverse transformation to (must support indexing with [0], [1], [2])
    /// @return The transformed coordinate T^-1(ijk)
    template <typename ScalarT, typename InVec3T>
    __hostdev__ nanovdb::math::Vec3<ScalarT> applyInv(const InVec3T& ijk) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        const nanovdb::math::Vec3<ScalarT> coord(ijk[0], ijk[1], ijk[2]);
        return (coord - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>();
    }



    /// @brief Apply the transformation (from xyz to ijk) to an input ray
    /// @tparam ScalarT The scalar type of the input ray
    /// @param ray The input ray to apply the transformation to
    /// @return The transformed ray T(ray)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Ray<ScalarT> applyToRay(nanovdb::math::Ray<ScalarT> ray) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        const nanovdb::math::Vec3<ScalarT> oVox = apply<ScalarT>(ray.eye());
        const nanovdb::math::Vec3<ScalarT> dVox = ray.dir() * mTransform.scale<ScalarT>();
        return nanovdb::math::Ray<ScalarT>(oVox, dVox, ray.t0(), ray.t1());
    }

    /// @brief Apply the transformation (from xyz to ijk) to an input ray
    /// @tparam ScalarT The scalar type of the input ray
    /// @param rayOx The x component of the input ray origin
    /// @param rayOy The y component of the input ray origin
    /// @param rayOz The z component of the input ray origin
    /// @param rayDx The x component of the input ray direction
    /// @param rayDy The y component of the input ray direction
    /// @param rayDz The z component of the input ray direction
    /// @param t0 The minimum ray time parameter
    /// @param t1 The maximum ray time parameter
    /// @return The transformed ray T(ray)
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Ray<ScalarT> applyToRay(ScalarT rayOx, ScalarT rayOy, ScalarT rayOz,
                                                 ScalarT rayDx, ScalarT rayDy, ScalarT rayDz,
                                                 ScalarT t0 = static_cast<ScalarT>(0),
                                                 ScalarT t1 = std::numeric_limits<ScalarT>::infinity()) const {
        static_assert(is_floating_point_or_half<ScalarT>::value);
        const nanovdb::math::Vec3<ScalarT> oVox = apply<ScalarT>(rayOx, rayOy, rayOz);
        const nanovdb::math::Vec3<ScalarT> dVox = nanovdb::math::Vec3<ScalarT>(rayDx, rayDy, rayDz) * mTransform.scale<ScalarT>();
        return nanovdb::math::Ray<ScalarT>(oVox, dVox, t0, t1);
    }

    /// @brief Get the scale component of this transformation
    /// @tparam ScalarT The scalar type to return the scale in
    /// @return The scale component of this transformation
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> scale() const {
        return mTransform.scale<ScalarT>();
    }

    /// @brief Get the translation component of this transformation
    /// @tparam ScalarT The scalar type to return the translation in
    /// @return The translation component of this transformation
    template <typename ScalarT>
    __hostdev__ nanovdb::math::Vec3<ScalarT> translate() const {
        return mTransform.translate<ScalarT>();
    }


private:
    /// @brief A struct representing the transformation from world space (xyz) to voxel space (ijk)
    ///        in float16, float32, and float64. You can access the scale and translation in any of these
    ///        by calling methods with the appropriate template paramter
    struct Transform {
        /// @brief Construct an identity transformation
        __hostdev__ Transform() {};

        /// @brief Construct a transformation that scales and translates each input point
        __hostdev__ Transform(nanovdb::Vec3d scale, const nanovdb::Vec3d& translate) :
            mScaleh(nanovdb::math::Vec3<c10::Half>(c10::Half(float(scale[0])), c10::Half(float(scale[1])), c10::Half(float(scale[2])))),
            mTranslateh(nanovdb::math::Vec3<c10::Half>(c10::Half(float(translate[0])), c10::Half(float(translate[1])), c10::Half(float(translate[2])))),
            mScalef(nanovdb::Vec3f(scale[0], scale[1], scale[2])),
            mTranslatef(nanovdb::Vec3f(translate[0], translate[1], translate[2])),
            mScaled(scale),
            mTranslated(translate) {}

        nanovdb::math::Vec3<c10::Half> mScaleh = nanovdb::math::Vec3<c10::Half>(c10::Half(1.0f), c10::Half(1.0f), c10::Half(1.0f));
        nanovdb::math::Vec3<c10::Half> mTranslateh = nanovdb::math::Vec3<c10::Half>(c10::Half(0.0f), c10::Half(0.0f), c10::Half(0.0f));
        nanovdb::Vec3f mScalef = nanovdb::Vec3f(1.0f, 1.0f, 1.0f);
        nanovdb::Vec3f mTranslatef = nanovdb::Vec3f(0.0f, 0.0f, 0.0f);
        nanovdb::Vec3d mScaled = nanovdb::Vec3d(1.0, 1.0, 1.0);
        nanovdb::Vec3d mTranslated = nanovdb::Vec3d(0.0, 0.0, 0.0);

        /// @brief Get the scale component of this transformation
        /// @tparam T The scalar type to return the scale in
        /// @return The scale component of this transformation
        template <typename T>
        __hostdev__ inline const nanovdb::math::Vec3<T>& scale() const;

        /// @brief Get the translation component of this transformation
        /// @tparam T The scalar type to return the translation in
        /// @return The translation component of this transformation
        template <typename T>
        __hostdev__ inline const nanovdb::math::Vec3<T>&  translate() const;
    } mTransform;
};

// Template specializations to return the appropriate types
template <>
__hostdev__ inline const nanovdb::math::Vec3<c10::Half>& VoxelCoordTransform::Transform::scale<c10::Half>() const {
    return mScaleh;
}
template <>
__hostdev__ inline const nanovdb::Vec3f& VoxelCoordTransform::Transform::scale<float>() const {
    return mScalef;
}
template <>
__hostdev__ inline const nanovdb::Vec3d& VoxelCoordTransform::Transform::scale<double>() const {
    return mScaled;
}
template <>
__hostdev__ inline const nanovdb::math::Vec3<c10::Half>& VoxelCoordTransform::Transform::translate<c10::Half>() const {
    return mTranslateh;
}
template <>
__hostdev__ inline const nanovdb::Vec3f& VoxelCoordTransform::Transform::translate<float>() const {
    return mTranslatef;
}
template <>
__hostdev__ inline const nanovdb::Vec3d& VoxelCoordTransform::Transform::translate<double>() const {
    return mTranslated;
}

/// @brief Get a primal voxel transform given a voxel size and the coordinate of the [0, 0, 0] voxel center
/// @param voxSize The size of each voxel in the grid
/// @param voxOrigin The coordinate of the [0, 0, 0] voxel center
/// @return The primal voxel transform
inline __hostdev__ VoxelCoordTransform primalVoxelTransformForSizeAndOrigin(const nanovdb::Vec3d& voxSize, const nanovdb::Vec3d& voxOrigin) {
    // TORCH_CHECK_VALUE(voxSize[0] > 0.0 && voxSize[1] > 0.0 && voxSize[2] > 0.0, "voxel_size must be positive");
    const nanovdb::Vec3d w = voxSize;
    const nanovdb::Vec3d tx = voxOrigin;
    const nanovdb::Vec3d invW = nanovdb::Vec3d(1.0, 1.0, 1.0) / w;
    const nanovdb::Vec3d half(0.5, 0.5, 0.5);

    return VoxelCoordTransform(invW, -tx / w);
}

/// @brief Get the primal and dual transforms for a grid given a voxel size and the coordinate of the [0, 0, 0] voxel center
/// @param voxSize The size of each voxel in the grid
/// @param voxOrigin The coordinate of the [0, 0, 0] voxel center
/// @param outPrimal Output primal transform
/// @param outDual Output dual transform
inline __hostdev__ void voxelTransformForSizeAndOrigin(const nanovdb::Vec3d& voxSize, const nanovdb::Vec3d& voxOrigin,
                                                       VoxelCoordTransform& outPrimal, VoxelCoordTransform& outDual) {
    // TORCH_CHECK_VALUE(voxSize[0] > 0.0 && voxSize[1] > 0.0 && voxSize[2] > 0.0, "voxel_size must be positive");
    const nanovdb::Vec3d w = voxSize;
    const nanovdb::Vec3d tx = voxOrigin;
    const nanovdb::Vec3d invW = nanovdb::Vec3d(1.0, 1.0, 1.0) / w;
    const nanovdb::Vec3d half(0.5, 0.5, 0.5);

    outPrimal = VoxelCoordTransform(invW, -tx / w);
    outDual = VoxelCoordTransform(invW, -tx / w + half);
}

} // namespace detail
} // namespace fvdb
