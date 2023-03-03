// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>

#include <gtest/gtest.h>


using namespace openvdb;

class TestTypes: public ::testing::Test
{
};


namespace { struct Dummy {}; }

// Work-around for a macro expansion bug in debug mode that presents as an
// undefined reference linking error. This happens in cases where template
// instantiation of a type trait is prevented from occurring as the template
// instantiation is deemed to be in an unreachable code block.
// The work-around is to wrap the EXPECT_TRUE macro in another which holds
// the expected value in a temporary.
#define EXPECT_TRUE_TEMP(expected) \
    { bool result = expected; EXPECT_TRUE(result); }

TEST_F(TestTypes, testVecTraits)
{
    { // VecTraits - IsVec

        // standard types (Vec3s, etc)
        EXPECT_TRUE_TEMP(VecTraits<Vec3s>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec3d>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec3i>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec2i>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec2s>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec2d>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec4i>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec4s>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec4d>::IsVec);

        // some less common types (Vec3U16, etc)
        EXPECT_TRUE_TEMP(VecTraits<Vec2R>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec3U16>::IsVec);
        EXPECT_TRUE_TEMP(VecTraits<Vec4H>::IsVec);

        // some non-vector types
        EXPECT_TRUE_TEMP(!VecTraits<int>::IsVec);
        EXPECT_TRUE_TEMP(!VecTraits<double>::IsVec);
        EXPECT_TRUE_TEMP(!VecTraits<bool>::IsVec);
        EXPECT_TRUE_TEMP(!VecTraits<Quats>::IsVec);
        EXPECT_TRUE_TEMP(!VecTraits<Mat4d>::IsVec);
        EXPECT_TRUE_TEMP(!VecTraits<ValueMask>::IsVec);
        EXPECT_TRUE_TEMP(!VecTraits<Dummy>::IsVec);
        EXPECT_TRUE_TEMP(!VecTraits<Byte>::IsVec);
    }

    { // VecTraits - Size

        // standard types (Vec3s, etc)
        EXPECT_TRUE(VecTraits<Vec3s>::Size == 3);
        EXPECT_TRUE(VecTraits<Vec3d>::Size == 3);
        EXPECT_TRUE(VecTraits<Vec3i>::Size == 3);
        EXPECT_TRUE(VecTraits<Vec2i>::Size == 2);
        EXPECT_TRUE(VecTraits<Vec2s>::Size == 2);
        EXPECT_TRUE(VecTraits<Vec2d>::Size == 2);
        EXPECT_TRUE(VecTraits<Vec4i>::Size == 4);
        EXPECT_TRUE(VecTraits<Vec4s>::Size == 4);
        EXPECT_TRUE(VecTraits<Vec4d>::Size == 4);

        // some less common types (Vec3U16, etc)
        EXPECT_TRUE(VecTraits<Vec2R>::Size == 2);
        EXPECT_TRUE(VecTraits<Vec3U16>::Size == 3);
        EXPECT_TRUE(VecTraits<Vec4H>::Size == 4);

        // some non-vector types
        EXPECT_TRUE(VecTraits<int>::Size == 1);
        EXPECT_TRUE(VecTraits<double>::Size == 1);
        EXPECT_TRUE(VecTraits<bool>::Size == 1);
        EXPECT_TRUE(VecTraits<Quats>::Size == 1);
        EXPECT_TRUE(VecTraits<Mat4d>::Size == 1);
        EXPECT_TRUE(VecTraits<ValueMask>::Size == 1);
        EXPECT_TRUE(VecTraits<Dummy>::Size == 1);
        EXPECT_TRUE(VecTraits<Byte>::Size == 1);
    }

    { // VecTraits - ElementType

        // standard types (Vec3s, etc)
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec3s>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec3d>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec3i>::ElementType, int>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec2i>::ElementType, int>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec2s>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec2d>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec4i>::ElementType, int>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec4s>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec4d>::ElementType, double>::value));

        // some less common types (Vec3U16, etc)
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec2R>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec3U16>::ElementType, uint16_t>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Vec4H>::ElementType, math::half>::value));

        // some non-vector types
        EXPECT_TRUE(bool(std::is_same<VecTraits<int>::ElementType, int>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<double>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<bool>::ElementType, bool>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Quats>::ElementType, Quats>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Mat4d>::ElementType, Mat4d>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<ValueMask>::ElementType, ValueMask>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Dummy>::ElementType, Dummy>::value));
        EXPECT_TRUE(bool(std::is_same<VecTraits<Byte>::ElementType, Byte>::value));
    }
}

TEST_F(TestTypes, testQuatTraits)
{
    { // QuatTraits - IsQuat

        // standard types (Quats, etc)
        EXPECT_TRUE_TEMP(QuatTraits<Quats>::IsQuat);
        EXPECT_TRUE_TEMP(QuatTraits<Quatd>::IsQuat);

        // some non-quaternion types
        EXPECT_TRUE_TEMP(!QuatTraits<Vec3s>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<Vec4d>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<Vec2i>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<Vec3U16>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<int>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<double>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<bool>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<Mat4s>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<ValueMask>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<Dummy>::IsQuat);
        EXPECT_TRUE_TEMP(!QuatTraits<Byte>::IsQuat);
    }

    { // QuatTraits - Size

        // standard types (Quats, etc)
        EXPECT_TRUE(QuatTraits<Quats>::Size == 4);
        EXPECT_TRUE(QuatTraits<Quatd>::Size == 4);

        // some non-quaternion types
        EXPECT_TRUE(QuatTraits<Vec3s>::Size == 1);
        EXPECT_TRUE(QuatTraits<Vec4d>::Size == 1);
        EXPECT_TRUE(QuatTraits<Vec2i>::Size == 1);
        EXPECT_TRUE(QuatTraits<Vec3U16>::Size == 1);
        EXPECT_TRUE(QuatTraits<int>::Size == 1);
        EXPECT_TRUE(QuatTraits<double>::Size == 1);
        EXPECT_TRUE(QuatTraits<bool>::Size == 1);
        EXPECT_TRUE(QuatTraits<Mat4s>::Size == 1);
        EXPECT_TRUE(QuatTraits<ValueMask>::Size == 1);
        EXPECT_TRUE(QuatTraits<Dummy>::Size == 1);
        EXPECT_TRUE(QuatTraits<Byte>::Size == 1);
    }

    { // QuatTraits - ElementType

        // standard types (Quats, etc)
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Quats>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Quatd>::ElementType, double>::value));

        // some non-matrix types
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Vec3s>::ElementType, Vec3s>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Vec4d>::ElementType, Vec4d>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Vec2i>::ElementType, Vec2i>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Vec3U16>::ElementType, Vec3U16>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<int>::ElementType, int>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<double>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<bool>::ElementType, bool>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Mat4s>::ElementType, Mat4s>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<ValueMask>::ElementType, ValueMask>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Dummy>::ElementType, Dummy>::value));
        EXPECT_TRUE(bool(std::is_same<QuatTraits<Byte>::ElementType, Byte>::value));
    }
}

TEST_F(TestTypes, testMatTraits)
{
    { // MatTraits - IsMat

        // standard types (Mat4d, etc)
        EXPECT_TRUE_TEMP(MatTraits<Mat3s>::IsMat);
        EXPECT_TRUE_TEMP(MatTraits<Mat3d>::IsMat);
        EXPECT_TRUE_TEMP(MatTraits<Mat4s>::IsMat);
        EXPECT_TRUE_TEMP(MatTraits<Mat4d>::IsMat);

        // some non-matrix types
        EXPECT_TRUE_TEMP(!MatTraits<Vec3s>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<Vec4d>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<Vec2i>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<Vec3U16>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<int>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<double>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<bool>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<Quats>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<ValueMask>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<Dummy>::IsMat);
        EXPECT_TRUE_TEMP(!MatTraits<Byte>::IsMat);
    }

    { // MatTraits - Size

        // standard types (Mat4d, etc)
        EXPECT_TRUE(MatTraits<Mat3s>::Size == 3);
        EXPECT_TRUE(MatTraits<Mat3d>::Size == 3);
        EXPECT_TRUE(MatTraits<Mat4s>::Size == 4);
        EXPECT_TRUE(MatTraits<Mat4d>::Size == 4);

        // some non-matrix types
        EXPECT_TRUE(MatTraits<Vec3s>::Size == 1);
        EXPECT_TRUE(MatTraits<Vec4d>::Size == 1);
        EXPECT_TRUE(MatTraits<Vec2i>::Size == 1);
        EXPECT_TRUE(MatTraits<Vec3U16>::Size == 1);
        EXPECT_TRUE(MatTraits<int>::Size == 1);
        EXPECT_TRUE(MatTraits<double>::Size == 1);
        EXPECT_TRUE(MatTraits<bool>::Size == 1);
        EXPECT_TRUE(MatTraits<Quats>::Size == 1);
        EXPECT_TRUE(MatTraits<ValueMask>::Size == 1);
        EXPECT_TRUE(MatTraits<Dummy>::Size == 1);
        EXPECT_TRUE(MatTraits<Byte>::Size == 1);
    }

    { // MatTraits - ElementType

        // standard types (Mat4d, etc)
        EXPECT_TRUE(bool(std::is_same<MatTraits<Mat3s>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Mat3d>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Mat4s>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Mat4d>::ElementType, double>::value));

        // some non-matrix types
        EXPECT_TRUE(bool(std::is_same<MatTraits<Vec3s>::ElementType, Vec3s>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Vec4d>::ElementType, Vec4d>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Vec2i>::ElementType, Vec2i>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Vec3U16>::ElementType, Vec3U16>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<int>::ElementType, int>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<double>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<bool>::ElementType, bool>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Quats>::ElementType, Quats>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<ValueMask>::ElementType, ValueMask>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Dummy>::ElementType, Dummy>::value));
        EXPECT_TRUE(bool(std::is_same<MatTraits<Byte>::ElementType, Byte>::value));
    }
}


TEST_F(TestTypes, testValueTraits)
{
    { // ValueTraits - IsVec, IsQuat, IsMat, IsScalar

        // vector types
        EXPECT_TRUE_TEMP(ValueTraits<Vec3s>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec3s>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec3s>::IsMat);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec3s>::IsScalar);
        EXPECT_TRUE_TEMP(ValueTraits<Vec4d>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec4d>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec4d>::IsMat);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec4d>::IsScalar);
        EXPECT_TRUE_TEMP(ValueTraits<Vec3U16>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec3U16>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec3U16>::IsMat);
        EXPECT_TRUE_TEMP(!ValueTraits<Vec3U16>::IsScalar);

        // quaternion types
        EXPECT_TRUE_TEMP(!ValueTraits<Quats>::IsVec);
        EXPECT_TRUE_TEMP(ValueTraits<Quats>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<Quats>::IsMat);
        EXPECT_TRUE_TEMP(!ValueTraits<Quats>::IsScalar);
        EXPECT_TRUE_TEMP(!ValueTraits<Quatd>::IsVec);
        EXPECT_TRUE_TEMP(ValueTraits<Quatd>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<Quatd>::IsMat);
        EXPECT_TRUE_TEMP(!ValueTraits<Quatd>::IsScalar);

        // matrix types
        EXPECT_TRUE_TEMP(!ValueTraits<Mat3s>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<Mat3s>::IsQuat);
        EXPECT_TRUE_TEMP(ValueTraits<Mat3s>::IsMat);
        EXPECT_TRUE_TEMP(!ValueTraits<Mat3s>::IsScalar);
        EXPECT_TRUE_TEMP(!ValueTraits<Mat4d>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<Mat4d>::IsQuat);
        EXPECT_TRUE_TEMP(ValueTraits<Mat4d>::IsMat);
        EXPECT_TRUE_TEMP(!ValueTraits<Mat4d>::IsScalar);

        // scalar types
        EXPECT_TRUE_TEMP(!ValueTraits<double>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<double>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<double>::IsMat);
        EXPECT_TRUE_TEMP(ValueTraits<double>::IsScalar);
        EXPECT_TRUE_TEMP(!ValueTraits<bool>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<bool>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<bool>::IsMat);
        EXPECT_TRUE_TEMP(ValueTraits<bool>::IsScalar);
        EXPECT_TRUE_TEMP(!ValueTraits<ValueMask>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<ValueMask>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<ValueMask>::IsMat);
        EXPECT_TRUE_TEMP(ValueTraits<ValueMask>::IsScalar);
        EXPECT_TRUE_TEMP(!ValueTraits<Dummy>::IsVec);
        EXPECT_TRUE_TEMP(!ValueTraits<Dummy>::IsQuat);
        EXPECT_TRUE_TEMP(!ValueTraits<Dummy>::IsMat);
        EXPECT_TRUE_TEMP(ValueTraits<Dummy>::IsScalar);
    }

    { // ValueTraits - Size

        // vector types
        EXPECT_TRUE(ValueTraits<Vec3s>::Size == 3);
        EXPECT_TRUE(ValueTraits<Vec4d>::Size == 4);
        EXPECT_TRUE(ValueTraits<Vec3U16>::Size == 3);

        // quaternion types
        EXPECT_TRUE(ValueTraits<Quats>::Size == 4);
        EXPECT_TRUE(ValueTraits<Quatd>::Size == 4);

        // matrix types
        EXPECT_TRUE(ValueTraits<Mat3s>::Size == 3);
        EXPECT_TRUE(ValueTraits<Mat4d>::Size == 4);

        // scalar types
        EXPECT_TRUE(ValueTraits<double>::Size == 1);
        EXPECT_TRUE(ValueTraits<bool>::Size == 1);
        EXPECT_TRUE(ValueTraits<ValueMask>::Size == 1);
        EXPECT_TRUE(ValueTraits<Dummy>::Size == 1);
    }

    { // ValueTraits - Elements

        // vector types
        EXPECT_TRUE(ValueTraits<Vec3s>::Elements == 3);
        EXPECT_TRUE(ValueTraits<Vec4d>::Elements == 4);
        EXPECT_TRUE(ValueTraits<Vec3U16>::Elements == 3);

        // quaternion types
        EXPECT_TRUE(ValueTraits<Quats>::Elements == 4);
        EXPECT_TRUE(ValueTraits<Quatd>::Elements == 4);

        // matrix types
        EXPECT_TRUE(ValueTraits<Mat3s>::Elements == 3*3);
        EXPECT_TRUE(ValueTraits<Mat4d>::Elements == 4*4);

        // scalar types
        EXPECT_TRUE(ValueTraits<double>::Elements == 1);
        EXPECT_TRUE(ValueTraits<bool>::Elements == 1);
        EXPECT_TRUE(ValueTraits<ValueMask>::Elements == 1);
        EXPECT_TRUE(ValueTraits<Dummy>::Elements == 1);
    }

    { // ValueTraits - ElementType

        // vector types
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Vec3s>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Vec4d>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Vec3U16>::ElementType, uint16_t>::value));

        // quaternion types
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Quats>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Quatd>::ElementType, double>::value));

        // matrix types
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Mat3s>::ElementType, float>::value));
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Mat4d>::ElementType, double>::value));

        // scalar types
        EXPECT_TRUE(bool(std::is_same<ValueTraits<double>::ElementType, double>::value));
        EXPECT_TRUE(bool(std::is_same<ValueTraits<bool>::ElementType, bool>::value));
        EXPECT_TRUE(bool(std::is_same<ValueTraits<ValueMask>::ElementType, ValueMask>::value));
        EXPECT_TRUE(bool(std::is_same<ValueTraits<Dummy>::ElementType, Dummy>::value));
    }
}


TEST_F(TestTypes, testConvertElementType)
{
    // Just replaces the type for non VDB math types
    static_assert((std::is_same<ConvertElementType<float, double>::Type, double>::value));
    static_assert((std::is_same<ConvertElementType<float, int32_t>::Type, int32_t>::value));

    // Replaces the element type for VDB Math types
    static_assert((std::is_same<ConvertElementType<Vec2f, int32_t>::Type, Vec2i>::value));
    static_assert((std::is_same<ConvertElementType<Vec2i, float>::Type, Vec2f>::value));
    static_assert((std::is_same<ConvertElementType<Vec2f, float>::Type, Vec2f>::value));
    static_assert((std::is_same<ConvertElementType<Vec2f, double>::Type, Vec2d>::value));

    static_assert((std::is_same<ConvertElementType<Vec3f, int32_t>::Type, Vec3i>::value));
    static_assert((std::is_same<ConvertElementType<Vec3f, uint32_t>::Type, math::Vec3ui>::value));
    static_assert((std::is_same<ConvertElementType<Vec3i, float>::Type, Vec3f>::value));
    static_assert((std::is_same<ConvertElementType<Vec3f, float>::Type, Vec3f>::value));
    static_assert((std::is_same<ConvertElementType<Vec3f, double>::Type, Vec3d>::value));

    static_assert((std::is_same<ConvertElementType<Vec4f, int32_t>::Type, Vec4i>::value));
    static_assert((std::is_same<ConvertElementType<Vec4i, float>::Type, Vec4f>::value));
    static_assert((std::is_same<ConvertElementType<Vec4f, float>::Type, Vec4f>::value));
    static_assert((std::is_same<ConvertElementType<Vec4f, double>::Type, Vec4d>::value));

    static_assert((std::is_same<ConvertElementType<Quats, float>::Type, Quats>::value));
    static_assert((std::is_same<ConvertElementType<Quats, double>::Type, Quatd>::value));
    static_assert((std::is_same<ConvertElementType<Quatd, float>::Type, Quats>::value));

    static_assert((std::is_same<ConvertElementType<Mat3s, float>::Type, Mat3s>::value));
    static_assert((std::is_same<ConvertElementType<Mat3s, double>::Type, Mat3d>::value));
    static_assert((std::is_same<ConvertElementType<Mat3d, float>::Type, Mat3s>::value));

    static_assert((std::is_same<ConvertElementType<Mat4s, float>::Type, Mat4s>::value));
    static_assert((std::is_same<ConvertElementType<Mat4s, double>::Type, Mat4d>::value));
    static_assert((std::is_same<ConvertElementType<Mat4d, float>::Type, Mat4s>::value));
}


TEST_F(TestTypes, testPromoteType)
{
    static_assert((std::is_same<PromoteType<math::half>::Lowest, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Previous, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Next, float>::value));
    static_assert((std::is_same<PromoteType<math::half>::Highest, double>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<0>, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<1>, float>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<2>, double>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<3>, double>::value));
    static_assert((std::is_same<PromoteType<math::half>::Demote<0>, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Demote<1>, math::half>::value));

    // floating point types
    static_assert((std::is_same<PromoteType<math::half>::Lowest, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Previous, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Next, float>::value));
    static_assert((std::is_same<PromoteType<math::half>::Highest, double>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<0>, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<1>, float>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<2>, double>::value));
    static_assert((std::is_same<PromoteType<math::half>::Promote<3>, double>::value));
    static_assert((std::is_same<PromoteType<math::half>::Demote<0>, math::half>::value));
    static_assert((std::is_same<PromoteType<math::half>::Demote<1>, math::half>::value));

    static_assert((std::is_same<PromoteType<float>::Lowest, math::half>::value));
    static_assert((std::is_same<PromoteType<float>::Previous, math::half>::value));
    static_assert((std::is_same<PromoteType<float>::Next, double>::value));
    static_assert((std::is_same<PromoteType<float>::Highest, double>::value));
    static_assert((std::is_same<PromoteType<float>::Promote<0>, float>::value));
    static_assert((std::is_same<PromoteType<float>::Promote<1>, double>::value));
    static_assert((std::is_same<PromoteType<float>::Promote<2>, double>::value));
    static_assert((std::is_same<PromoteType<float>::Demote<0>, float>::value));
    static_assert((std::is_same<PromoteType<float>::Demote<1>, math::half>::value));
    static_assert((std::is_same<PromoteType<float>::Demote<2>, math::half>::value));

    static_assert((std::is_same<PromoteType<double>::Lowest, math::half>::value));
    static_assert((std::is_same<PromoteType<double>::Previous, float>::value));
    static_assert((std::is_same<PromoteType<double>::Next, double>::value));
    static_assert((std::is_same<PromoteType<double>::Highest, double>::value));
    static_assert((std::is_same<PromoteType<double>::Promote<0>, double>::value));
    static_assert((std::is_same<PromoteType<double>::Promote<1>, double>::value));
    static_assert((std::is_same<PromoteType<double>::Demote<0>, double>::value));
    static_assert((std::is_same<PromoteType<double>::Demote<1>, float>::value));
    static_assert((std::is_same<PromoteType<double>::Demote<2>, math::half>::value));
    static_assert((std::is_same<PromoteType<double>::Demote<3>, math::half>::value));

    // int types
    static_assert((std::is_same<PromoteType<int8_t>::Lowest, int8_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Previous, int8_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Next, int16_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Highest, int64_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Promote<0>, int8_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Promote<1>, int16_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Promote<2>, int32_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Promote<3>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Promote<4>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Demote<0>, int8_t>::value));
    static_assert((std::is_same<PromoteType<int8_t>::Demote<1>, int8_t>::value));

    static_assert((std::is_same<PromoteType<int16_t>::Lowest, int8_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Previous, int8_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Next, int32_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Highest, int64_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Promote<0>, int16_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Promote<1>, int32_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Promote<2>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Promote<3>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Demote<0>, int16_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Demote<1>, int8_t>::value));
    static_assert((std::is_same<PromoteType<int16_t>::Demote<2>, int8_t>::value));

    static_assert((std::is_same<PromoteType<int32_t>::Lowest, int8_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Previous, int16_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Next, int64_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Highest, int64_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Promote<0>, int32_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Promote<1>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Promote<2>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Demote<0>, int32_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Demote<1>, int16_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Demote<2>, int8_t>::value));
    static_assert((std::is_same<PromoteType<int32_t>::Demote<3>, int8_t>::value));

    static_assert((std::is_same<PromoteType<int64_t>::Lowest, int8_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Previous, int32_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Next, int64_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Highest, int64_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Promote<0>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Promote<1>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Demote<0>, int64_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Demote<1>, int32_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Demote<2>, int16_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Demote<3>, int8_t>::value));
    static_assert((std::is_same<PromoteType<int64_t>::Demote<4>, int8_t>::value));

    // unsigned
    static_assert((std::is_same<PromoteType<uint8_t>::Lowest, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Previous, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Next, uint16_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Highest, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Promote<0>, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Promote<1>, uint16_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Promote<2>, uint32_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Promote<3>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Promote<4>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Demote<0>, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint8_t>::Demote<1>, uint8_t>::value));

    static_assert((std::is_same<PromoteType<uint16_t>::Lowest, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Previous, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Next, uint32_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Highest, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Promote<0>, uint16_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Promote<1>, uint32_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Promote<2>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Promote<3>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Demote<0>, uint16_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Demote<1>, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint16_t>::Demote<2>, uint8_t>::value));

    static_assert((std::is_same<PromoteType<uint32_t>::Lowest, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Previous, uint16_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Next, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Highest, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Promote<0>, uint32_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Promote<1>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Promote<2>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Demote<0>, uint32_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Demote<1>, uint16_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Demote<2>, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint32_t>::Demote<3>, uint8_t>::value));

    static_assert((std::is_same<PromoteType<uint64_t>::Lowest, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Previous, uint32_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Next, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Highest, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Promote<0>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Promote<1>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Demote<0>, uint64_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Demote<1>, uint32_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Demote<2>, uint16_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Demote<3>, uint8_t>::value));
    static_assert((std::is_same<PromoteType<uint64_t>::Demote<4>, uint8_t>::value));

    // Math types
#define CHECK_PROMOTED_FLOAT_MATH_TYPE(MATH_TYPE) \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Lowest, math::MATH_TYPE<math::half>>::value));    \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Previous, math::MATH_TYPE<math::half>>::value));  \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Next, math::MATH_TYPE<double>>::value));          \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Highest, math::MATH_TYPE<double>>::value));       \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Promote<0>, math::MATH_TYPE<float>>::value));     \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Promote<1>, math::MATH_TYPE<double>>::value));    \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Promote<2>, math::MATH_TYPE<double>>::value));    \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Demote<0>, math::MATH_TYPE<float>>::value));      \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Demote<1>, math::MATH_TYPE<math::half>>::value)); \
    static_assert((std::is_same<PromoteType<MATH_TYPE##s>::Demote<2>, math::MATH_TYPE<math::half>>::value));

    CHECK_PROMOTED_FLOAT_MATH_TYPE(Quat)
    CHECK_PROMOTED_FLOAT_MATH_TYPE(Vec2)
    CHECK_PROMOTED_FLOAT_MATH_TYPE(Vec3)
    CHECK_PROMOTED_FLOAT_MATH_TYPE(Vec4)

    CHECK_PROMOTED_FLOAT_MATH_TYPE(Mat3)
    CHECK_PROMOTED_FLOAT_MATH_TYPE(Mat4)

#define CHECK_PROMOTED_DOUBLE_MATH_TYPE(MATH_TYPE) \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Lowest, math::MATH_TYPE<math::half>>::value));    \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Previous, math::MATH_TYPE<float>>::value));       \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Next, math::MATH_TYPE<double>>::value));          \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Highest, math::MATH_TYPE<double>>::value));       \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Promote<0>, math::MATH_TYPE<double>>::value));    \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Promote<1>, math::MATH_TYPE<double>>::value));    \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Demote<0>, math::MATH_TYPE<double>>::value));     \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Demote<1>, math::MATH_TYPE<float>>::value));      \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Demote<2>, math::MATH_TYPE<math::half>>::value)); \
    static_assert((std::is_same<PromoteType<MATH_TYPE##d>::Demote<3>, math::MATH_TYPE<math::half>>::value));

    CHECK_PROMOTED_DOUBLE_MATH_TYPE(Quat)
    CHECK_PROMOTED_DOUBLE_MATH_TYPE(Vec2)
    CHECK_PROMOTED_DOUBLE_MATH_TYPE(Vec3)
    CHECK_PROMOTED_DOUBLE_MATH_TYPE(Vec4)

    CHECK_PROMOTED_DOUBLE_MATH_TYPE(Mat3)
    CHECK_PROMOTED_DOUBLE_MATH_TYPE(Mat4)
}
