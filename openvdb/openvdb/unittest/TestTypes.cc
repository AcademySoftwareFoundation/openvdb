// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>

#include <gtest/gtest.h>

#include <functional> // for std::ref()
#include <string>


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


////////////////////////////////////////


namespace {

template<typename T> char typeCode() { return '.'; }
template<> char typeCode<bool>()   { return 'b'; }
template<> char typeCode<char>()   { return 'c'; }
template<> char typeCode<double>() { return 'd'; }
template<> char typeCode<float>()  { return 'f'; }
template<> char typeCode<int>()    { return 'i'; }
template<> char typeCode<long>()   { return 'l'; }


struct TypeCodeOp
{
    std::string codes;
    template<typename T> void operator()(const T&) { codes.push_back(typeCode<T>()); }
};


template<typename TSet>
inline std::string
typeSetAsString()
{
    TypeCodeOp op;
    TSet::foreach(std::ref(op));
    return op.codes;
}

} // anonymous namespace


TEST_F(TestTypes, testTypeList)
{
    using T0 = TypeList<>;
    EXPECT_EQ(std::string(), typeSetAsString<T0>());

    using T1 = TypeList<int>;
    EXPECT_EQ(std::string("i"), typeSetAsString<T1>());

    using T2 = TypeList<float>;
    EXPECT_EQ(std::string("f"), typeSetAsString<T2>());

    using T3 = TypeList<bool, double>;
    EXPECT_EQ(std::string("bd"), typeSetAsString<T3>());

    using T4 = T1::Append<T2>;
    EXPECT_EQ(std::string("if"), typeSetAsString<T4>());
    EXPECT_EQ(std::string("fi"), typeSetAsString<T2::Append<T1>>());

    using T5 = T3::Append<T4>;
    EXPECT_EQ(std::string("bdif"), typeSetAsString<T5>());

    using T6 = T5::Append<T5>;
    EXPECT_EQ(std::string("bdifbdif"), typeSetAsString<T6>());

    using T7 = T5::Append<char, long>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T7>());

    using T8 = T5::Append<char>::Append<long>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T8>());

    using T9 = T8::Remove<TypeList<>>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T9>());

    using T10 = T8::Remove<std::string>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T10>());

    using T11 = T8::Remove<char>::Remove<int>;
    EXPECT_EQ(std::string("bdfl"), typeSetAsString<T11>());

    using T12 = T8::Remove<char, int>;
    EXPECT_EQ(std::string("bdfl"), typeSetAsString<T12>());

    using T13 = T8::Remove<TypeList<char, int>>;
    EXPECT_EQ(std::string("bdfl"), typeSetAsString<T13>());

    /// Compile time tests of TypeList
    /// @note  static_assert with no message requires C++17

    using IntTypes = TypeList<Int16, Int32, Int64>;
    using EmptyList = TypeList<>;

    // Size
    static_assert(IntTypes::Size == 3, "");
    static_assert(EmptyList::Size == 0, "");

    // Contains
    static_assert(IntTypes::Contains<Int16>, "");
    static_assert(IntTypes::Contains<Int32>, "");
    static_assert(IntTypes::Contains<Int64>, "");
    static_assert(!IntTypes::Contains<float>, "");

    // Index
    static_assert(IntTypes::Index<Int16> == 0, "");
    static_assert(IntTypes::Index<Int32> == 1, "");
    static_assert(IntTypes::Index<Int64> == 2, "");
    static_assert(IntTypes::Index<float> == -1, "");

    // Get
    static_assert(std::is_same<IntTypes::Get<0>, Int16>::value, "");
    static_assert(std::is_same<IntTypes::Get<1>, Int32>::value, "");
    static_assert(std::is_same<IntTypes::Get<2>, Int64>::value, "");
    static_assert(std::is_same<IntTypes::Get<3>,  typelist_internal::NullType>::value, "");
    static_assert(!std::is_same<IntTypes::Get<3>, void>::value, "");

    // Unique
    static_assert(std::is_same<IntTypes::Unique<>, IntTypes>::value, "");
    static_assert(std::is_same<EmptyList::Unique<>, EmptyList>::value, "");

    // Front/Back
    static_assert(std::is_same<IntTypes::Front, Int16>::value, "");
    static_assert(std::is_same<IntTypes::Back, Int64>::value, "");

    // PopFront/PopBack
    static_assert(std::is_same<IntTypes::PopFront, TypeList<Int32, Int64>>::value, "");
    static_assert(std::is_same<IntTypes::PopBack, TypeList<Int16, Int32>>::value, "");

    // RemoveByIndex
    static_assert(std::is_same<IntTypes::RemoveByIndex<0,0>, IntTypes::PopFront>::value, "");
    static_assert(std::is_same<IntTypes::RemoveByIndex<2,2>, IntTypes::PopBack>::value, "");
    static_assert(std::is_same<IntTypes::RemoveByIndex<0,2>, EmptyList>::value, "");
    static_assert(std::is_same<IntTypes::RemoveByIndex<1,2>, TypeList<Int16>>::value, "");
    static_assert(std::is_same<IntTypes::RemoveByIndex<1,1>, TypeList<Int16, Int64>>::value, "");
    static_assert(std::is_same<IntTypes::RemoveByIndex<0,1>, TypeList<Int64>>::value, "");
    static_assert(std::is_same<IntTypes::RemoveByIndex<0,10>, EmptyList>::value, "");

    // invalid indices do nothing
    static_assert(std::is_same<IntTypes::RemoveByIndex<2,1>, IntTypes>::value, "");
    static_assert(std::is_same<IntTypes::RemoveByIndex<3,3>, IntTypes>::value, "");

    //

    // Test methods on an empty list
    static_assert(!EmptyList::Contains<Int16>, "");
    static_assert(EmptyList::Index<Int16> == -1, "");
    static_assert(std::is_same<EmptyList::Get<0>, typelist_internal::NullType>::value, "");
    static_assert(std::is_same<EmptyList::Front, typelist_internal::NullType>::value, "");
    static_assert(std::is_same<EmptyList::Back, typelist_internal::NullType>::value, "");
    static_assert(std::is_same<EmptyList::PopFront, EmptyList>::value, "");
    static_assert(std::is_same<EmptyList::PopBack, EmptyList>::value, "");
    static_assert(std::is_same<EmptyList::RemoveByIndex<0,0>, EmptyList>::value, "");

    //

    // Test some methods on lists with duplicate types
    using DuplicateIntTypes = TypeList<Int32, Int16, Int64, Int16>;
    using DuplicateRealTypes = TypeList<float, float, float, float>;
    static_assert(DuplicateIntTypes::Size == 4, "");
    static_assert(DuplicateRealTypes::Size == 4, "");
    static_assert(DuplicateIntTypes::Index<Int16> == 1, "");
    static_assert(std::is_same<DuplicateIntTypes::Unique<>, TypeList<Int32, Int16, Int64>>::value, "");
    static_assert(std::is_same<DuplicateRealTypes::Unique<>, TypeList<float>>::value, "");

    //

    // Tests on VDB grid node chains - reverse node chains from leaf->root
    using Tree4Float = openvdb::tree::Tree4<float, 5, 4, 3>::Type; // usually the same as FloatTree
    using NodeChainT = Tree4Float::RootNodeType::NodeChainType;

    // Expected types
    using LeafT = openvdb::tree::LeafNode<float, 3>;
    using IternalT1 = openvdb::tree::InternalNode<LeafT, 4>;
    using IternalT2 = openvdb::tree::InternalNode<IternalT1, 5>;
    using RootT = openvdb::tree::RootNode<IternalT2>;

    static_assert(std::is_same<NodeChainT::Get<0>, LeafT>::value, "");
    static_assert(std::is_same<NodeChainT::Get<1>, IternalT1>::value, "");
    static_assert(std::is_same<NodeChainT::Get<2>, IternalT2>::value, "");
    static_assert(std::is_same<NodeChainT::Get<3>, RootT>::value, "");
    static_assert(std::is_same<NodeChainT::Get<4>, typelist_internal::NullType>::value, "");
}
