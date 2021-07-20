// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/codegen/Types.h>
#include <openvdb_ax/codegen/String.h>

#include <openvdb/math/Vec2.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/Vec4.h>
#include <openvdb/math/Mat3.h>
#include <openvdb/math/Mat4.h>

#include <cppunit/extensions/HelperMacros.h>

class TestTypes : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestTypes);
    CPPUNIT_TEST(testTypes);
    CPPUNIT_TEST(testVDBTypes);
    CPPUNIT_TEST(testString);
    CPPUNIT_TEST_SUITE_END();

    void testTypes();
    void testVDBTypes();
    void testString();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTypes);

void
TestTypes::testTypes()
{
    using openvdb::ax::codegen::LLVMType;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

    // scalar types

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getInt1Ty(C)), LLVMType<bool>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getInt8Ty(C)), LLVMType<int8_t>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getInt16Ty(C)), LLVMType<int16_t>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getInt32Ty(C)), LLVMType<int32_t>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getInt64Ty(C)), LLVMType<int64_t>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getFloatTy(C)), LLVMType<float>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getDoubleTy(C)), LLVMType<double>::get(C));

    // scalar values

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantInt::get(llvm::Type::getInt1Ty(C), true)),
        LLVMType<bool>::get(C, true));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantInt::get(llvm::Type::getInt8Ty(C), int8_t(1))),
        LLVMType<int8_t>::get(C, int8_t(1)));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantInt::get(llvm::Type::getInt16Ty(C), int16_t(2))),
        LLVMType<int16_t>::get(C, int16_t(2)));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), int32_t(3))),
        LLVMType<int32_t>::get(C, int32_t(3)));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantInt::get(llvm::Type::getInt64Ty(C), int64_t(4))),
        LLVMType<int64_t>::get(C, int64_t(4)));

    // array types

#if LLVM_VERSION_MAJOR > 6
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt1Ty(C), 1)),
        LLVMType<bool[1]>::get(C));
#endif

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt8Ty(C), 2)),
        LLVMType<int8_t[2]>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt16Ty(C), 3)),
        LLVMType<int16_t[3]>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)),
        LLVMType<int32_t[4]>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt64Ty(C), 5)),
        LLVMType<int64_t[5]>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 6)),
        LLVMType<float[6]>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 7)),
        LLVMType<double[7]>::get(C));

    // array values

#if LLVM_VERSION_MAJOR > 6
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantDataArray::get<bool>(C, {true})),
        LLVMType<bool[1]>::get(C, {true}));
#endif

    const std::vector<uint8_t> veci8{1,2};
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantDataArray::get(C, veci8)),
        LLVMType<uint8_t[2]>::get(C, {1,2}));

    const std::vector<uint16_t> veci16{1,2,3};
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantDataArray::get(C, veci16)),
        LLVMType<uint16_t[3]>::get(C, {1,2,3}));

    const std::vector<uint32_t> veci32{1,2,3,4};
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantDataArray::get(C, veci32)),
        LLVMType<uint32_t[4]>::get(C, {1,2,3,4}));

    const std::vector<uint64_t> veci64{1,2,3,4,5};
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantDataArray::get(C, veci64)),
        LLVMType<uint64_t[5]>::get(C, {1,2,3,4,5}));

    const std::vector<float> vecf{.0f,.1f,.2f,.3f,.4f,.5f};
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantDataArray::get(C, vecf)),
        LLVMType<float[6]>::get(C, {.0f,.1f,.2f,.3f,.4f,.5f}));

    const std::vector<double> vecd{.0,.1,.2,.3,.4,.5,.6};
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Constant>(llvm::ConstantDataArray::get(C, vecd)),
        LLVMType<double[7]>::get(C, {.0,.1,.2,.3,.4,.5,.6}));

    // void
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getVoidTy(C)), LLVMType<void>::get(C));
    // some special cases we alias
    CPPUNIT_ASSERT_EQUAL(llvm::Type::getInt8PtrTy(C), LLVMType<void*>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::Type::getInt8Ty(C)), LLVMType<char>::get(C));
}

void
TestTypes::testVDBTypes()
{
    using openvdb::ax::codegen::LLVMType;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)),
        LLVMType<openvdb::math::Vec2<int32_t>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2)),
        LLVMType<openvdb::math::Vec2<float>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)),
        LLVMType<openvdb::math::Vec2<double>>::get(C));

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)),
        LLVMType<openvdb::math::Vec3<int32_t>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3)),
        LLVMType<openvdb::math::Vec3<float>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3)),
        LLVMType<openvdb::math::Vec3<double>>::get(C));

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)),
        LLVMType<openvdb::math::Vec4<int32_t>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4)),
        LLVMType<openvdb::math::Vec4<float>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4)),
        LLVMType<openvdb::math::Vec4<double>>::get(C));

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9)),
        LLVMType<openvdb::math::Mat3<float>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)),
        LLVMType<openvdb::math::Mat3<double>>::get(C));

    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16)),
        LLVMType<openvdb::math::Mat4<float>>::get(C));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Type>(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16)),
        LLVMType<openvdb::math::Mat4<double>>::get(C));
}

void
TestTypes::testString()
{
    using openvdb::ax::codegen::LLVMType;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

    llvm::Type* type = LLVMType<openvdb::ax::codegen::String>::get(C);
    CPPUNIT_ASSERT(type->isAggregateType());
    CPPUNIT_ASSERT_EQUAL(llvm::Type::StructTyID, type->getTypeID());
    CPPUNIT_ASSERT_EQUAL(unsigned(3), type->getNumContainedTypes()); // char*, SSO, len
    CPPUNIT_ASSERT_EQUAL(unsigned(3), type->getStructNumElements()); // char*, SSO, len

    // Check members
    CPPUNIT_ASSERT_EQUAL((llvm::Type*)LLVMType<char*>::get(C), type->getContainedType(0));
    CPPUNIT_ASSERT_EQUAL(LLVMType<char[openvdb::ax::codegen::String::SSO_LENGTH]>::get(C), type->getContainedType(1));
    CPPUNIT_ASSERT_EQUAL(LLVMType<int64_t>::get(C), type->getContainedType(2));
}
