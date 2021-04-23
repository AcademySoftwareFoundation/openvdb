// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>

#include <gtest/gtest.h>

class TestExceptions : public ::testing::Test
{
protected:
    template<typename ExceptionT> void testException();
};


template<typename ExceptionT> struct ExceptionTraits
{ static std::string name() { return ""; } };
template<> struct ExceptionTraits<openvdb::ArithmeticError>
{ static std::string name() { return "ArithmeticError"; } };
template<> struct ExceptionTraits<openvdb::IndexError>
{ static std::string name() { return "IndexError"; } };
template<> struct ExceptionTraits<openvdb::IoError>
{ static std::string name() { return "IoError"; } };
template<> struct ExceptionTraits<openvdb::KeyError>
{ static std::string name() { return "KeyError"; } };
template<> struct ExceptionTraits<openvdb::LookupError>
{ static std::string name() { return "LookupError"; } };
template<> struct ExceptionTraits<openvdb::NotImplementedError>
{ static std::string name() { return "NotImplementedError"; } };
template<> struct ExceptionTraits<openvdb::ReferenceError>
{ static std::string name() { return "ReferenceError"; } };
template<> struct ExceptionTraits<openvdb::RuntimeError>
{ static std::string name() { return "RuntimeError"; } };
template<> struct ExceptionTraits<openvdb::TypeError>
{ static std::string name() { return "TypeError"; } };
template<> struct ExceptionTraits<openvdb::ValueError>
{ static std::string name() { return "ValueError"; } };


template<typename ExceptionT>
void
TestExceptions::testException()
{
    std::string ErrorMsg("Error message");

    EXPECT_THROW(OPENVDB_THROW(ExceptionT, ErrorMsg), ExceptionT);

    try {
        OPENVDB_THROW(ExceptionT, ErrorMsg);
    } catch (openvdb::Exception& e) {
        const std::string expectedMsg = ExceptionTraits<ExceptionT>::name() + ": " + ErrorMsg;
        EXPECT_EQ(expectedMsg, std::string(e.what()));
    }
}

TEST_F(TestExceptions, testArithmeticError) { testException<openvdb::ArithmeticError>(); }
TEST_F(TestExceptions, testIndexError) { testException<openvdb::IndexError>(); }
TEST_F(TestExceptions, testIoError) { testException<openvdb::IoError>(); }
TEST_F(TestExceptions, testKeyError) { testException<openvdb::KeyError>(); }
TEST_F(TestExceptions, testLookupError) { testException<openvdb::LookupError>(); }
TEST_F(TestExceptions, testNotImplementedError) { testException<openvdb::NotImplementedError>(); }
TEST_F(TestExceptions, testReferenceError) { testException<openvdb::ReferenceError>(); }
TEST_F(TestExceptions, testRuntimeError) { testException<openvdb::RuntimeError>(); }
TEST_F(TestExceptions, testTypeError) { testException<openvdb::TypeError>(); }
TEST_F(TestExceptions, testValueError) { testException<openvdb::ValueError>(); }
