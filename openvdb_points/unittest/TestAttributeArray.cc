///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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


#include <cppunit/extensions/HelperMacros.h>
#include <openvdb_points/tools/AttributeArray.h>
#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>

#include "ProfileTimer.h"

#include <sstream>
#include <iostream>

using namespace openvdb;
using namespace openvdb::tools;

class TestAttributeArray: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAttributeArray);
    CPPUNIT_TEST(testFixedPointConversion);
    CPPUNIT_TEST(testAttributeArray);
    CPPUNIT_TEST(testAttributeHandle);
    CPPUNIT_TEST(testProfile);

    CPPUNIT_TEST_SUITE_END();

    void testFixedPointConversion();
    void testAttributeArray();
    void testAttributeHandle();
    void testProfile();
}; // class TestAttributeArray

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeArray);


////////////////////////////////////////


namespace {

bool
matchingNamePairs(const openvdb::NamePair& lhs,
                  const openvdb::NamePair& rhs)
{
    if (lhs.first != rhs.first)     return false;
    if (lhs.second != rhs.second)     return false;

    return true;
}

} // namespace


////////////////////////////////////////

void
TestAttributeArray::testFixedPointConversion()
{
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(/*voxelSize=*/0.1));

    const float value = 33.5688040469035f;

    // convert to fixed-point value

    const openvdb::Vec3f worldSpaceValue(value);
    const openvdb::Vec3f indexSpaceValue = transform->worldToIndex(worldSpaceValue);
    const float voxelSpaceValue = indexSpaceValue.x() - math::Round(indexSpaceValue.x()) + 0.5f;
    const uint32_t intValue = floatingPointToFixedPoint<uint32_t>(voxelSpaceValue);

    // convert back to floating-point value

    const float newVoxelSpaceValue = fixedPointToFloatingPoint<float>(intValue);
    const openvdb::Vec3f newIndexSpaceValue(newVoxelSpaceValue + math::Round(indexSpaceValue.x()) - 0.5f);
    const openvdb::Vec3f newWorldSpaceValue = transform->indexToWorld(newIndexSpaceValue);

    const float newValue = newWorldSpaceValue.x();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(value, newValue, /*tolerance=*/1e-6);
}

void
TestAttributeArray::testAttributeArray()
{
    typedef openvdb::tools::TypedAttributeArray<float> AttributeArrayF;
    typedef openvdb::tools::TypedAttributeArray<double> AttributeArrayD;

    {
        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(attr->size(), size_t(50));
    }

    {
        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(size_t(50), attr->size());

        AttributeArrayD& typedAttr = static_cast<AttributeArrayD&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), value, /*tolerance=*/double(0.0));
    }

    typedef openvdb::tools::FixedPointAttributeCodec<uint16_t> FixedPointCodec;
    typedef openvdb::tools::TypedAttributeArray<double, FixedPointCodec> AttributeArrayC;

    { // test hasValueType()
        openvdb::tools::AttributeArray::Ptr attrC(new AttributeArrayC(50));
        openvdb::tools::AttributeArray::Ptr attrD(new AttributeArrayD(50));
        openvdb::tools::AttributeArray::Ptr attrF(new AttributeArrayF(50));

        CPPUNIT_ASSERT(attrD->hasValueType<double>());
        CPPUNIT_ASSERT(attrC->hasValueType<double>());
        CPPUNIT_ASSERT(!attrF->hasValueType<double>());

        CPPUNIT_ASSERT(!attrD->hasValueType<float>());
        CPPUNIT_ASSERT(!attrC->hasValueType<float>());
        CPPUNIT_ASSERT(attrF->hasValueType<float>());
    }

    {
        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayC(50));

        AttributeArrayC& typedAttr = static_cast<AttributeArrayC&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), value, /*tolerance=*/double(0.0001));
    }

    typedef openvdb::tools::TypedAttributeArray<int32_t> AttributeArrayI;

    { // Base class API

        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayI(50));

        CPPUNIT_ASSERT_EQUAL(size_t(50), attr->size());

        CPPUNIT_ASSERT_EQUAL((sizeof(AttributeArrayI) + sizeof(int)), attr->memUsage());

        CPPUNIT_ASSERT(attr->isType<AttributeArrayI>());
        CPPUNIT_ASSERT(!attr->isType<AttributeArrayD>());

        CPPUNIT_ASSERT(*attr == *attr);
    }

    { // Typed class API

        const size_t count = 50;
        const size_t uniformMemUsage = sizeof(AttributeArrayI) + sizeof(int);
        const size_t expandedMemUsage = sizeof(AttributeArrayI) + count * sizeof(int);

        AttributeArrayI attr(count);

        CPPUNIT_ASSERT_EQUAL(attr.get(0), 0);
        CPPUNIT_ASSERT_EQUAL(attr.get(10), 0);

        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        attr.set(0, 10);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        AttributeArrayI attr2(count);
        attr2.set(0, 10);

        CPPUNIT_ASSERT(attr == attr2);

        attr.collapse(5);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        CPPUNIT_ASSERT_EQUAL(attr.get(0), 5);
        CPPUNIT_ASSERT_EQUAL(attr.get(20), 5);

        attr.expand();
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), 5);
        }

        attr.fill(10);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), 10);
        }

        attr.collapse(7);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        CPPUNIT_ASSERT_EQUAL(attr.get(0), 7);
        CPPUNIT_ASSERT_EQUAL(attr.get(20), 7);

        attr.fill(5);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), 5);
        }

        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setTransient(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setHidden(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());

        attr.setTransient(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());

        AttributeArrayI attrB(attr);
        CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
        CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
        }
    }

    typedef openvdb::tools::FixedPointAttributeCodec<uint16_t> FixedPointCodec;

    { // Fixed codec range
        openvdb::tools::AttributeArray::Ptr attr1(new AttributeArrayC(50));

        AttributeArrayC& fixedPoint = static_cast<AttributeArrayC&>(*attr1);

        // fixed point range is -0.5 => 0.5

        fixedPoint.set(0, -0.6);
        fixedPoint.set(1, -0.4);
        fixedPoint.set(2, 0.4);
        fixedPoint.set(3, 0.6);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.5), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.4), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.4), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    { // IO
        const size_t count = 50;
        AttributeArrayI attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        attrA.setHidden(true);

        std::ostringstream ostr(std::ios_base::binary);
        attrA.write(ostr);

        AttributeArrayI attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        CPPUNIT_ASSERT(matchingNamePairs(attrA.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attrA.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());
        CPPUNIT_ASSERT_EQUAL(attrA.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attrA.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attrA.isHidden(), attrB.isHidden());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
        }

        AttributeArrayI attrC(count, 3);
        attrC.setTransient(true);

        std::ostringstream ostrC(std::ios_base::binary);
        attrC.write(ostrC);

        CPPUNIT_ASSERT_EQUAL(ostrC.str().size(), size_t(0));
    }

    // Registry
    AttributeArrayI::registerType();

    openvdb::tools::AttributeArray::Ptr attr =
        openvdb::tools::AttributeArray::create(
            AttributeArrayI::attributeType(), 34);

    { // Casting
        using namespace openvdb;
        using namespace openvdb::tools;

        AttributeArray::Ptr array = TypedAttributeArray<float>::create(0);
        CPPUNIT_ASSERT_NO_THROW(TypedAttributeArray<float>::cast(*array));
        CPPUNIT_ASSERT_THROW(TypedAttributeArray<int>::cast(*array), TypeError);

        AttributeArray::ConstPtr constArray = array;
        CPPUNIT_ASSERT_NO_THROW(TypedAttributeArray<float>::cast(*constArray));
        CPPUNIT_ASSERT_THROW(TypedAttributeArray<int>::cast(*constArray), TypeError);
    }
}


void
TestAttributeArray::testAttributeHandle()
{
    using namespace openvdb;
    using namespace openvdb::tools;
    using namespace openvdb::math;

    typedef TypedAttributeArray<int>                                                          AttributeI;
    typedef TypedAttributeArray<float, NullAttributeCodec<half> >                             AttributeFH;
    typedef TypedAttributeArray<Vec3f>                                                        AttributeVec3f;

    typedef AttributeWriteHandle<int> AttributeHandleRWI;

    AttributeI::registerType();
    AttributeFH::registerType();
    AttributeVec3f::registerType();

    // create a Descriptor and AttributeSet

    typedef AttributeSet::Descriptor Descriptor;

    Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
        .add("pos", AttributeVec3f::attributeType())
        .add("truncate", AttributeFH::attributeType())
        .add("int", AttributeI::attributeType())
        .vec);

    unsigned count = 50;
    AttributeSet attrSet(descr, /*arrayLength=*/count);

    // check uniform value implementation

    {
        AttributeArray* array = attrSet.get(2);

        AttributeHandleRWI handle(*array);

        CPPUNIT_ASSERT_EQUAL(handle.get(0), 0);
        CPPUNIT_ASSERT_EQUAL(handle.get(10), 0);

        CPPUNIT_ASSERT(handle.isUniform());

        handle.set(0, 10);
        CPPUNIT_ASSERT(!handle.isUniform());

        handle.collapse(5);
        CPPUNIT_ASSERT(handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(handle.get(0), 5);
        CPPUNIT_ASSERT_EQUAL(handle.get(20), 5);

        handle.expand();
        CPPUNIT_ASSERT(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(handle.get(i), 5);
        }

        handle.fill(10);
        CPPUNIT_ASSERT(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(handle.get(i), 10);
        }

        handle.collapse(7);
        CPPUNIT_ASSERT(handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(handle.get(0), 7);
        CPPUNIT_ASSERT_EQUAL(handle.get(20), 7);

        handle.fill(5);
        CPPUNIT_ASSERT(handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(handle.get(i), 5);
        }

        CPPUNIT_ASSERT(handle.isUniform());
    }

    {
        AttributeArray* array = attrSet.get(0);

        AttributeHandleRWVec3f handle(*array);

        handle.set(5, Vec3f(10));

        CPPUNIT_ASSERT_EQUAL(handle.get(5), Vec3f(10));
    }

    {
        AttributeArray* array = attrSet.get(1);

        array->compress();

        AttributeHandleRWF handle(*array);

        handle.set(6, float(11));

        CPPUNIT_ASSERT_EQUAL(handle.get(6), float(11));

        CPPUNIT_ASSERT(!array->isCompressed());

#ifdef OPENVDB_USE_BLOSC
        array->compress();

        CPPUNIT_ASSERT(array->isCompressed());

        {
            AttributeHandleROF handleRO(*array);

            CPPUNIT_ASSERT(array->isCompressed());

            CPPUNIT_ASSERT_EQUAL(handleRO.get(6), float(11));

            CPPUNIT_ASSERT(array->isCompressed());
        }

        CPPUNIT_ASSERT(array->isCompressed());
#endif
    }

    // check values have been correctly set without using handles

    {
        AttributeVec3f* array = static_cast<AttributeVec3f*>(attrSet.get(0));

        CPPUNIT_ASSERT(array);

        CPPUNIT_ASSERT_EQUAL(array->get(5), Vec3f(10));
    }

    {
        AttributeFH* array = static_cast<AttributeFH*>(attrSet.get(1));

        CPPUNIT_ASSERT(array);

        CPPUNIT_ASSERT_EQUAL(array->get(6), float(11));
    }
}


namespace profile {

typedef openvdb::util::ProfileTimer ProfileTimer;

template <typename AttrT>
void expand(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": expand");
    attr.expand();
}

template <typename AttrT>
void set(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": set");
    for (size_t i = 0; i < attr.size(); i++) {
        attr.set(i, typename AttrT::ValueType(i));
    }
}

template <typename AttrT>
void setH(const Name& prefix, AttrT& attr)
{
    typedef typename AttrT::ValueType ValueType;
    ProfileTimer timer(prefix + ": setHandle");
    AttributeWriteHandle<ValueType> handle(attr);
    for (size_t i = 0; i < attr.size(); i++) {
        handle.set(i, ValueType(i));
    }
}

template <typename AttrT>
void sum(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sum");
    typename AttrT::ValueType sum = 0;
    for (size_t i = 0; i < attr.size(); i++) {
        sum += attr.get(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

template <typename AttrT>
void sumH(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sumHandle");
    typedef typename AttrT::ValueType ValueType;
    ValueType sum = 0;
    AttributeHandle<ValueType> handle(attr);
    for (size_t i = 0; i < attr.size(); i++) {
        sum += handle.get(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

template <typename AttrT>
void sumWH(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": sumWriteHandle");
    typedef typename AttrT::ValueType ValueType;
    ValueType sum = 0;
    AttributeWriteHandle<ValueType> handle(attr);
    for (size_t i = 0; i < attr.size(); i++) {
        sum += handle.get(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

} // namespace profile

void
TestAttributeArray::testProfile()
{
    using namespace openvdb;
    using namespace openvdb::util;
    using namespace openvdb::math;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<float>                                      AttributeArrayF;
    typedef TypedAttributeArray<float,
                                FixedPointAttributeCodec<uint16_t> >        AttributeArrayF16;
    typedef TypedAttributeArray<float,
                                FixedPointAttributeCodec<uint8_t> >         AttributeArrayF8;

    ///////////////////////////////////////////////////

#ifdef PROFILE
    const int elements(1000 * 1000 * 1000);

    std::cerr << std::endl;
#else
    const int elements(10 * 1000 * 1000);
#endif

    // std::vector

    {
        std::vector<float> values;
        {
            ProfileTimer timer("Vector<float>: resize");
            values.resize(elements);
        }
        {
            ProfileTimer timer("Vector<float>: set");
            for (int i = 0; i < elements; i++) {
                values[i] = float(i);
            }
        }
        {
            ProfileTimer timer("Vector<float>: sum");
            float sum = 0;
            for (int i = 0; i < elements; i++) {
                sum += values[i];
            }
            // to prevent optimisation clean up
            CPPUNIT_ASSERT(sum);
        }
    }

    // AttributeArray

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeArray<float>", attr);
        profile::set("AttributeArray<float>", attr);
        profile::sum("AttributeArray<float>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeArray<float, fp16>", attr);
        profile::set("AttributeArray<float, fp16>", attr);
        profile::sum("AttributeArray<float, fp16>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeArray<float, fp8>", attr);
        profile::set("AttributeArray<float, fp8>", attr);
        profile::sum("AttributeArray<float, fp8>", attr);
    }

    // AttributeHandle

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeHandle<float>", attr);
        profile::setH("AttributeHandle<float>", attr);
        profile::sumH("AttributeHandle<float>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeHandle<float, fp16>", attr);
        profile::setH("AttributeHandle<float, fp16>", attr);
        profile::sumH("AttributeHandle<float, fp16>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeHandle<float, fp8>", attr);
        profile::setH("AttributeHandle<float, fp8>", attr);
        profile::sumH("AttributeHandle<float, fp8>", attr);
    }
}

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
