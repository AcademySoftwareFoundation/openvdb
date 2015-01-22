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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb_points/tools/AttributeArray.h>
#include <openvdb/Types.h>

#include <iostream>
#include <sstream>

class TestAttributeArray: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAttributeArray);
    CPPUNIT_TEST(testAttributeArray);
    CPPUNIT_TEST(testAttributeSetDescriptor);
    CPPUNIT_TEST(testAttributeSet);

    CPPUNIT_TEST_SUITE_END();

    void testAttributeArray();
    void testAttributeSetDescriptor();
    void testAttributeSet();
}; // class TestPointDataGrid

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeArray);


////////////////////////////////////////


namespace {

bool
matchingAttributeSets(const openvdb::tools::AttributeSet& lhs,
    const openvdb::tools::AttributeSet& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    if (lhs.memUsage() != rhs.memUsage()) return false;
    //if (lhs.descriptor() != rhs.descriptor()) return false;

    typedef openvdb::tools::AttributeArray AttributeArray;

    for (size_t n = 0, N = lhs.size(); n < N; ++n) {

        const AttributeArray* a = lhs.getConst(n);
        const AttributeArray* b = rhs.getConst(n);

        if (a->size() != b->size()) return false;
        if (a->isUniform() != b->isUniform()) return false;
        if (a->isCompressed() != b->isCompressed()) return false;
        if (a->isHidden() != b->isHidden()) return false;
        if (a->type() != b->type()) return false;
    }

    return true;
}

bool
attributeSetMatchesDescriptor(  const openvdb::tools::AttributeSet& attrSet,
                                const openvdb::tools::AttributeSet::Descriptor& descriptor)
{
    if (descriptor.size() != attrSet.size())    return false;

    // ensure descriptor and attributes are still in sync

    for (openvdb::tools::AttributeSet::Descriptor::ConstIterator  it = attrSet.descriptor().map().begin(),
                                    itEnd = attrSet.descriptor().map().end(); it != itEnd; ++it)
    {
        const size_t pos = descriptor.find(it->first);

        if (pos != size_t(it->second))  return false;
        if (descriptor.type(pos) != attrSet.get(pos)->type())   return false;
    }

    return true;
}

} //unnamed  namespace


////////////////////////////////////////


void
TestAttributeArray::testAttributeArray()
{
    typedef openvdb::tools::TypedAttributeArray<double> AttributeArrayD;

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

    {
        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayC(50));

        AttributeArrayC& typedAttr = static_cast<AttributeArrayC&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), value, /*tolerance=*/double(0.0001));
    }

    typedef openvdb::tools::TypedAttributeArray<int> AttributeArrayI;

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
        CPPUNIT_ASSERT_EQUAL(attr.type(), attrB.type());
        CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
        CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
        }
    }

    typedef openvdb::tools::FixedPositionAttributeCodec<uint16_t> FixedPositionCodec;
    typedef openvdb::tools::TypedAttributeArray<double, FixedPositionCodec> AttributeArrayP;

    { // Fixed codec range
        openvdb::tools::AttributeArray::Ptr attr1(new AttributeArrayC(50));
        openvdb::tools::AttributeArray::Ptr attr2(new AttributeArrayP(50));

        AttributeArrayC& fixedPoint = static_cast<AttributeArrayC&>(*attr1);
        AttributeArrayP& fixedPosition = static_cast<AttributeArrayP&>(*attr2);

        // fixed point range is 0.0 => 1.0

        fixedPoint.set(0, -0.1);
        fixedPoint.set(1, 0.1);
        fixedPoint.set(2, 0.9);
        fixedPoint.set(3, 1.1);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.1), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.9), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), fixedPoint.get(3), /*tolerance=*/double(0.0001));

        // fixed position range is -0.5 => 0.5

        fixedPosition.set(0, -0.6);
        fixedPosition.set(1, -0.4);
        fixedPosition.set(2, 0.4);
        fixedPosition.set(3, 0.6);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.5), fixedPosition.get(0), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.4), fixedPosition.get(1), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.4), fixedPosition.get(2), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), fixedPosition.get(3), /*tolerance=*/double(0.0001));
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

        CPPUNIT_ASSERT_EQUAL(attrA.type(), attrB.type());
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
}


void
TestAttributeArray::testAttributeSetDescriptor()
{
    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>  AttributeS;
    typedef openvdb::tools::TypedAttributeArray<double> AttributeD;
    typedef openvdb::tools::TypedAttributeArray<int>    AttributeI;

    AttributeS::registerType();
    AttributeD::registerType();
    AttributeI::registerType();

    typedef openvdb::tools::AttributeSet::Descriptor Descriptor;

    Descriptor::Inserter names;
    names.add("density", AttributeS::attributeType());
    names.add("id", AttributeI::attributeType());

    Descriptor::Ptr descrA = Descriptor::create(names.vec);

    Descriptor::Ptr descrB = Descriptor::create(Descriptor::Inserter()
        .add("density", AttributeS::attributeType())
        .add("id", AttributeI::attributeType())
        .vec);

    CPPUNIT_ASSERT_EQUAL(descrA->size(), descrB->size());

    CPPUNIT_ASSERT(*descrA == *descrB);

    // Rebuild NameAndTypeVec

    Descriptor::NameAndTypeVec rebuildNames;
    descrA->appendTo(rebuildNames);

    CPPUNIT_ASSERT_EQUAL(rebuildNames.size(), names.vec.size());

    for (Descriptor::NameAndTypeVec::const_iterator itA = rebuildNames.begin(), itB = names.vec.begin(),
                                                    itEndA = rebuildNames.end(), itEndB = names.vec.end();
                                                    itA != itEndA && itB != itEndB; ++itA, ++itB) {
        CPPUNIT_ASSERT_EQUAL(itA->name, itB->name);
        CPPUNIT_ASSERT_EQUAL(itA->type, itB->type);
    }

    // I/O test

    std::ostringstream ostr(std::ios_base::binary);
    descrA->write(ostr);

    Descriptor inputDescr;

    std::istringstream istr(ostr.str(), std::ios_base::binary);
    inputDescr.read(istr);

    CPPUNIT_ASSERT_EQUAL(descrA->size(), inputDescr.size());
    CPPUNIT_ASSERT(*descrA == inputDescr);
}


void
TestAttributeArray::testAttributeSet()
{
    typedef openvdb::tools::AttributeArray AttributeArray;

    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>          AttributeS;
    typedef openvdb::tools::TypedAttributeArray<int>            AttributeI;
    typedef openvdb::tools::TypedAttributeArray<openvdb::Vec3s> AttributeVec3s;

    AttributeS::registerType();
    AttributeI::registerType();
    AttributeVec3s::registerType();

    typedef openvdb::tools::AttributeSet AttributeSet;
    typedef openvdb::tools::AttributeSet::Descriptor Descriptor;

    // construct

    Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
        .add("pos", AttributeVec3s::attributeType())
        .add("id", AttributeI::attributeType())
        .vec);

    AttributeSet attrSetA(descr, /*arrayLength=*/50);

    // check equality against duplicate array

    Descriptor::Ptr descr2 = Descriptor::create(Descriptor::Inserter()
        .add("pos", AttributeVec3s::attributeType())
        .add("id", AttributeI::attributeType())
        .vec);

    AttributeSet attrSetA2(descr2, /*arrayLength=*/50);

    CPPUNIT_ASSERT(attrSetA == attrSetA2);

    // expand uniform values and check equality

    attrSetA.get("pos")->expand();
    attrSetA2.get("pos")->expand();

    CPPUNIT_ASSERT(attrSetA == attrSetA2);

    CPPUNIT_ASSERT_EQUAL(size_t(2), attrSetA.size());
    CPPUNIT_ASSERT_EQUAL(size_t(50), attrSetA.get(0)->size());
    CPPUNIT_ASSERT_EQUAL(size_t(50), attrSetA.get(1)->size());

    { // copy
        CPPUNIT_ASSERT(!attrSetA.isShared(0));
        CPPUNIT_ASSERT(!attrSetA.isShared(1));

        AttributeSet attrSetB(attrSetA);

        CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));

        CPPUNIT_ASSERT(attrSetA.isShared(0));
        CPPUNIT_ASSERT(attrSetA.isShared(1));
        CPPUNIT_ASSERT(attrSetB.isShared(0));
        CPPUNIT_ASSERT(attrSetB.isShared(1));

        attrSetB.makeUnique(0);
        attrSetB.makeUnique(1);

        CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));

        CPPUNIT_ASSERT(!attrSetA.isShared(0));
        CPPUNIT_ASSERT(!attrSetA.isShared(1));
        CPPUNIT_ASSERT(!attrSetB.isShared(0));
        CPPUNIT_ASSERT(!attrSetB.isShared(1));
    }

    { // value copy
        AttributeSet attrSetB(attrSetA);
        AttributeSet attrSetC(attrSetA);

        attrSetB.makeUnique(0);
        attrSetB.makeUnique(1);
        attrSetC.makeUnique(0);
        attrSetC.makeUnique(1);

        AttributeVec3s* attrB0 = static_cast<AttributeVec3s*>(attrSetB.get(0));
        AttributeI* attrB1 = static_cast<AttributeI*>(attrSetB.get(1));

        // assign arbitrary values to pos and id

        for (openvdb::Index i = 0; i < 50; i++) {
            attrB0->set(i, openvdb::Vec3s(i, i+50, i+100));
            attrB1->set(i, int(i));
        }

        // copy attribute values from attrSetB to attrSetC with an offset of 10 modulo

        for (openvdb::Index i = 0; i < 50; i++) {
            openvdb::Index offset = (i + 10) % 50;
            attrSetC.copyAttributeValues(i, attrSetB, offset);
        }

        // verify attribute set arrays match only when using the correct offset

        AttributeVec3s* attrC0 = static_cast<AttributeVec3s*>(attrSetC.get(0));
        AttributeI* attrC1 = static_cast<AttributeI*>(attrSetC.get(1));

        for (openvdb::Index i = 0; i < 50; i++) {
            openvdb::Index offset = (i + 10) % 50;

            CPPUNIT_ASSERT(attrC0->get(i) != attrB0->get(i));
            CPPUNIT_ASSERT(attrC1->get(i) != attrB1->get(i));

            CPPUNIT_ASSERT_EQUAL(attrC0->get(i), attrB0->get(offset));
            CPPUNIT_ASSERT_EQUAL(attrC1->get(i), attrB1->get(offset));
        }

        // ensure an attempt to copy values for attribute sets with mis-matching descriptors throws

        Descriptor::Ptr descrD = Descriptor::create(Descriptor::Inserter()
        .add("pos", AttributeVec3s::attributeType())
        .vec);

        AttributeSet attrSetD(descrD, /*arrayLength=*/50);

        CPPUNIT_ASSERT_THROW(attrSetD.copyAttributeValues(0, attrSetA, 0), openvdb::LookupError);
    }

    { // attribute insertion
        AttributeSet attrSetB(attrSetA);

        attrSetB.makeUnique(0);
        attrSetB.makeUnique(1);

        Descriptor::NameAndTypeVec newAttributes;
        newAttributes.push_back(Descriptor::NameAndType("test", AttributeS::attributeType()));

        Descriptor::Ptr targetDescr = Descriptor::create(Descriptor::Inserter()
            .add("pos", AttributeVec3s::attributeType())
            .add("id", AttributeI::attributeType())
            .add("test", AttributeS::attributeType())
            .vec);

        Descriptor::Ptr descrB = attrSetB.descriptor().duplicateAppend(newAttributes);

        // ensure attribute order persists

        CPPUNIT_ASSERT_EQUAL(descrB->find("pos"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(descrB->find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(descrB->find("test"), size_t(2));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttributes(newAttributes);

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *descrB));
        }
        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttributes(newAttributes, attrSetC.descriptor(), descrB);

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }
    }

    { // attribute removal

        Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
            .add("pos", AttributeVec3s::attributeType())
            .add("test", AttributeI::attributeType())
            .add("id", AttributeI::attributeType())
            .add("test2", AttributeI::attributeType())
            .vec);

        Descriptor::Ptr targetDescr = Descriptor::create(Descriptor::Inserter()
            .add("pos", AttributeVec3s::attributeType())
            .add("id", AttributeI::attributeType())
            .vec);

        AttributeSet attrSetB(descr, /*arrayLength=*/50);

        std::vector<size_t> toDrop;
        toDrop.push_back(descr->find("test"));
        toDrop.push_back(descr->find("test2"));

        CPPUNIT_ASSERT_EQUAL(toDrop[0], size_t(1));
        CPPUNIT_ASSERT_EQUAL(toDrop[1], size_t(3));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            attrSetC.dropAttributes(toDrop);

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(2));

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            Descriptor::Ptr descrB = attrSetB.descriptor().duplicateDrop(toDrop);

            attrSetC.dropAttributes(toDrop, attrSetC.descriptor(), descrB);

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(2));

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }
    }

    // replace existing arrays

    // this replace call should not take effect since the new attribute
    // array type does not match with the descriptor type for the given position.
    AttributeArray::Ptr floatAttr(new AttributeS(15));
    CPPUNIT_ASSERT(attrSetA.replace(1, floatAttr) == AttributeSet::INVALID_POS);

    AttributeArray::Ptr intAttr(new AttributeI(10));
    CPPUNIT_ASSERT(attrSetA.replace(1, intAttr) != AttributeSet::INVALID_POS);

    CPPUNIT_ASSERT_EQUAL(size_t(10), attrSetA.get(1)->size());

    // I/O test

    std::ostringstream ostr(std::ios_base::binary);
    attrSetA.write(ostr);

    AttributeSet attrSetB;
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    attrSetB.read(istr);

    CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));
}


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
