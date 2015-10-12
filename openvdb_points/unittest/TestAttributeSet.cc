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
#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb/Types.h>

#include <iostream>
#include <sstream>

class TestAttributeSet: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAttributeSet);
    CPPUNIT_TEST(testAttributeSetDescriptor);
    CPPUNIT_TEST(testAttributeSet);

    CPPUNIT_TEST_SUITE_END();

    void testAttributeSetDescriptor();
    void testAttributeSet();
}; // class TestPointDataGrid

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeSet);


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
TestAttributeSet::testAttributeSetDescriptor()
{
    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>  AttributeS;
    typedef openvdb::tools::TypedAttributeArray<double> AttributeD;
    typedef openvdb::tools::TypedAttributeArray<int32_t>    AttributeI;

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
        CPPUNIT_ASSERT_EQUAL(itA->type.first, itB->type.first);
        CPPUNIT_ASSERT_EQUAL(itA->type.second, itB->type.second);
    }

     // Test hasSameAttributes
    {
        Descriptor::Ptr descr1 = Descriptor::create(Descriptor::Inserter()
                .add("pos", AttributeD::attributeType())
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec);

        // Test same names with different types, should be false
        Descriptor::Ptr descr2 = Descriptor::create(Descriptor::Inserter()
                .add("pos", AttributeD::attributeType())
                .add("test", AttributeS::attributeType())
                .add("id", AttributeI::attributeType())
                .vec);

        CPPUNIT_ASSERT(!descr1->hasSameAttributes(*descr2));

        // Test different names, should be false
        Descriptor::Ptr descr3 = Descriptor::create(Descriptor::Inserter()
                .add("pos", AttributeD::attributeType())
                .add("test2", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec);

        CPPUNIT_ASSERT(!descr1->hasSameAttributes(*descr3));

        // Test same names and types but different order, should be true
        Descriptor::Ptr descr4 = Descriptor::create(Descriptor::Inserter()
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .add("pos", AttributeD::attributeType())
                .vec);

        CPPUNIT_ASSERT(descr1->hasSameAttributes(*descr4));
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
TestAttributeSet::testAttributeSet()
{
    typedef openvdb::tools::AttributeArray AttributeArray;

    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>          AttributeS;
    typedef openvdb::tools::TypedAttributeArray<int32_t>        AttributeI;
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

            attrSetC.appendAttribute(newAttributes[0]);

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *descrB));
        }
        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttribute(newAttributes[0], attrSetC.descriptor(), descrB);

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

    { // reorder attribute set
        Descriptor::Ptr descr1 = Descriptor::create(Descriptor::Inserter()
                .add("pos", AttributeVec3s::attributeType())
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .add("test2", AttributeI::attributeType())
                .vec);

        Descriptor::Ptr descr2 = Descriptor::create(Descriptor::Inserter()
                .add("test2", AttributeI::attributeType())
                .add("test", AttributeI::attributeType())
                .add("pos", AttributeVec3s::attributeType())
                .add("id", AttributeI::attributeType())
                .vec);

        AttributeSet attrSetA(descr1);
        AttributeSet attrSetB(descr2);

        CPPUNIT_ASSERT(attrSetA != attrSetB);

        attrSetB.reorderAttributes(descr1);

        CPPUNIT_ASSERT(attrSetA == attrSetB);
    }

    // I/O test

    std::ostringstream ostr(std::ios_base::binary);
    attrSetA.write(ostr);

    AttributeSet attrSetB;
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    attrSetB.read(istr);

    CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));
}

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
