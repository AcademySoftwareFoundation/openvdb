///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
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
#include <openvdb_points/tools/AttributeGroup.h>
#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb_points/openvdb.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/Metadata.h>

#include <iostream>
#include <sstream>

#include <boost/algorithm/string/predicate.hpp> // boost::startswith

class TestAttributeSet: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestAttributeSet);
    CPPUNIT_TEST(testAttributeSetDescriptor);
    CPPUNIT_TEST(testAttributeSet);
    CPPUNIT_TEST(testAttributeSetGroups);

    CPPUNIT_TEST_SUITE_END();

    void testAttributeSetDescriptor();
    void testAttributeSet();
    void testAttributeSetGroups();
}; // class TestAttributeSet

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeSet);


////////////////////////////////////////


namespace {

bool
matchingAttributeSets(const openvdb::tools::AttributeSet& lhs,
    const openvdb::tools::AttributeSet& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    if (lhs.memUsage() != rhs.memUsage()) return false;
    if (lhs.descriptor() != rhs.descriptor()) return false;

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

    // check default metadata

    const openvdb::MetaMap& meta1 = descriptor.getMetadata();
    const openvdb::MetaMap& meta2 = attrSet.descriptor().getMetadata();

    // build vector of all default keys

    std::vector<openvdb::Name> defaultKeys;

    for (openvdb::MetaMap::ConstMetaIterator    it = meta1.beginMeta(),
                                                itEnd = meta1.endMeta(); it != itEnd; ++it)
    {
        const openvdb::Name name = it->first;

        if (boost::starts_with(name, "default:")) {
            defaultKeys.push_back(name);
        }
    }

    for (openvdb::MetaMap::ConstMetaIterator    it = meta2.beginMeta(),
                                                itEnd = meta2.endMeta(); it != itEnd; ++it)
    {
        const openvdb::Name name = it->first;

        if (boost::starts_with(name, "default:"))
        {
            if (std::find(defaultKeys.begin(), defaultKeys.end(), name) != defaultKeys.end()) {
                defaultKeys.push_back(name);
            }
        }
    }

    // compare metadata value from each metamap

    for (std::vector<openvdb::Name>::const_iterator it = defaultKeys.begin(),
                                                    itEnd = defaultKeys.end(); it != itEnd; ++it) {
        const openvdb::Name name = *it;

        openvdb::Metadata::ConstPtr metaValue1 = meta1[name];
        openvdb::Metadata::ConstPtr metaValue2 = meta2[name];

        if (!metaValue1)    return false;
        if (!metaValue2)    return false;

        if (*metaValue1 != *metaValue2)     return false;
    }

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

bool testStringVector(std::vector<std::string>& input)
{
    return input.size() == 0;
}

bool testStringVector(std::vector<std::string>& input, const std::string& name1)
{
    if (input.size() != 1)  return false;
    if (input[0] != name1)  return false;
    return true;
}

bool testStringVector(std::vector<std::string>& input, const std::string& name1, const std::string& name2)
{
    if (input.size() != 2)  return false;
    if (input[0] != name1)  return false;
    if (input[1] != name2)  return false;
    return true;
}

} //unnamed  namespace


////////////////////////////////////////


void
TestAttributeSet::testAttributeSetDescriptor()
{
    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>      AttributeS;
    typedef openvdb::tools::TypedAttributeArray<double>     AttributeD;
    typedef openvdb::tools::TypedAttributeArray<int32_t>    AttributeI;

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

    descrB->setGroup("test", size_t(0));
    descrB->setGroup("test2", size_t(1));

    Descriptor descrC(*descrB);

    CPPUNIT_ASSERT(descrB->hasSameAttributes(descrC));
    CPPUNIT_ASSERT(descrC.hasGroup("test"));
    CPPUNIT_ASSERT(*descrB == descrC);

    descrC.dropGroup("test");
    descrC.dropGroup("test2");

    CPPUNIT_ASSERT(!descrB->hasSameAttributes(descrC));
    CPPUNIT_ASSERT(!descrC.hasGroup("test"));
    CPPUNIT_ASSERT(*descrB != descrC);

    descrC.setGroup("test2", size_t(1));
    descrC.setGroup("test3", size_t(0));

    CPPUNIT_ASSERT(!descrB->hasSameAttributes(descrC));

    descrC.dropGroup("test3");
    descrC.setGroup("test", size_t(0));

    CPPUNIT_ASSERT(descrB->hasSameAttributes(descrC));

    // rebuild NameAndTypeVec

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

    // hasSameAttributes
    {
        Descriptor::Ptr descr1 = Descriptor::create(Descriptor::Inserter()
                .add("pos", AttributeD::attributeType())
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec);

        // test same names with different types, should be false
        Descriptor::Ptr descr2 = Descriptor::create(Descriptor::Inserter()
                .add("pos", AttributeD::attributeType())
                .add("test", AttributeS::attributeType())
                .add("id", AttributeI::attributeType())
                .vec);

        CPPUNIT_ASSERT(!descr1->hasSameAttributes(*descr2));

        // test different names, should be false
        Descriptor::Ptr descr3 = Descriptor::create(Descriptor::Inserter()
                .add("pos", AttributeD::attributeType())
                .add("test2", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec);

        CPPUNIT_ASSERT(!descr1->hasSameAttributes(*descr3));

        // test same names and types but different order, should be true
        Descriptor::Ptr descr4 = Descriptor::create(Descriptor::Inserter()
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .add("pos", AttributeD::attributeType())
                .vec);

        CPPUNIT_ASSERT(descr1->hasSameAttributes(*descr4));
    }

    { // Test uniqueName
        Descriptor::Inserter names;
        Descriptor::Ptr emptyDescr = Descriptor::create(names.vec);
        const openvdb::Name uniqueNameEmpty = emptyDescr->uniqueName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueNameEmpty, openvdb::Name("test0"));

        names.add("test", AttributeS::attributeType());
        names.add("test1", AttributeI::attributeType());

        Descriptor::Ptr descr1 = Descriptor::create(names.vec);

        const openvdb::Name uniqueName1 = descr1->uniqueName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueName1, openvdb::Name("test0"));

        const Descriptor::NameAndType newAttr(uniqueName1, AttributeI::attributeType());
        Descriptor::Ptr descr2 = descr1->duplicateAppend(newAttr);

        const openvdb::Name uniqueName2 = descr2->uniqueName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueName2, openvdb::Name("test2"));
    }

    { // Test name validity

        CPPUNIT_ASSERT(Descriptor::validName("test1"));
        CPPUNIT_ASSERT(Descriptor::validName("abc_def"));
        CPPUNIT_ASSERT(Descriptor::validName("abc|def"));
        CPPUNIT_ASSERT(Descriptor::validName("abc:def"));

        CPPUNIT_ASSERT(!Descriptor::validName(""));
        CPPUNIT_ASSERT(!Descriptor::validName("test1!"));
        CPPUNIT_ASSERT(!Descriptor::validName("abc=def"));
        CPPUNIT_ASSERT(!Descriptor::validName("abc def"));
        CPPUNIT_ASSERT(!Descriptor::validName("abc*def"));
    }

    { // Test enforcement of valid names
        Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter().add("test1", AttributeS::attributeType()).vec);
        CPPUNIT_ASSERT_THROW(descr->rename("test1", "test1!"), openvdb::RuntimeError);
        CPPUNIT_ASSERT_THROW(descr->setGroup("group1!", 1), openvdb::RuntimeError);

        Descriptor::NameAndType invalidAttr("test1!", AttributeS::attributeType());
        CPPUNIT_ASSERT_THROW(descr->duplicateAppend(invalidAttr), openvdb::RuntimeError);

        Descriptor::Inserter names;
        names.add(invalidAttr);
        CPPUNIT_ASSERT_THROW(Descriptor::create(names.vec), openvdb::RuntimeError);
        CPPUNIT_ASSERT_THROW(descr->duplicateAppend(names.vec), openvdb::RuntimeError);

        const openvdb::Index64 offset(0);
        const openvdb::Index64 zeroLength(0);
        const openvdb::Index64 oneLength(1);

        // write a stream with an invalid attribute
        std::ostringstream attrOstr(std::ios_base::binary);

        attrOstr.write(reinterpret_cast<const char*>(&oneLength), sizeof(openvdb::Index64));
        openvdb::writeString(attrOstr, invalidAttr.type.first);
        openvdb::writeString(attrOstr, invalidAttr.type.second);
        openvdb::writeString(attrOstr, invalidAttr.name);
        attrOstr.write(reinterpret_cast<const char*>(&offset), sizeof(openvdb::Index64));

        attrOstr.write(reinterpret_cast<const char*>(&zeroLength), sizeof(openvdb::Index64));

        // write a stream with an invalid group
        std::ostringstream groupOstr(std::ios_base::binary);

        groupOstr.write(reinterpret_cast<const char*>(&zeroLength), sizeof(openvdb::Index64));

        groupOstr.write(reinterpret_cast<const char*>(&oneLength), sizeof(openvdb::Index64));
        openvdb::writeString(groupOstr, "group1!");
        groupOstr.write(reinterpret_cast<const char*>(&offset), sizeof(openvdb::Index64));

        // read the streams back
        Descriptor inputDescr;
        std::istringstream attrIstr(attrOstr.str(), std::ios_base::binary);
        CPPUNIT_ASSERT_THROW(inputDescr.read(attrIstr), openvdb::IoError);
        std::istringstream groupIstr(groupOstr.str(), std::ios_base::binary);
        CPPUNIT_ASSERT_THROW(inputDescr.read(groupIstr), openvdb::IoError);
    }

    { // Test empty string parse
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "");
        CPPUNIT_ASSERT(testStringVector(includeNames));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test single token parse
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1"));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test parse with two include tokens
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1", "group2"));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test parse with one include and one ^ exclude token
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 ^group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group2"));
    }

    { // Test parse with one include and one ! exclude token
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 !group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group2"));
    }

    { // Test parse one include and one exclude backwards
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "^group1 group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group2"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group1"));
    }

    { // Test parse with two exclude tokens
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "^group1 ^group2");
        CPPUNIT_ASSERT(testStringVector(includeNames));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group1", "group2"));
    }

    { // Test parse multiple includes and excludes at the same time
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 ^group2 ^group3 group4");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1", "group4"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group2", "group3"));
    }

    { // Test parse misplaced negate character failure
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        CPPUNIT_ASSERT_THROW(Descriptor::parseNames(includeNames, excludeNames, "group1 ^ group2"), openvdb::RuntimeError);
    }

    { // Test parse (*) character
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "*");
        CPPUNIT_ASSERT(testStringVector(includeNames));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test parse invalid character failure
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        CPPUNIT_ASSERT_THROW(Descriptor::parseNames(includeNames, excludeNames, "group$1"), openvdb::RuntimeError);
    }

    { //  Test hasGroup(), setGroup(), dropGroup(), clearGroups()
        Descriptor descr;

        CPPUNIT_ASSERT(!descr.hasGroup("test1"));

        descr.setGroup("test1", 1);

        CPPUNIT_ASSERT(descr.hasGroup("test1"));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test1"), size_t(1));

        descr.setGroup("test5", 5);

        CPPUNIT_ASSERT(descr.hasGroup("test1"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test1"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test5"), size_t(5));

        descr.setGroup("test1", 2);

        CPPUNIT_ASSERT(descr.hasGroup("test1"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test1"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test5"), size_t(5));

        descr.dropGroup("test1");

        CPPUNIT_ASSERT(!descr.hasGroup("test1"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));

        descr.setGroup("test3", 3);

        CPPUNIT_ASSERT(descr.hasGroup("test3"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));

        descr.clearGroups();

        CPPUNIT_ASSERT(!descr.hasGroup("test1"));
        CPPUNIT_ASSERT(!descr.hasGroup("test3"));
        CPPUNIT_ASSERT(!descr.hasGroup("test5"));
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
    using namespace openvdb::tools;

    typedef openvdb::tools::AttributeArray AttributeArray;

    // Define and register some common attribute types
    typedef openvdb::tools::TypedAttributeArray<float>          AttributeS;
    typedef openvdb::tools::TypedAttributeArray<int32_t>        AttributeI;
    typedef openvdb::tools::TypedAttributeArray<int64_t>        AttributeL;
    typedef openvdb::tools::TypedAttributeArray<openvdb::Vec3s> AttributeVec3s;

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

        openvdb::TypedMetadata<AttributeS::ValueType> defaultValueTest(5);

        // add a default value of the wrong type

        openvdb::TypedMetadata<int> defaultValueInt(5);

        CPPUNIT_ASSERT_THROW(descrB->setDefaultValue("test", defaultValueInt), openvdb::TypeError);

        // add a default value with a name that does not exist

        CPPUNIT_ASSERT_THROW(descrB->setDefaultValue("badname", defaultValueTest), openvdb::LookupError);

        // add a default value for test of 5

        descrB->setDefaultValue("test", defaultValueTest);

        {
            openvdb::Metadata::Ptr meta = descrB->getMetadata()["default:test"];
            CPPUNIT_ASSERT(meta);
            CPPUNIT_ASSERT(meta->typeName() == "float");
        }

        // ensure attribute order persists

        CPPUNIT_ASSERT_EQUAL(descrB->find("pos"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(descrB->find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(descrB->find("test"), size_t(2));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttribute(newAttributes[0], defaultValueTest.copy());

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *descrB));
        }
        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttribute(newAttributes[0], attrSetC.descriptor(), descrB);

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        // add a default value for pos of (1, 3, 1)

        openvdb::TypedMetadata<AttributeVec3s::ValueType> defaultValuePos(AttributeVec3s::ValueType(1, 3, 1));

        descrB->setDefaultValue("pos", defaultValuePos);

        {
            openvdb::Metadata::Ptr meta = descrB->getMetadata()["default:pos"];
            CPPUNIT_ASSERT(meta);
            CPPUNIT_ASSERT(meta->typeName() == "vec3s");
            CPPUNIT_ASSERT_EQUAL(descrB->getDefaultValue<AttributeVec3s::ValueType>("pos"), defaultValuePos.value());
        }

        // remove default value

        CPPUNIT_ASSERT(descrB->hasDefaultValue("test"));

        descrB->removeDefaultValue("test");

        CPPUNIT_ASSERT(!descrB->hasDefaultValue("test"));
    }

    { // attribute removal

        Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
            .add("pos", AttributeVec3s::attributeType())
            .add("test", AttributeI::attributeType())
            .add("id", AttributeL::attributeType())
            .add("test2", AttributeI::attributeType())
            .add("id2", AttributeL::attributeType())
            .add("test3", AttributeI::attributeType())
            .vec);

        Descriptor::Ptr targetDescr = Descriptor::create(Descriptor::Inserter()
            .add("pos", AttributeVec3s::attributeType())
            .add("id", AttributeL::attributeType())
            .add("id2", AttributeL::attributeType())
            .vec);

        AttributeSet attrSetB(descr, /*arrayLength=*/50);

        // add some default values

        openvdb::TypedMetadata<AttributeI::ValueType> defaultOne(AttributeI::ValueType(1));

        descr->setDefaultValue("test", defaultOne);
        descr->setDefaultValue("test2", defaultOne);

        openvdb::TypedMetadata<AttributeL::ValueType> defaultThree(AttributeL::ValueType(3));

        descr->setDefaultValue("id", defaultThree);

        std::vector<size_t> toDrop;
        toDrop.push_back(descr->find("test"));
        toDrop.push_back(descr->find("test2"));
        toDrop.push_back(descr->find("test3"));

        CPPUNIT_ASSERT_EQUAL(toDrop[0], size_t(1));
        CPPUNIT_ASSERT_EQUAL(toDrop[1], size_t(3));
        CPPUNIT_ASSERT_EQUAL(toDrop[2], size_t(5));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            CPPUNIT_ASSERT(attrSetC.descriptor().getMetadata()["default:test"]);

            attrSetC.dropAttributes(toDrop);

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(3));

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));

            // check default values have been removed for the relevant attributes

            const Descriptor& descrC = attrSetC.descriptor();

            CPPUNIT_ASSERT(!descrC.getMetadata()["default:test"]);
            CPPUNIT_ASSERT(!descrC.getMetadata()["default:test2"]);
            CPPUNIT_ASSERT(!descrC.getMetadata()["default:test3"]);

            CPPUNIT_ASSERT(descrC.getMetadata()["default:id"]);
        }

        { // reverse removal order
            std::vector<size_t> toDropReverse;
            toDropReverse.push_back(descr->find("test3"));
            toDropReverse.push_back(descr->find("test2"));
            toDropReverse.push_back(descr->find("test"));

            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            attrSetC.dropAttributes(toDropReverse);

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(3));

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

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(3));

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

    { // metadata test
        Descriptor::Ptr descr1 = Descriptor::create(Descriptor::Inserter()
            .add("pos", AttributeVec3s::attributeType())
            .add("id", AttributeI::attributeType())
            .vec);

        Descriptor::Ptr descr2 = Descriptor::create(Descriptor::Inserter()
            .add("pos", AttributeVec3s::attributeType())
            .add("id", AttributeI::attributeType())
            .vec);

        openvdb::MetaMap& meta = descr1->getMetadata();
        meta.insertMeta("exampleMeta", openvdb::FloatMetadata(2.0));

        AttributeSet attrSetA(descr1);
        AttributeSet attrSetB(descr2);
        AttributeSet attrSetC(attrSetA);

        CPPUNIT_ASSERT(attrSetA != attrSetB);
        CPPUNIT_ASSERT(attrSetA == attrSetC);
    }

    // add some metadata and register the type

    openvdb::MetaMap& meta = attrSetA.descriptor().getMetadata();
    meta.insertMeta("exampleMeta", openvdb::FloatMetadata(2.0));

    { // flag size test
        Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
            .add("hidden1", AttributeI::attributeType())
            .add("group1", GroupAttributeArray::attributeType())
            .add("hidden2", AttributeI::attributeType())
            .vec);

        AttributeSet attrSet(descr);

        GroupAttributeArray::cast(*attrSet.get("group1")).setGroup(true);

        attrSet.get("hidden1")->setHidden(true);
        attrSet.get("hidden2")->setHidden(true);

        CPPUNIT_ASSERT_EQUAL(attrSet.size(AttributeArray::TRANSIENT), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attrSet.size(AttributeArray::GROUP), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attrSet.size(AttributeArray::HIDDEN), size_t(2));
    }

    { // I/O test
        std::ostringstream ostr(std::ios_base::binary);
        attrSetA.write(ostr);

        AttributeSet attrSetB;
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrSetB.read(istr);

        CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));
    }

    { // I/O transient test
        AttributeArray* array = attrSetA.get(0);
        array->setTransient(true);

        std::ostringstream ostr(std::ios_base::binary);
        attrSetA.write(ostr);

        AttributeSet attrSetB;
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrSetB.read(istr);

        // ensures transient attribute is not written out

        CPPUNIT_ASSERT_EQUAL(attrSetB.size(), size_t(1));
    }
}


void
TestAttributeSet::testAttributeSetGroups()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    // Define and register some common attribute types
    typedef TypedAttributeArray<int32_t>        AttributeI;
    typedef TypedAttributeArray<openvdb::Vec3s> AttributeVec3s;

    typedef AttributeSet::Descriptor Descriptor;

    // construct

    Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
        .add("pos", AttributeVec3s::attributeType())
        .add("id", AttributeI::attributeType())
        .vec);

    AttributeSet attrSet(descr, /*arrayLength=*/3);

    {
        CPPUNIT_ASSERT(!descr->hasGroup("test1"));
    }

    { // group offset
        Descriptor::Ptr descr(new Descriptor);

        descr->setGroup("test1", 1);

        CPPUNIT_ASSERT(descr->hasGroup("test1"));
        CPPUNIT_ASSERT_EQUAL(descr->groupMap().at("test1"), size_t(1));

        AttributeSet attrSet(descr);

        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset("test1"), size_t(1));
    }

    { // group index
        Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
            .add("test", AttributeI::attributeType())
            .add("test2", AttributeI::attributeType())
            .add("group1", GroupAttributeArray::attributeType())
            .add("test3", AttributeI::attributeType())
            .add("group2", GroupAttributeArray::attributeType())
            .add("test4", AttributeI::attributeType())
            .add("group3", GroupAttributeArray::attributeType())
            .vec);

        AttributeSet attrSet(descr);

        GroupAttributeArray::cast(*attrSet.get("group1")).setGroup(true);
        GroupAttributeArray::cast(*attrSet.get("group2")).setGroup(true);
        GroupAttributeArray::cast(*attrSet.get("group3")).setGroup(true);

        std::stringstream ss;
        for (int i = 0; i < 17; i++) {
            ss.str("");
            ss << "test" << i;
            descr->setGroup(ss.str(), i);
        }

        Descriptor::GroupIndex index15 = attrSet.groupIndex(15);
        CPPUNIT_ASSERT_EQUAL(index15.first, size_t(4));
        CPPUNIT_ASSERT_EQUAL(index15.second, uint8_t(7));

        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset(index15), size_t(15));
        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset("test15"), size_t(15));

        Descriptor::GroupIndex index15b = attrSet.groupIndex("test15");
        CPPUNIT_ASSERT_EQUAL(index15b.first, size_t(4));
        CPPUNIT_ASSERT_EQUAL(index15b.second, uint8_t(7));

        Descriptor::GroupIndex index16 = attrSet.groupIndex(16);
        CPPUNIT_ASSERT_EQUAL(index16.first, size_t(6));
        CPPUNIT_ASSERT_EQUAL(index16.second, uint8_t(0));

        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset(index16), size_t(16));
        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset("test16"), size_t(16));

        Descriptor::GroupIndex index16b = attrSet.groupIndex("test16");
        CPPUNIT_ASSERT_EQUAL(index16b.first, size_t(6));
        CPPUNIT_ASSERT_EQUAL(index16b.second, uint8_t(0));

        // check out of range exception

        CPPUNIT_ASSERT_NO_THROW(attrSet.groupIndex(23));
        CPPUNIT_ASSERT_THROW(attrSet.groupIndex(24), LookupError);
    }
}


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
