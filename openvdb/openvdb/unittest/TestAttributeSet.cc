// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/AttributeGroup.h>
#include <openvdb/points/AttributeSet.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

class TestAttributeSet: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }

    void testAttributeSet();
    void testAttributeSetDescriptor();
}; // class TestAttributeSet


////////////////////////////////////////


using namespace openvdb;
using namespace openvdb::points;

namespace {

bool
matchingAttributeSets(const AttributeSet& lhs,
    const AttributeSet& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    if (lhs.memUsage() != rhs.memUsage()) return false;
    if (lhs.descriptor() != rhs.descriptor()) return false;

    for (size_t n = 0, N = lhs.size(); n < N; ++n) {

        const AttributeArray* a = lhs.getConst(n);
        const AttributeArray* b = rhs.getConst(n);

        if (a->size() != b->size()) return false;
        if (a->isUniform() != b->isUniform()) return false;
        if (a->isHidden() != b->isHidden()) return false;
        if (a->type() != b->type()) return false;
    }

    return true;
}

bool
attributeSetMatchesDescriptor(  const AttributeSet& attrSet,
                                const AttributeSet::Descriptor& descriptor)
{
    if (descriptor.size() != attrSet.size())    return false;

    // check default metadata

    const openvdb::MetaMap& meta1 = descriptor.getMetadata();
    const openvdb::MetaMap& meta2 = attrSet.descriptor().getMetadata();

    // build vector of all default keys

    std::vector<openvdb::Name> defaultKeys;

    for (auto it = meta1.beginMeta(), itEnd = meta1.endMeta(); it != itEnd; ++it)
    {
        const openvdb::Name& name = it->first;

        if (name.compare(0, 8, "default:") == 0) {
            defaultKeys.push_back(name);
        }
    }

    for (auto it = meta2.beginMeta(), itEnd = meta2.endMeta(); it != itEnd; ++it)
    {
        const openvdb::Name& name = it->first;

        if (name.compare(0, 8, "default:") == 0) {
            if (std::find(defaultKeys.begin(), defaultKeys.end(), name) != defaultKeys.end()) {
                defaultKeys.push_back(name);
            }
        }
    }

    // compare metadata value from each metamap

    for (const openvdb::Name& name : defaultKeys) {
        openvdb::Metadata::ConstPtr metaValue1 = meta1[name];
        openvdb::Metadata::ConstPtr metaValue2 = meta2[name];

        if (!metaValue1)    return false;
        if (!metaValue2)    return false;

        if (*metaValue1 != *metaValue2)     return false;
    }

    // ensure descriptor and attributes are still in sync

    for (const auto& namePos : attrSet.descriptor().map()) {
        const size_t pos = descriptor.find(namePos.first);

        if (pos != size_t(namePos.second))  return false;
        if (descriptor.type(pos) != attrSet.get(pos)->type())   return false;
    }

    return true;
}

bool testStringVector(std::vector<std::string>& input)
{
    return input.empty();
}

bool testStringVector(std::vector<std::string>& input, const std::string& name1)
{
    if (input.size() != 1)  return false;
    if (input[0] != name1)  return false;
    return true;
}

bool testStringVector(std::vector<std::string>& input,
    const std::string& name1, const std::string& name2)
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
    using AttributeVec3f    = TypedAttributeArray<openvdb::Vec3f>;
    using AttributeS        = TypedAttributeArray<float>;
    using AttributeI        = TypedAttributeArray<int32_t>;

    using Descriptor        = AttributeSet::Descriptor;

    { // error on invalid construction
        Descriptor::Ptr invalidDescr = Descriptor::create(AttributeVec3f::attributeType());
        EXPECT_THROW(invalidDescr->duplicateAppend("P", AttributeS::attributeType()),
            openvdb::KeyError);
    }

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3f::attributeType());

    descrA = descrA->duplicateAppend("density", AttributeS::attributeType());
    descrA = descrA->duplicateAppend("id", AttributeI::attributeType());

    Descriptor::Ptr descrB = Descriptor::create(AttributeVec3f::attributeType());

    descrB = descrB->duplicateAppend("density", AttributeS::attributeType());
    descrB = descrB->duplicateAppend("id", AttributeI::attributeType());

    EXPECT_EQ(descrA->size(), descrB->size());

    EXPECT_TRUE(*descrA == *descrB);

    descrB->setGroup("test", size_t(0));
    descrB->setGroup("test2", size_t(1));

    Descriptor descrC(*descrB);

    EXPECT_TRUE(descrB->hasSameAttributes(descrC));
    EXPECT_TRUE(descrC.hasGroup("test"));
    EXPECT_TRUE(*descrB == descrC);

    descrC.dropGroup("test");
    descrC.dropGroup("test2");

    EXPECT_TRUE(!descrB->hasSameAttributes(descrC));
    EXPECT_TRUE(!descrC.hasGroup("test"));
    EXPECT_TRUE(*descrB != descrC);

    descrC.setGroup("test2", size_t(1));
    descrC.setGroup("test3", size_t(0));

    EXPECT_TRUE(!descrB->hasSameAttributes(descrC));

    descrC.dropGroup("test3");
    descrC.setGroup("test", size_t(0));

    EXPECT_TRUE(descrB->hasSameAttributes(descrC));

    Descriptor::Inserter names;
    names.add("P", AttributeVec3f::attributeType());
    names.add("density", AttributeS::attributeType());
    names.add("id", AttributeI::attributeType());

    // rebuild NameAndTypeVec

    Descriptor::NameAndTypeVec rebuildNames;
    descrA->appendTo(rebuildNames);

    EXPECT_EQ(rebuildNames.size(), names.vec.size());

    for (auto itA = rebuildNames.cbegin(), itB = names.vec.cbegin(),
              itEndA = rebuildNames.cend(), itEndB = names.vec.cend();
              itA != itEndA && itB != itEndB; ++itA, ++itB) {
        EXPECT_EQ(itA->name, itB->name);
        EXPECT_EQ(itA->type.first, itB->type.first);
        EXPECT_EQ(itA->type.second, itB->type.second);
    }

    Descriptor::NameToPosMap groupMap;
    openvdb::MetaMap metadata;

    // hasSameAttributes (note: uses protected create methods)
    {
        Descriptor::Ptr descr1 = Descriptor::create(Descriptor::Inserter()
                .add("P", AttributeVec3f::attributeType())
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec, groupMap, metadata);

        // test same names with different types, should be false
        Descriptor::Ptr descr2 = Descriptor::create(Descriptor::Inserter()
                .add("P", AttributeVec3f::attributeType())
                .add("test", AttributeS::attributeType())
                .add("id", AttributeI::attributeType())
                .vec, groupMap, metadata);

        EXPECT_TRUE(!descr1->hasSameAttributes(*descr2));

        // test different names, should be false
        Descriptor::Ptr descr3 = Descriptor::create(Descriptor::Inserter()
                .add("P", AttributeVec3f::attributeType())
                .add("test2", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec, groupMap, metadata);

        EXPECT_TRUE(!descr1->hasSameAttributes(*descr3));

        // test same names and types but different order, should be true
        Descriptor::Ptr descr4 = Descriptor::create(Descriptor::Inserter()
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .add("P", AttributeVec3f::attributeType())
                .vec, groupMap, metadata);

        EXPECT_TRUE(descr1->hasSameAttributes(*descr4));
    }

    { // Test uniqueName
        Descriptor::Inserter names2;
        Descriptor::Ptr emptyDescr = Descriptor::create(AttributeVec3f::attributeType());
        const openvdb::Name uniqueNameEmpty = emptyDescr->uniqueName("test");
        EXPECT_EQ(uniqueNameEmpty, openvdb::Name("test"));

        names2.add("test", AttributeS::attributeType());
        names2.add("test1", AttributeI::attributeType());

        Descriptor::Ptr descr1 = Descriptor::create(names2.vec, groupMap, metadata);

        const openvdb::Name uniqueName1 = descr1->uniqueName("test");
        EXPECT_EQ(uniqueName1, openvdb::Name("test0"));

        Descriptor::Ptr descr2 = descr1->duplicateAppend(uniqueName1, AttributeI::attributeType());

        const openvdb::Name uniqueName2 = descr2->uniqueName("test");
        EXPECT_EQ(uniqueName2, openvdb::Name("test2"));
    }

    { // Test name validity

        EXPECT_TRUE(Descriptor::validName("test1"));
        EXPECT_TRUE(Descriptor::validName("abc_def"));
        EXPECT_TRUE(Descriptor::validName("abc|def"));
        EXPECT_TRUE(Descriptor::validName("abc:def"));

        EXPECT_TRUE(!Descriptor::validName(""));
        EXPECT_TRUE(!Descriptor::validName("test1!"));
        EXPECT_TRUE(!Descriptor::validName("abc=def"));
        EXPECT_TRUE(!Descriptor::validName("abc def"));
        EXPECT_TRUE(!Descriptor::validName("abc*def"));
    }

    { // Test enforcement of valid names
        Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter().add(
            "test1", AttributeS::attributeType()).vec, groupMap, metadata);
        EXPECT_THROW(descr->rename("test1", "test1!"), openvdb::RuntimeError);
        EXPECT_THROW(descr->setGroup("group1!", 1), openvdb::RuntimeError);

        Descriptor::NameAndType invalidAttr("test1!", AttributeS::attributeType());
        EXPECT_THROW(descr->duplicateAppend(invalidAttr.name, invalidAttr.type),
            openvdb::RuntimeError);

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
        EXPECT_THROW(inputDescr.read(attrIstr), openvdb::IoError);
        std::istringstream groupIstr(groupOstr.str(), std::ios_base::binary);
        EXPECT_THROW(inputDescr.read(groupIstr), openvdb::IoError);
    }

    { // Test empty string parse
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "");
        EXPECT_TRUE(testStringVector(includeNames));
        EXPECT_TRUE(testStringVector(excludeNames));
    }

    { // Test single token parse
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        bool includeAll = false;
        Descriptor::parseNames(includeNames, excludeNames, includeAll, "group1");
        EXPECT_TRUE(!includeAll);
        EXPECT_TRUE(testStringVector(includeNames, "group1"));
        EXPECT_TRUE(testStringVector(excludeNames));
    }

    { // Test parse with two include tokens
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 group2");
        EXPECT_TRUE(testStringVector(includeNames, "group1", "group2"));
        EXPECT_TRUE(testStringVector(excludeNames));
    }

    { // Test parse with one include and one ^ exclude token
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 ^group2");
        EXPECT_TRUE(testStringVector(includeNames, "group1"));
        EXPECT_TRUE(testStringVector(excludeNames, "group2"));
    }

    { // Test parse with one include and one ! exclude token
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 !group2");
        EXPECT_TRUE(testStringVector(includeNames, "group1"));
        EXPECT_TRUE(testStringVector(excludeNames, "group2"));
    }

    { // Test parse one include and one exclude backwards
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "^group1 group2");
        EXPECT_TRUE(testStringVector(includeNames, "group2"));
        EXPECT_TRUE(testStringVector(excludeNames, "group1"));
    }

    { // Test parse with two exclude tokens
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "^group1 ^group2");
        EXPECT_TRUE(testStringVector(includeNames));
        EXPECT_TRUE(testStringVector(excludeNames, "group1", "group2"));
    }

    { // Test parse multiple includes and excludes at the same time
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 ^group2 ^group3 group4");
        EXPECT_TRUE(testStringVector(includeNames, "group1", "group4"));
        EXPECT_TRUE(testStringVector(excludeNames, "group2", "group3"));
    }

    { // Test parse misplaced negate character failure
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        EXPECT_THROW(Descriptor::parseNames(includeNames, excludeNames, "group1 ^ group2"),
            openvdb::RuntimeError);
    }

    { // Test parse (*) character
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        bool includeAll = false;
        Descriptor::parseNames(includeNames, excludeNames, includeAll, "*");
        EXPECT_TRUE(includeAll);
        EXPECT_TRUE(testStringVector(includeNames));
        EXPECT_TRUE(testStringVector(excludeNames));
    }

    { // Test parse invalid character failure
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        EXPECT_THROW(Descriptor::parseNames(includeNames, excludeNames, "group$1"),
            openvdb::RuntimeError);
    }

    { //  Test hasGroup(), setGroup(), dropGroup(), clearGroups()
        Descriptor descr;

        EXPECT_TRUE(!descr.hasGroup("test1"));

        descr.setGroup("test1", 1);

        EXPECT_TRUE(descr.hasGroup("test1"));
        EXPECT_EQ(descr.groupMap().at("test1"), size_t(1));

        descr.setGroup("test5", 5);

        EXPECT_TRUE(descr.hasGroup("test1"));
        EXPECT_TRUE(descr.hasGroup("test5"));
        EXPECT_EQ(descr.groupMap().at("test1"), size_t(1));
        EXPECT_EQ(descr.groupMap().at("test5"), size_t(5));

        descr.setGroup("test1", 2);

        EXPECT_TRUE(descr.hasGroup("test1"));
        EXPECT_TRUE(descr.hasGroup("test5"));
        EXPECT_EQ(descr.groupMap().at("test1"), size_t(2));
        EXPECT_EQ(descr.groupMap().at("test5"), size_t(5));

        descr.dropGroup("test1");

        EXPECT_TRUE(!descr.hasGroup("test1"));
        EXPECT_TRUE(descr.hasGroup("test5"));

        descr.setGroup("test3", 3);

        EXPECT_TRUE(descr.hasGroup("test3"));
        EXPECT_TRUE(descr.hasGroup("test5"));

        descr.clearGroups();

        EXPECT_TRUE(!descr.hasGroup("test1"));
        EXPECT_TRUE(!descr.hasGroup("test3"));
        EXPECT_TRUE(!descr.hasGroup("test5"));
    }

    // I/O test

    std::ostringstream ostr(std::ios_base::binary);
    descrA->write(ostr);

    Descriptor inputDescr;

    std::istringstream istr(ostr.str(), std::ios_base::binary);
    inputDescr.read(istr);

    EXPECT_EQ(descrA->size(), inputDescr.size());
    EXPECT_TRUE(*descrA == inputDescr);
}
TEST_F(TestAttributeSet, testAttributeSetDescriptor) { testAttributeSetDescriptor(); }


void
TestAttributeSet::testAttributeSet()
{
    // Define and register some common attribute types
    using AttributeS        = TypedAttributeArray<float>;
    using AttributeB        = TypedAttributeArray<bool>;
    using AttributeI        = TypedAttributeArray<int32_t>;
    using AttributeL        = TypedAttributeArray<int64_t>;
    using AttributeVec3s    = TypedAttributeArray<Vec3s>;

    using Descriptor        = AttributeSet::Descriptor;

    Descriptor::NameToPosMap groupMap;
    openvdb::MetaMap metadata;

    { // construction
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        descr = descr->duplicateAppend("test", AttributeI::attributeType());
        AttributeSet attrSet(descr);
        EXPECT_EQ(attrSet.size(), size_t(2));

        Descriptor::Ptr newDescr = Descriptor::create(AttributeVec3s::attributeType());
        EXPECT_THROW(attrSet.resetDescriptor(newDescr), openvdb::LookupError);
        EXPECT_NO_THROW(
            attrSet.resetDescriptor(newDescr, /*allowMismatchingDescriptors=*/true));
    }

    { // transfer of flags on construction
        AttributeSet attrSet(Descriptor::create(AttributeVec3s::attributeType()));
        AttributeArray::Ptr array1 = attrSet.appendAttribute(
            "hidden", AttributeS::attributeType());
        array1->setHidden(true);
        AttributeArray::Ptr array2 = attrSet.appendAttribute(
            "transient", AttributeS::attributeType());
        array2->setTransient(true);
        AttributeSet attrSet2(attrSet, size_t(1));
        EXPECT_TRUE(attrSet2.getConst("hidden")->isHidden());
        EXPECT_TRUE(attrSet2.getConst("transient")->isTransient());
    }

    // construct

    { // invalid append
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        AttributeSet invalidAttrSetA(descr, /*arrayLength=*/50);

        EXPECT_THROW(invalidAttrSetA.appendAttribute("id", AttributeI::attributeType(),
            /*stride=*/0, /*constantStride=*/true), openvdb::ValueError);
        EXPECT_TRUE(invalidAttrSetA.find("id") == AttributeSet::INVALID_POS);
        EXPECT_THROW(invalidAttrSetA.appendAttribute("id", AttributeI::attributeType(),
            /*stride=*/49, /*constantStride=*/false), openvdb::ValueError);
        EXPECT_NO_THROW(
            invalidAttrSetA.appendAttribute("testStride1", AttributeI::attributeType(),
            /*stride=*/50, /*constantStride=*/false));
        EXPECT_NO_THROW(
            invalidAttrSetA.appendAttribute("testStride2", AttributeI::attributeType(),
            /*stride=*/51, /*constantStride=*/false));
    }

    { // copy construction with varying attribute types and strides
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        AttributeSet attrSet(descr, /*arrayLength=*/50);

        attrSet.appendAttribute("float1", AttributeS::attributeType(), /*stride=*/1);
        attrSet.appendAttribute("int1", AttributeI::attributeType(), /*stride=*/1);
        attrSet.appendAttribute("float3", AttributeS::attributeType(), /*stride=*/3);
        attrSet.appendAttribute("vector", AttributeVec3s::attributeType(), /*stride=*/1);
        attrSet.appendAttribute("vector3", AttributeVec3s::attributeType(), /*stride=*/3);
        attrSet.appendAttribute("bool100", AttributeB::attributeType(), /*stride=*/100);
        attrSet.appendAttribute("boolDynamic", AttributeB::attributeType(), /*size=*/100, false);
        attrSet.appendAttribute("intDynamic", AttributeI::attributeType(), /*size=*/300, false);

        EXPECT_EQ(std::string("float"), attrSet.getConst("float1")->type().first);
        EXPECT_EQ(std::string("int32"), attrSet.getConst("int1")->type().first);
        EXPECT_EQ(std::string("float"), attrSet.getConst("float3")->type().first);
        EXPECT_EQ(std::string("vec3s"), attrSet.getConst("vector")->type().first);
        EXPECT_EQ(std::string("vec3s"), attrSet.getConst("vector3")->type().first);
        EXPECT_EQ(std::string("bool"), attrSet.getConst("bool100")->type().first);
        EXPECT_EQ(std::string("bool"), attrSet.getConst("boolDynamic")->type().first);
        EXPECT_EQ(std::string("int32"), attrSet.getConst("intDynamic")->type().first);

        EXPECT_EQ(openvdb::Index(1), attrSet.getConst("float1")->stride());
        EXPECT_EQ(openvdb::Index(1), attrSet.getConst("int1")->stride());
        EXPECT_EQ(openvdb::Index(3), attrSet.getConst("float3")->stride());
        EXPECT_EQ(openvdb::Index(1), attrSet.getConst("vector")->stride());
        EXPECT_EQ(openvdb::Index(3), attrSet.getConst("vector3")->stride());
        EXPECT_EQ(openvdb::Index(100), attrSet.getConst("bool100")->stride());

        EXPECT_EQ(openvdb::Index(50), attrSet.getConst("float1")->size());

        // error as the new length is greater than the data size of the
        // 'boolDynamic' attribute
        EXPECT_THROW(AttributeSet(attrSet, /*arrayLength=*/200), openvdb::ValueError);

        AttributeSet attrSet2(attrSet, /*arrayLength=*/100);

        EXPECT_EQ(std::string("float"), attrSet2.getConst("float1")->type().first);
        EXPECT_EQ(std::string("int32"), attrSet2.getConst("int1")->type().first);
        EXPECT_EQ(std::string("float"), attrSet2.getConst("float3")->type().first);
        EXPECT_EQ(std::string("vec3s"), attrSet2.getConst("vector")->type().first);
        EXPECT_EQ(std::string("vec3s"), attrSet2.getConst("vector3")->type().first);
        EXPECT_EQ(std::string("bool"), attrSet2.getConst("bool100")->type().first);
        EXPECT_EQ(std::string("bool"), attrSet2.getConst("boolDynamic")->type().first);
        EXPECT_EQ(std::string("int32"), attrSet2.getConst("intDynamic")->type().first);

        EXPECT_EQ(openvdb::Index(1), attrSet2.getConst("float1")->stride());
        EXPECT_EQ(openvdb::Index(1), attrSet2.getConst("int1")->stride());
        EXPECT_EQ(openvdb::Index(3), attrSet2.getConst("float3")->stride());
        EXPECT_EQ(openvdb::Index(1), attrSet2.getConst("vector")->stride());
        EXPECT_EQ(openvdb::Index(3), attrSet2.getConst("vector3")->stride());
        EXPECT_EQ(openvdb::Index(100), attrSet2.getConst("bool100")->stride());
        EXPECT_EQ(openvdb::Index(0), attrSet2.getConst("boolDynamic")->stride());
        EXPECT_EQ(openvdb::Index(0), attrSet2.getConst("intDynamic")->stride());

        EXPECT_EQ(openvdb::Index(100), attrSet2.getConst("float1")->size());
        EXPECT_EQ(openvdb::Index(100), attrSet2.getConst("boolDynamic")->size());
        EXPECT_EQ(openvdb::Index(100), attrSet2.getConst("intDynamic")->size());
        EXPECT_EQ(openvdb::Index(100), attrSet2.getConst("boolDynamic")->dataSize());
        EXPECT_EQ(openvdb::Index(300), attrSet2.getConst("intDynamic")->dataSize());
    }

    Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
    AttributeSet attrSetA(descr, /*arrayLength=*/50);

    attrSetA.appendAttribute("id", AttributeI::attributeType());

    // check equality against duplicate array

    Descriptor::Ptr descr2 = Descriptor::create(AttributeVec3s::attributeType());
    AttributeSet attrSetA2(descr2, /*arrayLength=*/50);

    attrSetA2.appendAttribute("id", AttributeI::attributeType());

    EXPECT_TRUE(attrSetA == attrSetA2);

    // expand uniform values and check equality

    attrSetA.get("P")->expand();
    attrSetA2.get("P")->expand();

    EXPECT_TRUE(attrSetA == attrSetA2);

    EXPECT_EQ(size_t(2), attrSetA.size());
    EXPECT_EQ(openvdb::Index(50), attrSetA.get(0)->size());
    EXPECT_EQ(openvdb::Index(50), attrSetA.get(1)->size());

    { // copy
        EXPECT_TRUE(!attrSetA.isShared(0));
        EXPECT_TRUE(!attrSetA.isShared(1));

        AttributeSet attrSetB(attrSetA);

        EXPECT_TRUE(matchingAttributeSets(attrSetA, attrSetB));

        EXPECT_TRUE(attrSetA.isShared(0));
        EXPECT_TRUE(attrSetA.isShared(1));
        EXPECT_TRUE(attrSetB.isShared(0));
        EXPECT_TRUE(attrSetB.isShared(1));

        attrSetB.makeUnique(0);
        attrSetB.makeUnique(1);

        EXPECT_TRUE(matchingAttributeSets(attrSetA, attrSetB));

        EXPECT_TRUE(!attrSetA.isShared(0));
        EXPECT_TRUE(!attrSetA.isShared(1));
        EXPECT_TRUE(!attrSetB.isShared(0));
        EXPECT_TRUE(!attrSetB.isShared(1));
    }

    { // attribute insertion
        AttributeSet attrSetB(attrSetA);

        attrSetB.makeUnique(0);
        attrSetB.makeUnique(1);

        Descriptor::Ptr targetDescr = Descriptor::create(Descriptor::Inserter()
            .add("P", AttributeVec3s::attributeType())
            .add("id", AttributeI::attributeType())
            .add("test", AttributeS::attributeType())
            .vec, groupMap, metadata);

        Descriptor::Ptr descrB =
            attrSetB.descriptor().duplicateAppend("test", AttributeS::attributeType());

        // should throw if we attempt to add the same attribute name but a different type
        EXPECT_THROW(
            descrB->insert("test", AttributeI::attributeType()), openvdb::KeyError);

        // shouldn't throw if we attempt to add the same attribute name and type
        EXPECT_NO_THROW(descrB->insert("test", AttributeS::attributeType()));

        openvdb::TypedMetadata<AttributeS::ValueType> defaultValueTest(5);

        // add a default value of the wrong type

        openvdb::TypedMetadata<int> defaultValueInt(5);

        EXPECT_THROW(descrB->setDefaultValue("test", defaultValueInt), openvdb::TypeError);

        // add a default value with a name that does not exist

        EXPECT_THROW(descrB->setDefaultValue("badname", defaultValueTest),
            openvdb::LookupError);

        // add a default value for test of 5

        descrB->setDefaultValue("test", defaultValueTest);

        {
            openvdb::Metadata::Ptr meta = descrB->getMetadata()["default:test"];
            EXPECT_TRUE(meta);
            EXPECT_TRUE(meta->typeName() == "float");
        }

        // ensure attribute order persists

        EXPECT_EQ(descrB->find("P"), size_t(0));
        EXPECT_EQ(descrB->find("id"), size_t(1));
        EXPECT_EQ(descrB->find("test"), size_t(2));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttribute("test", AttributeS::attributeType(), /*stride=*/1,
                                        /*constantStride=*/true, defaultValueTest.copy().get());

            EXPECT_TRUE(attributeSetMatchesDescriptor(attrSetC, *descrB));
        }
        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttribute(attrSetC.descriptor(), descrB, size_t(2));

            EXPECT_TRUE(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        // add a default value for pos of (1, 3, 1)

        openvdb::TypedMetadata<AttributeVec3s::ValueType> defaultValuePos(
            AttributeVec3s::ValueType(1, 3, 1));

        descrB->setDefaultValue("P", defaultValuePos);

        {
            openvdb::Metadata::Ptr meta = descrB->getMetadata()["default:P"];
            EXPECT_TRUE(meta);
            EXPECT_TRUE(meta->typeName() == "vec3s");
            EXPECT_EQ(descrB->getDefaultValue<AttributeVec3s::ValueType>("P"),
                defaultValuePos.value());
        }

        // remove default value

        EXPECT_TRUE(descrB->hasDefaultValue("test"));

        descrB->removeDefaultValue("test");

        EXPECT_TRUE(!descrB->hasDefaultValue("test"));
    }

    { // attribute removal

        Descriptor::Ptr descr1 = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSetB(descr1, /*arrayLength=*/50);

        TypedMetadata<int> defaultValue(7);
        Metadata& baseDefaultValue = defaultValue;

        attrSetB.appendAttribute("test", AttributeI::attributeType(),
            Index(1), true, &baseDefaultValue);
        attrSetB.appendAttribute("id", AttributeL::attributeType());
        attrSetB.appendAttribute("test2", AttributeI::attributeType());
        attrSetB.appendAttribute("id2", AttributeL::attributeType());
        attrSetB.appendAttribute("test3", AttributeI::attributeType());

        // check default value of "test" attribute has been applied
        EXPECT_EQ(7, attrSetB.descriptor().getDefaultValue<int>("test"));
        EXPECT_EQ(7, AttributeI::cast(*attrSetB.getConst("test")).get(0));

        descr1 = attrSetB.descriptorPtr();

        Descriptor::Ptr targetDescr = Descriptor::create(AttributeVec3s::attributeType());

        targetDescr = targetDescr->duplicateAppend("id", AttributeL::attributeType());
        targetDescr = targetDescr->duplicateAppend("id2", AttributeL::attributeType());

        // add some default values

        openvdb::TypedMetadata<AttributeI::ValueType> defaultOne(AttributeI::ValueType(1));

        descr1->setDefaultValue("test", defaultOne);
        descr1->setDefaultValue("test2", defaultOne);

        openvdb::TypedMetadata<AttributeL::ValueType> defaultThree(AttributeL::ValueType(3));

        descr1->setDefaultValue("id", defaultThree);

        std::vector<size_t> toDrop{
            descr1->find("test"), descr1->find("test2"), descr1->find("test3")};

        EXPECT_EQ(toDrop[0], size_t(1));
        EXPECT_EQ(toDrop[1], size_t(3));
        EXPECT_EQ(toDrop[2], size_t(5));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            EXPECT_TRUE(attrSetC.descriptor().getMetadata()["default:test"]);

            attrSetC.dropAttributes(toDrop);

            EXPECT_EQ(attrSetC.size(), size_t(3));

            EXPECT_TRUE(attributeSetMatchesDescriptor(attrSetC, *targetDescr));

            // check default values have been removed for the relevant attributes

            const Descriptor& descrC = attrSetC.descriptor();

            EXPECT_TRUE(!descrC.getMetadata()["default:test"]);
            EXPECT_TRUE(!descrC.getMetadata()["default:test2"]);
            EXPECT_TRUE(!descrC.getMetadata()["default:test3"]);

            EXPECT_TRUE(descrC.getMetadata()["default:id"]);
        }

        { // reverse removal order
            std::vector<size_t> toDropReverse{
                descr1->find("test3"), descr1->find("test2"), descr1->find("test")};

            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            attrSetC.dropAttributes(toDropReverse);

            EXPECT_EQ(attrSetC.size(), size_t(3));

            EXPECT_TRUE(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            Descriptor::Ptr descrB = attrSetB.descriptor().duplicateDrop(toDrop);

            attrSetC.dropAttributes(toDrop, attrSetC.descriptor(), descrB);

            EXPECT_EQ(attrSetC.size(), size_t(3));

            EXPECT_TRUE(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        { // remove attribute
            AttributeSet attrSetC;
            attrSetC.appendAttribute("test1", AttributeI::attributeType());
            attrSetC.appendAttribute("test2", AttributeI::attributeType());
            attrSetC.appendAttribute("test3", AttributeI::attributeType());
            attrSetC.appendAttribute("test4", AttributeI::attributeType());
            attrSetC.appendAttribute("test5", AttributeI::attributeType());

            EXPECT_EQ(attrSetC.size(), size_t(5));

            { // remove test2
                AttributeArray::Ptr array = attrSetC.removeAttribute(1);
                EXPECT_TRUE(array);
                EXPECT_EQ(array.use_count(), long(1));
            }

            EXPECT_EQ(attrSetC.size(), size_t(4));
            EXPECT_EQ(attrSetC.descriptor().size(), size_t(4));

            { // remove test5
                AttributeArray::Ptr array = attrSetC.removeAttribute("test5");
                EXPECT_TRUE(array);
                EXPECT_EQ(array.use_count(), long(1));
            }

            EXPECT_EQ(attrSetC.size(), size_t(3));
            EXPECT_EQ(attrSetC.descriptor().size(), size_t(3));

            { // remove test3 unsafely
                AttributeArray::Ptr array = attrSetC.removeAttributeUnsafe(1);
                EXPECT_TRUE(array);
                EXPECT_EQ(array.use_count(), long(1));
            }

            // array of attributes and descriptor are not updated

            EXPECT_EQ(attrSetC.size(), size_t(3));
            EXPECT_EQ(attrSetC.descriptor().size(), size_t(3));

            const auto& nameToPosMap = attrSetC.descriptor().map();

            EXPECT_EQ(nameToPosMap.size(), size_t(3));
            EXPECT_EQ(nameToPosMap.at("test1"), size_t(0));
            EXPECT_EQ(nameToPosMap.at("test3"), size_t(1)); // this array does not exist
            EXPECT_EQ(nameToPosMap.at("test4"), size_t(2));

            EXPECT_TRUE(attrSetC.getConst(0));
            EXPECT_TRUE(!attrSetC.getConst(1)); // this array does not exist
            EXPECT_TRUE(attrSetC.getConst(2));
        }

        { // test duplicateDrop configures group mapping
            AttributeSet attrSetC;

            const size_t GROUP_BITS = sizeof(GroupType) * CHAR_BIT;

            attrSetC.appendAttribute("test1", AttributeI::attributeType());
            attrSetC.appendAttribute("__group1", GroupAttributeArray::attributeType());
            attrSetC.appendAttribute("test2", AttributeI::attributeType());
            attrSetC.appendAttribute("__group2", GroupAttributeArray::attributeType());
            attrSetC.appendAttribute("__group3", GroupAttributeArray::attributeType());
            attrSetC.appendAttribute("__group4", GroupAttributeArray::attributeType());

            // 5 attributes exist - append a group as the sixth and then drop

            Descriptor::Ptr descriptor = attrSetC.descriptorPtr();
            size_t count = descriptor->count(GroupAttributeArray::attributeType());
            EXPECT_EQ(count, size_t(4));

            descriptor->setGroup("test_group1", /*offset*/0); // __group1
            descriptor->setGroup("test_group2", /*offset=8*/GROUP_BITS); // __group2
            descriptor->setGroup("test_group3", /*offset=16*/GROUP_BITS*2); // __group3
            descriptor->setGroup("test_group4", /*offset=28*/GROUP_BITS*3 + GROUP_BITS/2); // __group4

            descriptor = descriptor->duplicateDrop({ 1, 2, 3 });
            count = descriptor->count(GroupAttributeArray::attributeType());
            EXPECT_EQ(count, size_t(2));

            EXPECT_EQ(size_t(3), descriptor->size());
            EXPECT_TRUE(!descriptor->hasGroup("test_group1"));
            EXPECT_TRUE(!descriptor->hasGroup("test_group2"));
            EXPECT_TRUE(descriptor->hasGroup("test_group3"));
            EXPECT_TRUE(descriptor->hasGroup("test_group4"));

            EXPECT_EQ(descriptor->find("__group1"), size_t(AttributeSet::INVALID_POS));
            EXPECT_EQ(descriptor->find("__group2"), size_t(AttributeSet::INVALID_POS));
            EXPECT_EQ(descriptor->find("__group3"), size_t(1));
            EXPECT_EQ(descriptor->find("__group4"), size_t(2));

            EXPECT_EQ(descriptor->groupOffset("test_group3"), size_t(0));
            EXPECT_EQ(descriptor->groupOffset("test_group4"), size_t(GROUP_BITS + GROUP_BITS/2));
        }
    }

    // replace existing arrays

    // this replace call should not take effect since the new attribute
    // array type does not match with the descriptor type for the given position.
    AttributeArray::Ptr floatAttr(new AttributeS(15));
    EXPECT_TRUE(attrSetA.replace(1, floatAttr) == AttributeSet::INVALID_POS);

    AttributeArray::Ptr intAttr(new AttributeI(10));
    EXPECT_TRUE(attrSetA.replace(1, intAttr) != AttributeSet::INVALID_POS);

    EXPECT_EQ(openvdb::Index(10), attrSetA.get(1)->size());

    { // reorder attribute set
        Descriptor::Ptr descr1 = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSetA1(descr1);

        attrSetA1.appendAttribute("test", AttributeI::attributeType());
        attrSetA1.appendAttribute("id", AttributeI::attributeType());
        attrSetA1.appendAttribute("test2", AttributeI::attributeType());

        descr1 = attrSetA1.descriptorPtr();

        Descriptor::Ptr descr2x = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSetB1(descr2x);

        attrSetB1.appendAttribute("test2", AttributeI::attributeType());
        attrSetB1.appendAttribute("test", AttributeI::attributeType());
        attrSetB1.appendAttribute("id", AttributeI::attributeType());

        EXPECT_TRUE(attrSetA1 != attrSetB1);

        attrSetB1.reorderAttributes(descr1);

        EXPECT_TRUE(attrSetA1 == attrSetB1);
    }

    { // metadata test
        Descriptor::Ptr descr1A = Descriptor::create(AttributeVec3s::attributeType());

        Descriptor::Ptr descr2A = Descriptor::create(AttributeVec3s::attributeType());

        openvdb::MetaMap& meta = descr1A->getMetadata();
        meta.insertMeta("exampleMeta", openvdb::FloatMetadata(2.0));

        AttributeSet attrSetA1(descr1A);
        AttributeSet attrSetB1(descr2A);
        AttributeSet attrSetC1(attrSetA1);

        EXPECT_TRUE(attrSetA1 != attrSetB1);
        EXPECT_TRUE(attrSetA1 == attrSetC1);
    }

    // add some metadata and register the type

    openvdb::MetaMap& meta = attrSetA.descriptor().getMetadata();
    meta.insertMeta("exampleMeta", openvdb::FloatMetadata(2.0));

    { // I/O test
        std::ostringstream ostr(std::ios_base::binary);
        attrSetA.write(ostr);

        AttributeSet attrSetB;
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrSetB.read(istr);

        EXPECT_TRUE(matchingAttributeSets(attrSetA, attrSetB));
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

        EXPECT_EQ(attrSetB.size(), size_t(1));

        std::ostringstream ostr2(std::ios_base::binary);
        attrSetA.write(ostr2, /*transient=*/true);

        AttributeSet attrSetC;
        std::istringstream istr2(ostr2.str(), std::ios_base::binary);
        attrSetC.read(istr2);

        EXPECT_EQ(attrSetC.size(), size_t(2));
    }
}
TEST_F(TestAttributeSet, testAttributeSet) { testAttributeSet(); }


TEST_F(TestAttributeSet, testAttributeSetGroups)
{
    // Define and register some common attribute types
    using AttributeI        = TypedAttributeArray<int32_t>;
    using AttributeVec3s    = TypedAttributeArray<openvdb::Vec3s>;

    using Descriptor        = AttributeSet::Descriptor;

    Descriptor::NameToPosMap groupMap;
    openvdb::MetaMap metadata;

    { // construct
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        AttributeSet attrSet(descr, /*arrayLength=*/3);
        attrSet.appendAttribute("id", AttributeI::attributeType());
        EXPECT_TRUE(!descr->hasGroup("test1"));
    }

    { // group offset
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());

        descr->setGroup("test1", 1);

        EXPECT_TRUE(descr->hasGroup("test1"));
        EXPECT_EQ(descr->groupMap().at("test1"), size_t(1));

        AttributeSet attrSet(descr);

        EXPECT_EQ(attrSet.groupOffset("test1"), size_t(1));
    }

    { // group index
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSet(descr);

        attrSet.appendAttribute("test", AttributeI::attributeType());
        attrSet.appendAttribute("test2", AttributeI::attributeType());
        attrSet.appendAttribute("group1", GroupAttributeArray::attributeType());
        attrSet.appendAttribute("test3", AttributeI::attributeType());
        attrSet.appendAttribute("group2", GroupAttributeArray::attributeType());
        attrSet.appendAttribute("test4", AttributeI::attributeType());
        attrSet.appendAttribute("group3", GroupAttributeArray::attributeType());

        descr = attrSet.descriptorPtr();

        std::stringstream ss;
        for (int i = 0; i < 17; i++) {
            ss.str("");
            ss << "test" << i;
            descr->setGroup(ss.str(), i);
        }

        Descriptor::GroupIndex index15 = attrSet.groupIndex(15);
        EXPECT_EQ(index15.first, size_t(5));
        EXPECT_EQ(index15.second, uint8_t(7));

        EXPECT_EQ(attrSet.groupOffset(index15), size_t(15));
        EXPECT_EQ(attrSet.groupOffset("test15"), size_t(15));

        Descriptor::GroupIndex index15b = attrSet.groupIndex("test15");
        EXPECT_EQ(index15b.first, size_t(5));
        EXPECT_EQ(index15b.second, uint8_t(7));

        Descriptor::GroupIndex index16 = attrSet.groupIndex(16);
        EXPECT_EQ(index16.first, size_t(7));
        EXPECT_EQ(index16.second, uint8_t(0));

        EXPECT_EQ(attrSet.groupOffset(index16), size_t(16));
        EXPECT_EQ(attrSet.groupOffset("test16"), size_t(16));

        Descriptor::GroupIndex index16b = attrSet.groupIndex("test16");
        EXPECT_EQ(index16b.first, size_t(7));
        EXPECT_EQ(index16b.second, uint8_t(0));

        // check out of range exception

        EXPECT_NO_THROW(attrSet.groupIndex(23));
        EXPECT_THROW(attrSet.groupIndex(24), LookupError);

        // check group attribute indices (group attributes are appended with indices 3, 5, 7)

        std::vector<size_t> groupIndices = attrSet.groupAttributeIndices();

        EXPECT_EQ(size_t(3), groupIndices.size());
        EXPECT_EQ(size_t(3), groupIndices[0]);
        EXPECT_EQ(size_t(5), groupIndices[1]);
        EXPECT_EQ(size_t(7), groupIndices[2]);
    }

    { // group unique name
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        const openvdb::Name uniqueNameEmpty = descr->uniqueGroupName("test");
        EXPECT_EQ(uniqueNameEmpty, openvdb::Name("test"));

        descr->setGroup("test", 1);
        descr->setGroup("test1", 2);

        const openvdb::Name uniqueName1 = descr->uniqueGroupName("test");
        EXPECT_EQ(uniqueName1, openvdb::Name("test0"));
        descr->setGroup(uniqueName1, 3);

        const openvdb::Name uniqueName2 = descr->uniqueGroupName("test");
        EXPECT_EQ(uniqueName2, openvdb::Name("test2"));
    }

    { // group rename
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        descr->setGroup("test", 1);
        descr->setGroup("test1", 2);

        size_t pos = descr->renameGroup("test", "test1");
        EXPECT_TRUE(pos == AttributeSet::INVALID_POS);
        EXPECT_TRUE(descr->hasGroup("test"));
        EXPECT_TRUE(descr->hasGroup("test1"));

        pos = descr->renameGroup("test", "test2");
        EXPECT_EQ(pos, size_t(1));
        EXPECT_TRUE(!descr->hasGroup("test"));
        EXPECT_TRUE(descr->hasGroup("test1"));
        EXPECT_TRUE(descr->hasGroup("test2"));
    }

    // typically 8 bits per group
    EXPECT_EQ(size_t(CHAR_BIT), Descriptor::groupBits());

    { // unused groups and compaction
        AttributeSet attrSet(Descriptor::create(AttributeVec3s::attributeType()));
        attrSet.appendAttribute("group1", GroupAttributeArray::attributeType());
        attrSet.appendAttribute("group2", GroupAttributeArray::attributeType());

        Descriptor& descriptor = attrSet.descriptor();

        Name sourceName;
        size_t sourceOffset, targetOffset;

        // no groups

        EXPECT_EQ(size_t(CHAR_BIT*2), descriptor.unusedGroups());
        EXPECT_EQ(size_t(0), descriptor.unusedGroupOffset());
        EXPECT_EQ(size_t(1), descriptor.unusedGroupOffset(/*hint=*/size_t(1)));
        EXPECT_EQ(size_t(5), descriptor.unusedGroupOffset(/*hint=*/size_t(5)));
        EXPECT_EQ(true, descriptor.canCompactGroups());
        EXPECT_EQ(false,
            descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));

        // add one group in first slot

        descriptor.setGroup("test0", size_t(0));

        EXPECT_EQ(size_t(CHAR_BIT*2-1), descriptor.unusedGroups());
        EXPECT_EQ(size_t(1), descriptor.unusedGroupOffset());
        // hint already in use
        EXPECT_EQ(size_t(1), descriptor.unusedGroupOffset(/*hint=*/size_t(0)));
        EXPECT_EQ(true, descriptor.canCompactGroups());
        EXPECT_EQ(false,
            descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));

        descriptor.dropGroup("test0");

        // add one group in a later slot of the first attribute

        descriptor.setGroup("test7", size_t(7));

        EXPECT_EQ(size_t(CHAR_BIT*2-1), descriptor.unusedGroups());
        EXPECT_EQ(size_t(0), descriptor.unusedGroupOffset());
        EXPECT_EQ(size_t(6), descriptor.unusedGroupOffset(/*hint=*/size_t(6)));
        EXPECT_EQ(size_t(0), descriptor.unusedGroupOffset(/*hint=*/size_t(7)));
        EXPECT_EQ(size_t(8), descriptor.unusedGroupOffset(/*hint=*/size_t(8)));
        EXPECT_EQ(true, descriptor.canCompactGroups());
        // note that requiresGroupMove() is not particularly clever because it
        // blindly recommends moving the group even if it ultimately remains in
        // the same attribute
        EXPECT_EQ(true,
            descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));
        EXPECT_EQ(Name("test7"), sourceName);
        EXPECT_EQ(size_t(7), sourceOffset);
        EXPECT_EQ(size_t(0), targetOffset);

        descriptor.dropGroup("test7");

        // this test assumes CHAR_BIT == 8 for convenience

        if (CHAR_BIT == 8) {

            EXPECT_EQ(size_t(16), descriptor.availableGroups());

            // add all but one group in the first attribute

            descriptor.setGroup("test0", size_t(0));
            descriptor.setGroup("test1", size_t(1));
            descriptor.setGroup("test2", size_t(2));
            descriptor.setGroup("test3", size_t(3));
            descriptor.setGroup("test4", size_t(4));
            descriptor.setGroup("test5", size_t(5));
            descriptor.setGroup("test6", size_t(6));
            // no test7

            EXPECT_EQ(size_t(9), descriptor.unusedGroups());
            EXPECT_EQ(size_t(7), descriptor.unusedGroupOffset());
            EXPECT_EQ(true, descriptor.canCompactGroups());
            EXPECT_EQ(false,
                descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));

            descriptor.setGroup("test7", size_t(7));

            EXPECT_EQ(size_t(8), descriptor.unusedGroups());
            EXPECT_EQ(size_t(8), descriptor.unusedGroupOffset());
            EXPECT_EQ(true, descriptor.canCompactGroups());
            EXPECT_EQ(false,
                descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));

            descriptor.setGroup("test8", size_t(8));

            EXPECT_EQ(size_t(7), descriptor.unusedGroups());
            EXPECT_EQ(size_t(9), descriptor.unusedGroupOffset());
            EXPECT_EQ(false, descriptor.canCompactGroups());
            EXPECT_EQ(false,
                descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));

            // out-of-order
            descriptor.setGroup("test13", size_t(13));

            EXPECT_EQ(size_t(6), descriptor.unusedGroups());
            EXPECT_EQ(size_t(9), descriptor.unusedGroupOffset());
            EXPECT_EQ(false, descriptor.canCompactGroups());
            EXPECT_EQ(true,
                descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));
            EXPECT_EQ(Name("test13"), sourceName);
            EXPECT_EQ(size_t(13), sourceOffset);
            EXPECT_EQ(size_t(9), targetOffset);

            descriptor.setGroup("test9", size_t(9));
            descriptor.setGroup("test10", size_t(10));
            descriptor.setGroup("test11", size_t(11));
            descriptor.setGroup("test12", size_t(12));
            descriptor.setGroup("test14", size_t(14));
            descriptor.setGroup("test15", size_t(15), /*checkValidOffset=*/true);

            // attempt to use an existing group offset
            EXPECT_THROW(descriptor.setGroup("test1000", size_t(15),
                /*checkValidOffset=*/true), RuntimeError);

            EXPECT_EQ(size_t(0), descriptor.unusedGroups());
            EXPECT_EQ(std::numeric_limits<size_t>::max(), descriptor.unusedGroupOffset());
            EXPECT_EQ(false, descriptor.canCompactGroups());
            EXPECT_EQ(false,
                descriptor.requiresGroupMove(sourceName, sourceOffset, targetOffset));

            EXPECT_EQ(size_t(16), descriptor.availableGroups());

            // attempt to use a group offset that is out-of-range
            EXPECT_THROW(descriptor.setGroup("test16", size_t(16),
                /*checkValidOffset=*/true), RuntimeError);
        }
    }

    { // group index collision
        Descriptor descr1;
        Descriptor descr2;

        // no groups - no collisions
        EXPECT_TRUE(!descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(!descr2.groupIndexCollision(descr1));

        descr1.setGroup("test1", 0);

        // only one descriptor has groups - no collision
        EXPECT_TRUE(!descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(!descr2.groupIndexCollision(descr1));

        descr2.setGroup("test1", 0);

        // both descriptors have same group - no collision
        EXPECT_TRUE(!descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(!descr2.groupIndexCollision(descr1));

        descr1.setGroup("test2", 1);
        descr2.setGroup("test2", 2);

        // test2 has different index - collision
        EXPECT_TRUE(descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(descr2.groupIndexCollision(descr1));

        descr2.setGroup("test2", 1);

        // overwrite test2 value to remove collision
        EXPECT_TRUE(!descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(!descr2.groupIndexCollision(descr1));

        // overwrite test1 value to introduce collision
        descr1.setGroup("test1", 4);

        // first index has collision
        EXPECT_TRUE(descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(descr2.groupIndexCollision(descr1));

        // add some additional groups
        descr1.setGroup("test0", 2);
        descr2.setGroup("test0", 2);
        descr1.setGroup("test9", 9);
        descr2.setGroup("test9", 9);

        // first index still has collision
        EXPECT_TRUE(descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(descr2.groupIndexCollision(descr1));

        descr1.setGroup("test1", 0);

        // first index no longer has collision
        EXPECT_TRUE(!descr1.groupIndexCollision(descr2));
        EXPECT_TRUE(!descr2.groupIndexCollision(descr1));
    }
}
