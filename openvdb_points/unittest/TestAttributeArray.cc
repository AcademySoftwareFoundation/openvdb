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
#include <openvdb/Types.h>

#include <openvdb_points/tools/AttributeArray.h>

#include <iostream>
#include <sstream>

class TestAttributeArray: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAttributeArray);
    CPPUNIT_TEST(testAttributeArray);
    CPPUNIT_TEST(testAttributeSet);
    CPPUNIT_TEST_SUITE_END();

    void testAttributeArray();
    void testAttributeSet();
}; // class TestPointDataGrid

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeArray);


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

	typedef openvdb::tools::FixedPointAttributeCodec<uint16_t> Codec;
	typedef openvdb::tools::TypedAttributeArray<double, Codec> AttributeArrayC;

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

		attr.collapse(5);
		CPPUNIT_ASSERT(attr.isUniform());
		CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

		CPPUNIT_ASSERT_EQUAL(attr.get(0), 5);
		CPPUNIT_ASSERT_EQUAL(attr.get(20), 5);

		attr.expand();
		CPPUNIT_ASSERT(!attr.isUniform());
		CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

		for (size_t i = 0; i < count; ++i) {
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

		for (size_t i = 0; i < count; ++i) {
			CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
		}
	}

	{ // IO
		const size_t count = 50;
		AttributeArrayI attrA(count);

		for (size_t i = 0; i < count; ++i) {
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

		for (size_t i = 0; i < count; ++i) {
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
TestAttributeArray::testAttributeSet()
{
	typedef openvdb::tools::AttributeArray AttributeArray;
	typedef openvdb::tools::AttributeSet::Descriptor Descriptor;

	// Define and register some common attribute types
	typedef openvdb::tools::TypedAttributeArray<float> 			AttributeS;
	typedef openvdb::tools::TypedAttributeArray<double> 		AttributeD;
	typedef openvdb::tools::TypedAttributeArray<int> 			AttributeI;
	typedef openvdb::tools::TypedAttributeArray<openvdb::Vec3s> AttributeVec3s;
	typedef openvdb::tools::TypedAttributeArray<openvdb::Vec3d> AttributeVec3d;

	AttributeS::registerType();
	AttributeD::registerType();
	AttributeI::registerType();
	AttributeVec3s::registerType();
	AttributeVec3d::registerType();

	Descriptor::Inserter names;
	names.add("p", AttributeVec3s::attributeType());
	names.add("t", AttributeS::attributeType());

	Descriptor::Ptr descrA = Descriptor::create(names.vec);

	Descriptor::Ptr descrB = Descriptor::create(Descriptor::Inserter()
		.add("p", AttributeVec3s::attributeType())
		.add("t", AttributeS::attributeType())
		.vec);

	CPPUNIT_ASSERT_EQUAL(descrA->size(), descrB->size());

	CPPUNIT_ASSERT(*descrA == *descrB);

	typedef openvdb::tools::AttributeSet AttributeSet;

	AttributeSet attributes(descrA);

	AttributeArray::Ptr pAttr(new AttributeVec3s(50));
	AttributeArray::Ptr tAttr(new AttributeS(50));

	CPPUNIT_ASSERT(attributes.replace(0, pAttr) != AttributeSet::INVALID_POS);
	CPPUNIT_ASSERT(attributes.replace(1, tAttr) != AttributeSet::INVALID_POS);

	CPPUNIT_ASSERT(attributes.replace(0, tAttr) == AttributeSet::INVALID_POS);
}


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
