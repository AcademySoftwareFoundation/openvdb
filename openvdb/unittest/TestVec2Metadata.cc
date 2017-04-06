///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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
#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

class TestVec2Metadata : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestVec2Metadata);
    CPPUNIT_TEST(testVec2i);
    CPPUNIT_TEST(testVec2s);
    CPPUNIT_TEST(testVec2d);
    CPPUNIT_TEST_SUITE_END();

    void testVec2i();
    void testVec2s();
    void testVec2d();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVec2Metadata);

void
TestVec2Metadata::testVec2i()
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec2IMetadata(openvdb::Vec2i(1, 1)));
    Metadata::Ptr m2 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Vec2IMetadata*>(m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Vec2IMetadata*>(m2.get()) != 0);

    CPPUNIT_ASSERT(m->typeName().compare("vec2i") == 0);
    CPPUNIT_ASSERT(m2->typeName().compare("vec2i") == 0);

    Vec2IMetadata *s = dynamic_cast<Vec2IMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2i(1, 1));
    s->value() = openvdb::Vec2i(2, 2);
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2i(2, 2));

    m2->copy(*s);

    s = dynamic_cast<Vec2IMetadata*>(m2.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2i(2, 2));
}

void
TestVec2Metadata::testVec2s()
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec2SMetadata(openvdb::Vec2s(1, 1)));
    Metadata::Ptr m2 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Vec2SMetadata*>(m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Vec2SMetadata*>(m2.get()) != 0);

    CPPUNIT_ASSERT(m->typeName().compare("vec2s") == 0);
    CPPUNIT_ASSERT(m2->typeName().compare("vec2s") == 0);

    Vec2SMetadata *s = dynamic_cast<Vec2SMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2s(1, 1));
    s->value() = openvdb::Vec2s(2, 2);
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2s(2, 2));

    m2->copy(*s);

    s = dynamic_cast<Vec2SMetadata*>(m2.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2s(2, 2));
}

void
TestVec2Metadata::testVec2d()
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec2DMetadata(openvdb::Vec2d(1, 1)));
    Metadata::Ptr m2 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Vec2DMetadata*>(m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Vec2DMetadata*>(m2.get()) != 0);

    CPPUNIT_ASSERT(m->typeName().compare("vec2d") == 0);
    CPPUNIT_ASSERT(m2->typeName().compare("vec2d") == 0);

    Vec2DMetadata *s = dynamic_cast<Vec2DMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2d(1, 1));
    s->value() = openvdb::Vec2d(2, 2);
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2d(2, 2));

    m2->copy(*s);

    s = dynamic_cast<Vec2DMetadata*>(m2.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec2d(2, 2));
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
