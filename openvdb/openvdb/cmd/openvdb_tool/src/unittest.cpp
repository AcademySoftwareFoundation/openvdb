// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <stdio.h>// for std::remove
#include <string>
#include <fstream>

#include "Tool.h"

#include "gtest/gtest.h"

// The fixture for testing class.
class Test_vdb_tool : public ::testing::Test
{
protected:
    Test_vdb_tool() {}

    ~Test_vdb_tool() override {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // produce vector of tokenized c-strings from a single input string
    static std::vector<char*> getArgs(const std::string &line)
    {
      const auto tmp = openvdb::vdb_tool::tokenize(line, " ");
      std::vector<char*> args;
      std::transform(tmp.begin(), tmp.end(), std::back_inserter(args),
      [](const std::string &s){
        char *c = new char[s.size()+1];
        std::strcpy(c, s.c_str());
        return c;
      });
      return args;
    }

}; // Test_vdb_tool

TEST_F(Test_vdb_tool, Util)
{
    {// findMatch
      EXPECT_EQ(2, openvdb::vdb_tool::findMatch("bc", {"abc,a", "ab,c,bc"}));
      EXPECT_EQ(4, openvdb::vdb_tool::findMatch("abc", {"abd", "cba", "ab", "abc"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findMatch("abc", {"abc", "abc ", "ab", "bc"}));
      EXPECT_EQ(2, openvdb::vdb_tool::findMatch("abc", {" abc", "abc", "ab", "abc"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findMatch("o", {"abc,o", "abc", "ab", "abc"}));
      EXPECT_EQ(3, openvdb::vdb_tool::findMatch("j", {"abc,o", "a,b,c", "ab,k,j", "abc,d,a,w"}));
      EXPECT_EQ(4, openvdb::vdb_tool::findMatch("aa", {"abc,o", "a,b,c", "ab,k,j", "abc,d,aa,w"}));
      EXPECT_EQ(2, openvdb::vdb_tool::findMatch("aaa", {"abc,o", "a,aaa,c,aa", "ab,k,j", "abc,d,bb,w"}));
    }
    {// to_lower_case
      EXPECT_EQ(" abc=", openvdb::vdb_tool::to_lower_case(" AbC="));
    }
    {// contains
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", "base"));
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", "base", 5));
      EXPECT_FALSE(openvdb::vdb_tool::contains("path/base.ext", "base", 6));
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", 'b'));
      EXPECT_FALSE(openvdb::vdb_tool::contains("path/base.ext", "bbase"));
    }
    {// getFileBase
      EXPECT_EQ("base", openvdb::vdb_tool::getFileBase("path/base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFileBase("/path/base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFileBase("C:\\path\\base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFileBase("/path/base"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFileBase("base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFileBase("base"));
    }
    {// getFileExt
      EXPECT_EQ("ext", openvdb::vdb_tool::getFileExt("path/file_100.ext"));
      EXPECT_EQ("ext", openvdb::vdb_tool::getFileExt("path/file.100.ext"));
      EXPECT_EQ("e", openvdb::vdb_tool::getFileExt("path/file_100.e"));
      EXPECT_EQ("", openvdb::vdb_tool::getFileExt("path/file_100."));
      EXPECT_EQ("", openvdb::vdb_tool::getFileExt("path/file_100"));
    }
     {// findFileExt
      EXPECT_EQ(0, openvdb::vdb_tool::findFileExt("path/file_002.eXt", {"ext", "abs", "ab"}, false));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.eXt", {"ext", "abs", "ab"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.EXT", {"ext", "ext", "ab"}));
      EXPECT_EQ(3, openvdb::vdb_tool::findFileExt("path/file_002.EXT", {"e",   "ex",  "ext"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.ext", {"ext", "ext", "ab"}));
      EXPECT_EQ(0, openvdb::vdb_tool::findFileExt("path/file_002.ext", {"abc", "efg", "ab"}));
    }
    {// replace
      EXPECT_EQ("base%",  openvdb::vdb_tool::replace("base%", 'i', "1"));
      EXPECT_EQ("base1234",  openvdb::vdb_tool::replace("base%i", 'i', "1234"));
      EXPECT_EQ("base%1", openvdb::vdb_tool::replace("base%1", 'i', "1"));
      EXPECT_EQ("base1",  openvdb::vdb_tool::replace("base%1i", 'i', "1"));
      EXPECT_EQ("path/base_0003.vdb", openvdb::vdb_tool::replace("path/base_%4i.vdb",  'i', "3"));
      EXPECT_EQ("path/base_0003_03.vdb", openvdb::vdb_tool::replace("path/base_%4i_%2i.vdb",  'i', "3"));
      EXPECT_EQ("path/base_0003_%2j.vdb", openvdb::vdb_tool::replace("path/base_%4i_%2j.vdb",  'i', "3"));
      EXPECT_EQ("path/base_20003_02.vdb", openvdb::vdb_tool::replace(openvdb::vdb_tool::replace("path/base_%j%4i_%2j.vdb",  'i', "3"),'j', "2"));
      EXPECT_EQ("path/base_1003.vdb", openvdb::vdb_tool::replace("path/base_%4i.vdb",  'i', "1003"));
      EXPECT_EQ("path/base_3.vdb", openvdb::vdb_tool::replace("path/base_%i.vdb",  'i', "3"));
      EXPECT_EQ("path/base_%4i.vdb", openvdb::vdb_tool::replace("path/base_%4i.vdb", 'j', "3"));
      EXPECT_EQ("path/base_0003.vdb", openvdb::vdb_tool::replace("path/base_0003.vdb", 'i', "3"));

      EXPECT_EQ("f=1.2", openvdb::vdb_tool::replace("f=%i", 'i', "1.2"));
      EXPECT_EQ("f=1.2", openvdb::vdb_tool::replace("f=%3i", 'i', "1.2"));
      EXPECT_EQ("f=01.2", openvdb::vdb_tool::replace("f=%4i", 'i', "1.2"));
      EXPECT_EQ("f=%", openvdb::vdb_tool::replace("f=%", 'i', "1.2"));
      EXPECT_EQ("f=1", openvdb::vdb_tool::replace("f=1", 'i', "1.2"));
      EXPECT_EQ("1", openvdb::vdb_tool::replace("1", 'i', "1.2"));
      EXPECT_EQ("%", openvdb::vdb_tool::replace("%", 'i', "1.2"));
    }

    {// test LoopParam
      struct LoopParam {
        char c; double value, end, stride;
        bool next() { value += stride; return value < end; }
        std::string str() const {return floor(value) == value ? std::to_string(int(value)) : std::to_string(float(value));}
      } p{'i', 0.0, 4.0, 2.0};

      EXPECT_EQ("value=0", openvdb::vdb_tool::replace("value=%i", p.c, p.str()));
      EXPECT_TRUE( p.next() );
      EXPECT_EQ("value=002", openvdb::vdb_tool::replace("value=%3i", p.c, p.str()));
      LoopParam q{'v', 0.1, 4.0, 0.5};
      EXPECT_EQ("value=0.100000", openvdb::vdb_tool::replace("value=%v", q.c, q.str()));
      EXPECT_TRUE( q.next() );
      EXPECT_EQ("value=0.600000", openvdb::vdb_tool::replace("value=%v", q.c, q.str()));
    }

    {// starts_with
      EXPECT_TRUE(openvdb::vdb_tool::starts_with("vfxvfxvfx",  "vfx"));
      EXPECT_FALSE(openvdb::vdb_tool::starts_with("vvfxvfxvfx", "vfx"));
    }

    {// ends_with
      EXPECT_TRUE(openvdb::vdb_tool::ends_with("vfxvfxvfx",  "vfx"));
      EXPECT_TRUE(openvdb::vdb_tool::ends_with("vvfxvfxvfx", "vfx"));
      EXPECT_TRUE(openvdb::vdb_tool::ends_with("file.ext", "ext"));
    }

    {// tokenize
      auto tokens = openvdb::vdb_tool::tokenize("1 2 3-4 5   6");
      EXPECT_EQ(5, tokens.size());
      EXPECT_EQ("1",   tokens[0]);
      EXPECT_EQ("2",   tokens[1]);
      EXPECT_EQ("3-4", tokens[2]);
      EXPECT_EQ("5",   tokens[3]);
      EXPECT_EQ("6",   tokens[4]);
      tokens = openvdb::vdb_tool::tokenize("1 2 3-4 5   6", " -");
      EXPECT_EQ(6, tokens.size());
      EXPECT_EQ("1",   tokens[0]);
      EXPECT_EQ("2",   tokens[1]);
      EXPECT_EQ("3",   tokens[2]);
      EXPECT_EQ("4",   tokens[3]);
      EXPECT_EQ("5",   tokens[4]);
      EXPECT_EQ("6",   tokens[5]);
    }
    {// tokenize vectors
      auto tokens = openvdb::vdb_tool::tokenize("(1,2,3)", ",()");
      EXPECT_EQ(3,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      tokens = openvdb::vdb_tool::tokenize("1,2,3", ",()");
      EXPECT_EQ(3,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      tokens = openvdb::vdb_tool::tokenize("((1,2,3),(4,5,6))", ",()");
      EXPECT_EQ(6,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      EXPECT_EQ("4", tokens[3]);
      EXPECT_EQ("5", tokens[4]);
      EXPECT_EQ("6", tokens[5]);
      tokens = openvdb::vdb_tool::tokenize("[(1,2,3),(4,5,6)]", ",()[]");
      EXPECT_EQ(6,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      EXPECT_EQ("4", tokens[3]);
      EXPECT_EQ("5", tokens[4]);
      EXPECT_EQ("6", tokens[5]);
    }

    {// vectorize
      auto vec = openvdb::vdb_tool::vectorize<float>("[(1.1,2.3,3.4),(4.3,5.6,6.7)]", ",()[]");
      EXPECT_EQ(   6, vec.size());
      EXPECT_EQ(1.1f, vec[0]);
      EXPECT_EQ(2.3f, vec[1]);
      EXPECT_EQ(3.4f, vec[2]);
      EXPECT_EQ(4.3f, vec[3]);
      EXPECT_EQ(5.6f, vec[4]);
      EXPECT_EQ(6.7f, vec[5]);
    }

    {// trim
      EXPECT_EQ("-a-=bs-=", openvdb::vdb_tool::trim(" -a-=bs-= "));
      EXPECT_EQ("a-=bs", openvdb::vdb_tool::trim(" -a-=bs-= ", " =-"));
    }

    {// findArg
      EXPECT_NO_THROW({
        EXPECT_EQ("bar", openvdb::vdb_tool::findArg({"v=foo", "val=bar"}, "val"));
        EXPECT_EQ("", openvdb::vdb_tool::findArg({"v=foo", "val="}, "val"));
      });
      EXPECT_THROW(openvdb::vdb_tool::findArg({"v=foo", "va=bar"}, "val"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::findArg({"v=foo", "val"}, "val"), std::invalid_argument);
    }

    {// is_int
      int i=-1;
      EXPECT_FALSE(openvdb::vdb_tool::is_int("", i));
      EXPECT_EQ(-1, i);
      EXPECT_TRUE(openvdb::vdb_tool::is_int("-5", i));
      EXPECT_EQ(-5, i);
      EXPECT_FALSE(openvdb::vdb_tool::is_int("-6.0", i));
    }

    {// str2int
      EXPECT_NO_THROW({
        EXPECT_EQ( 1, openvdb::vdb_tool::str2int("1"));
        EXPECT_EQ(-5, openvdb::vdb_tool::str2int("-5"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2int("1.0"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2int("1 "),  std::invalid_argument);
    }

    {// is_flt
      float v=-1.0f;
      EXPECT_FALSE(openvdb::vdb_tool::is_flt("", v));
      EXPECT_EQ(-1.0f, v);
      EXPECT_TRUE(openvdb::vdb_tool::is_flt("-5", v));
      EXPECT_EQ(-5.0f, v);
      EXPECT_TRUE(openvdb::vdb_tool::is_flt("-6.0", v));
      EXPECT_EQ(-6.0, v);
      EXPECT_FALSE(openvdb::vdb_tool::is_flt("-7.0f", v));
    }

    {// str2float
      EXPECT_NO_THROW({
        EXPECT_EQ(0.02f, openvdb::vdb_tool::str2float("0.02"));
        EXPECT_EQ( 1.0f, openvdb::vdb_tool::str2float("1"));
        EXPECT_EQ(-5.0f, openvdb::vdb_tool::str2float("-5.0"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2float(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2float("1.0f"), std::invalid_argument);
    }

    {// str2double
      EXPECT_NO_THROW({
        EXPECT_EQ(0.02, openvdb::vdb_tool::str2double("0.02"));
        EXPECT_EQ( 1.0, openvdb::vdb_tool::str2double("1"));
        EXPECT_EQ(-5.0, openvdb::vdb_tool::str2double("-5.0"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2double(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2double("1.0f"), std::invalid_argument);
    }

    {// str2bool
      EXPECT_NO_THROW({
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("1"));
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("true"));
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("TRUE"));
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("TrUe"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("0"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("false"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("FALSE"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("FaLsE"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2bool(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2bool("2"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2bool("t"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2bool("f"), std::invalid_argument);
    }

    {// is_number
      int i=0;
      float v=0;
      EXPECT_FALSE(openvdb::vdb_tool::is_number("", i, v));
      EXPECT_EQ(0, i);
      EXPECT_EQ(1, openvdb::vdb_tool::is_number("-5",i,  v));
      EXPECT_EQ(-5, i);
      EXPECT_EQ(2, openvdb::vdb_tool::is_number("-6.0", i, v));
      EXPECT_EQ(-6.0, v);
      EXPECT_FALSE(openvdb::vdb_tool::is_number("-7.0f", i, v));
    }
}

TEST_F(Test_vdb_tool, getArgs)
{
  const std::vector<char*> args = getArgs("cmd -action option=1.0");
  EXPECT_EQ(3, args.size());
  EXPECT_EQ(0, strcmp("cmd", args[0]));
  EXPECT_EQ(0, strcmp("-action", args[1]));
  EXPECT_EQ(0, strcmp("option=1.0", args[2]));
}

TEST_F(Test_vdb_tool, Geometry)
{
  openvdb::vdb_tool::Geometry geo;
  {// test empty
    EXPECT_TRUE(geo.isEmpty());
    EXPECT_FALSE(geo.isPoints());
    EXPECT_FALSE(geo.isMesh());
  }
  {// test non-empty
    geo.setName("test");
    geo.vtx().emplace_back(1.0f, 2.0f, 3.0f);
    geo.vtx().emplace_back(4.0f, 5.0f, 6.0f);
    geo.vtx().emplace_back(7.0f, 8.0f, 9.0f);
    geo.vtx().emplace_back(10.0f, 11.0f, 12.0f);
    geo.tri().emplace_back(0,1,2);
    geo.tri().emplace_back(1,2,3);
    geo.quad().emplace_back(0,1,2,3);
    EXPECT_FALSE(geo.isEmpty());
    EXPECT_FALSE(geo.isPoints());
    EXPECT_TRUE(geo.isMesh());
    EXPECT_EQ(4, geo.vtxCount());
    EXPECT_EQ(2, geo.triCount());
    EXPECT_EQ(1, geo.quadCount());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo.quad()[0]);
  }
  {// Geometry::Header
    openvdb::vdb_tool::Geometry::Header header(geo);
    EXPECT_EQ(4, header.name);
    EXPECT_EQ(4, header.vtx);
    EXPECT_EQ(2, header.tri);
    EXPECT_EQ(1, header.quad);
  }
  std::string buffer;
  {// test streaming to buffer
    std::ostringstream os(std::ios_base::binary);
    const size_t size = geo.write(os);
    EXPECT_TRUE(size>0);
    buffer = os.str();
    EXPECT_EQ(size, buffer.size());
  }
  {// test streaming from buffer
    std::istringstream is(buffer, std::ios_base::binary);
    openvdb::vdb_tool::Geometry geo2;
    EXPECT_EQ(buffer.size(), geo2.read(is));
    EXPECT_EQ(4, geo2.vtxCount());
    EXPECT_EQ(2, geo2.triCount());
    EXPECT_EQ(1, geo2.quadCount());
    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo.quad()[0]);
  }
  {// write to file
    std::ofstream os("data/test.geo", std::ios_base::binary);
    EXPECT_TRUE(geo.write(os));
  }
  {// read from file
    std::ifstream is("data/test.geo", std::ios_base::binary);
    openvdb::vdb_tool::Geometry geo2;
    EXPECT_TRUE(geo2.read(is));
    EXPECT_EQ(4, geo2.vtxCount());
    EXPECT_EQ(2, geo2.triCount());
    EXPECT_EQ(1, geo2.quadCount());
    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo.quad()[0]);
  }
}

TEST_F(Test_vdb_tool, ToolParser)
{
    using namespace openvdb::vdb_tool;
    int alpha = 0, alpha_sum = 0;
    float beta = 0.0f, beta_sum = 0.0f;

    Parser p({{"alpha", "64"}, {"beta", "4.56"}});
    p.addAction("process_a", "a", "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){alpha = p.getInt("alpha");
                     beta  = p.getFloat("beta");}
               );
    p.addAction("process_b", "b", "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){alpha_sum += p.getInt("alpha");
                     beta_sum  += p.getFloat("beta");}
               );
    p.finalize();
    auto args = getArgs("vdb_tool -quiet -process_a alpha=128 -for v=0.1,0.4,0.1 I=i -b alpha=%i beta=%v -end");
    p.parse(args.size(), args.data());
    EXPECT_EQ(0, alpha);
    EXPECT_EQ(0.0f, beta);
    EXPECT_EQ(0, alpha_sum);
    EXPECT_EQ(0.0f, beta_sum);
    p.run();
    EXPECT_EQ(128, alpha);// defined explicitly
    EXPECT_EQ(4.56f, beta);// default value
    EXPECT_EQ(1 + 2 + 3, alpha_sum);// derived from loop
    EXPECT_EQ(0.1f + 0.2f + 0.3f, beta_sum);// derived from loop
}

TEST_F(Test_vdb_tool, ToolBasic)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));

    char *argv[] = { "vdb_tool", "-quiet", "-sphere", "r=1.1", "-ls2mesh", "-write", "data/sphere.ply", "data/config.txt" };
    const int argc = sizeof(argv)/sizeof(char*);

    EXPECT_NO_THROW({
      Tool vdb_tool(argc, argv);
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));
}

TEST_F(Test_vdb_tool, GlobalCounter)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere_1.ply");
    std::remove("data/config_2.txt");

    EXPECT_FALSE(file_exists("data/sphere_1.ply"));
    EXPECT_FALSE(file_exists("data/config_2.txt"));

    char *argv[] = { "vdb_tool", "-quiet", "-sphere", "r=1.1", "-ls2mesh", "-write", "data/sphere_%G.ply", "data/config_%G.txt" };
    const int argc = sizeof(argv)/sizeof(char*);

    EXPECT_NO_THROW({
      Tool vdb_tool(argc, argv);
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere_1.ply"));
    EXPECT_TRUE(file_exists("data/config_2.txt"));
}


TEST_F(Test_vdb_tool, ToolForLoop)
{
    using namespace openvdb::vdb_tool;

    std::remove("data/config.txt");
    EXPECT_FALSE(file_exists("data/config.txt"));
    for (int i=0; i<4; ++i) {
      const std::string name("data/sphere_"+std::to_string(i)+".ply");
      std::remove(name.c_str());
      EXPECT_FALSE(file_exists(name));
    }

    // test single for-loop
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -for I=j i=0,3,1 -sphere r=1.%i dim=128 name=sphere_%i -ls2mesh -write data/sphere_%j.ply -end");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    for (int i=1; i<4; ++i) EXPECT_TRUE(file_exists("data/sphere_"+std::to_string(i)+".ply"));

    // test two nested for-loops
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -for v=0.1,0.3,0.1 -each s=sphere_1,sphere_3 -read ./data/%s.ply -mesh2ls voxel=%v -end -end -write data/test.vdb");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/test.vdb"));
}

TEST_F(Test_vdb_tool, ToolError)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));

    char *argv[] = { "vdb_tool", "-sphere", "bla=3", "-ls2mesh", "-write", "data/sphere.ply", "data/config.txt", "-quiet" };
    const int argc = sizeof(argv)/sizeof(char*);

    EXPECT_THROW({
      Tool vdb_tool(argc, argv);
      vdb_tool.run();
    }, std::invalid_argument);

    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));
}

TEST_F(Test_vdb_tool, ToolKeep)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.vdb");
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(file_exists("data/sphere.vdb"));
    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));

    char *argv[] = { "vdb_tool", "-quiet", "-default", "keep=1", "-sphere", "r=2", "-ls2mesh", "vdb=0", "-write", "vdb=0", "geo=0", "data/sphere.vdb", "data/sphere.ply", "data/config.txt" };
    const int argc = sizeof(argv)/sizeof(char*);

    EXPECT_NO_THROW({
      Tool vdb_tool(argc, argv);
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere.vdb"));
    EXPECT_TRUE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));
}

TEST_F(Test_vdb_tool, ToolConfig)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.vdb");
    std::remove("data/sphere.ply");

    EXPECT_FALSE(file_exists("data/sphere.vdb"));
    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));

    char *argv[] = { "vdb_tool", "-quiet", "-config", "data/config.txt" };
    const int argc = sizeof(argv)/sizeof(char*);

    EXPECT_NO_THROW({
      Tool vdb_tool(argc, argv);
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere.vdb"));
    EXPECT_TRUE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
