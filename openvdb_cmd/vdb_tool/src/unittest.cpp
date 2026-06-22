// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#define _USE_MATH_DEFINES

#include "Tool.h"
#include "Parser.h"
#include "Util.h"

#include <stdio.h>// for std::remove
#include <string>
#include <fstream>
#include <set>
#include <thread>

#if defined(_WIN32)
#include <direct.h>// for mkdir
int mkdir_wrapper(const char *dirname) { return _mkdir(dirname); }
#else
#include <sys/stat.h>// for mkdir
int mkdir_wrapper(const char *dirname) { return mkdir(dirname, 0777); }
#endif

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
        if (!openvdb::vdb_tool::fileExists("data")) {
            if (mkdir_wrapper("data") == -1) {
                std::cerr << "Test_vdb_tool::SetUp: Failed to create \"data\" directory :  " << strerror(errno) << std::endl;
            } else {
                std::cerr << "Successfully created \"data\" directory\n";
            }
        }
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

    {// findAll
      auto vec = openvdb::vdb_tool::findAll("%1234%678%0123%");
      EXPECT_EQ( 4, vec.size());
      EXPECT_EQ( 0, vec[0]);
      EXPECT_EQ( 5, vec[1]);
      EXPECT_EQ( 9, vec[2]);
      EXPECT_EQ(14, vec[3]);
    }

    {// toLowerCase
      EXPECT_EQ(" abc=", openvdb::vdb_tool::toLowerCase(" AbC="));
    }

    {// toUpperCase
      EXPECT_EQ(" ABC=", openvdb::vdb_tool::toUpperCase(" AbC="));
    }

    {// contains
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", "base"));
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", "base", 5));
      EXPECT_FALSE(openvdb::vdb_tool::contains("path/base.ext", "base", 6));
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", 'b'));
      EXPECT_FALSE(openvdb::vdb_tool::contains("path/base.ext", "bbase"));
    }

    {// getFile
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("path/base.ext"));
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("/path/base.ext"));
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("C:\\path\\base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFile("/path/base"));
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFile("base"));
    }

    {// getBase
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("path/base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("/path/base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("C:\\path\\base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("/path/base"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("base"));
    }

    {// getExt
      EXPECT_EQ("ext", openvdb::vdb_tool::getExt("path/file_100.ext"));
      EXPECT_EQ("ext", openvdb::vdb_tool::getExt("path/file.100.ext"));
      EXPECT_EQ("e", openvdb::vdb_tool::getExt("path/file_100.e"));
      EXPECT_EQ("", openvdb::vdb_tool::getExt("path/file_100."));
      EXPECT_EQ("", openvdb::vdb_tool::getExt("path/file_100"));
    }

    {// replaceExt
      EXPECT_EQ("path/file_100.abc", openvdb::vdb_tool::replaceExt("path/file_100.ext", "abc"));
      EXPECT_EQ("path/file.100.abc", openvdb::vdb_tool::replaceExt("path/file.100.ext", "abc"));
      EXPECT_EQ("path/file_100.abc", openvdb::vdb_tool::replaceExt("path/file_100.e", "abc"));
      EXPECT_EQ("path/file_100.abc", openvdb::vdb_tool::replaceExt("path/file_100.", "abc"));
      EXPECT_EQ("path/file_100.abc", openvdb::vdb_tool::replaceExt("path/file_100", "abc"));
    }

    {// replacePath
      EXPECT_EQ("path/file_100.ext", openvdb::vdb_tool::replacePath("abc/file_100.ext", "path"));
      EXPECT_EQ("/path/file.100.ext", openvdb::vdb_tool::replacePath("/abc/file.100.ext", "/path"));
      EXPECT_EQ("path/file_100.ext", openvdb::vdb_tool::replacePath("foo/bar/file_100.ext", "path"));
      EXPECT_EQ("path/abc/file_100.ext", openvdb::vdb_tool::replacePath("foo/bar/file_100.ext", "path/abc"));
      EXPECT_EQ("/path/abc/file_100.ext", openvdb::vdb_tool::replacePath("/foo/bar/file_100.ext", "/path/abc"));
    }

    {// findFileExt
      EXPECT_EQ(0, openvdb::vdb_tool::findFileExt("path/file_002.eXt", {"ext", "abs", "ab"}, false));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.eXt", {"ext", "abs", "ab"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.EXT", {"ext", "ext", "ab"}));
      EXPECT_EQ(3, openvdb::vdb_tool::findFileExt("path/file_002.EXT", {"e",   "ex",  "ext"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.ext", {"ext", "ext", "ab"}));
      EXPECT_EQ(0, openvdb::vdb_tool::findFileExt("path/file_002.ext", {"abc", "efg", "ab"}));
    }

    {// startsWith
      EXPECT_TRUE(openvdb::vdb_tool::startsWith("vfxvfxvfx",  "vfx"));
      EXPECT_FALSE(openvdb::vdb_tool::startsWith("vvfxvfxvfx", "vfx"));
    }

    {// endsWith
      EXPECT_TRUE(openvdb::vdb_tool::endsWith("vfxvfxvfx",  "vfx"));
      EXPECT_TRUE(openvdb::vdb_tool::endsWith("vvfxvfxvfx", "vfx"));
      EXPECT_TRUE(openvdb::vdb_tool::endsWith("file.ext", "ext"));
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

    {// isInt
      int i=-1;
      EXPECT_FALSE(openvdb::vdb_tool::isInt("", i));
      EXPECT_EQ(-1, i);
      EXPECT_TRUE(openvdb::vdb_tool::isInt("-5", i));
      EXPECT_EQ(-5, i);
      EXPECT_FALSE(openvdb::vdb_tool::isInt("-6.0", i));
    }

    {// strToInt
      EXPECT_NO_THROW({
        EXPECT_EQ( 1, openvdb::vdb_tool::strToInt("1"));
        EXPECT_EQ(-5, openvdb::vdb_tool::strToInt("-5"));
      });
      EXPECT_THROW(openvdb::vdb_tool::strToInt("1.0"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strToInt("1 "),  std::invalid_argument);
    }

    {// isFloat
      float v=-1.0f;
      EXPECT_FALSE(openvdb::vdb_tool::isFloat("", v));
      EXPECT_EQ(-1.0f, v);
      EXPECT_TRUE(openvdb::vdb_tool::isFloat("-5", v));
      EXPECT_EQ(-5.0f, v);
      EXPECT_TRUE(openvdb::vdb_tool::isFloat("-6.0", v));
      EXPECT_EQ(-6.0, v);
      EXPECT_FALSE(openvdb::vdb_tool::isFloat("-7.0f", v));
    }

    {// strToFloat
      EXPECT_NO_THROW({
        EXPECT_EQ(0.02f, openvdb::vdb_tool::strToFloat("0.02"));
        EXPECT_EQ( 1.0f, openvdb::vdb_tool::strToFloat("1"));
        EXPECT_EQ(-5.0f, openvdb::vdb_tool::strToFloat("-5.0"));
      });
      EXPECT_THROW(openvdb::vdb_tool::strToFloat(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strToFloat("1.0f"), std::invalid_argument);
    }

    {// strToDouble
      EXPECT_NO_THROW({
        EXPECT_EQ(0.02, openvdb::vdb_tool::strToDouble("0.02"));
        EXPECT_EQ( 1.0, openvdb::vdb_tool::strToDouble("1"));
        EXPECT_EQ(-5.0, openvdb::vdb_tool::strToDouble("-5.0"));
      });
      EXPECT_THROW(openvdb::vdb_tool::strToDouble(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strToDouble("1.0f"), std::invalid_argument);
    }

    {// strToBool
      EXPECT_NO_THROW({
        EXPECT_TRUE(openvdb::vdb_tool::strToBool("1"));
        EXPECT_TRUE(openvdb::vdb_tool::strToBool("true"));
        EXPECT_TRUE(openvdb::vdb_tool::strToBool("TRUE"));
        EXPECT_TRUE(openvdb::vdb_tool::strToBool("TrUe"));
        EXPECT_FALSE(openvdb::vdb_tool::strToBool("0"));
        EXPECT_FALSE(openvdb::vdb_tool::strToBool("false"));
        EXPECT_FALSE(openvdb::vdb_tool::strToBool("FALSE"));
        EXPECT_FALSE(openvdb::vdb_tool::strToBool("FaLsE"));
      });
      EXPECT_THROW(openvdb::vdb_tool::strToBool(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strToBool("2"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strToBool("t"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strToBool("f"), std::invalid_argument);
    }

    {// strSizeToByteSize
      EXPECT_NO_THROW({
        EXPECT_EQ(uint64_t(2), openvdb::vdb_tool::strSizeToByteSize("2"));
        EXPECT_EQ(uint64_t(2), openvdb::vdb_tool::strSizeToByteSize("2B"));
        EXPECT_EQ(uint64_t(2048), openvdb::vdb_tool::strSizeToByteSize("2KB"));
        EXPECT_EQ(uint64_t(40) << 20, openvdb::vdb_tool::strSizeToByteSize("40MB"));
        EXPECT_EQ(uint64_t(21) << 30, openvdb::vdb_tool::strSizeToByteSize("21GB"));
        EXPECT_EQ(uint64_t(12) << 40, openvdb::vdb_tool::strSizeToByteSize("12TB"));
        EXPECT_EQ(uint64_t(2), openvdb::vdb_tool::strSizeToByteSize(" 2  "));
        EXPECT_EQ(uint64_t(2), openvdb::vdb_tool::strSizeToByteSize(" 2B  "));
        EXPECT_EQ(uint64_t(2048), openvdb::vdb_tool::strSizeToByteSize(" 2KB  "));
        EXPECT_EQ(uint64_t(40) << 20, openvdb::vdb_tool::strSizeToByteSize(" 40MB  "));
        EXPECT_EQ(uint64_t(21) << 30, openvdb::vdb_tool::strSizeToByteSize(" 21GB  "));
        EXPECT_EQ(uint64_t(12) << 40, openvdb::vdb_tool::strSizeToByteSize(" 12TB  "));
      });
      EXPECT_THROW(openvdb::vdb_tool::strSizeToByteSize(""), std::out_of_range);// stoi
      EXPECT_THROW(openvdb::vdb_tool::strSizeToByteSize("2b"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strSizeToByteSize("foo"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strSizeToByteSize("1PB"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strSizeToByteSize(" B"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::strSizeToByteSize("KB1KB"), std::invalid_argument);
    }

    {// isNumber
      int i=0;
      float v=0;
      EXPECT_FALSE(openvdb::vdb_tool::isNumber("", i, v));
      EXPECT_EQ(0, i);
      EXPECT_EQ(1, openvdb::vdb_tool::isNumber("-5",i,  v));
      EXPECT_EQ(-5, i);
      EXPECT_EQ(2, openvdb::vdb_tool::isNumber("-6.0", i, v));
      EXPECT_EQ(-6.0, v);
      EXPECT_FALSE(openvdb::vdb_tool::isNumber("-7.0f", i, v));
    }

    {// uuid: uniqueness across many calls
      std::set<std::string> tmp;
      const size_t size = 10000;
      for (size_t i=0; i<size; ++i) {
        EXPECT_TRUE(tmp.emplace(openvdb::vdb_tool::uuid()).second);
      }
      EXPECT_EQ(size, tmp.size());
    }

    {// uuid: RFC 4122 v4 format conformance
      // "xxxxxxxx-xxxx-4xxx-Nxxx-xxxxxxxxxxxx" — 36 chars, hyphens at fixed
      // positions, version nibble (pos 14) = '4', variant nibble (pos 19) in
      // {8,9,a,b}, every other char a lowercase hex digit.
      auto isHex = [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      };
      for (int i = 0; i < 1000; ++i) {
        const std::string id = openvdb::vdb_tool::uuid();
        ASSERT_EQ(36u, id.size()) << "got: " << id;
        EXPECT_EQ('-', id[8])  << id;
        EXPECT_EQ('-', id[13]) << id;
        EXPECT_EQ('-', id[18]) << id;
        EXPECT_EQ('-', id[23]) << id;
        EXPECT_EQ('4', id[14]) << "version nibble must be 4: " << id;
        const char v = id[19];
        EXPECT_TRUE(v == '8' || v == '9' || v == 'a' || v == 'b')
            << "variant nibble must be 8/9/a/b: " << id;
        for (size_t k = 0; k < id.size(); ++k) {
          if (k == 8 || k == 13 || k == 18 || k == 23) continue;
          EXPECT_TRUE(isHex(id[k]))
              << "non-hex char at index " << k << " in " << id;
        }
      }
    }

    {// uuid: thread-safety — no two threads produce the same UUID and no
     // generated string is malformed under concurrent access. Catches
     // regressions if the thread_local PRNG is ever changed back to a
     // process-wide static.
      constexpr int kThreads = 8;
      constexpr int kPerThread = 2000;
      std::vector<std::vector<std::string>> perThread(kThreads);
      std::vector<std::thread> workers;
      workers.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t]() {
          perThread[t].reserve(kPerThread);
          for (int i = 0; i < kPerThread; ++i) {
            perThread[t].push_back(openvdb::vdb_tool::uuid());
          }
        });
      }
      for (auto& w : workers) w.join();
      std::set<std::string> all;
      for (const auto& bucket : perThread) {
        for (const auto& id : bucket) {
          EXPECT_EQ(36u, id.size()) << id;
          EXPECT_TRUE(all.insert(id).second) << "duplicate across threads: " << id;
        }
      }
      EXPECT_EQ(static_cast<size_t>(kThreads * kPerThread), all.size());
    }

    {//swapBytes
      const int i = 4, j = openvdb::vdb_tool::swapBytes(i);
      EXPECT_NE(i, j);
      EXPECT_EQ(i, openvdb::vdb_tool::swapBytes(j));

      const float a = 4, b = openvdb::vdb_tool::swapBytes(a);
      EXPECT_NE(a, b);
      EXPECT_EQ(a, openvdb::vdb_tool::swapBytes(b));

      const double x = 4, y = openvdb::vdb_tool::swapBytes(x);
      EXPECT_NE(x, y);
      EXPECT_EQ(x, openvdb::vdb_tool::swapBytes(y));

      int vec_i[3]={3,4,5}, vec_j[3];
      for (int n=0; n<3; ++n) {
        vec_j[n] = openvdb::vdb_tool::swapBytes(vec_i[n]);
        EXPECT_NE(vec_i[n], vec_j[n]);
      }
      openvdb::vdb_tool::swapBytes(vec_j, 3);
      for (int n=0; n<3; ++n) EXPECT_EQ(vec_i[n], vec_j[n]);

      float vec_a[3]={3,4,5}, vec_b[3];
      for (int n=0; n<3; ++n) {
        vec_b[n] = openvdb::vdb_tool::swapBytes(vec_a[n]);
        EXPECT_NE(vec_a[n], vec_b[n]);
      }
      openvdb::vdb_tool::swapBytes(vec_b, 3);
      for (int n=0; n<3; ++n) EXPECT_EQ(vec_a[n], vec_b[n]);

      double vec_x[3]={3,4,5}, vec_y[3];
      for (int n=0; n<3; ++n) {
        vec_y[n] = openvdb::vdb_tool::swapBytes(vec_x[n]);
        EXPECT_NE(vec_x[n], vec_y[n]);
      }
      openvdb::vdb_tool::swapBytes(vec_y, 3);
      for (int n=0; n<3; ++n) EXPECT_EQ(vec_x[n], vec_y[n]);
    }
    {// weird pointer behaviour
      float vec[4], *p = vec;
      EXPECT_EQ(vec, p);// of course
      EXPECT_EQ((char*)(vec),  (char*)p);// sure
      EXPECT_EQ((char*)(&vec), (char*)p);// wait, what?!
      EXPECT_NE((char*)(vec),  (char*)(&p));// yep
      EXPECT_NE((char*)(&p),   (char*)p);// of course
    }

    {// fileExists
      using namespace openvdb::vdb_tool;
      EXPECT_TRUE(fileExists("data"));// directory created by SetUp
      EXPECT_FALSE(fileExists("data/no_such_file_xyz_42.bin"));
      const std::string probe = "data/fileExists_probe.tmp";
      { std::ofstream os(probe); os << "x"; }
      EXPECT_TRUE(fileExists(probe));
      std::remove(probe.c_str());
      EXPECT_FALSE(fileExists(probe));
    }

    {// getPath
      using namespace openvdb::vdb_tool;
      EXPECT_EQ("path",     getPath("path/base.ext"));
      EXPECT_EQ("/path",    getPath("/path/base.ext"));
      // Backslash separators in the INPUT are accepted (normalized to '/'
      // before parsing) but the OUTPUT uses the canonical '/' separator
      // because std::filesystem on POSIX does not emit '\\'.
      EXPECT_EQ("C:/path",  getPath("C:\\path\\base.ext"));
      EXPECT_EQ("path/sub", getPath("path/sub/base.ext"));
      EXPECT_EQ(".",        getPath("base.ext"));// no separator → "."
      EXPECT_EQ(".",        getPath("base"));
      EXPECT_EQ(".",        getPath(""));        // empty → "."
      EXPECT_EQ("/",        getPath("/file"));   // root: keep separator
      EXPECT_EQ("/",        getPath("/"));
      EXPECT_EQ("/",        getPath("\\file"));  // backslash normalized to '/'
    }

    {// getName
      using namespace openvdb::vdb_tool;
      EXPECT_EQ("base",     getName("path/base.ext"));
      EXPECT_EQ("base",     getName("/path/base.ext"));
      EXPECT_EQ("base",     getName("C:\\path\\base.ext"));
      EXPECT_EQ("base",     getName("base.ext"));
      EXPECT_EQ("base0123", getName("path/base0123.ext"));
      EXPECT_EQ("base0123", getName("base0123.ext"));
    }

    {// getNumber
      using namespace openvdb::vdb_tool;
      EXPECT_EQ("0123", getNumber("path/base0123.ext"));
      EXPECT_EQ("100",  getNumber("base_100.ext"));
      EXPECT_EQ("0042", getNumber("file_0042.txt"));
    }

    {// in-place toLowerCase / toUpperCase return a reference to the input
      using namespace openvdb::vdb_tool;
      std::string s = "Hello World";
      std::string &refL = toLowerCase(s);
      EXPECT_EQ(&s, &refL);                  // same object
      EXPECT_EQ("hello world", s);
      std::string &refU = toUpperCase(s);
      EXPECT_EQ(&s, &refU);
      EXPECT_EQ("HELLO WORLD", s);
    }

    {// isInt(float)
      using namespace openvdb::vdb_tool;
      EXPECT_TRUE (isInt( 0.0f));
      EXPECT_TRUE (isInt( 1.0f));
      EXPECT_TRUE (isInt(-2.0f));
      EXPECT_FALSE(isInt( 1.5f));
      EXPECT_FALSE(isInt(-0.25f));
    }

    {// strTo<T> dispatcher with explicit template argument
      using namespace openvdb::vdb_tool;
      EXPECT_EQ(42,    strTo<int>("42"));
      EXPECT_EQ(3.14f, strTo<float>("3.14"));
      EXPECT_EQ(3.14,  strTo<double>("3.14"));
      EXPECT_TRUE (strTo<bool>("true"));
      EXPECT_TRUE (strTo<bool>("1"));
      EXPECT_FALSE(strTo<bool>("false"));
      EXPECT_FALSE(strTo<bool>("0"));
      EXPECT_THROW(strTo<int>("nope"),   std::invalid_argument);
      EXPECT_THROW(strTo<float>(""),     std::invalid_argument);
      EXPECT_THROW(strTo<double>("xyz"), std::invalid_argument);
      EXPECT_THROW(strTo<bool>(""),      std::invalid_argument);
    }

    {// vectorize<int>
      using namespace openvdb::vdb_tool;
      auto v = vectorize<int>("1 2 3 -4");
      ASSERT_EQ(4u, v.size());
      EXPECT_EQ( 1, v[0]);
      EXPECT_EQ( 2, v[1]);
      EXPECT_EQ( 3, v[2]);
      EXPECT_EQ(-4, v[3]);
      v = vectorize<int>("1,2,3", " ,");
      ASSERT_EQ(3u, v.size());
      EXPECT_EQ(1, v[0]);
      EXPECT_THROW(vectorize<int>("1 foo 3"), std::invalid_argument);
    }

    {// vectorize<bool>
      using namespace openvdb::vdb_tool;
      auto v = vectorize<bool>("1 0 true false");
      ASSERT_EQ(4u, v.size());
      EXPECT_TRUE (v[0]);
      EXPECT_FALSE(v[1]);
      EXPECT_TRUE (v[2]);
      EXPECT_FALSE(v[3]);
      EXPECT_THROW(vectorize<bool>("1 maybe 0"), std::invalid_argument);
    }

    {// vectorize<std::string>
      using namespace openvdb::vdb_tool;
      auto v = vectorize<std::string>("foo bar baz");
      ASSERT_EQ(3u, v.size());
      EXPECT_EQ("foo", v[0]);
      EXPECT_EQ("bar", v[1]);
      EXPECT_EQ("baz", v[2]);
    }

    {// findIntN: "option=1,3,6" -> {1,3,6}
      using namespace openvdb::vdb_tool;
      auto v = findIntN({"cmd", "rgb=1,3,6"}, "rgb");
      ASSERT_EQ(3u, v.size());
      EXPECT_EQ(1, v[0]);
      EXPECT_EQ(3, v[1]);
      EXPECT_EQ(6, v[2]);
      // space-separated values are also accepted (delimiter set is " ,")
      v = findIntN({"opt=1 2 3"}, "opt");
      ASSERT_EQ(3u, v.size());
      EXPECT_EQ(1, v[0]);
      EXPECT_EQ(2, v[1]);
      EXPECT_EQ(3, v[2]);
      EXPECT_THROW(findIntN({"opt=1,foo,3"}, "opt"), std::invalid_argument);
      EXPECT_THROW(findIntN({"abc=1"}, "missing"),   std::invalid_argument);
    }

    {// findFltN: "option=1.3,-3.1,6.0" -> {1.3f,-3.1f,6.0f}
      using namespace openvdb::vdb_tool;
      auto v = findFltN({"opt=1.3,-3.1,6.0"}, "opt");
      ASSERT_EQ(3u, v.size());
      EXPECT_FLOAT_EQ( 1.3f, v[0]);
      EXPECT_FLOAT_EQ(-3.1f, v[1]);
      EXPECT_FLOAT_EQ( 6.0f, v[2]);
      EXPECT_THROW(findFltN({"opt=1.0,bad"}, "opt"), std::invalid_argument);
      EXPECT_THROW(findFltN({"abc=1.0"}, "missing"), std::invalid_argument);
    }

    {// isLittleEndian: deterministic and cross-checks a byte-inspected uint16_t
      using namespace openvdb::vdb_tool;
      const bool le = isLittleEndian();
      EXPECT_EQ(le, isLittleEndian());// idempotent
      const uint16_t v = 0x0102;
      const bool actuallyLittle = (*reinterpret_cast<const uint8_t*>(&v) == 0x02);
      EXPECT_EQ(actuallyLittle, le);
    }

    {// dateStamp: format "YYYY-MM-DD_HH-MM-SS" (19 chars)
      using namespace openvdb::vdb_tool;
      const std::string s = dateStamp();
      ASSERT_EQ(19u, s.size());
      EXPECT_EQ('-', s[4]);
      EXPECT_EQ('-', s[7]);
      EXPECT_EQ('_', s[10]);
      EXPECT_EQ('-', s[13]);
      EXPECT_EQ('-', s[16]);
      for (size_t i : {0u,1u,2u,3u,5u,6u,8u,9u,11u,12u,14u,15u,17u,18u}) {
        EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(s[i]))) << "char " << i << " = " << s[i];
      }
    }

    {// Spinner: glyph cycles through |/-\ then wraps to |
      using namespace openvdb::vdb_tool;
      std::stringstream ss;
      Spinner spin(ss);
      spin("step1");
      spin("step2");
      spin("step3");
      spin("step4");
      spin("step5");// expected to wrap back to '|'
      const std::string out = ss.str();
      EXPECT_NE(std::string::npos, out.find("step1: |"));
      EXPECT_NE(std::string::npos, out.find("step2: /"));
      EXPECT_NE(std::string::npos, out.find("step3: -"));
      EXPECT_NE(std::string::npos, out.find("step4: \\"));
      EXPECT_NE(std::string::npos, out.find("step5: |"));// wrapped
    }

    {// tokenize edge cases not covered above
      using namespace openvdb::vdb_tool;
      EXPECT_TRUE(tokenize("").empty());          // empty input
      EXPECT_TRUE(tokenize("   ").empty());       // only delimiters
      auto t = tokenize("a  b   c", " ");         // consecutive delimiters → tokens skipped
      ASSERT_EQ(3u, t.size());
      EXPECT_EQ("a", t[0]);
      EXPECT_EQ("b", t[1]);
      EXPECT_EQ("c", t[2]);
    }

    {// startsWith / endsWith edge cases
      using namespace openvdb::vdb_tool;
      EXPECT_TRUE (startsWith("anything", ""));// empty pattern is always a prefix
      EXPECT_TRUE (endsWith  ("anything", ""));// empty pattern is always a suffix
      EXPECT_FALSE(startsWith("ab", "abc"));   // pattern longer than string
      EXPECT_FALSE(endsWith  ("ab", "abc"));
      EXPECT_TRUE (startsWith("abc", "abc"));  // equal
      EXPECT_TRUE (endsWith  ("abc", "abc"));
    }
}// Util

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
    const size_t size = geo.writeGEO(os);
    EXPECT_TRUE(size>0);
    buffer = os.str();
    EXPECT_EQ(size, buffer.size());
  }
  {// test streaming from buffer
    std::istringstream is(buffer, std::ios_base::binary);
    openvdb::vdb_tool::Geometry geo2;
    EXPECT_EQ(buffer.size(), geo2.readGEO(is));
    EXPECT_EQ(4, geo2.vtxCount());
    EXPECT_EQ(2, geo2.triCount());
    EXPECT_EQ(1, geo2.quadCount());
    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo2.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo2.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo2.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo2.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo2.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo2.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo2.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo2.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo2.quad()[0]);
  }
  {// write to file
    std::ofstream os("data/test.geo", std::ios_base::binary);
    EXPECT_TRUE(geo.writeGEO(os));
  }
  {// read from file
    std::ifstream is("data/test.geo", std::ios_base::binary);
    openvdb::vdb_tool::Geometry geo2;
    EXPECT_TRUE(geo2.readGEO(is));
    EXPECT_EQ(4, geo2.vtxCount());
    EXPECT_EQ(2, geo2.triCount());
    EXPECT_EQ(1, geo2.quadCount());
    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo2.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo2.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo2.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo2.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo2.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo2.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo2.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo2.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo2.quad()[0]);
  }
  {// test readOFF and writeOFF
    geo.write("data/test.off");
    openvdb::vdb_tool::Geometry geo2;
    geo2.read("data/test.off");
    EXPECT_EQ(4, geo2.vtxCount());
    EXPECT_EQ(2, geo2.triCount());
    EXPECT_EQ(1, geo2.quadCount());
    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo2.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo2.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo2.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo2.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo2.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo2.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo2.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo2.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo2.quad()[0]);
  }
  #ifdef VDB_TOOL_USE_PDAL
  {// read from PDAL-supported ASCII format file
   // (NOTE: PDAL also supports other formats e.g. LAS, LAZ, E57, Draco, FBX, NumPy, OBJ,…)

   // write a test file
    std::ofstream os("data/test.txt");
    os << "X,Y,Z\n";
    for (size_t i=0; i<geo.vtxCount(); ++i) {
      os << geo.vtx()[i][0] << "," << geo.vtx()[i][1] << "," << geo.vtx()[i][2] << "\n";
    }
    os.close();

    // read the test file
    openvdb::vdb_tool::Geometry geo2;
    geo2.read("data/test.txt");
    EXPECT_EQ(4, geo2.vtxCount());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo2.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo2.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo2.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo2.vtx()[3]);
  }
  #endif

  {// PLY round-trip (binary)
    geo.write("data/test.ply");// binary by default
    openvdb::vdb_tool::Geometry geo2;
    geo2.read("data/test.ply");
    EXPECT_EQ(geo.vtxCount(),  geo2.vtxCount());
    EXPECT_EQ(geo.triCount(),  geo2.triCount());
    EXPECT_EQ(geo.quadCount(), geo2.quadCount());
    for (size_t i = 0; i < geo.vtxCount(); ++i) EXPECT_EQ(geo.vtx()[i], geo2.vtx()[i]);
    for (size_t i = 0; i < geo.triCount(); ++i) EXPECT_EQ(geo.tri()[i], geo2.tri()[i]);
    for (size_t i = 0; i < geo.quadCount(); ++i) EXPECT_EQ(geo.quad()[i], geo2.quad()[i]);
  }

  {// PLY round-trip (ASCII)
    geo.write("data/test_ascii.ply", /*ascii=*/true);
    openvdb::vdb_tool::Geometry geo2;
    geo2.read("data/test_ascii.ply");
    EXPECT_EQ(geo.vtxCount(),  geo2.vtxCount());
    EXPECT_EQ(geo.triCount(),  geo2.triCount());
    EXPECT_EQ(geo.quadCount(), geo2.quadCount());
    for (size_t i = 0; i < geo.vtxCount(); ++i) EXPECT_EQ(geo.vtx()[i], geo2.vtx()[i]);
    for (size_t i = 0; i < geo.triCount(); ++i) EXPECT_EQ(geo.tri()[i], geo2.tri()[i]);
    for (size_t i = 0; i < geo.quadCount(); ++i) EXPECT_EQ(geo.quad()[i], geo2.quad()[i]);
  }

  {// OBJ round-trip
    geo.write("data/test.obj");
    openvdb::vdb_tool::Geometry geo2;
    geo2.read("data/test.obj");
    EXPECT_EQ(geo.vtxCount(),  geo2.vtxCount());
    EXPECT_EQ(geo.triCount(),  geo2.triCount());
    EXPECT_EQ(geo.quadCount(), geo2.quadCount());
    for (size_t i = 0; i < geo.vtxCount(); ++i) EXPECT_EQ(geo.vtx()[i], geo2.vtx()[i]);
  }

  {// STL: writing a mesh containing quads must throw (documented in Geometry.h)
    EXPECT_THROW(geo.writeSTL("data/test_quads.stl"), std::invalid_argument);
  }

  {// STL round-trip after triangulating quads.
   // Note: STL stores per-triangle vertex coordinates rather than vertex indices,
   //       so the round-tripped Geometry's vertex *count* differs from the source.
   //       The triangle count is the invariant we can check.
    openvdb::vdb_tool::Geometry tris;
    tris.vtx()  = geo.vtx();
    tris.tri()  = geo.tri();
    tris.quad() = geo.quad();
    const size_t added = tris.triangulateQuads();
    EXPECT_EQ(0,                       tris.quadCount());
    EXPECT_EQ(geo.triCount() + added,  tris.triCount());
    EXPECT_NO_THROW(tris.write("data/test.stl"));
    openvdb::vdb_tool::Geometry stl;
    EXPECT_NO_THROW(stl.read("data/test.stl"));
    EXPECT_EQ(tris.triCount(), stl.triCount());
    EXPECT_EQ(0,               stl.quadCount());
  }

  {// XYZ read (Geometry has no XYZ writer, so build the file directly)
    {
      std::ofstream os("data/test.xyz");
      for (size_t i = 0; i < geo.vtxCount(); ++i) {
        os << geo.vtx()[i][0] << " " << geo.vtx()[i][1] << " " << geo.vtx()[i][2] << "\n";
      }
    }
    openvdb::vdb_tool::Geometry pts;
    pts.read("data/test.xyz");
    EXPECT_EQ(geo.vtxCount(), pts.vtxCount());
    EXPECT_EQ(0,              pts.triCount());
    EXPECT_EQ(0,              pts.quadCount());
    EXPECT_TRUE(pts.isPoints());
    for (size_t i = 0; i < geo.vtxCount(); ++i) EXPECT_EQ(geo.vtx()[i], pts.vtx()[i]);
  }

  {// triangulateQuads
    openvdb::vdb_tool::Geometry tmp;
    tmp.vtx()  = geo.vtx();
    tmp.tri()  = geo.tri();
    tmp.quad() = geo.quad();
    const size_t origQuad = tmp.quadCount();
    const size_t origTri  = tmp.triCount();
    const size_t origVtx  = tmp.vtxCount();
    const size_t added    = tmp.triangulateQuads();
    EXPECT_EQ(2 * origQuad,                added);  // two triangles per quad
    EXPECT_EQ(0,                           tmp.quadCount());
    EXPECT_EQ(origTri + 2 * origQuad,      tmp.triCount());
    EXPECT_EQ(origVtx,                     tmp.vtxCount()); // vertex list unchanged
    // Re-triangulating a quad-free mesh is a no-op.
    EXPECT_EQ(0, tmp.triangulateQuads());
  }

  {// triangulate (static, planar convex N-gon)
    using openvdb::vdb_tool::Geometry;
    // Triangle: 3 vertices -> 1 triangle
    auto t3 = Geometry::triangulate({0, 1, 2});
    EXPECT_EQ(1u, t3.size());
    EXPECT_EQ(openvdb::Vec3I(0,1,2), t3[0]);
    // Quad: 4 vertices -> 2 triangles (fan from vertex 0)
    auto t4 = Geometry::triangulate({0, 1, 2, 3});
    EXPECT_EQ(2u, t4.size());
    EXPECT_EQ(openvdb::Vec3I(0,1,2), t4[0]);
    EXPECT_EQ(openvdb::Vec3I(0,2,3), t4[1]);
    // Pentagon: 5 vertices -> 3 triangles
    auto t5 = Geometry::triangulate({0, 1, 2, 3, 4});
    EXPECT_EQ(3u, t5.size());
    EXPECT_EQ(openvdb::Vec3I(0,1,2), t5[0]);
    EXPECT_EQ(openvdb::Vec3I(0,2,3), t5[1]);
    EXPECT_EQ(openvdb::Vec3I(0,3,4), t5[2]);
    // Degenerate inputs return an empty triangulation.
    EXPECT_TRUE(Geometry::triangulate({}).empty());
    EXPECT_TRUE(Geometry::triangulate({0}).empty());
    EXPECT_TRUE(Geometry::triangulate({0, 1}).empty());
  }

  {// transform: uniform scale via createLinearTransform(2.0) should double every coordinate.
    openvdb::vdb_tool::Geometry tmp;
    tmp.vtx() = geo.vtx();
    auto xform = openvdb::math::Transform::createLinearTransform(2.0);
    tmp.transform(*xform);
    EXPECT_EQ(geo.vtxCount(), tmp.vtxCount());
    for (size_t i = 0; i < geo.vtxCount(); ++i) {
      EXPECT_EQ(geo.vtx()[i] * 2.0f, tmp.vtx()[i]);
    }
  }

  {// deepCopy: copy must be independent of the source.
    auto copy = geo.deepCopy();
    ASSERT_TRUE(copy != nullptr);
    EXPECT_EQ(geo.vtxCount(),  copy->vtxCount());
    EXPECT_EQ(geo.triCount(),  copy->triCount());
    EXPECT_EQ(geo.quadCount(), copy->quadCount());
    EXPECT_EQ(geo.getName(),   copy->getName());
    for (size_t i = 0; i < geo.vtxCount(); ++i) EXPECT_EQ(geo.vtx()[i], copy->vtx()[i]);
    // Mutating the copy must not affect the source.
    copy->vtx().emplace_back(99.0f, 99.0f, 99.0f);
    EXPECT_EQ(geo.vtxCount() + 1, copy->vtxCount());
    EXPECT_EQ(4u,                 geo.vtxCount());
  }
}// Geometry

#ifdef VDB_TOOL_USE_USD
// Hand-author a minimal USD ASCII (.usda) file containing one Mesh inside an Xform
// (translated by +10 along X) and one Points prim at the root, then read it back via
// the Geometry extension dispatcher. Verifies that:
//   - both Mesh and Points contribute to mVtx,
//   - the Xform on the Mesh's parent is baked into the vertex positions,
//   - triangles and quads are preserved as-is,
//   - the order of prims in the file matches the order of vertices in the output.
TEST_F(Test_vdb_tool, GeometryUSD)
{
    using namespace openvdb::vdb_tool;

    const std::string fileName = "data/test.usda";
    std::remove(fileName.c_str());

    {
        std::ofstream os(fileName);
        os << "#usda 1.0\n"
              "\n"
              "def Xform \"Root\"\n"
              "{\n"
              "    double3 xformOp:translate = (10, 0, 0)\n"
              "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
              "\n"
              "    def Mesh \"M\"\n"
              "    {\n"
              "        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]\n"
              "        int[] faceVertexCounts = [3, 4]\n"
              "        int[] faceVertexIndices = [0, 1, 2, 0, 1, 3, 2]\n"
              "    }\n"
              "}\n"
              "\n"
              "def Points \"P\"\n"
              "{\n"
              "    point3f[] points = [(100, 0, 0), (101, 0, 0), (102, 0, 0)]\n"
              "}\n";
    }
    ASSERT_TRUE(fileExists(fileName));

    Geometry geo;
    EXPECT_NO_THROW(geo.read(fileName));

    // 4 (mesh) + 3 (points) = 7 vertices; 1 triangle and 1 quad from the mesh.
    EXPECT_EQ(7u, geo.vtxCount());
    EXPECT_EQ(1u, geo.triCount());
    EXPECT_EQ(1u, geo.quadCount());
    EXPECT_TRUE(geo.isMesh());

    // Mesh vertices are translated by (+10, 0, 0) from the Xform parent.
    EXPECT_EQ(openvdb::Vec3f(10, 0, 0), geo.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(11, 0, 0), geo.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(10, 1, 0), geo.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(11, 1, 0), geo.vtx()[3]);

    // Points prim has no parent Xform, so its positions are unmodified.
    EXPECT_EQ(openvdb::Vec3f(100, 0, 0), geo.vtx()[4]);
    EXPECT_EQ(openvdb::Vec3f(101, 0, 0), geo.vtx()[5]);
    EXPECT_EQ(openvdb::Vec3f(102, 0, 0), geo.vtx()[6]);

    // Topology indices are relative to the mesh's base (0 here).
    EXPECT_EQ(openvdb::Vec3I(0, 1, 2),    geo.tri()[0]);
    EXPECT_EQ(openvdb::Vec4I(0, 1, 3, 2), geo.quad()[0]);

    std::remove(fileName.c_str());
}// GeometryUSD
#endif// VDB_TOOL_USE_USD

#include "Calculator.h"

TEST_F(Test_vdb_tool, Calculator)
{
    using namespace openvdb::vdb_tool;

    auto approxEq = [](float a, float b, float tol = 1e-5f) {
        return std::fabs(a - b) <= tol * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    };

    {// RPN: identity, literal, simple binary op
      Calculator e;
      e.compile("$x");
      EXPECT_EQ( 0.0f, e.eval(0.0f));
      EXPECT_EQ( 3.5f, e.eval(3.5f));
      EXPECT_EQ(-7.0f, e.eval(-7.0f));

      e.compile("42");
      EXPECT_EQ(42.0f, e.eval());
      EXPECT_EQ(42.0f, e.eval(123.0f));

      e.compile("$x:2:+");
      EXPECT_EQ( 5.0f, e.eval( 3.0f));
      EXPECT_EQ(-1.0f, e.eval(-3.0f));
    }

    {// RPN: nontrivial expression with multiple operations
      Calculator e;
      e.compile("$x:sin:$x:pow2:2:*:+");// sin(x) + 2*x*x
      EXPECT_TRUE(approxEq(std::sin(1.5f) + 2.0f * 1.5f * 1.5f, e.eval(1.5f)));
      EXPECT_TRUE(approxEq(std::sin(0.0f) + 0.0f,               e.eval(0.0f)));
    }

    {// Infix: simple operators with standard precedence
      Calculator e;
      e.compile("2 + 3 * 4");
      EXPECT_EQ(14.0f, e.eval());// times binds tighter

      e.compile("(2 + 3) * 4");
      EXPECT_EQ(20.0f, e.eval());

      e.compile("10 - 4 - 3");// left-associative
      EXPECT_EQ(3.0f, e.eval());

      e.compile("2 ^ 3 ^ 2");// right-associative -> 2^(3^2) = 2^9 = 512
      EXPECT_EQ(512.0f, e.eval());
    }

    {// Infix: variable, named constants, unary minus
      Calculator e;
      e.compile("x");
      EXPECT_EQ(2.5f, e.eval(2.5f));

      e.compile("-x");
      EXPECT_EQ(-2.5f, e.eval(2.5f));

      e.compile("2 * -x");
      EXPECT_EQ(-5.0f, e.eval(2.5f));

      e.compile("pi");
      EXPECT_TRUE(approxEq(static_cast<float>(M_PI), e.eval()));

      e.compile("e");
      EXPECT_TRUE(approxEq(static_cast<float>(M_E), e.eval()));
    }

    {// Infix: function calls (unary and binary)
      Calculator e;
      e.compile("sin(x)");
      EXPECT_TRUE(approxEq(std::sin(0.7f), e.eval(0.7f)));

      e.compile("sqrt(x*x + 1)");
      EXPECT_TRUE(approxEq(std::sqrt(2.0f), e.eval(1.0f)));

      e.compile("pow(x, 3)");
      EXPECT_TRUE(approxEq(8.0f, e.eval(2.0f)));

      e.compile("max(x, 0)");
      EXPECT_EQ( 5.0f, e.eval( 5.0f));
      EXPECT_EQ( 0.0f, e.eval(-2.0f));

      e.compile("min(x, max(0, x*x - 1))");
      // x=0.5 -> max(0, -0.75)=0, min(0.5, 0)=0
      EXPECT_EQ(0.0f, e.eval(0.5f));
      // x=2 -> max(0, 3)=3, min(2, 3)=2
      EXPECT_EQ(2.0f, e.eval(2.0f));
    }

    {// Equivalence: same kernel expressed both ways must produce identical bytecode results
      Calculator rpn, infix;
      rpn.compile  ("$x:sin:$x:pow2:2:*:+");
      infix.compile("sin(x) + 2*x*x");
      for (float t : {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.14159f}) {
          EXPECT_TRUE(approxEq(rpn.eval(t), infix.eval(t)))
              << "mismatch at x=" << t
              << ": rpn=" << rpn.eval(t)
              << " infix=" << infix.eval(t);
      }
    }

    {// Three-argument support: x, y, z in both infix and RPN, bound by
     // initializer-list (since eval(x,y,z) was removed).
      Calculator e;

      // Infix: bare identifiers y and z
      e.compile("x + y + z");
      EXPECT_EQ(6.0f, e.eval({1.0f, 2.0f, 3.0f}));
      EXPECT_EQ(0.0f, e.eval({0.0f, 0.0f, 0.0f}));
      EXPECT_THROW(e.eval(1.0f),     std::invalid_argument);// eval(x) cannot bind y or z

      e.compile("sin(x)*cos(y) + z*z");
      EXPECT_TRUE(approxEq(std::sin(0.7f)*std::cos(1.3f) + 4.0f,
                           e.eval({0.7f, 1.3f, 2.0f})));

      // RPN: $y and $z tokens
      e.compile("$x:$y:+:$z:+");
      EXPECT_EQ(6.0f, e.eval({1.0f, 2.0f, 3.0f}));

      // Same expression, infix vs RPN must agree on (x,y,z) inputs
      Calculator rpn, infix;
      rpn.compile  ("$x:$y:*:$z:+");
      infix.compile("x*y + z");
      for (auto xyz : std::initializer_list<std::array<float,3>>{
              {1.0f, 2.0f, 3.0f}, {-1.5f, 4.0f, 0.5f}, {0.0f, 0.0f, 7.0f}}) {
          EXPECT_TRUE(approxEq(rpn.eval({xyz[0], xyz[1], xyz[2]}),
                               infix.eval({xyz[0], xyz[1], xyz[2]})));
      }
    }

    {// Error paths
      Calculator e;
      EXPECT_THROW(e.compile(""),                  std::invalid_argument); // empty
      EXPECT_THROW(e.compile("foo(x)"),            std::invalid_argument); // unknown function
      EXPECT_THROW(e.compile("$x:notanop"),        std::invalid_argument); // leaves 2 values on stack (notanop is a variable)
      EXPECT_THROW(e.compile("(x + 1"),            std::invalid_argument); // mismatched (
      EXPECT_THROW(e.compile("x + 1)"),            std::invalid_argument); // mismatched )
      EXPECT_THROW(e.compile("$x:+"),              std::invalid_argument); // binary op with one operand
      EXPECT_THROW(e.compile("1:2:3"),             std::invalid_argument); // leaves 3 values on the stack
      EXPECT_THROW(e.compile("@"),                 std::invalid_argument); // bad infix character
    }

    {// Arbitrary variable names: discovered and exposed via variables()
      Calculator c;
      c.compile("foo + bar*2");
      ASSERT_EQ(2u, c.variables().size());
      EXPECT_EQ("foo", c.variables()[0]);   // order of first appearance
      EXPECT_EQ("bar", c.variables()[1]);

      const float values[] = {1.0f, 3.0f};
      EXPECT_EQ(7.0f, c.eval(values));      // 1 + 3*2
      EXPECT_EQ(7.0f, c.eval({1.0f, 3.0f}));// initializer-list form

      // Underscores and digits are valid in identifiers.
      c.compile("_a1 * _b2 + 1");
      ASSERT_EQ(2u, c.variables().size());
      EXPECT_EQ("_a1", c.variables()[0]);
      EXPECT_EQ("_b2", c.variables()[1]);
      EXPECT_EQ(7.0f, c.eval({2.0f, 3.0f}));

      // Repeated references resolve to the same slot.
      c.compile("alpha*alpha + alpha");
      ASSERT_EQ(1u, c.variables().size());
      EXPECT_EQ(12.0f, c.eval({3.0f}));     // 9 + 3
    }

    {// RPN with arbitrary names: '$foo' and 'foo' are equivalent
      Calculator c;
      c.compile("$alpha:$beta:+");
      ASSERT_EQ(2u, c.variables().size());
      EXPECT_EQ(7.0f, c.eval({3.0f, 4.0f}));

      c.compile("alpha:beta:*");
      EXPECT_EQ(12.0f, c.eval({3.0f, 4.0f}));
    }

    {// Constant names (pi, e) are not promoted to variables, with or without '$'
      Calculator c;
      c.compile("pi + e");
      EXPECT_TRUE(c.variables().empty());
      EXPECT_TRUE(approxEq(static_cast<float>(M_PI + M_E), c.eval()));

      c.compile("$pi:$e:+");
      EXPECT_TRUE(c.variables().empty());
      EXPECT_TRUE(approxEq(static_cast<float>(M_PI + M_E), c.eval()));
    }

    {// eval(x) throws on any variable other than `x`
      Calculator c;
      c.compile("foo + 1");
      EXPECT_THROW(c.eval(2.0f),  std::invalid_argument);

      // Matching case still works.
      c.compile("x + 1");
      EXPECT_EQ(3.0f, c.eval(2.0f));
    }

    {// initializer-list eval throws on size mismatch
      Calculator c;
      c.compile("x + y");
      EXPECT_THROW(c.eval({1.0f}),                 std::invalid_argument);
      EXPECT_THROW(c.eval({1.0f, 2.0f, 3.0f}),     std::invalid_argument);
      EXPECT_EQ(3.0f, c.eval({1.0f, 2.0f}));
    }

    {// Persistent memory via evalAndRemember(): the user's `x = y + z` case
      Calculator c;
      c.compile("x = y + z");
      EXPECT_EQ("x", c.resultName());        // trailing LHS captured
      EXPECT_FALSE(c.has("x"));              // nothing remembered yet

      const float result = c.evalAndRemember({{"y", 1.0f}, {"z", 2.0f}});
      EXPECT_EQ(3.0f, result);
      EXPECT_EQ(3.0f, c.get("x"));           // trailing-LHS value
      EXPECT_EQ(1.0f, c.get("y"));           // input
      EXPECT_EQ(2.0f, c.get("z"));           // input
      EXPECT_THROW(c.get("missing"),         std::invalid_argument);

      // Plain final expression: no result name, but inputs are still stored.
      c.compile("y + z");
      EXPECT_EQ("", c.resultName());
      c.evalAndRemember({{"y", 1.0f}, {"z", 2.0f}});
      EXPECT_FALSE(c.has("x"));
      EXPECT_EQ(1.0f, c.get("y"));
      EXPECT_EQ(2.0f, c.get("z"));
    }

    {// Persistent memory exposes intermediate slot values
      Calculator c;
      c.compile("t = x*x; r = sin(t); r + t");
      EXPECT_EQ("", c.resultName());         // final stmt has no LHS

      const float ref = std::sin(4.0f) + 4.0f;
      EXPECT_TRUE(approxEq(ref, c.evalAndRemember({{"x", 2.0f}})));
      EXPECT_EQ(2.0f, c.get("x"));           // input
      EXPECT_EQ(4.0f, c.get("t"));           // intermediate slot
      EXPECT_TRUE(approxEq(std::sin(4.0f), c.get("r"))); // intermediate slot

      // memory() returns the whole map.
      EXPECT_EQ(3u, c.memory().size());

      // compile() clears stale memory.
      c.compile("y + 1");
      EXPECT_TRUE(c.memory().empty());
      EXPECT_FALSE(c.has("t"));
    }

    {// Positional evalAndRemember + missing-binding error
      Calculator c;
      c.compile("a = x*x; a + 1");
      const float values[] = {3.0f};
      EXPECT_EQ(10.0f, c.evalAndRemember(values));
      EXPECT_EQ(3.0f, c.get("x"));
      EXPECT_EQ(9.0f, c.get("a"));

      // Missing binding via the map form.
      EXPECT_THROW(c.evalAndRemember(std::unordered_map<std::string, float>{}),
                   std::invalid_argument);

      // eval() (const, no memory) does not populate mMemory.
      c.compile("y + 1");
      c.eval({{"y", 1.0f}});                 // const map overload
      EXPECT_FALSE(c.has("y"));              // still empty
    }

    {// By-name lookup: variableIndex and the map-based eval overload
      Calculator c;
      c.compile("foo + bar*2");
      EXPECT_EQ(0, c.variableIndex("foo"));
      EXPECT_EQ(1, c.variableIndex("bar"));
      EXPECT_EQ(-1, c.variableIndex("missing"));

      // Cached-index pattern: index lookup once, fill values per call.
      const int foo_i = c.variableIndex("foo");
      const int bar_i = c.variableIndex("bar");
      float values[2];
      values[foo_i] = 1.0f;
      values[bar_i] = 3.0f;
      EXPECT_EQ(7.0f, c.eval(values));

      // Map-based eval: convenient but slower.
      EXPECT_EQ(7.0f, c.eval(std::unordered_map<std::string, float>{
                                {"foo", 1.0f}, {"bar", 3.0f}}));
      // Missing binding throws.
      EXPECT_THROW(c.eval(std::unordered_map<std::string, float>{
                            {"foo", 1.0f}}), std::invalid_argument);
      // Extra entries are ignored.
      EXPECT_EQ(7.0f, c.eval(std::unordered_map<std::string, float>{
                                {"foo", 1.0f}, {"bar", 3.0f}, {"extra", 99.0f}}));
    }

    {// Single assignment: the LHS of the final statement is documentation
      Calculator c;
      c.compile("x = y + z");
      ASSERT_EQ(2u, c.variables().size());   // only RHS names are inputs
      EXPECT_EQ("y", c.variables()[0]);
      EXPECT_EQ("z", c.variables()[1]);
      EXPECT_EQ(5.0f, c.eval({2.0f, 3.0f}));

      // The LHS may also name an input; semantically still just documentation.
      c.compile("x = x*x + 1");
      ASSERT_EQ(1u, c.variables().size());
      EXPECT_EQ("x", c.variables()[0]);
      EXPECT_EQ(5.0f, c.eval(2.0f));
    }

    {// Multi-statement: intermediate assignments allocate local slots
      Calculator c;
      c.compile("t = x*x; t + sin(t)");
      ASSERT_EQ(1u, c.variables().size());   // 't' is a slot, not an input
      EXPECT_EQ("x", c.variables()[0]);
      const float ref = 4.0f + std::sin(4.0f);
      EXPECT_TRUE(approxEq(ref, c.eval(2.0f)));

      // Slot reassignment reuses the same storage.
      c.compile("t = 1; t = t + 2; t = t*3; t");
      EXPECT_TRUE(c.variables().empty());
      EXPECT_EQ(9.0f, c.eval());         // ((1+2)*3) = 9; x ignored

      // Slots can shadow input names — after `x = x*2`, subsequent reads of
      // `x` read the slot, not the input. Final statement returns slot.
      c.compile("x = x*2; x + 1");
      ASSERT_EQ(1u, c.variables().size());
      EXPECT_EQ("x", c.variables()[0]);
      EXPECT_EQ(11.0f, c.eval(5.0f));        // (5*2)+1
    }

    {// Multi-statement: trailing semicolons and whitespace are tolerated
      Calculator c;
      EXPECT_NO_THROW(c.compile("t = x*x;  ;  t + 1;"));
      EXPECT_EQ(5.0f, c.eval(2.0f));         // 4 + 1
    }

    {// Extended operators: modulo, comparisons, logical, ternary if()
      Calculator c;

      // Modulo, both punctuation and word
      c.compile("7 % 3");        EXPECT_EQ(1.0f, c.eval());
      c.compile("$x:3:mod");     EXPECT_EQ(1.0f, c.eval(7.0f));
      c.compile("fmod(10, 3)");  EXPECT_EQ(1.0f, c.eval());

      // Comparisons return 1.0 / 0.0
      c.compile("1 < 2");        EXPECT_EQ(1.0f, c.eval());
      c.compile("2 < 1");        EXPECT_EQ(0.0f, c.eval());
      c.compile("1 <= 1");       EXPECT_EQ(1.0f, c.eval());
      c.compile("2 >= 3");       EXPECT_EQ(0.0f, c.eval());
      c.compile("3 == 3");       EXPECT_EQ(1.0f, c.eval());
      c.compile("3 != 3");       EXPECT_EQ(0.0f, c.eval());

      // Logical
      c.compile("1 && 1");       EXPECT_EQ(1.0f, c.eval());
      c.compile("1 && 0");       EXPECT_EQ(0.0f, c.eval());
      c.compile("0 || 0");       EXPECT_EQ(0.0f, c.eval());
      c.compile("0 || 1");       EXPECT_EQ(1.0f, c.eval());
      c.compile("!0");           EXPECT_EQ(1.0f, c.eval());
      c.compile("!1");           EXPECT_EQ(0.0f, c.eval());

      // Ternary via if() / select()
      c.compile("if(1, 10, 20)");      EXPECT_EQ(10.0f, c.eval());
      c.compile("if(0, 10, 20)");      EXPECT_EQ(20.0f, c.eval());
      c.compile("select(1, 10, 20)");  EXPECT_EQ(10.0f, c.eval());

      // Combined inside a larger expression
      c.compile("if(x > 0, sqrt(x), 0)");
      EXPECT_EQ(0.0f, c.eval(-4.0f));
      EXPECT_EQ(2.0f, c.eval( 4.0f));
    }

    {// switch(selector, k1, v1, ..., kN, vN, default)
      Calculator c;

      // Basic case-match
      c.compile("switch(1, 0, 100, 1, 200, 2, 300, 999)");
      EXPECT_EQ(200.0f, c.eval());
      c.compile("switch(2, 0, 100, 1, 200, 2, 300, 999)");
      EXPECT_EQ(300.0f, c.eval());

      // No match → default
      c.compile("switch(5, 0, 100, 1, 200, 2, 300, 999)");
      EXPECT_EQ(999.0f, c.eval());

      // First match wins (k=1 appears twice)
      c.compile("switch(1, 1, 10, 1, 20, 99)");
      EXPECT_EQ(10.0f, c.eval());

      // Minimal: 1 case + default
      c.compile("switch(7, 7, 42, 99)");
      EXPECT_EQ(42.0f, c.eval());
      c.compile("switch(8, 7, 42, 99)");
      EXPECT_EQ(99.0f, c.eval());

      // Selector is a variable
      c.compile("switch(x, 0, 10, 1, 20, 2, 30, -1)");
      EXPECT_EQ(10.0f, c.eval(0.0f));
      EXPECT_EQ(20.0f, c.eval(1.0f));
      EXPECT_EQ(30.0f, c.eval(2.0f));
      EXPECT_EQ(-1.0f, c.eval(7.0f));

      // Case-value can also be a variable: switch(0, 0, x, ...) returns x
      c.compile("switch(0, 0, x, 999)");
      EXPECT_EQ(3.14f, c.eval(3.14f));

      // Arity errors: 3 args (too few) and 5 args (odd)
      EXPECT_THROW(c.compile("switch(1, 0, 100)"),       std::invalid_argument);
      EXPECT_THROW(c.compile("switch(1, 0, 100, 200, 300)"), std::invalid_argument);
    }

    {// Extended functions: hyperbolic, atan2, hypot, step, clamp, mix, smoothstep, sign, round, trunc
      Calculator c;

      c.compile("sinh(0)");        EXPECT_EQ(0.0f, c.eval());
      c.compile("cosh(0)");        EXPECT_EQ(1.0f, c.eval());
      c.compile("tanh(0)");        EXPECT_EQ(0.0f, c.eval());
      c.compile("asinh(0)");       EXPECT_EQ(0.0f, c.eval());
      c.compile("acosh(1)");       EXPECT_EQ(0.0f, c.eval());
      c.compile("atanh(0)");       EXPECT_EQ(0.0f, c.eval());

      c.compile("atan2(1, 1)");
      EXPECT_TRUE(approxEq(static_cast<float>(M_PI / 4.0), c.eval()));
      c.compile("hypot(3, 4)");    EXPECT_EQ(5.0f, c.eval());

      c.compile("step(1, 0.5)");   EXPECT_EQ(0.0f, c.eval());// x < edge → 0
      c.compile("step(1, 1.5)");   EXPECT_EQ(1.0f, c.eval());// x >= edge → 1

      c.compile("clamp(5, 0, 3)"); EXPECT_EQ(3.0f, c.eval());
      c.compile("clamp(-1, 0, 3)");EXPECT_EQ(0.0f, c.eval());
      c.compile("clamp(2, 0, 3)"); EXPECT_EQ(2.0f, c.eval());

      c.compile("mix(0, 10, 0.5)");   EXPECT_EQ(5.0f, c.eval());
      c.compile("lerp(0, 10, 0.25)"); EXPECT_EQ(2.5f, c.eval());

      c.compile("smoothstep(0, 1, 0.5)"); EXPECT_EQ(0.5f, c.eval());
      c.compile("smoothstep(0, 1, -1)");  EXPECT_EQ(0.0f, c.eval());
      c.compile("smoothstep(0, 1, 2)");   EXPECT_EQ(1.0f, c.eval());

      c.compile("sign(0)");        EXPECT_EQ(0.0f, c.eval());
      c.compile("sign(-3.5)");     EXPECT_EQ(-1.0f, c.eval());
      c.compile("sign(7)");        EXPECT_EQ(1.0f, c.eval());

      c.compile("round(2.4)");     EXPECT_EQ(2.0f, c.eval());
      c.compile("round(2.6)");     EXPECT_EQ(3.0f, c.eval());
      c.compile("trunc(2.9)");     EXPECT_EQ(2.0f, c.eval());
      c.compile("trunc(-2.9)");    EXPECT_EQ(-2.0f, c.eval());
    }

    {// No-arg eval() for constant expressions
      Calculator c;
      c.compile("1+2+3");
      EXPECT_TRUE(c.variables().empty());
      EXPECT_EQ(6.0f, c.eval());            // no dummy argument needed

      c.compile("sin(pi/4)*2");
      EXPECT_TRUE(c.variables().empty());
      EXPECT_TRUE(approxEq(std::sin(static_cast<float>(M_PI)/4.0f)*2.0f, c.eval()));

      // Throws when the expression actually references a variable.
      c.compile("x + 1");
      EXPECT_THROW(c.eval(), std::invalid_argument);
    }

    {// New named constants
      Calculator c;
      c.compile("tau");
      EXPECT_TRUE(approxEq(static_cast<float>(2.0 * M_PI), c.eval()));
      c.compile("phi");
      EXPECT_TRUE(approxEq(1.6180339887498949f, c.eval()));
      c.compile("inf");
      EXPECT_TRUE(std::isinf(c.eval()));
      c.compile("nan");
      EXPECT_TRUE(std::isnan(c.eval()));

      // Assignment to constants is rejected for the new constants too.
      EXPECT_THROW(c.compile("tau = 1; tau"), std::invalid_argument);
      EXPECT_THROW(c.compile("inf = 0; inf"), std::invalid_argument);
    }

    {// Operator precedence: comparisons bind tighter than logical, looser than arithmetic
      Calculator c;
      c.compile("1 + 2 == 3 && 1");       EXPECT_EQ(1.0f, c.eval());// (1+2)==3 && 1 → 1
      c.compile("1 < 2 && 3 > 2");        EXPECT_EQ(1.0f, c.eval());
      c.compile("x > 0 && x < 10");
      EXPECT_EQ(1.0f, c.eval(5.0f));
      EXPECT_EQ(0.0f, c.eval(-1.0f));
      EXPECT_EQ(0.0f, c.eval(15.0f));
    }

    {// Lazy if(): only the taken branch is evaluated. Without lazy semantics,
     // these expressions would either crash or produce inf/nan.
      Calculator c;
      c.compile("if(1, 42, 1/0)");
      EXPECT_EQ(42.0f, c.eval());      // 1/0 in non-taken branch is skipped

      c.compile("if(0, 1/0, 99)");
      EXPECT_EQ(99.0f, c.eval());      // again the bad branch is skipped

      // Signed square root: avoids sqrt of a negative number when x>=0.
      c.compile("if(x>=0, sqrt(x), -sqrt(-x))");
      EXPECT_EQ(3.0f, c.eval(9.0f));
      EXPECT_EQ(-3.0f, c.eval(-9.0f));

      // Nested: only one inner branch survives.
      c.compile("if(1, if(0, 100, 200), 300)");
      EXPECT_EQ(200.0f, c.eval());
      c.compile("if(0, if(1, 100, 200), 300)");
      EXPECT_EQ(300.0f, c.eval());
    }

    {// Constant folding: the bytecode for "1+2+3" should collapse to a
     // single PushLit.
      Calculator c;
      c.compile("1+2+3");
      EXPECT_EQ(1u, c.size());
      EXPECT_EQ(6.0f, c.eval());

      c.compile("2*pi");
      EXPECT_EQ(1u, c.size());
      EXPECT_TRUE(approxEq(static_cast<float>(2.0 * M_PI), c.eval()));

      c.compile("sqrt(16) + abs(-3)");
      EXPECT_EQ(1u, c.size());
      EXPECT_EQ(7.0f, c.eval());

      // Variable references prevent folding for the entire expression.
      c.compile("x + 1");
      EXPECT_EQ(3u, c.size());// PushVar x, PushLit 1, Add
      EXPECT_EQ(3.0f, c.eval(2.0f));
    }

    {// disassemble(): just confirm it returns a non-empty string with the
     // expected opcode names. Strict pattern matching would be brittle.
      Calculator c;
      c.compile("x*x + 1");
      const std::string out = c.disassemble();
      EXPECT_TRUE(out.find("PushVar") != std::string::npos);
      EXPECT_TRUE(out.find("Mul")     != std::string::npos);
      EXPECT_TRUE(out.find("Add")     != std::string::npos);
      EXPECT_TRUE(out.find("PushLit") != std::string::npos);
    }

    {// Batched eval_n
      Calculator c;
      c.compile("sin(x) + 1");
      constexpr size_t N = 5;
      const float in[N]  = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
      float out[N] = {};
      c.eval_n(in, out, N);
      for (size_t i = 0; i < N; ++i) {
          EXPECT_TRUE(approxEq(std::sin(in[i]) + 1.0f, out[i]));
      }

      // Custom variable name.
      c.compile("v*v");
      const float in2[3] = {2.0f, 3.0f, 4.0f};
      float out2[3] = {};
      c.eval_n(in2, out2, 3, "v");
      EXPECT_EQ(4.0f,  out2[0]);
      EXPECT_EQ(9.0f,  out2[1]);
      EXPECT_EQ(16.0f, out2[2]);

      // Constant expression: broadcast.
      c.compile("42");
      float out3[3] = {};
      c.eval_n(nullptr, out3, 3);
      EXPECT_EQ(42.0f, out3[0]);
      EXPECT_EQ(42.0f, out3[1]);
      EXPECT_EQ(42.0f, out3[2]);

      // Wrong variable name throws.
      c.compile("x + 1");
      float dummy[1];
      EXPECT_THROW(c.eval_n(in, dummy, 1, "y"), std::invalid_argument);
    }

    {// Column-aware error messages include "column N" and a caret line.
      Calculator c;
      try {
          c.compile("1 + @ + 2");
          FAIL() << "expected throw on '@'";
      } catch (const std::exception &e) {
          const std::string what = e.what();
          EXPECT_TRUE(what.find("column") != std::string::npos)
              << "error missing column info: " << what;
          EXPECT_TRUE(what.find("^") != std::string::npos)
              << "error missing caret marker: " << what;
      }
    }

    {// User-defined functions (def name(p1, p2, ...) = body)
      Calculator c;

      // Single-argument
      c.compile("def sq(x) = x*x; sq(3)");
      EXPECT_EQ(9.0f, c.eval());

      // Multiple calls
      c.compile("def sq(x) = x*x; sq(3) + sq(4)");
      EXPECT_EQ(25.0f, c.eval());

      // Two parameters
      c.compile("def hyp(a, b) = sqrt(a*a + b*b); hyp(3, 4)");
      EXPECT_EQ(5.0f, c.eval());

      // Composition: cube via square
      c.compile("def sq(x) = x*x; def cu(x) = x*sq(x); cu(3)");
      EXPECT_EQ(27.0f, c.eval());

      // Lazy if inside a UDF body still lazy:
      c.compile("def safe_inv(x) = if(x==0, 0, 1/x); safe_inv(0) + safe_inv(2)");
      EXPECT_EQ(0.5f, c.eval());

      // Inputs flow through: def's parameter is independent of the caller's name.
      // Here the def has parameter `t`; the caller passes its `x` as the argument.
      c.compile("def square(t) = t*t; square(x) + x");
      EXPECT_EQ(20.0f, c.eval(4.0f));// 16 + 4

      // Error: free variable in def body.
      EXPECT_THROW(c.compile("def f(x) = x + y; f(1)"),
                   std::invalid_argument);

      // Error: recursion (forward reference to self).
      EXPECT_THROW(c.compile("def f(x) = f(x-1); f(3)"),
                   std::invalid_argument);

      // Error: wrong argument count.
      EXPECT_THROW(c.compile("def f(x, y) = x+y; f(1)"),
                   std::invalid_argument);
      EXPECT_THROW(c.compile("def f(x) = x; f(1, 2)"),
                   std::invalid_argument);

      // Error: def cannot be the final statement.
      EXPECT_THROW(c.compile("def f(x) = x*x"),
                   std::invalid_argument);

      // Error: cannot redefine built-in.
      EXPECT_THROW(c.compile("def sin(x) = x; sin(0)"),
                   std::invalid_argument);

      // Error: duplicate def.
      EXPECT_THROW(c.compile("def f(x) = x; def f(y) = y+1; f(2)"),
                   std::invalid_argument);
    }

    {// Multi-statement error paths
      Calculator c;
      // Intermediate plain expression strands a value.
      EXPECT_THROW(c.compile("x + 1; x + 2"),        std::invalid_argument);
      // Invalid LHS identifier.
      EXPECT_THROW(c.compile("1+1 = x"),             std::invalid_argument);
      // Assignment to a reserved constant name.
      EXPECT_THROW(c.compile("pi = 3.14; pi"),       std::invalid_argument);
      EXPECT_THROW(c.compile("e = 1; e"),            std::invalid_argument);
      // Empty RHS.
      EXPECT_THROW(c.compile("t = ; t"),             std::invalid_argument);
      // Chained '=' is not supported (no '==').
      EXPECT_THROW(c.compile("a = b = 1"),           std::invalid_argument);
      // Mixing assignment with RPN markers is rejected.
      EXPECT_THROW(c.compile("t = $x; t"),           std::invalid_argument);
      EXPECT_THROW(c.compile("t = x ; $x"),          std::invalid_argument);
    }

    {// Parallel sanity check: a Calculator is re-entrant; many threads can call eval()
     // concurrently on the same instance via tools::foreach.
      Calculator expr;
      expr.compile("sin(x) + 2*x*x");

      // Build a small float grid with active values 0..N-1, run the kernel
      // through the same path Tool::forValues uses, and verify each cell.
      openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
      auto acc = grid->getAccessor();
      constexpr int N = 4096;
      for (int i = 0; i < N; ++i) acc.setValue(openvdb::Coord(i, 0, 0), float(i));

      openvdb::tools::foreach(grid->beginValueOn(),
          [&expr](const openvdb::FloatGrid::ValueOnIter &it) {
              it.setValue(expr.eval(*it));
          });

      for (int i = 0; i < N; ++i) {
          const float v   = float(i);
          const float ref = std::sin(v) + 2.0f * v * v;
          EXPECT_TRUE(approxEq(ref, acc.getValue(openvdb::Coord(i, 0, 0)), 1e-3f))
              << "i=" << i;
      }
    }
}// Calculator

TEST_F(Test_vdb_tool, ActionCalc)
{
    using namespace openvdb::vdb_tool;

    // Run @a cmd through a fresh Parser. Returns whatever the action wrote
    // to std::clog so tests can assert on the user-visible echo. State on
    // p (notably p.processor.memory()) is preserved for post-run inspection.
    auto runCapture = [](Parser &p, const std::string &cmd) -> std::string {
        auto args = getArgs(cmd);
        std::ostringstream oss;
        auto *old = std::clog.rdbuf(oss.rdbuf());
        try {
            p.parse(int(args.size()), args.data());
            p.run();
        } catch (...) {
            std::clog.rdbuf(old);
            throw;
        }
        std::clog.rdbuf(old);
        return oss.str();
    };

    // Note: getArgs() tokenizes on whitespace only and does not strip shell
    // quotes, so kernel values are passed bare (no surrounding apostrophes).

    {// 1. Plain expression: trailing statement has no LHS, so the result
     //    is echoed and nothing meaningful lands in the Processor's memory.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet -calc kernel=1+2+3");
      EXPECT_NE(std::string::npos, out.find("6")) << "expected '6' in: " << out;
    }

    {// 2. Single assignment: trailing LHS is silent on -calc, but the value
     //    is written to memory under that name.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet -calc kernel=x=1+2");
      EXPECT_EQ("", out);// silent on trailing assignment
      ASSERT_TRUE(p.processor.memory().isSet("x"));
      EXPECT_EQ(3.0f, strTo<float>(p.processor.memory().get("x")));
    }

    {// 3. Multi-statement: intermediate slot and trailing LHS both land in
     //    memory; the action remains silent because the final stmt assigns.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet -calc kernel=a=1+2;b=a*3");
      EXPECT_EQ("", out);
      ASSERT_TRUE(p.processor.memory().isSet("a"));
      ASSERT_TRUE(p.processor.memory().isSet("b"));
      EXPECT_EQ(3.0f, strTo<float>(p.processor.memory().get("a")));
      EXPECT_EQ(9.0f, strTo<float>(p.processor.memory().get("b")));
    }

    {// 4. Reading values seeded by a prior -eval, with a plain trailing
     //    statement so the result is echoed.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet -eval {2:@x} -calc kernel=3*x+1");
      EXPECT_NE(std::string::npos, out.find("7")) << "expected '7' in: " << out;
    }

    {// 5. Chained -calc: the second kernel reads what the first wrote.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p,
          "vdb_tool -quiet -calc kernel=a=4 -calc kernel=b=a+1");
      EXPECT_EQ("", out);// both kernels end in assignments
      EXPECT_EQ(4.0f, strTo<float>(p.processor.memory().get("a")));
      EXPECT_EQ(5.0f, strTo<float>(p.processor.memory().get("b")));
    }

    {// 6. Referencing an undefined variable throws at run time with a
     //    message naming the offending identifier.
      Parser p({});
      p.finalize();
      auto args = getArgs("vdb_tool -quiet -calc kernel=undef+1");
      p.parse(int(args.size()), args.data());
      try {
          p.run();
          FAIL() << "expected -calc to throw on undefined variable";
      } catch (const std::exception &e) {
          const std::string what = e.what();
          EXPECT_NE(std::string::npos, what.find("undef"))
              << "error did not name the offending variable: " << what;
      }
    }

    {// 7a. Bare positional kernel: greedy-anonymous parsing accepts
     //     "-calc x=1+2" alongside "-calc kernel=x=1+2", because -calc's
     //     anonymous option is registered with the greedy flag.
      Parser p({});
      p.finalize();
      runCapture(p, "vdb_tool -quiet -calc x=1+2");
      ASSERT_TRUE(p.processor.memory().isSet("x"));
      EXPECT_EQ(3.0f, strTo<float>(p.processor.memory().get("x")));

      // Plain expression as a bare positional still echoes the result.
      Parser p2({});
      p2.finalize();
      const std::string out = runCapture(p2, "vdb_tool -quiet -calc 1+2+3");
      EXPECT_NE(std::string::npos, out.find("6")) << "expected '6' in: " << out;
    }

    {// 7b. Reading an input doesn't rewrite it in memory. The float-formatted
     //     "0.000000" rewrite previously broke downstream int comparators
     //     (e.g. -for / -if integer ops).
      Parser p({});
      p.finalize();
      runCapture(p, "vdb_tool -quiet -eval {7:@n} -calc kernel=n");
      ASSERT_TRUE(p.processor.memory().isSet("n"));
      EXPECT_EQ("7", p.processor.memory().get("n"));// not "7.000000"
    }

    {// 7c. An input reassigned via trailing LHS IS overwritten (the new
     //     value is what the user explicitly asked for).
      Parser p({});
      p.finalize();
      runCapture(p, "vdb_tool -quiet -eval {5:@n} -calc kernel=n=n+1");
      EXPECT_EQ(6.0f, strTo<float>(p.processor.memory().get("n")));
    }

    {// 8. A memory entry that isn't a valid float (e.g. set with the typo
     //    "{n:@n}" instead of "{0:@n}") produces a diagnostic naming the
     //    variable and showing the stored value.
      Parser p({});
      p.finalize();
      auto args = getArgs("vdb_tool -quiet -eval {n:@n} -calc kernel=n+1");
      p.parse(int(args.size()), args.data());
      try {
          p.run();
          FAIL() << "expected -calc to throw on non-numeric variable value";
      } catch (const std::exception &e) {
          const std::string what = e.what();
          EXPECT_NE(std::string::npos, what.find("\"n\""))   // names the variable
              << "error did not name the offending variable: " << what;
          EXPECT_NE(std::string::npos, what.find("not a valid float"))
              << "error did not describe the float-parse failure: " << what;
      }
    }
}// ActionCalc

TEST_F(Test_vdb_tool, ActionForValuesGreedy)
{
    // The three forValues actions also register their `kernel` option with
    // the greedy flag, so a bare positional kernel works alongside the
    // explicit kernel='...' form even with multi-option actions where
    // other options like `keep=` must still parse normally. The voxel
    // value is bound to the variable "v"; any other identifier in the
    // kernel must already be set in Processor memory (else an error).
    using namespace openvdb::vdb_tool;

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -sphere -forOnValues sin(v)+1");
      Tool t(int(args.size()), args.data());
      t.run();
    });

    EXPECT_NO_THROW({
      // Bare kernel with assignment ('=' embedded), alongside a named option.
      auto args = getArgs("vdb_tool -quiet -sphere -forOnValues v=v+1 keep=true");
      Tool t(int(args.size()), args.data());
      t.run();
    });

    EXPECT_NO_THROW({
      // Explicit kernel= form continues to work.
      auto args = getArgs("vdb_tool -quiet -sphere -forOnValues kernel=2*v");
      Tool t(int(args.size()), args.data());
      t.run();
    });

    EXPECT_NO_THROW({
      // Mixing the voxel variable v with a memory-resolved constant.
      auto args = getArgs("vdb_tool -quiet -eval {2:@scale} -sphere -forOnValues scale*v+1");
      Tool t(int(args.size()), args.data());
      t.run();
    });

    EXPECT_NO_THROW({
      // use= renames the voxel variable. "x" is now the voxel
      // input (not a memory lookup), and the rest of the kernel uses it
      // freely.
      auto args = getArgs("vdb_tool -quiet -sphere -forOnValues sin(x)+1 use=x");
      Tool t(int(args.size()), args.data());
      t.run();
    });

    EXPECT_NO_THROW({
      // use= renames AND a memory variable is still resolved correctly.
      auto args = getArgs("vdb_tool -quiet -eval {3:@scale} -sphere -forOnValues scale*y use=y");
      Tool t(int(args.size()), args.data());
      t.run();
    });

    {// A kernel that references an unbound (non-"v") variable should throw
     // with mErrorOnWarning enabled. The default Tool path swallows the
     // error and logs a "skipping" message, so trigger the strict path by
     // adding -on-warning=throw equivalent (just observe the log path here
     // by checking the grid wasn't transformed — we can't easily without
     // intermediate inspection; instead, drive the Calculator's variable
     // discovery directly).
      Calculator c;
      c.compile("a*v+1");
      ASSERT_EQ(2u, c.variables().size());
      // "v" and "a" should both appear; the action implementation would
      // try to bind "a" from memory and throw if absent.
      const auto &vars = c.variables();
      EXPECT_TRUE((vars[0] == "a" && vars[1] == "v") ||
                  (vars[0] == "v" && vars[1] == "a"));
    }
}// ActionForValuesGreedy

TEST_F(Test_vdb_tool, ActionForValuesStencil)
{
    // -forValues supports relative neighbor access via v(dx, dy, dz). At
    // every active voxel the kernel sees the current value plus values of
    // arbitrary neighbors fetched through a per-thread const accessor; the
    // grid is internally deep-copied so reads are from a stable snapshot.
    using namespace openvdb::vdb_tool;
    using GridT = openvdb::FloatGrid;

    // Build a 5×5×5 grid where each voxel stores its x-coordinate (so the
    // x-derivative kernel v(1,0,0) - v(-1,0,0) has a known answer of 2.0
    // at every interior voxel, regardless of voxel topology).
    GridT::Ptr grid = GridT::create(0.0f);
    auto acc = grid->getAccessor();
    for (int i = 0; i <= 4; ++i)
      for (int j = 0; j <= 4; ++j)
        for (int k = 0; k <= 4; ++k) {
            acc.setValue(openvdb::Coord(i, j, k), static_cast<float>(i));
        }

    // Compile and run the stencil kernel through the Calculator + neighbor
    // binding plumbing. Build the Tool from a config so the action wiring
    // is the production path (not a bespoke API).
    {
      auto args = getArgs("vdb_tool -quiet");
      Tool tool(int(args.size()), args.data());
      // Inject our prebuilt grid into the Tool's stack via parser plumbing.
      // (The Tool doesn't expose a public "addGrid" but it offers writeFile.)
      // Easier: write to disk, then drive the full -read / -forOnValues / -read pipeline.
      const std::string tmp = "data/test_stencil_in.vdb";
      openvdb::io::File(tmp).write({grid});
      auto args2 = getArgs("vdb_tool -quiet -read " + tmp +
                          " -forOnValues v(1,0,0)-v(-1,0,0)"
                          " -write data/test_stencil_out.vdb");
      Tool t2(int(args2.size()), args2.data());
      EXPECT_NO_THROW(t2.run());
    }

    // Read the result back and verify the interior values.
    GridT::Ptr result;
    {
      openvdb::io::File f("data/test_stencil_out.vdb");
      f.open();
      auto baseGrid = f.readGrid(f.beginName().gridName());
      result = openvdb::gridPtrCast<GridT>(baseGrid);
    }
    ASSERT_TRUE(result != nullptr);
    auto racc = result->getConstAccessor();
    // Interior x-derivative should equal 2.0 (= (i+1) - (i-1)).
    for (int i = 1; i <= 3; ++i)
      for (int j = 0; j <= 4; ++j)
        for (int k = 0; k <= 4; ++k) {
            const float got = racc.getValue(openvdb::Coord(i, j, k));
            EXPECT_NEAR(2.0f, got, 1e-5f) << "at (" << i << "," << j << "," << k << ")";
        }
}// ActionForValuesStencil

TEST_F(Test_vdb_tool, ActionForValuesMultiGrid)
{
    // -forValues with multiple grids: use=x,y vdb=0,1 picks the first as
    // output (iterated and written) and the rest as read-only inputs. Each
    // grid is accessible as a center value (bare name) or a relative
    // neighbor (name(dx,dy,dz)).
    using namespace openvdb::vdb_tool;
    using GridT = openvdb::FloatGrid;

    // Build two 5x5x5 grids:
    //   A: A(i,j,k) = i  (so x-derivative kernel sees +/-1 across i)
    //   B: B(i,j,k) = 10 (a constant offset)
    GridT::Ptr A = GridT::create(0.0f);
    GridT::Ptr B = GridT::create(0.0f);
    auto accA = A->getAccessor();
    auto accB = B->getAccessor();
    for (int i = 0; i <= 4; ++i)
      for (int j = 0; j <= 4; ++j)
        for (int k = 0; k <= 4; ++k) {
            accA.setValue(openvdb::Coord(i, j, k), static_cast<float>(i));
            accB.setValue(openvdb::Coord(i, j, k), 10.0f);
        }
    const std::string aPath = "data/test_mg_a.vdb";
    const std::string bPath = "data/test_mg_b.vdb";
    A->setName("A");
    B->setName("B");
    openvdb::io::File(aPath).write({A});
    openvdb::io::File(bPath).write({B});

    // Pointwise sum: out(i,j,k) = a(i,j,k) + b(i,j,k). Note that vdb_tool's
    // "age" convention is most-recently-read = 0, so after reading A then B,
    // age 0 is B and age 1 is A. With `use=a,b vdb=0,1`, name "a" is bound
    // to B (the output grid; written and iterated) and name "b" is bound
    // to A (read-only input). The kernel value is therefore 10 + i.
    {
      auto args = getArgs("vdb_tool -quiet -read " + aPath + " " + bPath +
                          " -forOnValues a+b use=a,b vdb=0,1"
                          " -write vdb=0 data/test_mg_out1.vdb");
      Tool t(int(args.size()), args.data());
      EXPECT_NO_THROW(t.run());
    }
    {
      openvdb::io::File f("data/test_mg_out1.vdb");
      f.open();
      auto baseGrid = f.readGrid(f.beginName().gridName());
      auto result = openvdb::gridPtrCast<GridT>(baseGrid);
      ASSERT_TRUE(result != nullptr);
      auto racc = result->getConstAccessor();
      for (int i = 0; i <= 4; ++i)
        for (int j = 0; j <= 4; ++j)
          for (int k = 0; k <= 4; ++k) {
              const float got = racc.getValue(openvdb::Coord(i, j, k));
              EXPECT_NEAR(static_cast<float>(i) + 10.0f, got, 1e-5f)
                  << "at (" << i << "," << j << "," << k << ")";
          }
    }

    // Cross-grid neighbor read: out(i,j,k) = a(1,0,0) + b(0,0,0).
    // age 0 (= "a") is B (constant 10), age 1 (= "b") is A (= i). So:
    //   a(1,0,0) at coord (i,j,k) reads B(i+1, j, k) = 10 (for i+1 in [0,4]).
    //   b(0,0,0) at coord (i,j,k) reads A(i,   j, k) = i.
    // Result: 10 + i for i in [0,3] (interior). At i=4, a(1,0,0) reads B's
    // background (= 0), so we skip that boundary plane.
    {
      auto args = getArgs("vdb_tool -quiet -read " + aPath + " " + bPath +
                          " -forOnValues a(1,0,0)+b(0,0,0) use=a,b vdb=0,1"
                          " -write vdb=0 data/test_mg_out2.vdb");
      Tool t(int(args.size()), args.data());
      EXPECT_NO_THROW(t.run());
    }
    {
      openvdb::io::File f("data/test_mg_out2.vdb");
      f.open();
      auto baseGrid = f.readGrid(f.beginName().gridName());
      auto result = openvdb::gridPtrCast<GridT>(baseGrid);
      ASSERT_TRUE(result != nullptr);
      auto racc = result->getConstAccessor();
      for (int i = 0; i <= 3; ++i)// only interior x indices
        for (int j = 0; j <= 4; ++j)
          for (int k = 0; k <= 4; ++k) {
              const float got = racc.getValue(openvdb::Coord(i, j, k));
              EXPECT_NEAR(10.0f + static_cast<float>(i), got, 1e-5f)
                  << "at (" << i << "," << j << "," << k << ")";
          }
    }
}// ActionForValuesMultiGrid

TEST_F(Test_vdb_tool, ActionSwitch)
{
    using namespace openvdb::vdb_tool;

    // Captures std::clog while running cmd through a fresh Parser, so the
    // tests can assert on what -eval echoed inside the matched -case body.
    auto runCapture = [](Parser &p, const std::string &cmd) -> std::string {
        auto args = getArgs(cmd);
        std::ostringstream oss;
        auto *old = std::clog.rdbuf(oss.rdbuf());
        try {
            p.parse(int(args.size()), args.data());
            p.run();
        } catch (...) {
            std::clog.rdbuf(old);
            throw;
        }
        std::clog.rdbuf(old);
        return oss.str();
    };

    {// Exact numeric match — only the case whose key equals the selector runs.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-switch on=2 "
            "-case key=1 -eval str=hit-1 -end "
            "-case key=2 -eval str=hit-2 -end "
            "-case key=3 -eval str=hit-3 -end "
          "-end");
      EXPECT_EQ(std::string::npos, out.find("hit-1")) << out;
      EXPECT_NE(std::string::npos, out.find("hit-2")) << out;
      EXPECT_EQ(std::string::npos, out.find("hit-3")) << out;
    }

    {// Numeric equivalence: 1.0 selector matches key=1, since both parse as float.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-switch on=1.0 "
            "-case key=1 -eval str=numeric-hit -end "
          "-end");
      EXPECT_NE(std::string::npos, out.find("numeric-hit")) << out;
    }

    {// String selector with verbatim string keys.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-switch on=sphere "
            "-case key=cube   -eval str=hit-cube   -end "
            "-case key=sphere -eval str=hit-sphere -end "
          "-end");
      EXPECT_EQ(std::string::npos, out.find("hit-cube"))   << out;
      EXPECT_NE(std::string::npos, out.find("hit-sphere")) << out;
    }

    {// '*' catch-all fires only when no prior case matched.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-switch on=99 "
            "-case key=1 -eval str=hit-1 -end "
            "-case key=2 -eval str=hit-2 -end "
            "-case key=* -eval str=hit-default -end "
          "-end");
      EXPECT_EQ(std::string::npos, out.find("hit-1"))       << out;
      EXPECT_EQ(std::string::npos, out.find("hit-2"))       << out;
      EXPECT_NE(std::string::npos, out.find("hit-default")) << out;
    }

    {// 'default' keyword is a synonym for '*' (catch-all).
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-switch on=99 "
            "-case key=1       -eval str=hit-1 -end "
            "-case key=default -eval str=hit-default -end "
          "-end");
      EXPECT_NE(std::string::npos, out.find("hit-default")) << out;
    }

    {// Catch-all does NOT fire if any earlier case matched.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-switch on=1 "
            "-case key=1 -eval str=hit-1 -end "
            "-case key=* -eval str=hit-default -end "
          "-end");
      EXPECT_NE(std::string::npos, out.find("hit-1"))       << out;
      EXPECT_EQ(std::string::npos, out.find("hit-default")) << out;
    }

    {// Per-iteration evaluation: -switch inside -for sees the loop variable
     //  each pass, so different cases fire on different iterations.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-for n=0,3 quiet=true "
            "-switch on={$n} "
              "-case key=0 -eval str=iter-zero -end "
              "-case key=1 -eval str=iter-one  -end "
              "-case key=* -eval str=iter-other -end "
            "-end "
          "-end");
      EXPECT_NE(std::string::npos, out.find("iter-zero"))  << out;
      EXPECT_NE(std::string::npos, out.find("iter-one"))   << out;
      EXPECT_NE(std::string::npos, out.find("iter-other")) << out;
    }

    {// Nested -switch inside a matched -case: only the inner case for the
     //  inner selector runs.
      Parser p({});
      p.finalize();
      const std::string out = runCapture(p, "vdb_tool -quiet "
          "-switch on=outer "
            "-case key=outer "
              "-switch on=inner "
                "-case key=inner -eval str=nested-hit -end "
                "-case key=*     -eval str=nested-miss -end "
              "-end "
            "-end "
          "-end");
      EXPECT_NE(std::string::npos, out.find("nested-hit"))  << out;
      EXPECT_EQ(std::string::npos, out.find("nested-miss")) << out;
    }

    {// -case outside any -switch is a hard error.
      Parser p({});
      p.finalize();
      EXPECT_THROW(runCapture(p, "vdb_tool -case key=1 -end"), std::exception);
    }
}// ActionSwitch

TEST_F(Test_vdb_tool, Memory)
{
    using namespace openvdb::vdb_tool;
    Memory mem;
    EXPECT_EQ(std::to_string(2.718281828459), mem.get("e"));
    EXPECT_EQ(std::to_string(openvdb::math::pi<float>()), mem.get("pi"));
    EXPECT_EQ(0, mem.size());
    EXPECT_FALSE(mem.isSet("a"));
    mem.set("a", 1.4f);
    EXPECT_TRUE(mem.isSet("a"));
    EXPECT_EQ(std::to_string(1.4f), mem.get("a"));
    EXPECT_EQ(1, mem.size());
    mem.clear();
    EXPECT_EQ(std::to_string(2.718281828459), mem.get("e"));
    EXPECT_EQ(std::to_string(openvdb::math::pi<float>()), mem.get("pi"));
    EXPECT_EQ(0, mem.size());
    EXPECT_FALSE(mem.isSet("a"));
}

TEST_F(Test_vdb_tool, Stack)
{
    using namespace openvdb::vdb_tool;
    Stack s;
    EXPECT_EQ(0, s.depth());
    EXPECT_TRUE(s.empty());
    s.push("foo");
    EXPECT_EQ(1, s.depth());
    EXPECT_FALSE(s.empty());
    EXPECT_EQ("foo", s.pop());
    s.push("foo");
    s.push("bar");
    EXPECT_EQ(2, s.depth());
    EXPECT_FALSE(s.empty());
    s.drop();
    EXPECT_EQ(1, s.depth());
    EXPECT_EQ("foo", s.top());
    EXPECT_EQ("foo", s.peek());
    s.top() = "bar";
    EXPECT_EQ("bar", s.top());
    EXPECT_EQ("bar", s.peek());
    s.dup();
    EXPECT_EQ(2, s.depth());
    EXPECT_EQ(Stack({"bar", "bar"}), s);
    s.top() = "foo";
    EXPECT_EQ(Stack({"bar", "foo"}), s);
    s.swap();
    EXPECT_EQ(Stack({"foo", "bar"}), s);
    s.nip();
    EXPECT_EQ(Stack({"bar"}), s);
    s.push("foo");
    s.push("bla");
    EXPECT_EQ(Stack({"bar", "foo", "bla"}), s);
    s.scrape();
    EXPECT_EQ(Stack({"bla"}), s);
    s.push("foo");
    s.push("bar");
    EXPECT_EQ(Stack({"bla", "foo", "bar"}), s);
    s.over();
    EXPECT_EQ(Stack({"bla", "foo", "bar", "foo"}), s);
    s.top()="bob";
    EXPECT_EQ(Stack({"bla", "foo", "bar", "bob"}), s);
    s.rot();
    EXPECT_EQ(Stack({"bla", "bar", "bob", "foo"}), s);
    s.tuck();
    EXPECT_EQ(Stack({"bla", "foo", "bar", "bob"}), s);
    std::stringstream ss;
    s.print(ss);
    EXPECT_EQ(std::string("bla,foo,bar,bob"), ss.str());

    {// throw paths documented in Parser.h
      Stack t;
      EXPECT_THROW(t.pop(),    std::invalid_argument);
      EXPECT_THROW(t.drop(),   std::invalid_argument);
      EXPECT_THROW(t.top(),    std::invalid_argument);
      EXPECT_THROW(t.peek(),   std::invalid_argument);
      EXPECT_THROW(t.dup(),    std::invalid_argument);
      EXPECT_THROW(t.scrape(), std::invalid_argument);
      t.push("a");
      EXPECT_THROW(t.swap(),   std::invalid_argument); // size < 2
      EXPECT_THROW(t.nip(),    std::invalid_argument); // size < 2
      EXPECT_THROW(t.over(),   std::invalid_argument); // size < 2
      t.push("b");
      EXPECT_THROW(t.rot(),    std::invalid_argument); // size < 3
      EXPECT_THROW(t.tuck(),   std::invalid_argument); // size < 3
    }
}// Stack

TEST_F(Test_vdb_tool, Processor)
{
    using namespace openvdb::vdb_tool;
    Processor proc;

    // test set and get, i.e. @ and $
    EXPECT_THROW({proc("{$file}");}, std::invalid_argument);
    EXPECT_THROW({proc("{dup}");},   std::invalid_argument);
    EXPECT_THROW({proc("{drop}");},  std::invalid_argument);
    EXPECT_THROW({proc("{swap}");},  std::invalid_argument);

    EXPECT_EQ(0, proc.memory().size());
    EXPECT_EQ(std::to_string(openvdb::math::pi<float>()), proc("{$pi}"));
    EXPECT_EQ(std::to_string(openvdb::math::pi<float>()), proc("{pi:get}"));
    EXPECT_EQ(std::to_string(2.718281828459), proc("{$e}"));
    EXPECT_TRUE(proc("{path/base_0123.ext:@file}").empty());
    EXPECT_EQ("path/base_0123.ext", proc("{$file}"));
    EXPECT_TRUE(proc("{1:@G}").empty());
    EXPECT_EQ("1", proc("{$G}"));
    EXPECT_TRUE(proc("{$file:upper:@file2}").empty());
    EXPECT_EQ("PATH/BASE_0123.EXT", proc("{$file2}"));
    EXPECT_TRUE(proc("{$G:1000:+:@F}").empty());
    EXPECT_EQ("1001", proc("{$F}"));
    EXPECT_TRUE(proc("{0.1:@x:0.2:@y}").empty());
    EXPECT_EQ("0.1", proc("{$x}"));
    EXPECT_EQ("0.2", proc("{$y}"));
    EXPECT_TRUE(proc("{1:$G:+:@G}").empty());
    EXPECT_EQ("2", proc("{$G}"));
    EXPECT_TRUE(proc("{$G:++:@G}").empty());
    EXPECT_EQ("3", proc("{$G}"));
    EXPECT_EQ(6, proc.memory().size());

    // test file-name methods
    EXPECT_EQ("path", proc("{$file:path}"));
    EXPECT_EQ("base_0123.ext", proc("{$file:file}"));
    EXPECT_EQ("base_0123", proc("{$file:name}"));
    EXPECT_EQ("base_", proc("{$file:base}"));
    EXPECT_EQ("0123", proc("{$file:number}"));
    EXPECT_EQ("ext", proc("{$file:ext}"));

    EXPECT_EQ("6", proc("{5:1:+}"));
    EXPECT_EQ(std::to_string(6.0f), proc("{5.0:1:+}"));
    EXPECT_EQ(std::to_string(6.2f), proc("{5.0:1.2:+}"));

    EXPECT_EQ("4", proc("{5:1:-}"));
    EXPECT_EQ(std::to_string(4.0f), proc("{5.0:1:-}"));
    EXPECT_EQ(std::to_string(3.8f), proc("{5.0:1.2:-}"));

    EXPECT_EQ("10", proc("{5:2:*}"));
    EXPECT_EQ(std::to_string(10.0f), proc("{5.0:2:*}"));
    EXPECT_EQ(std::to_string(6.0f), proc("{5.0:1.2:*}"));

    EXPECT_EQ("5", proc("{10:2:/}"));
    EXPECT_EQ("0", proc("{2:10:/}"));
    EXPECT_EQ(std::to_string(5.0f), proc("{10.0:2.0:/}"));
    EXPECT_EQ(std::to_string(0.2f), proc("{2.0:10.0:/}"));

    EXPECT_EQ("6", proc("{5:++}"));
    EXPECT_EQ(std::to_string(6.2f), proc("{5.2:++}"));

    EXPECT_EQ("4", proc("{5:--}"));
    EXPECT_EQ(std::to_string(4.2f), proc("{5.2:--}"));

    EXPECT_EQ("0", proc("{5:2:==}"));
    EXPECT_EQ("0", proc("{5.0:2.0:==}"));
    EXPECT_EQ("1", proc("{5:5:==}"));
    EXPECT_EQ("1", proc("{5.0:5.0:==}"));
    EXPECT_EQ("0", proc("{foo:bar:==}"));
    EXPECT_EQ("1", proc("{foo:foo:==}"));

    EXPECT_EQ("1", proc("{5:2:!=}"));
    EXPECT_EQ("1", proc("{5.0:2.0:!=}"));
    EXPECT_EQ("0", proc("{5:5:!=}"));
    EXPECT_EQ("0", proc("{5.0:5.0:!=}"));
    EXPECT_EQ("1", proc("{foo:bar:!=}"));
    EXPECT_EQ("0", proc("{foo:foo:!=}"));

    EXPECT_EQ("0", proc("{5:2:<=}"));
    EXPECT_EQ("0", proc("{5.0:2.0:<=}"));
    EXPECT_EQ("0", proc("{foo:bar:<=}"));
    EXPECT_EQ("1", proc("{2:5:<=}"));
    EXPECT_EQ("1", proc("{2.0:5.0:<=}"));
    EXPECT_EQ("1", proc("{bar:foo:<=}"));
    EXPECT_EQ("1", proc("{5:5:<=}"));
    EXPECT_EQ("1", proc("{5.0:5.0:<=}"));
    EXPECT_EQ("1", proc("{foo:foo:<=}"));

    EXPECT_EQ("1", proc("{5:2:>=}"));
    EXPECT_EQ("1", proc("{5.0:2.0:>=}"));
    EXPECT_EQ("1", proc("{foo:bar:>=}"));
    EXPECT_EQ("0", proc("{2:5:>=}"));
    EXPECT_EQ("0", proc("{2.0:5.0:>=}"));
    EXPECT_EQ("0", proc("{bar:foo:>=}"));
    EXPECT_EQ("1", proc("{5:5:>=}"));
    EXPECT_EQ("1", proc("{5.0:5.0:>=}"));
    EXPECT_EQ("1", proc("{foo:foo:>=}"));

    EXPECT_EQ("1", proc("{5:2:>}"));
    EXPECT_EQ("1", proc("{5.0:2.0:>}"));
    EXPECT_EQ("1", proc("{foo:bar:>}"));
    EXPECT_EQ("0", proc("{2:5:>}"));
    EXPECT_EQ("0", proc("{2.0:5.0:>}"));
    EXPECT_EQ("0", proc("{bar:foo:>}"));
    EXPECT_EQ("0", proc("{5:5:>}"));
    EXPECT_EQ("0", proc("{5.0:5.0:>}"));
    EXPECT_EQ("0", proc("{foo:foo:>}"));

    EXPECT_EQ("0", proc("{5:2:<}"));
    EXPECT_EQ("0", proc("{5.0:2.0:<}"));
    EXPECT_EQ("0", proc("{foo:bar:<}"));
    EXPECT_EQ("1", proc("{2:5:<}"));
    EXPECT_EQ("1", proc("{2.0:5.0:<}"));
    EXPECT_EQ("1", proc("{bar:foo:<}"));
    EXPECT_EQ("0", proc("{5:5:<}"));
    EXPECT_EQ("0", proc("{5.0:5.0:<}"));
    EXPECT_EQ("0", proc("{foo:foo:<}"));

    EXPECT_EQ("1", proc("{0:!}"));
    EXPECT_EQ("0", proc("{1:!}"));
    EXPECT_EQ("1", proc("{false:!}"));
    EXPECT_EQ("0", proc("{true:!}"));

    EXPECT_EQ("1", proc("{0:1:|}"));
    EXPECT_EQ("1", proc("{1:0:|}"));
    EXPECT_EQ("1", proc("{1:1:|}"));
    EXPECT_EQ("0", proc("{0:0:|}"));
    EXPECT_EQ("1", proc("{false:true:|}"));
    EXPECT_EQ("0", proc("{false:false:|}"));

    EXPECT_EQ("0", proc("{0:1:&}"));
    EXPECT_EQ("0", proc("{1:0:&}"));
    EXPECT_EQ("1", proc("{1:1:&}"));
    EXPECT_EQ("0", proc("{0:0:&}"));
    EXPECT_EQ("0", proc("{false:true:&}"));
    EXPECT_EQ("0", proc("{false:false:&}"));

    EXPECT_EQ("1", proc("{1:abs}"));
    EXPECT_EQ("1", proc("{-1:abs}"));
    EXPECT_EQ(std::to_string(1.2f), proc("{1.2:abs}"));
    EXPECT_EQ(std::to_string(1.2f), proc("{-1.2:abs}"));

    EXPECT_EQ(std::to_string(1.0f), proc("{1:ceil}"));
    EXPECT_EQ(std::to_string(2.0f), proc("{1.2:ceil}"));
    EXPECT_EQ(std::to_string(-1.0f), proc("{-1.2:ceil}"));

    EXPECT_EQ(std::to_string(1.0f), proc("{1:floor}"));
    EXPECT_EQ(std::to_string(1.0f), proc("{1.2:floor}"));
    EXPECT_EQ(std::to_string(-2.0f), proc("{-1.2:floor}"));

    EXPECT_EQ("4", proc("{2:pow2}"));
    EXPECT_EQ(std::to_string(4.0f), proc("{2.0:pow2}"));

    EXPECT_EQ("8", proc("{2:pow3}"));
    EXPECT_EQ(std::to_string(8.0f), proc("{2.0:pow3}"));

    EXPECT_EQ("9", proc("{3:2:pow}"));
    EXPECT_EQ(std::to_string(9.0f), proc("{3.0:2.0:pow}"));

    EXPECT_EQ("2", proc("{3:2:min}"));
    EXPECT_EQ("-2", proc("{3:-2:min}"));
    EXPECT_EQ(std::to_string(2.0f), proc("{3.0:2.0:min}"));
    EXPECT_EQ(std::to_string(-2.0f), proc("{3.0:-2.0:min}"));

    EXPECT_EQ("3", proc("{3:2:max}"));
    EXPECT_EQ("3", proc("{3:-2:max}"));
    EXPECT_EQ(std::to_string(3.0f), proc("{3.0:2.0:max}"));
    EXPECT_EQ(std::to_string(2.0f), proc("{-3.0:2.0:max}"));

    EXPECT_EQ("-3", proc("{3:neg}"));
    EXPECT_EQ("3", proc("{-3:neg}"));
    EXPECT_EQ(std::to_string(-3.0f), proc("{3.0:neg}"));
    EXPECT_EQ(std::to_string(3.0f), proc("{-3.0:neg}"));

    EXPECT_EQ(std::to_string(sin(2.0f)), proc("{2:sin}"));
    EXPECT_EQ(std::to_string(sin(2.0f)), proc("{2.0:sin}"));

    EXPECT_EQ(std::to_string(cos(2.0f)), proc("{2:cos}"));
    EXPECT_EQ(std::to_string(cos(2.0f)), proc("{2.0:cos}"));

    EXPECT_EQ(std::to_string(tan(2.0f)), proc("{2:tan}"));
    EXPECT_EQ(std::to_string(tan(2.0f)), proc("{2.0:tan}"));

    EXPECT_EQ(std::to_string(asin(2.0f)), proc("{2:asin}"));
    EXPECT_EQ(std::to_string(asin(2.0f)), proc("{2.0:asin}"));

    EXPECT_EQ(std::to_string(acos(2.0f)), proc("{2:acos}"));
    EXPECT_EQ(std::to_string(acos(2.0f)), proc("{2.0:acos}"));

    EXPECT_EQ(std::to_string(atan(2.0f)), proc("{2:atan}"));
    EXPECT_EQ(std::to_string(atan(2.0f)), proc("{2.0:atan}"));

    EXPECT_NEAR(openvdb::math::pi<float>(), strToFloat(proc("{180.0:d2r}")), 1e-4);
    EXPECT_NEAR(180.0f, strToFloat(proc("{$pi:r2d}")), 1e-4);

    EXPECT_EQ(std::to_string(1.0f/2.0f), proc("{2:inv}"));
    EXPECT_EQ(std::to_string(1.0f), proc("{1.0:inv}"));
    EXPECT_EQ(std::to_string(1.0f/1.2f), proc("{1.2:inv}"));

    EXPECT_EQ(std::to_string(exp(1.2f)), proc("{1.2:exp}"));
    EXPECT_EQ(std::to_string(log(1.2f)), proc("{1.2:ln}"));
    EXPECT_EQ(std::to_string(log10(1.2f)), proc("{1.2:log}"));
    EXPECT_EQ(std::to_string(sqrt(1.2f)), proc("{1.2:sqrt}"));
    EXPECT_EQ("1", proc("{1:to_int}"));
    EXPECT_EQ("1", proc("{1.2:to_int}"));
    EXPECT_EQ(std::to_string(1.0f), proc("{1:to_float}"));
    EXPECT_EQ(std::to_string(1.2f), proc("{1.2:to_float}"));

    EXPECT_EQ("abcde012", proc("{AbCdE012:lower}"));
    EXPECT_EQ("ABCDE012", proc("{AbCdE012:upper}"));

    EXPECT_EQ("1", proc("{1:dup:==}"));
    EXPECT_EQ("2", proc("{1:2:nip}"));
    EXPECT_EQ("1", proc("{1:2:drop}"));
    EXPECT_EQ(std::to_string(0.5f), proc("{1.0:2.0:/}"));
    EXPECT_EQ(std::to_string(2.0f), proc("{1.0:2.0:swap:/}"));
    EXPECT_EQ(std::to_string(2.0f/1.0f+1.0f), proc("{1.0:2.0:over:/:+}"));

    EXPECT_EQ(std::to_string(2.0f/3.0f+1.0f), proc("{1.0:2.0:3.0:/:+}"));
    EXPECT_EQ(std::to_string(3.0f/1.0f+2.0f), proc("{1.0:2.0:3.0:rot:/:+}"));// rot(1 2 3) = 2 3 1
    EXPECT_EQ(std::to_string(1.0f/2.0f+3.0f), proc("{1.0:2.0:3.0:tuck:/:+}"));// tuck(1 2 3) = 3 1 2

    EXPECT_EQ("123", proc("{123:0:pad0}"));
    EXPECT_EQ("123", proc("{123:1:pad0}"));
    EXPECT_EQ("123", proc("{123:2:pad0}"));
    EXPECT_EQ("123", proc("{123:3:pad0}"));
    EXPECT_EQ("0123", proc("{123:4:pad0}"));
    EXPECT_EQ("00123", proc("{123:5:pad0}"));
    EXPECT_EQ("000123", proc("{123:6:pad0}"));

    EXPECT_EQ("0", proc("{depth}"));
    EXPECT_EQ("1", proc("{0:depth:scrape}"));
    EXPECT_EQ("2", proc("{0:1:depth:scrape}"));
    EXPECT_EQ("3", proc("{0:1:2:depth:scrape}"));
    EXPECT_EQ("4", proc("{0:1:2:3:depth:scrape}"));
    EXPECT_EQ("4", proc("{0:1:2:3:clear:4}"));
    EXPECT_EQ("4", proc("{0:1:2:3:depth:@size:clear:$size}"));

    EXPECT_EQ("0", proc("{e:is_set}"));
    EXPECT_EQ("0", proc("{pi:is_set}"));
    EXPECT_EQ("0", proc("{foo:is_set}"));
    EXPECT_EQ("1", proc("{8:@bar:bar:is_set}"));

    EXPECT_EQ(std::to_string(sqrt(0.1f*0.1f + 0.2f*0.2f)), proc("{$x:pow2:$y:pow2:+:sqrt}"));

    EXPECT_EQ("4",proc("{1:2:<:if(1:3:+)}"));
    EXPECT_EQ("",proc("{1:2:>:if(1:3:+)}"));
    EXPECT_EQ("1",proc("{5:@a:1:2:<:if(1:@a):$a}"));
    EXPECT_EQ("5",proc("{5:@a:1:2:>:if(1:@a):$a}"));

    EXPECT_EQ("4",proc("{1:2:<:if(1:3:+?2:2:-)}"));
    EXPECT_EQ("0",proc("{1:2:>:if(1:3:+?2:2:-)}"));
    EXPECT_EQ("1",proc("{1:2:<:if(1:@a?2:@a):$a}"));
    EXPECT_EQ("2",proc("{1:2:>:if(1:@a?2:@a):$a}"));
    EXPECT_EQ(std::to_string(sqrt(4+16)),proc("{$pi:2:>:if(2:pow2:4:pow2:+:sqrt?2:sin)}"));
    EXPECT_EQ(std::to_string(sin(2)),proc("{$pi:2:<:if(2:pow2:4:pow2:+:sqrt?2:sin)}"));

    EXPECT_EQ("a", proc("{1:switch(1:a?2:b?3:c)}"));
    EXPECT_EQ("b", proc("{2:switch(1:a?2:b?3:c)}"));
    EXPECT_EQ("c", proc("{3:switch(1:a?2:b?3:c)}"));
    //EXPECT_THROW({proc("{0:switch(1:a?2:b?3:c)}");}, std::invalid_argument);
    //EXPECT_THROW({proc("{4:switch(1:a?2:b?3:c)}");}, std::invalid_argument);
    EXPECT_EQ("SUPER", proc("{1:switch(1:super:upper?2:1:2:+?3:$pi)}"));
    EXPECT_EQ("3", proc("{2:switch(1:super:upper?2:1:2:+?3:$pi)}"));
    EXPECT_EQ(std::to_string(openvdb::math::pi<float>()), proc("{3:switch(1:super:upper?2:1:2:+?3:$pi)}"));

    EXPECT_EQ("a", proc("{a:squash}"));
    EXPECT_EQ("a,b,c,d", proc("{a:b:c:d:squash}"));

    EXPECT_EQ("1", proc("{a:length}"));
    EXPECT_EQ("3", proc("{foo:length}"));
    EXPECT_EQ("7", proc("{foo bar:length}"));

    EXPECT_EQ("foobar", proc("{foo:bar:append}"));
    EXPECT_EQ("foo,bar", proc("{foo:,:append:bar:append}"));

    EXPECT_EQ("3", proc("{1,2,3:,:tokenize:depth:scrape}"));
    EXPECT_EQ("5", proc("{1,2,3:,:tokenize:+:*}"));

    // find two real roots of a quadratic polynomial
    EXPECT_EQ("0.683375,7.316625", proc("{1:@a:-8:@b:5:@c:$b:pow2:4:$a:*:$c:*:-:@c:-2:$a:*:@a:$c:0:==:if($b:$a:/):$c:0:>:if($c:sqrt:dup:$b:+:$a:/:$b:rot:-:$a:/):squash}"));

    EXPECT_EQ("foo bar bla", proc("{foo_bar_bla:_: :replace}"));
    EXPECT_EQ("foo_bar_bla", proc("{foo bar bla: :_:replace}"));
    EXPECT_EQ("a b c d", proc("{a,b,c,d:,: :replace}"));
    EXPECT_EQ("a,b,c,d", proc("{a_b_c_d:_:,:replace}"));
    EXPECT_EQ("a,b,c,d", proc("{a b c d: :,:replace}"));
    EXPECT_EQ("a b c d", proc("{a:b:c:d:squash:,: :replace}"));

    EXPECT_EQ("_bar", proc("{foo_bar:foo:erase}"));
    EXPECT_EQ("f_bar", proc("{foo_bar:o:erase}"));
    EXPECT_EQ("foobar", proc("{foo bar: :erase}"));
}// Processor

TEST_F(Test_vdb_tool, ToolParser)
{
    using namespace openvdb::vdb_tool;
    int alpha = 0, alpha_sum = 0;
    float beta = 0.0f, beta_sum = 0.0f;
    std::string path, base, ext;

    Parser p({{"alpha", "64", "", ""}, {"beta", "4.56", "", ""}});
    p.addAction({"process_a", "a"}, "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){alpha = p.get<int>("alpha");
                     beta  = p.get<float>("beta");}
               );
    p.addAction({"process_b", "b"}, "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){alpha_sum += p.get<int>("alpha");
                     beta_sum  += p.get<float>("beta");}
               );
    p.addAction({"process_c", "c"}, "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""},{"gamma", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){path += (path.empty()?"":",") + p.getStr("alpha");
                     base += (base.empty()?"":",") + p.getStr("beta");
                     ext  += (ext.empty() ?"":",") + p.getStr("gamma");}
               );
    p.finalize();

    auto args = getArgs("vdb_tool -quiet -process_a alpha=128 -for v=0.1,0.4,0.1 -b alpha={$#v:++} beta={$v} -end");
    p.parse(int(args.size()), args.data());
    EXPECT_EQ(0, alpha);
    EXPECT_EQ(0.0f, beta);
    EXPECT_EQ(0, alpha_sum);
    EXPECT_EQ(0.0f, beta_sum);
    p.run();
    EXPECT_EQ(128, alpha);// defined explicitly
    EXPECT_EQ(4.56f, beta);// default value
    EXPECT_EQ(1 + 2 + 3, alpha_sum);// derived from loop
    EXPECT_EQ(0.1f + 0.2f + 0.3f, beta_sum);// derived from loop

    args = getArgs("vdb_tool -quiet -each file=path1/base1.ext1,path2/base2.ext2 -c alpha={$file:path} beta={$file:name} gamma={$file:ext} -end");
    p.parse(int(args.size()), args.data());
    p.run();
    EXPECT_EQ(path, "path1,path2");
    EXPECT_EQ(base, "base1,base2");
    EXPECT_EQ(ext,  "ext1,ext2");
}// ToolParser

TEST_F(Test_vdb_tool, ToolBasic)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(fileExists("data"));
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(fileExists("data/sphere.ply"));
    EXPECT_FALSE(fileExists("data/config.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -sphere r=1.1 -ls2mesh -write data/sphere.ply data/config.txt");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(fileExists("data/sphere.ply"));
    EXPECT_TRUE(fileExists("data/config.txt"));
}// ToolBasic

TEST_F(Test_vdb_tool, Counter)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(fileExists("data"));
    std::remove("data/sphere_1.ply");
    std::remove("data/config_2.txt");

    EXPECT_FALSE(fileExists("data/sphere_1.ply"));
    EXPECT_FALSE(fileExists("data/config_2.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -eval {1:@G} -sphere r=1.1 -ls2mesh -write data/sphere_{$G}.ply data/config_{$G:++}.txt");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(fileExists("data/sphere_1.ply"));
    EXPECT_TRUE(fileExists("data/config_2.txt"));
}// Counter

TEST_F(Test_vdb_tool, ToolForLoop)
{
    using namespace openvdb::vdb_tool;

    std::remove("data/config.txt");
    EXPECT_FALSE(fileExists("data/config.txt"));
    for (int i=0; i<4; ++i) {
      const std::string name("data/sphere_"+std::to_string(i)+".ply");
      std::remove(name.c_str());
      EXPECT_FALSE(fileExists(name));
    }

    // test single for-loop
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -for i=0,3 -sphere r=1.{$i} dim=128 name=sphere_{$i} -ls2mesh -write data/sphere_{$#i:++}.ply -end");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    for (int i=1; i<4; ++i) EXPECT_TRUE(fileExists("data/sphere_"+std::to_string(i)+".ply"));

    // test two nested for-loops
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -for v=0.1,0.3,0.1 -each s=sphere_1,sphere_3 -read ./data/{$s}.ply -mesh2ls voxel={$v} -end -end -write data/test.vdb");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(fileExists("data/test.vdb"));
}// ToolForLoop

TEST_F(Test_vdb_tool, ToolFilesLoop)
{
    using namespace openvdb::vdb_tool;
    auto myRemove = [](const std::string &file){
      std::remove(file.c_str());
      EXPECT_FALSE(fileExists(file));
    };

    // test single for-loop
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -for i=0,3 -sphere r=1.{$i} dim=128 name=sphere_{$i} -write data/sphere_{$#i}.vdb -end");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    for (int i=0; i<3; ++i) EXPECT_TRUE(fileExists("data/sphere_"+std::to_string(i)+".vdb"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -files path=data include=sphere_ ext=vdb min=8 -read {$file} -ls2mesh -write {$file:path}/{$file:name}.obj -end");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    for (int i=0; i<3; ++i) {
      EXPECT_TRUE(fileExists("data/sphere_"+std::to_string(i)+".obj"));
      myRemove("data/sphere_"+std::to_string(i)+".vdb");
      myRemove("data/sphere_"+std::to_string(i)+".obj");
    }
}// ToolFilesLoop

TEST_F(Test_vdb_tool, ToolError)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(fileExists("data"));
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(fileExists("data/sphere.ply"));
    EXPECT_FALSE(fileExists("data/config.txt"));

    EXPECT_THROW({
      auto args = getArgs("vdb_tool -sphere bla=3 -ls2mesh -write data/sphere.ply data/config.txt -quiet");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    }, std::invalid_argument);

    EXPECT_FALSE(fileExists("data/sphere.ply"));
    EXPECT_FALSE(fileExists("data/config.txt"));
}

TEST_F(Test_vdb_tool, ToolKeep)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(fileExists("data"));
    std::remove("data/sphere.vdb");
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(fileExists("data/sphere.vdb"));
    EXPECT_FALSE(fileExists("data/sphere.ply"));
    EXPECT_FALSE(fileExists("data/config.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -default keep=1 -sphere r=2 -ls2mesh vdb=0 -write vdb=0 geo=0 data/sphere.vdb data/sphere.ply data/config.txt");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(fileExists("data/sphere.vdb"));
    EXPECT_TRUE(fileExists("data/sphere.ply"));
    EXPECT_TRUE(fileExists("data/config.txt"));
}

TEST_F(Test_vdb_tool, ToolConfig)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(fileExists("data"));
    std::remove("data/sphere.vdb");
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(fileExists("data/sphere.vdb"));
    EXPECT_FALSE(fileExists("data/sphere.ply"));
    EXPECT_FALSE(fileExists("data/config.txt"));

    // Generate the config file this test exercises. Previously this depended
    // on test ordering with ToolKeep; producing it here keeps the test
    // self-contained and survives --gtest_filter.
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -default keep=1 -sphere r=2 -ls2mesh vdb=0 -write vdb=0 geo=0 data/sphere.vdb data/sphere.ply data/config.txt");
      Tool tool(int(args.size()), args.data());
      tool.run();
    });

    // Remove the data files so we can verify the config alone re-creates them.
    std::remove("data/sphere.vdb");
    std::remove("data/sphere.ply");
    EXPECT_FALSE(fileExists("data/sphere.vdb"));
    EXPECT_FALSE(fileExists("data/sphere.ply"));
    EXPECT_TRUE(fileExists("data/config.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -config data/config.txt");
      Tool vdb_tool(int(args.size()), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(fileExists("data/sphere.vdb"));
    EXPECT_TRUE(fileExists("data/sphere.ply"));
    EXPECT_TRUE(fileExists("data/config.txt"));
}

// Tool::Header is a private nested struct, so we exercise it indirectly:
// generate a config file, corrupt its header line, and verify -config rejects it.
TEST_F(Test_vdb_tool, ToolConfigHeader)
{
    using namespace openvdb::vdb_tool;
    EXPECT_TRUE(fileExists("data"));

    const std::string cfg = "data/header_test.txt";
    std::remove(cfg.c_str());

    // Produce a valid config.
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -sphere -write " + cfg);
      Tool tool(int(args.size()), args.data());
      tool.run();
    });
    ASSERT_TRUE(fileExists(cfg));

    // The unmodified config must load successfully.
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -config " + cfg);
      Tool tool(int(args.size()), args.data());
      tool.run();
    });

    // Helper: replace the first line of the config file with `newHeader`.
    auto rewriteHeader = [&](const std::string &newHeader) {
        std::ifstream in(cfg);
        std::string body((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
        in.close();
        const auto nl = body.find('\n');
        ASSERT_NE(std::string::npos, nl);
        std::ofstream out(cfg, std::ios::trunc);
        out << newHeader << "\n" << body.substr(nl + 1);
    };

    // Incompatible major version (Header parses, isCompatible() returns false).
    rewriteHeader("vdb_tool 999.0.0");
    EXPECT_THROW({
      auto args = getArgs("vdb_tool -quiet -config " + cfg);
      Tool tool(int(args.size()), args.data());
      tool.run();
    }, std::invalid_argument);

    // Bad magic (Header constructor throws).
    rewriteHeader("not_vdb_tool 10.8.0");
    EXPECT_THROW({
      auto args = getArgs("vdb_tool -quiet -config " + cfg);
      Tool tool(int(args.size()), args.data());
      tool.run();
    }, std::invalid_argument);

    // Malformed header (wrong number of fields).
    rewriteHeader("garbage");
    EXPECT_THROW({
      auto args = getArgs("vdb_tool -quiet -config " + cfg);
      Tool tool(int(args.size()), args.data());
      tool.run();
    }, std::invalid_argument);

    std::remove(cfg.c_str());
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
